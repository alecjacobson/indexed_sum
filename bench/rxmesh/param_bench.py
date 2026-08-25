"""IndexedSum side of the parameterization comparison (RXMesh AD paper, Table 2).

RXMesh's Param app minimizes the **symmetric Dirichlet** energy with a matrix-free CG Newton
solver (eval_terms_grad_only + Hessian-vector products, NO Hessian assembly). The paper's
Table 2 compares RXMesh vs *PyTorch* here (2.76x geomean), not IndexedSum.

This bench measures the IndexedSum way of providing Newton derivatives for the same energy:
gradient + a full **sparse** Hessian assembly per iteration, and shows the new
`compile`/`cuda_graphs` switches on it. Symmetric Dirichlet has a J^{-1} term (a determinant
in the denominator), so it is exactly the kind of energy the RXMesh paper described as
"reverse-mode AD" but that IndexedSum actually differentiates **forward-over-reverse** -- the
path with the known `torch.linalg.det`-under-vmap bug. We therefore:
  (1) write the 2x2 determinant/inverse with elementary ops (the library's `det`-helper
      remedy) and verify the eager Hessian matches finite differences;
  (2) demonstrate that the naive `torch.linalg.det` formulation gives a WRONG Hessian here;
  (3) time eager vs compile vs cuda_graphs for the sparse-Hessian assembly.

This is the per-Newton-iteration *derivative-provision* cost. It is NOT directly comparable
to RXMesh's matrix-free HVP stream (different algorithm); it IS the analog of the paper's
PyTorch dense-Hessian baseline, but sparse.

Usage:
  python bench/rxmesh/param_bench.py --check
  python bench/rxmesh/param_bench.py --sizes 100 500 1000
"""
import argparse
import os
import sys

import torch
import torch._dynamo

for _attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _attr):
        setattr(torch._dynamo.config, _attr, 256)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from indexed_sum import IndexedSum  # noqa: E402
from bench.rxmesh._common import (  # noqa: E402
    plane_grid, time_ms, warmup_cost_s, fd_hessian, reldiff, dense_from_sparse, sync,
)


def _inv2_elem(M):
    """Elementary 2x2 inverse (vmap/forward-mode safe; no torch.linalg)."""
    a, b, c, d = M[0, 0], M[0, 1], M[1, 0], M[1, 1]
    det = a * d - b * c
    inv = torch.stack([torch.stack([d, -b]), torch.stack([-c, a])])
    return inv / det


def make_symdir(F, Mrinv, A, elementary=True, **flags):
    """Symmetric Dirichlet per face over uv (local_size=3, dim=2).
    per-term constant packs [Mrinv(4), A(1)] = 5 values."""
    const = torch.cat([Mrinv.reshape(-1, 4), A.reshape(-1, 1)], dim=1)  # (nF,5)

    def symdir(uv, cst):                 # uv:(3,2), cst:(5,)
        a, b, c = uv[0], uv[1], uv[2]
        M = torch.stack([b - a, c - a], dim=1)     # 2x2, columns (b-a),(c-a)
        Mri = cst[:4].view(2, 2)
        area = cst[4]
        J = M @ Mri
        if elementary:
            Jinv = _inv2_elem(J)
        else:
            Jinv = torch.linalg.inv(J)             # still elementary-ish; det bug is in det/slogdet
        return area * ((J * J).sum() + (Jinv * Jinv).sum())

    return IndexedSum(symdir, F, per_term_constants=const, **flags)


def make_symdir_linalgdet(F, Mrinv, A, **flags):
    """Same energy but computing the inverse via a determinant path that routes through
    torch.linalg.det -- the forward-over-reverse-under-vmap bug case, for the demo."""
    const = torch.cat([Mrinv.reshape(-1, 4), A.reshape(-1, 1)], dim=1)

    def symdir(uv, cst):
        a, b, c = uv[0], uv[1], uv[2]
        M = torch.stack([b - a, c - a], dim=1)
        Mri = cst[:4].view(2, 2)
        area = cst[4]
        J = M @ Mri
        detJ = torch.linalg.det(J)                 # <-- the buggy-under-vmap-hessian op
        adj = torch.stack([torch.stack([J[1, 1], -J[0, 1]]),
                           torch.stack([-J[1, 0], J[0, 0]])])
        Jinv = adj / detJ
        return area * ((J * J).sum() + (Jinv * Jinv).sum())

    return IndexedSum(symdir, F, per_term_constants=const, **flags)


def rest_shapes(V, F):
    """Per-face 2x2 rest shape inverse Mrinv and area A from the mesh geometry (uses xy)."""
    X = V[:, :2]
    A0, B0, C0 = X[F[:, 0]], X[F[:, 1]], X[F[:, 2]]
    e1 = B0 - A0
    e2 = C0 - A0
    Mr = torch.stack([e1, e2], dim=2)              # (nF,2,2) columns e1,e2
    det = Mr[:, 0, 0] * Mr[:, 1, 1] - Mr[:, 0, 1] * Mr[:, 1, 0]
    inv = torch.empty_like(Mr)
    inv[:, 0, 0] = Mr[:, 1, 1]
    inv[:, 0, 1] = -Mr[:, 0, 1]
    inv[:, 1, 0] = -Mr[:, 1, 0]
    inv[:, 1, 1] = Mr[:, 0, 0]
    Mrinv = inv / det[:, None, None]
    Aface = 0.5 * det.abs()
    return Mrinv.detach(), Aface.detach()


def correctness(device):
    print("== correctness gate (grid n=6, f64): symmetric Dirichlet ==")
    dtype = torch.float64
    torch.manual_seed(0)
    V0, F, E = plane_grid(6, dtype=dtype, device=device)
    Mrinv, A = rest_shapes(V0, F)
    uv0 = V0[:, :2].contiguous()
    uvc = (uv0 + 0.03 * torch.randn_like(uv0)).detach()   # strained, flip-free

    term = make_symdir(F, Mrinv, A, elementary=True)

    def energy(uv):
        return term(uv)

    H_eager = dense_from_sparse(term.sparse_hessian(uvc))
    H_fd = fd_hessian(energy, uvc)
    fd_err = reldiff(H_fd, H_eager)

    H_comp = dense_from_sparse(make_symdir(F, Mrinv, A, elementary=True, compile=True).sparse_hessian(uvc))
    comp_err = reldiff(H_eager, H_comp)
    H_cg = dense_from_sparse(make_symdir(F, Mrinv, A, elementary=True, cuda_graphs=True).sparse_hessian(uvc))
    cg_err = reldiff(H_eager, H_cg)

    # bug demo: torch.linalg.det formulation vs FD
    term_bug = make_symdir_linalgdet(F, Mrinv, A)
    H_bug = dense_from_sparse(term_bug.sparse_hessian(uvc))
    bug_err = reldiff(H_fd, H_bug)

    print(f"  elementary-det eager Hessian vs FD   relerr = {fd_err:.2e}")
    print(f"  compile vs eager                     relerr = {comp_err:.2e}")
    print(f"  cuda_graphs vs eager                 relerr = {cg_err:.2e}")
    print(f"  [bug demo] torch.linalg.det vs FD    relerr = {bug_err:.2e}  "
          f"({'WRONG as expected' if bug_err > 1e-2 else 'unexpectedly ok'})")
    ok = fd_err < 1e-4 and comp_err < 1e-9 and cg_err < 1e-9
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def time_variant(n, dtype, device, compile, cuda_graphs, iters, cache_indices=False):
    if compile or cuda_graphs:
        torch._dynamo.reset()
    V, F, E = plane_grid(n, dtype=dtype, device=device)
    Mrinv, A = rest_shapes(V, F)
    uv = (V[:, :2] + 0.01 * torch.randn(V.shape[0], 2, device=device, dtype=dtype)).detach().requires_grad_(True)
    term = make_symdir(F, Mrinv, A, elementary=True, compile=compile, cuda_graphs=cuda_graphs,
                       cache_indices=cache_indices)

    def diff_call():
        g = term.dense_gradient(uv)  # honors compile/cuda_graphs/cache_indices like the Hessian
        return term.sparse_hessian(uv)
    warm = warmup_cost_s(diff_call, device)
    ms = time_ms(diff_call, device, iters=iters, repeats=7, warmup=3)

    def hess_only():
        return term.sparse_hessian(uv)
    hess_ms = time_ms(hess_only, device, iters=iters, repeats=5, warmup=2)
    return dict(ms=ms, hess_ms=hess_ms, warmup_s=warm, nV=V.shape[0], nF=F.shape[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--iters", type=int, default=20)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "f32" else torch.float64
    print(f"device={device} dtype={args.dtype} gpu="
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}\n")
    if not correctness(device):
        print("CORRECTNESS FAILED — aborting.")
        sys.exit(1)
    if args.check:
        return
    torch._dynamo.reset()
    print("\n== per-iteration derivative cost: grad + sparse Hessian, ms ==")
    print(f"{'n':>6} {'nV':>9} {'nF':>9} {'variant':<18} {'diff_ms':>10} {'hess_ms':>10} "
          f"{'warmup_s':>9} {'speedup':>8}")
    for n in args.sizes:
        base = None
        for name, cf in [("eager", (False, False, False)), ("compile", (True, False, False)),
                         ("cuda_graphs", (False, True, False)),
                         ("compile+cache", (True, False, True)),
                         ("cuda_graphs+cache", (False, True, True))]:
            r = time_variant(n, dtype, device, cf[0], cf[1], args.iters, cache_indices=cf[2])
            if name == "eager":
                base = r["ms"]
            sp = base / r["ms"] if base else 1.0
            print(f"{n:>6} {r['nV']:>9} {r['nF']:>9} {name:<18} {r['ms']:>10.3f} "
                  f"{r['hess_ms']:>10.3f} {r['warmup_s']:>9.2f} {sp:>7.2f}x")
            sync(device)
            if device == "cuda":
                torch.cuda.empty_cache()
        print()


if __name__ == "__main__":
    main()
