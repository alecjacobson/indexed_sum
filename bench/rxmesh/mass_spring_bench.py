"""IndexedSum side of the mass-spring cloth comparison (RXMesh AD paper, Table 1).

Replicates the paper's headline IndexedSum benchmark and extends it with the new
`compile` / `cuda_graphs` switches. The timed unit matches what RXMesh's MassSpring
app times as "Diff" ( = problem.eval_terms() ): **gradient + Hessian assembly** of the
total energy, per Newton evaluation, excluding the linear solve.

Energies replicate apps/MassSpring exactly (Flag scene):
  * spring   (per edge):   E = 0.5*k*h^2 * r * (|a-b|^2/r - 1)^2,  r = squared rest length
  * inertial (per vertex): E = 0.5*m * |x - x_pred|^2
  * gravity  (per vertex): E = -m*h^2 * (x . g),  g = (0,-9.81,0)

Constants (rho=100, k=4e4, h=0.01) match the app; their exact values do not affect the
Hessian-assembly op count / timing, only the correctness check's self-consistency.

Variants timed on the same GPU:
  eager        - IndexedSum default (forward-over-reverse), the paper's condition
  compile      - IndexedSum(compile=True), Inductor fusion (reverse-over-reverse)
  cuda_graphs  - IndexedSum(cuda_graphs=True), fusion + CUDA-graph replay
  pytorch_dense- torch.func.hessian over the flattened energy -> dense Hessian (paper's
                 "PyTorch" column; expected to OOM at large sizes)

Usage:
  python bench/rxmesh/mass_spring_bench.py --check           # correctness gate only
  python bench/rxmesh/mass_spring_bench.py --sizes 10 100    # timing at chosen grid n
  python bench/rxmesh/mass_spring_bench.py --csv out.csv     # full sweep
"""
import argparse
import csv
import os
import sys

import torch
import torch._dynamo

# Each IndexedSum term compiles the same `blocks` code line (indexed_sum.py:170), so
# spring/inertial/gravity x {compile, cuda_graphs} accumulate guard specializations on
# one dynamo cache entry. Raise the recompile cap so none silently fall back to eager
# (which would corrupt the compiled timings). We also reset dynamo between variants.
for _attr in ("recompile_limit", "cache_size_limit"):
    if hasattr(torch._dynamo.config, _attr):
        setattr(torch._dynamo.config, _attr, 256)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from indexed_sum import IndexedSum  # noqa: E402
from bench.rxmesh._common import (  # noqa: E402
    plane_grid, time_ms, warmup_cost_s, fd_hessian, fd_gradient, reldiff,
    dense_from_sparse, sync,
)

RHO = 100.0
K = 4e4
H = 0.01
G = (0.0, -9.81, 0.0)


# --------------------------------------------------------------------------------------
# Energy term builders (return IndexedSum objects with the requested compile flags)
# --------------------------------------------------------------------------------------
def build_terms(V, E, dtype, device, compile=False, cuda_graphs=False, cache_indices=False):
    nV = V.shape[0]
    # squared rest lengths (RXMesh stores rest_l = (a-b).squaredNorm())
    d0 = V[E[:, 0]] - V[E[:, 1]]
    r = (d0 * d0).sum(dim=1, keepdim=True).detach()               # (M,1)
    mass = RHO / nV                                               # plane area = 1x1
    x_pred = V.detach().clone()                                   # inertial target (v=0)
    I = torch.arange(nV, device=device).unsqueeze(1)             # (nV,1)
    gvec = torch.tensor(G, dtype=dtype, device=device)

    c_spring = 0.5 * K * H * H
    half_m = 0.5 * mass
    c_grav = -mass * H * H
    flags = dict(compile=compile, cuda_graphs=cuda_graphs, cache_indices=cache_indices)

    def spring(v, rr):          # v:(2,3), rr:(1,)
        d = v[1] - v[0]
        r0 = rr[0]
        s = (d * d).sum() / r0 - 1.0
        return c_spring * r0 * s * s

    def inertial(v, xp):        # v:(1,3), xp:(1,3)
        d = v[0] - xp[0]
        return half_m * (d * d).sum()

    def gravity(v):             # v:(1,3)
        return c_grav * (v[0] * gvec).sum()

    spring_term = IndexedSum(spring, E, per_term_constants=r, **flags)
    inertial_term = IndexedSum(inertial, I, per_variable_constants=x_pred, **flags)
    gravity_term = IndexedSum(gravity, I, **flags)
    return spring_term + inertial_term + gravity_term


def total_energy(terms, V):
    return terms(V)


# --------------------------------------------------------------------------------------
# Correctness gate: eager Hessian vs finite-difference; compiled vs eager
# --------------------------------------------------------------------------------------
def correctness(device):
    print("== correctness gate (grid n=6, f64) ==")
    dtype = torch.float64
    torch.manual_seed(0)
    V0, F, E = plane_grid(6, dtype=dtype, device=device)  # reference (rest) config
    # Build terms ONCE: rest lengths / inertial target are fixed constants from V0.
    terms = build_terms(V0, E, dtype, device)
    # Evaluate at a distinct strained config so gradient AND Hessian are nonzero.
    Vc = (V0 + 0.05 * torch.randn_like(V0)).detach()

    def energy_only(Vx):
        return terms(Vx)

    # gradient (autograd) vs finite differences at the strained config
    Vg = Vc.clone().requires_grad_(True)
    e = energy_only(Vg)
    (g_ad,) = torch.autograd.grad(e, Vg)
    g_fd = fd_gradient(energy_only, Vc).reshape_as(g_ad)
    grad_err = (g_ad - g_fd).abs().max().item() / (g_fd.abs().max().item() + 1e-30)

    # eager sparse Hessian vs finite differences
    H_eager = dense_from_sparse(terms.sparse_hessian(Vc))
    H_fd = fd_hessian(energy_only, Vc)
    hess_fd_err = reldiff(H_fd, H_eager)

    # compiled / cuda_graphs vs eager (same fixed constants)
    terms_c = build_terms(V0, E, dtype, device, compile=True)
    H_comp = dense_from_sparse(terms_c.sparse_hessian(Vc))
    comp_err = reldiff(H_eager, H_comp)
    terms_g = build_terms(V0, E, dtype, device, cuda_graphs=True)
    H_cg = dense_from_sparse(terms_g.sparse_hessian(Vc))
    cg_err = reldiff(H_eager, H_cg)

    print(f"  grad autograd vs FD           relerr = {grad_err:.2e}")
    print(f"  eager Hessian vs FD           relerr = {hess_fd_err:.2e}")
    print(f"  compile vs eager Hessian      relerr = {comp_err:.2e}")
    print(f"  cuda_graphs vs eager Hessian  relerr = {cg_err:.2e}")
    ok = grad_err < 1e-6 and hess_fd_err < 1e-4 and comp_err < 1e-9 and cg_err < 1e-9
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
def make_diff_call(terms, V):
    """One 'Diff' evaluation: gradient (backward) + full sparse Hessian assembly."""
    def call():
        if V.grad is not None:
            V.grad = None
        e = terms(V)
        e.backward()
        H = terms.sparse_hessian(V)
        return H
    return call


def time_variant(n, dtype, device, compile, cuda_graphs, iters, cache_indices=False):
    if compile or cuda_graphs:
        torch._dynamo.reset()  # fresh compile cache per variant -> no cross-variant fallback
    V, F, E = plane_grid(n, dtype=dtype, device=device)
    V = (V + 0.01 * torch.randn_like(V)).requires_grad_(True)
    terms = build_terms(V, E, dtype, device, compile=compile, cuda_graphs=cuda_graphs,
                        cache_indices=cache_indices)
    call = make_diff_call(terms, V)
    warm = warmup_cost_s(call, device)
    ms = time_ms(call, device, iters=iters, repeats=7, warmup=3)
    # decomposition: hessian-only vs grad-only
    def hess_only():
        return terms.sparse_hessian(V)
    def grad_only():
        if V.grad is not None:
            V.grad = None
        terms(V).backward()
    hess_ms = time_ms(hess_only, device, iters=iters, repeats=5, warmup=2)
    grad_ms = time_ms(grad_only, device, iters=iters, repeats=5, warmup=2)
    return dict(ms=ms, hess_ms=hess_ms, grad_ms=grad_ms, warmup_s=warm,
                nV=V.shape[0], nE=E.shape[0], nF=F.shape[0])


def time_pytorch_dense(n, dtype, device, iters):
    """Paper's PyTorch baseline: dense torch.func.hessian over flattened positions."""
    from torch.func import hessian as thess
    V, F, E = plane_grid(n, dtype=dtype, device=device)
    V = V + 0.01 * torch.randn_like(V)
    nV = V.shape[0]
    terms = build_terms(V, E, dtype, device)

    def energy_flat(vflat):
        return terms(vflat.view(nV, 3))

    vflat = V.reshape(-1).contiguous()
    try:
        def call():
            return thess(energy_flat)(vflat)
        warm = warmup_cost_s(call, device)
        ms = time_ms(call, device, iters=max(3, iters // 4), repeats=3, warmup=1)
        return dict(ms=ms, warmup_s=warm, oom=False)
    except torch.cuda.OutOfMemoryError:
        torch.cuda.empty_cache()
        return dict(ms=float("nan"), warmup_s=float("nan"), oom=True)
    except RuntimeError as ex:
        if "out of memory" in str(ex).lower():
            torch.cuda.empty_cache()
            return dict(ms=float("nan"), warmup_s=float("nan"), oom=True)
        raise


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true", help="run correctness gate only")
    ap.add_argument("--sizes", type=int, nargs="+", default=[10, 100, 500, 1000],
                    help="grid sizes n (vertices = n^2)")
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--iters", type=int, default=20)
    ap.add_argument("--no-dense", action="store_true", help="skip PyTorch dense baseline")
    ap.add_argument("--no-check", action="store_true",
                    help="skip correctness gate (use when isolating one size per process)")
    ap.add_argument("--csv", default=None)
    ap.add_argument("--csv-append", action="store_true",
                    help="append to --csv (header only if file is new/empty)")
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "f32" else torch.float64
    print(f"device={device} dtype={args.dtype} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}\n")

    if not args.no_check:
        if not correctness(device):
            print("CORRECTNESS FAILED — aborting timing.")
            sys.exit(1)
        torch._dynamo.reset()  # don't let gate's compiles pollute timing caches
    if args.check:
        return

    print("\n== timing: Diff (grad + full sparse Hessian) per evaluation, ms ==")
    hdr = (f"{'n':>5} {'nV':>9} {'nE':>9} {'variant':<13} "
           f"{'diff_ms':>10} {'hess_ms':>10} {'grad_ms':>9} {'warmup_s':>9} {'speedup':>8}")
    print(hdr)
    rows = []
    for n in args.sizes:
        base = None
        for name, cflags in [("eager", (False, False, False)),
                             ("compile", (True, False, False)),
                             ("cuda_graphs", (False, True, False)),
                             ("compile+cache", (True, False, True)),
                             ("cuda_graphs+cache", (False, True, True))]:
            try:
                r = time_variant(n, dtype, device, cflags[0], cflags[1], args.iters,
                                 cache_indices=cflags[2])
            except RuntimeError as ex:
                if "out of memory" in str(ex).lower():
                    torch.cuda.empty_cache()
                    print(f"{n:>5} {'':>9} {'':>9} {name:<13} OOM")
                    continue
                raise
            if name == "eager":
                base = r["ms"]
            sp = base / r["ms"] if base else float("nan")
            print(f"{n:>5} {r['nV']:>9} {r['nE']:>9} {name:<13} "
                  f"{r['ms']:>10.3f} {r['hess_ms']:>10.3f} {r['grad_ms']:>9.3f} "
                  f"{r['warmup_s']:>9.2f} {sp:>7.2f}x")
            rows.append(dict(n=n, dtype=args.dtype, variant=name, **r,
                             speedup_vs_eager=sp))
            sync(device)
            torch.cuda.empty_cache() if device == "cuda" else None
        if not args.no_dense:
            d = time_pytorch_dense(n, dtype, device, args.iters)
            tag = "OOM" if d["oom"] else f"{d['ms']:>10.3f} ms"
            print(f"{n:>5} {'':>9} {'':>9} {'pytorch_dense':<13} {tag}")
            rows.append(dict(n=n, dtype=args.dtype, variant="pytorch_dense",
                             ms=d["ms"], warmup_s=d["warmup_s"], oom=d["oom"]))
            torch.cuda.empty_cache() if device == "cuda" else None
        print()

    if args.csv:
        keys = ["n", "dtype", "variant", "nV", "nE", "nF", "ms", "hess_ms",
                "grad_ms", "warmup_s", "speedup_vs_eager", "oom"]
        mode = "a" if args.csv_append and os.path.exists(args.csv) and os.path.getsize(args.csv) > 0 else "w"
        with open(args.csv, mode, newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=keys, extrasaction="ignore")
            if mode == "w":
                w.writeheader()
            w.writerows(rows)
        print(f"{'appended' if mode == 'a' else 'wrote'} {len(rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
