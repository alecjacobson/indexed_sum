"""IndexedSum side of the manifold-optimization comparison (RXMesh AD paper).

RXMesh's ManiOpt app does spherical parameterization: minimize an injectivity-barrier +
Dirichlet + pole energy over 2D tangent coordinates that are *retracted* to the sphere. It is
a Newton method with an **assembled** Hessian (add_term<FV,true> + eval_terms()), so unlike
Param it is directly analogous to what IndexedSum does.

Crucially, the energy contains a **3x3 determinant** ( volume = det([a,b,c])/6 ) -- exactly the
`torch.linalg.det`-under-vmap forward-mode bug documented in bench/RESULTS.md (the neohookean
case). This is the second, independent energy showing why the RXMesh paper's description of
IndexedSum as "reverse-mode AD" is imprecise: the default path is forward-over-reverse, and
that path miscomputes a 3x3-det Hessian unless the determinant is written with the elementary
`indexed_sum.det` helper. We verify:
  (1) elementary-det Hessian matches finite differences;
  (2) the torch.linalg.det formulation gives a WRONG Hessian (the bug);
  (3) eager vs compile vs cuda_graphs timing (correct + capturable only with the helper).

Runs on the same mesh RXMesh uses (giraffe.obj + giraffe_embedding.obj for the sphere init).

Usage:
  python bench/rxmesh/maniopt_bench.py --check
  python bench/rxmesh/maniopt_bench.py --mesh <obj> --embed <obj>
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
from indexed_sum.det import det as det_helper  # noqa: E402
from bench.rxmesh._common import (  # noqa: E402
    load_mesh, time_ms, warmup_cost_s, fd_hessian, fd_gradient, reldiff,
    dense_from_sparse, sync,
)

RX_INPUT = "/home/horde/projects/RXMesh/input"


def tangent_basis(S):
    """Replicate RXMesh any_tangent: b1 = normalize(axis_min . cross S), b2 = S x b1."""
    axes = torch.eye(3, dtype=S.dtype, device=S.device)
    dots2 = (S @ axes.T) ** 2                      # (nV,3)
    idx = dots2.argmin(dim=1)
    ax = axes[idx]                                 # (nV,3)
    b1 = torch.cross(ax, S, dim=1)
    b1 = b1 / b1.norm(dim=1, keepdim=True)
    b2 = torch.cross(S, b1, dim=1)
    return b1, b2


def build_term(F, K, detfn, **flags):
    def maniopt(uv, k):                            # uv:(3,2), k:(3,9)
        def ret(j):
            s = k[j, :3]
            b1 = k[j, 3:6]
            b2 = k[j, 6:9]
            p = s + uv[j, 0] * b1 + uv[j, 1] * b2
            return p / (p * p).sum().sqrt()
        a, b, c = ret(0), ret(1), ret(2)
        Mv = torch.stack([a, b, c], dim=1)         # 3x3, columns a,b,c
        vol = detfn(Mv) / 6.0
        # RXMesh returns +inf when vol<=0 (injectivity barrier); a few faces of the giraffe
        # sphere-embedding init are inverted. Clamp to a tiny floor so the energy stays finite
        # (does not change the op count / timing, nor the Hessian of feasible faces).
        vol = torch.clamp(vol, min=1e-9)
        E = -0.1 * torch.log(vol)
        E = E + (a - b).pow(2).sum() + (b - c).pow(2).sum() + (c - a).pow(2).sum()
        E = E + a[1] ** 2 + b[1] ** 2 + c[1] ** 2   # equator term
        return E
    return IndexedSum(maniopt, F, per_variable_constants=K, **flags)


def setup(mesh, embed, dtype, device):
    _, F, _ = load_mesh(mesh, dtype=dtype, device=device)
    Vemb, Fe, _ = load_mesh(embed, dtype=dtype, device=device)
    S = Vemb / Vemb.norm(dim=1, keepdim=True)      # unit sphere positions
    B1, B2 = tangent_basis(S)
    K = torch.cat([S, B1, B2], dim=1).detach()     # (nV,9)
    return F, K, S.shape[0]


def correctness(device):
    print("== correctness gate (giraffe, f64): manifold-opt spherical energy ==")
    dtype = torch.float64
    torch.manual_seed(0)
    F, K, nV = setup(f"{RX_INPUT}/giraffe.obj", f"{RX_INPUT}/giraffe_embedding.obj", dtype, device)
    # tiny tangent coords -> feasible (volumes > 0), smooth log
    uv = (1e-3 * torch.randn(nV, 2, dtype=dtype, device=device)).detach()

    # Finite differences are O(dof^2): use a small patch of faces over the first ~40 vertices,
    # keeping only *feasible* faces (sphere volume > 0) so the log-barrier energy is smooth
    # (the giraffe embedding init has a few inverted faces).
    def face_volumes(Fx, Kx, uvx):
        selK, seluv = Kx[Fx], uvx[Fx]
        def ret(t, k):
            p = k[:3] + t[0] * k[3:6] + t[1] * k[6:9]
            return p / (p * p).sum().sqrt()
        vols = []
        for f in range(Fx.shape[0]):
            a = ret(seluv[f, 0], selK[f, 0]); b = ret(seluv[f, 1], selK[f, 1]); c = ret(seluv[f, 2], selK[f, 2])
            vols.append((det_helper(torch.stack([a, b, c], dim=1)) / 6).item())
        return torch.tensor(vols)

    vmax = 40
    Fsub = F[(F < vmax).all(dim=1)]
    Fsub = Fsub[face_volumes(Fsub, K, uv) > 1e-6]        # feasible only
    used = torch.unique(Fsub)
    remap = -torch.ones(nV, dtype=torch.long, device=device)
    remap[used] = torch.arange(used.numel(), device=device)
    Fsub_r = remap[Fsub]
    Ksub = K[used]
    uvsub = uv[used].clone()
    term_sub = build_term(Fsub_r, Ksub, det_helper)

    def energy_sub(x):
        return term_sub(x)

    # gradient vs FD (AD-independent, first order) -- the primary correctness anchor
    uvg = uvsub.clone().requires_grad_(True)
    (g_ad,) = torch.autograd.grad(term_sub(uvg), uvg)
    g_fd = fd_gradient(energy_sub, uvsub).reshape_as(g_ad)
    grad_err = (g_ad - g_fd).abs().max().item() / (g_fd.abs().max().item() + 1e-30)

    H_eager = dense_from_sparse(term_sub.sparse_hessian(uvsub))
    H_fd = fd_hessian(energy_sub, uvsub, eps=1e-5)
    fd_err = reldiff(H_fd, H_eager)

    H_comp = dense_from_sparse(build_term(Fsub_r, Ksub, det_helper, compile=True).sparse_hessian(uvsub))
    comp_err = reldiff(H_eager, H_comp)
    H_cg = dense_from_sparse(build_term(Fsub_r, Ksub, det_helper, cuda_graphs=True).sparse_hessian(uvsub))
    cg_err = reldiff(H_eager, H_cg)

    # does the 3x3 det bug trigger here? compare torch.linalg.det vs elementary-det (both eager)
    H_ldet = dense_from_sparse(build_term(Fsub_r, Ksub, torch.linalg.det).sparse_hessian(uvsub))
    ldet_vs_helper = reldiff(H_eager, H_ldet)

    print(f"  patch: {Fsub_r.shape[0]} feasible faces, {used.numel()} verts")
    print(f"  grad autograd vs FD                  relerr = {grad_err:.2e}")
    print(f"  elementary-det eager Hessian vs FD   relerr = {fd_err:.2e}")
    print(f"  compile vs eager                     relerr = {comp_err:.2e}")
    print(f"  cuda_graphs vs eager                 relerr = {cg_err:.2e}")
    print(f"  torch.linalg.det vs helper (eager)   relerr = {ldet_vs_helper:.2e}  "
          f"({'differs -> det bug triggers' if ldet_vs_helper > 1e-2 else 'agree -> bug does NOT trigger here'})")
    ok = grad_err < 1e-5 and fd_err < 1e-3 and comp_err < 1e-9 and cg_err < 1e-9
    print(f"  => {'PASS' if ok else 'FAIL'}")
    return ok


def time_all(mesh, embed, dtype, device, iters):
    F, K, nV = setup(mesh, embed, dtype, device)
    print(f"\n== per-iteration derivative cost on {os.path.basename(mesh)} "
          f"(nV={nV}, nF={F.shape[0]}): grad + sparse Hessian, ms ==")
    print(f"{'variant':<18} {'diff_ms':>10} {'hess_ms':>10} {'warmup_s':>9} {'speedup':>8}")
    base = None
    for name, cf in [("eager", (False, False, False)), ("compile", (True, False, False)),
                     ("cuda_graphs", (False, True, False)),
                     ("compile+cache", (True, False, True)),
                     ("cuda_graphs+cache", (False, True, True))]:
        if cf[0] or cf[1]:
            torch._dynamo.reset()
        uv = (1e-3 * torch.randn(nV, 2, dtype=dtype, device=device)).detach().requires_grad_(True)
        term = build_term(F, K, det_helper, compile=cf[0], cuda_graphs=cf[1], cache_indices=cf[2])

        def diff_call():
            g = term.dense_gradient(uv)  # honors compile/cuda_graphs/cache_indices like the Hessian
            return term.sparse_hessian(uv)
        warm = warmup_cost_s(diff_call, device)
        ms = time_ms(diff_call, device, iters=iters, repeats=7, warmup=3)
        hess_ms = time_ms(lambda: term.sparse_hessian(uv), device, iters=iters, repeats=5, warmup=2)
        if name == "eager":
            base = ms
        print(f"{name:<18} {ms:>10.3f} {hess_ms:>10.3f} {warm:>9.2f} {base / ms:>7.2f}x")
        sync(device)
        if device == "cuda":
            torch.cuda.empty_cache()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--mesh", default=f"{RX_INPUT}/giraffe.obj")
    ap.add_argument("--embed", default=f"{RX_INPUT}/giraffe_embedding.obj")
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--iters", type=int, default=30)
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
    time_all(args.mesh, args.embed, dtype, device, args.iters)


if __name__ == "__main__":
    main()
