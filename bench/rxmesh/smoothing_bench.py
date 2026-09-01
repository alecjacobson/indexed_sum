"""IndexedSum side of the Laplacian-smoothing comparison (RXMesh AD paper, Fig 6).

RXMesh's Smoothing app is **gradient-only** (gradient descent; DiffScalarProblem<...,false>,
add_term without Hessian). Its timed quantity is eval_terms() = the gradient of the total
energy per iteration. Default energy is edge-based: E = sum_edges |x0 - x1|^2.

Important fairness note: IndexedSum's `compile` / `cuda_graphs` switches only wire into
`sparse_hessian` -- there is **no Hessian here**, so those switches do not apply to a
gradient-only workload. We therefore measure IndexedSum's eager autograd gradient (forward +
backward), which is exactly what the library offers for this task, and report it as-is. For
context we also time a "what-if" torch.compile of the vmapped forward+grad (NOT part of the
IndexedSum API) to show the ceiling if a compiled-gradient path were added.

Usage:
  python bench/rxmesh/smoothing_bench.py --sizes 100 500 1000
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
from bench.rxmesh._common import plane_grid, time_ms, fd_gradient, warmup_cost_s  # noqa: E402


def edge_energy(v):            # v: (2,3) -> scalar, matches RXMesh default (dist_sq)
    d = v[0] - v[1]
    return (d * d).sum()


def correctness(device):
    print("== correctness gate (grid n=6, f64): gradient vs finite differences ==")
    dtype = torch.float64
    torch.manual_seed(0)
    V0, F, E = plane_grid(6, dtype=dtype, device=device)
    term = IndexedSum(edge_energy, E)
    Vc = (V0 + 0.05 * torch.randn_like(V0)).detach()

    Vg = Vc.clone().requires_grad_(True)
    (g_ad,) = torch.autograd.grad(term(Vg), Vg)
    g_fd = fd_gradient(lambda X: term(X), Vc).reshape_as(g_ad)
    err = (g_ad - g_fd).abs().max().item() / (g_fd.abs().max().item() + 1e-30)
    g_dense = term.dense_gradient(Vc).reshape_as(g_ad)
    derr = (g_dense - g_ad).abs().max().item() / (g_ad.abs().max().item() + 1e-30)
    print(f"  grad autograd vs FD       relerr = {err:.2e}")
    print(f"  dense_gradient vs autograd relerr = {derr:.2e}  "
          f"=> {'PASS' if err < 1e-6 and derr < 1e-9 else 'FAIL'}")
    return err < 1e-6 and derr < 1e-9


def time_autograd_backward(n, dtype, device, iters):
    """The paper-era gradient path: energy forward + autograd backward (no dense_gradient)."""
    V, F, E = plane_grid(n, dtype=dtype, device=device)
    V = V.detach().requires_grad_(True)
    term = IndexedSum(edge_energy, E)

    def call():
        if V.grad is not None:
            V.grad = None
        term(V).backward()
    return time_ms(call, device, iters=iters, repeats=7, warmup=3), V.shape[0], E.shape[0]


def time_dense_gradient(n, dtype, device, iters, compile=False, cuda_graphs=False, cache_indices=False):
    """The compiled/capturable gradient path (IndexedSum.dense_gradient, merged in the core
    library). For a gradient-only workload this IS accelerated by compile/cuda_graphs/cache."""
    if compile or cuda_graphs:
        torch._dynamo.reset()
    V, F, E = plane_grid(n, dtype=dtype, device=device)
    V = V.detach()
    term = IndexedSum(edge_energy, E, compile=compile, cuda_graphs=cuda_graphs,
                      cache_indices=cache_indices)

    def call():
        return term.dense_gradient(V)
    warmup_cost_s(call, device)
    return time_ms(call, device, iters=iters, repeats=7, warmup=3)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sizes", type=int, nargs="+", default=[100, 500, 1000])
    ap.add_argument("--dtype", choices=["f32", "f64"], default="f32")
    ap.add_argument("--iters", type=int, default=30)
    args = ap.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float32 if args.dtype == "f32" else torch.float64
    print(f"device={device} dtype={args.dtype} gpu="
          f"{torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}\n")
    if not correctness(device):
        sys.exit(1)
    torch._dynamo.reset()
    print("\n== gradient time per iteration, ms (edge energy |x0-x1|^2) ==")
    print(f"{'n':>6} {'nV':>9} {'nE':>9} {'autograd':>10} {'dense_eager':>12} "
          f"{'dense_compile':>14} {'dense_cg+cache':>15}")
    for n in args.sizes:
        ms_bw, nV, nE = time_autograd_backward(n, dtype, device, args.iters)
        ms_e = time_dense_gradient(n, dtype, device, args.iters)
        ms_c = time_dense_gradient(n, dtype, device, args.iters, compile=True)
        ms_cg = time_dense_gradient(n, dtype, device, args.iters, cuda_graphs=True, cache_indices=True)
        print(f"{n:>6} {nV:>9} {nE:>9} {ms_bw:>10.3f} {ms_e:>12.3f} {ms_c:>14.3f} {ms_cg:>15.3f}")


if __name__ == "__main__":
    main()
