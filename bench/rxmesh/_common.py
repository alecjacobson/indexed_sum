"""Shared utilities for the RXMesh-vs-IndexedSum comparison benchmarks.

Mesh generation, robust GPU timing (CUDA events, median over repeats, warmup
excluded), and finite-difference correctness checks. Kept dependency-light
(torch + numpy + igl) and reused across the per-app bench scripts.
"""
import statistics
import time

import numpy as np
import torch

try:
    import igl
    _HAVE_IGL = True
except Exception:  # pragma: no cover
    _HAVE_IGL = False


# --------------------------------------------------------------------------------------
# Meshes
# --------------------------------------------------------------------------------------
def plane_grid(n, dtype=torch.float64, device="cpu"):
    """An n x n triangulated grid embedded in 3D (z=0): n**2 vertices.

    Matches the topology RXMesh's MassSpring generates via create_plane(n, n, ...):
    a regular grid of n**2 vertices triangulated into 2*(n-1)**2 faces. Returns
    (V[n**2,3], F[·,3], E[·,2]) as torch tensors. Vertex count is exactly n**2 so
    the paper's "n**2 vertices" sizes map to grid size n.
    """
    if _HAVE_IGL:
        V2, F = igl.triangulated_grid(n, n)  # V2:(n*n,2) in [0,1]^2, F:(2(n-1)^2,3)
        V = np.zeros((V2.shape[0], 3), dtype=np.float64)
        V[:, :2] = V2
        F = np.asarray(F, dtype=np.int64)
    else:  # pragma: no cover - fallback grid builder
        xs, ys = np.meshgrid(np.linspace(0, 1, n), np.linspace(0, 1, n), indexing="xy")
        V = np.stack([xs.ravel(), ys.ravel(), np.zeros(n * n)], axis=1)
        faces = []
        idx = np.arange(n * n).reshape(n, n)
        for i in range(n - 1):
            for j in range(n - 1):
                a, b, c, d = idx[i, j], idx[i, j + 1], idx[i + 1, j], idx[i + 1, j + 1]
                faces += [[a, b, d], [a, d, c]]
        F = np.asarray(faces, dtype=np.int64)
    # undirected unique edges
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    e = np.sort(e, axis=1)
    E = np.unique(e, axis=0)
    Vt = torch.tensor(V, dtype=dtype, device=device)
    Ft = torch.tensor(F, dtype=torch.int64, device=device)
    Et = torch.tensor(E, dtype=torch.int64, device=device)
    return Vt, Ft, Et


def load_mesh(path, dtype=torch.float64, device="cpu"):
    """Read an OBJ/OFF/PLY mesh -> (V[·,3], F[·,3], E[·,2]) torch tensors."""
    V, F = igl.read_triangle_mesh(str(path))
    V = np.asarray(V, dtype=np.float64)
    F = np.asarray(F, dtype=np.int64)
    e = np.concatenate([F[:, [0, 1]], F[:, [1, 2]], F[:, [2, 0]]], axis=0)
    E = np.unique(np.sort(e, axis=1), axis=0)
    return (torch.tensor(V, dtype=dtype, device=device),
            torch.tensor(F, dtype=torch.int64, device=device),
            torch.tensor(E, dtype=torch.int64, device=device))


# --------------------------------------------------------------------------------------
# Timing
# --------------------------------------------------------------------------------------
def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize()


def time_ms(fn, device, iters=20, repeats=7, warmup=3):
    """Median per-call time in ms. `warmup` calls excluded (they also pay any
    one-time torch.compile / CUDA-graph capture cost). GPU uses CUDA events;
    CPU uses perf_counter. Median over `repeats` blocks of `iters` calls each.
    """
    for _ in range(warmup):
        fn()
    sync(device)
    samples = []
    if str(device).startswith("cuda"):
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn()
            e.record()
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e) / iters)
    else:
        for _ in range(repeats):
            t = time.perf_counter()
            for _ in range(iters):
                fn()
            samples.append((time.perf_counter() - t) / iters * 1e3)
    return statistics.median(samples)


def warmup_cost_s(fn, device):
    """Wall time of the first call (includes any compile / capture cost)."""
    sync(device)
    t = time.perf_counter()
    fn()
    sync(device)
    return time.perf_counter() - t


# --------------------------------------------------------------------------------------
# Correctness
# --------------------------------------------------------------------------------------
def dense_from_sparse(H):
    return H.coalesce().to_dense() if H.is_sparse else H


def fd_gradient(energy_fn, V, eps=1e-6):
    """Central-difference gradient of a scalar energy_fn(V) -> flat (dof,) tensor.
    AD-independent ground truth. Uses float64 V."""
    V = V.detach().clone()
    flat = V.reshape(-1)
    g = torch.zeros_like(flat)
    for i in range(flat.numel()):
        orig = flat[i].item()
        flat[i] = orig + eps
        fp = energy_fn(V).item()
        flat[i] = orig - eps
        fm = energy_fn(V).item()
        flat[i] = orig
        g[i] = (fp - fm) / (2 * eps)
    return g


def fd_hessian(energy_fn, V, eps=1e-4):
    """Central-difference Hessian of scalar energy_fn(V) -> dense (dof,dof).
    AD-independent ground truth for small problems. Uses float64 V."""
    V = V.detach().clone()
    flat = V.reshape(-1)
    n = flat.numel()

    def grad_at():
        Vc = V.detach().clone().requires_grad_(True)
        e = energy_fn(Vc)
        (gg,) = torch.autograd.grad(e, Vc)
        return gg.reshape(-1).detach()

    H = torch.zeros(n, n, dtype=flat.dtype, device=flat.device)
    for i in range(n):
        orig = flat[i].item()
        flat[i] = orig + eps
        gp = grad_at()
        flat[i] = orig - eps
        gm = grad_at()
        flat[i] = orig
        H[i] = (gp - gm) / (2 * eps)
    return 0.5 * (H + H.T)


def reldiff(ref, out):
    ref = dense_from_sparse(ref)
    out = dense_from_sparse(out)
    denom = ref.abs().max().item()
    return (ref - out).abs().max().item() / (denom + 1e-30)
