"""
Tests for IndexedSum.dense_gradient and its opt-in compile / cuda_graphs acceleration.

`dense_gradient` assembles the dense gradient vector `[num_vars * dim]` from batched
per-element gradients (`vmap(jacrev(g))`) scattered into the global vector -- mirroring
`sparse_hessian`, and accelerated by the same `compile`/`cuda_graphs` switches. Facts locked in:

  1. The eager `dense_gradient` equals `torch.autograd.grad` of the scalar energy, AND a
     finite-difference gradient (an AD-independent ground truth).
  2. `compile=True` and `cuda_graphs=True` reproduce the eager gradient to floating-point
     rounding (gradient is plain reverse-mode, so compilation is exact -- no Hessian-style
     forward/reverse reformulation is needed).
  3. It composes with `SumNode` (the `+` of IndexedSums).

Run:  pytest tests/test_gradient_compile.py -v
"""
import pytest
import torch

from indexed_sum import IndexedSum


# ------------------------------------------------------------------ workloads
def spring(v):
    d = v[1] - v[0]
    return 0.5 * (d * d).sum()


def area(v):  # non-quadratic; gradient genuinely depends on the point
    a = torch.linalg.norm(v[1] - v[0])
    b = torch.linalg.norm(v[2] - v[1])
    c = torch.linalg.norm(v[2] - v[0])
    s = (a + b + c) / 2
    return torch.sqrt(torch.clamp(s * (s - a) * (s - b) * (s - c), min=1e-12))


SMOOTH_WORKLOADS = [("spring", spring, 2, 2), ("area", area, 3, 3)]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.float64]


# ------------------------------------------------------------------ helpers
def _cloud_inputs(local_size, dim, device, dtype, n=64, seed=0):
    """A shared vertex cloud + n elements with distinct indices."""
    g = torch.Generator().manual_seed(seed)
    N = n + local_size
    V = torch.randn(N, dim, generator=g, dtype=dtype).to(device)
    idx = torch.stack([torch.randperm(N, generator=g)[:local_size] for _ in range(n)]).to(device)
    return V, idx


def _reldiff(ref, out):
    return (ref - out).abs().max().item() / (ref.abs().max().item() + 1e-30)


def _rtol(dtype):
    return 2e-5 if dtype == torch.float32 else 1e-11


def _autograd_gradient(fn, V, idx):
    """The eager reference: gradient of the scalar summed energy via torch.autograd.grad."""
    Vg = V.clone().requires_grad_(True)
    energy = IndexedSum(fn, idx)(Vg).sum()
    return torch.autograd.grad(energy, Vg)[0].reshape(-1)


def _fd_gradient(total_energy, x, eps=1e-6):
    """Finite-difference gradient (central diff) -- an AD-independent ground truth. `x` is a
    flat vector; `total_energy` takes the flat vector and returns the scalar energy."""
    n = x.numel()
    g = torch.zeros(n, dtype=x.dtype, device=x.device)
    for i in range(n):
        e = torch.zeros(n, dtype=x.dtype, device=x.device)
        e[i] = eps
        g[i] = (total_energy(x + e) - total_energy(x - e)) / (2 * eps)
    return g


# (compile, cuda_graphs) combinations to exercise. cuda_graphs requires a GPU.
_COMPILE_OPTS = [dict(compile=True)]
if torch.cuda.is_available():
    _COMPILE_OPTS.append(dict(cuda_graphs=True))  # cuda_graphs implies compilation


def _opt_id(opt):
    return "cuda_graphs" if opt.get("cuda_graphs") else "compile"


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name,fn,local_size,dim", SMOOTH_WORKLOADS)
def test_dense_gradient_matches_autograd(name, fn, local_size, dim, device):
    """Eager dense_gradient == torch.autograd.grad of the scalar energy (f64)."""
    V, idx = _cloud_inputs(local_size, dim, device, torch.float64)
    ref = _autograd_gradient(fn, V, idx)
    got = IndexedSum(fn, idx).dense_gradient(V)
    assert got.shape == ref.shape
    assert _reldiff(ref, got) < _rtol(torch.float64)


@pytest.mark.parametrize("name,fn,local_size,dim", SMOOTH_WORKLOADS)
def test_dense_gradient_matches_finite_differences(name, fn, local_size, dim):
    """Eager dense_gradient == a finite-difference gradient (AD-independent ground truth, f64)."""
    V, idx = _cloud_inputs(local_size, dim, "cpu", torch.float64, n=16)
    num_vars, d = V.shape

    def total(z):
        z = z.view(num_vars, d)
        return sum(fn(z[idx[i]]) for i in range(idx.shape[0]))

    ref = _fd_gradient(total, V.reshape(-1))
    got = IndexedSum(fn, idx).dense_gradient(V)
    assert _reldiff(ref, got) < 1e-5


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("opt", _COMPILE_OPTS, ids=_opt_id)
@pytest.mark.parametrize("name,fn,local_size,dim", SMOOTH_WORKLOADS)
def test_compiled_gradient_matches_eager(name, fn, local_size, dim, opt, dtype, device):
    """dense_gradient with compile / cuda_graphs == the eager default, to rounding."""
    if opt.get("cuda_graphs") and device != "cuda":
        pytest.skip("cuda_graphs requires CUDA")
    V, idx = _cloud_inputs(local_size, dim, device, dtype)

    ref = IndexedSum(fn, idx).dense_gradient(V)
    obj = IndexedSum(fn, idx, **opt)
    g1 = obj.dense_gradient(V)
    g2 = obj.dense_gradient(V)  # reuse compiled kernel; clone must keep g1 valid

    tol = 1e-6 if dtype == torch.float32 else 1e-12
    assert _reldiff(ref, g1) < max(tol, _rtol(dtype))
    # g1 must not be corrupted by g2 (CUDA-graph memory reuse) -> the .clone() in _batched_gradient
    assert _reldiff(g1, g2) < max(tol, _rtol(dtype))


@pytest.mark.parametrize("opt", _COMPILE_OPTS, ids=_opt_id)
def test_dense_gradient_cache_indices_matches(opt):
    """cache_indices=True must not change the assembled gradient (composes with compile)."""
    device = "cuda" if opt.get("cuda_graphs") else "cpu"
    V, idx = _cloud_inputs(2, 2, device, torch.float64)
    ref = IndexedSum(spring, idx).dense_gradient(V)
    obj = IndexedSum(spring, idx, cache_indices=True, **opt)
    g1 = obj.dense_gradient(V)
    assert obj._cached_grad_indices is not None
    g2 = obj.dense_gradient(V + 0.5)  # new values, same topology -> cached indices reused
    ref2 = IndexedSum(spring, idx).dense_gradient(V + 0.5)
    assert _reldiff(ref, g1) < 1e-11
    assert _reldiff(ref2, g2) < 1e-11


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("opt", [dict()] + _COMPILE_OPTS, ids=lambda o: _opt_id(o) if o else "eager")
def test_sumnode_dense_gradient(opt, device):
    """(a + b).dense_gradient == gradient of the summed energy, eager and compiled."""
    if opt.get("cuda_graphs") and device != "cuda":
        pytest.skip("cuda_graphs requires CUDA")
    V, idx_s = _cloud_inputs(2, 2, device, torch.float64, seed=1)
    _, idx_a = _cloud_inputs(3, 3, device, torch.float64, seed=2)
    # match vertex count / dim: rebuild on a common cloud
    N, d = 66, 3
    g = torch.Generator().manual_seed(3)
    V = torch.randn(N, d, generator=g, dtype=torch.float64).to(device)
    idx_s = torch.stack([torch.randperm(N, generator=g)[:2] for _ in range(30)]).to(device)
    idx_a = torch.stack([torch.randperm(N, generator=g)[:3] for _ in range(30)]).to(device)

    a = IndexedSum(spring, idx_s, **opt)
    b = IndexedSum(area, idx_a, **opt)
    node = a + b

    Vg = V.clone().requires_grad_(True)
    energy = (a(Vg) + b(Vg)).sum()
    ref = torch.autograd.grad(energy, Vg)[0].reshape(-1)

    got = node.dense_gradient(V)
    assert got.shape == ref.shape
    assert _reldiff(ref, got) < 1e-11
