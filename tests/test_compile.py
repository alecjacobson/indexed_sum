"""
Regression tests for a torch.compile-based Hessian path for IndexedSum.

Facts these tests lock in (see bench/RESULTS.md for the full study):

  1. torch.func.hessian == jacfwd(jacrev(f)) (forward-over-reverse, what the library uses)
     does NOT compile under dynamo in torch 2.11 (`_fw_primal` inference-mode assert). To
     compile at all we reformulate the Hessian as jacrev(jacrev(f)) (reverse-over-reverse).

  2. Compilation is *faithful*: compiled jacrev(jacrev(f)) reproduces eager jacrev(jacrev(f))
     essentially exactly, for every workload / dtype / device.

  3. For smooth, well-conditioned energies the reverse-mode Hessian equals the library's
     forward-mode Hessian, so the compiled kernel is a drop-in replacement -- including the
     full sparse-assembly path of IndexedSum.sparse_hessian.

  4. CAVEAT (test_neohookean_*): forward- and reverse-mode Hessians can disagree on
     *degenerate* elements (e.g. near-singular tets hitting a clamp), by AD-mode rounding,
     independent of compilation. Compiled output always tracks eager reverse mode.

Run:  pytest tests/test_compile.py -v
"""
import pytest
import torch
from torch.func import vmap, hessian, jacrev

from indexed_sum import IndexedSum


# ------------------------------------------------------------------ workloads
def spring(v):
    d = v[1] - v[0]
    return 0.5 * (d * d).sum()


def area(v):  # non-quadratic; Hessian genuinely depends on the point
    a = torch.linalg.norm(v[1] - v[0])
    b = torch.linalg.norm(v[2] - v[1])
    c = torch.linalg.norm(v[2] - v[0])
    s = (a + b + c) / 2
    return torch.sqrt(torch.clamp(s * (s - a) * (s - b) * (s - c), min=1e-12))


def neohookean(v):
    F = v[1:] - v[0:1]
    J = torch.linalg.det(F)
    return (F * F).sum() - 3 - 2 * torch.log(torch.clamp(J, min=1e-3)) + (J - 1) ** 2


SMOOTH_WORKLOADS = [("spring", spring, 2, 2), ("area", area, 3, 3)]
ALL_WORKLOADS = SMOOTH_WORKLOADS + [("neohookean", neohookean, 4, 3)]

DEVICES = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
DTYPES = [torch.float32, torch.float64]


# ------------------------------------------------------------------ helpers
def _reshaped(fn, local_size, dim):
    def r(inp):
        return fn(inp.view(local_size, dim))
    return r


def rev_hessian_kernel(reshaped):
    """The compilable reverse-over-reverse batched Hessian kernel."""
    return lambda sel: vmap(jacrev(jacrev(reshaped)))(sel)


def _cloud_inputs(local_size, dim, device, dtype, n=64, seed=0):
    """A shared vertex cloud + n elements with distinct indices. Generic geometry (fine for
    the smooth workloads); may include degenerate elements for neohookean by design."""
    g = torch.Generator().manual_seed(seed)
    N = n + local_size
    V = torch.randn(N, dim, generator=g, dtype=dtype).to(device)
    idx = torch.stack([torch.randperm(N, generator=g)[:local_size] for _ in range(n)]).to(device)
    return V, idx


def _well_conditioned_tets(device, dtype, n=64, seed=0):
    """n non-degenerate tetrahedra: a regular tet + small per-vertex noise (detF ~ 1)."""
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=dtype)
    sel = base.unsqueeze(0) + 0.05 * torch.randn(n, 4, 3, generator=g, dtype=dtype)
    return sel.to(device)


def _reldiff(ref, out):
    return (ref - out).abs().max().item() / (ref.abs().max().item() + 1e-30)


def _rtol(dtype):
    return 2e-4 if dtype == torch.float32 else 1e-9


@pytest.fixture(autouse=True)
def _reset_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


# ------------------------------------------------------------------ tests
@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn,local_size,dim", ALL_WORKLOADS)
def test_compile_is_faithful_to_eager_reverse(name, fn, local_size, dim, dtype, device):
    """Compiled jacrev(jacrev(f)) == eager jacrev(jacrev(f)). Isolates compile correctness."""
    reshaped = _reshaped(fn, local_size, dim)
    V, idx = _cloud_inputs(local_size, dim, device, dtype)
    sel = V[idx]

    eager = rev_hessian_kernel(reshaped)(sel)
    compiled = torch.compile(rev_hessian_kernel(reshaped))(sel).clone()

    assert not torch.isnan(compiled).any()
    assert _reldiff(eager, compiled) < _rtol(dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn,local_size,dim", SMOOTH_WORKLOADS)
def test_compiled_matches_library_hessian(name, fn, local_size, dim, dtype, device):
    """For smooth energies, compiled reverse-mode == library forward-mode (vmap(hessian))."""
    reshaped = _reshaped(fn, local_size, dim)
    V, idx = _cloud_inputs(local_size, dim, device, dtype)
    sel = V[idx]

    ref = vmap(hessian(reshaped))(sel)  # library formulation
    compiled = torch.compile(rev_hessian_kernel(reshaped))(sel).clone()
    assert _reldiff(ref, compiled) < _rtol(dtype)


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("name,fn,local_size,dim", SMOOTH_WORKLOADS)
def test_compiled_sparse_hessian_matches_library(name, fn, local_size, dim, dtype, device):
    """End-to-end: a compiled reverse-mode kernel dropped into the sparse scatter reproduces
    IndexedSum.sparse_hessian exactly (same nnz, same values)."""
    reshaped = _reshaped(fn, local_size, dim)
    V, idx = _cloud_inputs(local_size, dim, device, dtype)
    Vg = V.clone().requires_grad_(True)

    H_eager = IndexedSum(fn, idx).sparse_hessian(Vg).coalesce()

    # Mirror indexed_sum.sparse_hessian's index math, compiled kernel for the blocks.
    num_vars, d = V.shape
    sum_length = idx.shape[0]
    dof = num_vars * d
    blocks = torch.compile(rev_hessian_kernel(reshaped))(V[idx]).clone()
    gi = (d * idx[:, :, None] + torch.arange(d, device=device)).reshape(sum_length, local_size * d)
    rows = gi[:, :, None].expand(sum_length, local_size * d, local_size * d).reshape(-1)
    cols = gi[:, None, :].expand(sum_length, local_size * d, local_size * d).reshape(-1)
    H_comp = torch.sparse_coo_tensor(torch.stack([rows, cols]), blocks.reshape(-1),
                                     size=(dof, dof)).coalesce()

    assert H_comp._nnz() == H_eager._nnz()
    assert _reldiff(H_eager.to_dense(), H_comp.to_dense()) < _rtol(dtype)


def _fd_hessian(fn, x, eps=1e-6):
    """Finite-difference Hessian (central diff of the analytic gradient) -- an AD-independent
    ground truth. x is a flat vector; fn takes the flat vector."""
    n = x.numel()

    def grad(z):
        z = z.detach().requires_grad_(True)
        return torch.autograd.grad(fn(z), z)[0]

    H = torch.zeros(n, n, dtype=x.dtype, device=x.device)
    for i in range(n):
        e = torch.zeros(n, dtype=x.dtype, device=x.device)
        e[i] = eps
        H[i] = (grad(x + e) - grad(x - e)) / (2 * eps)
    return 0.5 * (H + H.T)


def test_compiled_reverse_matches_finite_differences_for_det():
    """The compiled reverse-over-reverse Hessian is CORRECT for a torch.det summand, verified
    against an AD-independent finite-difference ground truth. This is the correctness case
    that the library's forward-mode formulation gets wrong (next test)."""
    dtype = torch.float64
    sel = _well_conditioned_tets("cpu", dtype, n=8)
    compiled = torch.compile(rev_hessian_kernel(_reshaped(neohookean, 4, 3)))(sel).clone()
    for i in range(sel.shape[0]):
        Hfd = _fd_hessian(lambda z: neohookean(z.view(4, 3)), sel[i].reshape(12))
        assert _reldiff(Hfd, compiled[i].reshape(12, 12)) < 1e-4


@pytest.mark.xfail(reason="Known bug: forward-mode AD of torch.det under vmap is wrong "
                          "(pytorch#149694). The library's vmap(hessian(...)) uses it. The "
                          "reverse-mode formulation required by torch.compile fixes this. If "
                          "this test starts PASSING, torch fixed the bug -- revisit.",
                   strict=True)
def test_library_forward_hessian_is_correct_for_det_under_vmap():
    """Documents the latent correctness bug: IndexedSum.sparse_hessian (forward-over-reverse
    via torch.func.hessian) disagrees with finite differences for a torch.det summand."""
    dtype = torch.float64
    reshaped = _reshaped(neohookean, 4, 3)
    sel = _well_conditioned_tets("cpu", dtype, n=8)
    lib = vmap(hessian(reshaped))(sel)  # what IndexedSum.sparse_hessian computes
    for i in range(sel.shape[0]):
        Hfd = _fd_hessian(lambda z: neohookean(z.view(4, 3)), sel[i].reshape(12))
        assert _reldiff(Hfd, lib[i].reshape(12, 12)) < 1e-4


def test_forward_over_reverse_still_uncompilable():
    """Documents WHY we can't just wrap the library's own formulation in torch.compile. If a
    future torch makes this pass, revisit the compile strategy (and delete this test)."""
    reshaped = _reshaped(spring, 2, 2)
    V, idx = _cloud_inputs(2, 2, "cpu", torch.float64)
    with pytest.raises(Exception):
        torch.compile(lambda s: vmap(hessian(reshaped))(s), fullgraph=False)(V[idx])


# ---------------------------------------------------------------- IndexedSum(compile=...) API
def _det_energy(v):  # det via the vmap-safe helper, so both eager and compiled are correct
    from indexed_sum.det import det
    F = v[1:] - v[0:1]
    J = det(F)
    return (F * F).sum() - 3 - 2 * torch.log(torch.clamp(J, min=1e-3)) + (J - 1) ** 2


# (compile, cuda_graphs) combinations to exercise. cuda_graphs requires a GPU.
_COMPILE_OPTS = [dict(compile=True)]
if torch.cuda.is_available():
    _COMPILE_OPTS.append(dict(cuda_graphs=True))  # cuda_graphs implies compilation


def _opt_id(opt):
    return "cuda_graphs" if opt.get("cuda_graphs") else "compile"


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("opt", _COMPILE_OPTS, ids=_opt_id)
@pytest.mark.parametrize("name,fn,local_size,dim", SMOOTH_WORKLOADS)
def test_indexed_sum_compile_option_matches_eager(name, fn, local_size, dim, opt, device):
    """IndexedSum(compile=True) and (cuda_graphs=True).sparse_hessian == the eager default."""
    if opt.get("cuda_graphs") and device != "cuda":
        pytest.skip("cuda_graphs requires CUDA")
    V, idx = _cloud_inputs(local_size, dim, device, torch.float64)
    Vg = V.clone().requires_grad_(True)

    H_eager = IndexedSum(fn, idx).sparse_hessian(Vg).coalesce()
    obj = IndexedSum(fn, idx, **opt)
    H1 = obj.sparse_hessian(Vg).coalesce()
    H2 = obj.sparse_hessian(Vg).coalesce()  # reuse compiled kernel; clone must keep H1 valid

    assert H1._nnz() == H_eager._nnz()
    assert _reldiff(H_eager.to_dense(), H1.to_dense()) < _rtol(torch.float64)
    # H1 must not be corrupted by H2 (CUDA-graph memory reuse) -> the .clone() in _batched_hessian
    assert _reldiff(H1.to_dense(), H2.to_dense()) < 1e-9


@pytest.mark.parametrize("opt", _COMPILE_OPTS, ids=_opt_id)
def test_indexed_sum_compile_correct_for_det_energy(opt):
    """End-to-end: a det-helper energy assembles a correct sparse Hessian with compile enabled,
    verified against finite differences on the total energy."""
    device = "cuda" if opt.get("cuda_graphs") else "cpu"
    g = torch.Generator().manual_seed(0)
    Nv = 16
    V = torch.randn(Nv, 3, generator=g, dtype=torch.float64).to(device)
    idx = torch.stack([torch.randperm(Nv, generator=g)[:4] for _ in range(12)]).to(device)
    Vg = V.clone().requires_grad_(True)

    H = IndexedSum(_det_energy, idx, **opt).sparse_hessian(Vg).coalesce().to_dense()

    def total(z):
        z = z.view(Nv, 3)
        return sum(_det_energy(z[idx[i]]) for i in range(idx.shape[0]))

    Hfd = _fd_hessian(total, V.reshape(-1))  # device-aware
    assert _reldiff(Hfd, H) < 1e-4
