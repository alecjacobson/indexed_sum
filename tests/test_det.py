"""
Tests for the vmap-safe determinant helpers (indexed_sum.det).

The point: a summand using `indexed_sum.det.det` gets a CORRECT sparse Hessian through the
library's existing forward-over-reverse path, whereas `torch.linalg.det` silently gives a
wrong one (pytorch#149694). Ground truth is finite differences (AD-independent).

Run:  pytest tests/test_det.py -v
"""
import pytest
import torch
from torch.func import vmap, hessian

from indexed_sum import IndexedSum
from indexed_sum.det import det, logabsdet

torch.set_default_dtype(torch.float64)


def _reldiff(a, b):
    return (a - b).abs().max().item() / (a.abs().max().item() + 1e-30)


@pytest.mark.parametrize("n", [1, 2, 3, 4, 5])
def test_det_values_match_torch(n):
    torch.manual_seed(n)
    A = torch.randn(7, n, n)
    mine = torch.stack([det(A[i]) for i in range(7)])
    assert _reldiff(mine, torch.linalg.det(A)) < 1e-12


def test_det_supports_batched_leading_dims():
    torch.manual_seed(0)
    A = torch.randn(4, 6, 3, 3)
    assert _reldiff(det(A), torch.linalg.det(A)) < 1e-12


def test_logabsdet_matches_slogdet():
    torch.manual_seed(0)
    A = torch.randn(5, 3, 3)
    mine = torch.stack([logabsdet(A[i]) for i in range(5)])
    assert _reldiff(mine, torch.linalg.slogdet(A)[1]) < 1e-12


def _fd_hessian(fn, x, eps=1e-6):
    n = x.numel()

    def grad(z):
        z = z.detach().requires_grad_(True)
        return torch.autograd.grad(fn(z), z)[0]

    H = torch.zeros(n, n)
    for i in range(n):
        e = torch.zeros(n)
        e[i] = eps
        H[i] = (grad(x + e) - grad(x - e)) / (2 * eps)
    return 0.5 * (H + H.T)


def _neo(v, detfn):
    F = v[1:] - v[0:1]
    J = detfn(F)
    return (F * F).sum() - 3 - 2 * torch.log(torch.clamp(J, min=1e-3)) + (J - 1) ** 2


def _well_conditioned_tets(n=8, seed=1):
    g = torch.Generator().manual_seed(seed)
    base = torch.tensor([[0, 0, 0], [1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=torch.float64)
    return base.unsqueeze(0) + 0.05 * torch.randn(n, 4, 3, generator=g)


def test_det_helper_gives_correct_hessian_through_vmap_hessian():
    """The whole point: det-helper summand -> correct Hessian via forward-over-reverse."""
    sel = _well_conditioned_tets()
    H = vmap(hessian(lambda inp: _neo(inp.view(4, 3), det)))(sel)
    for i in range(sel.shape[0]):
        Hfd = _fd_hessian(lambda z: _neo(z.view(4, 3), det), sel[i].reshape(12))
        assert _reldiff(Hfd, H[i].reshape(12, 12)) < 1e-4


@pytest.mark.xfail(reason="torch.linalg.det is wrong under vmap+forward-mode AD "
                          "(pytorch#149694); this is exactly why the det helper exists. "
                          "If it starts passing, torch fixed the batching rule.",
                   strict=True)
def test_torch_det_gives_correct_hessian_through_vmap_hessian():
    sel = _well_conditioned_tets()
    H = vmap(hessian(lambda inp: _neo(inp.view(4, 3), torch.linalg.det)))(sel)
    for i in range(sel.shape[0]):
        Hfd = _fd_hessian(lambda z: _neo(z.view(4, 3), torch.linalg.det), sel[i].reshape(12))
        assert _reldiff(Hfd, H[i].reshape(12, 12)) < 1e-4


def test_indexed_sum_sparse_hessian_correct_with_det_helper():
    """End-to-end through the public API: a det-helper energy assembles a correct sparse
    Hessian (vs finite differences on the total energy)."""
    g = torch.Generator().manual_seed(0)
    Nv = 20
    V = torch.randn(Nv, 3, generator=g)
    idx = torch.stack([torch.randperm(Nv, generator=g)[:4] for _ in range(15)])
    Vg = V.clone().requires_grad_(True)

    H = IndexedSum(lambda v: _neo(v, det), idx).sparse_hessian(Vg).coalesce().to_dense()

    def total(z):
        z = z.view(Nv, 3)
        return sum(_neo(z[idx[i]], det) for i in range(idx.shape[0]))

    Hfd = _fd_hessian(total, V.reshape(-1))
    assert _reldiff(Hfd, H) < 1e-4
