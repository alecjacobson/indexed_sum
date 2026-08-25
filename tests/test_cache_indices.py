"""Tests for IndexedSum(..., cache_indices=True).

The assembled sparse Hessian's row/col index tensors depend only on `all_indices`
(the topology), so they are constant across calls. `cache_indices` computes them once
and reuses them; the result must be identical to the default, on repeated calls and at
changed variable values (same topology).
"""
import torch

from indexed_sum import IndexedSum


def _spring(v):
    d = v[1] - v[0]
    return 0.5 * (d * d).sum()


def _mesh(device="cpu"):
    torch.manual_seed(0)
    V = torch.randn(12, 3, dtype=torch.float64, device=device)
    E = torch.tensor(
        [[0, 1], [1, 2], [2, 3], [3, 4], [4, 5], [0, 5],
         [6, 7], [7, 8], [8, 9], [9, 10], [10, 11], [0, 6], [3, 9]],
        dtype=torch.int64, device=device,
    )
    return V, E


def test_cache_indices_matches_default():
    V, E = _mesh()
    ref = IndexedSum(_spring, E).sparse_hessian(V).to_dense()
    cached = IndexedSum(_spring, E, cache_indices=True).sparse_hessian(V).to_dense()
    assert torch.equal(ref, cached)


def test_cache_indices_stable_across_calls_and_values():
    V, E = _mesh()
    f = IndexedSum(_spring, E, cache_indices=True)
    f.sparse_hessian(V)                       # populate cache
    assert f._cached_indices is not None
    # New variable values, same topology: cached indices must still assemble correctly.
    V2 = V + 0.5
    got = f.sparse_hessian(V2).to_dense()
    want = IndexedSum(_spring, E).sparse_hessian(V2).to_dense()
    assert torch.equal(got, want)


def test_cache_indices_default_off():
    _, E = _mesh()
    assert IndexedSum(_spring, E).cache_indices is False
