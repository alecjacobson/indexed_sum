"""
vmap-safe determinant helpers for use inside `local_summand` functions.

Why this exists
---------------
`IndexedSum.sparse_hessian` computes per-element Hessians with
`torch.func.hessian == jacfwd(jacrev(f))` (forward-over-reverse) under `vmap`. PyTorch's
batching rule for **forward-mode** autodiff of `torch.linalg.det` / `torch.det` /
`torch.linalg.slogdet` is **wrong** (silently returns incorrect, non-NaN values) -- so a
summand that calls `torch.linalg.det` gets a wrong Hessian. This is an open upstream bug
(pytorch#149694), still present on the latest release; reverse-mode and non-vmapped code are
unaffected, which is why casual checks miss it. See `bench/RESULTS.md`.

These helpers compute the determinant with elementary multiply/add ops only, so forward-mode
AD composes correctly and the fast forward-over-reverse Hessian path stays correct. Use them
instead of `torch.linalg.det` / `slogdet` inside your `local_summand`.

Only the determinant family is affected; `torch.inverse`, `@`, `trace`, elementwise ops, etc.
compose fine and need no replacement.

    from indexed_sum.det import det, logabsdet
    def neohookean(v):
        F = v[1:] - v[0:1]
        J = det(F)                      # NOT torch.linalg.det(F)
        return (F*F).sum() - 3 - 2*torch.log(torch.clamp(J, min=1e-3)) + (J-1)**2
"""
import torch


def det(A):
    """Determinant of a square matrix via elementary ops (vmap/forward-mode safe).

    Supports any leading batch dims: `A` has shape `(..., n, n)`. Fast closed forms for
    n in {1, 2, 3}; general n uses Laplace (cofactor) expansion -- O(n!), intended for the
    small local matrices (typically 2x2 or 3x3) that arise in FEM/FVM summands.
    """
    n = A.shape[-1]
    if A.shape[-2] != n:
        raise ValueError(f"det expects a square matrix, got shape {tuple(A.shape)}")

    if n == 1:
        return A[..., 0, 0]
    if n == 2:
        return A[..., 0, 0] * A[..., 1, 1] - A[..., 0, 1] * A[..., 1, 0]
    if n == 3:
        return (
            A[..., 0, 0] * (A[..., 1, 1] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 1])
            - A[..., 0, 1] * (A[..., 1, 0] * A[..., 2, 2] - A[..., 1, 2] * A[..., 2, 0])
            + A[..., 0, 2] * (A[..., 1, 0] * A[..., 2, 1] - A[..., 1, 1] * A[..., 2, 0])
        )

    # General case: Laplace expansion along the first row.
    total = None
    for j in range(n):
        minor = _minor(A, 0, j)
        term = A[..., 0, j] * det(minor)
        if j % 2:
            term = -term
        total = term if total is None else total + term
    return total


def logabsdet(A, eps=0.0):
    """log|det(A)| computed via `det` above -- a vmap-safe replacement for the log-abs-det
    component of `torch.linalg.slogdet`. `eps` optionally floors |det| before the log to
    avoid -inf on (near-)singular inputs."""
    d = det(A).abs()
    if eps:
        d = torch.clamp(d, min=eps)
    return torch.log(d)


def _minor(A, i, j):
    """The (i, j) minor: `A` with row i and column j removed, preserving leading dims."""
    n = A.shape[-1]
    rows = [r for r in range(n) if r != i]
    cols = [c for c in range(n) if c != j]
    return A[..., rows, :][..., :, cols]
