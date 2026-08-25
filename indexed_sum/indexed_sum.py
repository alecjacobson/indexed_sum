import torch
from torch.func import vmap, hessian, jacrev

class IndexedSum:
    """
    Constructs sparse Hessians for functions of the form:

        f(X) = ∑ g(Sᵢ X, SᵢK, cᵢ)

    where `g` is the `local_summand` function, and `f` is the summed function.

    This class facilitates the efficient computation of both the gradient and sparse Hessian
    of `f` by leveraging automatic differentiation.

    Forward evaluation is equivalent to:

        sum([local_summand(X[elem], K[elem], c[i]) for i, elem in enumerate(all_indices)])

    Parameters:
        local_summand (callable): The function `g`, which operates on individual indexed components.
        all_indices (tensor or array-like): The indices defining the structure of the sum.
        per_variable_constants (tensor, optional): A tensor of shape `(num_vars, k_dim)`, containing constants associated with variables.
        per_term_constants (tensor, optional): A tensor of shape `(sum-length, |c|)`, containing constants specific to each term.
        compile (bool, optional): Speed up `sparse_hessian`/`dense_gradient` with `torch.compile`. See below.
        cuda_graphs (bool, optional): Additionally replay via CUDA graphs. See below.

    Speeding up ``sparse_hessian`` / ``dense_gradient`` (``compile`` / ``cuda_graphs``)
    ---------------------------------------------------------------------------------
    Both `sparse_hessian` and `dense_gradient` compute a batched per-element quantity (Hessian
    blocks / gradient blocks), which is many small autograd graphs -- a workload dominated by
    kernel-launch overhead. Two opt-in, *layered* switches cut that overhead (both default
    ``False``, i.e. plain eager):

      * ``compile=True`` -- run the batched Hessian/gradient through ``torch.compile`` (Inductor
        kernel fusion). ~30-50x on an L40 for cheap summands.
      * ``cuda_graphs=True`` -- *also* record the launches into a CUDA graph and replay them
        (``torch.compile(mode="reduce-overhead")``). Fastest when it applies (up to ~85x), but
        it can only capture summands with no host<->device sync.

    They are **compatible, not exclusive** -- ``cuda_graphs`` is a refinement of ``compile``, so
    ``cuda_graphs=True`` implies compilation (setting ``compile=True`` alongside it is optional
    and harmless). The combinations:

      | compile | cuda_graphs | behavior                                   |
      |---------|-------------|--------------------------------------------|
      | False   | False       | eager (unchanged default)                  |
      | True    | False       | torch.compile, Inductor fusion             |
      | any     | True        | torch.compile + CUDA graphs (fastest)      |

    Notes / caveats:
      * The compiled Hessian path uses a reverse-over-reverse Hessian (``jacrev(jacrev(g))``);
        the eager default uses forward-over-reverse (``torch.func.hessian``). For well-behaved
        summands these are mathematically equal (differing only at floating-point rounding). (For
        a ``torch.linalg.det`` summand they differ more: forward-over-reverse is buggy -- see the
        next note -- so the compiled reverse-mode result is actually the correct one.) The
        gradient path is plain reverse-mode (``jacrev(g)``) in both eager and compiled forms, so
        no such reformulation is needed and the compiled gradient matches the eager one exactly.
      * ``cuda_graphs=True`` cannot capture summands that trigger a host<->device sync -- notably
        `torch.linalg.det`/`slogdet`. Use `indexed_sum.det.det`/`logabsdet` instead (which are also
        required for a *correct* Hessian, independent of compile: see `indexed_sum/det.py`).
      * First call pays a one-time compilation cost; shapes should be stable across calls to avoid
        recompilation. Best for repeated-call loops (e.g. Newton iterations). See `bench/RESULTS.md`.

    Reusing the sparsity pattern (``cache_indices``)
    ------------------------------------------------
    ``sparse_hessian`` also rebuilds the assembled matrix's row/col index tensors on every call.
    Those indices depend only on ``all_indices`` (the topology), so at a fixed problem they are
    constant. ``cache_indices=True`` computes them once and reuses them (as RXMesh caches its CSR
    pattern), a large assembly saving in repeated-call loops -- orthogonal to, and composable
    with, ``compile``/``cuda_graphs``. Default ``False``. See `bench/rxmesh/RESULTS.md`. The same
    per-element global-DOF index map drives ``dense_gradient``'s scatter, so it is cached too.
    """

    def __init__(self, local_summand, all_indices, per_variable_constants=None, per_term_constants=None,
                 compile=False, cuda_graphs=False, cache_indices=False):
        """
        Initializes a vectorized summed function.

        Parameters:
        - local_summand: Function of the form `local_summand(a, b, c, ..., cᵢ, k0, k1, ...)`.
        - all_indices: Tensor of shape `[sum-length, local indices size]` specifying subsets of V.
        - per_variable_constants: Tensor of shape `[num_vars, k_dim]` containing constants per variable.
        - per_term_constants: Tensor of shape `[sum-length, |c|]` containing constants for each term.
        - compile: if True, torch.compile the batched Hessian (Inductor fusion).
        - cuda_graphs: if True, also replay via CUDA graphs (implies compilation). See the
          class docstring for how compile/cuda_graphs combine.
        - cache_indices: if True, cache the Hessian's row/col index tensors (a pure function of
          `all_indices`, hence fixed across calls) and reuse them, instead of rebuilding them on
          every `sparse_hessian` call. Trades memory (a persistent `[2, nnz]` int64 tensor) for
          speed; a large win in repeated-call loops (Newton/timestep) at fixed topology. Like
          RXMesh, which computes its sparse pattern once. Default False (unchanged behavior).
        """
        self.local_summand = local_summand
        self.all_indices = all_indices
        self.set_per_variable_constants(per_variable_constants)
        self.set_per_term_constants(per_term_constants)
        self.compile = compile
        self.cuda_graphs = cuda_graphs
        self.cache_indices = cache_indices
        self._compiled_blocks = None  # lazily-built compiled batched-Hessian kernel
        self._compiled_grad = None    # lazily-built compiled batched-gradient kernel
        self._cached_indices = None   # lazily-built [2, nnz] COO index tensor (if cache_indices)
        self._cached_indices_key = None
        self._cached_grad_indices = None  # lazily-built flat scatter index map (if cache_indices)
        self._cached_grad_indices_key = None

    # set per_variable_constants
    def set_per_variable_constants(self, per_variable_constants):
        self.per_variable_constants = per_variable_constants
        if self.per_variable_constants is not None:
            self.selected_K = self.per_variable_constants[self.all_indices]
        else:
            self.selected_K = None

    def set_per_term_constants(self, per_term_constants):
        self.per_term_constants = per_term_constants

    def __call__(self, V):
        """
        Computes the sum of all local summands using vmap.

        Parameters:
        - V: Tensor of values (e.g., vertex positions).

        Returns:
        - Scalar tensor representing the summed function.
        """
        args = self.prepare_args(V)
        
        results = vmap(self.local_summand)(*args)
        return results.sum()

    def prepare_args(self, V):
        selected_V = V[self.all_indices]  # Shape: [sum-length, local indices size, feature_dim]
        args = [selected_V]
        
        if self.per_variable_constants is not None:
            args.append(self.selected_K)
        
        if self.per_term_constants is not None:
            args.append(self.per_term_constants)

        return args

    def sparse_hessian(self, V):
        """
        Computes the sparse Hessian using batched Hessian computation.

        Parameters:
        - V: Tensor with requires_grad=True.

        Returns:
        - Sparse Hessian as a PyTorch sparse tensor.
        """
        num_vars, dim = V.shape
        sum_length, local_size = self.all_indices.shape
        dof = num_vars * dim  # Total degrees of freedom

        args = self.prepare_args(V)
        # Shape: (sum-length, local_size * dim, local_size * dim)
        batched_hessian = self._batched_hessian(args, local_size, dim)

        indices = self._hessian_indices(V, local_size, dim, sum_length)
        values = batched_hessian.reshape(-1)

        H_sparse = torch.sparse_coo_tensor(
            indices=indices,
            values=values,
            size=(dof, dof)
        )
        return H_sparse

    def _hessian_indices(self, V, local_size, dim, sum_length):
        """The [2, nnz] COO (row, col) index tensor. Depends only on `all_indices`/`dim`, so it
        is fixed across calls at a given topology; cached and reused when `cache_indices=True`
        (RXMesh likewise computes its sparse pattern once)."""
        key = (int(sum_length), int(local_size), int(dim), V.device)
        if self.cache_indices and self._cached_indices is not None and self._cached_indices_key == key:
            return self._cached_indices

        global_indices = (dim * self.all_indices[:, :, None] + torch.arange(dim, device=V.device)).reshape(sum_length, local_size * dim)
        row_indices = global_indices[:, :, None].expand(sum_length, local_size * dim, local_size * dim).reshape(-1)
        col_indices = global_indices[:, None, :].expand(sum_length, local_size * dim, local_size * dim).reshape(-1)
        indices = torch.stack([row_indices, col_indices])

        if self.cache_indices:
            self._cached_indices = indices
            self._cached_indices_key = key
        return indices

    def _batched_hessian(self, args, local_size, dim):
        """Compute the batched per-element Hessian blocks, eagerly or via torch.compile.

        Eager uses forward-over-reverse (`torch.func.hessian`). The compiled path uses
        reverse-over-reverse (`jacrev(jacrev)`) -- mathematically identical, and the only
        formulation `torch.compile` currently traces. See the class docstring.
        """
        def reshaped_summand(inputs, *rest):
            return self.local_summand(inputs.view(local_size, dim), *rest)

        # cuda_graphs implies compilation; either switch selects the compiled path.
        if not (self.compile or self.cuda_graphs):
            return vmap(hessian(reshaped_summand))(*args)

        if self._compiled_blocks is None:
            mode = "reduce-overhead" if self.cuda_graphs else None

            def blocks(*a):
                return vmap(jacrev(jacrev(reshaped_summand)))(*a)

            # dynamic=False: specialize on concrete shapes. The batched functorch Hessian can't
            # trace with symbolic sizes, and shapes are fixed per problem, so static is correct.
            self._compiled_blocks = torch.compile(blocks, mode=mode, dynamic=False)

        out = self._compiled_blocks(*args)
        # CUDA-graph outputs are backed by reused memory the next call overwrites; clone before
        # the values escape into the returned sparse tensor. (Not needed for plain fusion.)
        return out.clone() if self.cuda_graphs else out

    def dense_gradient(self, V):
        """
        Computes the dense gradient vector of `f` by assembling per-element gradients.

        Equivalent to ``torch.autograd.grad(self(V).sum(), V)[0].reshape(-1)`` (the eager
        autograd gradient), but computed as a batched per-element Jacobian that is scattered
        into the global vector -- the same structure as `sparse_hessian`, and accelerated by the
        same opt-in ``compile``/``cuda_graphs`` switches (see the class docstring). Unlike the
        Hessian, the gradient is plain reverse-mode (``vmap(jacrev(g))``), which compiles cleanly
        under dynamo, so no reformulation is needed.

        Parameters:
        - V: Tensor of shape `[num_vars, dim]`.

        Returns:
        - Dense gradient as a flat tensor of shape `[num_vars * dim]`.
        """
        num_vars, dim = V.shape
        sum_length, local_size = self.all_indices.shape
        dof = num_vars * dim  # Total degrees of freedom

        args = self.prepare_args(V)
        # Shape: (sum-length, local_size * dim)
        batched_grad = self._batched_gradient(args, local_size, dim)

        # Flat global-DOF index of each per-element gradient entry; scatter-add into the vector.
        scatter_index = self._gradient_indices(V, local_size, dim, sum_length)
        grad = torch.zeros(dof, dtype=batched_grad.dtype, device=V.device)
        grad.index_add_(0, scatter_index, batched_grad.reshape(-1))
        return grad

    def _gradient_indices(self, V, local_size, dim, sum_length):
        """The flat `[sum_length * local_size * dim]` scatter-target index of each per-element
        gradient entry (its global DOF). This is exactly the `global_indices` map that
        `_hessian_indices` builds, flattened -- fixed across calls at a given topology; cached
        and reused when `cache_indices=True`."""
        key = (int(sum_length), int(local_size), int(dim), V.device)
        if (self.cache_indices and self._cached_grad_indices is not None
                and self._cached_grad_indices_key == key):
            return self._cached_grad_indices

        global_indices = (dim * self.all_indices[:, :, None] + torch.arange(dim, device=V.device)).reshape(-1)

        if self.cache_indices:
            self._cached_grad_indices = global_indices
            self._cached_grad_indices_key = key
        return global_indices

    def _batched_gradient(self, args, local_size, dim):
        """Compute the batched per-element gradient blocks, eagerly or via torch.compile.

        Both eager and compiled paths use reverse-mode `vmap(jacrev(g))`, which compiles cleanly
        under dynamo -- so, unlike `_batched_hessian`, no forward/reverse reformulation is needed
        and the compiled result matches the eager one to floating-point rounding.
        """
        def reshaped_summand(inputs, *rest):
            return self.local_summand(inputs.view(local_size, dim), *rest)

        # cuda_graphs implies compilation; either switch selects the compiled path.
        if not (self.compile or self.cuda_graphs):
            return vmap(jacrev(reshaped_summand))(*args).reshape(args[0].shape[0], local_size * dim)

        if self._compiled_grad is None:
            mode = "reduce-overhead" if self.cuda_graphs else None

            def grad_blocks(*a):
                return vmap(jacrev(reshaped_summand))(*a)

            # dynamic=False: specialize on concrete shapes (fixed per problem), matching the
            # batched-Hessian kernel.
            self._compiled_grad = torch.compile(grad_blocks, mode=mode, dynamic=False)

        out = self._compiled_grad(*args).reshape(args[0].shape[0], local_size * dim)
        # CUDA-graph outputs alias reused memory the next call overwrites; clone before the values
        # escape into the returned gradient vector. (Not needed for plain fusion.)
        return out.clone() if self.cuda_graphs else out

    def __add__(self, other):
        if isinstance(other, IndexedSum):
            return SumNode(self, other)  # Create a tree node instead of a list-based collection
        elif isinstance(other, SumNode):
            return SumNode(self, other)  # Attach to an existing SumNode
        else:
            raise ValueError("Can only add IndexedSum or SumNode.")

class SumNode:
    """Tree-based structure to store sums of `IndexedSum`s."""
    
    def __init__(self, left, right=None):
        self.left = left
        self.right = right

    def __call__(self, V):
        if self.right is None:
            return self.left(V)
        return self.left(V) + self.right(V)

    def sparse_hessian(self, V):
        if self.right is None:
            return self.left.sparse_hessian(V)
        # Should they be coalesced before adding?
        # Does that depend on whether using coo or csr?
        return self.left.sparse_hessian(V) + self.right.sparse_hessian(V)  # Efficient summation

    def dense_gradient(self, V):
        if self.right is None:
            return self.left.dense_gradient(V)
        return self.left.dense_gradient(V) + self.right.dense_gradient(V)

    def __add__(self, other):
        return SumNode(self, other)  # Create a new tree node instead of copying lists
