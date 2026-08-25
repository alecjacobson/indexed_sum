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
        compile (bool, optional): Speed up `sparse_hessian` with `torch.compile`. See below.
        cuda_graphs (bool, optional): Additionally replay via CUDA graphs. See below.

    Speeding up ``sparse_hessian`` (``compile`` / ``cuda_graphs``)
    -------------------------------------------------------------
    `sparse_hessian` computes a batched per-element Hessian, which is many small autograd
    graphs -- a workload dominated by kernel-launch overhead. Two opt-in, *layered* switches
    cut that overhead (both default ``False``, i.e. plain eager):

      * ``compile=True`` -- run the batched Hessian through ``torch.compile`` (Inductor kernel
        fusion). ~30-50x on an L40 for cheap summands.
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
      * The compiled path uses a reverse-over-reverse Hessian (``jacrev(jacrev(g))``); the eager
        default uses forward-over-reverse (``torch.func.hessian``). For well-behaved summands these
        are mathematically equal (differing only at floating-point rounding). (For a
        ``torch.linalg.det`` summand they differ more: forward-over-reverse is buggy -- see the
        next note -- so the compiled reverse-mode result is actually the correct one.)
      * ``cuda_graphs=True`` cannot capture summands that trigger a host<->device sync -- notably
        `torch.linalg.det`/`slogdet`. Use `indexed_sum.det.det`/`logabsdet` instead (which are also
        required for a *correct* Hessian, independent of compile: see `indexed_sum/det.py`).
      * First call pays a one-time compilation cost; shapes should be stable across calls to avoid
        recompilation. Best for repeated-call loops (e.g. Newton iterations). See `bench/RESULTS.md`.
    """

    def __init__(self, local_summand, all_indices, per_variable_constants=None, per_term_constants=None,
                 compile=False, cuda_graphs=False):
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
        """
        self.local_summand = local_summand
        self.all_indices = all_indices
        self.set_per_variable_constants(per_variable_constants)
        self.set_per_term_constants(per_term_constants)
        self.compile = compile
        self.cuda_graphs = cuda_graphs
        self._compiled_blocks = None  # lazily-built compiled batched-Hessian kernel

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

        # Compute global indices efficiently
        global_indices = (dim * self.all_indices[:, :, None] + torch.arange(dim, device=V.device)).reshape(sum_length, local_size * dim)
        row_indices = global_indices[:, :, None].expand(sum_length, local_size * dim, local_size * dim).reshape(-1)
        col_indices = global_indices[:, None, :].expand(sum_length, local_size * dim, local_size * dim).reshape(-1)
        values = batched_hessian.reshape(-1)

        H_sparse = torch.sparse_coo_tensor(
            indices=torch.stack([row_indices, col_indices]),
            values=values,
            size=(dof, dof)
        )
        return H_sparse

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

    def __add__(self, other):
        return SumNode(self, other)  # Create a new tree node instead of copying lists
