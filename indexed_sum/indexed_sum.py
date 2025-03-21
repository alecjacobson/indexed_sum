import torch
from torch.func import vmap, hessian

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
    """
    
    def __init__(self, local_summand, all_indices, per_variable_constants=None, per_term_constants=None):
        """
        Initializes a vectorized summed function.

        Parameters:
        - local_summand: Function of the form `local_summand(a, b, c, ..., cᵢ, k0, k1, ...)`.
        - all_indices: Tensor of shape `[sum-length, local indices size]` specifying subsets of V.
        - per_variable_constants: Tensor of shape `[num_vars, k_dim]` containing constants per variable.
        - per_term_constants: Tensor of shape `[sum-length, |c|]` containing constants for each term.
        """
        self.local_summand = local_summand
        self.all_indices = all_indices
        self.set_per_variable_constants(per_variable_constants)
        self.set_per_term_constants(per_term_constants)

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

        def reshaped_summand(inputs, *args):
            """Reshape input for Hessian computation."""
            split_inputs = inputs.view(local_size, dim)  # Reshape to (local_size, feature_dim)
            return self.local_summand(split_inputs, *args)

        args = self.prepare_args(V)
        batched_hessian = vmap(hessian(reshaped_summand))(*args)  # Shape: (sum-length, local_size * dim, local_size * dim)
        
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
