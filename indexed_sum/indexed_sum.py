import torch
from torch.func import vmap, hessian

class IndexedSum:
    """
    Constructs sparse Hessians for functions of the form:

        f(X) = ∑ g(Sᵢ X)

    where `g` is the `local_summand` function, and `f` is the summed function.

    The Hessian of `f` is given by:

        ∂²f/∂X² = ∑ Sᵢᵀ (∂²g/∂Xᵢ²) Sᵢ

    This class facilitates the efficient computation of both the gradient and sparse Hessian
    of `f` by leveraging automatic differentiation.

    Parameters:
        local_summand (callable): The function `g`, which operates on individual indexed components.
        all_indices (tensor or array-like): The indices defining the structure of the sum.
    """
    
    def __init__(self, local_summand, all_indices):
        """
        Initializes a vectorized summed function.

        Parameters:
        - local_summand: Function of the form `local_summand(a, b, c, ...)`.
        - all_indices: Tensor of shape `[sum-length, local indices size]` specifying subsets of V.
        """
        self.local_summand = local_summand
        self.all_indices = all_indices

    def __call__(self, V):
        """
        Computes the sum of all local summands using vmap.

        Parameters:
        - V: Tensor of values (e.g., vertex positions).

        Returns:
        - Scalar tensor representing the summed function.
        """
        selected_V = V[self.all_indices]  # Shape: [sum-length, local indices size, feature_dim]
        results = vmap(self.local_summand)(*selected_V.unbind(dim=1))  # Correct unpacking
        return results.sum()

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

        def reshaped_summand(inputs):
            """Reshape input for Hessian computation."""
            split_inputs = inputs.view(local_size, dim)
            return self.local_summand(*split_inputs)
        
        selected_V = V[self.all_indices].reshape(sum_length, -1)  # Shape: (sum-length, local_size * dim)
        batched_hessian = vmap(hessian(reshaped_summand))(selected_V)  # Shape: (sum-length, local_size * dim, local_size * dim)
        
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
