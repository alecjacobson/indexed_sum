import numpy as np
import igl
import torch
from torch.func import hessian

from timeit import default_timer as timer
from indexed_sum import IndexedSum


def test_indexed_sum():
    # 3D Tetrahedron
    #V = np.array([[0,0,0],[1,0,0],[0,1,0],[0,0,1]],dtype='float64')
    #F = np.array([[3, 1, 2],[2, 0, 3],[3, 0, 1],[1, 0, 2]], dtype='int64')
    # 2D Triangle
    V = np.array([[0,0],[1,0],[0,1]],dtype='float64')
    F = np.array([[0,1,2]],dtype='int64')
    V,F = igl.loop(V,F,number_of_subdivs=6)
    device = torch.device("cpu")
    V = torch.tensor(V,dtype=torch.float32,requires_grad=True,device=device)
    F = torch.tensor(F,dtype=torch.int64,device=device)
    # Function that computes the area given three vertices
    def face_area(v0, v1, v2):
        # Compute edge lengths
        a = torch.linalg.norm(v1 - v0, dim=-1)
        b = torch.linalg.norm(v2 - v1, dim=-1)
        c = torch.linalg.norm(v2 - v0, dim=-1)
    
        # Compute semi-perimeter
        s = (a + b + c) / 2
    
        # Compute area using Heron's formula
        area = torch.sqrt(s * (s - a) * (s - b) * (s - c))
        return area
    area_function = IndexedSum(face_area, F)
    A = area_function(V)
    print("Total Area:", A.item())
    # Compute gradient
    A.backward()
    print("∂A/∂V.shape: ", V.grad.shape)
    # Compute sparse Hessian
    start = timer()
    H_sparse = area_function.sparse_hessian(V)
    end = timer()
    print(f"Sparse Hessian computation time: {end - start:.4f} seconds")
    print("Sparse Hessian shape:", H_sparse.shape)
    print("Nonzero elements in Hessian:", H_sparse._nnz())
    
    # Compare with full Hessian from autograd
    start = timer()
    #H_dense = torch.autograd.functional.hessian(lambda V: area_function(V), V)
    H_dense = hessian(lambda V: area_function(V))(V)
    end = timer()
    print(f"Full Hessian computation time: {end - start:.4f} seconds")
    print("Full Hessian shape:", H_dense.shape)
    
    H_dense = H_dense.reshape(V.shape[0] * V.shape[1], V.shape[0] * V.shape[1])
    diff = torch.norm(H_dense - H_sparse.to_dense())
    print("Hessian Difference:", diff.item())

    # ensure that diff.item() is less than 1e-6
    assert diff.item() < 1e-6

