import numpy as np
import igl
import torch
from torch.func import hessian

from timeit import default_timer as timer
from indexed_sum import IndexedSum


def test_total_area():
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
    def face_area(v):
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
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


def test_multiple_terms():
    X = torch.tensor([[0,0],[1,0],[1,1],[0,1]],dtype=torch.float64,requires_grad=True)
    # list of indices 0 to size of X
    F = torch.tensor([[0,1,2],[0,2,3]],dtype=torch.int64)
    E = torch.tensor([[0,1],[1,2],[2,0],[0,3],[2,3]],dtype=torch.int64)
    I = torch.arange(X.shape[0],dtype=torch.int64).unsqueeze(1)
    
    def vertex_magnitude(v):
        return torch.linalg.norm(v)
    
    def edge_length(v):
        return torch.linalg.norm(v[1] - v[0])
    
    def face_area(v):
        v0 = v[0]
        v1 = v[1]
        v2 = v[2]
        a = torch.linalg.norm(v1 - v0, dim=-1)
        b = torch.linalg.norm(v2 - v1, dim=-1)
        c = torch.linalg.norm(v2 - v0, dim=-1)
        s = (a + b + c) / 2
        area = torch.sqrt(s * (s - a) * (s - b) * (s - c))
        return area
    
    sum_vertex_magnitude = IndexedSum(local_summand=vertex_magnitude,all_indices=I)
    sum_edge_length = IndexedSum(local_summand=edge_length,all_indices=E)
    sum_face_area = IndexedSum(local_summand=face_area,all_indices=F)
    sum_all = sum_vertex_magnitude + sum_edge_length + sum_face_area
    
    X.grad = None  # Reset gradients
    value_vertex_magnitude = sum_vertex_magnitude(X)
    value_vertex_magnitude.backward()
    grad_vertex_magnitude = X.grad.detach().clone()
    hess_vertex_magnitude = sum_vertex_magnitude.sparse_hessian(X)
    
    X.grad = None  # Reset gradients
    value_edge_length = sum_edge_length(X)
    value_edge_length.backward()
    grad_edge_length = X.grad.detach().clone()
    hess_edge_length = sum_edge_length.sparse_hessian(X)
    
    X.grad = None  # Reset gradients
    value_face_area = sum_face_area(X)
    value_face_area.backward()
    grad_face_area = X.grad.detach().clone()
    hess_face_area = sum_face_area.sparse_hessian(X)
    
    X.grad = None  # Reset gradients
    value_all = sum_all(X)
    value_all.backward()
    grad_all = X.grad.detach().clone()
    hess_all = sum_all.sparse_hessian(X)
    
    assert torch.allclose( value_vertex_magnitude + value_edge_length + value_face_area, value_all)
    assert torch.allclose( grad_vertex_magnitude + grad_edge_length + grad_face_area, grad_all)
    assert torch.allclose((hess_vertex_magnitude + hess_edge_length + hess_face_area).to_dense(), hess_all.to_dense())

def test_mass_spring_with_constants():

    # A square made of 4 springs and one diagonal
    X = torch.tensor([[0,0],[1,0],[1,1],[0,1]],dtype=torch.float64,requires_grad=True)
    E = torch.tensor([[0,1],[1,2],[2,3],[3,0],[0,2]],dtype=torch.int64)
    M = torch.tensor([1,2,3,4],dtype=torch.float64).unsqueeze(1)
    # compute rest_l[i] = rest length of spring from X[E[i,0]] to X[E[i,1]]
    rest_l = 0.5*torch.norm(X[E[:,0]] - X[E[:,1]],dim=1).detach().unsqueeze(1)
    
    
    def spring(v,m,rest_l):
        v0 = v[0]
        v1 = v[1]
        m0 = m[0]
        m1 = m[1]
        l = torch.norm(v1 - v0)
        return 0.5 * (m0 + m1) * (l - rest_l)**2;
    
    #simple_total = sum([spring(X[edge]) for edge in E])
    simple_total_func = lambda X: sum([spring(X[edge],M[edge],rest_l[i]) for i,edge in enumerate(E)])
    
    simple_total = simple_total_func(X)
    simple_total.backward()
    grad_simple  = X.grad
    H_simple = torch.func.hessian(simple_total_func)(X)
    H_simple = H_simple.reshape(X.shape[0] * X.shape[1], X.shape[0] * X.shape[1])
    print("Simple Total:", simple_total.item())
    
    total_energy_function = IndexedSum(local_summand=spring,all_indices=E,per_variable_constants=M,per_term_constants=rest_l)
    value = total_energy_function(X)
    # gradient in the usual pytorch way
    value.backward()
    grad = X.grad
    # second derivatives using IndexedSum's sparse scatter-gather
    H = total_energy_function.sparse_hessian(X)
    
    # much faster than torch.func.hessian(lambda X: total_energy_function(X), X)
    H_dense = torch.func.hessian(lambda X: total_energy_function(X))(X)
    H_dense = H_dense.reshape(X.shape[0] * X.shape[1], X.shape[0] * X.shape[1])
    print("‖H‖ = ", torch.norm(H.to_dense()).item())
    print("‖H - H_dense‖ = ", torch.norm(H.to_dense() - H_dense).item())
    print("‖grad - grad_simple‖ = ", torch.norm(grad - grad_simple).item())
    print("‖H - H_simple‖ = ", torch.norm(H._to_dense() - H_simple).item())
    assert torch.allclose(H.to_dense(), H_dense)
    assert torch.allclose(H.to_dense(), H_simple)

