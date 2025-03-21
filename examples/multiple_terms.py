from indexed_sum import IndexedSum
import torch

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
