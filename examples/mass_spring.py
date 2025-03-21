from indexed_sum import IndexedSum
import torch

# A square made of 4 springs and one diagonal
X = torch.tensor([[0,0],[1,0],[1,1],[0,1]],dtype=torch.float64,requires_grad=True)
E = torch.tensor([[0,1],[1,2],[2,3],[3,0],[0,2]],dtype=torch.int64)
M = torch.tensor([1,2,2,1],dtype=torch.float64).unsqueeze(1)
# compute rest_l[i] = rest length of spring from X[E[i,0]] to X[E[i,1]]
R = 0.5*torch.norm(X[E[:,0]] - X[E[:,1]],dim=1).detach().unsqueeze(1)


def spring(v,m,r):
    v0 = v[0]
    v1 = v[1]
    m0 = m[0]
    m1 = m[1]
    l = torch.norm(v1 - v0)
    return 0.5 * (m0 + m1) * (l - r)**2;

#simple_total = sum([spring(X[edge]) for edge in E])
simple_total_func = lambda X: sum([spring(X[edge],M[edge],R[i]) for i,edge in enumerate(E)])

simple_total = simple_total_func(X)
simple_total.backward()
grad_simple  = X.grad
H_simple = torch.func.hessian(simple_total_func)(X)
H_simple = H_simple.reshape(X.shape[0] * X.shape[1], X.shape[0] * X.shape[1])
print("Simple Total:", simple_total.item())

total_energy_function = IndexedSum(local_summand=spring,all_indices=E,per_variable_constants=M,per_term_constants=R)
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
