from indexed_sum import IndexedSum
import torch
import polyscope as ps
import numpy as np
import igl
import scipy.sparse

# Some helper functions
def torch_sparse_to_scipy_csc(tensor: torch.Tensor) -> scipy.sparse.csc_matrix:
    """
    Convert a PyTorch sparse tensor to a SciPy CSC sparse matrix.

    Parameters:
    - tensor: PyTorch sparse tensor.
    Returns:
    - scipy sparse CSC matrix.
    """
    if not tensor.is_sparse:
        raise ValueError("Input tensor must be a sparse tensor")

    coalesced = tensor.coalesce()  # Ensure COO format
    indices = coalesced.indices().numpy()
    values = coalesced.detach().values().numpy()
    shape = coalesced.shape

    return scipy.sparse.csc_matrix((values, (indices[0], indices[1])), shape=shape)


def backtracking_line_search(f,x0,dfx0,dx,alpha,beta,max_iter):
    """
    Backtracking line search

    Parameters:
    - f: function to minimize
    - x0: starting point
    - dfx0: gradient at x0
    - dx: search direction
    - alpha: sufficient decrease parameter
    - beta: step size reduction parameter
    - max_iter: maximum number of iterations
    Returns:
    - t: step size
    - x: new point
    - fx: function value at new point
    """
    assert(alpha>0 and alpha<0.5)
    assert(beta>0 and beta<1)
    t = 1
    fx0 = f(x0)
    if max_iter is None:
        max_iter = 30
    alpha_dfdx0_dx = alpha * torch.sum(dfx0 * dx)
    for iter in range(1,max_iter+1):
        x = x0 + t*dx
        fx = f(x)
        if fx <= fx0 + t*alpha_dfdx0_dx:
            return t,x,fx
        t = beta*t
    t = 0
    x = x0
    fx = fx0
    return t,x,fx

# Create a simple cantilevered beam
aspect_ratio = 5
ns =  16
X,F = igl.triangulated_grid(aspect_ratio*ns+1,ns+1)
X /= ns
X[:,0] *= aspect_ratio

# Convert to torch types
X = torch.tensor(X,dtype=torch.float64,requires_grad=False)
F = torch.tensor(F,dtype=torch.int64)

# index sets for per-vertex energy
I = torch.arange(X.shape[0],dtype=torch.int64).unsqueeze(1)

# per-vertex mass
M = igl.massmatrix(X,F,igl.MASSMATRIX_TYPE_VORONOI).diagonal()
M = torch.tensor(M,dtype=torch.float64).unsqueeze(1)

# initialize deformation to the rest state
x = X.clone().detach().requires_grad_(True)

# define local summand of stable neo-Hookean energy
def stable_neohookean(x,X,scale):
    def build_M(x0,x1,x2):
        return torch.stack([x1-x0,x2-x0],dim=1)
    M = build_M(X[0],X[1],X[2])
    m = build_M(x[0],x[1],x[2])
    youngs_modulus = 0.37e3
    poissons_ratio = 0.4
    mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    lam = youngs_modulus * poissons_ratio / ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    J = m @ torch.inverse(M)
    A = 0.5 * torch.det(M)
    Ic = torch.trace(J.T @ J)
    # https://github.com/pytorch/pytorch/issues/149694
    det2 = lambda J: J[0,0] * J[1,1] - J[0,1] * J[1,0]
    detF = det2(J)
    alpha = 1.0 + mu / lam
    W = mu / 2.0 * (Ic - 2.0) + lam / 2.0 * (detF - alpha) * (detF - alpha)
    return scale * A * W

# define local summand of momentum potential energy
def momentum_potential(x,xtilde,m):
    delta = x - xtilde
    return 0.5 * m * torch.sum(delta * delta)

dt = 1/66

# prepare each as indexed sum with appropriate constant parameters
elastic_potential_func = IndexedSum(local_summand=lambda x,X: stable_neohookean(x,X,scale=dt**2),all_indices=F,per_variable_constants=X)
momentum_potential_func = IndexedSum(local_summand=momentum_potential,all_indices=I,per_term_constants=M)

# sum together
total_energy_func = elastic_potential_func + momentum_potential_func

# Find all the rows in X with X[:,0] == 0
fixed = torch.where(X[:,0] == 0)[0]
fixed = torch.cat([2*fixed,2*fixed+1])
fixed = fixed.numpy()
bc = np.zeros((fixed.shape[0],1))


xdot = torch.zeros_like(x)
xprev = x.clone().detach()

# mkdir out
import os
os.makedirs("out",exist_ok=True)

t = 0
g = torch.tensor([0,-9.8],dtype=torch.float64)

def integrate():
    global t,xdot,xprev,x,g,fixed,bc,total_energy_func,dt,elastic_potential_func,momentum_potential_func

    print(f"Integrating t={t}")
    for i in range(30):
        xtilde = xprev + dt * xdot + dt**2 * g
        momentum_potential_func.set_per_variable_constants(xtilde)

        total_energy = total_energy_func(x)
        total_energy.backward()
        grad = x.grad

        np_grad = grad.detach().numpy().reshape(-1)[:,np.newaxis]
        np_grad[fixed] = 0
        if np.linalg.norm(np_grad) < 1e-8:
            print("    Converged")
            break

        H = total_energy_func.sparse_hessian(x)
        # convert tensor sparse matrix to scipy sparse matrix
        H = torch_sparse_to_scipy_csc(H)

        # min_x ½ dxᵀ H dx + gᵀ dx
        # subject to dx(fixed) = 0
        dx = igl.min_quad_with_fixed(H,np_grad,fixed,bc)
        # reshape dx like x
        dx = torch.tensor(dx.reshape(x.shape[0],x.shape[1]),dtype=torch.float64)

        x0 = x.clone().detach()
        ss,x_new,fx = backtracking_line_search(total_energy_func,x0,grad,dx,0.1,0.5,30)
        print(f"  {i}: {total_energy} → {fx} {ss} | {np.linalg.norm(np_grad)} {dx.norm()}")
        if ss == 0:
            print("    Line search failed")
            break

        x.data = x_new
        x.grad = None
    t = t + dt
    damping = 0.98
    xdot = ((x - xprev) / dt * damping).detach()
    xprev = x.clone().detach()

#while t < 5:
#    igl.write_triangle_mesh(f"out/frame_{t:04f}.ply",np.hstack([x.detach().numpy(),np.zeros((x.shape[0],1))]),F.detach().numpy())
#    integrate()




ps.init()
ps.set_give_focus_on_show(True)
ps_mesh = ps.register_surface_mesh("x", x.detach().numpy(), F.detach().numpy());
##ps_vec = ps_mesh.add_vector_quantity("grad", dx*100, enabled=True)
#ps.set_ground_height(-2.) # in world coordinates
ps.set_ground_plane_height_factor(0.125, is_relative=False) # in world coordinates

def callback():
    global t, x, xdot, xprev
    if t > 5:
        t = 0
        x = X.clone().detach().requires_grad_(True)
        xdot = torch.zeros_like(x)
        xprev = x.clone().detach()
    integrate()
    ps_mesh.update_vertex_positions(x.detach().numpy())


ps.set_user_callback(callback)

ps.show()
