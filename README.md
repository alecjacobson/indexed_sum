# indexed_sum: fast sparse Hessians in pytorch

This small library provides a python class `IndexedSum` to use pytorch to construct the sparse Hessian of functions of
the form:

$$
f(x) = \sum_i g(S_i x)
$$

where $g : \mathbb{R}^m \to \mathbb{R}$ is a scalar function conducting the
`local_summand` contribution and $S_i$ is a selection matrix picking out the $m$
rows of $x$ that are relevant to the $i$-th local summand. These types of energies are very common for problems defined on graphs or meshes (e.g., Finite Element Methods or Finite Volume Methods).

The Hessian of $f$ is then:

$$
\frac{\partial^2 f}{\partial x^2} = \sum_i S_i^T \frac{\partial^2 g}{\partial x^2} S_i
$$

where each $\frac{\partial^2 g}{\partial x^2}$ is a (dense) $m \times m$ matrix.

The `IndexedSum` class is a thin wrapper around pytorch's autograd machinery.

This provides similar functionality to the C++ library [TinyAD](https://github.com/patr-schm/TinyAD).

## Installation

Try it out with:

```
pip install git+https://github.com/alecjacobson/indexed_sum.git
```

## Simple Example

Let's consider a very small mass-spring system example


```python
from indexed_sum import IndexedSum
import torch

# A square made of 4 springs
X = torch.tensor([[0,0],[1,0],[1,1],[0,1]],dtype=torch.float64,requires_grad=True)
E = torch.tensor([[0,1],[1,2],[2,3],[3,0]],dtype=torch.int64)

# local contribution for a single spring
def spring(x):
    return 0.5*(torch.norm(x[1] - x[0])**2)

# sum([spring(X[edge]) for edge in E])
total_energy_function = IndexedSum(spring,E)
value = total_energy_function(X)

# gradient in the usual pytorch way
value.backward()
grad = X.grad

# second derivatives using IndexedSum's sparse scatter-gather
H = total_energy_function.sparse_hessian(X)
```

The alternative would be to use pytorch to compute the Hessian but this will be
dense:

```python
H_dense = torch.func.hessian(lambda X: sum([spring(X[edge]) for edge in E]))(X)
```

For small examples this doesn't matter, but if the number of variables in X is
large, the sparse Hessian can be much more efficient to compute and store.

## Constants

We support constant (non-differentiated) parameters to the local summands, in
the form:

$$
f(x) = \sum_i g(S_i x, S_i k,  c_i)
$$

where $k$ is a matrix of constant parameters where each row corresponds to a row
in the differentiated variables $x$ (aka, "nodal" parameters) and $c$ is a
matrix of constant parameters where each row corresponds to a summation term
(aka, "element" parameters).


The `local_summand` function should implemented so that forward evaluation is
equivelent to:

```python
sum([local_summand(X[elem], K[elem], c[i]) for i, elem in enumerate(all_indices)])
```

We can adapt the mass-spring example above to take into account per-node factors
(`m`) and per-edge rest lengths (`rest_l`):

```python
# A square made of 4 springs and one diagonal
X = torch.tensor([[0,0],[1,0],[1,1],[0,1]],dtype=torch.float64,requires_grad=True)
E = torch.tensor([[0,1],[1,2],[2,3],[3,0],[0,2]],dtype=torch.int64)
M = torch.tensor([1,2,2,1],dtype=torch.float64).unsqueeze(1)
R = 0.5*torch.norm(X[E[:,0]] - X[E[:,1]],dim=1).detach().unsqueeze(1)

def spring(v,m,r):
    v0 = v[0]
    v1 = v[1]
    m0 = m[0]
    m1 = m[1]
    l = torch.norm(v1 - v0)
    return 0.5 * (m0 + m1) * (l - r)**2;

total_energy_function = IndexedSum(
    local_summand=spring,
    all_indices=E,
    per_variable_constants=M,
    per_term_constants=R)
H = total_energy_function.sparse_hessian(X)
```

## Complex Example

https://github.com/user-attachments/assets/a0d45b90-37ae-474e-9fc6-ecc735ee6c67

In [examples/neohookean_elasticity.py](examples/neohookean_elasticity.py), we
build out a Newton-Raphson solver for a 2D neohookean material elasticity
animation. For a triangle mesh,

```python
# simulation timestep duration
dt = 1/66
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
```


The key energies are the "Stable Neo-Hookean" energy and the
momentum term. These are defined as 

```python
# define local summand of stable neo-Hookean energy
def stable_neohookean(x,X,scale):
    def build_M(x0,x1,x2):
        return torch.stack([x1-x0,x2-x0],dim=1)
    M = build_M(X[0],X[1],X[2])
    m = build_M(x[0],x[1],x[2])
    youngs_modulus = 0.37e3
    poissons_ratio = 0.4
    mu = youngs_modulus / (2.0 * (1.0 + poissons_ratio))
    lam = youngs_modulus * poissons_ratio / 
        ((1.0 + poissons_ratio) * (1.0 - 2.0 * poissons_ratio))
    J = m @ torch.inverse(M)
    A = 0.5 * torch.det(M)
    Ic = torch.trace(J.T @ J)
    # https://github.com/pytorch/pytorch/issues/149694
    det2 = lambda J: J[0,0] * J[1,1] - J[0,1] * J[1,0]
    detF = det2(J)
    alpha = 1.0 + mu / lam
    W = mu / 2.0 * (Ic - 2.0) + 
        lam / 2.0 * (detF - alpha) * (detF - alpha)
    return scale * A * W

# define local summand of momentum potential energy
def momentum_potential(x,xtilde,m):
    delta = x - xtilde
    return 0.5 * m * torch.sum(delta * delta)
```

These terms are then simply added together as `IndexedSum`s.

```python
elastic_potential_func = IndexedSum(
    local_summand=lambda x,X: stable_neohookean(x,X,scale=dt**2),
    all_indices=F,
    per_variable_constants=X)
# momentum_potential_func.per_variable_constants 
# will be updated later each timestep
momentum_potential_func = IndexedSum(
    local_summand=momentum_potential,
    all_indices=I,
    per_term_constants=M)
total_energy_func = elastic_potential_func + momentum_potential_func
```

Later in the simulation loop we can gather the derivatives to prepare the
Newton solve:

```python
# momentum's prediction 
xtilde = xprev + dt * xdot + dt**2 * g
# update constant parameters
momentum_potential_func.set_per_variable_constants(xtilde)
# collect gradient and hessian
total_energy = total_energy_func(x)
total_energy.backward()
grad = x.grad
H = total_energy_func.sparse_hessian(x)
```


## Determinants in summands (important)

`sparse_hessian` differentiates each `local_summand` with `torch.func.hessian`
(forward-over-reverse) under `vmap`. PyTorch's forward-mode autodiff of
`torch.linalg.det` / `torch.det` / `torch.linalg.slogdet` is **wrong under
`vmap`** (an open upstream bug, [pytorch#149694](https://github.com/pytorch/pytorch/issues/149694)),
so a summand that calls `torch.linalg.det` silently produces an **incorrect
Hessian** (verified off finite differences by >100×). The symptom is
data-dependent, so it can hide in small tests.

Use the drop-in, `vmap`-safe helpers instead — they compute the determinant with
elementary ops, which differentiate correctly:

```python
from indexed_sum.det import det, logabsdet   # instead of torch.linalg.det / slogdet

def neohookean(v):
    F = v[1:] - v[0:1]
    J = det(F)                                # NOT torch.linalg.det(F)
    return (F*F).sum() - 3 - 2*torch.log(torch.clamp(J, min=1e-3)) + (J-1)**2
```

Only the determinant family is affected; `torch.inverse`, matmul, `trace`,
elementwise ops, etc. are fine.

## Speeding up with `torch.compile`

`sparse_hessian` is many small per-element autograd graphs, a workload dominated
by kernel-launch overhead. Two opt-in, *layered* switches cut that overhead:

```python
f = IndexedSum(neohookean, T, compile=True)      # Inductor kernel fusion
f = IndexedSum(neohookean, T, cuda_graphs=True)  # + CUDA graphs (fastest)
H = f.sparse_hessian(x)                           # first call compiles; then reused
```

`cuda_graphs` is a refinement of `compile` (it records the fused launches into a
CUDA graph and replays them), so `cuda_graphs=True` implies compilation — they're
compatible, not exclusive:

| `compile` | `cuda_graphs` | behavior                          |
|-----------|---------------|-----------------------------------|
| `False`   | `False`       | eager (default)                   |
| `True`    | `False`       | torch.compile, Inductor fusion    |
| any       | `True`        | torch.compile + CUDA graphs       |

This helps most for cheap summands on GPU inside a repeated-call loop (e.g. Newton
iterations at fixed shapes): ~30–50× on an L40, up to ~85× with `cuda_graphs=True`.
Both default to `False`. Note `cuda_graphs` requires a summand free of host↔device
syncs — another reason to use `indexed_sum.det.det` rather than `torch.linalg.det`.
See `bench/RESULTS.md` for the full study.

## Reusing the sparsity pattern with `cache_indices`

Beyond the per-element block compute, `sparse_hessian` assembles the result by
building the matrix's row/col index tensors — but those depend only on
`all_indices` (the mesh topology), so at a fixed problem they are **constant across
every call**. `cache_indices=True` computes them once and reuses them, the same
"pattern computed once" strategy sparse solvers use:

```python
f = IndexedSum(neohookean, T, compile=True, cache_indices=True)
H = f.sparse_hessian(x)   # index tensors built once, then reused each call
```

It is numerically identical to the default, composes with `compile`/`cuda_graphs`,
and trades a persistent `[2, nnz]` index tensor for the per-call rebuild — a large
assembly saving in fixed-topology solve loops (at 1M vertices it roughly halves the
`compile` assembly time). Defaults to `False`.

## Compiling the gradient with `dense_gradient`

`compile`/`cuda_graphs`/`cache_indices` all target `sparse_hessian`; the gradient
via `energy(x).backward()` stays eager. `IndexedSum.dense_gradient(x)` assembles the
dense gradient vector from per-element gradients the same way, and honors the same
flags — so the gradient is compilable and CUDA-graph capturable too (useful when the
gradient dominates, e.g. gradient-descent loops):

```python
f = IndexedSum(energy, T, compile=True, cache_indices=True)
g = f.dense_gradient(x)   # equals autograd's gradient; compiled/captured
```

_You might also be interested in https://github.com/alecjacobson/tinyremo and https://github.com/alecjacobson/pytorch-sparse-solve_
