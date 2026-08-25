# RXMesh vs IndexedSum, revisited with `compile` / `cuda_graphs`

**What this is.** The RXMesh AD paper — *"Locality-Aware Automatic Differentiation on the GPU
for Mesh-Based Computations"* (Mahmoud, Goel, Ragan-Kelley, Solomon; arXiv:2509.00406) —
benchmarks RXMesh against this library, `indexed_sum`. Its headline IndexedSum result
(Table 1, mass-spring cloth) is **RXMesh ≈ 6.2× faster than IndexedSum at ≈1M vertices**. That
measurement predates this repo's `IndexedSum(compile=True)` / `IndexedSum(cuda_graphs=True)`
switches. This study re-runs the comparison **with both systems on the same GPU**, first
reproducing the paper's finding with eager IndexedSum, then quantifying how the new switches
change it.

**TL;DR.**
- On equal hardware (one L40), **RXMesh's mass-spring Diff time reproduces the paper almost
  exactly** (13.6 ms vs the paper's 13.53 ms at 1M vertices) — the RXMesh side is faithful.
- With **current PyTorch (2.11)**, *eager* IndexedSum is already faster than the paper measured
  (50.9 ms vs 83.9 ms at 1M), so the gap at 1M is **3.7×, not 6.2×**, before any new feature.
- Turning on **`cuda_graphs` narrows the 1M gap to ≈2.35×** (and `compile` to ≈2.24×). RXMesh
  still wins at every size, but by less than the paper reports.
- The compile win is **large at small/medium meshes (≈5×) and shrinks at 1M (≈1.7×)** because at
  scale the *eager gradient pass and the sparse-COO assembly* — neither of which the switches
  touch — dominate, not the block-Hessian kernel.
- A **correction to the paper's methods text**: it says IndexedSum "performs … reverse-mode
  AD." The default path is **forward-over-reverse** (`torch.func.hessian = jacfwd∘jacrev`),
  which uses *both* modes (details below).

Everything here is measured on **one NVIDIA L40 (48 GB), CUDA 12.6 toolkit, PyTorch
2.11.0+cu128**, single precision (RXMesh uses `float`), medians over repeats with warmup
excluded. The paper used an RTX 4090; absolute ms therefore differ, so the object of study is
the **RXMesh/IndexedSum ratio measured for both systems on the same L40**, with the paper's
numbers used only as a cross-check.

---

## 1. A correction to the paper's description of IndexedSum

> The paper states: *"IndexedSum performs a vectorized dense AD on the local Hessians using
> reverse-mode AD."*

This is imprecise. `IndexedSum.sparse_hessian` builds each element block with
`torch.func.hessian`, and PyTorch defines (verified in `torch/_functorch/eager_transforms.py`,
and the docstring: *"via a forward-over-reverse strategy"*):

```
torch.func.hessian(f) == jacfwd(jacrev(f))
```

i.e. **forward-over-reverse** — an *outer forward-mode* pass over an inner reverse-mode pass.
It is not reverse-mode. This matters because the *outer forward mode under `vmap`* is exactly
the path with the known `torch.linalg.det`/`slogdet` miscomputation (see `bench/RESULTS.md`
and `tests/test_det.py`), which is why this repo ships the elementary-op `indexed_sum.det`
helper. (The `compile`/`cuda_graphs` path instead uses reverse-over-reverse,
`jacrev(jacrev)`.)

---

## 2. Setup and what is measured (fairness)

**Same GPU for both systems.** All RXMesh and IndexedSum numbers below are on the one L40.

**Same quantity.** RXMesh's apps time a **"Diff"** phase = `problem.eval_terms()` = gradient +
Hessian assembly, reported separately from the linear solver / line search. The IndexedSum
side times the identical thing: **gradient (autograd `backward`) + the full `sparse_hessian`**
(block compute *and* the `sparse_coo_tensor` assembly), excluding the solve. We report the
*end-to-end* number, not the block-Hessian kernel alone — reporting kernel-only would be unfair
to RXMesh (the eager assembly is a real IndexedSum cost).

**Per single evaluation.** We compare **one grad+Hessian evaluation** on each side
(RXMesh's Diff/Newton-iter vs one IndexedSum grad+`sparse_hessian`). This is invariant to how
many Newton iterations a solver takes, so it avoids conflating derivative cost with solver
convergence. (Per-timestep just multiplies both sides by the shared iteration count; the ratio
is identical.)

**Same energies and mesh.** For mass-spring we replicate RXMesh's Flag-scene energies exactly
(inertial + spring with squared rest length + gravity) and its `create_plane(n,n)` grid; our
`igl.triangulated_grid(n,n)` produces the *identical* vertex/edge/face counts (e.g. n=1000 →
1,000,000 V / 2,996,001 E / 1,996,002 F). Single precision throughout.

**Faithfulness check.** RXMesh's own Diff time on the L40 lands on the paper's RTX-4090 number:

| grid n | vertices | RXMesh Diff/eval, L40 (this study) | paper Table 1, RTX 4090 |
|-------:|---------:|-----------------------------------:|------------------------:|
| 1000   | 1,000,000 | **13.61 ms** | **13.53 ms** |

So the RXMesh measurement pipeline here is trustworthy; differences on the IndexedSum side are
about IndexedSum, not a mis-measured RXMesh.

**How RXMesh's Diff is isolated.** The shipped app couples Diff to a direct linear solve; with
no cuDSS on this box that solve (CPU Cholesky fallback) would dominate wall time at 1M vertices
but is *excluded from Table 1 anyway*. We therefore added a minimal harness,
`apps/MassSpring/mass_spring_diff.cu`, that sets up the identical problem and times
`eval_terms()` in a loop — measuring exactly the paper's Diff quantity. (Build accommodations
for this headless machine are listed in §7.)

---

## 3. Headline: mass-spring cloth (the paper's IndexedSum claim)

Per grad+Hessian evaluation, f32, one L40. "×faster" = IndexedSum time ÷ RXMesh time (how much
faster RXMesh is).

| grid n | vertices | RXMesh | IS eager | IS compile | IS cuda_graphs | RXMesh× vs eager | RXMesh× vs cuda_graphs |
|-------:|---------:|-------:|---------:|-----------:|---------------:|-----------------:|-----------------------:|
| 10     | 100       | 0.121 ms | 12.65 ms | 2.52 ms | 2.59 ms | 104×  | 21× |
| 100    | 10,000    | 0.263 ms | 13.34 ms | 2.52 ms | 2.55 ms | 51×   | 9.7× |
| 500    | 250,000   | 3.50 ms  | 12.32 ms | 7.31 ms | 7.63 ms | 3.5×  | 2.2× |
| 1000   | 1,000,000 | 13.61 ms | 50.93 ms | 30.48 ms | 31.94 ms | **3.7×** | **2.35×** |

**Reading it.**
- At the paper's headline size (**1M vertices**): the paper reported **6.2×**. On equal hardware
  with current PyTorch, *eager* IndexedSum already closes it to **3.7×**, and **`cuda_graphs`
  brings it to 2.35×** (`compile` to 2.24×). RXMesh is still faster, by roughly **half to a
  third** of the originally reported factor.
- At **small/medium meshes** the gap is huge (20–100×) and the new switches help a lot in
  *relative* terms (IndexedSum's eager ~13 ms is almost pure Python/launch overhead; compile
  collapses it to ~2.5 ms). But these are sub-millisecond RXMesh problems where IndexedSum's
  fixed per-call overhead dominates — RXMesh's near-zero-overhead design wins decisively.
- The compile **speedup shrinks with size** (5.0× → 5.3× → 1.7× → 1.7×). Decomposition explains
  why: of IndexedSum's ~13 ms eager Diff at small n, the Hessian block-compute is the launch-
  bound part the switches remove; but the **eager gradient pass (~1.3 ms floor) and the
  sparse-COO assembly (~6 ms at 250k, growing with nnz)** are *not* compiled. At 1M the eager
  Hessian is genuinely compute-bound (47 ms) and assembly-bound, so removing launch overhead
  buys less.

**Why our eager IndexedSum beats the paper's (50.9 vs 83.9 ms).** Same code path, newer
PyTorch (2.11 vs whatever the paper used) and the L40. We did not change the library's eager
path. This is the honest reason the "before" gap is 3.7×, not 6.2×, independent of the new
switches.

Reproduce: `bench/rxmesh/run_rxmesh_massspring.sh` (RXMesh side) and
`python bench/rxmesh/mass_spring_bench.py --sizes 10 100 500 1000` (IndexedSum side). Raw
numbers in `mass_spring_rxmesh.csv` / `mass_spring_is.csv`.

---

## 4. The other three applications

The paper compares these against *PyTorch/JAX/etc., not IndexedSum*, so there is no prior
IndexedSum number to move. We still port each energy to IndexedSum on the same L40, for an
apples-to-apples extension — with honest framing of where the comparison is or isn't clean.

### 4a. Laplacian smoothing — gradient only (paper Fig 6)
RXMesh's Smoothing app is **gradient descent**: its Diff is a *gradient*, no Hessian.
**IndexedSum's `compile`/`cuda_graphs` only wire into `sparse_hessian`, so they do not apply
to a gradient-only workload.** We therefore report IndexedSum's eager autograd gradient.

| grid n | vertices | RXMesh grad/iter | IS eager grad/iter | RXMesh× |
|-------:|---------:|-----------------:|-------------------:|--------:|
| 100    | 10,000    | 0.020 ms | 0.41 ms | 20× |
| 500    | 250,000   | 0.108 ms | 0.47 ms | 4.4× |
| 1000   | 1,000,000 | 0.420 ms | 2.84 ms | 6.7× |

Honest note: IndexedSum is a **sparse-Hessian** tool; for pure gradient descent it offers no
Hessian to accelerate and pays per-call autograd overhead, so RXMesh is several× faster and the
new switches change nothing. (A hypothetical *compiled-gradient* path — not in the library —
would help; a partial what-if is in `smoothing_bench.py`.)

### 4b. Parameterization — symmetric Dirichlet (paper Table 2)
RXMesh's Param uses a **matrix-free CG Newton** solver (`eval_terms_grad_only` + Hessian-vector
products, **no Hessian assembly**). IndexedSum's niche — an *assembled sparse Hessian* — is a
*different algorithm*, so a direct per-iteration RXMesh-vs-IndexedSum number would be
apples-to-oranges. What we can show cleanly is the IndexedSum derivative-provision cost for the
same energy, and that the new switches apply to it (this energy has a `J⁻¹` / determinant
term):

| grid n | vertices | IS eager | IS compile | IS cuda_graphs | compile speedup |
|-------:|---------:|---------:|-----------:|---------------:|----------------:|
| 100    | 10,000    | 10.56 ms | 2.16 ms | 1.99 ms | 4.9× |
| 500    | 250,000   | 15.03 ms | 7.46 ms | 7.58 ms | 2.0× |
| 1000   | 1,000,000 | 81.30 ms | 33.92 ms | 34.78 ms | 2.4× |

The symmetric-Dirichlet Hessian is verified against finite differences (below), computed with
the elementary 2×2 determinant/inverse (the `indexed_sum.det` remedy) — required for
`cuda_graphs` capturability regardless of the det bug. RXMesh's paper result here is 2.76× over
*PyTorch's dense* Hessian; a sparse IndexedSum + `cuda_graphs` is a stronger PyTorch-side
baseline than the paper's dense one.

### 4c. Manifold optimization — spherical parameterization
RXMesh's ManiOpt is a **Newton method with an assembled Hessian** (like mass-spring), so this
*is* directly analogous. Its energy contains a **3×3 determinant** (a signed volume) — the same
family as the documented det bug. On the paper's giraffe mesh (3,130 V / 6,256 F):

| | RXMesh Diff/iter | IS eager | IS compile | IS cuda_graphs |
|-|-----------------:|---------:|-----------:|---------------:|
| giraffe | 0.193 ms | 29.42 ms | 4.26 ms (6.9×) | 3.96 ms (7.4×) |

giraffe is small, so IndexedSum is deep in the overhead-bound regime where the switches help
most (7.4×) — yet RXMesh is still ~20–150× faster at this size, consistent with the
mass-spring small-n rows. (A larger genus-0 mesh with a sphere embedding would narrow this as
at mass-spring's 1M row; we only had giraffe's embedding to match RXMesh exactly.)

---

## 5. Correctness (gates run before any timing)

Every configuration timed above passes an AD-independent check. Representative (f64):

| app | eager Hessian vs finite-diff | compile vs eager | cuda_graphs vs eager |
|-----|-----------------------------:|-----------------:|---------------------:|
| mass-spring | 6.1e-08 | 1.7e-16 | 1.7e-16 |
| param (sym-Dirichlet) | 5.0e-06 | 6.6e-16 | 6.6e-16 |
| mani-opt (spherical) | 6.6e-04 (grad vs FD 2.1e-06) | 6.7e-13 | 6.7e-13 |

Two honest sub-findings on the determinant bug:
- The documented `torch.linalg.det`-under-vmap Hessian bug is **narrow**: in the *full* param
  (2×2 det) and mani-opt (3×3 det) energies at feasible configs, `torch.linalg.det` and the
  elementary helper **agree** (≤1e-12) — the bug does *not* trigger. The canonical, isolated
  reproduction remains `tests/test_det.py` (neohookean, off by ~250×).
- The elementary `indexed_sum.det` helper is still **required for `cuda_graphs`** on these
  energies: `torch.linalg.det` forces a host sync and cannot be captured.

---

## 6. Honest caveats and limitations

- **Hardware differs from the paper** (L40 vs RTX 4090). We do *not* mix hardware in any ratio;
  every RXMesh-vs-IndexedSum comparison is same-L40. The paper's numbers appear only as the §2
  cross-check, which matched.
- **Newer PyTorch** is the main reason eager IndexedSum improved on its own (3.7× vs the paper's
  6.2×). This is not attributable to the new switches; we separate the two effects.
- **What the switches don't touch:** the gradient pass and the sparse-COO *assembly* are eager
  and graph-break — they dilute the block-Hessian speedup, increasingly so at scale. This is
  why the mass-spring win drops to ~1.7× at 1M. An assembly/gradient that were also compiled
  (a library change, out of scope here) would push the ratio further.
- **Warmup / fixed shapes.** `compile`/`cuda_graphs` pay a one-time compile cost (~2–14 s,
  reported per row) and require fixed shapes — realistic for a Newton/timestep loop that calls
  `sparse_hessian` repeatedly at one shape, which is the regime here. Warmup is excluded from
  the steady-state medians (as in the paper).
- **PyTorch dense baseline** (`torch.func.hessian`) OOMs by 10k vertices on the 48 GB L40 in our
  formulation; the paper's PyTorch column survived further (28 s at 10k), suggesting a different
  PyTorch formulation. PyTorch is not the object of study here, so we did not chase this.
- **ManiOpt is single-size** (only giraffe's embedding was available to match RXMesh exactly);
  Param's RXMesh comparison is algorithmically different (matrix-free) and so is reported as an
  IndexedSum-side result rather than a head-to-head ratio.

---

## 7. Reproduce

Environment: one NVIDIA L40, driver 570 (CUDA 12.8 runtime); PyTorch 2.11.0+cu128; RXMesh built
with a CUDA 12.6 toolkit.

IndexedSum side (this repo):
```
python bench/rxmesh/mass_spring_bench.py --check          # correctness gate
python bench/rxmesh/mass_spring_bench.py --sizes 10 100 500 1000
python bench/rxmesh/smoothing_bench.py  --sizes 100 500 1000
python bench/rxmesh/param_bench.py      --sizes 100 500 1000
python bench/rxmesh/maniopt_bench.py                       # giraffe
python bench/rxmesh/make_grids.py                          # regenerate plane-grid OBJs
```

RXMesh side (external checkout of owensgroup/RXMesh @ 13cc82c):
```
# built headless: -DRX_USE_POLYSCOPE=OFF -DRX_BUILD_TESTS=OFF -DRX_USE_CUDSS=OFF
#                 -DCMAKE_CUDA_ARCHITECTURES=89 -DCMAKE_POLICY_VERSION_MINIMUM=3.5
bench/rxmesh/run_rxmesh_massspring.sh    # MassSpringDiff harness -> mass_spring_rxmesh.csv
build/bin/Smoothing -i grid_<n>.obj -n 500
build/bin/Param     -i grid_<n>.obj -s cg_mat_free
build/bin/ManiOpt   -i giraffe.obj -e giraffe_embedding.obj -s newton -m 100
```

Minimal RXMesh source touches for this headless L40 (documented, do not affect timed code):
`apps/MassSpring/mass_spring_diff.cu` (new eval_terms timing harness); a `#if USE_POLYSCOPE`
guard around one Drop-scene `registerSurfaceMesh` call in `apps/MassSpring/draw.h`; explicit
`glm/gtx/norm.hpp` + `glm/gtc/constants.hpp` includes in `apps/ManiOpt/mean_curv.h`; and a
2-line `cuda_profiler_api.h` shim (absent from the assembled toolkit; symbols live in
libcudart). CUDA math libs (cusparse/cusolver/cublas) were assembled into a complete 12.6
toolkit from the system's runtime libs plus matching cu12 headers.
```
```
