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
- Three post-publication library features **stack** to close the 1M gap, each measured:
  `compile` (block-Hessian fusion) → **2.24×**, then `cache_indices` (reuse the fixed sparse
  pattern, as RXMesh does) → **1.60×**, then `dense_gradient` (a compilable/capturable gradient,
  so the gradient stops being an eager floor) → **1.38×**. Net: the paper's **6.2× becomes
  ≈1.38×** on equal hardware.
- The wins are **larger at small/medium meshes** (mass-spring compile+cache is **~9×** vs eager
  at ≤10k vertices) and taper at 1M as real compute/assembly dominates. At the very smallest
  sizes RXMesh's near-zero-overhead design still wins by 6–13× — those are sub-millisecond
  problems where IndexedSum's fixed per-call overhead dominates.
- The two library additions here (`cache_indices`, `dense_gradient`) are opt-in and default-off;
  `dense_gradient` also lifts the other apps — notably **manifold-opt** (whose gradient was its
  dominant cost: best 3.95→**0.74 ms**) and **smoothing** (a gradient-only workload the switches
  now *do* help: 2.86→**0.95 ms** at 1M).
- A **correction to the paper's methods text**: it says IndexedSum "performs … reverse-mode
  AD." That is not the configuration the paper benchmarked — the **eager default** (`IS eager`,
  what RXMesh compared against) is **forward-over-reverse** (`torch.func.hessian = jacfwd∘jacrev`),
  which uses *both* modes; only the new opt-in `compile`/`cuda_graphs` paths are reverse-mode
  (details below).

Everything here is measured on **one NVIDIA L40 (48 GB), CUDA 12.6 toolkit, PyTorch
2.11.0+cu128**, single precision (RXMesh uses `float`), medians over repeats with warmup
excluded. The paper used an RTX 4090; absolute ms therefore differ, so the object of study is
the **RXMesh/IndexedSum ratio measured for both systems on the same L40**, with the paper's
numbers used only as a cross-check.

---

## 1. A correction to the paper's description of IndexedSum

> The paper states: *"IndexedSum performs a vectorized dense AD on the local Hessians using
> reverse-mode AD."*

This is imprecise, and it describes the configuration the paper actually benchmarked — the
**eager default** (`IndexedSum(...).sparse_hessian(...)` with no flags; the `IS eager` row in
§3, which is what RXMesh compared against). That default builds each element block with
`torch.func.hessian`, and PyTorch defines (verified in `torch/_functorch/eager_transforms.py`,
and the docstring: *"via a forward-over-reverse strategy"*):

```
torch.func.hessian(f) == jacfwd(jacrev(f))
```

i.e. **forward-over-reverse** — an *outer forward-mode* pass over an inner reverse-mode pass
(both modes, not reverse-mode alone). This matters because the *outer forward mode under
`vmap`* is exactly the path with the known `torch.linalg.det`/`slogdet` miscomputation (see the
repo's `bench/RESULTS.md` and `tests/test_det.py`), which is why this repo ships the
elementary-op `indexed_sum.det` helper. Only the *new, opt-in* `compile` / `cuda_graphs` paths
(the faster rows in §3, which post-date the paper) use reverse-over-reverse (`jacrev(jacrev)`)
— so "reverse-mode AD" describes those, not the benchmarked default.

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

All IndexedSum configurations below use the compilable `dense_gradient` for the gradient (so
the gradient is accelerated alongside the Hessian in the compiled variants). "×faster" =
IndexedSum time ÷ RXMesh time (how much faster RXMesh is).

| grid n | vertices | RXMesh | IS eager | IS compile | IS compile+cache | RXMesh× vs eager | RXMesh× vs best IS |
|-------:|---------:|-------:|---------:|-----------:|-----------------:|-----------------:|-------------------:|
| 10     | 100       | 0.121 ms | 14.32 ms | 2.05 ms | 1.54 ms | 118× | 13× |
| 100    | 10,000    | 0.263 ms | 13.79 ms | 2.06 ms | 1.56 ms | 52×  | 5.9× |
| 500    | 250,000   | 3.50 ms  | 13.74 ms | 6.70 ms | 4.61 ms | 3.9× | 1.32× |
| 1000   | 1,000,000 | 13.61 ms | 49.60 ms | 27.97 ms | 18.78 ms | **3.6×** | **1.38×** |

("best IS" = fastest IndexedSum config, `compile+cache`.)

**How the 1M gap closed — each post-publication feature, measured.** Rows add one library
feature at a time (rows 2–4 use autograd `backward` for the gradient, the paper-era path; the
last switches to the compiled `dense_gradient`):

| at 1M vertices | ms | RXMesh× |
|-|-----:|--:|
| paper (RTX 4090; none of the below existed) | — | 6.2× |
| eager default, this L40 (PyTorch 2.11) | 50.9 | 3.7× |
| + `compile` (block-Hessian fusion) | 30.5 | 2.24× |
| + `cache_indices` (reuse fixed sparse pattern) | 21.8 | 1.60× |
| + `dense_gradient` (compiled gradient) | 18.8 | **1.38×** |

**Reading it.**
- At the paper's headline size (**1M vertices**): the paper reported **6.2×**. On equal hardware
  the eager default is already **3.7×** (newer PyTorch), and the three opt-in features bring it to
  **1.38×** — RXMesh still wins, but by ~a fifth of the originally reported factor.
- At **small/medium meshes** the *relative* wins are largest (compile+cache is ~9× over eager at
  ≤10k vertices), yet RXMesh is still 6–13× faster there — these are sub-millisecond problems
  where IndexedSum's fixed per-call overhead dominates and RXMesh's near-zero-overhead design
  wins decisively. The closest IndexedSum gets is at 250k–1M (1.3–1.4×).
- The per-feature win **shrinks with size**: at ≤10k vertices the Diff is nearly pure
  launch/dispatch overhead that fusion + graphs remove (→ ~9×), while at 1M real compute and the
  sparse-COO assembly dominate. §3a decomposes the assembly (and the `cache_indices` fix);
  §3b covers the gradient (`dense_gradient`), which was the remaining eager floor.

**Why our eager IndexedSum beats the paper's (50.9 vs 83.9 ms).** Same code path, newer
PyTorch (2.11 vs whatever the paper used) and the L40. We did not change the library's eager
path. This is the honest reason the "before" gap is 3.7×, not 6.2×, independent of the new
switches.

Reproduce: `bench/rxmesh/run_rxmesh_massspring.sh` (RXMesh side) and
`python bench/rxmesh/mass_spring_bench.py --sizes 10 100 500 1000` (IndexedSum side). Raw
numbers in `mass_spring_rxmesh.csv` / `mass_spring_is.csv`.

### 3a. Where the assembly time goes, and a fairness fix (`cache_indices`)

Neither `compile` nor `cuda_graphs` touches `sparse_hessian`'s **assembly**. Decomposing that
assembly (1M, the dominant spring term) shows it is almost entirely one avoidable thing:

| step | ms |
|-|--:|
| rebuild row/col **index** tensors every call | 7.50 |
| construct the COO once indices exist | 0.008 |
| block-Hessian compute (compiled) | 4.53 |
| **full `sparse_hessian`** | **12.02** |

The returned matrix is already a **device-side** `torch.sparse_coo` tensor (not CPU) — but it is
rebuilt from scratch each call, and its row/col indices are a pure function of the mesh topology
(`all_indices`), i.e. **fixed across every Newton/timestep iteration**. RXMesh, by contrast,
computes its CSR sparsity pattern **once** and only scatters new values. Rebuilding that fixed
index tensor (an `arange` + two `expand`+`reshape` into a 108M-entry int64 tensor) is the 7.5 ms.

This repo now has an **opt-in `IndexedSum(..., cache_indices=True)`** that caches the index
tensor and reuses it — the same "pattern computed once" strategy RXMesh uses, appropriate to a
fixed-topology solve loop (trades a persistent `[2, nnz]` int64 tensor for the per-call rebuild).
It is numerically identical to the default (verified, 0.0 rel-err) and composes with
`compile`/`cuda_graphs`. Effect on the mass-spring Diff:

| grid n | vertices | RXMesh | IS `compile` | IS `compile+cache` | RXMesh× vs `compile+cache` |
|-------:|---------:|-------:|-------------:|-------------------:|---------------------------:|
| 500    | 250,000   | 3.50 ms  | 7.31 ms  | 5.16 ms  | 1.47× |
| 1000   | 1,000,000 | 13.61 ms | 30.48 ms | 21.76 ms | **1.60×** |

With `cache_indices` the 1M gap goes **2.24× (compile) → 1.60× (compile+cache)**. The next
remaining piece was the gradient (§3b).

### 3b. The gradient floor, and a second fix (`dense_gradient`)

After the Hessian was compiled and the pattern cached, the **eager gradient became the floor**:
IndexedSum got its gradient via `energy(V).backward()` (eager autograd), ~3.8 ms at 1M — larger
than the compiled+cached Hessian assembly it sat next to. RXMesh instead scatters per-element
gradients on-device with the rest of `eval_terms`.

The core library now has an opt-in **`IndexedSum.dense_gradient(V)`** (a compilable/capturable
per-element gradient assembly, mirroring `sparse_hessian`; honors the same
`compile`/`cuda_graphs`/`cache_indices` flags). It is numerically identical to autograd
(verified to ~1e-16). Switching the benchmark's gradient to it drops the 1M mass-spring
`compile+cache` Diff from **21.8 → 18.8 ms** (gap **1.60× → 1.38×**), and — because `jacrev` is
reverse-mode — it compiles cleanly with no reformulation. Per-element gradient alone at 1M:
**eager 3.2 ms → compiled+cached 0.95 ms** (~3.4×).

So the 1M gap collapses across the full sequence **6.2× (paper) → 3.7× (eager, same HW) → 2.24×
(+compile) → 1.60× (+cache_indices) → 1.38× (+dense_gradient)**. The residual ~1.4× is now the
**multi-term `SumNode` sparse-adds** (IndexedSum assembles spring/inertial/gravity as three
separate matrices and adds them, where RXMesh accumulates all terms into one shared CSR pattern)
plus IndexedSum computing the gradient and Hessian in two passes where RXMesh fuses them — a
single shared assembly pattern and a fused grad+Hessian would be the next steps.

---

## 4. The other three applications

The paper compares these against *PyTorch/JAX/etc., not IndexedSum*, so there is no prior
IndexedSum number to move. We still port each energy to IndexedSum on the same L40, for an
apples-to-apples extension — with honest framing of where the comparison is or isn't clean.

### 4a. Laplacian smoothing — gradient only (paper Fig 6)
RXMesh's Smoothing app is **gradient descent**: its Diff is a *gradient*, no Hessian. With the
new **`dense_gradient`** path (§3b) IndexedSum now *does* have a compilable/capturable gradient,
so unlike the Hessian-only `compile`/`cuda_graphs` this workload benefits:

| grid n | vertices | RXMesh grad/iter | IS eager autograd | IS `dense_gradient` compiled+cached | RXMesh× |
|-------:|---------:|-----------------:|------------------:|-----------------------------------:|--------:|
| 100    | 10,000    | 0.020 ms | 0.41 ms | 0.20 ms | 10× |
| 500    | 250,000   | 0.108 ms | 0.47 ms | 0.21 ms | 2.0× |
| 1000   | 1,000,000 | 0.420 ms | 2.86 ms | **0.95 ms** | **2.3×** |

The compiled gradient cuts the 1M gap from **6.7× to 2.3×**. Honest note: IndexedSum is still a
**sparse-Hessian** tool at heart — for pure gradient descent it pays a full per-element
`vmap(jacrev)` + scatter, so at the smallest meshes RXMesh's near-zero overhead still wins ~10×,
and `dense_gradient` eager is actually a touch slower than plain `backward` there (it only pays
off once compiled). `cache_indices` applies via the gradient's own scatter map.

### 4b. Parameterization — symmetric Dirichlet (paper Table 2)
RXMesh's Param uses a **matrix-free CG Newton** solver (`eval_terms_grad_only` + Hessian-vector
products, **no Hessian assembly**). IndexedSum's niche — an *assembled sparse Hessian* — is a
*different algorithm*, so a direct per-iteration RXMesh-vs-IndexedSum number would be
apples-to-oranges. What we can show cleanly is the IndexedSum derivative-provision cost for the
same energy, and that the new switches apply to it (this energy has a `J⁻¹` / determinant
term):

| grid n | vertices | IS eager | IS compile | IS compile+cache | IS cuda_graphs+cache | best speedup |
|-------:|---------:|---------:|-----------:|-----------------:|---------------------:|-------------:|
| 100    | 10,000    | 10.11 ms | 1.04 ms | 0.88 ms | **0.64 ms** | 15.7× |
| 500    | 250,000   | 14.82 ms | 6.74 ms | 5.48 ms | 5.61 ms | 2.7× |
| 1000   | 1,000,000 | 80.23 ms | 30.54 ms | **25.18 ms** | 26.36 ms | 3.2× |

(All configs use `dense_gradient`; `cache_indices` and the compiled gradient both stack with
`compile`/`cuda_graphs`. "best speedup" is the fastest configuration vs eager.)

The symmetric-Dirichlet Hessian is verified against finite differences (below), computed with
the elementary 2×2 determinant/inverse (the `indexed_sum.det` remedy) — required for
`cuda_graphs` capturability regardless of the det bug. RXMesh's paper result here is 2.76× over
*PyTorch's dense* Hessian; a sparse IndexedSum + `cuda_graphs` is a stronger PyTorch-side
baseline than the paper's dense one.

### 4c. Manifold optimization — spherical parameterization
RXMesh's ManiOpt is a **Newton method with an assembled Hessian** (like mass-spring), so this
*is* directly analogous. Its energy contains a **3×3 determinant** (a signed volume) — the same
family as the documented det bug. On the paper's giraffe mesh (3,130 V / 6,256 F):

| | RXMesh Diff/iter | IS eager | IS compile | IS cuda_graphs | IS cuda_graphs+cache |
|-|-----------------:|---------:|-----------:|---------------:|---------------------:|
| giraffe | 0.20 ms | 33.3 ms | 1.45 ms (23×) | 0.94 ms (35×) | **0.74 ms (45×)** |

This app benefits most from `dense_gradient`: its retraction+barrier energy has an expensive
per-element gradient, so with autograd `backward` the gradient *was* ~90% of the compiled Diff
(best was 3.95 ms). Compiling the gradient too collapses it — best **0.74 ms (45× over eager)**,
now within **3.7×** of RXMesh (0.20 ms) on this small mesh. (A larger genus-0 mesh with a sphere
embedding would narrow it further, as at mass-spring's 1M row; we only had giraffe's embedding to
match RXMesh exactly.)

---

## 5. Correctness (gates run before any timing)

Every configuration timed above passes an AD-independent check. Representative (f64):

| app | eager Hessian vs finite-diff | compile vs eager | cuda_graphs vs eager |
|-----|-----------------------------:|-----------------:|---------------------:|
| mass-spring | 6.1e-08 | 1.7e-16 | 1.7e-16 |
| param (sym-Dirichlet) | 5.0e-06 | 6.6e-16 | 6.6e-16 |
| mani-opt (spherical) | 6.6e-04 (grad vs FD 2.1e-06) | 6.7e-13 | 6.7e-13 |

The `dense_gradient` used for timing is checked against autograd in every gate (≤1e-16 f64;
mass-spring/smoothing print it explicitly) and carries its own unit tests in the core library
(`tests/test_gradient_compile.py`).

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
- **What's addressed vs. what remains:** the two largest avoidable costs are now fixed —
  rebuilding fixed indices (`cache_indices`, §3a) and the eager gradient (`dense_gradient`, §3b)
  — taking the 1M gap to ≈1.38×. What remains: IndexedSum still assembles multi-term energies as
  separate matrices it sparse-adds (vs RXMesh's one shared CSR pattern) and computes grad/Hessian
  in two passes (vs RXMesh's fused `eval_terms`). A shared assembly pattern and a fused
  grad+Hessian are the natural next steps.
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

The exact RXMesh source changes are captured in `bench/rxmesh/rxmesh_headless.patch`
(`git apply` from an RXMesh checkout). Minimal RXMesh source touches for this headless L40
(documented, do not affect timed code):
`apps/MassSpring/mass_spring_diff.cu` (new eval_terms timing harness); a `#if USE_POLYSCOPE`
guard around one Drop-scene `registerSurfaceMesh` call in `apps/MassSpring/draw.h`; explicit
`glm/gtx/norm.hpp` + `glm/gtc/constants.hpp` includes in `apps/ManiOpt/mean_curv.h`; and a
2-line `cuda_profiler_api.h` shim (absent from the assembled toolkit; symbols live in
libcudart). CUDA math libs (cusparse/cusolver/cublas) were assembled into a complete 12.6
toolkit from the system's runtime libs plus matching cu12 headers.
```
```
