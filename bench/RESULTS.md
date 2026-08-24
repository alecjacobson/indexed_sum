# Does `torch.compile` speed up `IndexedSum` sparse Hessians?

**Short answer:** Sometimes — and getting there uncovered a *correctness bug* in the current
library that matters more than the speed.

- `torch.compile` cannot wrap the library's Hessian as-is. `torch.func.hessian` is
  forward-over-reverse (`jacfwd∘jacrev`), which **fails to compile** under dynamo (torch
  2.11). Compiling requires reformulating the Hessian as **reverse-over-reverse**
  (`jacrev∘jacrev`).
- Separately, the study uncovered a **correctness bug**: **forward-over-reverse under `vmap`
  computes the wrong Hessian for `torch.det`-based energies** (e.g. neohookean elasticity) — a
  latent, data-dependent bug the current `IndexedSum.sparse_hessian` inherits (the repo's own
  example already hand-expands a 2×2 determinant to dodge it, citing pytorch#149694). The
  narrow, low-risk fix is a **vmap-safe determinant helper** (`indexed_sum.det`) used inside
  summands instead of `torch.linalg.det`; the fast default Hessian path is unchanged.
- **Speed: it depends on whether the per-element work is overhead-bound or compute-bound.**
  For cheap summands (constant/low arithmetic, GPU) the Hessian is dominated by kernel-launch
  overhead, and `torch.compile(mode="reduce-overhead")` (CUDA graphs) gives **~35×** on the
  hot kernel. For expensive summands (neohookean's `det`/`log`) the GPU is already
  compute-bound and compile gives **no win** — and CUDA-graph capture is impossible anyway
  because those ops force a host sync.

Environment: torch 2.11.0+cu128, NVIDIA L40, driver 570.158 (CUDA 12.8).

---

## 1. Can you compile it at all?

No, not the library's formulation. `torch.func.hessian(f) == jacfwd(jacrev(f))`. Under
`torch.compile` the forward-mode (`jacfwd`) part raises:

```
RuntimeError: InferenceMode::is_enabled() ... native::_fw_primal ... INTERNAL ASSERT FAILED
```

for every backend (`inductor`, `aot_eager`). Isolating the transforms:

| transform under `torch.compile`         | result   |
|-----------------------------------------|----------|
| `vmap(grad(f))`                         | compiles |
| `hessian(f)` (no vmap)                  | compiles |
| `vmap(jacrev(jacrev(f)))` (rev-over-rev)| compiles |
| `vmap(hessian(f))` = `vmap(jacfwd(jacrev(f)))` | **fails** |

So the only route to a compiled batched Hessian is **reverse-over-reverse**:
`vmap(jacrev(jacrev(f)))`. It is numerically equivalent to the library's Hessian for
well-behaved energies (verified to ~1e-15 in f64).

## 2. The correctness bug (the important finding)

Determined against an **AD-independent finite-difference ground truth** (central differences
of the analytic gradient):

| Hessian of a `torch.det` summand, batched over distinct elements | vs finite-diff |
|------------------------------------------------------------------|----------------|
| `vmap(hessian(f))` — **what `IndexedSum.sparse_hessian` uses**    | **WRONG** (off by 100–250×) |
| `vmap(jacrev(jacrev(f)))` — the compile path                      | correct (~1e-11) |
| non-vmapped `hessian(f)` per element (loop)                       | correct |

The bug is **forward-mode AD of `det` composed with `vmap`**. It is data-dependent: a batch
of *identical* elements happens to come out right, so casual spot-checks miss it; a batch of
*distinct* elements (i.e. any real mesh) is wrong. Reproduced through the public API:
`IndexedSum(neohookean_with_det, idx).sparse_hessian(V)` disagrees with finite differences by
254×. Captured as a regression test in `tests/test_compile.py`
(`test_library_forward_hessian_is_correct_for_det_under_vmap`, `xfail(strict)`).

**Whose bug is it? — PyTorch's, triggered by the library's formulation choice.** Minimal repro,
already visible at the *first* derivative of `det(A)` (3×3), batched over distinct matrices,
against the analytic Jacobian (the cofactor matrix `det(A)·inv(A)ᵀ`):

| batched 1st derivative of `det` | vs analytic |
|---|---|
| `vmap(jacrev(det))` — batched reverse-mode | ✓ 1.3e-15 |
| `vmap(jacfwd(det))` — batched **forward**-mode | ✗ **1.01 (100% wrong)** |
| `jacfwd(det)` in a loop — forward-mode, **no vmap** | ✓ 9.8e-16 |

So the broken primitive is **PyTorch functorch's batching rule for forward-mode AD (jvp) of
`torch.linalg.det`** (`vmap ∘ jacfwd`). Forward-mode alone is fine; vmap alone is fine; only
their *composition over `det`* is wrong (pytorch#149694). The library is correct code that
happens to select exactly that path: `IndexedSum.sparse_hessian` uses
`torch.func.hessian == jacfwd(jacrev(f))` and wraps it in `vmap`, i.e. `vmap ∘ jacfwd ∘ jacrev`
— the outer `vmap ∘ jacfwd` runs over the user's summand, so any summand touching `det`/
`linalg.det` (and plausibly other forward-mode-fragile linalg ops) is hit. Reverse-over-reverse
(`jacrev ∘ jacrev`) never invokes forward-mode, so it dodges the upstream bug entirely — and is
also the only formulation `torch.compile` accepts.

**Still broken on the latest PyTorch (checked 2026-08).** Issue #149694 is still *Open*. Its
original symptom (nans for the *non-vmapped* `torch.func.hessian(det)`) is gone in current
releases, but the case the library actually hits — the *batched* `vmap(hessian(det))` — is still
wrong, now *silently* (finite but incorrect, no nan). Verified against ground truth on both
torch **2.11.0** (in use here) and the latest stable **2.13.0**:

| torch | `hessian(det)` (no vmap) | `vmap(hessian(det))` (the library path) |
|---|---|---|
| 2.13.0 | correct (6.7e-16) | **wrong (0.94)**, no nan |
| 2.11.0 | correct (1.0e-15) | **wrong (0.99)**, no nan |

So upgrading PyTorch does **not** fix it.

**The chosen remedy — a vmap-safe determinant helper, not an AD-strategy change.** The blast
radius is narrow: only the determinant *operator* family is affected. Tested under
`vmap(hessian(·))` against reverse-mode ground truth:

| op inside a summand | under `vmap(hessian)` |
|---|---|
| `torch.linalg.det` | ✗ **wrong** |
| `torch.linalg.slogdet` / `logdet` | ✗ **wrong** (same family) |
| `torch.inverse` | ✓ fine |
| `@`, `trace`, elementwise | ✓ fine |

A determinant written with elementary multiply/add ops composes correctly through forward-mode
AD, so it fixes the Hessian while keeping the library's fast, well-tested forward-over-reverse
path unchanged. Shipped as `indexed_sum.det.det` / `logabsdet` (FD-verified: 9e-11 vs the
254× error of `torch.linalg.det`). Use them instead of `torch.linalg.det` / `slogdet` in
summands. This is preferred over globally switching to reverse-over-reverse, which would also
be correct but is a larger change and **regresses GPU perf** (eager reverse-mode was ~1.8×
slower than forward-mode for the neohookean Hessian on the L40). Regression tests:
`tests/test_det.py`; the `torch.linalg.det` failure is pinned with a strict `xfail`.

## 3. Is the workload even launch-bound? (whether fusion *can* help)

Comparing GPU-active time (CUDA events) to wall time on the eager kernel:

- **spring** (cheap, quadratic): time is a **flat ~2.9 ms floor** from N=1e3 to 5e5 — the GPU
  is never saturated; it's pure dispatch/launch overhead. → fusion / CUDA graphs *can* help.
- **neohookean** (`det`/`log`): GPU-active ≈ wall (149.9 ms vs 149.97 ms at N=1e5) — fully
  **compute-bound**. → nothing for fusion to reclaim.

So the ceiling on any `torch.compile` speedup is set by the per-element arithmetic intensity.

## 4. Performance sweep

Kernel-only median time (the `vmap(hessian)`-equivalent block compute), L40.
Baseline = eager forward-over-reverse (the library today). `compiled_rev[*]` = compiled
reverse-over-reverse. `reduce-overhead` = CUDA graphs.

Speedup in **bold** is vs the eager baseline (higher is better; <1× means slower). "capture
fails" = `reduce-overhead` cannot build a CUDA graph because the summand forces a host sync.
"— (cannot compile)" = the forward-over-reverse formulation itself.

| device | workload | dtype | N | eager_fwd (lib) | compiled `default` | compiled `reduce-overhead` |
|---|---|---|--:|--:|--:|--:|
| cuda | spring | float32 | 1,000 | 2.91ms | 0.13ms (**23×**) | 0.08ms (**35×**) |
| cuda | spring | float32 | 10,000 | 3.02ms | 0.09ms (**32×**) | 0.08ms (**36×**) |
| cuda | spring | float32 | 100,000 | 2.85ms | 0.10ms (**30×**) | 0.08ms (**35×**) |
| cuda | spring | float64 | 100,000 | 2.79ms | 0.10ms (**29×**) | 0.08ms (**33×**) |
| cuda | area | float32 | 1,000 | 4.96ms | 0.15ms (**32×**) | 0.10ms (**51×**) |
| cuda | area | float32 | 100,000 | 4.99ms | 0.16ms (**32×**) | 0.14ms (**36×**) |
| cuda | area | float64 | 10,000 | 4.97ms | 0.19ms (**26×**) | 0.20ms (**25×**) |
| cuda | area | float64 | 100,000 | 5.02ms | 1.18ms (**4.3×**) | 1.20ms (**4.2×**) |
| cuda | neohookean | float32 | 1,000 | 7.34ms | 2.79ms (**2.6×**) | **capture fails** |
| cuda | neohookean | float32 | 10,000 | 9.42ms | 14.71ms (**0.6×**) | **capture fails** |
| cuda | neohookean | float32 | 100,000 | 83.44ms | 138.55ms (**0.6×**) | **capture fails** |
| cuda | neohookean | float64 | 100,000 | 255.62ms | 415.97ms (**0.6×**) | **capture fails** |
| cpu | spring | float32 | 1,000 | 2.67ms | 0.07ms (**38×**) | 0.08ms (**33×**) |
| cpu | spring | float32 | 10,000 | 179.66ms | 0.12ms (**1536×**) | 0.10ms (**1744×**) |
| cpu | area | float32 | 10,000 | 530.22ms | 10.00ms (**53×**) | 10.02ms (**53×**) |

_(Abridged; full grid incl. all N/dtype in `bench/results.csv`.)_

**Reading of the sweep:**

- **spring, CUDA (f32 & f64):** eager is a launch-bound ~2.8–3.0 ms floor (flat across N);
  `reduce-overhead` collapses it to ~0.08 ms → **~35×**. Even `default` inductor (no CUDA
  graphs) fuses the tiny ops down to **~30×**. dtype barely matters — it's overhead, not FLOPs.
- **area, CUDA:** non-quadratic but still cheap + capturable → **up to 51×** at small/f32.
  Note the trend: f64 at N=100k drops to **~4×** — as real arithmetic grows it starts to
  matter and the overhead-floor win shrinks. This is the crossover from overhead- to
  compute-bound in action.
- **neohookean, CUDA:** compute-bound. `default` is **0.6× (slower)** at realistic sizes (the
  reverse-mode graph inductor produces is heavier than eager forward-mode), and
  `reduce-overhead` **cannot capture** (`det` forces a host copy). No win in any cell.
- **CPU:** eager functorch pays per-element Python dispatch (e.g. spring N=10k = 180 ms);
  inductor fuses it into a vectorized loop → enormous *apparent* speedups (1500×+), but this
  just reflects how slow eager CPU functorch is. Absolute CPU times stay far above GPU; CPU is
  not this library's target regime.

### Costs that dilute the win

- **Compile/warmup** is a one-time ~0.3–12 s (mode-dependent). Break-even vs eager is a handful
  of `sparse_hessian` calls for the big-win cases — fine for a Newton/optimization loop that
  calls it repeatedly at fixed shapes, not worth it for a one-shot Hessian.
- **End-to-end `sparse_hessian`** also builds indices + a `sparse_coo_tensor` (eager, and not
  compilable — it graph-breaks). That fixed cost dilutes the kernel speedup at the assembled-
  matrix level; the 35× is on the block-compute kernel, not the whole call.
- **Recompiles**: shapes must be fixed (or `dynamic=True`) or every new `sum_length`
  retriggers compilation.

## 5. Recommendation

1. **Correctness (done, low-risk):** use the vmap-safe `indexed_sum.det.det` / `logabsdet`
   inside summands instead of `torch.linalg.det` / `slogdet`. This fixes the `det`-energy
   Hessian while leaving the fast, well-tested forward-over-reverse `sparse_hessian` path
   unchanged — preferred over globally switching to reverse-mode (which is also correct but
   larger and ~1.8× slower on GPU for neohookean). Regression-tested in `tests/test_det.py`.
2. **Speed (optional, situational):** `torch.compile` is worth adding only as an opt-in for
   **cheap, CUDA-graph-capturable summands on GPU inside a repeated-call loop** (fixed shapes),
   where `reduce-overhead` gives ~an order of magnitude. It requires the reverse-over-reverse
   formulation (the only one that compiles), does **not** help compute-bound energies
   (neohookean), and cannot capture summands with host syncs. Keep eager as the default.

## Reproduce

```
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.x driver
pytest tests/test_det.py -v                # determinant helper: correctness + bug guard
pytest tests/test_compile.py -v            # compile viability + faithfulness
python bench/benchmark_compile.py --quick  # fast smoke sweep
python bench/benchmark_compile.py --csv bench/results.csv   # full sweep
```
