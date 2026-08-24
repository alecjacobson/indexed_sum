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
- **Speed: `torch.compile(mode="reduce-overhead")` — i.e. record the kernel launches into a
  CUDA graph once and replay them — gives ~an order of magnitude when the Hessian is
  kernel-launch-overhead-bound** (cheap per-element work), which covers spring, area, **and
  neohookean once its determinant uses the elementary `det` helper** (~20–85× on GPU). Two
  things must hold: reverse-over-reverse (the only formulation that compiles), and no host
  syncs in the summand. `torch.linalg.det` violates both — it's the reason the naive
  neohookean can't be captured (and its eager Hessian is wrong too). So the `det` helper is
  what makes the elasticity case both **correct and compilable**.

Environment: torch 2.11.0+cu128, NVIDIA L40, driver 570.158 (CUDA 12.8).

---

> **Two *separate* problems, both caused by the forward-mode (`jacfwd`) outer of
> `hessian = jacfwd(jacrev)`:**
> 1. **Correctness (§2):** forward-mode AD of `torch.linalg.det` under `vmap` returns *wrong
>    values*. Hits the library's eager path. Fixed by the elementary `det` helper — *without*
>    leaving forward mode. Affects `det`/`slogdet` only.
> 2. **Compilability (§1):** `jacfwd` won't *trace* under dynamo at all (any summand). Forces
>    the compile path onto reverse-over-reverse. Independent of `det`.
>
> They're related (same culprit) but distinct: the `det` helper resolves #1; #2 is why the
> compile path uses `jacrev(jacrev)`. Reverse-mode happens to sidestep #1 for free.

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
summands. This is preferred over globally switching `sparse_hessian` to reverse-over-reverse:
the helper is a **surgical, opt-in** change that leaves the default AD path (and every existing
energy's results) untouched, versus swapping the core AD strategy for all summands. (Perf is
not the deciding factor — eager reverse-mode is actually a bit *faster* here for most workloads;
it's the blast radius that argues for the helper.) Regression tests: `tests/test_det.py`; the
`torch.linalg.det` failure is pinned with a strict `xfail`.

## 3. Is the workload even launch-bound? (whether fusion *can* help)

Comparing GPU-active time (CUDA events) to wall time on the eager kernel:

- **spring** (cheap, quadratic): time is a **flat ~2.9 ms floor** from N=1e3 to 5e5 — the GPU
  is never saturated; it's pure dispatch/launch overhead. → fusion / CUDA graphs *can* help.
- **neohookean with `torch.linalg.det`** (`det`/`log`): GPU-active ≈ wall (149.9 ms vs 149.97 ms
  at N=1e5) — fully **compute-bound** (LAPACK `det` kernels dominate). → nothing for fusion.
- **neohookean with the elementary `det` helper**: the determinant becomes a handful of
  multiply/adds, so the summand is cheap and the Hessian is **launch-overhead-bound again**
  (eager time is flat ~9 ms across N, like spring/area). → fusion / CUDA graphs *can* help, a
  lot. (The helper is also ~9× faster than `torch.linalg.det` in plain eager at N=1e5, because
  LAPACK is poor on tiny 3×3 blocks.)

So the ceiling on any `torch.compile` speedup is set by per-element arithmetic intensity — and
using elementary ops for the determinant *lowers* that intensity into the regime where compile
pays off.

## 4. Performance sweep

Kernel-only median time (the `vmap(hessian)`-equivalent block compute), L40. Baseline =
eager forward-over-reverse (the library today). `compiled_rev[*]` = compiled reverse-over-reverse.
Speedup in **bold** is vs the eager baseline (higher is better; <1× = slower). "capture fails" =
`reduce-overhead` cannot build a CUDA graph because the summand forces a host sync. The two
`neohookean*` rows differ *only* in how the determinant is computed.

| device | workload | dtype | N | eager fwd (lib) | eager rev | compiled `default` | compiled `reduce-overhead` (CUDA graphs) |
|---|---|---|--:|--:|--:|--:|--:|
| cuda | spring | f32 | 10,000 | 2.81ms | 1.44 (**2.0×**) | 0.09 (**30×**) | 0.08 (**34×**) |
| cuda | spring | f64 | 100,000 | 2.77ms | 1.42 (**1.9×**) | 0.09 (**29×**) | 0.08 (**33×**) |
| cuda | area | f32 | 10,000 | 4.90ms | 4.22 (**1.2×**) | 0.15 (**32×**) | 0.10 (**50×**) |
| cuda | area | f64 | 100,000 | 4.93ms | 4.28 (**1.2×**) | 1.17 (**4.2×**) | 1.20 (**4.1×**) |
| cuda | **neohookean** (`det` helper) | f32 | 1,000 | 8.56ms | 6.58 (**1.3×**) | 0.17 (**52×**) | 0.10 (**86×**) |
| cuda | **neohookean** (`det` helper) | f32 | 10,000 | 8.43ms | 6.51 (**1.3×**) | 0.16 (**51×**) | 0.10 (**89×**) |
| cuda | **neohookean** (`det` helper) | f32 | 100,000 | 8.45ms | 6.69 (**1.3×**) | 0.46 (**18×**) | 0.47 (**18×**) |
| cuda | **neohookean** (`det` helper) | f64 | 10,000 | 8.42ms | 6.63 (**1.3×**) | 0.17 (**49×**) | 0.13 (**65×**) |
| cuda | neohookean_torchdet | f32 | 10,000 | 9.42ms | 17.23 (**0.5×**) | 14.68 (**0.6×**) | **capture fails** |
| cuda | neohookean_torchdet | f32 | 100,000 | 83.38ms | 147.0 (**0.6×**) | 138.3 (**0.6×**) | **capture fails** |
| cpu | spring | f32 | 10,000 | 159.78ms | 69.9 (**2.3×**) | 0.10 (**1567×**) | 0.10 (**1664×**) |
| cpu | neohookean (`det` helper) | f32 | 10,000 | 739.94ms | 650.0 (**1.1×**) | 19.95 (**37×**) | 10.13 (**73×**) |

_(Abridged; full grid — all N/dtype/device, both neohookean variants — in `bench/results.csv`.)_

**Reading of the sweep:**

- **spring / area, CUDA:** eager is a launch-overhead floor (flat ~2.8 / ~4.9 ms across N);
  `reduce-overhead` collapses it to ~0.08–0.10 ms → **~34× / ~50×**. `default` inductor (no CUDA
  graphs) also fuses well (~30×). For area, f64 at N=100k falls to **~4×** — as real arithmetic
  grows, the overhead-floor win shrinks (the overhead- → compute-bound crossover).
- **neohookean with the `det` helper, CUDA — the headline:** now overhead-bound (eager flat
  ~8.5 ms), so `reduce-overhead` gives **~86–89× (f32)** at N≤10k, tapering to **~18×** at
  N=100k f32 and **~13×** at N=100k f64 as real FLOPs finally dominate. Correct in every cell
  (`ok`; the N=100k `ok=False` flags are the fwd-vs-reverse rounding gap on the odd degenerate
  random tet, not a compile error — see `tests/test_det.py` for the controlled check).
- **neohookean_torchdet, CUDA — the contrast:** `reduce-overhead` **cannot capture** (`det`'s
  host sync), `default` is **0.6× (slower)**, and note eager is itself ~10× slower (83 ms vs
  8.5 ms at N=100k) because LAPACK `det` is costly on tiny blocks. This row is what the `det`
  helper replaces.
- **CPU:** eager functorch pays per-element Python dispatch; inductor fuses it into a vectorized
  loop for large *apparent* speedups (neohookean **~73×**, spring 1500×+), but absolute CPU
  times stay far above GPU — CPU isn't this library's target regime.

### Costs that dilute the win

- **Compile/warmup** is a one-time ~0.3–12 s (mode-dependent). Break-even vs eager is a handful
  of `sparse_hessian` calls for the big-win cases — fine for a Newton/optimization loop that
  calls it repeatedly at fixed shapes, not worth it for a one-shot Hessian.
- **End-to-end `sparse_hessian`** also builds indices + a `sparse_coo_tensor` (eager, and not
  compilable — it graph-breaks). That fixed cost dilutes the kernel speedup at the assembled-
  matrix level; the reported speedups are on the per-element block-compute kernel (the part
  `torch.compile` covers), not the whole `sparse_hessian` call.
- **Recompiles**: shapes must be fixed (or `dynamic=True`) or every new `sum_length`
  retriggers compilation.

## 5. Recommendation

1. **Correctness (done, low-risk):** use the vmap-safe `indexed_sum.det.det` / `logabsdet`
   inside summands instead of `torch.linalg.det` / `slogdet`. This fixes the `det`-energy
   Hessian while leaving the fast, well-tested forward-over-reverse `sparse_hessian` path
   unchanged — a surgical fix, versus globally swapping the core AD strategy to reverse-mode
   (also correct, but a far larger blast radius). Regression-tested in `tests/test_det.py`.
2. **Speed (optional, situational):** `torch.compile` is worth adding as an opt-in for
   **cheap, CUDA-graph-capturable summands on GPU inside a repeated-call loop** (fixed shapes),
   where `mode="reduce-overhead"` (CUDA graphs) gives ~an order of magnitude — spring/area **and
   neohookean, once it uses the elementary `det` helper** (~20–88× on the L40). It requires the
   reverse-over-reverse formulation (the only one that compiles) and a summand free of host
   syncs — which is exactly why `torch.linalg.det` must be replaced by the helper (it both
   corrupts the eager Hessian *and* blocks CUDA-graph capture). Keep eager as the default; the
   compile path is not yet wired into `IndexedSum` (it lives in `bench/` as the study).

## Reproduce

```
pip install torch --index-url https://download.pytorch.org/whl/cu128   # CUDA 12.x driver
pytest tests/test_det.py -v                # determinant helper: correctness + bug guard
pytest tests/test_compile.py -v            # compile viability + faithfulness
python bench/benchmark_compile.py --quick  # fast smoke sweep
python bench/benchmark_compile.py --csv bench/results.csv   # full sweep
```
