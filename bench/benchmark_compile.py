"""
Benchmark: does torch.compile (kernel fusion / CUDA graphs) speed up IndexedSum's
sparse-Hessian construction, and when?

The hot path is `vmap(hessian(local_summand))` (indexed_sum/indexed_sum.py). This script
A/B-compares the eager baseline against several torch.compile variants across:
  workload (per-element cost) x problem size x dtype x device x compile mode.

Key facts this harness encodes (established empirically, see bench/RESULTS.md):
  * torch.func.hessian == jacfwd(jacrev(f)) (forward-over-reverse) does NOT compile under
    dynamo in torch 2.11 (`_fw_primal` inference-mode assert). To compile at all we must
    reformulate the Hessian as jacrev(jacrev(f)) (reverse-over-reverse), which is
    numerically identical but ~2x slower in eager.
  * mode="reduce-overhead" (CUDA graphs) can only capture summands with no CPU<->CUDA
    syncs; e.g. torch.linalg.det triggers a host copy and capture fails.

Timing uses CUDA events on GPU (device-time) and perf_counter on CPU, median over repeats,
after warmup that also pays the one-time compile cost (reported separately).

Usage:
  python bench/benchmark_compile.py --quick          # fast smoke sweep
  python bench/benchmark_compile.py --csv out.csv    # full sweep -> CSV
"""
import argparse
import csv
import statistics
import sys
import time

import torch
from torch.func import vmap, hessian, jacrev


# --------------------------------------------------------------------------------------
# Workloads: local_summand, local_size, feature dim, and a capturability/cost note.
# --------------------------------------------------------------------------------------
def spring(v):
    # Quadratic (constant Hessian), cheapest possible, no CPU-sync ops -> CUDA-graph OK.
    d = v[1] - v[0]
    return 0.5 * (d * d).sum()


def area(v):
    # Heron's formula: non-quadratic, uses sqrt (GPU-only, no host copy) -> CUDA-graph OK.
    a = torch.linalg.norm(v[1] - v[0])
    b = torch.linalg.norm(v[2] - v[1])
    c = torch.linalg.norm(v[2] - v[0])
    s = (a + b + c) / 2
    return torch.sqrt(torch.clamp(s * (s - a) * (s - b) * (s - c), min=1e-12))


def neohookean(v):
    # Uses det + log: heavy per-element compute AND det forces a host copy -> NOT
    # CUDA-graph capturable; representative of "expensive" elasticity summands.
    F = v[1:] - v[0:1]  # (3,3)
    J = torch.linalg.det(F)
    I1 = (F * F).sum()
    return I1 - 3 - 2 * torch.log(torch.clamp(J, min=1e-3)) + (J - 1) ** 2


WORKLOADS = {
    #  name         fn          local_size  dim  note
    "spring": (spring, 2, 2, "cheap/quadratic, capturable"),
    "area": (area, 3, 3, "cheap/nonquadratic, capturable"),
    "neohookean": (neohookean, 4, 3, "expensive (det/log), NOT capturable"),
}


def make_reshaped(fn, local_size, dim):
    def reshaped(inp):
        return fn(inp.view(local_size, dim))
    return reshaped


def eager_fwd_kernel(reshaped):
    # The library's actual formulation: forward-over-reverse.
    def k(sel):
        return vmap(hessian(reshaped))(sel)
    return k


def rev_kernel(reshaped):
    # Reverse-over-reverse: numerically identical, and the only thing that compiles.
    def k(sel):
        return vmap(jacrev(jacrev(reshaped)))(sel)
    return k


# --------------------------------------------------------------------------------------
# Timing helpers
# --------------------------------------------------------------------------------------
def _sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def time_call(fn, arg, device, iters, repeats=5):
    """Median per-call time in ms over `repeats` measurements of `iters` calls each."""
    # warmup (also triggers compilation)
    for _ in range(3):
        fn(arg)
    _sync(device)
    samples = []
    if device == "cuda":
        for _ in range(repeats):
            s = torch.cuda.Event(enable_timing=True)
            e = torch.cuda.Event(enable_timing=True)
            s.record()
            for _ in range(iters):
                fn(arg)
            e.record()
            torch.cuda.synchronize()
            samples.append(s.elapsed_time(e) / iters)
    else:
        for _ in range(repeats):
            t = time.perf_counter()
            for _ in range(iters):
                fn(arg)
            samples.append((time.perf_counter() - t) / iters * 1e3)
    return statistics.median(samples)


def measure_compile_time(fn, arg, device):
    """Wall time of the first call (compilation + first execution)."""
    _sync(device)
    t = time.perf_counter()
    fn(arg)
    _sync(device)
    return time.perf_counter() - t


# --------------------------------------------------------------------------------------
# Sweep
# --------------------------------------------------------------------------------------
COMPILE_MODES = ["default", "reduce-overhead", "max-autotune"]


def run_config(wl_name, N, dtype, device, iters, modes, do_maxautotune):
    fn, local_size, dim, _note = WORKLOADS[wl_name]
    reshaped = make_reshaped(fn, local_size, dim)

    idx = torch.randint(0, N, (N, local_size), device=device)
    V = torch.randn(N, dim, device=device, dtype=dtype)
    sel = V[idx]

    rows = []

    # Baseline: eager forward-over-reverse (what the library does today).
    kf = eager_fwd_kernel(reshaped)
    ref = kf(sel).clone()
    base_ms = time_call(kf, sel, device, iters)
    rows.append(dict(workload=wl_name, N=N, dtype=str(dtype).split(".")[-1], device=device,
                     variant="eager_fwd(lib)", ms=base_ms, speedup=1.0,
                     compile_s=0.0, maxdiff=0.0, ok=True, note="baseline"))

    # Eager reverse-over-reverse (the formulation compile is forced to use).
    kr = rev_kernel(reshaped)
    try:
        r = kr(sel)
        md = reldiff(ref, r)
        er_ms = time_call(kr, sel, device, iters)
        rows.append(dict(workload=wl_name, N=N, dtype=str(dtype).split(".")[-1], device=device,
                         variant="eager_rev", ms=er_ms, speedup=base_ms / er_ms,
                         compile_s=0.0, maxdiff=md, ok=True, note=""))
    except Exception as ex:  # pragma: no cover - defensive
        rows.append(dict(workload=wl_name, N=N, dtype=str(dtype).split(".")[-1], device=device,
                         variant="eager_rev", ms=float("nan"), speedup=float("nan"),
                         compile_s=0.0, maxdiff=float("nan"), ok=False, note=repr(ex)[:80]))

    # Document that compiling the library's fwd-over-rev formulation fails.
    try:
        cfwd = torch.compile(kf, fullgraph=False)
        cfwd(sel)
        note = "compiled unexpectedly"
        ok = True
    except Exception as ex:
        note = "fwd-over-rev does NOT compile: " + repr(ex).split("'")[1][:40] if "'" in repr(ex) else "no compile"
        ok = False
    rows.append(dict(workload=wl_name, N=N, dtype=str(dtype).split(".")[-1], device=device,
                     variant="compiled_fwd", ms=float("nan"), speedup=float("nan"),
                     compile_s=float("nan"), maxdiff=float("nan"), ok=ok, note=note))
    torch._dynamo.reset()

    # Compiled reverse-over-reverse across modes.
    for mode in modes:
        if mode == "max-autotune" and not do_maxautotune:
            continue
        variant = f"compiled_rev[{mode}]"
        try:
            ck = torch.compile(kr, mode=mode, fullgraph=False)
            comp_s = measure_compile_time(ck, sel, device)
            outs = [ck(sel).clone() for _ in range(3)]  # clone: CUDA graphs reuse buffers
            _sync(device)
            md = reldiff(ref, outs[-1])
            ms = time_call(ck, sel, device, iters)
            rows.append(dict(workload=wl_name, N=N, dtype=str(dtype).split(".")[-1], device=device,
                             variant=variant, ms=ms, speedup=base_ms / ms,
                             compile_s=comp_s, maxdiff=md, ok=(md < tol(dtype)), note=""))
        except Exception as ex:
            msg = repr(ex)
            short = "CUDA-graph capture fails (CPU sync)" if "CUDA graph" in msg or "pin_memory" in msg else msg[:70]
            rows.append(dict(workload=wl_name, N=N, dtype=str(dtype).split(".")[-1], device=device,
                             variant=variant, ms=float("nan"), speedup=float("nan"),
                             compile_s=float("nan"), maxdiff=float("nan"), ok=False, note=short))
        torch._dynamo.reset()

    return rows


def reldiff(ref, out):
    """Max relative difference: ||ref-out||_inf / ||ref||_inf. Robust to Hessian magnitude
    and to the fwd-vs-reverse-mode rounding gap in low precision."""
    denom = ref.abs().max().item()
    return (ref - out).abs().max().item() / (denom + 1e-30)


def tol(dtype):
    # Relative tolerance. fwd-over-rev vs rev-over-rev differ at ~1e-5 (f32) rounding.
    return 1e-4 if dtype == torch.float32 else 1e-10


def fmt_row(r):
    def f(x, p="8.3f"):
        return "   nan  " if x != x else format(x, p)
    return (f"{r['workload']:<11} N={r['N']:>7} {r['dtype']:>7} {r['device']:>4} "
            f"{r['variant']:<22} {f(r['ms'])}ms  {f(r['speedup'],'6.2f')}x  "
            f"comp={f(r['compile_s'],'6.1f')}s  ok={str(r['ok']):>5}  {r['note']}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true", help="fast smoke sweep")
    ap.add_argument("--csv", default=None, help="write results to CSV path")
    ap.add_argument("--cpu-only", action="store_true")
    ap.add_argument("--maxautotune", action="store_true",
                    help="also benchmark mode='max-autotune' (slow to compile)")
    args = ap.parse_args()

    devices = ["cpu"]
    if torch.cuda.is_available() and not args.cpu_only:
        devices = ["cuda", "cpu"]

    if args.quick:
        workloads = ["spring", "neohookean"]
        sizes = [10_000]
        dtypes = [torch.float32]
        iters = 20
        modes = ["default", "reduce-overhead"]
        do_maxauto = False
    else:
        workloads = ["spring", "area", "neohookean"]
        sizes = [1_000, 10_000, 100_000]
        dtypes = [torch.float32, torch.float64]
        iters = 30
        modes = COMPILE_MODES if args.maxautotune else ["default", "reduce-overhead"]
        do_maxauto = args.maxautotune

    print(f"device(s)={devices} torch={torch.__version__} "
          f"gpu={torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'n/a'}\n")

    all_rows = []
    for device in devices:
        # CPU is slow for big N and reverse-mode; cap sizes there.
        dev_sizes = sizes if device == "cuda" else [n for n in sizes if n <= 10_000]
        dev_iters = iters if device == "cuda" else max(5, iters // 3)
        for wl in workloads:
            for dtype in dtypes:
                for N in dev_sizes:
                    rows = run_config(wl, N, dtype, device, dev_iters, modes, do_maxauto)
                    for r in rows:
                        print(fmt_row(r), flush=True)
                    all_rows.extend(rows)
                    print(flush=True)

    if args.csv:
        with open(args.csv, "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"wrote {len(all_rows)} rows -> {args.csv}")


if __name__ == "__main__":
    main()
