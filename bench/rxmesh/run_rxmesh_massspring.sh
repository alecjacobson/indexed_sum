#!/usr/bin/env bash
# Reproduce the RXMesh side of the mass-spring "Diff" (grad+Hessian assembly) timing.
#
# Requires RXMesh built with the MassSpringDiff harness (apps/MassSpring/mass_spring_diff.cu)
# and the assembled CUDA 12.6 toolkit on LD_LIBRARY_PATH. Edit RXMESH_DIR / CUDA_ROOT for
# your setup. Prints Diff/eval (ms) per grid size (vertices = n^2), matching the quantity
# RXMesh's MassSpring app reports as the "Diff" timer and the paper's Table 1 reports.
set -euo pipefail
RXMESH_DIR="${RXMESH_DIR:-/home/horde/projects/RXMesh}"
CUDA_ROOT="${CUDA_ROOT:-/home/horde/projects/cuda126}"
export LD_LIBRARY_PATH="$CUDA_ROOT/lib64:${LD_LIBRARY_PATH:-}"
BIN="$RXMESH_DIR/build/bin/MassSpringDiff"
OUT="${1:-$(dirname "$0")/mass_spring_rxmesh.csv}"

echo "n,nV,nE,nF,diff_per_eval_ms" > "$OUT"
for n in 10 100 500 1000; do
  line=$("$BIN" -n "$n" -e 200 -w 30 2>&1 | grep DiffBench | tail -1)
  nV=$(echo "$line"  | sed -n 's/.*#V= \([0-9]*\).*/\1/p')
  nE=$(echo "$line"  | sed -n 's/.*#E= \([0-9]*\).*/\1/p')
  nF=$(echo "$line"  | sed -n 's/.*#F= \([0-9]*\).*/\1/p')
  ms=$(echo "$line"  | sed -n 's/.*Diff\/eval= \([0-9.]*\).*/\1/p')
  echo "$n,$nV,$nE,$nF,$ms" | tee -a "$OUT"
done
echo "wrote -> $OUT"
