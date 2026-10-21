#!/bin/bash
# a8 sweep, stage 2: the stage-1 grid was monotone in the free-bits budget up to
# lambda = 0.5, so extend to larger per-dimension budgets to find the turning point.
cd analysis/a8_retrain
mkdir -p logs checkpoints
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=()
for B in 0.01 0.1 1.0; do
  for L in 1.0 2.0 4.0; do
    JOBS+=("--tag fb${L}_b${B} --loss fixed --free-bits-per-dim $L --beta-max $B")
  done
done
JOBS+=("--tag fb8.0_b0.1 --loss fixed --free-bits-per-dim 8.0 --beta-max 0.1")
i=0
for J in "${JOBS[@]}"; do
  set -- $J
  python a8_train.py "$@" > "logs/$2.log" 2>&1 &
  i=$((i+1)); if (( i % 2 == 0 )); then wait; fi
done
wait
echo "SWEEP2 DONE"
