#!/bin/bash
# a8 sweep: full factorial beta_max x per-dimension free-bits budget, plus the
# buggy-loss CPU control. 2 jobs in parallel, 1 torch thread each (2-core box).
cd analysis/a8_retrain
mkdir -p logs checkpoints
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
run() { python a8_train.py "$@" > "logs/$2.log" 2>&1; }

JOBS=()
for B in 0.01 0.1 0.5 1.0; do
  for L in 0.0 0.05 0.25 0.5; do
    JOBS+=("--tag fb${L}_b${B} --loss fixed --free-bits-per-dim $L --beta-max $B")
  done
done
JOBS+=("--tag buggy_b0.1 --loss buggy --beta-max 0.1")

i=0
for J in "${JOBS[@]}"; do
  set -- $J
  run "$@" &
  i=$((i+1))
  if (( i % 2 == 0 )); then wait; fi
done
wait
echo "SWEEP DONE"
