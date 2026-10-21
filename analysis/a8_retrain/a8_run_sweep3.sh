#!/bin/bash
# a8 sweep, stage 3: refine around the peak in unique valid formulas (lambda ~ 1.0,
# beta_max small) found in stage 2.
cd analysis/a8_retrain
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
JOBS=("--tag fb0.75_b0.01 --loss fixed --free-bits-per-dim 0.75 --beta-max 0.01"
      "--tag fb1.5_b0.01 --loss fixed --free-bits-per-dim 1.5 --beta-max 0.01"
      "--tag fb1.0_b0.001 --loss fixed --free-bits-per-dim 1.0 --beta-max 0.001"
      "--tag fb1.25_b0.01 --loss fixed --free-bits-per-dim 1.25 --beta-max 0.01")
i=0
for J in "${JOBS[@]}"; do set -- $J; python a8_train.py "$@" > "logs/$2.log" 2>&1 &
  i=$((i+1)); if (( i % 2 == 0 )); then wait; fi; done
wait; echo "SWEEP3 DONE"
