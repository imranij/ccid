#!/usr/bin/env bash
# Retrain the a5 surrogates with the UNMODIFIED a5 scripts (the .pkl files are not in the
# repo).  Run single-core (taskset + LOKY_MAX_CPU_COUNT=1, so the scripts' n_jobs=-1 random
# forests use one core; forests are seeded, so n_jobs does not change the fitted models).
# The a5 scripts also rewrite their own metric/figure files under $CCID_ROOT/results; to
# avoid touching the committed a5 results they are run against a shadow root whose
# results/ directory is separate (SHADOW/results), with every other top-level directory
# symlinked to the repository.  The .pkl files land in analysis/a5_surrogate/ as usual.
# Afterwards a9_check_a5_retrain.py compares the shadow metrics with results/a5_*.json.
set -eu
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
SHADOW="${A9_A5_SHADOW:-${TMPDIR:-/tmp}/a9_a5_shadow_root}"   # outside the repo (its symlinks would loop)
mkdir -p "$SHADOW/results"
for d in dgx shared_datasets analysis Paper_1_Uncertainty_Calibrated_Data_Repair_for_Ceramic_Materials; do
  [ -e "$SHADOW/$d" ] || ln -s "$ROOT/$d" "$SHADOW/$d"
done
export CCID_ROOT="$SHADOW" OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 LOKY_MAX_CPU_COUNT=1
cd "$ROOT/analysis/a5_surrogate"
for s in a5_featurize_training.py a5_train_stability_gap.py a5_train_dielectric.py; do
  echo "$(date -Is) start $s"
  taskset -c 1 python "$s"
  echo "$(date -Is) done $s"
done
