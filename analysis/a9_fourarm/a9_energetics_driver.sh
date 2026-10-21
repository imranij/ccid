#!/usr/bin/env bash
# Sequential driver: one relaxation job at a time (2-core machine shared with the
# generation agent).  Arm B first; then wait for the generation agent's READY flag
# (poll every 5 min, no busy-wait); then arms C and D.
set -u
HERE="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$HERE/../.." && pwd)"
R="$ROOT/results"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
cd "$HERE"
python a9_energetics_run.py --arm B --calc chgnet > a9_energetics_B_chgnet.log 2>&1
python a9_energetics_run.py --arm B --calc mace   > a9_energetics_B_mace.log 2>&1
echo "$(date -Is) arm B done; waiting for $R/a9_umlip_sample_READY.flag" > a9_energetics_wait.log
while [ ! -f "$R/a9_umlip_sample_READY.flag" ]; do sleep 300; echo "$(date -Is) waiting" >> a9_energetics_wait.log; done
echo "$(date -Is) flag found" >> a9_energetics_wait.log
python a9_energetics_sample_CD.py > a9_energetics_sample_CD.log 2>&1
for ARM in C D; do
  python a9_energetics_run.py --arm $ARM --calc chgnet > a9_energetics_${ARM}_chgnet.log 2>&1
  python a9_energetics_run.py --arm $ARM --calc mace   > a9_energetics_${ARM}_mace.log 2>&1
done
echo "$(date -Is) ALL DONE" >> a9_energetics_wait.log
