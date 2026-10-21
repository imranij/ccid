#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
python a4_run_screen.py --calc chgnet --refs
python a4_run_screen.py --calc chgnet --tier ALL
echo "CHGNET_ALL_DONE"
