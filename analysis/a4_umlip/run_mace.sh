#!/bin/bash
cd "$(dirname "$0")"
export OMP_NUM_THREADS=2
python a4_mace_refine.py
echo "MACE_DONE"
