#!/bin/sh
# Rebuild every a6 figure. Outputs go to results/figures/.
set -e
cd "$(dirname "$0")"
for f in a6_fig1_concept.py a6_fig2_funnel.py a6_fig3_freebits.py \
         a6_fig4_energetics.py a6_fig5_chemistry.py a6_fig6_stage1.py \
         a6_fig7_stage1_workflow.py a6_fig8_stage2_pipeline.py; do
    echo "== $f"
    python3 "$f"
done
