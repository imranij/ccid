"""fig7_stage1_workflow.pdf -- the corrected Stage-1 screening workflow (manuscript Fig. 2).

Drawn in the authors' Visio style (a6_visio.py) with the same snake layout as their
original method figures. Content is the revised (corrected) workflow. Counts:
  1,056 dielectric records; 757 with Eg >= 1 eV      -- manuscript Table 'tab:scoring'
  18,928 / 4,914 / 1,306 perovskite/oxide records     -- manuscript Table 'tab:scoring'
  pooled records after the harmonised gate             -- a1_corrected_pool_summary.csv
  Pareto-front size                                    -- a1_corrected_pareto.csv
  acquisition-batch size                               -- a1_corrected_acquisition_batch.csv
The four scalarisation rules are those of a1_true_scoring_rules.csv.
"""
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import a6_style as S
import a6_visio as V

R = _CCID_ROOT + "/results"
pool = pd.read_csv(f"{R}/a1_corrected_pool_summary.csv")
n_pool = int(pool["candidates"].sum())
n_batch = len(pd.read_csv(f"{R}/a1_corrected_acquisition_batch.csv"))
n_pareto = len(pd.read_csv(f"{R}/a1_corrected_pareto.csv"))

W, H = 6.6, 3.55
fig, ax = V.canvas(W, H)

GAP = 0.30
w1, w2, w3 = 1.42, 1.42, 2.88
x1 = (W - (w1 + w2 + w3 + 2 * GAP)) / 2
x2 = x1 + w1 + GAP
x3 = x2 + w2 + GAP
TY, TH = 1.86, 1.64           # top row
BY, BH = 0.06, 1.42           # bottom row
ttop = TY + TH - 0.30

# ---------------------------------------------------------------- top row
V.container(ax, x1, TY, w1, TH, "Raw Materials Databases")
V.stack(ax, x1 + 0.09, ttop, w1 - 0.18,
        ["MP dielectric set\n1,056 records", "Castelli perovskites\n18,928",
         "Wolverton oxides\n4,914", "Double perovskites\n1,306"], h=0.27, gap=0.055)

V.container(ax, x2, TY, w2, TH, "Harmonised Gates\n(Corrections 1–2)", title_pad=0.06)
V.stack(ax, x2 + 0.09, ttop - 0.11, w2 - 0.18,
        ["$E_g \\geq 1.0$ eV on\nevery population", "One element-\nexclusion list",
         f"25,148 $\\rightarrow$ {n_pool:,}\nrecords", "757 of 1,056\ndielectric records"],
        h=0.25, gap=0.05)

V.container(ax, x3, TY, w3, TH,
            "Per-Population Scalarisation", title_pad=0.06,
            sub="four rules, each normalised over its own population")
V.stack(ax, x3 + 0.10, ttop - 0.12, w3 - 0.20,
        ["Dielectric: $0.80D + 0.20N$,  "
         "$D = 0.45\\hat{\\varepsilon}_{\\mathrm{tot}} + 0.35\\hat{n} + 0.20\\hat{E}_g$",
         "Castelli: $0.45U_s + 0.35U_g + 0.20U_d$",
         "Wolverton: $0.65U_s + 0.35U_g$",
         "Double perovskites: $U_g$ only, $U_s \\equiv \\rho$ (imputed)"],
        h=0.235, gap=0.055)

V.arrow(ax, (x1 + w1, TY + TH / 2), (x2 - 0.015, TY + TH / 2))
V.arrow(ax, (x2 + w2, TY + TH / 2), (x3 - 0.015, TY + TH / 2))

# ---------------------------------------------------------------- bottom row
w4 = w3
x4 = x3
V.container(ax, x4, BY, w4, BH, "Pooled Ranking")
btop = BY + BH - 0.32
V.stack(ax, x4 + 0.10, btop, w4 - 0.20,
        [f"{n_pool:,} records after the harmonised gates",
         "Imputed stability $\\rho$ reported as a sensitivity,\nnot fixed silently (correction 3)"],
        h=0.36, gap=0.08)

x5, w5 = x1, (x4 - GAP) - x1
V.highlight(ax, x5, BY, w5, BH, "Ranked Baseline")
V.stack(ax, x5 + 0.10, btop, w5 - 0.20,
        [f"Pareto front: {n_pareto} known-stability records (Fig. 4a)",
         f"Acquisition batch: {n_batch} records, greedy on $0.65S + 0.35d_{{\\min}}$",
         "Reference chemistries for Stage 2: hafnates,\nzirconates and aluminates (Fig. 7b)"],
        h=0.27, gap=0.06, fc="white", ec=S.C["ink"])

V.arrow(ax, (x3 + w3 / 2, TY), (x4 + w4 / 2, BY + BH + 0.015))
V.arrow(ax, (x4, BY + BH / 2), (x5 + w5 + 0.015, BY + BH / 2))

S.save(fig, "fig7_stage1_workflow")
print("pool", n_pool, "batch", n_batch, "pareto", n_pareto)
