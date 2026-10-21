"""fig2_funnel.pdf -- measured attrition through the screening funnel, four arms
(manuscript Fig. 5).

Every number is read from a result file; nothing is interpolated.

Sources
  gates 1-4 (counts)     a9_counts_summary.csv, rows arm in {A, B, C, D}:
                         valid_count_mean/_std, unique_valid_count_mean/_std,
                         novel_unique_count_mean/_std (five seeds 0-4,
                         20,000 proposals per run; 'Sampled' = 20,000 by construction).
  gate 5 (rate)          a9_energetics_summary.csv, calc == 'chgnet', mode == 'full':
                         frac_le0, frac_le0_wilson_lo, frac_le0_wilson_hi, k_le0,
                         n_with_reference.
  gate 6 (rate)          a9_energetics_novel_yield.csv: k_le0_both_novel, yield_novel,
                         yield_novel_wilson_lo / _hi, n_relaxed_with_reference.
  known in C's sample    a9_energetics_summary.csv, n_in_training (arm C).

Gates 5-6 are NOT count-comparable: for every arm, 50 candidates were drawn from the
reference-eligible set (all cations in the 14-cation reference set) and relaxed.  They are
therefore drawn as rates (% of the 50 relaxed, Wilson 95 % CI), hatched, below a dashed
rule, and panel (a) carries no bars for them.
"""
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import a6_style as S

S.apply()
C = S.C
R = _CCID_ROOT + "/results"
ARMS = S.ARMS4
N = 20000

# ----------------------------------------------------------------- data -----
cs = pd.read_csv(f"{R}/a9_counts_summary.csv").set_index("arm")
pop = {k: [float(N), cs.loc[k, "valid_count_mean"], cs.loc[k, "unique_valid_count_mean"],
           cs.loc[k, "novel_unique_count_mean"]] for k in ARMS}
psd = {k: [0.0, cs.loc[k, "valid_count_std"], cs.loc[k, "unique_valid_count_std"],
           cs.loc[k, "novel_unique_count_std"]] for k in ARMS}

es = pd.read_csv(f"{R}/a9_energetics_summary.csv")
es = es[(es.calc == "chgnet") & (es["mode"] == "full")].set_index("arm")
ny = pd.read_csv(f"{R}/a9_energetics_novel_yield.csv").set_index("arm")
rate = {k: [(es.loc[k, "frac_le0"], es.loc[k, "frac_le0_wilson_lo"],
             es.loc[k, "frac_le0_wilson_hi"], int(es.loc[k, "k_le0"]),
             int(es.loc[k, "n_with_reference"])),
            (ny.loc[k, "yield_novel"], ny.loc[k, "yield_novel_wilson_lo"],
             ny.loc[k, "yield_novel_wilson_hi"], int(ny.loc[k, "k_le0_both_novel"]),
             int(ny.loc[k, "n_relaxed_with_reference"]))] for k in ARMS}
nC_known = int(es.loc["C", "n_in_training"])
n_known = {k: int(es.loc[k, "n_in_training"]) for k in ARMS}

GATES = ["Sampled",
         "Valid\n(charge-balanced, whitelist)",
         "Distinct formulas",
         "Novel\n(absent from training table)",
         "Favourable, $\\Delta E_{\\mathrm{dec}} \\leq 0$\n(CHGNet)",
         "Novel and $\\Delta E_{\\mathrm{dec}} \\leq 0$\nunder both potentials"]
POP = 4


def fmt(v, sd):
    return f"{v:,.0f}" if sd < 0.5 else f"{v:,.0f} $\\pm$ {sd:,.0f}"


def pass_rates(v):
    return [np.nan, 100 * v[1] / v[0], 100 * v[2] / v[1], 100 * v[3] / v[2]]


# --------------------------------------------------------------- figure -----
fig, (axL, axR) = plt.subplots(
    1, 2, figsize=(S.PAGE, 4.95), sharey=True,
    gridspec_kw=dict(width_ratios=[1.0, 0.52], wspace=0.07))
fig.subplots_adjust(left=0.235, right=0.985, top=0.885, bottom=0.215)

ypos = np.arange(len(GATES))[::-1].astype(float)
H = 0.19
OFF = {k: (1.5 - i) * H for i, k in enumerate(ARMS)}
TXT = {k: ("white" if k != "A" else C["ink"]) for k in ARMS}

handles = [Patch(fc=S.ARM4[k], ec=S.ARM4[k]) for k in ARMS]
axL.legend(handles, [S.ARM4_LABEL[k] for k in ARMS],
           loc="lower left", bbox_to_anchor=(-0.315, 1.045), ncol=4,
           fontsize=7.0, handlelength=1.0, handleheight=0.8, handletextpad=0.4,
           columnspacing=1.2)

# ---- (a) absolute counts, log axis, gates 1-4 only ---------------------------
for k in ARMS:
    for i in range(POP):
        y = ypos[i] + OFF[k]
        v, sd = pop[k][i], psd[k][i]
        axL.barh(y, v, height=H, left=0.9, color=S.ARM4[k], edgecolor=S.ARM4[k],
                 linewidth=0.6, zorder=3)
        axL.text(1.12, y, k, va="center", ha="left", fontsize=5.6, fontweight="bold",
                 color=TXT[k], zorder=5)
        axL.text(v * 1.22, y, fmt(v, sd), va="center", ha="left", fontsize=6.0,
                 color=S.ARM4_TEXT[k], fontweight="bold", zorder=5)

axL.set_xscale("log")
axL.set_xlim(0.9, 4.0e6)
axL.set_xticks([1, 10, 100, 1000, 10000])
axL.set_xticklabels(["1", "10", "100", "1,000", "10,000"])
axL.set_xlabel("candidates surviving the gate (count, log scale)")
axL.set_yticks(ypos)
axL.set_yticklabels(GATES, fontsize=7.0, linespacing=1.25)
axL.set_ylim(ypos.min() - 0.56, ypos.max() + 0.56)
axL.tick_params(axis="y", length=0)
S.hgrid(axL, axis="x")
axL.spines["left"].set_visible(False)
axL.text(30, (ypos[4] + ypos[5]) / 2,
         "not comparable as counts: 50 reference-eligible\n"
         "candidates were relaxed per arm; rates in (b)",
         fontsize=6.3, style="italic", color=C["ink_soft"], ha="center", va="center",
         linespacing=1.35, zorder=6,
         bbox=dict(fc="white", ec="none", pad=1.5))

XR = 1.45e6
axL.text(XR, ypos[0], "ratio\nA : D", fontsize=6.4,
         color=C["ink_soft"], ha="center", va="center", linespacing=1.2)
ratios = []
for i in (1, 2, 3):
    r = pop["A"][i] / pop["D"][i]
    ratios.append(r)
    axL.text(XR, ypos[i], f"${r:.1f}\\times$", fontsize=7.2, fontweight="bold",
             color=C["ink"], ha="center", va="center")

# ---- (b) pass rates ---------------------------------------------------------
for k in ARMS:
    pr = pass_rates(pop[k])
    for i in (1, 2, 3):
        y, f = ypos[i] + OFF[k], pr[i]
        axR.barh(y, f, height=H, color=S.ARM4[k], edgecolor=S.ARM4[k],
                 linewidth=0.6, zorder=3)
        axR.text(f + 2.5, y, f"{f:.1f}" if f < 10 else f"{f:.0f}", va="center",
                 ha="left", fontsize=5.9, color=S.ARM4_TEXT[k], zorder=5)
    for j, i in enumerate((4, 5)):
        p, lo, hi, kk, nn = rate[k][j]
        y = ypos[i] + OFF[k]
        axR.barh(y, 100 * p, height=H, color=S.ARM4[k], edgecolor=S.ARM4[k],
                 alpha=0.35, hatch="///", linewidth=0.6, zorder=3)
        axR.plot([100 * lo, 100 * hi], [y, y], color=S.ARM4[k], lw=0.8, zorder=4,
                 solid_capstyle="butt")
        axR.text(100 * hi + 2.5, y, f"{k}  {100*p:.0f}  ({kk}/{nn})", va="center",
                 ha="left", fontsize=5.6, color=S.ARM4_TEXT[k], zorder=5)

axR.text(3, ypos[0], "—", fontsize=7, color=C["ink_soft"], va="center")
axR.set_xlim(0, 128)
axR.set_xticks([0, 50, 100])
axR.set_xlabel("passing the gate (%)\nabove the rule: of the preceding gate\n"
               "below: of 50 relaxed per arm (95 % CI)", linespacing=1.3,
               fontsize=7.6)
S.hgrid(axR, axis="x")
axR.spines["left"].set_visible(False)

ysep = ypos[POP] + 0.50
for a in (axL, axR):
    a.axhline(ysep, color=C["ink_soft"], lw=0.7, ls=(0, (3, 2)), zorder=4)

fig.text(0.012, 0.012,
         "Gates 1–4: mean ± s.d. over five seeds, 20,000 proposals per run, one shared "
         "filter. Below the dashed rule is the energetic screening subset, not a "
         "population gate:\n50 candidates per arm were drawn from the reference-eligible "
         "formulas (all cations in the 14-cation set) and relaxed, so the arms are "
         "comparable there only as rates.\n"
         f"{nC_known} of arm C's 50 relaxed candidates are training-table compounds "
         f"(A {n_known['A']}, B {n_known['B']}, D {n_known['D']}); the last gate counts "
         "novel formulas (≥ 2 cations) only. Pass rates use the "
         "five-seed means.",
         fontsize=5.9, color=C["ink_soft"], ha="left", va="bottom", linespacing=1.5)

S.panel_letter(axL, "(a)", x=-0.315, y=1.10)
S.panel_letter(axR, "(b)", x=-0.10, y=1.10)

S.save(fig, "fig2_funnel")
for k in ARMS:
    print(k, "counts", [round(x, 1) for x in pop[k]], "sd", [round(x, 1) for x in psd[k]],
          "pass", [None if np.isnan(x) else round(x, 2) for x in pass_rates(pop[k])],
          "rates", [(round(r[0], 3), r[3], r[4]) for r in rate[k]],
          "in_training", int(es.loc[k, "n_in_training"]))
print("ratios A:D", [round(r, 2) for r in ratios])
