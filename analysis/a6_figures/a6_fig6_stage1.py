"""fig6_stage1.pdf -- the corrected Stage-1 landscape.

Panel (a): stability utility U_s against band-gap utility U_g for the 200 highest-scoring
           records of the corrected pool, with the known-stability Pareto front marked and
           the leading hafnates, zirconates and aluminates labelled.  Records whose
           stability utility is imputed (the double-perovskite source, which carries no
           formation-energy field) are drawn as open symbols at U_s = rho = 0.5, because
           their vertical position is an assumption and not a measurement.
Panel (b): the sensitivity of the head of the ranking to that imputation.

Sources
  (a) a1_corrected_top200.csv
        cols stability_utility, gap_utility, candidate_score, source_dataset,
             stability_known, formula
      a1_corrected_pareto.csv
        cols stability_utility, gap_utility, formula  (7 known-stability records)
  (b) a1_imputation_sensitivity.csv
        cols rho, rank_CaSiYAlO6, double_perovskites_in_top20, top_formula
"""
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import a6_style as S

S.apply()
C = S.C
R = _CCID_ROOT + "/results"

T = pd.read_csv(f"{R}/a1_corrected_top200.csv")
PF = pd.read_csv(f"{R}/a1_corrected_pareto.csv").sort_values("stability_utility")
IM = pd.read_csv(f"{R}/a1_imputation_sensitivity.csv")

SRC = {"castelli_perovskites": ("Castelli perovskites ($e_{\\mathrm{form}}$)", C["cvae"], "o"),
       "wolverton_oxides": ("Wolverton oxides ($e_{\\mathrm{hull}}$)", C["acc2"], "s"),
       "double_perovskites_gap": ("Double perovskites ($U_s$ imputed)", C["rand"], "^")}

fig, (ax, ax2) = plt.subplots(2, 1, figsize=(S.COL, 4.60),
                              gridspec_kw=dict(height_ratios=[3.05, 1.6], hspace=0.50))
fig.subplots_adjust(left=0.165, right=0.965, top=0.865, bottom=0.105)

# ================================================================ panel a ===
for src, (lab, col, mk) in SRC.items():
    d = T[T.source_dataset == src]
    known = d.stability_known == 1
    ax.scatter(d.loc[known, "gap_utility"], d.loc[known, "stability_utility"],
               s=11, marker=mk, facecolor=col, edgecolor="white", linewidth=0.35,
               alpha=0.85, zorder=4)
    ax.scatter(d.loc[~known, "gap_utility"], d.loc[~known, "stability_utility"],
               s=11, marker=mk, facecolor="none", edgecolor=col, linewidth=0.55,
               alpha=0.60, zorder=3)

ax.plot(PF.gap_utility, PF.stability_utility, color=C["acc1"], lw=1.1,
        ls=(0, (4, 2)), zorder=6)
ax.scatter(PF.gap_utility, PF.stability_utility, s=26, marker="o",
           facecolor="none", edgecolor=C["acc1"], linewidth=1.0, zorder=7)

LABELS = {"BaZrO3": (-36, 9), "BaHfO3": (-7, 14), "SrHfO3": (13, 7),
          "CaHfO3": (14, -5), "SrZrO3": (-74, -48), "LaAlO3": (42, -36),
          "KTaO3": (9, 3), "CaSiYAlO6": (-18, 13)}
SUB = str.maketrans("0123456789", "₀₁₂₃₄₅₆₇₈₉")
for f, (dx, dy) in LABELS.items():
    r = T[T.formula == f]
    if r.empty:
        continue
    r = r.iloc[0]
    ax.annotate(f.translate(SUB), xy=(r.gap_utility, r.stability_utility),
                xytext=(dx, dy), textcoords="offset points", fontsize=6.0,
                color=C["ink"], fontweight="bold", zorder=8,
                arrowprops=dict(arrowstyle="-", lw=0.45, color=C["ink_soft"],
                                shrinkA=0, shrinkB=1.5))

ax.axhline(0.5, color=C["rand"], lw=0.6, ls=(0, (1.6, 1.6)), zorder=1)
ax.text(0.015, 0.512, "$\\rho$ = 0.5, the imputed stability utility\n"
        "(double perovskites carry no formation energy)",
        fontsize=5.7, color=C["rand_dk"], ha="left", va="bottom", linespacing=1.35)

ax.set_xlim(0, 1.04)
ax.set_ylim(0.42, 1.09)
ax.set_xlabel("band-gap utility $U_g$ (dimensionless, min–max over the pool)")
ax.set_ylabel("stability utility $U_s$\n(dimensionless)", linespacing=1.3)
S.hgrid(ax, axis="both")

handles = [Line2D([], [], marker=mk, ls="none", ms=3.6, color=col,
                  markeredgecolor="white", markeredgewidth=0.35)
           for _, (lab, col, mk) in SRC.items()]
handles.append(Line2D([], [], color=C["acc1"], lw=1.1, ls=(0, (4, 2)),
                      marker="o", ms=4.2, markerfacecolor="none"))
ax.legend(handles, [lab for _, (lab, _, _) in SRC.items()]
          + ["Pareto front (known stability)"],
          loc="lower left", bbox_to_anchor=(-0.185, 1.015), ncol=2,
          fontsize=6.0, handletextpad=0.35, columnspacing=0.9, labelspacing=0.22)

# ================================================================ panel b ===
ax2.plot(IM.rho, IM.rank_CaSiYAlO6, marker="o", ms=3.4, color=C["rand"],
         zorder=5, clip_on=False)
ax2.set_yscale("log")
ax2.set_ylim(0.8, 400)
ax2.set_yticks([1, 10, 100])
ax2.set_yticklabels(["1", "10", "100"])
ax2.set_xlim(-0.03, 1.03)
ax2.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
ax2.set_xlabel("imputed stability utility $\\rho$ for double-perovskite records")
ax2.set_ylabel("rank of \\ce{CaSiYAlO6}".replace("\\ce{CaSiYAlO6}", "CaSiYAlO$_6$")
               + "\n(log scale)", color=C["rand_dk"], linespacing=1.3)
ax2.tick_params(axis="y", colors=C["rand_dk"])
ax2.spines["left"].set_color(C["rand_dk"])
S.hgrid(ax2, axis="y")

ax3 = ax2.twinx()
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(True)
ax3.spines["right"].set_color(C["ink_soft"])
ax3.bar(IM.rho, IM.double_perovskites_in_top20, width=0.07, color=C["pale"],
        edgecolor=C["ink_soft"], linewidth=0.5, zorder=2)
ax3.set_ylim(0, 22)
ax3.set_yticks([0, 10, 20])
ax3.set_ylabel("double perovskites\nin the top 20", color=C["ink_soft"],
               linespacing=1.3)
ax3.tick_params(axis="y", colors=C["ink_soft"])
ax2.set_zorder(ax3.get_zorder() + 1)
ax2.patch.set_visible(False)

for _, r in IM.iterrows():
    ax2.annotate(f"{int(r.rank_CaSiYAlO6)}", xy=(r.rho, r.rank_CaSiYAlO6),
                 xytext=(0, 6), textcoords="offset points", fontsize=5.9,
                 color=C["rand_dk"], ha="center", va="bottom", zorder=8)
ax2.text(0.03, 3.2, "top-ranked:\nBaHfO$_3$", fontsize=5.9, color=C["ink"],
         ha="left", va="top", linespacing=1.3)
ax2.text(1.01, 40, "top-ranked:\nCaSiYAlO$_6$", fontsize=5.9, color=C["ink"],
         ha="right", va="top", linespacing=1.3)
ax2.axvspan(0.625, 1.03, color=C["pale"], alpha=0.55, zorder=0)

S.panel_letter(ax, "(a)", x=-0.185, y=1.17)
S.panel_letter(ax2, "(b)", x=-0.185, y=1.03)

S.save(fig, "fig6_stage1")
print("top200 by source:\n", T.source_dataset.value_counts())
print("pareto:\n", PF[["formula", "stability_utility", "gap_utility"]].to_string(index=False))
print(IM[["rho", "top_formula", "rank_CaSiYAlO6", "double_perovskites_in_top20"]].to_string(index=False))
