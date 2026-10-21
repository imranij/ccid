"""fig5_chemistry.pdf -- chemical plausibility across the four arms (manuscript Fig. 7).

Panel (a): the four chemistry proportions of the chemistry table, all four arms, with
           Wilson 95 % intervals on each proportion.
Panel (b): per-element occurrence for the three Stage-1-favoured cations, same form.

Every arm is drawn in the same style on every axis.  The Ti3+/Sn2+ row, on which arm D
is the WORST arm, sits in panel (a) exactly like the others and is not set apart, so the
panel cannot be read as a set of wins.

Sources (pooled unique valid formulas, seeds 0-4)
  (a) a9_compare_proportions.csv, rows comparison in {'D vs A','D vs B','D vs C'},
      axes has_open_shell, has_stage1_good, passes_quality_gate, requires_reduced_cation:
        arm D      -> p_a, p_a_ci95, n_a  (identical in the three rows)
        arm A/B/C  -> p_b, p_b_ci95, n_b  of 'D vs A' / 'D vs B' / 'D vs C'
        right column: diff (D minus arm, pp) and fisher_p of the same row
  (b) a9_element_enrichment.csv, element in {Zr, Al, Hf}, same row logic:
        p_a / p_a_ci95 (D), p_b / p_b_ci95 (A, B, C), odds_ratio, fisher_p
"""
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, os, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import a6_style as S

S.apply()
C = S.C
R = _CCID_ROOT + "/results"
ARMS = S.ARMS4

P = pd.read_csv(f"{R}/a9_compare_proportions.csv")
EN = pd.read_csv(f"{R}/a9_element_enrichment.csv")


def per_arm(df, key_col, key):
    """{arm: (p, (lo, hi), n)} plus {arm: (diff, fisher_p, odds_ratio)} for D vs arm."""
    out, cmp_ = {}, {}
    for k in "ABC":
        r = df[(df.comparison == f"D vs {k}") & (df[key_col] == key)].iloc[0]
        assert r.group_a == "D" and r.group_b == k
        out[k] = (r.p_b, ast.literal_eval(r.p_b_ci95), int(r.n_b))
        cmp_[k] = (r["diff"], r.fisher_p, r.odds_ratio)
        dd = (r.p_a, ast.literal_eval(r.p_a_ci95), int(r.n_a))
        if "D" in out:
            assert abs(out["D"][0] - dd[0]) < 1e-12
        out["D"] = dd
    return out, cmp_


ROWS_A = [("has_open_shell", "Contains an open-shell\n$d$/$f$ cation"),
          ("has_stage1_good", "Contains Hf, Zr or Al\n(Stage-1 favoured)"),
          ("passes_quality_gate", "Passes the wide-gap\ndielectric sanity gate"),
          ("requires_reduced_cation",
           "Needs Ti$^{3{+}}$/Sn$^{2{+}}$\nto charge balance")]
ROWS_B = ["Zr", "Al", "Hf"]
DY = 0.205
OFF = {k: (1.5 - i) * DY for i, k in enumerate(ARMS)}

fig, (axA, axB) = plt.subplots(
    2, 1, figsize=(S.COL, 5.75),
    gridspec_kw=dict(height_ratios=[4.0, 3.0], hspace=0.36))
fig.subplots_adjust(left=0.335, right=0.875, top=0.885, bottom=0.115)


def draw_group(ax, y0, vals):
    for k in ARMS:
        p, ci, _ = vals[k]
        y = y0 + OFF[k]
        ax.plot([ci[0] * 100, ci[1] * 100], [y, y], color=S.ARM4[k], lw=1.5,
                solid_capstyle="butt", alpha=0.45, zorder=3)
        ax.plot([p * 100], [y], marker=S.MARK4[k], ms=3.9 if k != "D" else 3.4,
                color=S.ARM4[k], markeredgecolor="white", markeredgewidth=0.5,
                ls="none", zorder=5)


def right_col(ax, y0, cmp_, fmt):
    for k in "ABC":
        ax.text(1.025, y0 + OFF[k], fmt(cmp_[k]), transform=ax.get_yaxis_transform(),
                fontsize=5.9, color=S.ARM4_TEXT[k], ha="left", va="center")
    ax.text(1.025, y0 + OFF["D"], "—", transform=ax.get_yaxis_transform(),
            fontsize=5.9, color=C["ink_soft"], ha="left", va="center")


pmax = 0.0
log = []
# ================================================================ panel a ===
yA = np.arange(len(ROWS_A))[::-1].astype(float)
for y, (key, lab) in zip(yA, ROWS_A):
    vals, cmp_ = per_arm(P, "axis", key)
    draw_group(axA, y, vals)
    right_col(axA, y, cmp_, lambda c: f"{100*c[0]:+.1f}".replace("-", "−"))
    pmax = max(pmax, max(c[1] for c in cmp_.values()))
    log.append((key, {k: round(100 * vals[k][0], 1) for k in ARMS},
                {k: round(100 * cmp_[k][0], 1) for k in "ABC"}))
nA = {k: vals[k][2] for k in ARMS}

axA.set_yticks(yA)
axA.set_yticklabels([l for _, l in ROWS_A], fontsize=6.8, linespacing=1.3)
axA.set_ylim(yA.min() - 0.52, yA.max() + 0.52)
axA.set_xlim(0, 100)
axA.set_xticks([0, 25, 50, 75, 100])
axA.set_xlabel("formulas with the property (%)")
axA.tick_params(axis="y", length=0)
S.hgrid(axA, axis="x")
axA.spines["left"].set_visible(False)
axA.text(1.025, 1.005, "D $-$ arm\n(pp)", transform=axA.transAxes, fontsize=5.8,
         color=C["ink_soft"], ha="left", va="bottom", linespacing=1.2)

handles = [Line2D([], [], marker=S.MARK4[k], ls="none", ms=4.0, color=S.ARM4[k],
                  markeredgecolor="white", markeredgewidth=0.5) for k in ARMS]
axA.legend(handles, [f"{S.ARM4_LABEL[k]} ($n$ = {nA[k]:,})" for k in ARMS],
           loc="lower left", bbox_to_anchor=(-0.50, 1.075), ncol=2,
           fontsize=6.3, handletextpad=0.3, labelspacing=0.25, columnspacing=0.8)

# ================================================================ panel b ===
yB = np.arange(len(ROWS_B))[::-1].astype(float)
for y, el in zip(yB, ROWS_B):
    vals, cmp_ = per_arm(EN, "element", el)
    draw_group(axB, y, vals)
    right_col(axB, y, cmp_, lambda c: f"{c[2]:.1f}$\\times$")
    pmax = max(pmax, max(c[1] for c in cmp_.values()))
    log.append((el, {k: round(100 * vals[k][0], 1) for k in ARMS},
                {k: round(cmp_[k][2], 2) for k in "ABC"}))

axB.set_yticks(yB)
axB.set_yticklabels(ROWS_B, fontsize=7.2)
axB.set_ylim(yB.min() - 0.52, yB.max() + 0.52)
axB.set_xlim(0, 40)
axB.set_xticks([0, 10, 20, 30, 40])
axB.set_xlabel("formulas containing the element (%)")
axB.tick_params(axis="y", length=0)
S.hgrid(axB, axis="x")
axB.spines["left"].set_visible(False)
axB.text(1.025, 1.005, "odds ratio\nD : arm", transform=axB.transAxes, fontsize=5.8,
         color=C["ink_soft"], ha="left", va="bottom", linespacing=1.2)

e = int(np.ceil(np.log10(pmax)))
fig.text(0.012, 0.006,
         "Bars: Wilson 95 % CI on each proportion (invisibly narrow for A and B). "
         "Pooled unique\nvalid formulas, seeds 0–4. Fisher exact $p$ for D vs every "
         f"other arm $\\leq$ 10$^{{{e}}}$ on all seven rows.",
         fontsize=5.7, color=C["ink_soft"], ha="left", va="bottom", linespacing=1.45)

S.panel_letter(axA, "(a)", x=-0.505, y=1.21)
S.panel_letter(axB, "(b)", x=-0.505, y=1.04)

S.save(fig, "fig5_chemistry")
for row in log:
    print(row)
print("max Fisher p, D vs arm:", pmax)
