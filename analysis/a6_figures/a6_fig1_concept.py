"""fig1_concept.pdf -- conventional vs controlled evaluation of a composition-only generator
(manuscript Fig. 1).

Drawn in the authors' Visio style (a6_visio.py), laid out like the original concept
figure: two dashed frames stacked, each a left-to-right chain of containers.
Panel (b) shows the four-arm comparative study (a9) as a ladder of increasing chemical
knowledge, A random -> B element statistics -> C ion substitution -> D corrected CVAE,
all through one filter.  Outcomes are summarised in words only; every statement is a
direction read from the a9 result files (a9_counts_summary.csv, a9_compare_proportions.csv,
a9_energetics_summary.csv / a9_energetics_pairwise_vs_D.csv) and is checked by the
assertions at the bottom of this script, so a re-run on changed results fails loudly
instead of printing a stale sentence.
"""
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import a6_style as S
import a6_visio as V

W, H = 6.6, 4.47
fig, ax = V.canvas(W, H)

CW, GAP = 1.36, 0.27            # container width, gap between containers
X0 = (W - (4 * CW + 3 * GAP)) / 2
xs = [X0 + i * (CW + GAP) for i in range(4)]
PW = CW - 0.18                  # pill width
px = lambda i: xs[i] + 0.09


def chain_arrows(y):
    for i in range(3):
        V.arrow(ax, (xs[i] + CW, y), (xs[i + 1] - 0.015, y))


# ------------------------------------------------------------- (a) conventional
FA_Y, FA_H = 2.82, 1.60
V.frame(ax, 0.05, FA_Y, W - 0.10, FA_H, "(a) Conventional evaluation",
        "one arm; the model is credited with the whole number")
CY, CH = FA_Y + 0.12, 1.20
top = CY + CH - 0.30

V.container(ax, xs[0], CY, CW, CH, "Trained Generator")
V.stack(ax, px(0), top - 0.22, PW, ["Composition-CVAE"], h=0.30)

V.container(ax, xs[1], CY, CW, CH, "Rule-Based Filter")
V.stack(ax, px(1), top, PW, ["Integer stoichiometry", "Charge balance",
                              "41-element whitelist"], h=0.24, gap=0.06)

V.container(ax, xs[2], CY, CW, CH, "Reported Metrics")
V.stack(ax, px(2), top, PW, ["Validity", "Uniqueness", "Novelty"], h=0.24, gap=0.06)

V.highlight(ax, xs[3], CY, CW, CH, "Attributed to the Model")
V.stack(ax, px(3), top, PW, ["The filter alone is\nalready a generator",
                              "Contributions are\nnot separable"], h=0.36, gap=0.08,
        fc="white", ec=S.C["ink"])
chain_arrows(CY + CH / 2)

# ---------------------------------------------------------------- heading
V.label(ax, W / 2, 2.69, "Conventional vs. Controlled Evaluation", size=10.0,
        italic=True, bold=True)

# --------------------------------------------------------- (b) controlled
FB_Y, FB_H = 0.05, 2.50
V.frame(ax, 0.05, FB_Y, W - 0.10, FB_H, "(b) Controlled evaluation (this work)",
        "four strategies, one filter; each step adds chemical knowledge")
CY, CH = FB_Y + 0.12, 2.10
top = CY + CH - 0.30

# container 1: the knowledge ladder, A at the top, an arrow down the left edge
V.container(ax, xs[0], CY, CW, CH, "Four Strategies")
LAD = [("A", "A  Random elements\n(whitelist)"),
       ("B", "B  Element statistics\n(training frequency)"),
       ("C", "C  Ion substitution\nof known compounds"),
       ("D", "D  Learned model\n(corrected CVAE)")]
lx, lw_, lh, lg = xs[0] + 0.25, CW - 0.34, 0.37, 0.065
ly0 = top - 0.02
for i, (k, t) in enumerate(LAD):
    V.pill_arm(ax, lx, ly0 - (i + 1) * lh - i * lg, lw_, lh, t, k, size=V.PILL_PT - 0.4)
ybot = ly0 - 4 * lh - 3 * lg
V.arrow(ax, (xs[0] + 0.135, ly0 - 0.02), (xs[0] + 0.135, ybot + 0.02), lw=1.1, head=6.5)
ax.text(xs[0] + 0.085, (ly0 + ybot) / 2, "more chemical knowledge", rotation=90,
        ha="center", va="center", fontsize=5.9, style="italic", color=S.C["ink_soft"],
        zorder=5)

V.container(ax, xs[1], CY, CW, CH, "Identical Filter")
V.stack(ax, px(1), top - 0.02, PW, ["Same code path", "Same budget\n(20,000 × 5 seeds)",
                                     "Charge balance", "41-element whitelist"],
        h=0.37, gap=0.065)

V.container(ax, xs[2], CY, CW, CH, "Outcomes")
V.stack(ax, px(2), top - 0.02, PW,
        ["Validity falls at\nevery step (Fig. 5)",
         "Chemical plausibility\nrises at every step (Fig. 7)",
         "Lower $\\Delta E_{\\mathrm{dec}}$ only for C,\nmany of them known\ncompounds (Fig. 8)"],
        h=0.50, gap=0.085)

V.highlight(ax, xs[3], CY, CW, CH, "What Learning Adds")
V.stack(ax, px(3), top - 0.02, PW, ["Strongest chemical-\nplausibility prior",
                                     "Below random on\nvalidity and diversity",
                                     "Ties random on\nuMLIP energetics",
                                     "Most Ti$^{3{+}}$/Sn$^{2{+}}$\ncharge balancing"],
        h=0.37, gap=0.065, fc="white", ec=S.C["ink"])
chain_arrows(CY + CH / 2)

S.save(fig, "fig1_concept")

# ---- every sentence above is a direction in the a9 results: check it ------------
import pandas as pd
R = _CCID_ROOT + "/results"
cs = pd.read_csv(f"{R}/a9_counts_summary.csv").set_index("arm")
v = cs.loc[["A", "B", "C", "D"], "validity_rate_mean"].values
u = cs.loc[["A", "B", "C", "D"], "unique_valid_count_mean"]
assert all(v[i] > v[i + 1] for i in range(3)), "validity no longer falls at every step"
assert u.idxmax() == "A" and v.argmax() == 0, "random no longer highest on counts"
P = pd.read_csv(f"{R}/a9_compare_proportions.csv")
pa = {}
for k in "ABC":
    r = P[(P.comparison == f"D vs {k}")].set_index("axis")
    pa[k] = r.p_b
pa["D"] = P[P.comparison == "D vs A"].set_index("axis").p_a
osh = [pa[k]["has_open_shell"] for k in "ABCD"]
gate = [pa[k]["passes_quality_gate"] for k in "ABCD"]
red = [pa[k]["requires_reduced_cation"] for k in "ABCD"]
assert all(osh[i] > osh[i + 1] for i in range(3)), "open-shell no longer falls each step"
assert all(gate[i] < gate[i + 1] for i in range(3)), "gate no longer rises each step"
assert max(red) == red[3], "D no longer the most Ti3+/Sn2+"
pw = pd.read_csv(f"{R}/a9_energetics_pairwise_vs_D.csv")
pw = pw[(pw["mode"] == "full")]
for pot in ("chgnet", "mace"):
    q = pw[pw.calc == pot].set_index("arm")
    assert q.loc["C", "cliffs_delta"] < 0 and q.loc["C", "mannwhitney_p"] < 0.01
    assert q.loc["A", "mannwhitney_p"] > 0.05 and q.loc["B", "mannwhitney_p"] > 0.05
es = pd.read_csv(f"{R}/a9_energetics_summary.csv")
es = es[(es.calc == "chgnet") & (es["mode"] == "full")].set_index("arm")
print("validity", [round(x, 4) for x in v], "unique", list(u.round(1)))
print("open-shell", [round(x, 3) for x in osh], "gate", [round(x, 3) for x in gate],
      "Ti3+/Sn2+", [round(x, 3) for x in red])
print("C relaxed known", int(es.loc["C", "n_in_training"]), "of",
      int(es.loc["C", "n_formulas_relaxed"]))
