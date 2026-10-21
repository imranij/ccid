"""fig8_stage2_pipeline.pdf -- Stage-2 generation as a four-arm comparative study
(manuscript Fig. 3): four generation strategies, one identical rule-based filter.

Drawn in the authors' Visio style (a6_visio.py).

Sources
  arm D architecture, 25,148 records   manuscript Methods section 2.2 (unchanged)
  arm D loss settings                  checkpoint tag fb0.5_b0.01 (a8_sweep_all.csv:
                                       free_bits_per_dim = 0.5, beta_max = 0.01)
  arm C parents / threshold            analysis/a9_fourarm/a9_arm_c_parents.json:
                                       n_parents, threshold
  filter steps                         manuscript Algorithm 1
  reachable output set 129,205         a2_decoder_reachability.json ->
                                       total_reachable_valid_formulas
  outcomes per arm                     a9_counts_summary.csv, rows arm in {A,B,C,D}:
                                       validity_rate_mean / _std,
                                       unique_valid_count_mean / _std
                                       (five seeds 0-4, 20,000 proposals each)
"""
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, os, json
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import pandas as pd
import a6_style as S
import a6_visio as V

R = _CCID_ROOT + "/results"
cs = pd.read_csv(f"{R}/a9_counts_summary.csv").set_index("arm")
PAR = json.load(open(_CCID_ROOT + "/analysis/a9_fourarm/a9_arm_c_parents.json"))
n_par, thr = PAR["n_parents"], PAR["threshold"]
REACH = json.load(open(f"{R}/a2_decoder_reachability.json"))["total_reachable_valid_formulas"]
sw = pd.read_csv(f"{R}/a8_sweep_all.csv")
dck = sw[sw.tag == "fb0.5_b0.01"].iloc[0]
lam, bmax = float(dck.free_bits_per_dim), float(dck.beta_max)
thr_exp = int(round(__import__("math").log10(thr)))
assert abs(thr - 10 ** thr_exp) < 1e-12

W, H = 6.6, 5.20
fig, ax = V.canvas(W, H)

GAP = 0.30
wA, wB, wC = 2.15, 1.65, 1.95
xA = (W - (wA + wB + wC + 2 * GAP)) / 2
xB = xA + wA + GAP
xC = xB + wB + GAP
TOP, BOT = H - 0.05, 0.05

# ------------------------------------------------- column A: the four arms
G = 0.11
hA = {"A": 0.80, "B": 0.80, "C": 1.30, "D": 1.87}
assert abs(sum(hA.values()) + 3 * G - (TOP - BOT)) < 1e-9
ys, y = {}, TOP
for k in "ABCD":
    y -= hA[k]
    ys[k] = y
    y -= G
pw_ = wA - 0.18
px_ = xA + 0.09


def arm_box(k, title, sub, pills, ph):
    V.container(ax, xA, ys[k], wA, hA[k], title, sub=sub)
    yt = ys[k] + hA[k] - 0.43
    for t in pills:
        V.pill_arm(ax, px_, yt - ph, pw_, ph, t, k)
        yt -= ph + 0.05


arm_box("A", "A  Random Elements", "no training of any kind",
        ["Cations uniform on the 41-element\nwhitelist; fraction profile kept"], 0.30)
arm_box("B", "B  Element Statistics", "no training; table frequencies only",
        ["Cations $\\propto$ training-table frequency;\nfraction profile kept"], 0.30)
arm_box("C", "C  Ion Substitution", "Hautier et al. (2011) probabilities, pymatgen",
        [f"Parent: one of {n_par:,} known oxides\nfrom the training table",
         "Substitute ions $\\propto$ probability; joint\n"
         f"$p \\geq 10^{{{thr_exp}}}$, charge-compensating pairs"], 0.36)
arm_box("D", "D  Composition-CVAE (corrected)", "trained on 25,148 records",
        ["Input: 77 element fractions\n+ 2 conditions ($U_s$, $U_g$)",
         "Encoder 79 $\\rightarrow$ 256 $\\rightarrow$ 128; 32-dim latent",
         "Decoder 32 + 2 $\\rightarrow$ 128 $\\rightarrow$ 256 $\\rightarrow$ 77",
         f"Free bits $\\lambda$ = {lam:g} per dim., $\\beta_{{\\max}}$ = {bmax:g}",
         "Sample $z \\sim \\mathcal{N}(0, I_{32})$;\ndecode with the condition"], 0.235)

# ------------------------------------------------- column B: identical filter
bY, bH = BOT, TOP - BOT
V.container(ax, xB, bY, wB, bH, "Identical Rule-Based\nChemical Filter", title_pad=0.07)
ax.text(xB + wB / 2, bY + bH - 0.47, "one code path, all four arms\n"
        "20,000 proposals per run, five seeds", ha="center",
        va="top", fontsize=V.PILL_PT - 0.4, color=S.C["ink_soft"], zorder=4)
steps = ["Take the three\nhighest-fraction cations",
         "Round to integer counts\n(max 4); reduce by GCD",
         "Search O counts 1–12\nfor charge balance",
         "Reject any element outside\nthe 41-element whitelist"]
py, ph, pg = bY + bH - 1.02, 0.46, 0.36
for i, t in enumerate(steps):
    y0 = py - i * (ph + pg)
    V.pill(ax, xB + 0.10, y0 - ph, wB - 0.20, ph, t)
    if i < 3:
        V.arrow(ax, (xB + wB / 2, y0 - ph), (xB + wB / 2, y0 - ph - pg + 0.015),
                lw=1.3, head=7.5)
V.pill(ax, xB + 0.10, bY + 0.10, wB - 0.20, 0.50,
       f"Reachable output set:\n{REACH:,} distinct formulas;\nthe filter does not limit diversity",
       fc="white", italic=True, size=V.PILL_PT - 0.5)

for k in "ABCD":
    ym = ys[k] + hA[k] / 2
    V.arrow(ax, (xA + wA, ym), (xB - 0.015, ym))

# ------------------------------------------------- column C: outcomes + downstream
oH = 2.62
oY = TOP - oH
V.container(ax, xC, oY, wC, oH, "Outcomes", sub="five-seed means $\\pm$ s.d.")
yt = oY + oH - 0.47
for k in "ABCD":
    v, vs = cs.loc[k, "validity_rate_mean"], cs.loc[k, "validity_rate_std"]
    u, us = cs.loc[k, "unique_valid_count_mean"], cs.loc[k, "unique_valid_count_std"]
    V.pill_arm(ax, xC + 0.10, yt - 0.46, wC - 0.20, 0.46,
               f"{S.ARM4_LABEL[k]}\nvalidity {v:.3f} $\\pm$ {vs:.3f}\n"
               f"{u:,.0f} $\\pm$ {us:.0f} distinct valid formulas", k,
               size=V.PILL_PT - 0.3)
    yt -= 0.46 + 0.06

dY = BOT
dH = oY - 0.30 - dY
V.highlight(ax, xC, dY, wC, dH, "Matched Downstream\nAnalysis")
V.stack(ax, xC + 0.10, dY + dH - 0.55, wC - 0.20,
        ["Chemistry (Fig. 7)",
         "uMLIP energetics, CHGNet\nand MACE-MP-0 (Fig. 8)",
         "Novelty against the\ntraining table",
         "Every step identical\nacross the four arms"],
        h=0.33, gap=0.06, fc="white", ec=S.C["ink"])

V.arrow(ax, (xB + wB, oY + oH / 2), (xC - 0.015, oY + oH / 2))
V.arrow(ax, (xC + wC / 2, oY), (xC + wC / 2, dY + dH + 0.015))

S.save(fig, "fig8_stage2_pipeline")
for k in "ABCD":
    print(k, round(cs.loc[k, "validity_rate_mean"], 5), round(cs.loc[k, "validity_rate_std"], 5),
          round(cs.loc[k, "unique_valid_count_mean"], 1), round(cs.loc[k, "unique_valid_count_std"], 1))
print("parents", n_par, "threshold", thr, "lambda", lam, "beta_max", bmax, "reach", REACH)
