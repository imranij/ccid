"""fig3_freebits.pdf -- posterior collapse: diagnosis (per-dimension KL) and fix (lambda sweep).

Sources
  (a) deposited checkpoint per-dimension KL
        a8_baseline_pathology.json -> latent.kl_per_dim_nats_sorted  (32 values)
        latent.total_kl_nats, latent.active_units_kl_gt_0.01
      corrected model per-dimension KL
        a8_sweep_all.csv, tag='fb0.5_b0.01', cols kl_per_dim_sorted,
        total_kl_nats, active_units_kl_gt_0.01
  (b) lambda sweep, beta_max = 0.01 series of a8_sweep_all.csv
        free_bits_per_dim, test_composition_mse, gen_validity_rate_mean,
        gen_unique_valid_count_mean
      deposited-checkpoint reference rules:
        MSE 0.12786 from dgx/outputs_dgx/composition_cvae_test_metrics.json
                       (composition_mse)
        69 distinct formulas from a8_headline_comparison.csv, arm=deposited_cvae,
                       unique_valid_count_mean (5 seeds)
      The deposited model has no per-dimension lambda, so it is drawn as a rule, not
      as a sweep point.
"""
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import sys, os, json, ast
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import a6_style as S

S.apply()
C = S.C
R = _CCID_ROOT + "/results"
DGX = _CCID_ROOT + "/dgx/outputs_dgx"

path = json.load(open(f"{R}/a8_baseline_pathology.json"))
sw = pd.read_csv(f"{R}/a8_sweep_all.csv")
tm = json.load(open(f"{DGX}/composition_cvae_test_metrics.json"))
hc = pd.read_csv(f"{R}/a8_headline_comparison.csv").set_index("arm")

kl_dep = np.array(path["latent"]["kl_per_dim_nats_sorted"], float)
row_fix = sw[sw.tag == "fb0.5_b0.01"].iloc[0]
kl_fix = np.array(ast.literal_eval(row_fix["kl_per_dim_sorted"]), float)
au_dep = int(path["latent"]["active_units_kl_gt_0.01"])
au_fix = int(row_fix["active_units_kl_gt_0.01"])

FLOOR = 3e-7                       # below every observed value; no clipping occurs
n_below = int((kl_dep < 0.01).sum())
kl_dep_p, kl_fix_p = kl_dep, kl_fix

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(S.COL, 4.75))
fig.subplots_adjust(left=0.180, right=0.795, top=0.955, bottom=0.092, hspace=0.66)

# ================================================================ panel a ===
x = np.arange(1, 33)
ax1.plot(x, kl_fix_p, marker="o", ms=2.8, color=C["cvae"], lw=1.1, zorder=5,
         clip_on=False)
ax1.plot(x, kl_dep_p, marker="o", ms=2.8, color=C["acc1"], lw=1.1, zorder=5,
         markerfacecolor="white", clip_on=False)

ax1.axhspan(FLOOR, 0.01, color=C["pale"], zorder=0)
ax1.axhline(0.01, color=C["ink_soft"], lw=0.7, ls=(0, (3, 2)), zorder=4)
ax1.text(0.9, 0.014, "active-unit threshold, 0.01 nats", fontsize=5.9,
         color=C["ink_soft"], ha="left", va="bottom")
ax1.text(32.5, 1.1 * FLOOR, "dead dimensions", fontsize=5.9, style="italic",
         color=C["ink_soft"], ha="right", va="bottom")

ax1.set_yscale("log")
ax1.set_ylim(FLOOR, 2.0)
ax1.set_xlim(0.5, 32.5)
ax1.set_xticks([1, 8, 16, 24, 32])
ax1.set_yticks([1e-6, 1e-4, 1e-2, 1e0])
ax1.set_xlabel("latent dimension, ranked by KL")
ax1.set_ylabel("KL divergence (nats dim$^{-1}$)")
S.hgrid(ax1, axis="y")

ax1.annotate(f"corrected ($\\lambda$ = 0.5 nats)\n{au_fix} of 32 active, "
             f"{row_fix['total_kl_nats']:.1f} nats total",
             xy=(20, kl_fix_p[19]), xytext=(13.0, 0.045),
             fontsize=6.4, color=C["cvae"], fontweight="bold", linespacing=1.3,
             ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=C["cvae"],
                             shrinkA=1, shrinkB=2))
ax1.annotate(f"deposited checkpoint\n{au_dep} of 32 active, "
             f"{path['latent']['total_kl_nats']:.2f} nats total",
             xy=(6.0, kl_dep_p[5]), xytext=(9.0, 6e-5),
             fontsize=6.4, color=C["acc1"], fontweight="bold", linespacing=1.3,
             ha="left", va="bottom",
             arrowprops=dict(arrowstyle="-", lw=0.6, color=C["acc1"],
                             shrinkA=1, shrinkB=2))


# ================================================================ panel b ===
b = sw[(sw.loss == "fixed") & (sw.beta_max == 0.01)].sort_values("free_bits_per_dim")
lam, mse = b["free_bits_per_dim"].values, b["test_composition_mse"].values
val, uni = b["gen_validity_rate_mean"].values, b["gen_unique_valid_count_mean"].values
mse_dep = tm["composition_mse"]
uni_dep = hc.loc["deposited_cvae", "unique_valid_count_mean"]

ax2.set_xscale("symlog", linthresh=0.25, linscale=0.9)
ax2.plot(lam, mse, marker="o", ms=3.0, color=C["acc1"], zorder=6, clip_on=False)
ax2.plot(lam, val, marker="s", ms=3.0, color=C["cvae"], zorder=6, clip_on=False)
ax2.set_xlim(0, 4.6)
ax2.set_ylim(0, 0.60)
ax2.set_xticks([0, 0.25, 0.5, 1.0, 2.0, 4.0])
ax2.set_xticklabels(["0", "0.25", "0.5", "1", "2", "4"])
ax2.set_xlabel("per-dimension KL floor $\\lambda$ (nats)")
ax2.set_ylabel("reconstruction MSE   /   validity")
S.hgrid(ax2, axis="y")

ax3 = ax2.twinx()
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(True)
ax3.spines["right"].set_color(C["acc2"])
ax3.plot(lam, uni, marker="^", ms=3.2, color=C["acc2"], zorder=6, clip_on=False)
ax3.set_yscale("log")
ax3.set_ylim(30, 9000)
ax3.set_yticks([100, 1000])
ax3.set_yticklabels(["100", "1,000"])
ax3.set_ylabel("distinct valid formulas\nper 20,000 samples", color=C["acc2"],
               linespacing=1.25)
ax3.tick_params(axis="y", colors=C["acc2"])

ax2.axhline(mse_dep, color=C["acc1"], lw=0.7, ls=(0, (1.6, 1.6)), zorder=2)
ax3.axhline(uni_dep, color=C["acc2"], lw=0.7, ls=(0, (1.6, 1.6)), zorder=2)
ax2.text(0.015, 0.575, f"dotted lines:\ndeposited checkpoint\n(MSE {mse_dep:.3f}; {uni_dep:.0f} formulas)",
         fontsize=5.9, color=C["ink_soft"], ha="left", va="top", linespacing=1.2)

# operating point highlighted in the manuscript
ax2.axvline(0.5, color=C["rule"], lw=0.7, zorder=1)
ax2.text(0.53, 0.585, "$\\lambda$ = 0.5", fontsize=6.0, color=C["ink_soft"],
         ha="left", va="top")

ax2.text(0.02, 0.063, "reconstruction MSE", fontsize=6.4, color=C["acc1"],
         fontweight="bold", ha="left", va="top")
S.label_line(ax2, lam[6], val[6], "validity", C["cvae"], dx=7, dy=1,
             ha="left", fontsize=6.4)
ax3.annotate("distinct formulas", xy=(lam[1], uni[1]), xytext=(2, 7),
             textcoords="offset points", fontsize=6.4, color=C["acc2"],
             fontweight="bold", ha="left", va="bottom")

S.panel_letter(ax1, "(a)", x=-0.215, y=1.02)
S.panel_letter(ax2, "(b)", x=-0.215, y=1.02)

S.save(fig, "fig3_freebits")
print("dims below 0.01 nats:", n_below, " deposited total KL:",
      path["latent"]["total_kl_nats"], " corrected:", row_fix["total_kl_nats"])
