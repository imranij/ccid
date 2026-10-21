"""a7_figures.py -- figures for the quality comparison."""
from __future__ import annotations
import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import json
import numpy as np, pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

R = _CCID_ROOT + "/results"
C_CV, C_RD = "#1f77b4", "#d62728"


def main():
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))

    # (a) decomposition energy, CHGNet, full protocol
    e = pd.read_csv(f"{R}/a7_energetics_all.csv")
    e["pot"] = np.where(e["calc"].str.endswith("mace"), "mace", "chgnet")
    g = e[(e["pot"] == "chgnet") & (e["mode"] == "full")]
    a = g[g.population == "cvae"]["decomposition_energy_eV_per_atom"].dropna()
    b = g[g.population == "random"]["decomposition_energy_eV_per_atom"].dropna()
    bins = np.linspace(-0.2, 0.5, 24)
    ax[0].hist(a, bins=bins, alpha=0.6, color=C_CV, label=f"CVAE (n={len(a)})", density=True)
    ax[0].hist(b, bins=bins, alpha=0.6, color=C_RD, label=f"random (n={len(b)})", density=True)
    ax[0].axvline(0, color="k", lw=1, ls="--")
    ax[0].set_xlabel(r"$\Delta E_{\rm dec}$ into binary oxides (eV/atom), CHGNet")
    ax[0].set_ylabel("density"); ax[0].legend(fontsize=8)
    ax[0].set_title("(a) energetics: no separation", fontsize=10)

    # (b) chemical sanity proportions
    s = json.load(open(f"{R}/a7_compare_summary.json"))
    keys = ["cvae", "random_matched", "random_freq", "random"]
    names = ["CVAE", "rand\n(n=66)", "rand\nfreq-wt", "rand\nuniform"]
    metrics = [("frac_open_shell", "open-shell d/f cation"),
               ("frac_reduced_cation", r"needs Ti$^{3+}$/Sn$^{2+}$"),
               ("frac_pass_gate", "passes wide-gap gate")]
    w = 0.26
    x = np.arange(len(keys))
    for i, (k, lab) in enumerate(metrics):
        ax[1].bar(x + (i - 1) * w, [100 * s[p][k] for p in keys], w, label=lab)
    ax[1].set_xticks(x); ax[1].set_xticklabels(names, fontsize=8)
    ax[1].set_ylabel("% of population"); ax[1].legend(fontsize=7)
    ax[1].set_title("(b) chemical sanity: CVAE wins", fontsize=10)

    # (c) size-controlled steelman
    st = pd.read_csv(f"{R}/a7_steelman_size_controlled.csv")
    st = st[(st.axis == "predicted_property_rank_score") & (st.pool == "random")]
    x = np.arange(len(st))
    ax[2].bar(x - 0.2, st["cvae_topN_median"], 0.4, color=C_CV, label="CVAE top-$N$")
    ax[2].bar(x + 0.2, st["null_median_of_topN_median"], 0.4, color=C_RD,
              label="equal-budget random top-$N$")
    for i, (_, r) in enumerate(st.iterrows()):
        ax[2].text(i, max(r["cvae_topN_median"], r["null_median_of_topN_median"]) + 0.01,
                   f"p={r['p_random_draw_at_least_as_good']:.3f}", ha="center", fontsize=7)
    ax[2].set_xticks(x); ax[2].set_xticklabels([f"N={int(n)}" for n in st["top_N"]], fontsize=8)
    ax[2].set_ylim(0.5, 0.95)
    ax[2].set_ylabel("composite predicted-property score")
    ax[2].legend(fontsize=7)
    ax[2].set_title("(c) at matched budget, CVAE's best is better", fontsize=10)

    fig.tight_layout()
    fig.savefig(f"{R}/a7_quality_comparison.pdf")
    fig.savefig(f"{R}/a7_quality_comparison.png", dpi=160)
    print("wrote a7_quality_comparison.pdf/.png")


if __name__ == "__main__":
    main()
