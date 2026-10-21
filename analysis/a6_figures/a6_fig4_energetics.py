"""fig4_energetics.pdf -- decomposition-energy distributions of the four arms under CHGNet
and MACE-MP-0 (manuscript Fig. 8).

Empirical CDFs on a common axis (no smoothing choice to argue about).  Under each ECDF a
strip marks every arm's median (marker) and inter-quartile range (bar).  Within arm C,
training-table compounds are marked on the curve itself, because C is the only arm whose
distribution is shifted to lower energies and much of that shift is known compounds.
The deposited (buggy) CVAE is drawn as a thin dashed reference, not as an arm.

Sources (spinel-corrected values; the *_asrun files are NOT used)
  distributions : a9_energetics_all.csv, mode == 'full', non-null
                  decomposition_energy_eV_per_atom; pot in {chgnet, mace};
                  arm in {A, B, C, D, CVAE_dep}; in_training_table marks known compounds.
  medians, IQR  : a9_energetics_summary.csv, mode == 'full': median_dE, q25, q75
                  (asserted equal to the median of the plotted values)
  % <= 0        : a9_energetics_summary.csv, frac_le0
  delta vs D, p : a9_energetics_pairwise_vs_D.csv, mode == 'full': cliffs_delta,
                  cliffs_delta_ci_lo / _hi, mannwhitney_p
  Kruskal-Wallis: a9_energetics_kruskal.csv, mode == 'full': kruskal_p, kruskal_ABD_p
  C novel median: a9_energetics_summary.csv, arm C, mode == 'full_novel', median_dE
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

E = pd.read_csv(f"{R}/a9_energetics_all.csv")
E = E[(E["mode"] == "full") & E.decomposition_energy_eV_per_atom.notna()]
SM = pd.read_csv(f"{R}/a9_energetics_summary.csv")
PW = pd.read_csv(f"{R}/a9_energetics_pairwise_vs_D.csv")
PW = PW[PW["mode"] == "full"]
KW = pd.read_csv(f"{R}/a9_energetics_kruskal.csv")
KW = KW[KW["mode"] == "full"].set_index("calc")

POT = [("chgnet", "CHGNet"), ("mace", "MACE-MP-0")]
SER = ["dep", "A", "B", "C", "D"]                 # drawing order (D on top)
ROWS = ["A", "B", "C", "D", "dep"]                # strip / table order
KEY = {"A": "A", "B": "B", "C": "C", "D": "D", "dep": "CVAE_dep"}
LW = {"A": 1.15, "B": 1.05, "C": 1.15, "D": 1.35, "dep": 0.85}
XLO, XHI = -0.32, 0.46


def vals(pot, k):
    d = E[(E.pot == pot) & (E.arm == KEY[k])]
    return (d.decomposition_energy_eV_per_atom.values.astype(float),
            d.in_training_table.values.astype(bool))


def summ(pot, k, mode="full"):
    return SM[(SM.calc == pot) & (SM["mode"] == mode) & (SM.arm == KEY[k])].iloc[0]


def ecdf(v):
    v = np.sort(v)
    return (np.concatenate([[XLO - 1], v, [XHI + 1]]),
            np.concatenate([[0.0], np.arange(1, len(v) + 1) / len(v), [1.0]]))


fig = plt.figure(figsize=(S.COL, 6.80))
gs = fig.add_gridspec(5, 1, height_ratios=[1.0, 0.40, 0.17, 1.0, 0.40], hspace=0.0,
                      left=0.165, right=0.972, top=0.875, bottom=0.315)
axes = [(fig.add_subplot(gs[0]), fig.add_subplot(gs[1])),
        (fig.add_subplot(gs[3]), fig.add_subplot(gs[4]))]
for (a, b) in axes:
    a.sharex(b)

printed = []
for (ax, sx), (pot, potlab) in zip(axes, POT):
    ax.axvspan(XLO, 0.0, color=C["pale"], zorder=0)
    ax.axvline(0.0, color=C["ink_soft"], lw=0.7, ls=(0, (3, 2)), zorder=2)
    for k in SER:
        v, known = vals(pot, k)
        x, y = ecdf(v)
        ax.step(x, y, where="post", color=S.ARM4[k], lw=LW[k], ls=S.LS4[k],
                zorder=5 if k != "dep" else 4)
        if k == "C":                                   # known compounds on C's curve
            order = np.argsort(v, kind="stable")
            vs, ks = v[order], known[order]
            Fy = np.arange(1, len(v) + 1) / len(v)
            ax.plot(vs[ks], Fy[ks], ls="none", marker=S.MARK4["C"], ms=2.9,
                    color=S.ARM4["C"], markeredgecolor="white", markeredgewidth=0.35,
                    zorder=7)
        n_hi = int((v > XHI).sum())
        if n_hi:
            ax.annotate(f"{n_hi} off scale, to {v.max():+.3f}", xy=(XHI, 1 - n_hi / len(v)),
                        xytext=(-3, -9), textcoords="offset points", fontsize=5.4,
                        color=S.ARM4_TEXT[k], ha="right", va="top")
        assert (v >= XLO).all()
    ax.text(0.985, 0.06, potlab, transform=ax.transAxes, fontsize=7.4,
            fontweight="bold", color=C["ink"], ha="right", va="bottom")
    ax.set_ylim(0, 1.02)
    ax.set_xlim(XLO, XHI)
    ax.set_yticks([0, 0.25, 0.5, 0.75, 1.0])
    ax.set_yticklabels(["0", "0.25", "0.50", "0.75", "1"])
    ax.set_ylabel("cumulative fraction\nof candidates", linespacing=1.3)
    S.hgrid(ax, axis="y")
    ax.tick_params(axis="x", labelbottom=False)

    # ---- median + IQR strip -------------------------------------------------
    sx.axvspan(XLO, 0.0, color=C["pale"], zorder=0)
    sx.axvline(0.0, color=C["ink_soft"], lw=0.7, ls=(0, (3, 2)), zorder=2)
    yy = {k: len(ROWS) - 1 - i for i, k in enumerate(ROWS)}
    for k in ROWS:
        v, _ = vals(pot, k)
        s = summ(pot, k)
        assert abs(np.median(v) - s.median_dE) < 1e-9, (pot, k)
        assert len(v) == int(s.n_with_reference), (pot, k)
        hollow = k == "dep"
        sx.plot([s.q25, s.q75], [yy[k]] * 2, color=S.ARM4[k], lw=1.6 if not hollow else 1.0,
                alpha=0.45 if not hollow else 0.9, solid_capstyle="butt", zorder=3,
                ls="-" if not hollow else S.LS4["dep"])
        sx.plot([s.median_dE], [yy[k]], marker=S.MARK4[k], ms=3.6 if k != "D" else 3.2,
                color=S.ARM4[k], markerfacecolor="white" if hollow else S.ARM4[k],
                markeredgecolor=S.ARM4[k] if hollow else "white",
                markeredgewidth=0.8 if hollow else 0.5, ls="none", zorder=5)
        printed.append((pot, k, len(v), round(float(s.median_dE), 4),
                        round(float(s.q25), 4), round(float(s.q75), 4),
                        round(float(s.frac_le0), 3)))
    sx.set_ylim(-0.7, len(ROWS) - 0.3)
    sx.set_yticks([yy[k] for k in ROWS])
    sx.set_yticklabels(["A", "B", "C", "D", "dep."], fontsize=5.6)
    for t, k in zip(sx.get_yticklabels(), ROWS):
        t.set_color(S.ARM4_TEXT[k])
    sx.tick_params(axis="y", length=0, pad=1.5)
    sx.spines["left"].set_visible(False)
    sx.text(0.985, 0.97, "median, IQR", transform=sx.transAxes, fontsize=5.5,
            color=C["ink_soft"], ha="right", va="top", style="italic")
    S.hgrid(sx, axis="x")

axes[1][1].set_xlabel("$\\Delta E_{\\mathrm{dec}}$ into binary oxides (eV atom$^{-1}$)")
axes[0][1].tick_params(axis="x", labelbottom=True)
axes[0][0].text(XLO + 0.012, 0.985, "stable against this\ndecomposition channel",
                fontsize=5.8, color=C["ink_soft"], ha="left", va="top",
                linespacing=1.3, style="italic")

nn = {k: len(vals("chgnet", k)[0]) for k in ROWS}
nC_known = int(summ("chgnet", "C").n_in_training)
handles = [Line2D([], [], color=S.ARM4[k], lw=LW[k], ls=S.LS4[k]) for k in ROWS]
labels = [f"{S.ARM4_LABEL[k]} ($n$ = {nn[k]})" for k in ROWS]
handles.append(Line2D([], [], color=S.ARM4["C"], marker=S.MARK4["C"], ls="none", ms=3.0,
                      markeredgecolor="white", markeredgewidth=0.35))
labels.append(f"C: training-table compound ({nC_known} of {nn['C']})")
axes[0][0].legend(handles, labels, loc="lower left", bbox_to_anchor=(-0.19, 1.035),
                  ncol=2, fontsize=5.9, handlelength=1.9, labelspacing=0.25,
                  columnspacing=0.8, handletextpad=0.4)

# ---- statistics strip --------------------------------------------------------
COLX = [0.020, 0.225, 0.425, 0.535, 0.800]
HEAD = ["", "median $\\Delta E_{\\mathrm{dec}}$", "$\\leq 0$ (%)",
        "Cliff's $\\delta$ vs D, CHGNet", "$p$ vs D"]
ytop = 0.214
for cx, h in zip(COLX, HEAD):
    fig.text(cx, ytop, h, fontsize=5.7, color=C["ink"], ha="left", va="bottom",
             fontweight="bold")
fig.add_artist(plt.Line2D([0.020, 0.985], [ytop - 0.005] * 2, color=C["ink_soft"], lw=0.6))
m = lambda x: f"{x:+.3f}".replace("-", "−")
for i, k in enumerate(ROWS):
    yy_ = ytop - 0.022 - i * 0.0185
    sc, sm = summ("chgnet", k), summ("mace", k)
    if k == "D":
        dl, pp = "ref.", "ref."
    else:
        pc = PW[(PW.calc == "chgnet") & (PW.arm == KEY[k])].iloc[0]
        pm = PW[(PW.calc == "mace") & (PW.arm == KEY[k])].iloc[0]
        dl = (f"{pc.cliffs_delta:+.2f} ({pc.cliffs_delta_ci_lo:+.2f}, "
              f"{pc.cliffs_delta_ci_hi:+.2f})").replace("-", "−")
        f = lambda p: f"{p:.1g}" if p < 0.01 else f"{p:.2f}"
        pp = f"{f(pc.mannwhitney_p)} / {f(pm.mannwhitney_p)}"
    cells = [S.ARM4_SHORT[k] if k != "dep" else "Deposited",
             f"{m(sc.median_dE)} / {m(sm.median_dE)}",
             f"{100*sc.frac_le0:.0f} / {100*sm.frac_le0:.0f}", dl, pp]
    for j, (cx, t) in enumerate(zip(COLX, cells)):
        fig.text(cx, yy_, t, fontsize=5.7, color=S.ARM4_TEXT[k] if j == 0 else C["ink_soft"],
                 ha="left", va="top", fontweight="bold" if (j == 0 and k == "C") else "normal")

ncm = summ("chgnet", "C", "full_novel")
kc, km = KW.loc["chgnet"], KW.loc["mace"]
fig.text(0.020, 0.006,
         "Cells are CHGNet / MACE-MP-0. $\\delta$ = arm minus D (negative: lower "
         "$\\Delta E_{\\mathrm{dec}}$ than D);\n$p$ two-sided Mann–Whitney. "
         f"Kruskal–Wallis over A–D $p$ = {kc.kruskal_p:.4f} / {km.kruskal_p:.3f};\n"
         f"over A, B, D only $p$ = {kc.kruskal_ABD_p:.2f} / {km.kruskal_ABD_p:.2f}. "
         "Only C is shifted; its five lowest are\nknown compounds, and its "
         f"{int(ncm.n_with_reference)} novel formulas have median {m(ncm.median_dE)} (CHGNet).\n"
         "$n$ = 50 per arm resolves only differences of $\\gtrsim$ 20 points in the "
         "$\\leq 0$ fraction.",
         fontsize=5.4, color=C["ink_soft"], ha="left", va="bottom", linespacing=1.45)

S.panel_letter(axes[0][0], "(a)", x=-0.215, y=0.93)
S.panel_letter(axes[1][0], "(b)", x=-0.215, y=0.93)

# the "five lowest are known" sentence is a claim: check it on both potentials
for pot, _ in POT:
    v, kn = vals(pot, "C")
    assert kn[np.argsort(v)[:5]].all(), pot

S.save(fig, "fig4_energetics")
for row in printed:
    print(row)
print("C novel-only CHGNet median", ncm.median_dE, "n", ncm.n_with_reference)
