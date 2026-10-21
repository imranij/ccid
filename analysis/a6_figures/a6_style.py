"""
a6_style.py -- shared figure style for MTCOMM-D-26-04679 (revision).

One style module imported by every a6 figure script so the set reads as a system.

Design decisions (all deliberate, none seaborn defaults):
  * Serif (Times New Roman; Liberation Serif has identical metrics), matching the
    authors' Visio schematics, at 8 pt base, 7 pt ticks,
    9 pt panel letters.  Figures are built at their FINAL PRINT SIZE, so the
    effective print size equals the nominal size (no rescaling in LaTeX).
  * Colour-blind-safe palette that also survives greyscale: the two arms that
    are compared everywhere (CVAE / random) differ in BOTH hue and lightness
    (L* 27 vs L* 72), so they remain distinguishable in a monochrome print.
  * No chart junk: no top/right spines, no grid unless it carries information,
    no titles inside the image (captions carry them), direct labelling in
    preference to legends wherever it fits.
"""

import os as _os, pathlib as _pl
_CCID_ROOT = _os.environ.get("CCID_ROOT") or str(_pl.Path(__file__).resolve().parents[2])  # repo root
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.path import Path
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, PathPatch
import numpy as np
import os

# ----------------------------------------------------------------------------
# Canvas sizes (inches).  elsarticle 3p is two-column.
#   COL   -> include with width=\columnwidth
#   PAGE  -> include with width=0.85\textwidth   (0.85 * 7.05 in)
# ----------------------------------------------------------------------------
COL = 3.30
PAGE = 6.00

OUTDIR = _CCID_ROOT + "/results/figures"

# ----------------------------------------------------------------------------
# Palette.  Hex, approximate CIE L* in the comment.
# ----------------------------------------------------------------------------
C = {
    # Office theme, matching the authors' Visio schematics (sampled from the
    # original ceramic_*_rendered.pdf exports): outline #4472C4, pill fill
    # #DEE9F7, teal highlight #D9EDF3, arrow / dark text blue #1F497D.
    "ink":      "#000000",   # L*  0   text, as in the Visio figures
    "ink_soft": "#595959",   # L* 38   secondary text, axes
    "rule":     "#BFBFBF",   # L* 77   hairlines
    "pale":     "#F2F2F2",   # L* 95   neutral fills

    "cvae":     "#1F497D",   # L* 31   Office dark blue -- the generative model
    "cvae_lt":  "#DEE9F7",   # L* 92   Visio pill fill
    "rand":     "#4BACC6",   # L* 66   Office aqua      -- the random control
    "rand_lt":  "#D9EDF3",   # L* 93   Visio teal highlight
    "rand_dk":  "#31859C",   # L* 51   aqua for text / strokes on white

    "acc1":     "#C0504D",   # L* 47   Office red    -- emphasis / the bug
    "acc2":     "#77933C",   # L* 57   Office green (dark) -- third series
    "acc3":     "#8064A2",   # L* 47   Office purple -- fourth series
    "grey":     "#7F7F7F",   # L* 53   neutral series

    "v_edge":   "#4472C4",   # Visio container / pill outline
    "v_arrow":  "#1F497D",   # Visio connector
    "v_frame":  "#7E99B3",   # Visio dashed frame
}

# Ordered categorical sequence (monotone-ish in lightness, hue-distinct)
SEQ = [C["cvae"], C["acc1"], C["acc2"], C["acc3"], C["rand"], C["grey"]]

# Named semantic aliases used across figures
ARM = {"cvae": C["cvae"], "random": C["rand"]}
ARM_LT = {"cvae": C["cvae_lt"], "random": C["rand_lt"]}
ARM_LABEL = {"cvae": "Composition-CVAE", "random": "Random control"}

# ----------------------------------------------------------------------------
# Four-arm comparative study (a9).  A ladder of increasing chemical knowledge.
# Colours stay inside the Office theme; greyscale lightness steps down the ladder
# (L* 66 / 53 / 39 / 31), and every figure adds a second, colour-free cue:
# marker shape (MARK4) and line style (LS4).
#   A random            Office aqua, unchanged                     L* 66
#   B element stats     Office grey (Background 1, darker 50 %)    L* 53
#   C ion substitution  Office olive green, darker 50 % (#4F6228); the
#                       darker-25 % #77933C (L* 57) sat within 4 L* of the
#                       grey and was indistinguishable in a monochrome print
#                                                                  L* 39
#   D corrected CVAE    Office dark blue                           L* 31
#   deposited CVAE      reference only: Text-2 lighter 40 %, always dashed
#                       or hollow                                   L* 58
# ----------------------------------------------------------------------------
C.update({
    "arm_A": "#4BACC6", "arm_A_lt": "#D9EDF3",
    "arm_B": "#7F7F7F", "arm_B_lt": "#EDEDED",
    "arm_C": "#4F6228", "arm_C_lt": "#EBF1DE",
    "arm_D": "#1F497D", "arm_D_lt": "#DEE9F7",
    "dep":   "#558ED5",
})
ARMS4 = ["A", "B", "C", "D"]
ARM4 = {k: C[f"arm_{k}"] for k in ARMS4}
ARM4["dep"] = C["dep"]
ARM4_LT = {k: C[f"arm_{k}_lt"] for k in ARMS4}
ARM4_TEXT = dict(ARM4, A=C["rand_dk"])           # aqua is too light for text on white
MARK4 = {"A": "o", "B": "s", "C": "^", "D": "D", "dep": "D"}
LS4 = {"A": (0, (1.0, 1.1)), "B": (0, (3.4, 1.5)),
       "C": (0, (5.0, 1.3, 1.2, 1.3)), "D": "-", "dep": (0, (2.2, 2.2))}
ARM4_LABEL = {"A": "A  Random", "B": "B  Element statistics",
              "C": "C  Ion substitution", "D": "D  Corrected CVAE",
              "dep": "Deposited CVAE (reference)"}
ARM4_SHORT = {"A": "Random", "B": "Element statistics", "C": "Ion substitution",
              "D": "Corrected CVAE", "dep": "Deposited CVAE"}


def apply():
    """Install the rcParams.  Call once at the top of every figure script."""
    mpl.rcParams.update({
        # --- vector output -------------------------------------------------
        "pdf.fonttype": 42,          # TrueType, editable/embeddable
        "ps.fonttype": 42,
        "svg.fonttype": "none",
        "figure.dpi": 150,
        "savefig.dpi": 600,
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.015,
        "savefig.transparent": False,
        "figure.facecolor": "white",
        "axes.facecolor": "white",

        # --- type ----------------------------------------------------------
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Liberation Serif", "Tinos", "DejaVu Serif"],
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "axes.titlesize": 8.0,
        "xtick.labelsize": 7.0,
        "ytick.labelsize": 7.0,
        "legend.fontsize": 7.0,
        "mathtext.fontset": "stix",
        "axes.unicode_minus": True,

        # --- ink -----------------------------------------------------------
        "text.color": C["ink"],
        "axes.labelcolor": C["ink"],
        "axes.edgecolor": C["ink_soft"],
        "xtick.color": C["ink_soft"],
        "ytick.color": C["ink_soft"],
        "xtick.labelcolor": C["ink"],
        "ytick.labelcolor": C["ink"],

        # --- no chart junk --------------------------------------------------
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.linewidth": 0.6,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.major.size": 2.6,
        "ytick.major.size": 2.6,
        "xtick.minor.size": 1.4,
        "ytick.minor.size": 1.4,
        "xtick.direction": "out",
        "ytick.direction": "out",
        "axes.grid": False,
        "grid.color": C["rule"],
        "grid.linewidth": 0.4,
        "grid.alpha": 0.8,

        # --- marks ----------------------------------------------------------
        "lines.linewidth": 1.2,
        "lines.markersize": 3.4,
        "lines.markeredgewidth": 0.6,
        "patch.linewidth": 0.6,
        "hatch.linewidth": 0.5,
        "errorbar.capsize": 1.8,

        # --- legend (used sparingly) -----------------------------------------
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "legend.columnspacing": 1.1,
        "legend.borderaxespad": 0.2,

        "axes.prop_cycle": mpl.cycler(color=SEQ),
    })


# ----------------------------------------------------------------------------
# Small helpers
# ----------------------------------------------------------------------------
def panel_letter(ax, letter, x=-0.16, y=1.04):
    """(a), (b), ... in 9 pt bold, outside the axes."""
    ax.text(x, y, letter, transform=ax.transAxes, fontsize=9,
            fontweight="bold", va="bottom", ha="left", color=C["ink"])


def despine(ax, left=True, bottom=True):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_visible(left)
    ax.spines["bottom"].set_visible(bottom)


def hgrid(ax, axis="y"):
    """Information-carrying reference lines only, always behind the data."""
    ax.grid(True, axis=axis, color=C["rule"], lw=0.4, alpha=0.7, zorder=0)
    ax.set_axisbelow(True)


def label_line(ax, x, y, text, color, dx=0.0, dy=0.0, ha="left", va="center",
               fontsize=7.0, weight="bold"):
    """Direct labelling in place of a legend."""
    return ax.annotate(text, xy=(x, y), xytext=(dx, dy),
                       textcoords="offset points", color=color,
                       fontsize=fontsize, fontweight=weight, ha=ha, va=va,
                       zorder=6)


def box(ax, x, y, w, h, label, fc="white", ec=None, lw=0.8, fontsize=7.0,
        weight="normal", tc=None, radius=0.02, ls="-", zorder=3, va="center"):
    """Rounded schematic box in axes (0-1) coordinates, text centred."""
    ec = ec if ec is not None else C["ink"]
    tc = tc if tc is not None else C["ink"]
    p = FancyBboxPatch((x, y), w, h,
                       boxstyle=f"round,pad=0,rounding_size={radius}",
                       linewidth=lw, edgecolor=ec, facecolor=fc,
                       linestyle=ls, transform=ax.transAxes, zorder=zorder,
                       mutation_aspect=1.0)
    ax.add_patch(p)
    ax.text(x + w / 2, y + h / 2, label, transform=ax.transAxes,
            ha="center", va=va, fontsize=fontsize, fontweight=weight,
            color=tc, zorder=zorder + 1, linespacing=1.35)
    return p


def arrow(ax, p0, p1, color=None, lw=0.9, ls="-", rad=0.0, head=4.5, zorder=2,
          transform=None):
    color = color if color is not None else C["ink_soft"]
    a = FancyArrowPatch(p0, p1,
                        transform=(transform if transform is not None else ax.transAxes),
                        arrowstyle="-|>,head_width=0.30,head_length=0.62",
                        connectionstyle=f"arc3,rad={rad}",
                        mutation_scale=head, linewidth=lw, linestyle=ls,
                        color=color, zorder=zorder, shrinkA=0, shrinkB=0,
                        joinstyle="miter", capstyle="butt")
    ax.add_patch(a)
    return a


def blank_axes(ax):
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, stem, outdir=OUTDIR):
    """Vector PDF + 600 dpi PNG, same basename."""
    os.makedirs(outdir, exist_ok=True)
    pdf = os.path.join(outdir, stem + ".pdf")
    png = os.path.join(outdir, stem + ".png")
    fig.savefig(pdf)
    fig.savefig(png, dpi=600)
    plt.close(fig)
    print(f"  wrote {pdf}")
    print(f"  wrote {png}")
    return pdf, png
