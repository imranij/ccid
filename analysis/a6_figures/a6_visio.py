"""
a6_visio.py -- Visio-style schematic primitives matching the authors' original
diagrams (ceramic_concept/architecture/method1/method2_rendered.pdf).

Visual grammar copied from those exports (colours sampled from the PDFs):
  * container : white rounded rectangle, #4472C4 outline, bold Times title inside the top
  * pill      : rounded rectangle filled #DEE9F7 with #4472C4 outline, regular Times text
  * highlight : dashed black outline, teal fill #D9EDF3 (the "constraint checker" / "Outputs" groups)
  * frame     : dashed #7E99B3 outline, white fill, regular title at top-left (concept figure)
  * connector : thick #1F497D line with a filled triangular head
Everything is drawn in inch coordinates on an equal-aspect canvas, so corner radii and
arrow heads look identical in every figure.
"""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import a6_style as S

S.apply()
C = S.C

EDGE = C["v_edge"]          # #4472C4
PILL = C["cvae_lt"]         # #DEE9F7
TEAL = C["rand_lt"]         # #D9EDF3
ARROW = C["v_arrow"]        # #1F497D
FRAME = C["v_frame"]        # #7E99B3
TEAL_EDGE = C["rand"]       # #4BACC6, used for random-arm pills

TITLE_PT = 7.6
PILL_PT = 6.9


def canvas(w, h):
    fig = plt.figure(figsize=(w, h))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.set_aspect("equal")
    ax.axis("off")
    return fig, ax


def _rect(ax, x, y, w, h, fc, ec, lw, ls="-", r=0.09, z=1):
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                       fc=fc, ec=ec, lw=lw, ls=ls, zorder=z)
    ax.add_patch(p)
    return p


def container(ax, x, y, w, h, title, bold=True, size=TITLE_PT, title_pad=0.07, sub=""):
    _rect(ax, x, y, w, h, "white", EDGE, 0.9, r=0.10, z=1)
    if title:
        ax.text(x + w / 2, y + h - title_pad, title, ha="center", va="top",
                fontsize=size, fontweight="bold" if bold else "normal",
                color=C["ink"], zorder=4, linespacing=1.15)
    if sub:
        ax.text(x + w / 2, y + h - title_pad - 0.17, sub, ha="center", va="top",
                fontsize=PILL_PT - 0.4, color=C["ink_soft"], zorder=4)


def highlight(ax, x, y, w, h, title="", size=TITLE_PT, title_pad=0.07):
    _rect(ax, x, y, w, h, TEAL, C["ink"], 0.8, ls=(0, (3.0, 1.6)), r=0.08, z=1)
    if title:
        ax.text(x + w / 2, y + h - title_pad, title, ha="center", va="top",
                fontsize=size, fontweight="bold", color=C["ink"], zorder=4,
                linespacing=1.15)


def frame(ax, x, y, w, h, title, sub=""):
    _rect(ax, x, y, w, h, "white", FRAME, 0.8, ls=(0, (4.0, 2.2)), r=0.0, z=0)
    ax.text(x + 0.10, y + h - 0.06, title, ha="left", va="top", fontsize=9.2,
            color=C["ink"], zorder=4)
    if sub:
        ax.text(x + w - 0.10, y + h - 0.08, sub, ha="right", va="top", fontsize=6.9,
                style="italic", color=C["ink_soft"], zorder=4)


def pill(ax, x, y, w, h, text, size=PILL_PT, fc=PILL, ec=EDGE, bold=False,
         color=None, italic=False):
    _rect(ax, x, y, w, h, fc, ec, 0.6, r=0.05, z=2)
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=size,
            fontweight="bold" if bold else "normal",
            style="italic" if italic else "normal",
            color=color or C["ink"], zorder=4, linespacing=1.18)


def pill_rand(ax, x, y, w, h, text, **kw):
    """Random-control pill: Visio teal fill, aqua outline."""
    pill(ax, x, y, w, h, text, fc=TEAL, ec=TEAL_EDGE, **kw)


def pill_arm(ax, x, y, w, h, text, arm, **kw):
    """Four-arm pill: the arm's light tint as fill, the arm colour as outline."""
    pill(ax, x, y, w, h, text, fc=S.ARM4_LT[arm], ec=S.ARM4[arm], **kw)


def stack(ax, x, y_top, w, texts, h=0.24, gap=0.07, fn=pill, **kw):
    """Vertical list of pills starting at y_top (top edge); returns the bottom y."""
    y = y_top
    for t in texts:
        fn(ax, x, y - h, w, h, t, **kw)
        y -= h + gap
    return y + gap


def arrow(ax, p0, p1, lw=1.7, color=ARROW, head=9.0):
    a = FancyArrowPatch(p0, p1, arrowstyle="-|>,head_width=0.32,head_length=0.55",
                        mutation_scale=head, lw=lw, color=color, zorder=3,
                        shrinkA=0, shrinkB=0, joinstyle="miter", capstyle="butt")
    ax.add_patch(a)


def elbow(ax, pts, lw=1.7, color=ARROW, head=9.0):
    """Orthogonal connector through pts; arrow head on the last segment."""
    for (x0, y0), (x1, y1) in zip(pts[:-2], pts[1:-1]):
        ax.plot([x0, x1], [y0, y1], color=color, lw=lw, solid_capstyle="butt", zorder=3)
    arrow(ax, pts[-2], pts[-1], lw=lw, color=color, head=head)


def label(ax, x, y, text, size=7.0, italic=False, bold=False, ha="center",
          color=None):
    ax.text(x, y, text, ha=ha, va="center", fontsize=size,
            style="italic" if italic else "normal",
            fontweight="bold" if bold else "normal",
            color=color or C["ink"], zorder=5)
