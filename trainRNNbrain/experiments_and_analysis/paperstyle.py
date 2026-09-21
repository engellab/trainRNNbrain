#!/usr/bin/env python3
"""
House style and schematic primitives for the MANUSCRIPT figures (fig_paper_*.py).

Distinct from `plotstyle.py` on purpose. `plotstyle.py` is the style for internal diagnostics:
screen-sized, grid on, 9pt, one variable one visual channel. That is the right style for a figure
whose only reader is us. A manuscript figure has a different job and therefore a different style:

  - it is printed at a fixed physical width (Nature Communications: 88 mm single, 180 mm double)
    and must stay legible after reduction, so sizes are specified in millimetres and the base font
    is 7 pt, the journal's recommended minimum;
  - it must EXPLAIN, not merely plot. A row of dot plots states a result without saying what the
    intervention was or why the number moves. So this module carries schematic primitives -
    unit-grid pictograms, circuit glyphs, annotated arrows - and every manuscript figure spends its
    top row on them;
  - grid lines are off by default; a manuscript axes earns its ink.

THE PALETTE IS VALIDATED, NOT CHOSEN. The five categorical slots below pass all five checks of the
dataviz validator at the light surface (#fcfcfb):

    node scripts/validate_palette.js "#2a78d6,#d94f2b,#12916a,#8b46d6,#a67c00" --mode light
    PASS lightness band | PASS chroma floor | PASS CVD separation (worst adjacent dE 10.3 deutan,
    11.5 tritan) | PASS normal-vision floor (dE 27.3) | PASS contrast vs surface (all >= 3:1)

An earlier version of this palette failed the contrast check on its green and gold; those two slots
were darkened until it passed. Do not substitute colours by eye - re-run the validator.

BASELINE IS NOT A SERIES. The unpenalised network is the reference every remedy is measured
against, so it is drawn in neutral ink (`BASE`), never in a categorical slot. `BASE` deliberately
fails the chroma floor: that is what makes it read as "reference", not "condition 5".

Self-check: `python paperstyle.py` - writes a swatch-and-primitives sheet to img/internal_figures.
"""

import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

from common import IMG_DIR

MM = 1.0 / 25.4                 # millimetres -> inches, the unit the journal specifies widths in
W1, W2 = 88 * MM, 180 * MM      # single- and double-column widths

# Neutral ink. Text greys and the reference condition.
INK, MUTED, FAINT, GRID = "#0b0b0b", "#6b6963", "#a8a69e", "#e1e0d9"
BASE = "#7a7a72"                # the unpenalised reference - neutral by design, not a series
PAPER = "#fcfcfb"               # the surface the palette was validated against

# The five validated categorical slots, in fixed assignment order. A sixth condition does not get a
# generated hue; it gets a facet or folds into "other".
SLOTS = ["#2a78d6", "#d94f2b", "#12916a", "#8b46d6", "#a67c00"]

# Semantic assignment used across every manuscript figure, so one colour means one thing everywhere.
COND_COL = {
    "baseline":  BASE,
    "none":      BASE,
    "dropout":   SLOTS[0],
    "mute":      SLOTS[0],
    "dead":      SLOTS[3],
    "frm":       SLOTS[1],
    "rws":       SLOTS[4],
    "both":      SLOTS[2],
    "frm + rws": SLOTS[2],
}

# Semantic non-categorical colours. `GOOD`/`BAD` mark a verdict (recovered / failed) and are only
# ever used with an accompanying word, never as the sole encoding.
GOOD, BAD = "#12916a", "#c0392b"


def setup():
    """Apply the manuscript rcParams. Call once at the top of every fig_paper_*.py."""
    plt.rcParams.update({
        "figure.dpi": 110,
        "savefig.dpi": 400,                # journals want >= 300 dpi for raster panels
        "savefig.facecolor": PAPER,
        "figure.facecolor": PAPER,
        "font.size": 7,
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "axes.titlesize": 7.5,
        "axes.labelsize": 7,
        "axes.linewidth": 0.6,
        "axes.edgecolor": "#3a3a36",
        "axes.facecolor": PAPER,
        "axes.grid": False,                # a manuscript axes earns its ink
        "axes.spines.top": False,
        "axes.spines.right": False,
        "grid.alpha": 0.22,
        "grid.linewidth": 0.5,
        "xtick.labelsize": 6.5,
        "ytick.labelsize": 6.5,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.major.size": 2.4,
        "ytick.major.size": 2.4,
        "legend.fontsize": 6.5,
        "legend.frameon": False,
        "legend.handlelength": 1.4,
        "legend.handletextpad": 0.5,
        "legend.labelspacing": 0.3,
        "lines.linewidth": 1.1,
        "lines.markersize": 3.2,
        "errorbar.capsize": 1.8,
        "pdf.fonttype": 42,                # embed as TrueType so editors can select the text
        "ps.fonttype": 42,
    })


def panel_letter(ax, letter, dx=-0.085, dy=1.045, size=9):
    """Put a bold panel letter in the axes' top-left corner, outside the plot box.

    Args:
        ax: axes; letter: the label, e.g. 'a'; dx, dy: position in axes coordinates;
        size: font size in points.
    Returns:
        the Text artist.
    """
    return ax.text(dx, dy, letter, transform=ax.transAxes, fontsize=size,
                   fontweight="bold", va="bottom", ha="left", color=INK)


def despine(ax, keep=("left", "bottom")):
    """Hide every spine not named in `keep`.

    Args:
        ax: axes; keep: spine names to leave visible.
    Returns:
        None.
    """
    for side, sp in ax.spines.items():
        sp.set_visible(side in keep)


def ygrid(ax, alpha=0.22):
    """Turn on a recessive horizontal-only grid, behind the data.

    Horizontal only because every quantitative panel in this paper reads a value off the y axis;
    vertical lines would add ink without adding a reading.

    Args:
        ax: axes; alpha: grid opacity.
    Returns:
        None.
    """
    ax.set_axisbelow(True)
    ax.yaxis.grid(True, alpha=alpha, linewidth=0.5, color=GRID)
    ax.xaxis.grid(False)


# --------------------------------------------------------------------------------------------
# schematic primitives - the reason this module exists
# --------------------------------------------------------------------------------------------

def unit_grid(ax, x0, y0, n_on, n_total=100, col="#2a78d6", off_col="#d8d7d0",
              side=None, pitch=1.0, s=7.0, lw=0.55, zorder=3):
    """Draw a square grid of unit glyphs with `n_on` of them filled - a pictogram of a live fraction.

    This is the paper's recurring visual idiom: the network as 100 dots, filled to the measured
    active fraction. It puts a number the reader must otherwise decode from an axis into a shape
    they read at a glance, and it is the same picture as the quantitative panel below it.

    `pitch` takes a pair because these grids are drawn on schematic axes whose x and y ranges are
    unrelated (a blank axes is usually 0..1 in x and 0..1 in y over a panel that is far from
    square). A single pitch in data units then renders the "square" grid as a row of tall columns,
    which is what happened the first time this was used. Pass (dx, dy) chosen so that dx/dy matches
    the panel's data-to-display aspect, or use `square_pitch` to compute it.

    Args:
        ax: axes; x0, y0: top-left corner in data coordinates; n_on: filled dots;
        n_total: dots in the grid (100 gives one dot per percent); col: fill colour for live units;
        off_col: edge colour for silent units (they are drawn hollow, never omitted - a silent unit
            is present in the network and that is the whole point);
        side: dots per row, default sqrt(n_total); pitch: spacing in data units, a scalar or
            (dx, dy); s: marker area; lw: edge width; zorder: draw order.
    Returns:
        (width, height) of the drawn grid in data units.
    """
    side = int(round(n_total ** 0.5)) if side is None else side
    dx, dy = (pitch, pitch) if np.isscalar(pitch) else pitch
    n_on = int(np.clip(round(n_on), 0, n_total))
    for d in range(n_total):
        r, c = divmod(d, side)
        on = d < n_on
        ax.scatter(x0 + c * dx, y0 - r * dy, s=s,
                   color=col if on else "none",
                   edgecolor=col if on else off_col, linewidth=lw, zorder=zorder)
    rows = int(np.ceil(n_total / side))
    return (side - 1) * dx, (rows - 1) * dy


def square_pitch(ax, dx):
    """The y pitch that makes a grid of x-pitch `dx` look square on this axes.

    Args:
        ax: axes, already given its final xlim/ylim and position; dx: pitch along x in data units.
    Returns:
        dy in data units such that dx and dy span the same number of display points.
    """
    bb = ax.get_window_extent()
    (x0, x1), (y0, y1) = ax.get_xlim(), ax.get_ylim()
    px_per_x = bb.width / max(abs(x1 - x0), 1e-12)
    px_per_y = bb.height / max(abs(y1 - y0), 1e-12)
    return dx * px_per_x / max(px_per_y, 1e-12)


def arrow(ax, xy_from, xy_to, col=MUTED, lw=0.9, style="-|>", rad=0.0, ls="-", zorder=2,
          mutation_scale=7, shrink=1.5):
    """Draw an annotated connector between two points in data coordinates.

    CURVATURE. `rad` bows the arc to the LEFT of the direction of travel, so for a pair of opposite
    connectors between the same two nodes one sign gives two arcs on opposite sides (what you want)
    and the endpoints must be the node CENTRES. Offsetting the endpoints perpendicular to the line
    and then bowing produces a pinched bowtie whose heads land on the nodes - which is what the
    first version of Figure 2a did. Give `shrink` in points, large enough to clear the node glyph.

    Args:
        ax: axes; xy_from, xy_to: (x, y) endpoints; col: colour; lw: line width;
        style: arrow style; rad: curvature (0 straight; bows left of travel); ls: line style;
        zorder: draw order; mutation_scale: arrow-head size; shrink: points trimmed from BOTH ends,
            so a connector between node centres stops clear of the glyphs.
    Returns:
        the FancyArrowPatch.
    """
    p = FancyArrowPatch(xy_from, xy_to, arrowstyle=style, lw=lw, color=col, zorder=zorder,
                        linestyle=ls, mutation_scale=mutation_scale,
                        connectionstyle=f"arc3,rad={rad}", shrinkA=shrink, shrinkB=shrink)
    ax.add_patch(p)
    return p


def box(ax, x, y, w, h, label=None, col=MUTED, face="none", lw=0.7, fs=6.5, pad=0.02,
        text_col=None, zorder=2, ls="-"):
    """Draw a rounded box, optionally with a centred label - the unit of a circuit schematic.

    Args:
        ax: axes; x, y: lower-left corner; w, h: size; label: centred text or None;
        col: edge colour; face: fill colour; lw: edge width; fs: label font size;
        pad: corner rounding; text_col: label colour, defaults to `col`; zorder: draw order;
        ls: edge line style.
    Returns:
        the FancyBboxPatch.
    """
    p = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={pad}",
                       linewidth=lw, edgecolor=col, facecolor=face, zorder=zorder, linestyle=ls)
    ax.add_patch(p)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=fs,
                color=text_col or col, zorder=zorder + 1)
    return p


def blank(ax):
    """Turn an axes into a bare drawing surface for a schematic: no ticks, no spines, no grid."""
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)
    ax.grid(False)
    return ax


def strip(ax, xs, groups, cols, width=0.24, jitter=0.055, rng=None, mean_lw=1.6, ms=3.0,
          alpha=0.85, zorder=3):
    """Per-seed dots with a mean bar - the project's standard for "n seeds per condition".

    Every seed is drawn. A bar chart of means over n = 3..7 hides exactly the thing a reader needs
    to judge (the spread, and whether one outlier carries the effect), and this project has already
    retracted one claim that rested on the worst of seven seeds being quoted as the result.

    Args:
        ax: axes; xs: one x position per group; groups: list of 1-D arrays of per-seed values;
        cols: one colour per group; width: half-width of the mean bar; jitter: x jitter sd;
        rng: np.random.Generator for reproducible jitter (seeded by the caller); mean_lw: mean bar
        width; ms: marker size; alpha: dot opacity; zorder: draw order.
    Returns:
        list of (mean, sd, n) per group.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    out = []
    for x, g, c in zip(xs, groups, cols):
        g = np.asarray(g, float)
        g = g[np.isfinite(g)]
        if not len(g):
            out.append((np.nan, np.nan, 0))
            continue
        ax.plot(x + rng.normal(0, jitter, len(g)), g, "o", ms=ms, color=c, alpha=alpha,
                mec="none", zorder=zorder, clip_on=False)
        ax.plot([x - width, x + width], [g.mean()] * 2, "-", lw=mean_lw, color=c,
                zorder=zorder + 1, solid_capstyle="butt")
        out.append((float(g.mean()), float(g.std(ddof=1)) if len(g) > 1 else 0.0, len(g)))
    return out


def save(fig, name, w_mm=180, h_mm=None):
    """Save a manuscript figure at a physical width, into img/internal_figures.

    Args:
        fig: the figure; name: file stem; w_mm: printed width in millimetres (88 or 180);
        h_mm: printed height, or None to keep whatever the figure was created with.
    Returns:
        the written path.
    """
    if h_mm is not None:
        fig.set_size_inches(w_mm * MM, h_mm * MM)
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"{name}.png")
    fig.savefig(out, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"wrote {out}")
    return out


def _self_check():
    """Draw every primitive once and assert the palette invariants. Raises on failure."""
    assert len(SLOTS) == len(set(SLOTS)), "categorical slots must be distinct"
    assert COND_COL["baseline"] == BASE, "baseline must be neutral ink, not a categorical slot"
    assert BASE not in SLOTS, "the reference colour must not double as a series colour"

    setup()
    fig, axes = plt.subplots(1, 3, figsize=(W2, 38 * MM))

    ax = blank(axes[0])
    for i, c in enumerate(SLOTS + [BASE]):
        ax.add_patch(Rectangle((i, 0), 0.82, 1, color=c))
        ax.text(i + 0.41, -0.22, c, ha="center", fontsize=5, color=MUTED, rotation=90, va="top")
    ax.set(xlim=(-0.3, len(SLOTS) + 1.2), ylim=(-1.4, 1.3), title="validated slots + reference ink")

    ax = blank(axes[1])
    ax.set(xlim=(0, 1), ylim=(0, 1), title="unit_grid: 26% vs 97% alive")
    fig.canvas.draw()                                  # square_pitch needs a laid-out axes
    dx = 0.035
    dy = square_pitch(ax, dx)
    assert dy > 0, "square_pitch must return a positive pitch"
    w, _ = unit_grid(ax, 0.04, 0.92, 26, 100, col=SLOTS[0], pitch=(dx, dy))
    unit_grid(ax, 0.04 + w + 0.12, 0.92, 97, 100, col=SLOTS[1], pitch=(dx, dy))

    ax = blank(axes[2])
    box(ax, 0.05, 0.55, 0.3, 0.3, "unit", col=SLOTS[2])
    box(ax, 0.65, 0.55, 0.3, 0.3, "read-out", col=SLOTS[3])
    arrow(ax, (0.36, 0.70), (0.64, 0.70), col=INK)
    arrow(ax, (0.36, 0.60), (0.64, 0.50), col=BAD, ls=":", rad=-0.3)
    ax.set(xlim=(0, 1), ylim=(0.2, 1.05), title="box / arrow")

    p = save(fig, "paperstyle_selfcheck")
    assert os.path.exists(p)

    rng = np.random.default_rng(1)
    fig, ax = plt.subplots(figsize=(W1, 40 * MM))
    res = strip(ax, [0, 1], [rng.normal(263, 13, 7), rng.normal(362, 8, 7)], [BASE, SLOTS[0]])
    assert res[0][2] == 7 and np.isfinite(res[0][0]), "strip must report mean, sd and n"
    plt.close(fig)

    # strip must survive an empty group rather than raising - some cells are still training
    fig, ax = plt.subplots()
    res = strip(ax, [0, 1], [np.array([]), np.array([1.0, 2.0])], [BASE, SLOTS[0]])
    assert res[0][2] == 0 and res[1][2] == 2
    plt.close(fig)

    print("paperstyle.py self-check passed")


if __name__ == "__main__":
    _self_check()
