#!/usr/bin/env python3
"""
One full-slide schematic per intervention: what the rule DOES to the network, drawn not described.

These five figures replace a five-column strip (slide_rules.pdf). That strip was legible but each
column was a fifth of a slide carrying three stacked blocks of prose, so the mechanism itself was
never drawn -- a reader learned the rule's NAME and a sentence about it. Here each rule gets a whole
slide, and the slide spends its room on the operation: which units enter the pool, which row is
overwritten, which weight moves where, where the noise enters, what the cost is a function of.

EVERY DETAIL HERE WAS READ OFF THE SOURCE, not off the earlier slides, because the earlier slides
had already dropped the parts that make the rules different from their plausible misreadings:

  dropout    RNN_torch.get_dropout_mask -- the pool is the FIRING units, the drop probability rises
             with firing rank, and each unit is then dropped by its own independent coin. "k units
             crossed out at random from N" is the misreading this picture exists to prevent.
  duplicate  Trainer.prune_and_reinit_, mode="copy" -- N never changes; the donor's incoming row is
             copied into the dormant unit while the donor's outgoing column is HALVED and shared,
             which is why the function survives the operation.
  rescale    Trainer.rescale_rows_ -- positive incoming weights x alpha, negative ones / alpha, with
             the row's length pinned. Weight MOVES from inhibition to excitation; nothing is added.
  synnoise   RNN_torch.forward -- a fresh multiplicative perturbation of a COPY of both weight
             matrices at every timestep, so structural zeros stay zero and the stored weights stay
             clean.
  penalty    Penalties.fr_magnitude_penalty / rec_weights_sparsity_penalty -- a two-sided target on
             activity, and a charge on how many inputs a unit effectively uses.

NO DATA IS PLOTTED HERE. Every one of these is a schematic; the measured results live in the other
fig_* scripts. Shapes that look like curves (the cost curves, the transfer curve) are the actual
functional forms from the source, evaluated here for drawing, not fitted to anything.

TEXT BUDGET. At most 25 words inside any one figure, titles and axis labels included. The deck
supplies the sentence; the panel supplies the picture.

Usage:  python fig_mechanisms.py           (from the repository root)
Output: img/internal_figures/slide_mech_*.pdf (+ .svg)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paperstyle as ps

# Colours come from the shared semantic map so one colour means one thing across the whole deck.
MUTE = ps.COND_COL["mute"]
# `mute` and `dead` are two variants of ONE rule, so `dead` takes a second value of the DROPOUT hue
# and is told apart by fill - hollow against filled - exactly as slide_24_dropout_tradeoff draws it.
# ps.COND_COL["dead"] is the same hex as ps.COND_COL["duplicate"], so drawing `dead` with it handed
# this diagram prune-and-duplicate's colour, five other figures' meaning for that purple.
DEAD = ps.COND_COL["mute"]
DUP = ps.COND_COL["duplicate"]
RESC = ps.COND_COL["rescale"]
SYN = ps.COND_COL["synnoise"]
FRM = ps.COND_COL["frm"]
RWS = ps.COND_COL["rws"]
OFF = ps.FAINT            # a unit or a synapse that is doing nothing
CUT = ps.BAD              # the red X, used only where something is severed
AMBIENT = ps.INK          # noise every network here already runs with: not BASE, not a condition

# The activity target the frm term drives units toward, read off Trainer.py: Penalties.UpV is the
# hard constant 100 (Trainer.py line 31) and cap = cap_fr * log1p(UpV) / log1p(N)
# (Penalties.fr_magnitude_penalty, and Trainer.frm_activity_cap_ which reuses the same expression).
UPV = 100                 # Trainer.UpV, "N units per unit of volume, hard constant"
CAP_FR = 0.3              # frm_args.cap_fr in configs/trainer/trainer.yaml


# -------------------------------------------------------------------------------------------------
# small glyphs shared by the five schematics
# -------------------------------------------------------------------------------------------------

def cross(ax, x, y, size=5.0, col=CUT, lw=1.3, zorder=9):
    """Draw the deck's red X - the mark for a severed connection.

    Args:
        ax: axes; x, y: centre in data coordinates; size: marker size in points;
        col: colour; lw: stroke width; zorder: draw order.
    Returns:
        None.
    """
    ax.plot([x], [y], marker="x", ms=size, mew=lw, color=col, zorder=zorder, clip_on=False)


def squiggle(ax, x0, x1, y, amp, col, n=140, lw=0.8, seed=0, zorder=4):
    """Draw a noise glyph: a short band-limited wiggle along a horizontal segment.

    Args:
        ax: axes; x0, x1: end points in x; y: centre line; amp: wiggle amplitude in y units;
        col: colour; n: samples; lw: stroke width; seed: RNG seed; zorder: draw order.
    Returns:
        None.
    """
    rng = np.random.default_rng(seed)
    x = np.linspace(x0, x1, n)
    v = rng.normal(0, 1, n)
    k = np.exp(-np.linspace(-2, 2, 13) ** 2)
    v = np.convolve(v, k / k.sum(), mode="same")
    v = v / max(np.abs(v).max(), 1e-9)
    ax.plot(x, y + amp * v, lw=lw, color=col, zorder=zorder, clip_on=False)


def rate_trace(ax, x0, w, y, h, col, firing=True, seed=0, lw=0.7, zorder=6, cycles=2.2):
    """Draw a small firing-rate sparkline - the glyph for "this source emits something".

    Args:
        ax: axes; x0: left edge; w: width in x units; y: baseline; h: height in y units;
        col: colour; firing: True draws bumps, False draws a flat line at zero;
        seed: RNG seed; lw: stroke width; zorder: draw order; cycles: bumps across the width,
        kept low for a glyph only a few millimetres wide or it renders as a coil.
    Returns:
        None.
    """
    t = np.linspace(0, 1, 120)
    if firing:
        rng = np.random.default_rng(seed)
        v = np.abs(np.sin(np.pi * cycles * t + 3 * rng.random()))
        v = v * (0.55 + 0.45 * rng.random())
    else:
        v = np.zeros_like(t)
    ax.plot(x0 + t * w, y + v * h, lw=lw, color=col, zorder=zorder, clip_on=False,
            solid_capstyle="round")


def unit(ax, x, y, r, col, filled=True, lw=0.9, zorder=6, ec=None):
    """Draw one unit as a circle: filled if it fires, hollow if it is silent.

    Args:
        ax: axes; x, y: centre; r: radius in x units (circles are drawn with a fixed display
        aspect by using a scatter marker, so r is interpreted as a marker size in points);
        col: colour; filled: solid or hollow; lw: edge width; zorder: draw order;
        ec: edge colour for a hollow circle, default OFF - pass the rule's own colour where the
        hollow circle is a VARIANT of that rule rather than a unit that is merely doing nothing.
    Returns:
        None.
    """
    ax.plot([x], [y], marker="o", ms=r, mfc=col if filled else "none",
            mec=col if filled else (ec or OFF), mew=lw, color="none", zorder=zorder,
            clip_on=False)


# -------------------------------------------------------------------------------------------------
# 1. targeted dropout
# -------------------------------------------------------------------------------------------------

def mech_dropout(name="slide_mech_dropout"):
    """Targeted dropout: who enters the pool, how the coin is weighted, and mute against dead.

    PANEL (a) IS THE WHOLE POINT. The rule is a two-stage funnel, and both stages are routinely
    misremembered. The pool is the FIRING units only - masking a silent unit's column provably
    changes nothing, so sampling over all N merely dilutes the dose - and within the pool the drop
    probability rises with firing rank, after which every unit is dropped by its own independent
    coin. A picture of k units struck out uniformly from N is a different rule.

    THERE IS NO SECOND PANEL ON THE DOSE. One drew two rows of dots, "early" and "late", braced
    "same setting", to say that a share of the LIVE pool is fewer units once fewer units fire. Both
    row lengths were drawn rather than measured, nothing named what a row or a crossed-out dot was,
    and the deck never pointed at it; the dose against training is measured in fig_slides.py
    (slide_24c). The room goes to the funnel and to the two kinds instead.

    PANEL (b) separates the two kinds. `mute` zeroes only the read-out column (RNN_torch.forward),
    so the unit keeps integrating its drive and keeps driving its neighbours; `dead` also carries a
    binary `silence` mask into the unit's own right-hand side, so its dynamics stop.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 96 * ps.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.82],
                  left=0.03, right=0.985, top=0.96, bottom=0.03, hspace=0.26)

    # ---- (a) the funnel -------------------------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[0]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    dx = 0.0145
    dy = ps.square_pitch(ax, dx)
    ps.panel_letter(ax, "a", dx=-0.012, dy=0.99)

    ps.unit_grid(ax, 0.015, 0.90, 25, 100, col=MUTE, off_col="#d8d7d0", side=10,
                 pitch=(dx, dy), s=11, lw=0.6)
    ax.text(0.015 + 4.5 * dx, 0.055, "all units", ha="center", va="bottom", fontsize=7.0,
            color=ps.MUTED)

    ps.arrow(ax, (0.175, 0.52), (0.225, 0.52), col=ps.INK, lw=1.0, mutation_scale=8)

    ps.unit_grid(ax, 0.245, 0.80, 25, 25, col=MUTE, side=5, pitch=(dx, dy), s=11, lw=0.6)
    ax.text(0.245 + 2.0 * dx, 0.055, "firing only", ha="center", va="bottom", fontsize=7.0,
            color=MUTE)

    ps.arrow(ax, (0.335, 0.52), (0.385, 0.52), col=ps.INK, lw=1.0, mutation_scale=8)

    # the ranked drop probability, as bars, with the outcome of one draw underneath
    x0, x1, base, top = 0.46, 0.975, 0.40, 0.86
    n = 25
    xs = np.linspace(x0, x1, n)
    bw = (x1 - x0) / n * 0.62
    p = 0.10 + 0.90 * (np.arange(n) / (n - 1)) ** 1.7
    for xi, pi in zip(xs, p):
        ax.add_patch(plt.Rectangle((xi - bw / 2, base), bw, pi * (top - base),
                                   facecolor=MUTE, edgecolor="none", alpha=0.75, zorder=3))
    ax.plot([x0 - bw, x1 + bw], [base, base], lw=0.6, color=ps.INK, zorder=4)
    ps.arrow(ax, (x0 - 0.022, base), (x0 - 0.022, top), col=ps.INK, lw=0.7, mutation_scale=6,
             shrink=0)
    ax.text(x0 - 0.034, (base + top) / 2, "drop chance", rotation=90, ha="center", va="center",
            fontsize=7.0, color=ps.INK)

    dropped = {24, 22, 19, 14, 8}
    for i, xi in enumerate(xs):
        unit(ax, xi, 0.24, 4.4, MUTE, filled=True, lw=0.6)
        if i in dropped:
            cross(ax, xi, 0.24, size=5.2, lw=1.2)
    ps.arrow(ax, (x0 + 0.062, 0.095), (x1 - 0.055, 0.095), col=ps.MUTED, lw=0.7,
             mutation_scale=6, shrink=0)
    ax.text(x0 - 0.005, 0.095, "quietest", ha="left", va="center", fontsize=6.6, color=ps.MUTED)
    ax.text(x1 + 0.005, 0.095, "busiest", ha="right", va="center", fontsize=6.6, color=ps.MUTED)

    # ---- (b) mute against dead -----------------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[1]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "b", dx=-0.012, dy=0.97)
    ax.text(0.30, 0.95, "network", ha="center", va="center", fontsize=7.2, color=ps.INK)
    ax.text(0.80, 0.95, "read-out", ha="center", va="center", fontsize=7.2, color=ps.INK)

    for y, col, lab, dead in ((0.62, MUTE, "mute", False), (0.20, DEAD, "dead", True)):
        ps.box(ax, 0.20, y - 0.11, 0.20, 0.22, None, col=ps.MUTED, lw=0.8)
        for k, sy in enumerate((y + 0.045, y - 0.065)):
            rate_trace(ax, 0.225, 0.15, sy, 0.055, ps.MUTED, firing=True, seed=7 + k, lw=0.5)
        # hollow, but hollow IN THE DROPOUT COLOUR: `dead` is a variant of this rule, not a unit
        # that happens to be doing nothing, and the fill is the only thing telling the two apart
        unit(ax, 0.555, y, 11.0, col, filled=not dead, lw=1.1, ec=col)
        rate_trace(ax, 0.515, 0.08, y + 0.115, 0.075, col if not dead else OFF,
                   firing=not dead, seed=2, lw=0.7)
        ps.box(ax, 0.72, y - 0.11, 0.16, 0.22, None, col=ps.MUTED if not dead else OFF, lw=0.8)
        ax.text(0.145, y, lab, ha="right", va="center", fontsize=8.0, color=col,
                fontweight="bold")
        # drive in, rate out, read-out
        a_col = OFF if dead else ps.INK
        ps.arrow(ax, (0.41, y + 0.05), (0.525, y + 0.05), col=a_col, lw=1.0, mutation_scale=8)
        ps.arrow(ax, (0.525, y - 0.05), (0.41, y - 0.05), col=a_col, lw=1.0, mutation_scale=8)
        ps.arrow(ax, (0.585, y), (0.71, y), col=OFF, lw=1.0, mutation_scale=8, ls=(0, (2, 1.6)))
        cross(ax, 0.648, y, size=6.0, lw=1.5)
        if dead:
            cross(ax, 0.467, y + 0.05, size=6.0, lw=1.5)
            cross(ax, 0.467, y - 0.05, size=6.0, lw=1.5)
    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# 2. prune and duplicate
# -------------------------------------------------------------------------------------------------

def _weight_image(mag, tint_rows=(), tint_col=(0.0, 0.0, 0.0)):
    """Turn a matrix of weight magnitudes into an RGB image, with named rows re-tinted.

    Args:
        mag: (n, n) array of magnitudes in [0, 1]; tint_rows: row indices to draw in `tint_col`
        instead of grey; tint_col: (r, g, b) in [0, 1], the colour at magnitude 1;
        rows_rgb: unused placeholder kept out of the signature of callers.
    Returns:
        (n, n, 3) float array suitable for imshow.
    """
    mag = np.clip(mag, 0.0, 1.0)
    grey = np.array([0.80, 0.80, 0.82])
    img = 1.0 - mag[:, :, None] * grey[None, None, :]
    if len(tint_rows):
        t = 1.0 - np.asarray(tint_col, float)
        idx = np.asarray(list(tint_rows), int)
        img[idx] = 1.0 - mag[idx][:, :, None] * t[None, None, :]
    return img


def _square_width(ax, h):
    """The width in x data units that renders as a square of height `h` on a 0..1 x 0..1 axes.

    Args:
        ax: axes, already laid out, with both limits spanning one unit; h: height in y units.
    Returns:
        the matching width in x units.
    """
    bb = ax.get_window_extent()
    return h * bb.height / max(bb.width, 1e-9)


def _matrix(ax, mag, x0, y0, w, h, tint_rows=(), tint_col=(0, 0, 0), grid=True):
    """Draw a weight matrix as a cell grid at a given rectangle of a schematic axes.

    Args:
        ax: a blank axes spanning 0..1 in both directions; mag: (n, n) magnitudes in [0, 1];
        x0, y0: lower-left corner; w, h: extent in data units; tint_rows: row indices drawn in
        `tint_col`; tint_col: (r, g, b) in [0, 1]; grid: draw cell separators.
    Returns:
        None.
    """
    ax.imshow(_weight_image(mag, tint_rows=tint_rows, tint_col=tint_col),
              extent=[x0, x0 + w, y0, y0 + h], aspect="auto", zorder=2,
              interpolation="nearest")
    n = mag.shape[0]
    if grid:
        for k in range(1, n):
            ax.plot([x0 + w * k / n] * 2, [y0, y0 + h], lw=0.3, color=ps.PAPER, zorder=3)
            ax.plot([x0, x0 + w], [y0 + h * k / n] * 2, lw=0.3, color=ps.PAPER, zorder=3)
    ax.add_patch(plt.Rectangle((x0, y0), w, h, fill=False, lw=0.7, edgecolor=ps.MUTED, zorder=5))


def mech_duplicate(name="slide_mech_duplicate"):
    """Prune and duplicate: the matrix never changes size, and the row and the column differ.

    PANEL (a) IS THERE TO KILL "a unit is added". `prune_and_reinit_` with mode="copy" overwrites
    one row of W_rec in place and writes one column; the matrix is N x N before and after, so the
    two matrices are drawn at exactly the same size with nothing created and nothing removed.

    PANEL (b) IS THE ASYMMETRY, which is the entire reason the operation preserves the function and
    the part every earlier slide left out. The donor's INCOMING row is copied, so the twin listens
    to the same sources. The donor's OUTGOING column is HALVED and shared, so for any downstream
    unit the pair contributes (w/2) r + (w/2) r = w r, exactly what the donor contributed alone.

    THE SILENT UNIT KEEPS ITS SEAT in both halves of panel (b) and is named in both ("silent", then
    "twin"), so a viewer can follow one particular unit through the operation. Drawn with the lower
    circle labelled only as one of a pair of "twins" afterwards, the picture read as a unit
    appearing out of nowhere, which is the one thing panel (a) is there to rule out.

    THERE IS NO JITTER PANEL. An earlier third panel drew the copied row beside its jittered copy as
    a bare stem plot with no axis of any kind, so nothing told a viewer what was plotted; the jitter
    is a detail of the copy and the room is worth more to the two panels that carry the operation.
    How much jitter the copy gets, and what it does to the result, is measured in
    fig_duplication_audit.py (slide_26f_duplication_noise).

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 116 * ps.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.24, 1.0],
                  left=0.03, right=0.98, top=0.96, bottom=0.04, hspace=0.24)
    dup_rgb = tuple(int(DUP[i:i + 2], 16) / 255.0 for i in (1, 3, 5))

    # ---- (a) the matrix, before and after ------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[0]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()
    ps.panel_letter(ax, "a", dx=-0.012, dy=0.98)

    n = 16
    rng = np.random.default_rng(5)
    mag = np.clip(np.abs(rng.normal(0, 0.40, (n, n))), 0, 1)
    donor, dorm = 4, 11
    mag[dorm, :] *= 0.10                       # the dormant unit's weak incoming row

    after = mag.copy()
    after[dorm, :] = mag[donor, :]             # the donor's incoming row, copied
    after[:, donor] *= 0.5                     # the donor's outgoing column, halved
    after[:, dorm] = after[:, donor]           # and shared with the twin

    h = 0.74
    w = _square_width(ax, h)
    y0 = 0.14
    for bx, m, tint, lab in ((0.14, mag, (donor,), "before"),
                             (0.60, after, (donor, dorm), "after")):
        _matrix(ax, m, bx, y0, w, h, tint_rows=tint, tint_col=dup_rgb)
        ax.text(bx + w / 2, y0 + h + 0.055, lab, ha="center", va="bottom", fontsize=7.6,
                color=ps.INK)
        for r, c in ((donor, DUP), (dorm, DUP if lab == "after" else OFF)):
            yr = y0 + h * (n - 1 - r) / n
            ax.add_patch(plt.Rectangle((bx, yr), w, h / n, fill=False, lw=1.0, edgecolor=c,
                                       zorder=6))
        cols = (donor,) if lab == "before" else (donor, dorm)
        for c in cols:
            xc = bx + w * c / n
            ax.add_patch(plt.Rectangle((xc, y0), w / n, h, fill=False, lw=1.0, edgecolor=DUP,
                                       ls=(0, (1.6, 1.2)), zorder=6))

    # the incoming row is copied
    yd = y0 + h * (n - 0.5 - donor) / n
    ym = y0 + h * (n - 0.5 - dorm) / n
    ps.arrow(ax, (0.14 + w + 0.006, yd), (0.595, ym), col=DUP, lw=1.2, rad=-0.26,
             mutation_scale=9, zorder=7)
    ax.text((0.14 + w + 0.595) / 2, yd + 0.035, "copy", ha="center", va="bottom", fontsize=7.4,
            color=DUP)

    # the outgoing column is halved and shared, braced over the two columns it now occupies
    xa = 0.60 + w * (donor + 0.5) / n
    xb = 0.60 + w * (dorm + 0.5) / n
    yb = y0 - 0.030
    for xc in (xa, xb):
        ps.arrow(ax, (xc, yb - 0.075), (xc, yb), col=DUP, lw=0.8, mutation_scale=6, shrink=0)
    ax.text((xa + xb) / 2, yb - 0.085, "halve", ha="center", va="top", fontsize=7.4, color=DUP)

    # orientation, so a newcomer knows which index is which
    ps.arrow(ax, (0.145, y0 - 0.045), (0.145 + w * 0.45, y0 - 0.045), col=ps.MUTED, lw=0.7,
             mutation_scale=6, shrink=0)
    ax.text(0.145 + w * 0.22, y0 - 0.075, "from unit", ha="center", va="top", fontsize=6.6,
            color=ps.MUTED)
    ps.arrow(ax, (0.126, y0 + h), (0.126, y0 + h * 0.5), col=ps.MUTED, lw=0.7,
             mutation_scale=6, shrink=0)
    ax.text(0.105, y0 + h * 0.75, "to unit", rotation=90, ha="center", va="center", fontsize=6.6,
            color=ps.MUTED)

    # ---- (b) the circuit: row copied, column split ---------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[1]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "b", dx=-0.012, dy=0.97)
    src_y = (0.88, 0.64, 0.40, 0.16)

    for sx, top_col, bot_fill, down_x, tag in ((0.035, DUP, False, 0.385, "before"),
                                               (0.565, DUP, True, 0.915, "after")):
        ux = sx + 0.155
        for sy in src_y:
            unit(ax, sx, sy, 7.0, ps.MUTED, filled=True, lw=0.8)
            ps.arrow(ax, (sx, sy), (ux - 0.010, 0.66), col=ps.MUTED, lw=0.7, mutation_scale=6,
                     shrink=5)
            if bot_fill:
                ps.arrow(ax, (sx, sy), (ux - 0.010, 0.24), col=DUP, lw=0.7, mutation_scale=6,
                         shrink=5)
            else:
                # the silent unit is wired, it simply delivers nothing - drawn, not implied
                ps.arrow(ax, (sx, sy), (ux - 0.010, 0.24), col=OFF, lw=0.4, mutation_scale=4,
                         shrink=5)
        unit(ax, ux, 0.66, 12.5, top_col, filled=True, lw=1.1)
        unit(ax, ux, 0.24, 12.5, DUP if bot_fill else OFF, filled=bot_fill, lw=1.1)
        unit(ax, down_x, 0.45, 11.0, ps.MUTED, filled=True, lw=0.9)
        if bot_fill:
            ps.arrow(ax, (ux + 0.012, 0.63), (down_x - 0.008, 0.48), col=DUP, lw=1.15,
                     mutation_scale=8, shrink=5)
            ps.arrow(ax, (ux + 0.012, 0.27), (down_x - 0.008, 0.42), col=DUP, lw=1.15,
                     mutation_scale=8, shrink=5)
            ax.text(ux + 0.085, 0.46, "split", ha="center", va="center", fontsize=7.4, color=DUP)
            ax.text(ux, 0.98, "donor", ha="center", va="top", fontsize=7.4, color=DUP)
            ax.text(ux, 0.09, "twin", ha="center", va="top", fontsize=7.4, color=DUP)
            ax.text(sx + 0.075, 0.03, "copied", ha="center", va="bottom", fontsize=7.4, color=DUP)
        else:
            ps.arrow(ax, (ux + 0.012, 0.63), (down_x - 0.008, 0.47), col=DUP, lw=2.4,
                     mutation_scale=12, shrink=5)
            ax.text(ux, 0.98, "donor", ha="center", va="top", fontsize=7.4, color=DUP)
            ax.text(ux, 0.09, "silent", ha="center", va="top", fontsize=7.4, color=ps.MUTED)
    ax.plot([0.505, 0.505], [0.04, 0.96], lw=0.6, color=ps.FAINT, zorder=1)
    ps.arrow(ax, (0.445, 0.50), (0.495, 0.50), col=ps.INK, lw=1.0, mutation_scale=8)
    # ONE UNIT, FOLLOWED ACROSS: the silent circle keeps its seat in both halves and is named
    # there, so the rule reads as something done to a particular unit, not as a unit appearing.

    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# 3. synaptic rescaling
# -------------------------------------------------------------------------------------------------

def mech_rescale(name="slide_mech_rescale"):
    """Synaptic rescaling: weight MOVES from inhibition to excitation until the drive crosses zero.

    PANEL (a) IS THE MECHANISM AND THE REASON IT WORKS. A dormant unit is dormant because its net
    drive, the sum over sources of weight times that source's rate, sits at or below zero, and the
    rectifier then emits exactly nothing. Tilting the row raises the drive; the transfer curve is
    what turns "the drive rose" into "the unit fires", so the threshold crossing has to be on the
    page or the panel shows an operation with no consequence.

    PANEL (b) IS WHAT THE RULE IS NOT. Positive incoming weights are multiplied by a factor just
    above one and negative ones divided by it, and then the row is rescaled back to the length it
    had (rescale_rows_, normalize=True). So the excitatory and inhibitory masses trade against each
    other at a constant total: a see-saw. The un-normalised version of exactly this rule pumped
    magnitude every event and diverged, which is why the pinned total is drawn as the invariant.

    PANEL (c) is the restriction to firing sources. Drive is a weighted sum of the sources' rates,
    so a synapse from a unit emitting nothing delivers nothing however large it is made; those
    columns are left exactly alone.

    PANEL (d) is the stop: the unit reaches its activity target, or its boost budget runs out.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 112 * ps.MM))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.62], width_ratios=[1.30, 1.0],
                  left=0.035, right=0.985, top=0.95, bottom=0.04, hspace=0.30, wspace=0.16)

    # ---- (a) drive, threshold, output ----------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[0, 0]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "a", dx=-0.02, dy=0.97)

    srcs = ((0.86, +1, True, 1), (0.63, +1, True, 2), (0.40, -1, True, 3), (0.17, -1, True, 4))
    for sy, sign, firing, sd in srcs:
        unit(ax, 0.035, sy, 7.0, ps.MUTED, filled=firing, lw=0.8)
        rate_trace(ax, 0.072, 0.072, sy - 0.030, 0.070, ps.MUTED, firing=firing, seed=sd, lw=0.6,
                   cycles=1.8)
        col = RESC if sign > 0 else ps.MUTED
        lw = 1.9 if sign > 0 else 0.7
        ps.arrow(ax, (0.165, sy), (0.325, 0.52), col=col, lw=lw, mutation_scale=8, shrink=5)
        ax.text(0.185, sy + (0.035 if sign > 0 else -0.045), "+" if sign > 0 else "−",
                ha="left", va="center", fontsize=9.0, color=col, fontweight="bold")
    ax.text(0.145, 0.975, "excitatory", ha="center", va="top", fontsize=6.8, color=RESC)
    ax.text(0.145, 0.055, "inhibitory", ha="center", va="bottom", fontsize=6.8, color=ps.MUTED)
    unit(ax, 0.355, 0.52, 13.0, OFF, filled=False, lw=1.1)

    # the transfer curve, drawn in the right half of the same drawing surface
    gx0, gx1, gy0, gy1 = 0.52, 0.975, 0.22, 0.86
    h = np.linspace(-1.0, 1.0, 400)
    r = np.maximum(h, 0.0)
    px = gx0 + (h + 1.0) / 2.0 * (gx1 - gx0)
    py = gy0 + r / 1.0 * (gy1 - gy0)
    ax.plot([gx0, gx1], [gy0, gy0], lw=0.7, color=ps.INK, zorder=3)
    ax.plot([gx0, gx0], [gy0, gy1], lw=0.7, color=ps.INK, zorder=3)
    zx = gx0 + 0.5 * (gx1 - gx0)
    ax.plot([zx, zx], [gy0, gy1], lw=0.7, color=ps.FAINT, ls=(0, (2, 2)), zorder=2)
    ax.plot(px, py, lw=1.6, color=ps.INK, zorder=4)
    ax.text((gx0 + gx1) / 2, 0.035, "drive", ha="center", va="bottom", fontsize=7.0,
            color=ps.INK)
    ax.text(gx0 - 0.022, (gy0 + gy1) / 2, "output", rotation=90, ha="center", va="center",
            fontsize=7.0, color=ps.INK)
    ax.text(zx + 0.012, gy0 - 0.02, "0", ha="left", va="top", fontsize=6.6, color=ps.MUTED)

    hb, ha_ = -0.52, 0.34
    # the unit's own drive is the point on the curve, so the two halves of the panel are one claim
    ps.arrow(ax, (0.372, 0.46), (gx0 + (hb + 1) / 2 * (gx1 - gx0) - 0.014, gy0 + 0.004),
             col=OFF, lw=0.8, mutation_scale=7, shrink=4, rad=0.55, zorder=5)
    for hv, col, fill in ((hb, OFF, False), (ha_, RESC, True)):
        bx = gx0 + (hv + 1.0) / 2.0 * (gx1 - gx0)
        by = gy0 + max(hv, 0.0) * (gy1 - gy0)
        unit(ax, bx, by, 7.5, col, filled=fill, lw=1.2)
    ps.arrow(ax, (gx0 + (hb + 1) / 2 * (gx1 - gx0), gy0 + 0.055),
             (gx0 + (ha_ + 1) / 2 * (gx1 - gx0), gy0 + max(ha_, 0) * (gy1 - gy0) + 0.055),
             col=RESC, lw=1.3, rad=-0.30, mutation_scale=9, zorder=6)

    # ---- (b) the total length is pinned --------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[0, 1]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "b", dx=-0.03, dy=0.97)
    bx0, bx1 = 0.18, 0.92
    fracs = (0.34, 0.46, 0.58, 0.70)
    ys = np.linspace(0.80, 0.22, len(fracs))
    hgt = 0.105
    for y, f in zip(ys, fracs):
        xm = bx0 + f * (bx1 - bx0)
        ax.add_patch(plt.Rectangle((bx0, y - hgt / 2), xm - bx0, hgt, facecolor=RESC,
                                   edgecolor="none", zorder=3))
        ax.add_patch(plt.Rectangle((xm, y - hgt / 2), bx1 - xm, hgt, facecolor="none",
                                   edgecolor=ps.MUTED, lw=0.8, hatch="////", zorder=3))
    ax.text(bx0 + 0.06, ys[0] + 0.105, "excitatory", ha="left", va="bottom", fontsize=6.8,
            color=RESC)
    ax.text(bx1 - 0.02, ys[0] + 0.105, "inhibitory", ha="right", va="bottom", fontsize=6.8,
            color=ps.MUTED)
    for x in (bx0, bx1):
        ax.plot([x, x], [ys[0] + 0.085, ys[-1] - 0.10], lw=0.7, color=ps.FAINT,
                ls=(0, (2, 2)), zorder=2)
    ax.plot([bx0, bx0, bx1, bx1], [ys[-1] - 0.085, ys[-1] - 0.115, ys[-1] - 0.115,
                                   ys[-1] - 0.085], lw=0.7, color=ps.INK, zorder=4)
    ax.text((bx0 + bx1) / 2, ys[-1] - 0.155, "total unchanged", ha="center", va="top",
            fontsize=7.0, color=ps.INK)
    ps.arrow(ax, (0.10, ys[0]), (0.10, ys[-1]), col=ps.MUTED, lw=0.8, mutation_scale=7, shrink=0)
    ax.text(0.075, (ys[0] + ys[-1]) / 2, "each step", rotation=90, ha="center", va="center",
            fontsize=7.0, color=ps.MUTED)

    # ---- (c) silent sources are skipped --------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[1, 0]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "c", dx=-0.02, dy=0.95)
    firing = (True, True, False, True, False, True, True, False, True, True)
    rng = np.random.default_rng(17)
    wv = rng.normal(0, 0.36, len(firing))
    xs = np.linspace(0.26, 0.97, len(firing))
    base = 0.50
    ax.plot([0.22, 0.99], [base, base], lw=0.6, color=ps.INK, zorder=3)
    first_skip = None                       # (x, y) of the leftmost skipped synapse's own cross
    for xi, f, wi in zip(xs, firing, wv):
        col = (RESC if wi > 0 else ps.MUTED) if f else OFF
        ax.plot([xi, xi], [base, base + wi * 0.40], lw=3.0, color=col, solid_capstyle="butt",
                zorder=4)
        unit(ax, xi, 0.12, 7.0, ps.MUTED if f else "none", filled=f, lw=0.8)
        tip = base + wi * 0.40
        if f:
            ps.arrow(ax, (xi, tip), (xi, tip + (0.11 if wi > 0 else -0.11)), col=col, lw=0.9,
                     mutation_scale=6, shrink=0)
        else:
            cross(ax, xi, tip, size=5.6, lw=1.3)
            first_skip = (xi, tip) if first_skip is None else first_skip
    ax.text(0.20, base + 0.30, "tilted", ha="right", va="center", fontsize=7.0, color=RESC)
    ax.text(first_skip[0], 0.02, "silent", ha="center", va="bottom", fontsize=7.0, color=ps.MUTED)
    # THE LEADER ENDS ON THE CROSS IT NAMES. Drawn to a fixed height it landed in blank space,
    # because which side of the baseline a skipped synapse sits on depends on its weight's sign.
    ax.annotate("skipped", first_skip, textcoords="offset points", xytext=(-30, 0),
                ha="right", va="center", fontsize=7.0, color=CUT, zorder=6,
                arrowprops=dict(arrowstyle="-", lw=0.5, color=CUT, linestyle=(0, (1.6, 1.4)),
                                shrinkA=1.0, shrinkB=4.0))

    # ---- (d) it stops ---------------------------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[1, 1]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "d", dx=-0.03, dy=0.95)
    ps.box(ax, 0.10, 0.40, 0.34, 0.26, "tilting", col=RESC, lw=0.9, fs=7.4, text_col=RESC)
    ps.arrow(ax, (0.19, 0.67), (0.35, 0.67), col=RESC, lw=0.9, rad=-1.9, mutation_scale=7,
             shrink=0, zorder=4)
    unit(ax, 0.74, 0.70, 12.0, RESC, filled=True, lw=1.1)
    unit(ax, 0.74, 0.22, 12.0, OFF, filled=False, lw=1.1)
    ps.arrow(ax, (0.45, 0.58), (0.705, 0.69), col=RESC, lw=1.0, mutation_scale=8, shrink=4)
    ps.arrow(ax, (0.45, 0.46), (0.705, 0.25), col=ps.MUTED, lw=1.0, mutation_scale=8, shrink=4)
    ax.text(0.80, 0.70, "fires", ha="left", va="center", fontsize=7.4, color=RESC)
    ax.text(0.80, 0.22, "gives up", ha="left", va="center", fontsize=7.4, color=ps.MUTED)
    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# 4. synaptic noise
# -------------------------------------------------------------------------------------------------

def wobbly(ax, p0, p1, amp, col, lw=1.0, seed=0, zorder=5, head=7):
    """Draw a connection whose path flickers - a synapse carrying a fresh perturbation.

    The wobble is band-limited noise added to y and windowed to zero at both ends, so the synapse
    still starts on its source and lands on its target. Amplitude is the caller's job: it is set
    proportional to the weight, which is what makes the perturbation multiplicative on the page.

    Args:
        ax: axes; p0, p1: (x, y) endpoints in data coordinates; amp: wobble amplitude in y units;
        col: colour; lw: stroke width; seed: RNG seed; zorder: draw order; head: arrow-head size.
    Returns:
        None.
    """
    t = np.linspace(0, 1, 180)
    x = p0[0] + t * (p1[0] - p0[0])
    y = p0[1] + t * (p1[1] - p0[1])
    rng = np.random.default_rng(seed)
    v = rng.normal(0, 1, t.size)
    k = np.exp(-np.linspace(-2, 2, 21) ** 2)
    v = np.convolve(v, k / k.sum(), mode="same")
    v = v / max(np.abs(v).max(), 1e-9) * np.sin(np.pi * t)
    ax.plot(x, y + amp * v, lw=lw, color=col, zorder=zorder, clip_on=False,
            solid_capstyle="round")
    ps.arrow(ax, (x[-12], y[-12] + amp * v[-12]), p1, col=col, lw=lw, mutation_scale=head,
             shrink=4, zorder=zorder)


def mech_synnoise(name="slide_mech_synnoise"):
    """Synaptic noise: where the jitter enters, that it is multiplicative, and that it is transient.

    PANEL (a) IS ABOUT THE ROUTE, AND ONLY THE ROUTE. State noise is a current injected at the cell
    body, the same draw whatever a unit is wired to. Synaptic noise is a flicker on each synapse
    that scales with that synapse's weight, so it arrives through the wiring the task is still
    learning. Both reach a weakly wired unit; they differ in how they get there, and the panel is
    two injection sites side by side with the same two units under each.

    THE WEAKLY WIRED UNIT'S SYNAPSES FLICKER TOO, by an amount set from their own small weights.
    Drawn with those synapses thin, grey and straight while the well-wired unit's jittered, the
    panel said state noise reaches a quiet unit and synaptic noise does not - the opposite of the
    section it sits in, and a claim neither the deck nor the data makes. Which rule recruits more
    units is measured in fig_slides.py; it is not asserted by this drawing.

    PANEL (b): the perturbation is w -> w (1 + sigma * normal), so it scales with the weight and
    leaves every structural zero at zero.

    PANEL (c): the draw is fresh at every timestep and applied to a COPY of both matrices
    (RNN_torch.forward), so the trained network never carries the jitter.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 104 * ps.MM))
    # panel (b) carries the claim, so it takes the wider column and the taller row
    gs = GridSpec(2, 2, figure=fig, height_ratios=[0.80, 1.0], width_ratios=[1.55, 1.0],
                  left=0.035, right=0.98, top=0.95, bottom=0.05, hspace=0.26, wspace=0.14)

    # ---- (a) two injection sites ---------------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[0, :]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "a", dx=-0.012, dy=0.99)
    ax.plot([0.495, 0.495], [0.03, 0.99], lw=0.6, color=ps.FAINT, zorder=1)
    # AMBIENT, not ps.BASE: BASE is the unpenalised reference's ink and state noise is a condition
    ax.text(0.23, 0.995, "state noise", ha="center", va="top", fontsize=7.6, color=AMBIENT,
            fontweight="bold")
    ax.text(0.76, 0.995, "synaptic noise", ha="center", va="top", fontsize=7.6, color=SYN,
            fontweight="bold")
    # the route, said once per half, because the route is the only difference the panel draws
    ax.text(0.23, 0.915, "at the cell body", ha="center", va="top", fontsize=6.8, color=ps.MUTED)
    ax.text(0.76, 0.915, "on every synapse", ha="center", va="top", fontsize=6.8, color=ps.MUTED)

    # A WEIGHT'S SIZE SETS BOTH ITS STROKE AND ITS WOBBLE, so the weakly wired unit's synapses
    # flicker at a visible fraction of the well-wired unit's rather than not at all: the jitter is
    # proportional to the weight (w -> w (1 + sigma n)), which is panel (b)'s claim, drawn here on
    # the wiring itself. Thin-against-thick is the weight; jitter against none is the route.
    LW_ON, LW_OFF = 2.1, 0.7                 # stroke widths standing for a large and a small weight
    AMP_ON = 0.030                           # the well-wired synapse's wobble, in y units
    for half, (ox, col) in enumerate(((0.0, AMBIENT), (0.53, SYN))):
        for row, (y, wired) in enumerate(((0.60, True), (0.19, False))):
            ucx = ox + 0.345
            unit(ax, ucx, y, 13.0, ps.MUTED if wired else OFF, filled=wired, lw=1.0)
            lw = LW_ON if wired else LW_OFF
            for k, sy in enumerate((y + 0.125, y, y - 0.125)):
                unit(ax, ox + 0.055, sy, 6.5, ps.MUTED, filled=True, lw=0.7)
                if half == 0:
                    # the wiring is just wiring here: the noise does not travel along it
                    ps.arrow(ax, (ox + 0.055, sy), (ucx - 0.012, y), col=ps.MUTED, lw=lw,
                             mutation_scale=7 if wired else 5, shrink=5)
                else:
                    # every synapse flickers, by an amount proportional to its own weight
                    wobbly(ax, (ox + 0.062, sy), (ucx - 0.014, y),
                           AMP_ON * lw / LW_ON, col, lw=lw, seed=10 * row + k,
                           head=7 if wired else 5)
            if half == 0:
                squiggle(ax, ucx - 0.050, ucx + 0.050, y + 0.195, 0.020, col, lw=0.9,
                         seed=3 + row, zorder=6, n=60)
                ps.arrow(ax, (ucx, y + 0.165), (ucx, y + 0.035), col=col, lw=1.3,
                         mutation_scale=8, shrink=0)

    # ---- (b) multiplicative ---------------------------------------------------------------------
    ax = ps.blank(fig.add_subplot(gs[1, 0]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(ax, "b", dx=-0.012, dy=0.97)
    # THE WEIGHTS ARE ORDERED BY SIZE on purpose: the jitter band is 45% of each weight, so a row
    # drawn largest-first renders the proportionality as a shape that narrows to nothing at the two
    # structural zeros, where the band has no width at all because w (1 + s n) is 0 for w = 0.
    w = np.array([0.88, -0.66, 0.47, -0.34, 0.19, -0.09, 0.0, 0.0])
    xs = np.linspace(0.14, 0.93, w.size)
    base, sc = 0.50, 0.40
    ax.plot([0.07, 0.98], [base, base], lw=0.6, color=ps.INK, zorder=3)
    hw = 0.019
    for xi, wi in zip(xs, w):
        if wi == 0.0:
            unit(ax, xi, base, 4.6, ps.FAINT, filled=True, lw=0.6)
            continue
        tip = base + wi * sc
        band = 0.45 * abs(wi) * sc
        ax.add_patch(plt.Rectangle((xi - hw, tip - band), 2 * hw, 2 * band,
                                   facecolor=SYN, alpha=0.26, edgecolor="none", zorder=3))
        for yc in (tip - band, tip + band):                    # the reach of one fresh draw
            ax.plot([xi - hw, xi + hw], [yc] * 2, lw=0.8, color=SYN, alpha=0.75, zorder=5)
        ax.plot([xi, xi], [base, tip], lw=3.2, color=SYN, solid_capstyle="butt", zorder=4)
    ax.text(0.045, base + 0.26, "weight", rotation=90, ha="center", va="center", fontsize=7.0,
            color=ps.INK)
    ax.plot([xs[-2] - 0.03, xs[-2] - 0.03, xs[-1] + 0.03, xs[-1] + 0.03],
            [base - 0.09, base - 0.12, base - 0.12, base - 0.09], lw=0.7, color=ps.MUTED,
            zorder=4)
    ax.annotate("zero", ((xs[-2] + xs[-1]) / 2, base - 0.12), textcoords="offset points",
                xytext=(0, -3), ha="center", va="top", fontsize=7.0, color=ps.MUTED)

    # ---- (c) a copy is perturbed, the stored matrix is not --------------------------------------
    ax = ps.blank(fig.add_subplot(gs[1, 1]))
    ax.set(xlim=(0, 1), ylim=(0, 1))
    fig.canvas.draw()                      # the card sizes below are measured off the laid-out axes
    ps.panel_letter(ax, "c", dx=-0.03, dy=0.97)
    rng = np.random.default_rng(23)
    m = 10
    mag = np.clip(np.abs(rng.normal(0, 0.36, (m, m))), 0, 1)

    # A WEIGHT MATRIX IS SQUARE, so the cards are sized from the axes' own aspect rather than from
    # a pair of numbers that stop being square the moment the panel changes shape.
    k = _square_width(ax, 1.0)             # y units per x unit of the same display length
    cy, x_cp0, gap = 0.66, 0.44, 0.016
    w_cp = (0.99 - x_cp0 - 2 * gap) / 3
    h_cp = w_cp / max(k, 1e-9)
    h_st = h_cp * 1.16
    w_st = h_st * k
    ax.imshow(_weight_image(mag), extent=[0.02, 0.02 + w_st, cy - h_st / 2, cy + h_st / 2],
              aspect="auto", zorder=2, interpolation="nearest")
    ax.add_patch(plt.Rectangle((0.02, cy - h_st / 2), w_st, h_st, fill=False, lw=0.8,
                               edgecolor=ps.INK, zorder=4))
    ax.text(0.02 + w_st / 2, cy - h_st / 2 - 0.035, "stored", ha="center", va="top", fontsize=7.2,
            color=ps.INK)
    syn_rgb = tuple(int(SYN[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    for j in range(3):
        bx = x_cp0 + j * (w_cp + gap)
        pert = np.clip(mag * (1.0 + 0.55 * np.random.default_rng(40 + j).normal(0, 1, mag.shape)),
                       0, 1)
        ax.imshow(_weight_image(pert, tint_rows=range(m), tint_col=syn_rgb),
                  extent=[bx, bx + w_cp, cy - h_cp / 2, cy + h_cp / 2], aspect="auto", zorder=2,
                  interpolation="nearest")
        ax.add_patch(plt.Rectangle((bx, cy - h_cp / 2), w_cp, h_cp, fill=False, lw=0.7,
                                   edgecolor=SYN, zorder=4))
    ps.arrow(ax, (0.03 + w_st, cy + 0.045), (x_cp0 - 0.015, cy + 0.045), col=SYN, lw=1.0,
             mutation_scale=8, shrink=0)
    ax.text((0.03 + w_st + x_cp0) / 2, cy + 0.07, "copy", ha="center", va="bottom", fontsize=7.2,
            color=SYN)
    # nothing flows back: the trained network never carries the jitter
    ps.arrow(ax, (x_cp0 - 0.015, cy - 0.075), (0.03 + w_st, cy - 0.075), col=OFF, lw=0.9,
             mutation_scale=7, shrink=0, ls=(0, (2, 1.8)))
    cross(ax, (0.03 + w_st + x_cp0) / 2, cy - 0.075, size=6.0, lw=1.5)
    ps.arrow(ax, (x_cp0, cy - h_cp / 2 - 0.05), (0.99, cy - h_cp / 2 - 0.05), col=ps.MUTED,
             lw=0.7, mutation_scale=6, shrink=0)
    ax.text((x_cp0 + 0.99) / 2, cy - h_cp / 2 - 0.075, "each step", ha="center", va="top",
            fontsize=7.0, color=ps.MUTED)
    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# 5. the two penalty terms
# -------------------------------------------------------------------------------------------------

def _profile(ax, x0, w, y0, total, n, col, seed):
    """Draw one unit's incoming weight profile as n bars whose heights sum to `total`.

    Pinning the SUM is what makes two profiles drawn with this function carry the same total
    weight, and it is why the ratio between them is not a drawing choice: with the same sum over
    15 bars and over 200, the concentrated profile is about thirteen times taller bar for bar, so
    the spread one is genuinely a sliver.

    Args:
        ax: a blank axes spanning 0..1; x0: left edge; w: width in x units; y0: baseline;
        total: the sum of all bar heights, in y units; n: number of inputs; col: colour;
        seed: RNG seed.
    Returns:
        None.
    """
    rng = np.random.default_rng(seed)
    amp = np.abs(rng.normal(0, 1, n)) + 0.15
    amp = amp / amp.sum() * total
    xs = np.linspace(x0, x0 + w, n)
    lw = 2.0 if n < 40 else 0.5
    for xi, ai in zip(xs, amp):
        ax.plot([xi, xi], [y0, y0 + ai], lw=lw, color=col, solid_capstyle="butt", zorder=5)


def frm_target(n):
    """The activity the frm term drives a unit toward, for a network of `n` units.

    Read off the source rather than redescribed: `Penalties.fr_magnitude_penalty` builds the cap as
    `cap_fr * log1p(UpV) / log1p(N)` and `Trainer.frm_activity_cap_` reuses the same expression as
    the maturity test, so one `lambda_frm` setting asks a smaller peak rate of a larger network.

    Args:
        n: network size in units, a scalar or an array.
    Returns:
        the target activity in the units of the soft-max-over-time rate, same shape as `n`.
    """
    return CAP_FR * np.log1p(UPV) / np.log1p(n)


def mech_penalty(name="slide_mech_penalty"):
    """The two loss terms: a two-sided activity target, and a charge on how many inputs are used.

    PANEL (a) IS A TARGET, NOT A FLOOR, and drawing it as a floor is the error this panel exists to
    prevent. `fr_magnitude_penalty` charges (under / cap)^5 below the cap AND (over / cap)^5 above
    it, so the cost rises on both sides and the busy units are dragged down as hard as the quiet
    ones are pushed up. That two-sidedness is how the term flattens the population.

    PANEL (b) IS BLIND TO WEIGHT SIZE, and both weight profiles are therefore drawn in neutral ink:
    the gold belongs to the cost curve, because the gold IS the penalty. With the 200-input profile
    in gold the panel read as if the penalty had produced those 200 inputs. `rec_weights_sparsity_penalty` charges on a row's effective
    support, the squared ratio of its first to its second norm, which counts how many inputs the
    unit uses and is unchanged if the whole row is scaled. Two rows carrying the same total weight
    are therefore charged completely differently, and a picture of weights shrinking toward zero
    would be drawing weight decay instead.

    PANEL (c) IS THE SETTING'S HIDDEN DEPENDENCE ON SIZE. The activity target is not a constant: it
    is cap_fr * log1p(UpV) / log1p(N), so one lambda_frm asks a peak rate of 0.30 of a network of
    100 units and 0.15 of one of 10,000. The earlier version of this panel drew the rectifier with a
    unit stuck on its flat arm, which is the whole claim of the zero-gradient slide (wh1) and was
    being told twice.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 76 * ps.MM))
    gs = GridSpec(1, 3, figure=fig, width_ratios=[1.0, 1.26, 0.82],
                  left=0.055, right=0.985, top=0.92, bottom=0.15, wspace=0.34)

    # ---- (a) a two-sided target on activity ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    cap = 1.0
    a = np.linspace(0, 2.3, 600)
    cost = (np.maximum(cap - a, 0) / cap) ** 5 + (np.maximum(a - cap, 0) / cap) ** 5
    ax.plot(a, cost, lw=1.7, color=FRM, zorder=4)
    ax.axvline(cap, color=ps.FAINT, lw=0.8, ls=(0, (2, 2)), zorder=2)
    for av, dxs in ((0.30, +1), (1.80, -1)):
        cv = (max(cap - av, 0) / cap) ** 5 + (max(av - cap, 0) / cap) ** 5
        ax.plot([av], [cv], "o", ms=5.0, color=FRM, mec="none", zorder=6)
        ps.arrow(ax, (av, cv), (av + dxs * 0.40, cv - 0.25 * cv - 0.04), col=FRM, lw=1.2,
                 mutation_scale=8, zorder=6, shrink=3)
    ax.set(xlim=(0, 2.3), ylim=(-0.05, 1.25), xlabel="a unit's activity", ylabel="cost")
    ax.set_xticks([cap])
    ax.set_xticklabels(["target"])
    ax.set_yticks([])
    ps.panel_letter(ax, "a")

    # ---- (b) how many inputs the unit uses -----------------------------------------------------
    sub = gs[0, 1].subgridspec(2, 1, height_ratios=[0.62, 1.0], hspace=0.30)
    axp = ps.blank(fig.add_subplot(sub[0]))
    axp.set(xlim=(0, 1), ylim=(0, 1))
    ps.panel_letter(axp, "b", dx=-0.10, dy=0.92)
    # BOTH profiles are neutral ink. The gold is the penalty, and the penalty is the cost curve
    # below -- painting the 200-input profile gold said the opposite of what the curve says.
    for x0, n, col, sd in ((0.05, 15, ps.MUTED, 3), (0.56, 200, ps.MUTED, 4)):
        _profile(axp, x0, 0.39, 0.40, 3.0, n, col, sd)
        axp.plot([x0, x0 + 0.39], [0.28, 0.28], lw=3.4, color=col, solid_capstyle="butt",
                 zorder=5)
    axp.plot([0.05, 0.05, 0.95, 0.95], [0.19, 0.13, 0.13, 0.19], lw=0.7, color=ps.INK, zorder=4)
    axp.text(0.50, 0.09, "same total weight", ha="center", va="top", fontsize=7.0, color=ps.INK)

    ax = fig.add_subplot(sub[1])
    tg = 20.0
    s = np.linspace(0, 235, 700)
    c = (np.maximum(s - tg, 0) / tg) ** 2
    ax.plot(s, c, lw=1.7, color=RWS, zorder=4)
    ax.axvline(tg, color=ps.FAINT, lw=0.8, ls=(0, (2, 2)), zorder=2)
    for sv in (15.0, 200.0):
        cv = (max(sv - tg, 0) / tg) ** 2
        ax.plot([sv], [cv], "o", ms=5.5, color=ps.MUTED, mec="none", zorder=6, clip_on=False)
    ax.set(xlim=(0, 235), ylim=(-4, 110), xlabel="inputs a unit uses", ylabel="cost")
    ax.set_xticks([tg, 200])
    ax.set_yticks([])

    # ---- (c) the target itself shrinks as the network grows ------------------------------------
    ax = fig.add_subplot(gs[0, 2])
    nn = np.logspace(2, 4, 400)
    ax.plot(nn, frm_target(nn), lw=1.7, color=FRM, zorder=4)
    for nv in (1000, 4000):
        tv = frm_target(nv)
        ax.plot([nv], [tv], "o", ms=5.0, color=FRM, mec="none", zorder=6)
        ax.annotate(f"{tv:.2f}", (nv, tv), textcoords="offset points", xytext=(3, 4),
                    ha="left", va="bottom", fontsize=6.8, color=FRM)
    ax.set_xscale("log")
    ax.set(xlim=(100, 10000), ylim=(0, 0.34), xlabel="network size", ylabel="target activity")
    ax.set_xticks([100, 1000, 10000])
    ax.set_xticklabels(["100", "1000", "10000"])
    ax.set_yticks([0, 0.3])
    ps.panel_letter(ax, "c", dx=-0.13)
    return ps.save(fig, name)


def main():
    """Write every mechanism schematic. Returns the list of output paths."""
    out = []
    for fn in (mech_dropout, mech_duplicate, mech_rescale, mech_synnoise, mech_penalty):
        out.append(fn())
    return out


if __name__ == "__main__":
    main()
