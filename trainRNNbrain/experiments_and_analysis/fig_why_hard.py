#!/usr/bin/env python3
"""
The talk's "why is this hard?" section: three pictures for the three obstacles, and nothing else.

The deck used to go straight from "most units never fire" to a list of things we tried. An audience
told only that the units are silent has no reason to expect fixing them to be difficult, so the
remedies arrived looking like a shopping list rather than like answers to a problem. These three
figures supply the problem.

  WH1  a silent unit is FROZEN, not merely quiet       -- the gradient on its weights is exactly 0
  WH2  switch one back on and the network switches it off again
  WH3  what part of a working unit does the recruiting

WH1's gradients are produced by autograd on a network wired silent, not asserted; the same premise
is checked by `tests/test_prune_and_reinit.py::test_dead_units_have_exactly_zero_gradient`, which
passes. WH2 and WH3 read measured counts out of `why_hard_cache.py` -- nothing on either figure is
typed in, and a number the cache cannot produce is not drawn.

TEXT IS CAPPED AND THE CAP IS ENFORCED. Every figure here is checked by `count_words`, which walks
the rendered figure's text artists and counts; `main` fails rather than writing a wordy panel. The
limit is 25 words per figure including titles, tick labels and annotations, because the brief for
this section is that the pictures carry the claims and the deck carries the sentences.

COLOUR. Two tints of one hue, the duplication hue, because the rules in WH2 and WH3 are the same
mechanism told apart by what the revived unit is handed: the dark tint (ps.COND_COL["rescale"]) is
every rule that rewrites the silent unit's OWN weights, which is what fails, and the light tint
(ps.COND_COL["duplicate"]) is handing it a working unit's wiring, which works. The untouched
network is ps.BASE, neutral, never a series colour.

Usage:  python fig_why_hard.py [--refresh]
Output: img/internal_figures/slide_wh_*.pdf (+ .svg)
"""

import argparse
import os
import re
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Ellipse, Rectangle

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import paperstyle as ps
import why_hard_cache as whc

MAX_WORDS = 25                      # the brief's hard limit on text inside one figure

# ⚠️ THE ZERO-GRADIENT FIGURE USES NO CONDITION COLOUR. It comes before any intervention exists and
# its two colours mean "this unit fires" and "this unit does not", which is not a condition. Drawn in
# ps.SLOTS[0] the firing unit wore the dropout hue, so the one slide in the deck that is purely about
# the rectifier borrowed a remedy's colour. Plain ink instead.
FIRING = ps.INK
OWN = ps.COND_COL["rescale"]        # rules that rewrite the silent unit's own weights -- they fail
DONOR = ps.COND_COL["duplicate"]    # handing it a working unit's wiring -- this works
REST = ps.BASE                      # the untouched network: neutral reference, never a series


def cache():
    """The measured numbers behind WH2 and WH3, or None if they have not been built.

    Returns:
        dict of arrays from why_hard_cache.py, or None.
    """
    if not os.path.exists(whc.OUT):
        print(f"  (build it with why_hard_cache.py: {whc.OUT} missing)")
        return None
    return dict(np.load(whc.OUT, allow_pickle=True))


def count_words(fig):
    """Words of visible text in a rendered figure, and the strings they came from.

    Counts every text artist matplotlib will draw -- titles, axis labels, tick labels, annotations,
    legend entries -- so the brief's limit is checked against what a viewer sees rather than against
    what the author remembers writing. A token that is purely numeric or punctuation is not a word;
    a bare unit or a name is.

    Args:
        fig: a matplotlib Figure, already drawn (call fig.canvas.draw() first so tick labels exist).
    Returns:
        (n_words, list of the strings found).
    """
    found, n = [], 0
    for t in fig.findobj(matplotlib.text.Text):
        s = (t.get_text() or "").strip()
        if not s or not t.get_visible():
            continue
        words = [w for w in re.split(r"[\s/,·()]+", s)
                 if re.search(r"[A-Za-z]", w) and not re.fullmatch(r"[A-Za-z]", w)]
        if words:
            found.append(s)
            n += len(words)
    return n, found


def _report(fig, stem):
    """Print a figure's word count and raise if it breaks the brief's limit.

    Args:
        fig: the figure, about to be saved; stem: its output stem, for the message.
    Returns:
        the word count.
    """
    fig.canvas.draw()
    n, found = count_words(fig)
    print(f"  {stem}: {n} words -> {found}")
    assert n <= MAX_WORDS, f"{stem} has {n} words, over the limit of {MAX_WORDS}: {found}"
    return n


# -------------------------------------------------------------------------------------------------
# WH1: a silent unit is frozen
# -------------------------------------------------------------------------------------------------

def _wiring(ax, x0, firing, col, grad_frac, gauge_span=0.30):
    """One unit with its incoming and outgoing wires, and a gauge of how much its weights move.

    The two halves of WH1 are the same drawing with one thing changed, so they come from one
    function: a newcomer should be able to see that the only difference is whether the unit fires.

    Args:
        ax: a blank axes whose x spans both halves and whose y spans 0..1;
        x0: left edge of this half in data coordinates (the half is 1.0 wide);
        firing: True to draw the unit filled and its wires live, False for hollow and dead;
        col: colour for the unit and its live wires;
        grad_frac: length of the weight-change gauge as a fraction of its track, 0 for frozen;
        gauge_span: track length in data coordinates.
    Returns:
        None.
    """
    ys = (0.26, 0.44, 0.62)
    cx, cy = x0 + 0.50, 0.44
    wire = col if firing else ps.FAINT
    for y in ys:                                                   # incoming
        ax.plot([x0 + 0.09], [y], "o", ms=3.0, color=ps.MUTED, mec="none", zorder=4)
        ps.arrow(ax, (x0 + 0.09, y), (cx, cy), col=wire, lw=0.8, shrink=12.0, mutation_scale=6)
    for y in ys:                                                   # outgoing
        ax.plot([x0 + 0.91], [y], "o", ms=3.0, color=ps.MUTED, mec="none", zorder=4)
        ps.arrow(ax, (cx, cy), (x0 + 0.91, y), col=wire, lw=0.8, shrink=12.0, mutation_scale=6)

    rx = 0.062
    ax.add_patch(Ellipse((cx, cy), 2 * rx, 2 * ps.square_pitch(ax, rx),
                         facecolor=col if firing else "none",
                         edgecolor=col if firing else ps.FAINT, lw=1.1, zorder=6))

    # the gauges: one per direction, on a shared empty track, so "nothing happens" has a shape
    for side, gx in (("in", x0 + 0.13), ("out", x0 + 0.57)):
        ax.add_patch(Rectangle((gx, 0.845), gauge_span, 0.042, facecolor=ps.GRID,
                               edgecolor="none", zorder=3))
        if grad_frac > 0:
            ax.add_patch(Rectangle((gx, 0.845), gauge_span * grad_frac, 0.042, facecolor=col,
                                   edgecolor="none", zorder=4))
        else:
            ax.text(gx + 0.015, 0.866, "0", ha="left", va="center", fontsize=9.5,
                    fontweight="bold", color=ps.BAD, zorder=5)
        ps.arrow(ax, (gx + gauge_span * 0.5, 0.825),
                 (x0 + (0.30 if side == "in" else 0.70), 0.60),
                 col=ps.FAINT, lw=0.6, ls=":", mutation_scale=5, shrink=1.0, zorder=2)


def wh1_zero_gradient(name="slide_wh_zero_gradient"):
    """WH1: the gradient on a silent unit's weights is exactly zero, in both directions.

    THE NUMBERS COME FROM AUTOGRAD, not from the algebra. `why_hard_cache.frozen_gradients` wires a
    few units of a small ReLU network permanently below threshold, backpropagates, and reads the
    largest absolute gradient on the recurrent weights into and out of those units. It is zero to
    the last bit, while the firing units' is not. The repository's own
    `tests/test_prune_and_reinit.py` asserts the same premise and passes.

    The gauge is the figure's whole argument: the same track in both halves, empty on the left and
    full on the right. A relative gauge rather than a printed number because 0.00111 is not a
    quantity a newcomer can picture, and the contrast is what the slide is for.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    g = whc.frozen_gradients()
    assert g["silent_in"] == 0.0 and g["silent_out"] == 0.0, "the frozen premise did not hold"
    scale = max(g["firing_in"], g["firing_out"])

    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 68 * ps.MM))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.44], hspace=0.26, wspace=0.26,
                  left=0.10, right=0.985, top=0.95, bottom=0.15)

    ax = ps.blank(fig.add_subplot(gs[0, :]))
    ax.set(xlim=(-0.46, 2.30), ylim=(0.12, 1.02))
    fig.canvas.draw()                                   # square_pitch needs a laid-out axes
    ax.text(-0.44, 0.866, "weight change", ha="left", va="center", fontsize=6.6, color=ps.MUTED)
    ax.plot([1.16, 1.16], [0.16, 0.95], "-", lw=0.5, color=ps.GRID, zorder=1)
    for x0, firing, col, lab in ((0.0, False, ps.MUTED, "silent unit"),
                                 (1.32, True, FIRING, "firing unit")):
        _wiring(ax, x0, firing, col, 0.0 if not firing else 1.0)
        ax.text(x0 + 0.50, 0.995, lab, ha="center", va="top", fontsize=8.0, color=col,
                fontweight="bold")
    ps.panel_letter(ax, "a", dx=-0.035, dy=0.99)

    # the activation function underneath, with each unit's operating point on it
    h = np.linspace(-1.6, 1.6, 400)
    ax0 = None
    for j, (hstar, col) in enumerate(((-0.95, ps.MUTED), (0.55, FIRING))):
        a = fig.add_subplot(gs[1, j], sharey=ax0)
        ax0 = ax0 or a
        a.plot(h, np.maximum(h, 0.0), "-", lw=1.2, color=ps.INK, zorder=3)
        a.plot([hstar], [max(hstar, 0.0)], "o", ms=6.5, color=col, mec=ps.PAPER, mew=0.8, zorder=5)
        a.plot([hstar, hstar], [-0.12, max(hstar, 0.0)], ":", lw=0.7, color=col, zorder=2)
        a.set(xlim=(-1.6, 1.6), ylim=(-0.12, 1.25), xticks=[0], yticks=[])
        a.set_xticklabels(["0"])
        ps.despine(a, keep=("bottom",))
        if j == 0:
            a.set_xlabel("input", labelpad=1.0)
            a.text(-0.055, 0.5, "rate", transform=a.transAxes, rotation=90, ha="center",
                   va="center", fontsize=7.0, color=ps.INK)
            ps.panel_letter(a, "b", dx=-0.055, dy=1.02)

    _report(fig, name)
    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# WH2: the treadmill
# -------------------------------------------------------------------------------------------------

def _band(ax, x, ys, col, lw=1.0, alpha_fill=0.20):
    """A per-seed mean curve with the seed-to-seed range shaded behind it.

    Args:
        ax: axes; x: (K,) iterations; ys: (seeds, K) values; col: colour; lw: mean line width;
        alpha_fill: opacity of the range band.
    Returns:
        the (K,) mean curve.
    """
    m = ys.mean(axis=0)
    ax.fill_between(x, ys.min(axis=0), ys.max(axis=0), color=col, alpha=alpha_fill, lw=0,
                    zorder=2)
    ax.plot(x, m, "-", lw=lw, color=col, zorder=4)
    return m


def _measure_glyph(ax, counts_above, key):
    """A thumbnail of the counting rule one panel uses, drawn as two units' rates over time.

    The two lower panels of WH2 count different things, and a reader who takes them for one thing
    measured twice reads a contradiction off them. The glyph says which rule its panel uses in two
    words: the unit the panel COUNTS is drawn in ink, the unit it does not count in faint grey, and
    the activity bar appears only in the panel whose rule has one.

    THE GLYPH IS KEYED. Unkeyed, it is two sparklines a viewer has no reason to read as a thumbnail
    of a counting rule at all, which is what a reader new to the deck reported.

    Args:
        ax: the panel's axes; counts_above: True for the rule "the unit's rate clears the activity
            bar", False for the rule "the unit never leaves zero"; key: two words naming what the
            ink curve does, written beside the thumbnail.
    Returns:
        the inset axes holding the glyph.
    """
    gx = ax.inset_axes([0.60, 0.07, 0.30, 0.34])
    t = np.linspace(0, 1, 240)
    bumps = np.abs(np.sin(np.pi * 2.4 * t)) * (0.45 + 0.55 * np.sin(np.pi * t) ** 2)
    if counts_above:
        gx.axhline(0.40, lw=0.7, color=ps.MUTED, ls=(0, (2.2, 2.0)), zorder=2)
        gx.plot(t, 0.16 * bumps, lw=0.8, color=ps.FAINT, zorder=3)     # under the bar: not counted
        gx.plot(t, bumps, lw=1.0, color=ps.INK, zorder=4)              # over the bar: counted
    else:
        gx.plot(t, 0.62 * bumps, lw=0.8, color=ps.FAINT, zorder=3)     # it fires: not counted
        gx.plot(t, np.zeros_like(t), lw=1.4, color=ps.INK, zorder=4)   # flat at zero: counted
    gx.set(xlim=(-0.03, 1.03), ylim=(-0.16, 1.18), xticks=[], yticks=[])
    # beside the thumbnail, not above it: the panel's own curves run across the top of this band
    gx.text(-0.07, 0.45, key, transform=gx.transAxes, ha="right", va="center", fontsize=6.4,
            color=ps.INK)
    for sp in gx.spines.values():
        sp.set_visible(False)
    gx.patch.set_alpha(0.0)
    return gx


def wh2_treadmill(name="slide_wh_treadmill"):
    """WH2: a rule that redraws dormant units fires tens of thousands of times and buys four units.

    THE EFFORT AND THE RESULT ARE DRAWN ON THE SAME TIME AXIS, which is the only way the mismatch
    reads as one fact rather than two. The redraw counter climbs to 26,012 while the two curves for
    working units lie on top of each other.

    THE MECHANISM DOES REACH THE UNITS, and the bottom panel is there so the figure is not read as
    "the rule did nothing". The count of units that never fire at all falls by about 130 of 1000.
    They come back on and the network puts them back off; what does not move is how many units the
    network ends up using.

    Both panels are read off the participation trace the trainer wrote during the run, which the
    cache cross-checks against an independent rebuild-and-rerun of every network.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cache is missing.
    """
    z = cache()
    if z is None:
        return None
    need = [f"treadmill|{a}|curve_{k}" for a in ("control", "redraw") for k in ("it", "active")]
    if any(k not in z for k in need):
        print(f"  (no treadmill curves in {whc.OUT}; rebuild it with --refresh)")
        return None

    it = np.asarray(z["treadmill|control|curve_it"], float)
    ev_it = np.asarray(z["treadmill|redraw|curve_ev_it"], float)
    events = np.asarray(z["treadmill|redraw|curve_events"], float)
    arms = (("control", REST, "left alone"), ("redraw", OWN, "redrawn"))

    ps.setup()
    fig = plt.figure(figsize=(118 * ps.MM, 104 * ps.MM))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.78, 1.0, 0.86], hspace=0.22,
                  left=0.165, right=0.745, top=0.975, bottom=0.095)

    ax_e = fig.add_subplot(gs[0])
    _band(ax_e, ev_it, events, OWN, lw=1.3)
    fin = events[:, -1].mean()
    ax_e.annotate(f"{fin:,.0f}", (ev_it[-1], fin), textcoords="offset points", xytext=(4, 2),
                  ha="left", va="bottom", fontsize=8.6, color=OWN, fontweight="bold")
    # the two counts that make the point together: how often the rule fired, and how few units it
    # ever reached -- 26,012 over 591 units is 44 redraws each
    ever = float(np.asarray(z["treadmill|redraw|units_ever"], float).mean())
    ax_e.annotate(f"{ever:.0f} different units", (ev_it[-1], fin), textcoords="offset points",
                  xytext=(4, -3), ha="left", va="top", fontsize=6.6, color=ps.MUTED)
    # THE AXIS CARRIES THE RULE. "redraws" named the count without naming the operation: a unit that
    # has been silent long enough has its weights thrown away and drawn again at random, and this is
    # the running total of how often the trainer has done that.
    ax_e.set_ylabel("silent units given\nrandom weights", labelpad=2, fontsize=6.2)
    ax_e.set(ylim=(0, 1.18 * events.max()))
    ps.panel_letter(ax_e, "a", dx=-0.135, dy=0.98)

    ax_a = fig.add_subplot(gs[1], sharex=ax_e)
    ax_n = fig.add_subplot(gs[2], sharex=ax_e)
    hi_a, hi_n = 0.0, 0.0
    for arm, col, lab in arms:
        ya = np.asarray(z[f"treadmill|{arm}|curve_active"], float)
        yn = np.asarray(z[f"treadmill|{arm}|curve_never"], float)
        hi_a, hi_n = max(hi_a, ya.max()), max(hi_n, yn.max())
        m = _band(ax_a, it, ya, col)
        # the two end counts are four units apart on an axis a thousand units tall, so they are
        # nudged off each other rather than drawn on top of one another
        ax_a.annotate(f"{m[-1]:.0f}", (it[-1], m[-1]), textcoords="offset points",
                      xytext=(4, 6 if arm == "redraw" else -6), ha="left",
                      va="bottom" if arm == "redraw" else "top", fontsize=7.6, color=col)
        mn = _band(ax_n, it, yn, col)
        ax_n.annotate(f"{mn[-1]:.0f} {lab}", (it[-1], mn[-1]), textcoords="offset points",
                      xytext=(4, 0), ha="left", va="center", fontsize=7.2, color=col)
    # THE TWO PANELS COUNT DIFFERENT THINGS and have to say so, or 414 against 410 and 219 against
    # 348 read as one measure contradicting itself. The wording is deliberately not interchangeable,
    # and each panel carries a thumbnail of its own rule.
    ax_a.set_ylabel("units above the activity bar", labelpad=2, fontsize=6.2)
    ax_n.set_ylabel("units that never fire", labelpad=2, fontsize=6.2)
    _measure_glyph(ax_a, counts_above=True, key="fires again")
    _measure_glyph(ax_n, counts_above=False, key="never fires")
    ax_n.set_xlabel("training step", labelpad=1.5)
    ax_a.set(ylim=(0, 1.08 * hi_a))
    ax_n.set(ylim=(0, 1.10 * hi_n))
    ax_e.set(xlim=(0, it[-1] * 1.005))
    for a, letter in ((ax_a, "b"), (ax_n, "c")):
        ps.panel_letter(a, letter, dx=-0.135, dy=0.98)
        ps.ygrid(a)
    ps.ygrid(ax_e)
    for a in (ax_e, ax_a):
        a.tick_params(labelbottom=False)
    # a hairline where the measure changes, so the lower panel reads as a second measurement of the
    # same networks rather than as a continuation of the one above it
    fig.canvas.draw()
    y_rule = (ax_a.get_position().y0 + ax_n.get_position().y1) / 2
    fig.add_artist(plt.Line2D([0.10, 0.995], [y_rule] * 2, lw=0.6, color=ps.GRID,
                              transform=fig.transFigure, zorder=0))

    _report(fig, name)
    return ps.save(fig, name, w_mm=118)


# -------------------------------------------------------------------------------------------------
# WH3: what a copy carries
# -------------------------------------------------------------------------------------------------

def _tint(hexcol, f):
    """A lighter version of a palette colour, mixed toward the paper surface.

    The three steps of WH3 are three parts of one thing, so they are three tints of ONE palette
    colour rather than three hexes chosen by eye: a hand-picked tint drifts off the hue and off the
    contrast the palette was validated for, and nothing in paperstyle then governs it.

    Args:
        hexcol: a "#rrggbb" palette colour; f: fraction of ps.PAPER in the mix, 0 to 1 (0 returns
            the colour itself, 1 returns the paper surface).
    Returns:
        the mixed colour as a "#rrggbb" string.
    """
    a = np.array([int(hexcol[i:i + 2], 16) for i in (1, 3, 5)], float)
    b = np.array([int(ps.PAPER[i:i + 2], 16) for i in (1, 3, 5)], float)
    return "#%02x%02x%02x" % tuple(int(round(v)) for v in a + (b - a) * float(f))


def _on(hexcol):
    """Ink or paper, whichever is readable on a filled patch of this colour.

    Args:
        hexcol: a "#rrggbb" fill colour.
    Returns:
        ps.INK for a light fill, ps.PAPER for a dark one, picked on relative luminance rather than
        by naming the colours, so a changed tint cannot leave text unreadable.
    """
    r, g, b = (int(hexcol[i:i + 2], 16) / 255.0 for i in (1, 3, 5))
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in (r, g, b)]
    return ps.INK if 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2] > 0.32 else ps.PAPER


def wh3_decomposition(name="slide_wh_decomposition"):
    """WH3: copying a working unit recruits, split into the three things the copy carries.

    FOUR MATCHED CELLS, ONE DIFFERENCE EACH, so the split is measured rather than reasoned. All are
    the 3-bit flip-flop at N = 1000 for 40,000 iterations with the same replacement rate; a replaced
    unit is handed, in turn, nothing from the donor, the donor's outgoing wires only, the donor's
    incoming weight magnitudes in scrambled positions, and finally those magnitudes in their own
    positions. Each step's height is the difference between two measured cells and the per-seed dots
    are drawn on top of it, so a reader can see how much of each step the seeds support.

    THE GREY BAR CARRIES ONE DOT SERIES AND ONE NAME. It used to carry a second, unkeyed series --
    the cells that redraw the unit's incoming weights with nothing of the donor's, which land on the
    untouched network's count -- annotated "random weights", so the bar the axis calls "left alone"
    was named twice and two kinds of dot sat on it with nothing to tell them apart. That cell is
    measured and it is the same failure WH2 draws on the other task; the deck makes the point there.

    THE STEPS ARE TIED TOGETHER. Each bar starts where the one before it finished, and the connector
    between them is what makes the staggered bases read as a waterfall rather than as a mistake.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cache is missing.
    """
    z = cache()
    if z is None:
        return None
    arms = ("control", "copy_iid", "permute", "copy")
    if any(f"decompose|{a}|active" not in z for a in arms):
        print(f"  (decomposition arms missing from {whc.OUT})")
        return None
    seeds = {a: np.asarray(z[f"decompose|{a}|active"], float) for a in arms}
    lv = {a: seeds[a].mean() for a in arms}

    # the three steps, each a difference between two measured cells
    # THE LABELS NAME WHAT THE UNIT IS HANDED, under the bracket that says whose it is. "outgoing
    # wires", "weight sizes" and "weight places" were each read wrongly by someone new to the deck:
    # the three cells hand the revived unit the donor's outgoing column (it drives the donor's
    # targets), then incoming weights of the donor's magnitudes in scrambled positions, then those
    # same magnitudes on the donor's own sources (it listens to the donor's inputs).
    steps = [("output targets", lv["control"], lv["copy_iid"]),
             ("input sizes, shuffled", lv["copy_iid"], lv["permute"]),
             ("input sources", lv["permute"], lv["copy"])]
    # one hue, light to dark in the step order, ending at the full-strength colour of the bar that
    # is their sum
    tints = [_tint(DONOR, f) for f in (0.62, 0.38, 0.14)]

    ps.setup()
    fig, ax = plt.subplots(figsize=(128 * ps.MM, 82 * ps.MM))
    fig.subplots_adjust(left=0.105, right=0.985, top=0.80, bottom=0.165)
    w = 0.62

    ax.bar([0], [lv["control"]], width=w, color=REST, edgecolor="none", zorder=3)
    # THE TWO END BARS CARRY THEIR VALUES TOO. With only the three increments numbered, the bars the
    # increments run between - where the count starts and where it ends - had to be read off the
    # grid lines while everything between them was printed.
    for i, lvl, col in ((0, lv["control"], REST), (4, lv["copy"], DONOR)):
        ax.annotate(f"{lvl:.0f}", (i, lvl / 2), ha="center", va="center", fontsize=7.8,
                    color=_on(col), fontweight="bold", zorder=5)
    for i, ((lab, lo, hi), col) in enumerate(zip(steps, tints), start=1):
        ax.bar([i], [hi - lo], bottom=[lo], width=w, color=col, edgecolor="none", zorder=3)
        # EVERY BAR IS TIED TO THE ONE BEFORE IT. Without the connector the staggered bases read as
        # a drawing mistake rather than as each step starting where the last one finished.
        ax.plot([i - 1 + w / 2, i + w / 2], [lo, lo], "-", lw=0.6, color=ps.MUTED, zorder=2)
        ax.annotate(f"+{hi - lo:.0f}", (i, (lo + hi) / 2), ha="center", va="center", fontsize=7.8,
                    color=_on(col), fontweight="bold", zorder=5)
    ax.bar([4], [lv["copy"]], width=w, color=DONOR, edgecolor="none", zorder=3)
    # the last bar is the three steps added up, not a fourth step, so its top is tied back
    ax.plot([3 + w / 2, 4 + w / 2], [lv["copy"]] * 2, ":", lw=0.7, color=ps.MUTED, zorder=2)

    rng = np.random.default_rng(3)
    ps.strip(ax, [0, 1, 2, 3, 4],
             [seeds["control"], seeds["copy_iid"], seeds["permute"], seeds["copy"], seeds["copy"]],
             [ps.INK] * 5, width=0.30, jitter=0.05, rng=rng, mean_lw=0.0, ms=2.6, alpha=0.95,
             zorder=6)

    ax.text(2.0, 1.035, "a working unit's", transform=ax.get_xaxis_transform(), ha="center",
            va="bottom", fontsize=7.4, color=ps.INK)
    ax.plot([0.66, 3.34], [1.025, 1.025], "-", lw=0.7, color=ps.MUTED,
            transform=ax.get_xaxis_transform(), clip_on=False)
    ax.set(xlim=(-0.6, 4.6), ylim=(0, 1.02 * (lv["copy"] + seeds["copy"].std(ddof=1) * 2)),
           xticks=[0, 1, 2, 3, 4],
           xticklabels=["left alone"] + [s[0] for s in steps] + ["all three"])
    ax.set_ylabel("units above the activity bar", labelpad=2, fontsize=6.2)
    ps.ygrid(ax)
    _report(fig, name)
    return ps.save(fig, name, w_mm=128)


def main(refresh=False):
    """Write every "why is this hard?" figure. Returns the list of output paths."""
    if refresh or not os.path.exists(whc.OUT):
        whc.main(refresh=True)
    out = []
    for fn in (wh1_zero_gradient, wh2_treadmill, wh3_decomposition):
        got = fn()
        if got:
            out.append(got)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="rebuild the measurement cache first")
    main(refresh=ap.parse_args().refresh)
