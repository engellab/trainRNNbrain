#!/usr/bin/env python3
"""
Manuscript Figure 2 - THREE WAYS TO KEEP UNITS ALIVE, measured on the same four axes.

Three interventions that need no change to the loss function, each asked the same four questions:
does the network still solve the task, how many units end up active, how many directions does the
population use, and does the weight distribution still look like the one biology has.

  (a) WHAT THE THREE RULES DO       drawn as the same picture three times - a silent unit among live
                                    ones - so the difference between the rules is a cut edge, a
                                    copied row and a tilted row, not a paragraph.
                                      dropout: mute      the sampled unit's READ-OUT weight is
                                                         zeroed, so it goes on driving its
                                                         neighbours but the task loss cannot see it.
                                                         Sampling is biased toward busy units.
                                      prune + duplicate  a silent unit is deleted and rebuilt as a
                                                         copy of a working one; the donor's outgoing
                                                         column is halved and the pair's 2x2 weight
                                                         block set so the network's output is
                                                         unchanged at the moment of surgery.
                                      rescale            the silent unit keeps its wiring. Its
                                                         incoming excitatory weights are multiplied
                                                         by alpha and its inhibitory ones divided by
                                                         it, tilting its drive toward excitation at
                                                         a preserved row norm.
  (b) ACTIVE UNITS                  the scale-free rule, every network drawn, out of 1000.
  (c) HELD-OUT r2                   recomputed on a fresh batch, so it can be checked against the
                                    value stored at training time.
  (d) DIMENSIONS USED               participation ratio of the noise-free rate covariance over the
                                    active units: how many directions the population actually uses.
                                    The count of components carrying 95% of the variance moves the
                                    same way but further - 12 for the control, 25 under dropout, 23
                                    under duplication, 12 under rescale - so the panel's ratio is
                                    the conservative reading of the same effect. Both are printed.
  (e) WEIGHT MAGNITUDES             the distribution of |W_rec| over all 10^6 entries. Cortical
                                    synaptic strengths are lognormal over roughly two orders of
                                    magnitude (Song et al. 2005; Lefort et al. 2009), so an
                                    intervention that recruits units by manufacturing a weight
                                    distribution biology does not produce has bought them with an
                                    artifact. This is the panel that catches one: dropout and
                                    duplication leave the range where the control has it, and
                                    rescale does not.

MATCHED, AND WHY THAT COST A CELL. Every arm is gamma = 0, N = 1000, 3-bit flip-flop, 40,000
iterations, lr 1e-3, weight decay 1e-6, sigma_rec = sigma_inp = 0.05, batch 1024 - the intervention
is the only difference. The duplication sweep at gamma = 0.1 (`ff_revive_g01_fix`) is NOT pooled in:
gamma is cubic saturation in the dynamics, so it changes the base network and a cross-gamma
comparison is not like for like. Duplication here is the corrected construction of 2026-09-24; the
cells carrying the earlier detuned self-weight are excluded (see f2_remedies_cache.py).

RESCALE IS SHOWN AT ITS BEST CONFIGURATION, NOT ITS FIRST. The rule was developed over five sweeps
-- an alpha-only form, then a fixed activity target that holds a boosted unit until it fires, then a
refractory tail after it graduates -- and the early alpha-only form recruits nothing at all. Quoting
that form as "rescale" would report the rule at its weakest, and pooling the grid would average the
developed rule with it. The cell drawn is picked by `select_rescale_cell`, on a rule fixed before
the cells were scored: most active units among the cells whose r2 is within 5% of the control's.
Every matched cell is printed under the figure, the drawn one marked.

WHAT REPLACED WHAT. The previous Figure 2 was dropout alone: a mute schematic, a characterisation of
the dropout sampler, the 150k training curve and a cost panel. The sampler panel documented a
sampler that was corrected on 2026-09-21 and was already marked for redesign; the training curve and
the cost panel are superseded by (b) and (c) here, which carry the same read-out for three
interventions instead of one.

CRITERION AND READ-OUT as Figure 1: scale-free participation, matched compute, every network drawn.

Usage:  python fig_paper_F2.py
Output: img/internal_figures/fig_paper_F2.pdf (+ .svg; vector only - see paperstyle.save)
Cache:  data/fig_paper_F2_cache.npz, built on the cluster by f2_remedies_cache.py
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
from flipflop_dropout_readout import welch

CACHE = "data/fig_paper_F2_cache.npz"
N_UNITS = 1000

# (key in the cache, x tick label, full name, colour). The x tick labels are short because panel
# (a) names the rules directly above them. The control is neutral ink: it is the reference every
# remedy is measured against, not a fifth condition.
ARMS = [("control", "none", "no intervention", ps.BASE),
        ("mute", "dropout", "dropout: mute", ps.COND_COL["mute"]),
        ("duplicate", "duplicate", "prune + duplicate", ps.COND_COL["duplicate"]),
        ("rescale", "rescale", "rescale", ps.COND_COL["rescale"])]


def load(path=CACHE):
    """Read the per-network cache.

    Args:
        path: path to the npz written by f2_remedies_cache.py.
    Returns:
        dict of arrays, one row per network, all the same length. Raises SystemExit if absent.
    """
    if not os.path.exists(path):
        raise SystemExit(f"{path} is missing - build it on the cluster with f2_remedies_cache.py")
    z = np.load(path, allow_pickle=True)
    return {k: z[k] for k in z.files}


def by_arm(c, key):
    """One array of per-network values per arm, in ARMS order.

    Args:
        c: the cache dict; key: the field to pull, e.g. 'n_active'.
    Returns:
        list of 1-D float arrays, one per arm.
    """
    return [np.asarray(c[key][c["arm"] == a], float) for a, _, _, _ in ARMS]


def select_rescale_cell(c, margin_frac=0.05):
    """Which of the matched rescale cells the figure draws, by a rule fixed before the numbers.

    The rescale rule was developed over five sweeps: an alpha-only form, then a fixed activity
    target, then a refractory tail after a unit graduates. Its cells therefore range from "does
    nothing" to whatever the rule can do, and pooling them, or quoting the earliest, reports the rule
    at its weakest. Dropout and duplication are each shown at one configuration, so rescale is too.

    THE RULE, fixed before these cells were scored: take the cell with the highest mean active-unit
    count among those whose mean r2 is within `margin_frac` of the control's - the same equivalence
    margin the cost read-out uses. Recruiting units by destroying the task is not recruiting.

    Args:
        c: the full cache dict; margin_frac: how far below the control's r2 a cell may sit.
    Returns:
        the chosen cell string, or None if the cache holds no rescale cell.
    """
    ref = np.mean(c["r2"][c["arm"] == "control"].astype(float))
    best, best_active = None, -1.0
    for cell in sorted(set(c["cell"][c["arm"] == "rescale"])):
        m = (c["arm"] == "rescale") & (c["cell"] == cell)
        if np.mean(c["r2"][m].astype(float)) < ref * (1 - margin_frac):
            continue
        active = float(np.mean(c["n_active"][m].astype(float)))
        if active > best_active:
            best, best_active = cell, active
    return best


def keep_rescale_cell(c, cell):
    """Drop every rescale network that is not from `cell`, leaving the other arms untouched.

    Args:
        c: the full cache dict; cell: the rescale cell to keep.
    Returns:
        a new dict of the same keys, with the rescale rows filtered. 'log_bins' is passed through.
    """
    keep = (c["arm"] != "rescale") | (c["cell"] == cell)
    return {k: (v if k == "log_bins" else v[keep]) for k, v in c.items()}


def tost(a, b, margin_frac=0.05):
    """Two one-sided tests for equivalence of two means within +-margin_frac of b's mean.

    "p = 0.95 so it is free" is absence of evidence, not evidence of equivalence. TOST asks the
    question the paper actually means: is the difference small enough to be uninteresting? The margin
    is fixed at 5% of the reference, chosen because the rate penalty of Figure 3 costs 7%, so the bar
    is "cheaper than the remedy we recommend".

    Args:
        a, b: 1-D samples (a = intervention, b = reference); margin_frac: equivalence margin as a
            fraction of mean(b).
    Returns:
        (p_tost, diff_frac, lo_frac, hi_frac): the larger of the two one-sided p-values, the relative
        difference, and its 95% CI, all as fractions of mean(b).
    """
    a, b = np.asarray(a, float), np.asarray(b, float)
    if len(a) < 2 or len(b) < 2:
        return float("nan"), float("nan"), float("nan"), float("nan")
    d = a.mean() - b.mean()
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    df = (a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)) ** 2 / (
        (a.var(ddof=1) / len(a)) ** 2 / (len(a) - 1) + (b.var(ddof=1) / len(b)) ** 2 / (len(b) - 1))
    delta = margin_frac * b.mean()
    try:
        from scipy import stats
        p = max(stats.t.sf((d + delta) / se, df), stats.t.cdf((d - delta) / se, df))
        crit = stats.t.ppf(0.975, df)
    except ImportError:
        from math import erfc, sqrt
        p = max(erfc((d + delta) / se / sqrt(2)) / 2, erfc(-(d - delta) / se / sqrt(2)) / 2)
        crit = 1.96
    return float(p), d / b.mean(), (d - crit * se) / b.mean(), (d + crit * se) / b.mean()


def _units(ax, x, y, states, col):
    """Draw a row of unit glyphs for a schematic.

    Args:
        ax: schematic axes; x: list of x centres; y: shared y centre;
        states: one of 'live', 'silent', 'new' per unit. Live is filled in `col`; silent is hollow
            and dotted, because a silent unit is still present in the network and that is the whole
            point; new is filled in `col` inside the dotted ring of the unit it replaced.
        col: the arm's colour.
    Returns:
        None.
    """
    for xi, st in zip(x, states):
        if st in ("silent", "new"):
            ax.scatter(xi, y, s=150, color="none", edgecolor=ps.FAINT, linewidth=0.8,
                       linestyle=":", zorder=4)
        if st != "silent":
            ax.scatter(xi, y, s=62, color=col, edgecolor=col, linewidth=0.8, zorder=5)


def panel_a(ax):
    """Panel (a): the three rules, each drawn on the same three-unit picture.

    Every sub-schematic has the same anatomy, so a reader can set the three against each other: the
    rule's name, the operation written above the arrow that performs it, a label under each unit the
    rule touches saying what that unit is, and one line at the bottom giving the consequence. The
    rows sit at the same heights in all three, and everything a rule draws below the units - the
    read-out under `mute`, the inhibitory arc under `rescale` - is kept above the label row.

    Args:
        ax: a blank axes spanning the figure's top row.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 3.18), ylim=(0, 1))
    y = 0.68                                                  # the row of units
    y_title, y_op, y_unit, y_foot = 1.00, 0.885, 0.30, 0.13
    for i, (kind, _, title, col) in enumerate(ARMS[1:]):
        x0 = i * 1.06
        left, mid, right = x0 + 0.18, x0 + 0.50, x0 + 0.82
        ax.text(x0 + 0.50, y_title, title, ha="center", va="top", fontsize=7.2,
                color=col, fontweight="bold")

        if kind == "mute":
            # the unit the sampler picks is an ACTIVE one, and only its read-out weight is cut
            _units(ax, [left, mid, right], y, ["live", "live", "silent"], col)
            ry = 0.40
            ps.box(ax, x0 + 0.50 - 0.15, ry, 0.30, 0.085, "read-out", col=ps.MUTED,
                   face="#f2f1ec", lw=0.6, fs=5.6)
            for j, xi in enumerate((left, mid, right)):
                cut = (j == 0)
                ps.arrow(ax, (xi, y - 0.055), (x0 + 0.50 + (j - 1) * 0.085, ry + 0.085),
                         col=ps.BAD if cut else ps.MUTED, lw=0.75, ls=":" if cut else "-")
                if cut:
                    mx, my = (xi + x0 + 0.50 - 0.085) / 2, (y - 0.055 + ry + 0.085) / 2
                    for sgn in (1, -1):
                        ax.plot([mx - 0.026, mx + 0.026], [my - sgn * 0.028, my + sgn * 0.028],
                                lw=1.0, color=ps.BAD, zorder=7)
            ax.text(x0 + 0.50, y_op, "set its read-out weight to zero", ha="center", va="center",
                    fontsize=5.8, color=col)
            ax.text(left, y_unit, "sampled:\nan active unit", ha="center", va="top", fontsize=5.5,
                    color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "the loss can no longer see this unit,\n"
                    "but it still drives the others", ha="center", va="top", fontsize=5.6,
                    color=ps.INK, linespacing=1.35)

        elif kind == "duplicate":
            # the silent unit is deleted and rebuilt as a copy of the live donor on the left
            _units(ax, [left, mid, right], y, ["live", "live", "new"], col)
            ps.arrow(ax, (left, y), (right, y), col=col, rad=-0.26, lw=0.9, shrink=7.0,
                     mutation_scale=6)
            ax.text(x0 + 0.50, y_op, "copy the donor's incoming connections",
                    ha="center", va="center", fontsize=5.8, color=col)
            ax.text(left, y_unit, "donor:\nan active unit", ha="center", va="top", fontsize=5.5,
                    color=ps.MUTED, linespacing=1.3)
            ax.text(right, y_unit, "pruned silent unit,\nrebuilt as the copy", ha="center",
                    va="top", fontsize=5.5, color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "divide the donor's outgoing connections by 2,\n"
                    "and the network's output does not change", ha="center", va="top",
                    fontsize=5.6, color=ps.INK, linespacing=1.35)

        else:
            # the silent unit keeps every synapse it has; only their balance changes. The label goes
            # under `right`, which is the silent one - under `left` it named a unit the rule does
            # not touch.
            _units(ax, [left, mid, right], y, ["live", "live", "silent"], col)
            ps.arrow(ax, (left, y), (right, y), col=col, lw=0.9, rad=-0.26, shrink=7.0,
                     mutation_scale=6)
            ps.arrow(ax, (mid, y), (right, y), col=ps.MUTED, lw=0.9, rad=0.55, shrink=7.0,
                     mutation_scale=6)
            ax.text(x0 + 0.50, y_op, r"excitatory inputs $\times\,\alpha$", ha="center",
                    va="center", fontsize=5.8, color=col)
            ax.text(x0 + 0.60, 0.545, r"inhibitory inputs $\div\,\alpha$", ha="center",
                    va="top", fontsize=5.8, color=ps.MUTED)
            ax.text(right, y_unit, "silent unit,\nkept in place", ha="center", va="top",
                    fontsize=5.5, color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "no new wiring: more excitation and less\n"
                    "inhibition, at the same total synaptic weight", ha="center", va="top",
                    fontsize=5.6, color=ps.INK, linespacing=1.35)


def _cat_axes(ax, ylabel):
    """Shared cosmetics for the three per-arm dot panels.

    Args:
        ax: axes; ylabel: y axis label.
    Returns:
        the x positions of the arms.
    """
    xs = np.arange(len(ARMS), dtype=float)
    ax.set_xticks(xs)
    ax.set_xticklabels([lab for _, lab, _, _ in ARMS], fontsize=6.2, rotation=30,
                       ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.55, len(ARMS) - 0.45)
    ax.set_ylabel(ylabel)
    ps.ygrid(ax)
    return xs


def panel_b(ax, c):
    """Panel (b): active units per network, out of 1000. Returns per-arm (mean, sd, n)."""
    xs = _cat_axes(ax, "active units")
    groups = by_arm(c, "n_active")
    res = ps.strip(ax, xs, groups, [col for _, _, _, col in ARMS], rng=np.random.default_rng(3))
    ax.axhline(N_UNITS, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
    ax.set_ylim(0, N_UNITS * 1.12)
    # Offsets are in POINTS from the highest seed of each arm, not in data units from its mean: the
    # arms differ in spread (12 rescale seeds against 3 elsewhere), so a fixed data-unit offset
    # clears the dots in one arm and lands on them in the next.
    ax.annotate(f"all {N_UNITS}", (len(ARMS) - 0.5, N_UNITS), textcoords="offset points",
                xytext=(0, 3), ha="right", va="bottom", fontsize=5.4, color=ps.MUTED)
    for x, g, (m, sd, n) in zip(xs, groups, res):
        ax.annotate(f"{m:.0f}", (x, g.max()), textcoords="offset points", xytext=(0, 5),
                    ha="center", va="bottom", fontsize=5.8, color=ps.INK)
    return res


def panel_c(ax, c):
    """Panel (c): held-out r2 per network. Returns per-arm (mean, sd, n)."""
    xs = _cat_axes(ax, "held-out $r^2$")
    groups = by_arm(c, "r2")
    res = ps.strip(ax, xs, groups, [col for _, _, _, col in ARMS], rng=np.random.default_rng(4))
    ref = res[0][0]
    ax.axhline(ref, color=ps.BASE, lw=0.7, ls=":", zorder=1)
    # No +-5% equivalence band is drawn: the margin is 0.047 of r2 and the largest cost here is
    # 0.017, so the band would fill the panel and say nothing. The TOST verdicts are printed below.
    # The delta row sits in a band cleared BELOW the lowest seed drawn, not at a fixed ylim: which
    # rescale cell is drawn changes the floor of this panel by more than the band is tall.
    lo = min(g.min() for g in groups)
    ax.set_ylim(lo - 0.011, max(0.952, max(g.max() for g in groups) + 0.003))
    for x, (m, sd, n) in zip(xs[1:], res[1:]):
        ax.text(x, lo - 0.0085, f"{(m - ref) / ref:+.1%}", ha="center", va="center", fontsize=5.6,
                color=ps.MUTED)
    return res


def panel_d(ax, c):
    """Panel (d): dimensions the active population uses. Returns per-arm (mean, sd, n)."""
    xs = _cat_axes(ax, "dimensions used")
    res = ps.strip(ax, xs, by_arm(c, "dims"), [col for _, _, _, col in ARMS],
                   rng=np.random.default_rng(5))
    ax.set_ylim(0, 10)
    return res


def _fold(log10_range):
    """A log10 range written as a fold-range a reader can picture, rounded, never in exponent form.

    Args:
        log10_range: the log10 of a ratio, e.g. 3.22 for a 1,660-fold range.
    Returns:
        a string such as "2,000-fold" or "5 billion-fold".
    """
    v = 10.0 ** log10_range
    for cut, name in ((1e9, "billion"), (1e6, "million"), (1e3, "thousand")):
        if v >= cut:
            return f"{v / cut:.0f} {name}-fold"
    return f"{v:,.0f}-fold"


def panel_e(ax, c):
    """Panel (e): the distribution of recurrent-weight magnitudes, one curve per arm.

    Densities are normalised per network and then averaged within an arm, so a network with more
    nonzero weights does not weigh more than its neighbour.

    Args:
        ax: axes; c: the cache dict.
    Returns:
        dict with the per-arm median magnitude, the sd of log|W| and the log10 q99/q01 range.
    """
    edges = np.asarray(c["log_bins"], float)
    mid = 0.5 * (edges[1:] + edges[:-1])
    out, lo_x, hi_x = {}, [], []
    for kind, short, _, col in ARMS:
        h = np.asarray(c["w_hist"][c["arm"] == kind], float)
        if not len(h):
            continue
        d = (h / h.sum(axis=1, keepdims=True)).mean(axis=0)
        dens = d / (mid[1] - mid[0])
        ax.plot(mid, np.where(dens > 0, dens, np.nan), lw=1.1, color=col, zorder=4,
                label=short)
        cdf = np.cumsum(d)
        # the arms differ by orders of magnitude in spread, so the window is taken from the data:
        # a fixed one either clips the widest arm or squeezes the others into a spike
        lo_x.append(mid[np.searchsorted(cdf, 0.01)])
        hi_x.append(mid[np.searchsorted(cdf, 0.99)])
        out[kind] = dict(median=float(mid[np.searchsorted(cdf, 0.5)]),
                         sigma_log=float(np.mean(c["w_sigma_log"][c["arm"] == kind])),
                         spread=float(np.mean(c["w_spread"][c["arm"] == kind])))
    # LOG DENSITY, not linear. What separates the arms is the TAIL: rescale's bulk sits where every
    # other arm's does and its range comes from weights driven far below the rest, so on a linear
    # density the four curves are one peak and a 5-billion-fold range reads as a faint shoulder.
    ax.set(xlim=(min(lo_x) - 0.3, max(hi_x) + 0.3), yscale="log", ylim=(2e-4, 6.0),
           xlabel="recurrent weight\n$\\log_{10}|W_{ij}|$", ylabel="density")
    ax.legend(loc="upper left", fontsize=5.6, handlelength=1.0, borderpad=0.1, borderaxespad=0.2)
    # The panel's result, in the title so it cannot land on a curve. The magnitude RANGE is the
    # statistic quoted rather than the sd, because a fold-range is a number a reader can picture.
    others = max(out[k]["spread"] for k in out if k != "rescale")
    ax.set_title(f"a {_fold(others)} range of magnitudes,\n"
                 f"{_fold(out['rescale']['spread'])} under rescale",
                 fontsize=5.6, color=ps.MUTED, linespacing=1.3, pad=3)
    ps.ygrid(ax)
    return out


def _ink_points(artist, renderer, max_step=2.0):
    """Every point of a drawn artist, in display coordinates, densified along its segments.

    A bounding box is the wrong test for a curve. A density curve on a log axis has a box covering
    the whole panel, so every in-panel label reads as a collision; a diagonal line has a box whose
    corners it never visits, so a label in one corner reads as clear when it is. This returns the
    ink itself, with long segments subdivided so a label cannot slip between two vertices.

    Args:
        artist: a Line2D or a collection with offsets; renderer: the active renderer;
        max_step: longest gap left between consecutive returned points, in display units.
    Returns:
        (n, 2) array of display-space points, empty if the artist draws nothing.
    """
    if hasattr(artist, "get_xydata"):
        pts = artist.get_transform().transform(np.asarray(artist.get_xydata(), float))
    else:
        off = np.asarray(artist.get_offsets(), float)
        pts = artist.get_offset_transform().transform(off) if len(off) else np.empty((0, 2))
    pts = pts[np.isfinite(pts).all(axis=1)]
    if len(pts) < 2:
        return pts
    out = [pts[:1]]
    for a, b in zip(pts[:-1], pts[1:]):
        n = int(np.ceil(np.hypot(*(b - a)) / max_step))
        if n > 1:
            out.append(a + np.outer(np.linspace(0, 1, n + 1)[1:], b - a))
        else:
            out.append(b[None, :])
    return np.vstack(out)


def check_labels_clear(fig, pad=2.0):
    """Raise if any label overlaps another label, or drawn data, anywhere in the figure.

    Two separate failures, because both have happened here. Labels drift onto DATA as soon as the
    data move: an offset that clears a three-seed arm lands on a twelve-seed one, and a rescale cell
    with a lower r2 drops a point into the row of deltas beneath it. Labels also collide with EACH
    OTHER, which is how the schematic's rule names ended up sitting on the lines describing them.

    Label-against-label is checked in every panel, the schematic included. Label-against-data is
    checked only where "data" means a measurement: the schematic's arrows and unit glyphs are drawn
    to be annotated, so text is meant to sit against them.

    Args:
        fig: the drawn figure. Its canvas is drawn here, so call it before saving.
        pad: display-unit margin added around each label, so a curve grazing a glyph counts.
    Returns:
        the number of labels checked.
    Raises:
        AssertionError naming every colliding pair and every label on top of data.
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad, checked = [], 0
    for ax in fig.axes:
        where = ax.get_ylabel() or "schematic"
        labels = [(t.get_text().replace("\n", " "), t.get_window_extent(r))
                  for t in ax.texts if t.get_text().strip()]
        if ax.get_title():
            labels.append((ax.get_title().replace("\n", " "), ax.title.get_window_extent(r)))
        if ax.get_legend() is not None:
            labels.append(("<legend>", ax.get_legend().get_window_extent(r)))
        checked += len(labels)

        for i, (ta, ba) in enumerate(labels):
            for tb, bb in labels[i + 1:]:
                if ba.overlaps(bb):
                    bad.append(f"{where}: {ta!r} overlaps {tb!r}")

        if where == "schematic":
            continue
        ink = [p for a in list(ax.lines) + list(ax.collections)
               for p in (_ink_points(a, r),) if len(p)]
        for text, box in labels:
            hit = any(((p[:, 0] > box.x0 - pad) & (p[:, 0] < box.x1 + pad) &
                       (p[:, 1] > box.y0 - pad) & (p[:, 1] < box.y1 + pad)).any() for p in ink)
            if hit:
                bad.append(f"{where}: {text!r} sits on the data")
    assert not bad, "Figure 2 label collisions:\n  " + "\n  ".join(bad)
    return checked


def main():
    """Assemble Figure 2, write it, and print the numbers the caption quotes. Returns the path."""
    c_all = load()
    rescale_cell = select_rescale_cell(c_all)
    c = keep_rescale_cell(c_all, rescale_cell)
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 112 * ps.MM))
    gs = GridSpec(2, 4, figure=fig, height_ratios=[0.95, 1.0], hspace=0.30, wspace=0.44)

    ax_a = fig.add_subplot(gs[0, :])
    panel_a(ax_a)
    ps.panel_letter(ax_a, "a", dx=-0.008, dy=0.98)

    axes = {}
    for j, (key, fn, letter) in enumerate((("b", panel_b, "b"), ("c", panel_c, "c"),
                                           ("d", panel_d, "d"), ("e", panel_e, "e"))):
        ax = fig.add_subplot(gs[1, j])
        axes[letter] = (ax, fn(ax, c))
        ps.panel_letter(ax, letter, dx=-0.30, dy=1.02)

    n_checked = check_labels_clear(fig)
    out = ps.save(fig, "fig_paper_F2")
    print(f"label check: {n_checked} labels, none touching data")

    # ---- the numbers the caption quotes -------------------------------------------------------
    ref = {k: np.asarray(c[k][c["arm"] == "control"], float)
           for k in ("n_active", "r2", "dims", "w_sigma_log")}
    print("\n--- per arm: mean +- sd (n networks) ---")
    print(f"{'arm':>12s} {'n':>2s} {'active':>14s} {'r2':>15s} {'r2 noise-free':>15s} "
          f"{'dims (PR)':>13s} {'dims 95%':>10s} {'sd log|W|':>11s} {'log10 q99/q01':>13s}")
    for kind, _, label, _ in ARMS:
        m = c["arm"] == kind
        g = {k: np.asarray(c[k][m], float) for k in
             ("n_active", "r2", "r2_clean", "dims", "dims95", "w_sigma_log", "w_spread")}
        print(f"{kind:>12s} {m.sum():2d} "
              f"{g['n_active'].mean():7.1f} ±{g['n_active'].std(ddof=1):5.1f} "
              f"{g['r2'].mean():8.4f} ±{g['r2'].std(ddof=1):5.4f} "
              f"{g['r2_clean'].mean():8.3f} ±{g['r2_clean'].std(ddof=1):5.3f} "
              f"{g['dims'].mean():6.2f} ±{g['dims'].std(ddof=1):5.2f} "
              f"{g['dims95'].mean():6.1f}    "
              f"{g['w_sigma_log'].mean():5.2f} ±{g['w_sigma_log'].std(ddof=1):4.2f} "
              f"{g['w_spread'].mean():10.2f}")
    print("  r2 is recomputed WITH the network's own noise, which is the quantity stored at training")
    print("  time and what panel (c) draws. The noise-free column is lower and far more variable for")
    print("  every arm, control included - the noise-free trajectory is not one these networks take.")

    print("\n--- against the control ---")
    for kind, _, label, _ in ARMS[1:]:
        m = c["arm"] == kind
        da = np.asarray(c["n_active"][m], float)
        dq = np.asarray(c["r2"][m], float)
        dd = np.asarray(c["dims"][m], float)
        _, pa = welch(da, ref["n_active"])
        _, pq = welch(dq, ref["r2"])
        _, pdi = welch(dd, ref["dims"])
        p_eq, diff, lo, hi = tost(dq, ref["r2"])
        print(f"  {kind:>10s}: active {da.mean() - ref['n_active'].mean():+7.1f} (Welch p={pa:.3g})   "
              f"dims {dd.mean() - ref['dims'].mean():+5.2f} (p={pdi:.3g})   "
              f"r2 {diff:+.2%} [{lo:+.2%}, {hi:+.2%}] (Welch p={pq:.3g}, TOST p={p_eq:.3g})")

    print(f"\n--- every matched rescale cell; the figure draws {rescale_cell} ---")
    ref_r2 = np.mean(c_all["r2"][c_all["arm"] == "control"].astype(float))
    rm = c_all["arm"] == "rescale"
    rows = []
    for cell in sorted(set(c_all["cell"][rm])):
        s = rm & (c_all["cell"] == cell)
        rows.append((float(np.mean(c_all["n_active"][s].astype(float))), cell, s))
    for active, cell, s in sorted(rows, reverse=True):
        r2 = float(np.mean(c_all["r2"][s].astype(float)))
        print(f"  {'->' if cell == rescale_cell else '  '} {cell.split('/')[-1][:58]:60s} "
              f"n={s.sum():2d}  active {active:5.1f}  r2 {r2:.4f}"
              f"{'' if r2 >= ref_r2 * 0.95 else '  (fails the r2 bar)'}"
              f"  dims {float(np.mean(c_all['dims'][s].astype(float))):5.2f}")

    print("\n--- weight magnitudes ---")
    for kind, info in axes["e"][1].items():
        print(f"  {kind:>10s}: median |W| = 10^{info['median']:+.2f}, sd of log|W| "
              f"= {info['sigma_log']:.2f}")
    return out


if __name__ == "__main__":
    main()
