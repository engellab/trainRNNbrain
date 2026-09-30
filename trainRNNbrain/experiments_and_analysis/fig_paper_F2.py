#!/usr/bin/env python3
"""
Manuscript Figure 2 - FIVE WAYS TO KEEP UNITS ALIVE, measured on the same four axes.

Every intervention the paper offers, each asked the same four questions:
does the network still solve the task, how many units end up active, how many directions does the
population use, and does the weight distribution still look like the one biology has. Then, for the
cheapest of them, whether any of it survives a change of network size.

  (a) WHAT THE FIVE RULES DO        drawn as the same picture five times - a silent unit among live
                                    ones - so the difference between the rules is a cut edge, a
                                    copied row, a tilted row, a redrawn matrix and a changed loss,
                                    not a paragraph. Four act on the network, one on the objective.
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
                                      synaptic noise     no unit is selected and no weight is
                                                         rewritten. W_rec is redrawn around its mean
                                                         at EVERY timestep, each synapse with sd
                                                         sigma_w * |W_ij|, so the network has to
                                                         stay accurate while its own wiring
                                                         fluctuates. It is the one arm that changes
                                                         the dynamics rather than the update rule.
                                      frm + rws          the only arm that changes the LOSS. A
                                                         firing-rate penalty whose minimum sits at a
                                                         non-zero target makes silence the most
                                                         expensive state, and a weight-sparsity term
                                                         caps a unit's inputs so the rate term
                                                         cannot be paid off with transients. The
                                                         PAIR is the arm: neither half is a remedy
                                                         alone, which is why Figures 3-5 take them
                                                         apart and this one carries them together.
  (b) ACTIVE UNITS                  the scale-free rule, every network drawn, out of 1000.
  (c) HELD-OUT r2                   in ONE condition for every arm: sigma_w = 0 with the recurrent
                                    and input noise they all share, averaged over eight draws. Each
                                    network's own trained condition is the wrong basis for a
                                    comparison - only the synaptic-noise arm is then measured with
                                    its wiring fluctuating, and its stored score is a single draw,
                                    untrustworthy once sigma_w is large. The trained-condition and
                                    fully noise-free values are printed beside it.
  (d) DIMENSIONS USED               participation ratio of the noise-free rate covariance over the
                                    active units: how many directions the population actually uses.
                                    The count of components carrying 95% of the variance is printed
                                    beside it, because a ratio and a variance threshold can
                                    disagree; here they move the same way.
  (e) WEIGHT MAGNITUDES             the distribution of |W_rec| over all 10^6 entries, as a LOG
                                    density: what separates the arms is a tail, which a linear
                                    density flattens into one peak. Cortical synaptic strengths are
                                    lognormal over roughly two orders of magnitude (Song et al. 2005;
                                    Lefort et al. 2009), so an intervention that recruits units by
                                    manufacturing a weight distribution biology does not produce has
                                    bought them with an artifact. This is the panel that catches
                                    one: three arms leave the range where the control has it, and
                                    rescale does not.
  (f), (g) DOES IT SURVIVE SIZE     every arm measured at more than one size, against the control,
                                    in active units and in r2. Which arms those are depends on what
                                    has finished training, so the panels fill in as the size series
                                    land rather than being edited. The diagonal in (f) is "every
                                    unit active", the thing the gap is measured against.

MATCHED, AND WHY THAT COST A CELL. Every arm is gamma = 0, 3-bit flip-flop, 40,000 iterations,
lr 1e-3, weight decay 1e-6, sigma_rec = sigma_inp = 0.05, batch 1024 - the intervention is the only
difference. The duplication sweep at gamma = 0.1 (`ff_revive_g01_fix`) is NOT pooled in: gamma is
cubic saturation in the dynamics, so it changes the base network and a cross-gamma comparison is not
like for like. Duplication here is the corrected construction of 2026-09-24; the cells carrying the
earlier detuned self-weight are excluded (see f2_remedies_cache.py). The N = 4000 control comes from
`paper_grid`, because `ff_revive` never ran one; its config was checked field by field against the
N = 1000 control rather than assumed from the folder name.

TWO ARMS ARE GRIDS AND ARE SHOWN AT ONE SETTING. Rescale was developed over five sweeps -- an
alpha-only form, then a fixed activity target that holds a boosted unit until it fires, then a
refractory tail after it graduates -- and the early alpha-only form recruits nothing at all. Quoting
that form as "rescale" reports the rule at its weakest, and pooling the grid averages the developed
rule with it; the first version of this figure did exactly that and had rescale sitting below the
control. Synaptic noise is a sweep over sigma_w in the same way. Both cells are picked by
`select_cell`, on a rule fixed before the cells were scored: most active units among the cells whose
r2 is within 5% of the control's. Every matched cell is printed under the figure, the drawn one
marked.

WHAT REPLACED WHAT. The previous Figure 2 was dropout alone: a mute schematic, a characterisation of
the dropout sampler, the 150k training curve and a cost panel. The sampler panel documented a
sampler that was corrected on 2026-09-21 and was already marked for redesign; the training curve and
the cost panel are superseded by (b) and (c) here, which carry the same read-out for four
interventions instead of one.

CRITERION AND READ-OUT as Figure 1: scale-free participation, matched compute, every network drawn.
`check_labels_clear` asserts that no label overlaps another label or any drawn datum, testing the
ink of each curve rather than its bounding box.

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
N_MAIN = 1000                    # the size the four-way comparison is made at
SIZES = (500, 1000, 2000, 4000)  # the sizes the dropout series covers

# (key in the cache, x tick label, full name, colour). The x tick labels are short because panel
# (a) names the rules directly above them. The control is neutral ink: it is the reference every
# remedy is measured against, not a fifth condition.
ARMS = [("control", "none", "no intervention", ps.BASE),
        ("mute", "dropout", "dropout: mute", ps.COND_COL["mute"]),
        ("duplicate", "duplicate", "prune + duplicate", ps.COND_COL["duplicate"]),
        ("rescale", "rescale", "rescale", ps.COND_COL["rescale"]),
        ("synnoise", "syn. noise", "synaptic noise", ps.COND_COL["synnoise"]),
        ("both", "frm + rws", "penalty: frm + rws", ps.COND_COL["both"])]

# The arms whose cache holds a hyperparameter grid rather than one setting, so the figure has to
# choose. See `select_cell` for the rule, which is fixed before the cells are scored.
# EVERY intervention is a grid, not just two: dropout has a rate x beta sweep in `std_bernoulli`
# that the first version of this figure never saw, duplication has a jitter sweep, rescale five
# sweeps and synaptic noise a noise ladder. All four are therefore shown at the setting `select_cell`
# picks, on one rule, and the full grid of each is printed beneath the figure.
GRID_ARMS = ("mute", "duplicate", "rescale", "synnoise", "both")

# THE READ-OUT. `r2` is each network scored in its OWN trained condition - the quantity stored at
# training time, which the cache builder uses as its gate. It is the wrong thing to compare arms on:
# only the synaptic-noise arm is then measured with its wiring fluctuating, and its stored score is
# a single draw, untrustworthy once sigma_w is large. `r2_common` is every arm under ONE condition,
# sigma_w = 0 with the recurrent and input noise they all share, averaged over eight draws.
R2_KEY = "r2_common"


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
        c: the cache dict, already restricted to one N and one cell per grid arm;
        key: the field to pull, e.g. 'n_active'.
    Returns:
        list of 1-D float arrays, one per arm.
    """
    return [np.asarray(c[key][c["arm"] == a], float) for a, _, _, _ in ARMS]


# THE OPERATING POINT OF EACH ARM, which is NOT chosen from these results. Every arm is a grid, so
# one cell has to be drawn, and picking it by "most active units" is unsound twice over: active
# units is one of the four measures the figure compares, so maximising it biases the other three,
# and the grids are dense enough that the rule chases noise - it put duplication at copy_noise 3.0
# over the paper's own cell on a 10-unit difference, and rescale at a cell that recruits 583 units
# while the population collapses to 1.97 dimensions.
#
# Each arm is therefore drawn at the setting its SIZE SERIES was launched with. Those settings were
# fixed by the size-series and paper-grid designs before this figure existed, independently of what
# is measured here, and using them makes panels (b)-(e) and (f)-(g) the same networks rather than
# two different choices of cell. `select_cell` survives as the fallback for an arm with no committed
# operating point, and the full grid of every arm is printed beneath the figure either way.
OPERATING_POINT = {
    "mute": ("do=mute_rate=0.20_beta=4",),          # paper_grid / dropout_sizes
    "duplicate": ("paper_grid/EqType=h_N=",),       # copy_noise 1.0, capped 0.025, maturity 1000
    "rescale": ("tgt=10.0", "arm=rescale_tgt10"),   # target ladder's operating point
    "synnoise": ("sw=1.0", "sw1.0"),                # the level the size series runs
    "both": ("ff_both40k",),                        # the only matched frm+rws cell
}


def pick_cell(c, arm, margin_frac=0.05):
    """The cell of a grid arm the figure draws: its operating point, or the fallback rule.

    Args:
        c: the cache dict restricted to one N; arm: the grid arm;
        margin_frac: passed to the fallback rule.
    Returns:
        (cell string or None, "operating point" or "fallback rule" or None).
    """
    cells = sorted(set(c["cell"][c["arm"] == arm]))
    for pat in OPERATING_POINT.get(arm, ()):
        hits = [x for x in cells if pat in x]
        if len(hits) == 1:
            return hits[0], "operating point"
        if len(hits) > 1:
            raise SystemExit(f"{arm}: {len(hits)} cells match the operating point {pat!r}: {hits}")
    got = select_cell(c, arm, margin_frac)
    return got, (None if got is None else "fallback rule")


def select_cell(c, arm, margin_frac=0.05):
    """Fallback when an arm has no committed operating point: most active units, task intact.

    THE RULE, fixed before the cells were scored: take the cell with the highest mean active-unit
    count among those whose mean r2 is within `margin_frac` of the control's - the same equivalence
    margin the cost read-out uses. Recruiting units by destroying the task is not recruiting, and
    the bar does real work: one rescale cell reaches 983 active units at r2 0.62 and is excluded.

    Args:
        c: the cache dict restricted to one N; arm: the arm to choose within;
        margin_frac: how far below the control's r2 a cell may sit.
    Returns:
        the chosen cell string, or None if no cell qualifies.
    """
    ref = np.mean(c[R2_KEY][c["arm"] == "control"].astype(float))
    best, best_active = None, -1.0
    for cell in sorted(set(c["cell"][c["arm"] == arm])):
        m = (c["arm"] == arm) & (c["cell"] == cell)
        if np.mean(c[R2_KEY][m].astype(float)) < ref * (1 - margin_frac):
            continue
        active = float(np.mean(c["n_active"][m].astype(float)))
        if active > best_active:
            best, best_active = cell, active
    return best


def restrict(c, n_units=None, chosen=None):
    """Keep the rows one panel should see: one network size, one cell per grid arm.

    Args:
        c: the full cache dict; n_units: keep only this N, or None for every size;
        chosen: {arm: cell} for the grid arms, or None to keep all of their cells.
    Returns:
        a new dict of the same keys, filtered. 'log_bins' is passed through unfiltered.
    """
    keep = np.ones(len(c["arm"]), bool)
    if n_units is not None:
        keep &= c["N"].astype(int) == n_units
    for arm, cell in (chosen or {}).items():
        keep &= (c["arm"] != arm) | (c["cell"] == cell)
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
    """Panel (a): the four interventions, each drawn on the same three-unit picture.

    Every sub-schematic has the same anatomy, so a reader can set them against each other: the rule's
    name, the operation written above the arrow that performs it, a label under each unit the rule
    touches saying what that unit is, and one line at the bottom giving the consequence. The rows sit
    at the same heights in all four, and everything a rule draws below the units - the read-out under
    `mute`, the inhibitory arc under `rescale` - is kept above the label row.

    Args:
        ax: a blank axes spanning the figure's top row.
    Returns:
        None.
    """
    ps.blank(ax)
    n = len(ARMS) - 1
    ax.set(xlim=(0, n * 1.06 + 0.02), ylim=(0, 1))
    y = 0.68                                                  # the row of units
    y_title, y_op, y_unit, y_foot = 1.00, 0.845, 0.34, 0.06
    for i, (kind, _, title, col) in enumerate(ARMS[1:]):
        x0 = i * 1.06
        left, mid, right = x0 + 0.18, x0 + 0.50, x0 + 0.82
        ax.text(x0 + 0.50, y_title, title, ha="center", va="top", fontsize=6.2,
                color=col, fontweight="bold", linespacing=1.15)

        if kind == "mute":
            # the unit the sampler picks is an ACTIVE one, and only its read-out weight is cut
            _units(ax, [left, mid, right], y, ["live", "live", "silent"], col)
            ry = 0.40
            ps.box(ax, x0 + 0.50 - 0.15, ry, 0.30, 0.085, "read-out", col=ps.MUTED,
                   face="#f2f1ec", lw=0.6, fs=5.4)
            for j, xi in enumerate((left, mid, right)):
                cut = (j == 0)
                ps.arrow(ax, (xi, y - 0.055), (x0 + 0.50 + (j - 1) * 0.085, ry + 0.085),
                         col=ps.BAD if cut else ps.MUTED, lw=0.75, ls=":" if cut else "-")
                if cut:
                    mx, my = (xi + x0 + 0.50 - 0.085) / 2, (y - 0.055 + ry + 0.085) / 2
                    for sgn in (1, -1):
                        ax.plot([mx - 0.026, mx + 0.026], [my - sgn * 0.028, my + sgn * 0.028],
                                lw=1.0, color=ps.BAD, zorder=7)
            ax.text(x0 + 0.50, y_op, "zero its read-out weight", ha="center", va="center",
                    fontsize=5.6, color=col)
            ax.text(left, y_unit, "sampled:\nan active unit", ha="center", va="top", fontsize=5.3,
                    color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "the loss cannot see it,\n"
                    "but it still drives the rest", ha="center", va="top", fontsize=5.4,
                    color=ps.INK, linespacing=1.35)

        elif kind == "duplicate":
            # the silent unit is deleted and rebuilt as a copy of the live donor on the left
            _units(ax, [left, mid, right], y, ["live", "live", "new"], col)
            ps.arrow(ax, (left, y), (right, y), col=col, rad=-0.26, lw=0.9, shrink=7.0,
                     mutation_scale=6)
            ax.text(x0 + 0.50, y_op, "copy the donor's incoming row", ha="center", va="center",
                    fontsize=5.6, color=col)
            ax.text(left, y_unit, "donor:\nan active unit", ha="center", va="top", fontsize=5.3,
                    color=ps.MUTED, linespacing=1.3)
            ax.text(right, y_unit, "pruned silent unit,\nrebuilt as the copy", ha="center",
                    va="top", fontsize=5.3, color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "the donor's outgoing weights\n"
                    "halve, so the output holds", ha="center", va="top",
                    fontsize=5.1, color=ps.INK, linespacing=1.35)

        elif kind == "rescale":
            # the silent unit keeps every synapse it has; only their balance changes. The label goes
            # under `right`, which is the silent one - under `left` it named a unit the rule does
            # not touch.
            _units(ax, [left, mid, right], y, ["live", "live", "silent"], col)
            ps.arrow(ax, (left, y), (right, y), col=col, lw=0.9, rad=-0.26, shrink=7.0,
                     mutation_scale=6)
            ps.arrow(ax, (mid, y), (right, y), col=ps.MUTED, lw=0.9, rad=0.55, shrink=7.0,
                     mutation_scale=6)
            ax.text(x0 + 0.50, y_op, r"excitatory inputs $\times\,\alpha$", ha="center",
                    va="center", fontsize=5.6, color=col)
            ax.text(x0 + 0.60, 0.545, r"inhibitory inputs $\div\,\alpha$", ha="center",
                    va="top", fontsize=5.6, color=ps.MUTED)
            ax.text(right, y_unit, "silent unit,\nkept in place", ha="center", va="top",
                    fontsize=5.3, color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "no new wiring: more excitation,\n"
                    "less inhibition, same norm", ha="center", va="top",
                    fontsize=5.1, color=ps.INK, linespacing=1.35)

        elif kind == "both":
            # the only arm that changes the LOSS. Nothing is done to any unit: silence simply stops
            # being free, and the sparsity term stops the rate term being paid off with transients.
            _units(ax, [left, mid, right], y, ["live", "live", "silent"], col)
            ry = 0.40
            ps.box(ax, x0 + 0.50 - 0.19, ry, 0.38, 0.085, "loss + penalty", col=col,
                   face="#f2f1ec", lw=0.7, fs=5.4)
            for xi in (left, mid, right):
                ps.arrow(ax, (xi, y - 0.055), (x0 + 0.50 + (xi - mid) * 0.55, ry + 0.085),
                         col=ps.MUTED, lw=0.75)
            ax.text(x0 + 0.50, y_op, "make silence expensive", ha="center", va="center",
                    fontsize=5.6, color=col)
            ax.text(right, y_unit, "silent unit,\nnow costly", ha="center", va="top",
                    fontsize=5.3, color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "the loss changes, not the\n"
                    "units; transients cannot pay", ha="center", va="top",
                    fontsize=5.1, color=ps.INK, linespacing=1.35)

        else:
            # no unit is selected and no weight is rewritten: the whole matrix is redrawn around its
            # mean on every timestep, which is drawn as several arcs where the others draw one
            _units(ax, [left, mid, right], y, ["live", "live", "silent"], col)
            for k, (rad, alpha) in enumerate(((-0.40, 0.30), (-0.26, 0.55), (-0.12, 0.95))):
                p = ps.arrow(ax, (left, y), (right, y), col=col, lw=0.9, rad=rad, shrink=7.0,
                             mutation_scale=6)
                p.set_alpha(alpha)
            ax.text(x0 + 0.50, y_op, "redraw every weight, every step", ha="center", va="center",
                    fontsize=5.6, color=col)
            ax.text(right, y_unit, "silent unit,\nnot targeted", ha="center", va="top",
                    fontsize=5.3, color=ps.MUTED, linespacing=1.3)
            ax.text(x0 + 0.50, y_foot, "no unit is singled out; the\n"
                    "dynamics are noisy, not the rule", ha="center", va="top",
                    fontsize=5.1, color=ps.INK, linespacing=1.35)


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
    ax.axhline(N_MAIN, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
    ax.set_ylim(0, N_MAIN * 1.12)
    # Offsets are in POINTS from the highest seed of each arm, not in data units from its mean: the
    # arms differ in spread (12 rescale seeds against 3 elsewhere), so a fixed data-unit offset
    # clears the dots in one arm and lands on them in the next.
    ax.annotate(f"all {N_MAIN}", (len(ARMS) - 0.5, N_MAIN), textcoords="offset points",
                xytext=(0, 3), ha="right", va="bottom", fontsize=5.4, color=ps.MUTED)
    for x, g, (m, sd, n) in zip(xs, groups, res):
        if not n:                       # a cell that has not finished training yet
            continue
        ax.annotate(f"{m:.0f}", (x, g.max()), textcoords="offset points", xytext=(0, 5),
                    ha="center", va="bottom", fontsize=5.8, color=ps.INK)
    return res


def panel_c(ax, c):
    """Panel (c): held-out r2 per network. Returns per-arm (mean, sd, n)."""
    xs = _cat_axes(ax, "held-out $r^2$")
    groups = by_arm(c, R2_KEY)
    res = ps.strip(ax, xs, groups, [col for _, _, _, col in ARMS], rng=np.random.default_rng(4))
    ref = res[0][0]
    ax.axhline(ref, color=ps.BASE, lw=0.7, ls=":", zorder=1)
    # No +-5% equivalence band is drawn: the margin is 0.047 of r2 and the largest cost here is
    # 0.017, so the band would fill the panel and say nothing. The TOST verdicts are printed below.
    # The delta row sits in a band cleared BELOW the lowest seed drawn, not at a fixed ylim: which
    # rescale cell is drawn changes the floor of this panel by more than the band is tall.
    drawn = [g for g in groups if len(g)]
    lo = min(g.min() for g in drawn)
    ax.set_ylim(lo - 0.011, max(0.952, max(g.max() for g in drawn) + 0.003))
    for x, (m, sd, n) in zip(xs[1:], res[1:]):
        if not n:
            continue
        ax.text(x, lo - 0.0085, f"{(m - ref) / ref:+.1%}", ha="center", va="center", fontsize=4.3,
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
    for cut, name in ((1e9, "billion"), (1e6, "million")):
        if v >= cut:
            return f"{v / cut:.0f} {name}-fold"
    return f"{round(v, -2):,.0f}-fold"


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
    # the headroom above the peak has to hold a legend that grows with the number of arms
    ax.set(xlim=(min(lo_x) - 0.3, max(hi_x) + 0.3), yscale="log", ylim=(2e-4, 40.0),
           xlabel="recurrent weight\n$\\log_{10}|W_{ij}|$", ylabel="density")
    ax.legend(loc="upper left", fontsize=5.2, handlelength=0.9, borderpad=0.1,
              borderaxespad=0.2, ncol=2, columnspacing=0.8, labelspacing=0.25)
    # The panel's result, in the title so it cannot land on a curve. The magnitude RANGE is the
    # statistic quoted rather than the sd, because a fold-range is a number a reader can picture.
    others = max(out[k]["spread"] for k in out if k != "rescale")
    ax.set_title(f"a {_fold(others)} range of magnitudes,\n"
                 f"{_fold(out['rescale']['spread'])} under rescale",
                 fontsize=5.6, color=ps.MUTED, linespacing=1.3, pad=3)
    ps.ygrid(ax)
    return out


def _size_series(c, arm, key):
    """Per-size mean of one field for one arm, choosing a cell per size by the same rule.

    An arm whose cache holds a grid needs one cell per SIZE, not one cell overall: the size series
    trains its own cell at each N, and at N = 1000 there are a dozen to choose between. The rule is
    the one `select_cell` applies, evaluated inside each size against that size's own control.

    Args:
        c: the full cache dict (every N); arm: the arm key; key: the field to pull.
    Returns:
        (sizes, means, sds, ns) as four arrays, over the sizes where that arm has networks.
    """
    sizes, mu, sd, ns = [], [], [], []
    for n_units in SIZES:
        at_n = restrict(c, n_units=n_units)
        if not (at_n["arm"] == arm).any() or not (at_n["arm"] == "control").any():
            continue
        if arm in GRID_ARMS:
            cell, _ = pick_cell(at_n, arm)
            if cell is None:
                continue
            at_n = restrict(at_n, chosen={arm: cell})
        g = at_n[key][at_n["arm"] == arm].astype(float)
        if not len(g):
            continue
        sizes.append(n_units)
        mu.append(g.mean())
        sd.append(g.std(ddof=1) if len(g) > 1 else 0.0)
        ns.append(len(g))
    return np.array(sizes, float), np.array(mu), np.array(sd), np.array(ns)


def _size_arms(c):
    """The arms worth drawing in the size panels: those measured at more than one size.

    Args:
        c: the full cache dict.
    Returns:
        list of (arm, short label, colour) in ARMS order.
    """
    out = []
    for arm, short, _, col in ARMS:
        n_sizes = len({int(v) for v in c["N"][c["arm"] == arm]})
        if n_sizes > 1:
            out.append((arm, short, col))
    return out


def panel_f(ax, c):
    """Panel (f): active units against network size, every arm measured at more than one size.

    Args:
        ax: axes; c: the FULL cache dict, every size.
    Returns:
        dict {N: {arm: (mean, n)}}.
    """
    out = {}
    ns_all = np.array(SIZES, float)
    ax.plot(ns_all, ns_all, lw=0.7, ls=":", color=ps.FAINT, zorder=1)
    ax.annotate("every unit active", (ns_all[-1], ns_all[-1]), textcoords="offset points",
                xytext=(-2, 3), ha="right", va="bottom", fontsize=5.2, color=ps.MUTED)
    for arm, short, col in _size_arms(c):
        sizes, mu, sd, ns = _size_series(c, arm, "n_active")
        if not len(sizes):
            continue
        ax.plot(sizes, mu, "-o", lw=1.1, ms=3.0, color=col, mec="white", mew=0.5, zorder=4,
                label=short)
        for n_units, v, k in zip(sizes, mu, ns):
            out.setdefault(int(n_units), {})[arm] = (float(v), int(k))
    ax.set(xscale="log", yscale="log", xlabel="network size $N$", ylabel="active units",
           xlim=(400, 5200), ylim=(150, 5200))
    ax.set_xticks(list(SIZES))
    ax.set_xticklabels([str(s) for s in SIZES])
    ax.set_yticks([200, 500, 1000, 2000, 4000])
    ax.set_yticklabels(["200", "500", "1000", "2000", "4000"])
    ax.legend(loc="upper left", fontsize=5.5, handlelength=1.1, borderaxespad=0.2, ncol=2,
              columnspacing=0.9)
    ps.ygrid(ax)
    return out


def panel_g(ax, c):
    """Panel (g): held-out r2 in the common test condition, against network size.

    Args:
        ax: axes; c: the FULL cache dict, every size.
    Returns:
        dict {N: {arm: mean}}.
    """
    out = {}
    for arm, short, col in _size_arms(c):
        sizes, mu, sd, ns = _size_series(c, arm, R2_KEY)
        if not len(sizes):
            continue
        ax.plot(sizes, mu, "-o", lw=1.1, ms=3.0, color=col, mec="white", mew=0.5, zorder=4,
                label=short)
        for n_units, v in zip(sizes, mu):
            out.setdefault(int(n_units), {})[arm] = float(v)
    ax.set(xscale="log", xlabel="network size $N$", ylabel="held-out $r^2$", xlim=(400, 5200))
    ax.set_xticks(list(SIZES))
    ax.set_xticklabels([str(s) for s in SIZES])
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
    # Figure-level text belongs to no axes, so it is tested against every panel's ink and against
    # every panel's labels. The task banner sits in a gap between rows and has to stay in one.
    fig_labels = [(t.get_text()[:40], t.get_window_extent(r)) for t in fig.texts
                  if t.get_text().strip()]
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
            for tb, bb in labels[i + 1:] + fig_labels:
                if ba.overlaps(bb):
                    bad.append(f"{where}: {ta!r} overlaps {tb!r}")

        if where == "schematic":
            continue
        labels = labels + fig_labels
        ink = [p for a in list(ax.lines) + list(ax.collections)
               for p in (_ink_points(a, r),) if len(p)]
        for text, box in labels:
            hit = any(((p[:, 0] > box.x0 - pad) & (p[:, 0] < box.x1 + pad) &
                       (p[:, 1] > box.y0 - pad) & (p[:, 1] < box.y1 + pad)).any() for p in ink)
            if hit:
                bad.append(f"{where}: {text!r} sits on the data")
    assert not bad, "Figure 2 label collisions:\n  " + "\n  ".join(bad)
    return checked


def task_banner(fig, ax_above, ax_below):
    """Write the task and training settings in the gap between the schematic and the data panels.

    Every panel below it is one task at one budget, and a reader should not have to find that in a
    caption. It goes in the FIGURE's coordinates, in the gap between two rows, so it belongs to all
    four panels rather than sitting inside one of them.

    Args:
        fig: the figure; ax_above, ax_below: the axes bounding the gap it is centred in.
    Returns:
        the Text artist.
    """
    top, bottom = ax_above.get_position().y0, ax_below.get_position().y1
    # The thousands separator is escaped for mathtext on its own. Adjacent string literals are
    # concatenated before any method call, so a .replace() chained onto the last one rewrites the
    # whole sentence - which is how the banner first printed "ReLU RNN{,} 40{,}000 iterations{,}".
    n_main = f"{N_MAIN:,}".replace(",", "{,}")
    return fig.text(0.5, (top + bottom) / 2,
                    "all interventions trained on the 3-bit flip-flop: ReLU RNN, 40,000 iterations, "
                    "$\\gamma=0$, three networks per cell    ·    "
                    f"b–e at $N={n_main}$",
                    ha="center", va="center", fontsize=6.0, color=ps.MUTED,
                    bbox=dict(boxstyle="round,pad=0.35", facecolor="#f2f1ec", edgecolor=ps.GRID,
                              linewidth=0.5))


def main():
    """Assemble Figure 2, write it, and print the numbers the caption quotes. Returns the path."""
    c_all = load()
    at_main = restrict(c_all, n_units=N_MAIN)
    picked = {a: pick_cell(at_main, a) for a in GRID_ARMS}
    chosen = {a: cell for a, (cell, _) in picked.items()}
    c = restrict(c_all, n_units=N_MAIN, chosen=chosen)
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 168 * ps.MM))
    gs = GridSpec(3, 4, figure=fig, height_ratios=[1.18, 1.0, 0.92], hspace=0.62, wspace=0.46)

    ax_a = fig.add_subplot(gs[0, :])
    panel_a(ax_a)
    ps.panel_letter(ax_a, "a", dx=-0.008, dy=0.98)

    results = {}
    for j, (letter, fn) in enumerate((("b", panel_b), ("c", panel_c), ("d", panel_d),
                                      ("e", panel_e))):
        ax = fig.add_subplot(gs[1, j])
        results[letter] = fn(ax, c)
        ps.panel_letter(ax, letter, dx=-0.30, dy=1.02)
        if letter == "b":
            ax_b = ax

    ax_f = fig.add_subplot(gs[2, 0:2])
    results["f"] = panel_f(ax_f, c_all)
    ps.panel_letter(ax_f, "f", dx=-0.135, dy=1.02)
    ax_g = fig.add_subplot(gs[2, 2:4])
    results["g"] = panel_g(ax_g, c_all)
    ps.panel_letter(ax_g, "g", dx=-0.135, dy=1.02)

    task_banner(fig, ax_a, ax_b)
    n_checked = check_labels_clear(fig)
    out = ps.save(fig, "fig_paper_F2")
    print(f"label check: {n_checked} labels, none touching data")

    # ---- the numbers the caption quotes -------------------------------------------------------
    ref = {k: np.asarray(c[k][c["arm"] == "control"], float)
           for k in ("n_active", R2_KEY, "dims", "w_sigma_log")}
    print(f"\n--- at N = {N_MAIN}: mean +- sd (n networks) ---")
    print(f"{'arm':>12s} {'n':>2s} {'active':>14s} {'r2 common':>15s} {'r2 as trained':>9s} "
          f"{'r2 clean':>9s} {'dims (PR)':>13s} {'dims 95%':>9s} {'sd log|W|':>11s} "
          f"{'log10 q99/q01':>13s}")
    for kind, _, label, _ in ARMS:
        m = c["arm"] == kind
        if not m.any():
            continue
        g = {k: np.asarray(c[k][m], float) for k in
             ("n_active", R2_KEY, "r2", "r2_clean", "dims", "dims95", "w_sigma_log",
              "w_spread")}
        print(f"{kind:>12s} {m.sum():2d} "
              f"{g['n_active'].mean():7.1f} ±{g['n_active'].std(ddof=1):5.1f} "
              f"{g[R2_KEY].mean():8.4f} ±{g[R2_KEY].std(ddof=1):5.4f} "
              f"{g['r2'].mean():9.4f} {g['r2_clean'].mean():9.4f} "
              f"{g['dims'].mean():6.2f} ±{g['dims'].std(ddof=1):5.2f} "
              f"{g['dims95'].mean():6.1f}    "
              f"{g['w_sigma_log'].mean():5.2f} ±{g['w_sigma_log'].std(ddof=1):4.2f} "
              f"{g['w_spread'].mean():10.2f}")
    print("  'r2 common' is the read-out and what panel (c) draws: sigma_w = 0 with the recurrent")
    print("  and input noise every arm shares, averaged over eight draws, so one condition scores")
    print("  every arm. 'as trained' is each network in its own condition - the quantity stored at")
    print("  training time, used as the rebuild's gate, not as a comparison. 'clean' is fully")
    print("  noise-free, which is a trajectory none of these networks normally takes.")

    print("\n--- against the control ---")
    for kind, _, label, _ in ARMS[1:]:
        m = c["arm"] == kind
        if not m.any():
            continue
        da, dq, dd = (np.asarray(c[k][m], float) for k in ("n_active", R2_KEY, "dims"))
        _, pa = welch(da, ref["n_active"])
        _, pq = welch(dq, ref[R2_KEY])
        _, pdi = welch(dd, ref["dims"])
        p_eq, diff, lo, hi = tost(dq, ref[R2_KEY])
        print(f"  {kind:>10s}: active {da.mean() - ref['n_active'].mean():+7.1f} (Welch p={pa:.3g})"
              f"   dims {dd.mean() - ref['dims'].mean():+5.2f} (p={pdi:.3g})"
              f"   r2 {diff:+.2%} [{lo:+.2%}, {hi:+.2%}] (Welch p={pq:.3g}, TOST p={p_eq:.3g})")

    for arm in GRID_ARMS:
        cell, how = picked[arm]
        print(f"\n--- every matched {arm} cell; the figure draws {cell} ({how}) ---")
        ref_r2 = np.mean(at_main[R2_KEY][at_main["arm"] == "control"].astype(float))
        rm = at_main["arm"] == arm
        rows = [(float(np.mean(at_main["n_active"][rm & (at_main["cell"] == cl)].astype(float))),
                 cl, rm & (at_main["cell"] == cl)) for cl in sorted(set(at_main["cell"][rm]))]
        for active, cl, s in sorted(rows, reverse=True):
            r2 = float(np.mean(at_main[R2_KEY][s].astype(float)))
            print(f"  {'->' if cl == chosen[arm] else '  '} {cl.split('/')[-1][:56]:58s} "
                  f"n={s.sum():2d}  active {active:5.1f}  r2 {r2:.4f}"
                  f"{'' if r2 >= ref_r2 * 0.95 else '  (fails the r2 bar)'}"
                  f"  dims {float(np.mean(at_main['dims'][s].astype(float))):5.2f}")

    print("\n--- against the control, by network size (panels f and g) ---")
    print(f"{'arm':>12s} {'N':>6s} {'control':>9s} {'arm':>9s} {'gained':>8s} {'ratio':>6s} "
          f"{'control r2':>11s} {'arm r2':>9s}")
    for arm, _, _, _ in ARMS[1:]:
        for n_units in SIZES:
            cf, cg = results["f"].get(n_units, {}), results["g"].get(n_units, {})
            if arm not in cf or "control" not in cf:
                continue
            ctl, val = cf["control"][0], cf[arm][0]
            print(f"{arm:>12s} {n_units:6d} {ctl:9.1f} {val:9.1f} {val - ctl:+8.1f} "
                  f"{val / ctl:5.2f}x {cg['control']:11.4f} {cg[arm]:9.4f}")
    missing = [(a, n) for a, _, _, _ in ARMS[1:] for n in SIZES
               if a not in results["f"].get(n, {})]
    if missing:
        print("  absent from the size panels (an arm needs two sizes to be drawn at all): "
              + ", ".join(f"{a} N={n}" for a, n in missing))

    print("\n--- weight magnitudes ---")
    for kind, info in results["e"].items():
        print(f"  {kind:>10s}: median |W| = 10^{info['median']:+.2f}, sd of log|W| "
              f"= {info['sigma_log']:.2f}, range 10^{info['spread']:.2f}")
    return out


if __name__ == "__main__":
    main()
