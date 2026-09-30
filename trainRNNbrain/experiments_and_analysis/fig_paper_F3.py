#!/usr/bin/env python3
"""
Manuscript Figure 3 - THE RATE PENALTY, and what it costs. What each penalty asks for, and what
each one does to a trained network across sizes and tasks.

The figure is built around one asymmetry. All three penalties are activity or connectivity
regularisers and all three are one line of code, but they put their minima in different places, and
where the minimum sits decides whether a unit is paid to fire or paid to fall silent:

  (a) WHY A UNIT CAN DIE FOR FREE      the task loss reads the output alone, so a solution
                                       carried by a quarter of the units scores exactly as well as
                                       one carried by all of them.
  (b-d) WHAT EACH PENALTY ASKS FOR     the three terms as functions of the quantity they act on,
                                       drawn from the implementations in Trainer.py at the argument
                                       values the runs were actually trained with, with each
                                       minimum marked. The metabolic cost - the standard activity
                                       regulariser in this literature - is minimised at r = 0, so
                                       it PAYS a unit to be silent. `frm` is two-sided about a
                                       non-zero target, so silence is the most expensive state a
                                       unit can occupy. `rws` is one-sided on a row's effective
                                       in-degree and says nothing about rates at all.

  (e-h) WHAT EACH ONE DOES             four measures against network size, on CDDM, and
  (i-l) ON TWO TASKS, ACROSS SIZE      the same four on the 3-bit flip-flop. Four,
                                       because a remedy that fixes the count and wrecks the rest
                                       has not fixed anything:
                                         active units    does the population come back
                                         task r2         what it costs to bring it back
                                         dimensionality  whether the recovered units do anything
                                                         the original ones were not already doing
                                         rate spread     whether the resulting population looks
                                                         like a cortical one, which is lognormal
                                                         over roughly a hundredfold range

Every point is one trained network, rebuilt from its own config and gated on reproducing its stored
r2 (f3_penalty_cache.py). DMTS is absent: those runs saved no weights and have no penalised arm.

⚠️ THE ARMS WERE NOT TRAINED TO A COMMON BUDGET. The unpenalised flip-flop cells run to 500,000
iterations and the penalised ones to 400,000, and only final weights were saved, so dimensionality
and the rate distribution can only be read at each run's own endpoint. That handicaps the arm with
MORE training, since units keep falling silent after the loss plateaus. It does not carry the
result: read from the participation traces at a matched 400,000 iterations the unpenalised k=3
N=1000 cell holds 198.3 active units against 189.7 at its own endpoint, a difference of 8.6 units
against a none-to-frm gap of 787.

Usage:  python fig_paper_F3.py
Output: img/internal_figures/fig_paper_F3.pdf (+ .svg; vector only - see paperstyle.save)
"""

import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec, GridSpecFromSubplotSpec
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR

CACHE = os.path.join(os.path.dirname(DATA_DIR), "fig_paper_F3_cache.npz")
MATCHED = os.path.join(os.path.dirname(DATA_DIR), "fig_paper_F3_matched.npz")

# Penalty constants read off trainRNNbrain/trainer/Trainer.py and the frm_args/rws_args recorded in
# the runs' own configs, so panel (a) draws the functions that were optimised rather than sketches.
UPV = 100                       # Trainer.UpV, "N units per unit of volume, hard constant"
CAP_FR = 0.3                    # frm_args.cap_fr
G_TOP = G_BOT = 3.0             # frm_args.g_top / g_bot, as trained (the signature default is 5)
TG_DEG = 20                     # rws_args.tg_deg, the target effective in-degree
LAM = dict(frm=0.1, rws=0.05)   # lambda_frm / lambda_rws in every penalised cell

# The four arms. `none` is the reference every panel is read against; `met` is not an arm here, it
# is the comparator drawn in panel (a) only, because no cell in this sweep was trained with it.
ARMS = [("none", "no penalty", ps.BASE),
        ("rws", "sparsity (rws)", ps.COND_COL["rws"]),
        ("frm", "rate penalty (frm)", ps.COND_COL["frm"]),
        ("both", "frm + rws", ps.COND_COL["both"])]

# ⚠️ THE ACTIVE-UNIT PANELS USE A DIFFERENT READ-OUT FROM THE OTHER THREE, and they have to. The
# arms were trained to different budgets - on the flip-flop `rws` stops at 150,000 iterations,
# `frm` and `frm+rws` at 400,000 and the unpenalised cells at 500,000 - and units go on falling
# silent after the loss plateaus. Read at each run's own endpoint, `rws` holds 218 units against
# the unpenalised arm's 190 and looks like an improvement; read at a matched 150,000 it holds 218
# against 263 and is the loss the manuscript reports. So the count comes from the participation
# traces at matched compute (f3_matched_counts.py), the project's standard rule, while
# dimensionality and rate spread come from the rebuilt final weights, which is the only checkpoint
# that was saved. Between 150,000 and the endpoint the penalised arms barely move (979 -> 977,
# 1000 -> 1000 at k=3, N=1000); it is the unpenalised arm that keeps losing units.
#
# (cache field, axis label, log y axis, transform applied to the cached value)
# The spread measure is sigma_log10 of the mean rate over the ACTIVE units, which is the statistic
# Figure 5 and section 6 already use, so the two figures can be read against each other. The cache
# stores the natural-log sd, hence the conversion.
MEASURES = [
    ("n_active", "active units\n(matched compute)", True, None),
    ("r2_clean", "task $r^2$ (noise-free)", False, None),
    ("dim_pr", "dimensionality\n(participation ratio)", False, None),
    ("rate_sigma_log", "rate spread $\\sigma_{\\log_{10}}$\n(active units)", False,
     lambda v: v / np.log(10.0)),
]
# Cortical mean rates are roughly lognormal with sigma_log10 near one decade
# (buzsaki2014log, roxin2011distribution, wohrer2013population), the same reference the manuscript
# already uses in section 6. Every arm here sits below it.
CORTEX_SIGMA = 1.0

# Both rows are the same four measures against network size, one task each. The flip-flop is
# taken at k = 3, the bit count used everywhere else in the paper, and its sizes stop at 2000
# because that is where its penalised cells stop.
FLIPFLOP_K = 3
CDDM_N = (500, 1000, 2000, 5000)
FLIPFLOP_N = (500, 1000, 2000)

# A network that never learned its task has population statistics, and they mean nothing: its
# active units are not solving anything, so averaging them into a cell misreports what the penalty
# did. Runs below this noise-free r2 are dropped and named on stdout, and a cell left with fewer
# than MIN_SEEDS survivors is not plotted. Both fixed before the cache was read, not tuned to it.
MIN_R2 = 0.5
MIN_SEEDS = 2

# Which sweep each arm is read from. NBitFlipFlop_std_pen and NBitFlipFlop_std_penlong both hold
# `pen=frm` cells at the same k and N, trained to different budgets, so an arm that does not name
# its sweep silently pools two conditions and reports six seeds for a three-seed cell.
SWEEP = {
    ("CDDM", "none"): "CDDM_std_g0_drift",
    ("CDDM", "rws"): "CDDM_std_g0_penalties",
    ("CDDM", "frm"): "CDDM_std_g0_penalties",
    ("CDDM", "both"): "CDDM_std_g0_penalties",
    ("flip-flop", "none"): "NBitFlipFlop_std_ksweep",
    ("flip-flop", "rws"): "NBitFlipFlop_std_pen",
    ("flip-flop", "frm"): "NBitFlipFlop_std_penlong",
    ("flip-flop", "both"): "NBitFlipFlop_std_penlong",
}


def load():
    """The per-network cache as a dict of arrays.

    Returns:
        dict field -> array, one entry per gated network.
    Raises:
        SystemExit if the cache has not been built.
    """
    if not os.path.exists(CACHE):
        raise SystemExit(f"{CACHE} missing - run f3_penalty_cache.py first")
    z = np.load(CACHE, allow_pickle=True)
    d = {k: z[k] for k in z.files}
    keep = d["r2_clean"] >= MIN_R2
    if not keep.all():
        for t, k, n, pen in sorted({(str(a), int(b), int(c), str(e)) for a, b, c, e in
                                    zip(d["task"][~keep], d["k"][~keep], d["N"][~keep],
                                        d["pen"][~keep])}):
            lost = int((~keep & (d["task"] == t) & (d["k"] == k) & (d["N"] == n)
                        & (d["pen"] == pen)).sum())
            print(f"  .. dropped {lost} run(s) below r2 {MIN_R2} in {t} k={k} N={n} {pen}")
    return {key: v[keep] for key, v in d.items()}


def cells(d, key, **sel):
    """Mean, SD and n of one measure over the seeds of every cell matching a selection.

    Args:
        d: the cache; key: the measure's field name; transform: a function applied to each value
           before averaging, or None; sel: field -> value filters, where a value may be a scalar or
           a sequence.
    Returns:
        list of (x_value, mean, sd, n) sorted by x, where x is whichever of N or k is passed as a
        sequence - the axis the panel varies.
    """
    transform = sel.pop("transform", None)
    m = np.ones(len(d[key]), bool)
    axis = None
    for f, v in sel.items():
        if np.ndim(v) == 0:
            m &= d[f] == v
        else:
            m &= np.isin(d[f], v)
            axis = f
    out = []
    for x in sorted(set(d[axis][m])):
        v = d[key][m & (d[axis] == x)]
        v = v[np.isfinite(v)]
        if transform is not None:
            v = transform(v)
        if len(v) >= MIN_SEEDS:
            out.append((x, v.mean(), v.std(ddof=1) if len(v) > 1 else 0.0, len(v)))
    return out


def matched_points(dm, task, sweep, pen, xfield, xvals, fixed):
    """Matched-compute active counts for one arm, already aggregated per cell.

    Args:
        dm: the matched-count cache; task, sweep, pen: which arm; xfield: "N" or "k"; xvals: the
        values of it to plot; fixed: dict of the other coordinates held fixed.
    Returns:
        list of (x, mean, sd, n) sorted by x.
    """
    m = ((dm["task"] == task) & (dm["sweep"] == sweep) & (dm["pen"] == pen)
         & np.isin(dm[xfield], xvals))
    for f, v in fixed.items():
        m &= dm[f] == v
    order = np.argsort(dm[xfield][m])
    return [(dm[xfield][m][i], dm["n_active_matched"][m][i], dm["sd"][m][i],
             int(dm["n_seeds"][m][i])) for i in order]


def frm_curve(a, N):
    """The firing-rate-magnitude penalty for one unit, as a function of its activity.

    Mirrors Trainer.Penalties.fr_magnitude_penalty for a single unit at the trained arguments:
    ((cap - a)_+ / cap)^g_bot + ((a - cap)_+ / cap)^g_top, with cap = cap_fr log(1+UpV)/log(1+N).
    `a` is the unit's soft-max activity over time and trials, not its mean rate.

    Args:
        a: activity values (array); N: network size, which sets the cap.
    Returns:
        (penalty values, cap).
    """
    cap = CAP_FR * np.log1p(UPV) / np.log1p(N)
    a = np.asarray(a, float)
    return ((np.maximum(cap - a, 0.0) / cap) ** G_BOT
            + (np.maximum(a - cap, 0.0) / cap) ** G_TOP), cap


def rws_curve(s):
    """The recurrent-weight-sparsity penalty for one row, as a function of its effective in-degree.

    Mirrors Trainer.Penalties.rec_weights_sparsity_penalty for a single row: (S - tg_deg)_+^2 /
    tg_deg^2, where S = (sum|w|)^2 / sum w^2 is the row's effective number of inputs.

    Args:
        s: effective in-degree values (array).
    Returns:
        penalty values, same shape.
    """
    return (np.maximum(np.asarray(s, float) - TG_DEG, 0.0) ** 2) / TG_DEG ** 2


def panel_mechanism(ax):
    """The first cell of row (a): two networks with the same output and different active counts.

    Kept from the previous version of this figure because section 4 opens on it - the task loss
    reads the output alone, so nothing in the objective asks for the units that are not being used.
    The panel carries short labels only and the argument is in the caption; an earlier version put
    the argument inside the panel and it was unreadable (Pavel, 2026-09-21).

    Args:
        ax: a blank Axes.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.figure.canvas.draw()
    dx = 0.030
    dy = ps.square_pitch(ax, dx)
    for col_i, (n_on, lab) in enumerate([(26, "260 active"), (100, "1000 active")]):
        x0 = 0.10 + col_i * 0.52
        ps.unit_grid(ax, x0, 0.86, n_on, 100, col=ps.SLOTS[0], off_col="#d9d8d1",
                     pitch=(dx, dy), s=3.0, lw=0.30)
        ax.text(x0 + 4.5 * dx, 0.95, lab, ha="center", fontsize=5.6, color=ps.INK)
    ax.text(0.5, 0.60, "=", ha="center", va="center", fontsize=11, color=ps.INK)
    ax.text(0.5, 0.17, "same output, same task loss", ha="center", fontsize=5.6, color=ps.INK)


def panel_penalties(axes):
    """Panel (a): the three penalties as functions of what they act on, with their minima marked.

    Args:
        axes: three Axes, for the metabolic cost, frm and rws in that order.
    Returns:
        the frm cap at N = 1000 and at N = 5000, for the caption.
    """
    ax_met, ax_frm, ax_rws = axes

    # metabolic: the field-standard activity regulariser, minimised where the unit is silent
    r = np.linspace(0, 0.5, 400)
    ax_met.plot(r, r ** 2, "-", lw=1.3, color=ps.BAD, zorder=4)
    ax_met.plot(0, 0, "o", ms=4.2, color=ps.BAD, zorder=5, mec="none")
    ax_met.annotate("minimum at $r=0$:\nsilence is free", xy=(0, 0), xycoords="data",
                    xytext=(0.30, 0.93), textcoords="axes fraction",
                    fontsize=5.4, color=ps.BAD, va="top",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=ps.BAD, shrinkA=0, shrinkB=2))
    ax_met.set(xlabel="unit firing rate $r$", ylabel="penalty", xlim=(0, 0.5), ylim=(0, 0.27))
    ax_met.set_title(r"metabolic cost  $\langle r^2\rangle$", fontsize=6.0, color=ps.INK, pad=3)

    # frm: two-sided about a target that falls as 1/log N, so the cap is a function of size
    a = np.linspace(0, 0.5, 600)
    caps = {}
    for N, alpha, lw in ((1000, 1.0, 1.3), (5000, 0.45, 1.0)):
        y, cap = frm_curve(a, N)
        caps[N] = cap
        ax_frm.plot(a, y, "-", lw=lw, color=ps.COND_COL["frm"], alpha=alpha, zorder=4,
                    label=f"$N={N:,}$")
        ax_frm.plot(cap, 0, "o", ms=4.2, color=ps.COND_COL["frm"], alpha=alpha, zorder=5, mec="none")
    ax_frm.annotate("minimum at the target:\nsilence is now the most\nexpensive state",
                    xy=(caps[1000], 0), xycoords="data",
                    xytext=(0.40, 0.70), textcoords="axes fraction", fontsize=5.4,
                    color=ps.COND_COL["frm"], va="top",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=ps.COND_COL["frm"],
                                    shrinkA=0, shrinkB=2))
    ax_frm.legend(loc="upper left", fontsize=5.2, handlelength=1.0)
    ax_frm.set(xlabel="unit activity $a$ (soft-max over time)", xlim=(0, 0.5), ylim=(0, 1.15))
    ax_frm.set_title(r"rate penalty  $\left(\frac{(c-a)_+}{c}\right)^3"
                     r"+\left(\frac{(a-c)_+}{c}\right)^3$", fontsize=6.0, color=ps.INK, pad=3)

    # rws: one-sided on a row's effective in-degree, and silent about rates
    s = np.linspace(0, 60, 400)
    ax_rws.plot(s, rws_curve(s), "-", lw=1.3, color=ps.COND_COL["rws"], zorder=4)
    ax_rws.plot([0, TG_DEG], [0, 0], "-", lw=3.0, color=ps.COND_COL["rws"], alpha=0.35, zorder=3)
    ax_rws.annotate("flat below the target degree,\nand silent about rates",
                    xy=(TG_DEG * 0.45, 0), xycoords="data",
                    xytext=(0.06, 0.93), textcoords="axes fraction", fontsize=5.4,
                    color=ps.COND_COL["rws"], va="top",
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=ps.COND_COL["rws"],
                                    shrinkA=0, shrinkB=2))
    ax_rws.axvline(TG_DEG, color=ps.MUTED, lw=0.6, ls=(0, (3, 2)), zorder=2)
    ax_rws.set(xlabel="effective in-degree $S$ of a row", xlim=(0, 60), ylim=(0, 3.5))
    ax_rws.set_title(r"weight sparsity  $(S-20)_+^2/20^2$", fontsize=6.0, color=ps.INK, pad=3)
    for ax in axes:
        ps.ygrid(ax)
    return caps[1000], caps[5000]


def measure_row(axes, d, dm, task, xfield, xvals, fixed, xlabel, logx):
    """One row of the four measures against one axis, four arms per panel.

    Args:
        axes: four Axes, in MEASURES order; d: the rebuild cache; dm: the matched-count cache;
        task: which task's arms to draw;
        xfield: "N" or "k", the axis varied; xvals: the values of it to plot; fixed: dict of the
        other coordinates held fixed; xlabel: the x axis label; logx: log-scale the x axis.
    Returns:
        list of (measure, arm, x, mean, sd, n) rows for the caption.
    """
    rows = []
    for ax, (key, ylab, logy, fn) in zip(axes, MEASURES):
        for arm, _, col in ARMS:
            if key == "n_active":
                pts = matched_points(dm, task, SWEEP[(task, arm)], arm, xfield, xvals, fixed)
            else:
                pts = cells(d, key, pen=arm, task=task, sweep=SWEEP[(task, arm)], transform=fn,
                            **{xfield: xvals}, **fixed)
            if not pts:
                continue
            x = np.array([p[0] for p in pts], float)
            y = np.array([p[1] for p in pts], float)
            e = np.array([p[2] for p in pts], float)
            ax.errorbar(x, y, yerr=e, fmt="o-", ms=3.0, lw=1.0, color=col, zorder=4, capsize=1.5)
            rows += [(key, arm) + p for p in pts]
        if key == "n_active" and xfield == "N":
            nn = np.array([min(xvals) * 0.8, max(xvals) * 1.25], float)
            ax.plot(nn, nn, "-", lw=0.7, color=ps.MUTED, zorder=2)
            # off the diagonal, not on it: under frm and frm+rws the data lie exactly on this line
            ax.text(nn[0] * 1.05, nn[1] * 0.85, "every unit active", fontsize=5.2,
                    color=ps.MUTED, ha="left", va="top")
        if key == "rate_sigma_log":
            ax.axhline(CORTEX_SIGMA, color=ps.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
            ax.text(0.03, CORTEX_SIGMA, " cortex", transform=ax.get_yaxis_transform(),
                    fontsize=5.2, color=ps.MUTED, va="bottom")
        ax.set(xlabel=xlabel, ylabel=ylab)
        if logx:
            # ticks at the sizes actually trained, written out: matplotlib's log locator labels
            # this range 4x10^2, 6x10^2, 10^3, 2x10^3, which is four ways of writing three numbers
            ax.set_xscale("log")
            ax.set_xticks(list(xvals))
            ax.set_xticklabels([f"{v:,}" for v in xvals])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        if logy:
            ax.set_yscale("log")
        ps.ygrid(ax)
    return rows


def main():
    """Assemble Figure 3 and write it. Returns the output path."""
    d = load()
    if not os.path.exists(MATCHED):
        raise SystemExit(f"{MATCHED} missing - run f3_matched_counts.py first")
    zm = np.load(MATCHED, allow_pickle=True)
    dm = {k: zm[k] for k in zm.files}
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 168 * ps.MM))
    gs = GridSpec(3, 1, figure=fig, height_ratios=[0.86, 1.0, 1.0], hspace=0.62,
                  left=0.065, right=0.985, top=0.955, bottom=0.065)

    gs_a = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[0], wspace=0.40,
                                   width_ratios=[1.0, 1.0, 1.0, 1.0])
    axes_a = [fig.add_subplot(gs_a[0, i]) for i in range(4)]
    panel_mechanism(axes_a[0])
    cap1k, cap5k = panel_penalties(axes_a[1:])
    for ax, letter in zip(axes_a, "abcd"):
        ps.panel_letter(ax, letter, dx=-0.22)

    gs_b = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[1], wspace=0.46)
    axes_b = [fig.add_subplot(gs_b[0, i]) for i in range(4)]
    rows_n = measure_row(axes_b, d, dm, "CDDM", "N", CDDM_N, {},
                         "network size $N$", True)
    for ax, letter in zip(axes_b, "efgh"):
        ps.panel_letter(ax, letter, dx=-0.30)
    axes_b[0].set_title("CDDM, four sizes", fontsize=6.2, color=ps.INK, pad=3, loc="left")

    gs_c = GridSpecFromSubplotSpec(1, 4, subplot_spec=gs[2], wspace=0.46)
    axes_c = [fig.add_subplot(gs_c[0, i]) for i in range(4)]
    rows_k = measure_row(axes_c, d, dm, "flip-flop", "N", FLIPFLOP_N, dict(k=FLIPFLOP_K),
                         "network size $N$", True)
    for ax, letter in zip(axes_c, "ijkl"):
        ps.panel_letter(ax, letter, dx=-0.30)
    axes_c[0].set_title(f"{FLIPFLOP_K}-bit flip-flop, three sizes", fontsize=6.2,
                        color=ps.INK, pad=3, loc="left")

    # the arm key goes between the rows rather than inside a data panel: every panel carries all
    # four arms, and in the active-units panel the frm arm sits exactly where a legend would
    handles = [Line2D([], [], color=c, marker="o", ms=3.0, lw=1.0, label=lab)
               for _, lab, c in ARMS]
    fig.legend(handles=handles, loc="center", bbox_to_anchor=(0.5, 0.678), ncol=4,
               fontsize=6.0, handlelength=1.3, columnspacing=1.8)

    out = ps.save(fig, "fig_paper_F3")

    print("\n--- frm target cap ---")
    print(f"  N=1000: {cap1k:.4f}   N=5000: {cap5k:.4f}  (falls as 1/log N)")
    for tag, rows in (("CDDM vs N", rows_n),
                      (f"{FLIPFLOP_K}-bit flip-flop vs N", rows_k)):
        print(f"\n--- {tag} ---")
        for key, arm, x, mean, sd, n in rows:
            print(f"  {key:12s} {arm:5s} x={x:6g}  {mean:9.3f} +- {sd:7.3f}  n={n}")
    return out


if __name__ == "__main__":
    main()
