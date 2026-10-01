#!/usr/bin/env python3
"""
Presentation figures: one claim per figure, one panel per figure, no composites.

A manuscript figure packs six panels onto a page because the reader controls the pace. A talk is the
opposite: the speaker controls the pace, so a slide carrying six panels is six slides the audience
cannot be pointed at one at a time. This script re-emits the manuscript panels as standalone figures
and builds one figure per failed intervention, so `docs/presentation.md` can show them in order.

NOTHING HERE RE-ANALYSES ANYTHING. The panels are the manuscript's own panel functions called on a
one-axes figure, and the intervention figures read their cells from `fig_paper_F1`'s own family
tables through its own loaders. If a number here disagrees with the paper, that is a bug in this
file, not a second opinion - which is the point: a talk that quotes different numbers from the paper
it is about has two problems.

THE INTERVENTION FIGURES are the expansion of Figure 1d. That panel plots every knob we varied as a
change from its own reference, which is the right summary and the wrong slide: it asks an audience
to read twenty intervals at once. Here each family gets its own figure, drawn as COUNTS rather than
changes so that the reference is visible on the same axis, with every seed shown.

Usage:  python fig_slides.py            # writes every slide
        python fig_slides.py --list     # names only
Output: img/internal_figures/slide_*.pdf (+ .svg, which is what the Markdown deck links)
"""

import argparse
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_paper_F1 as F1
import pr_matrix as PR
import fig_paper_F2 as F2

W = 110 * ps.MM             # a single-panel slide: wider than a journal column, same 7 pt type
H = 72 * ps.MM


def _family_values(family, trace=True):
    """Active-unit counts for a family's reference and each of its items.

    Args:
        family: an entry of F1.TRACE_FAMILIES, or F1.ARCHIVE_FAMILY / F1.NOISE_FAMILY;
        trace: True for the trace families (glob patterns), False for the CSV/noise families.
    Returns:
        (reference array, [(label, values array, group), ...]) with empty arrays where a cell is
        missing, so a slide shows the gap rather than silently dropping a condition.
    """
    if trace:
        title, ref_pat, cap, items = family
        ref = (F1.live_matched(ref_pat, cap) or (np.array([]),))[0]
        out = []
        for lab, pat, grp in items:
            v = np.array([]) if pat is None else (F1.live_matched(pat, cap) or (np.array([]),))[0]
            out.append((lab, np.asarray(v, float), grp))
        return np.asarray(ref, float), out
    title, ref_spec, items = family
    if isinstance(ref_spec, tuple):
        ref = F1.csv_active(ref_spec[0], **ref_spec[1])
        out = [(lab, np.asarray(F1.csv_active(s[0], **s[1]), float), grp) for lab, s, grp in items]
    else:
        # ⚠️ THE NOISE SWEEP SAVED NO PER-NETWORK ROWS, only a per-condition mean, sd and n, and it
        # uses a peak-rate silence rule rather than participation. Its slide therefore draws mean
        # and a 95% interval where the others draw every seed, and says so on the figure.
        ref = F1.noise_active(ref_spec)
        out = [(lab, (float("nan"), float("nan"), 0) if s is None else F1.noise_active(s), grp)
               for lab, s, grp in items]
    return ref, out


def collect():
    """Every intervention in Figure 1d, keyed by (family title, group).

    Returns:
        dict {(family title, group): (reference array, [(label, values), ...])}.
    """
    out = {}
    fams = [(f, True) for f in F1.TRACE_FAMILIES]
    fams += [(F1.ARCHIVE_FAMILY, False), (F1.NOISE_FAMILY, False)]
    for fam, trace in fams:
        title = fam[0]
        ref, items = _family_values(fam, trace)
        for lab, vals, grp in items:
            if grp == "reference":
                continue
            out.setdefault((title, grp), (ref, []))[1].append((lab, vals))
    return out


def summary_slide(name, title, ref_label, entries, ref, col, note=None, ref_at=0):
    """A family that saved only per-condition summaries: mean and a 95% interval, no seeds.

    Args:
        name: output stem; title: the claim; ref_label: the reference rung's name;
        entries: [(label, (mean, sd, n)), ...]; ref: the reference's (mean, sd, n);
        col: the family's colour; note: a second title line.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    rows = [(lab, v, col) for lab, v in entries]
    rows.insert(ref_at, (ref_label, ref, ps.BASE))
    xs = np.arange(len(rows), dtype=float)
    for x, (lab, (m, sd, n), c) in zip(xs, rows):
        if not n:
            continue
        ax.errorbar(x, m, yerr=1.96 * sd / np.sqrt(n), fmt="o", ms=4.0, color=c, mec="white",
                    mew=0.6, lw=1.1, capsize=2.0, zorder=4)
        ax.annotate(f"{m:.0f}", (x, m + 1.96 * sd / np.sqrt(n)), textcoords="offset points",
                    xytext=(0, 5), ha="center", va="bottom", fontsize=6.2, color=ps.INK)
    ax.axhline(ref[0], color=ps.BASE, lw=0.7, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels([r[0] for r in rows], fontsize=6.4, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel("active units")
    ax.set_title(title + ("\n" + note if note else ""), fontsize=7.4, color=ps.INK,
                 linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, name)


def dose_slide(name, title, ref_label, entries, ref, col, ylabel="active units", note=None,
               ref_at=0):
    """One intervention family as counts: the reference, then each setting, every seed drawn.

    Counts rather than changes, because a slide has no room for a reader to hold a reference in
    their head - it has to be on the axis.

    Args:
        name: output file stem; title: the claim, drawn above the axes;
        ref_label: what the reference rung is called; entries: [(label, values array), ...];
        ref: the reference's values; col: the family's colour; ylabel: y axis label;
        note: an optional second line under the title; ref_at: where the reference sits among the
        entries - on a dose ladder it is a rung, not a preamble, and putting it first makes the
        ladder read out of order.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    labels = [lab for lab, _ in entries]
    groups = [np.asarray(v, float) for _, v in entries]
    cols = [col] * len(entries)
    labels.insert(ref_at, ref_label)
    groups.insert(ref_at, np.asarray(ref, float))
    cols.insert(ref_at, ps.BASE)
    xs = np.arange(len(groups), dtype=float)
    res = ps.strip(ax, xs, groups, cols, rng=np.random.default_rng(0))
    if len(ref):
        ax.axhline(np.mean(ref), color=ps.BASE, lw=0.7, ls=":", zorder=1)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=6.4, rotation=22, ha="right", rotation_mode="anchor")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel)
    drawn = [g for g in groups if len(g)]
    if drawn:
        lo = min(g.min() for g in drawn)
        hi = max(g.max() for g in drawn)
        pad = max(0.08 * (hi - lo), 5.0)
        ax.set_ylim(max(0, lo - pad), hi + pad * 1.5)
    for x, g, (m, sd, n) in zip(xs, groups, res):
        if not n:
            ax.text(x, ax.get_ylim()[0] + 0.04 * (ax.get_ylim()[1] - ax.get_ylim()[0]),
                    "no data", ha="center", fontsize=5.6, color=ps.FAINT, rotation=90)
            continue
        ax.annotate(f"{m:.0f}", (x, g.max()), textcoords="offset points", xytext=(0, 5),
                    ha="center", va="bottom", fontsize=6.2, color=ps.INK)
    ax.set_title(title + ("\n" + note if note else ""), fontsize=7.4, color=ps.INK,
                 linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, name)


def panel_slide(name, fn, width=W, height=H, **kw):
    """One manuscript panel on its own figure.

    Args:
        name: output stem; fn: a callable taking the axes (and anything in kw);
        width, height: figure size in inches; kw: passed to fn.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(width, height))
    fn(ax, **kw)
    return ps.save(fig, name)


# (slide stem, title, family title, group, reference label, reference rung, note)
INTERVENTIONS = [
    ("slide_x_activation_cddm", "A different activation does not help",
     "CDDM, 200k", "activation", "ReLU (default)", 0,
     "CDDM, N = 1000, read at 200k. Every seed drawn."),
    ("slide_x_weightdecay", "Weight decay makes it monotonically worse",
     "CDDM, 200k", "weight decay", "W.D. 10$^{-6}$ (default)", 1,
     "CDDM, N = 1000. The default is a rung of this ladder, not a separate condition."),
    ("slide_x_activation_ff", "Nor on the other task",
     "3-bit flip-flop, 150k", "activation", "ReLU (default)", 0,
     "3-bit flip-flop, N = 1000, read at 150k."),
    ("slide_x_inputscale", "Scaling the input weights does not help either",
     "3-bit flip-flop, 150k", "input scale", "input w. $\\times$1 (default)", 1,
     "3-bit flip-flop, N = 1000."),
    ("slide_x_metabolic", "The field-standard metabolic penalty moves nothing beyond seed scatter",
     "CDDM, 30k (archived)", "metabolic", "$\\lambda$ = 0 (default)", 0,
     "CDDM, N = 1000, four decades of $\\lambda$."),
    ("slide_x_architecture", "Nor does the equation form, nor a trainable bias",
     "CDDM, 30k (archived)", "architecture", "standard", 0,
     "CDDM, N = 1000."),
    ("slide_x_recnoise", "Removing recurrent noise is the largest effect we found — and it is negative",
     "CDDM, 30k (peak-rate criterion)", "noise", "$\\sigma$ = 0.05 (default)", 2,
     "CDDM, N = 1000, peak-rate criterion. Mean and 95% interval: this sweep saved no per-seed rows."),
]


# The three tasks, and the arms that exist on all of them. Rescale, synaptic noise and the penalty
# pair were only ever run on the flip-flop, so putting them in one panel and not the others would
# make the three panels answer different questions.
TASKS = [("CDDM", "CDDM", 100_000), ("NBitFlipFlop", "3-bit flip-flop", 40_000),
         ("DMTS", "DMTS, 7$\\tau$ delay", 150_000)]
SHARED_ARMS = ("control", "mute", "duplicate")


def weights_by_task(c_all, n_units=1000):
    """Weight-magnitude distributions for the three tasks, side by side.

    The weight check is the one measure that separated the arms when the other three did not, so it
    is the one worth asking on more than one task. Only the arms present on every task are drawn:
    a panel carrying six arms next to one carrying three is not a comparison.

    Args:
        c_all: the full cache dict, every task and size; n_units: the size to read at.
    Returns:
        the output path, or None if the cache has no task column yet.
    """
    if "task" not in c_all:
        print("  SKIP weights_by_task: this cache predates the task column")
        return None
    edges = np.asarray(c_all["log_bins"], float)
    mid = 0.5 * (edges[1:] + edges[:-1])
    ps.setup()
    fig, axes = plt.subplots(1, len(TASKS), figsize=(ps.W2, 62 * ps.MM), sharey=True)
    lo_x, hi_x, drawn_any = [], [], False
    for ax, (task, nice, _) in zip(axes, TASKS):
        for arm, short, _, col in F2.ARMS:
            if arm not in SHARED_ARMS:
                continue
            m = (c_all["task"] == task) & (c_all["arm"] == arm) & \
                (c_all["N"].astype(int) == n_units)
            if not m.any():
                continue
            h = np.asarray(c_all["w_hist"][m], float)
            d = (h / h.sum(axis=1, keepdims=True)).mean(axis=0)
            dens = d / (mid[1] - mid[0])
            ax.plot(mid, np.where(dens > 0, dens, np.nan), lw=1.1, color=col, zorder=4,
                    label=f"{short} ({m.sum()})")
            cdf = np.cumsum(d)
            lo_x.append(mid[np.searchsorted(cdf, 0.01)])
            hi_x.append(mid[np.searchsorted(cdf, 0.99)])
            drawn_any = True
        ctl = (c_all["task"] == task) & (c_all["arm"] == "control") & \
              (c_all["N"].astype(int) == n_units)
        sub = []
        for arm in SHARED_ARMS:
            m = (c_all["task"] == task) & (c_all["arm"] == arm) & \
                (c_all["N"].astype(int) == n_units)
            if m.any() and ctl.any() and arm != "control":
                r = (np.mean(c_all["w_sigma_log"][m].astype(float)) /
                     np.mean(c_all["w_sigma_log"][ctl].astype(float)))
                sub.append(f"{arm} {r:.2f}×")
        ax.set_title(nice + ("\n" + "  ".join(sub) + " the control's width" if sub else ""),
                     fontsize=6.8, color=ps.INK, linespacing=1.3, pad=4)
        ax.set_xlabel("recurrent weight\n$\\log_{10}|W_{ij}|$")
        ps.ygrid(ax)
    if not drawn_any:
        plt.close(fig)
        print("  SKIP weights_by_task: no networks at N=%d" % n_units)
        return None
    axes[0].set_ylabel("density")
    for ax in axes:
        ax.set(xlim=(min(lo_x) - 0.3, max(hi_x) + 0.3), yscale="log", ylim=(2e-4, 40.0))
    axes[0].legend(loc="upper left", fontsize=5.6, handlelength=1.0, borderpad=0.1,
                   borderaxespad=0.2)
    return ps.save(fig, "slide_f2_weights_by_task")


DRIFT_CACHE = "data/f2_drift_traces.npz"
DRIFT_TASKS = ["CDDM", "NBitFlipFlop", "DMTS"]
DRIFT_VARS = [("W_inp", ps.SLOTS[0]), ("W_rec", ps.SLOTS[3]), ("W_out", ps.SLOTS[2])]


def participation_by_task(at_iteration=None, stem="slide_02_participation_by_task"):
    """The participation distribution on all three tasks, three panels in a row.

    Args:
        at_iteration: read every task at the probe nearest this iteration, for a matched-budget
            comparison; None reads each at the end of its own budget.
        stem: output file stem.

    ⚠️ THE TWO VERSIONS ANSWER DIFFERENT QUESTIONS and the figure says which it is showing. Reading
    each task at its own end asks "what does a trained network look like", and the budgets differ
    because the tasks do - DMTS needs 150,000 iterations to be solved, CDDM 100,000. Reading all
    three at one iteration asks "what do they look like after the same amount of training", which
    is matched on exactly the axis this talk argues is the wrong clock: at a common 40,000 the CDDM
    and DMTS networks are far less converged than the flip-flop. Neither is wrong; they are not
    interchangeable.

    Returns:
        the output path, or None if the trace cache is missing.
    """
    if not os.path.exists(DRIFT_CACHE):
        print(f"  SKIP participation_by_task: {DRIFT_CACHE} missing")
        return None
    z = np.load(DRIFT_CACHE, allow_pickle=True)
    have = [t for t in DRIFT_TASKS if f"{t}|participation" in z.files]
    if not have:
        print("  SKIP participation_by_task: no participation vectors in the cache")
        return None
    ps.setup()
    # x is shared so the three distributions sit on one scale; y is NOT, because CDDM puts 600 of
    # its 1000 units in a single bin and a shared count axis flattens the other two panels to a line
    fig, axes = plt.subplots(1, len(have), figsize=(ps.W2, 58 * ps.MM), sharex=True)
    axes = np.atleast_1d(axes)
    for ax, task in zip(axes, have):
        if at_iteration is None:
            vec = np.asarray(z[f"{task}|participation"], float)
            it = float(z[f"{task}|participation_iter"]) \
                if f"{task}|participation_iter" in z.files else float("nan")
        else:
            P = np.asarray(z[f"{task}|participation_all"], float)
            pit = np.asarray(z[f"{task}|participation_all_iters"], float)
            j = int(np.argmin(np.abs(pit - at_iteration)))
            vec, it = P[j], float(pit[j])
        F1.panel_b(ax, vec)
        lab = str(z[f"{task}|label"]) if f"{task}|label" in z.files else task
        ax.set_title(lab + (f"\n{it:,.0f} iterations" if np.isfinite(it) else ""),
                     fontsize=6.8, color=ps.INK, linespacing=1.3, pad=4)
    # panel_b labels and glosses its own axes. Three copies of each overlap and say nothing extra,
    # so the x label is kept on the middle panel only and the gloss is dropped - the deck defines
    # participation and the criterion once, in text, under the figure.
    mid = len(axes) // 2
    for i, ax in enumerate(axes):
        for t in list(ax.texts):
            if "rate moves over a trial" in t.get_text():
                t.remove()
        if i != mid:
            ax.set_xlabel("")
        if i != 0:
            ax.set_ylabel("")
    return ps.save(fig, stem)


def readout_time_slide(k=3):
    """How long each size takes to reach its own loss floor — unpenalised only, one panel.

    `excess_time_matrix.py` draws this over the (N, k) grid for four penalty conditions, which is a
    2 x 4 sheet of heat maps. Three of those conditions have not been introduced at the point this
    slide appears, and k is not a dimension the talk needs, so this is the `none` column at one k,
    as a curve against N with every run drawn.

    T is the iteration at which a run's noise-free loss first reaches (1 + EXCESS_DELTA) times its
    OWN fitted floor, so each network is judged against what it can do rather than against a shared
    target - which is the whole point: a larger network has a lower floor and takes longer to get
    there.

    Args:
        k: task complexity to show, the 3-bit flip-flop by default.
    Returns:
        the output path, or None if no run yields a finite time.
    """
    runs = [r for r in PR.load() if r["pen"] == "none" and r["k"] == k]
    for r in runs:
        r["floor"] = PR.fit_floor(r["loss"], r["budget"])
        r["T"] = PR.excess_time(r["loss"], r["floor"], PR.EXCESS_DELTA)
    runs = [r for r in runs if np.isfinite(r["T"])]
    if not runs:
        print("  SKIP readout_time_slide: no finite read-out times")
        return None
    Ns = sorted({r["N"] for r in runs})
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    for r in runs:
        ax.plot(r["N"], r["T"], "o", ms=3.0, color=ps.BASE, alpha=0.55, mec="none", zorder=3)
    mu = [np.mean([r["T"] for r in runs if r["N"] == n]) for n in Ns]
    ax.plot(Ns, mu, "-o", lw=1.3, ms=4.0, color=ps.SLOTS[0], mec="white", mew=0.6, zorder=5)
    for n, m in zip(Ns, mu):
        ax.annotate(f"{m/1000:.0f}k", (n, m), textcoords="offset points", xytext=(0, 7),
                    ha="center", fontsize=6.2, color=ps.INK)
    ax.set(xscale="log", yscale="log", xlabel="network size $N$",
           ylabel="iterations to reach\n$1.10\\times$ its own loss floor")
    ax.set_xticks(Ns, [str(n) for n in Ns])
    # ⚠️ THE TITLE STATES WHAT THIS PANEL SHOWS, which is not what it was built to show. Over an 8x
    # size range the read-out time moves from 31k to 38k and the within-size scatter overlaps, so
    # "bigger networks need longer" is not readable here. The pooled fit across the whole
    # unpenalised grid does give a positive size exponent, beta = +0.159 [0.090, 0.249], but that is
    # 1.39x over 8x and it controls for k, which this panel holds fixed - so it is quoted in the
    # deck text rather than asserted over a panel that cannot support it.
    lo, hi = min(mu), max(mu)
    ax.set_title(f"Read-out time hardly moves with size on one task\n"
                 f"{k}-bit flip-flop, unpenalised, {lo/1000:.0f}k to {hi/1000:.0f}k "
                 f"over an 8$\\times$ size range ({len(runs)} runs)",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, "slide_05_readout_time")


def readout_rule_slide(k=3, sizes=(500, 4000)):
    """The comparison rule itself: read every network at 1.10x its OWN loss floor.

    This is the slide between "iteration count is the wrong clock" and the scaling result, and it
    has to show the rule rather than state it. Two sizes, one task, unpenalised: each run's loss
    against iteration, its own fitted floor as a dotted line, 1.10x that floor as the threshold, and
    a marker where it crosses. Different networks have different floors, so the same rule lands at
    different iterations - which is the whole point of not fixing the iteration.

    Args:
        k: task complexity; sizes: the two network sizes to contrast.
    Returns:
        the output path, or None if no run is usable.
    """
    runs = [r for r in PR.load() if r["pen"] == "none" and r["k"] == k and r["N"] in sizes]
    for r in runs:
        r["floor"] = PR.fit_floor(r["loss"], r["budget"])
        r["T"] = PR.excess_time(r["loss"], r["floor"], PR.EXCESS_DELTA)
    runs = [r for r in runs if np.isfinite(r["T"]) and np.isfinite(r["floor"])]
    if not runs:
        print("  SKIP readout_rule_slide: no usable runs")
        return None
    cols = {sizes[0]: ps.BASE, sizes[-1]: ps.SLOTS[0]}
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    seen = set()
    for r in runs:
        c = cols.get(r["N"], ps.MUTED)
        L = np.asarray(r["loss"], float)
        it = np.arange(1, len(L) + 1, dtype=float)
        lab = f"N = {r['N']}" if r["N"] not in seen else None
        seen.add(r["N"])
        ax.plot(it, F1.running_median(L), lw=0.9, color=c, alpha=0.85, zorder=3, label=lab)
    for n in sizes:
        g = [r for r in runs if r["N"] == n]
        if not g:
            continue
        fl = float(np.mean([r["floor"] for r in g]))
        T = float(np.mean([r["T"] for r in g]))
        c = cols.get(n, ps.MUTED)
        ax.axhline(fl, color=c, lw=0.6, ls=":", zorder=2)
        ax.axhline(fl * (1 + PR.EXCESS_DELTA), color=c, lw=0.8, ls="--", zorder=2)
        ax.plot([T], [fl * (1 + PR.EXCESS_DELTA)], "o", ms=5.0, color=c, mec="white", mew=0.8,
                zorder=6)
        # the two sizes reach almost the same floor on this task, so the markers nearly coincide:
        # the labels are staggered rather than left to overlap
        dy = 9 if n == sizes[0] else -13
        ax.annotate(f"N={n}: {T/1000:.0f}k", (T, fl * (1 + PR.EXCESS_DELTA)),
                    textcoords="offset points", xytext=(8, dy), fontsize=6.2, color=c)
    ax.set(xscale="log", yscale="log", xlabel="training iteration", ylabel="noise-free task loss")
    ax.legend(loc="upper right", fontsize=6.2, handlelength=1.2)
    ax.set_title("Read every network where ITS OWN loss stops falling\n"
                 r"dotted: that run's fitted floor;  dashed: $1.10\times$ it;  dot: the read-out",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, "slide_06_readout_rule")


def drift_slides():
    """Do the parameters stop moving? One panel per task, trajectories against iteration.

    THE QUESTION IS NOT THE LAG EXPONENT. An earlier version of this function drew the final
    lag-scaling exponent over the (N, k) grid, which answers "is the motion biased or unbiased" for
    the flip-flop only, and k is not a dimension this talk needs. What the audience has to see is
    whether the weights are still moving at the end, so the y axis is the displacement itself and
    the x axis is the iteration count.

    Each curve is |W(t+L) - W(t)| / |W(t)| at L = 10,000, logged every 100 iterations, every seed
    drawn. A curve falling to zero means training has stopped changing the weights; one that
    flattens at a non-zero value means they are still moving.

    THE BUDGETS DIFFER AND THE PANELS SAY SO. DMTS needs 150,000 iterations to be solved and CDDM
    100,000; the flip-flop's long runs predate drift logging, so 40,000 is the longest one that
    carries these metrics.

    Returns:
        list of output paths - one figure, or an empty list if the cache is missing.
    """
    if not os.path.exists(DRIFT_CACHE):
        print(f"  SKIP drift_slides: {DRIFT_CACHE} missing (build it with f2_drift_traces.py)")
        return []
    z = np.load(DRIFT_CACHE, allow_pickle=True)
    lag = int(z["lag"])
    ps.setup()
    fig, axes = plt.subplots(1, len(DRIFT_TASKS), figsize=(ps.W2, 62 * ps.MM), sharey=True)
    for ax, task in zip(axes, DRIFT_TASKS):
        last = []
        for var, col in DRIFT_VARS:
            seeds = sorted({int(k.split("|")[2]) for k in z.files
                            if k.startswith(f"{task}|{var}|") and k.endswith("|it")})
            curves = []
            for s in seeds:
                it, v = z[f"{task}|{var}|{s}|it"], z[f"{task}|{var}|{s}|v"]
                ax.plot(it, v, lw=0.5, color=col, alpha=0.35, zorder=3)
                curves.append((it, v))
            if not curves:
                continue
            grid_t = curves[0][0]
            stack = [np.interp(grid_t, it, v) for it, v in curves]
            ax.plot(grid_t, np.mean(stack, axis=0), lw=1.3, color=col, zorder=5,
                    label=f"{var} ({len(curves)})")
            last.append(np.mean([v[-1] for _, v in curves]))
        lab = str(z[f"{task}|label"]) if f"{task}|label" in z.files else task
        end = max((z[f"{task}|{v}|0|it"].max() for v, _ in DRIFT_VARS
                   if f"{task}|{v}|0|it" in z.files), default=0)
        ax.set_title(f"{lab}\nstill moving at {end:,.0f} iterations",
                     fontsize=6.8, color=ps.INK, linespacing=1.3, pad=4)
        ax.set(xscale="log", yscale="log", xlabel="training iteration")
        ps.ygrid(ax)
    axes[0].set_ylabel("relative weight change\n"
                       r"$\|W(t)-W(t-L)\|_F/\|W(t)\|_F$, " + f"$L$ = {lag:,}")
    axes[0].legend(loc="lower left", fontsize=5.6, handlelength=1.1, borderaxespad=0.2)
    return [ps.save(fig, "slide_04_drift_trajectories")]


def main(list_only=False):
    """Write every slide. Returns the list of output paths."""
    if list_only:
        for stem, title, *_ in INTERVENTIONS:
            print(f"  {stem:28s} {title}")
        return []
    out = []

    # ---- the manuscript panels, one per figure ------------------------------------------------
    everything = F2.load()
    at_main = F2.restrict(everything, n_units=F2.N_MAIN)
    at_1000 = F2.restrict(everything, n_units=F2.N_MAIN,
                          chosen={a: F2.pick_cell(at_main, a)[0] for a in F2.GRID_ARMS})
    # ---- Figure 1's panels, in the order the talk needs them ----------------------------------
    rates, _, pvec = F1.example_network()
    fig = plt.figure(figsize=(W, 78 * ps.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.72], hspace=0.42)
    ax_net, ax_tr = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])
    F1.panel_a(ax_net, ax_tr, rates, pvec)
    out.append(ps.save(fig, "slide_01_schematic"))

    out.append(panel_slide("slide_02_participation", F1.panel_b, p=pvec))
    got = participation_by_task()
    if got:
        out.append(got)
    # the same three panels read at one common iteration, which is the flip-flop's ceiling
    got = participation_by_task(at_iteration=40_000,
                                stem="slide_02_participation_by_task_matched")
    if got:
        out.append(got)
    out.append(panel_slide("slide_06_scaling", F1.panel_c, width=W, height=78 * ps.MM))

    # panel (e) is one sub-panel per task, stacked; DMTS is kept even though it breaks the pattern
    fig = plt.figure(figsize=(W, 96 * ps.MM))
    gs = GridSpec(len(F1.TRAJ), 1, figure=fig, hspace=0.30)
    F1.panel_e([fig.add_subplot(gs[i]) for i in range(len(F1.TRAJ))])
    out.append(ps.save(fig, "slide_03_silencing_vs_training"))

    out += drift_slides()
    got = readout_time_slide()
    if got:
        out.append(got)
    got = readout_rule_slide()
    if got:
        out.append(got)

    out.append(panel_slide("slide_rules", F2.panel_a, width=ps.W2, height=52 * ps.MM))
    out.append(panel_slide("slide_f2_active", F2.panel_b, c=at_1000))
    out.append(panel_slide("slide_f2_r2", F2.panel_c, c=at_1000))
    out.append(panel_slide("slide_f2_dims", F2.panel_d, c=at_1000))
    out.append(panel_slide("slide_f2_weights", F2.panel_e, c=at_1000))
    got = weights_by_task(everything)
    if got:
        out.append(got)
    out.append(panel_slide("slide_f2_size_active", F2.panel_f, c=everything))
    out.append(panel_slide("slide_f2_size_r2", F2.panel_g, c=everything))

    # ---- one figure per failed intervention ---------------------------------------------------
    fams = collect()
    for stem, title, fam_title, group, ref_label, ref_at, note in INTERVENTIONS:
        got = fams.get((fam_title, group))
        if got is None:
            print(f"  SKIP {stem}: no cells for ({fam_title}, {group})")
            continue
        ref, entries = got
        draw = summary_slide if group == "noise" else dose_slide
        out.append(draw(stem, title, ref_label, entries, ref,
                        F1.GROUP_COL.get(group, ps.MUTED), note=note, ref_at=ref_at))
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="name the slides without drawing them")
    args = ap.parse_args()
    main(list_only=args.list)
