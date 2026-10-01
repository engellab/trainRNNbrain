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

import glob
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.ticker import NullLocator

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import fig_paper_F1 as F1
import pr_matrix as PR
import fig_paper_F2 as F2
from common import DATA_DIR, active_count

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
    # The reference is the DEFAULT DRAW, whose rows sit at norm 0.050 at N = 1000 - the lowest rung
    # of this ladder, not a middle one, which is why ref_at is 0. Labelling it "×1" and putting it
    # second (every version before 2026-10-01) was what made this panel look non-monotone.
    ("slide_x_inputscale", "Scaling the input weights up adds 40\u201375 units of 1000, peaking at row norm 2",
     "3-bit flip-flop, 150k", "input scale", "row norm 0.05 (default draw)", 0,
     "3-bit flip-flop, N = 1000, read at 150,000 iterations. Every seed drawn.\n"
     "Rungs are the absolute L2 norm of each W$_{inp}$ row at init; the default draw is 0.050."),
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


# ---- slide 12's companion: is the flat metabolic ladder bought with performance? ----------------
#
# The metabolic slide above reads the archived CSV, which has no r2 column, so it cannot answer the
# first objection to a null result: "the penalty did nothing to the count because it was busy
# breaking the task". These cells are the SAME networks, read from their own run folders instead,
# where the score prefix carries r2. The counts reproduce the CSV element for element (414 / 393 /
# 431 / 432 / 380 active at the scale-free rule), which is the check that the two paths agree.
#
# COLOUR CARRIES lambda HERE, which the dose slides do not need - they put lambda on the x axis and
# use one hue for the whole family. Five conditions is exactly the number of validated slots, and
# the unpenalised cell keeps BASE rather than taking a slot, per this project's rule that the
# baseline is not a series. Because categorical hues carry no order, the dose ORDER is encoded a
# second time, as a line through the cell means.
MET_CELL = f"{DATA_DIR}/CDDM_std_g0_metabolic/EqType=h_N=1000_LmbdMet={{lam}}"
MET_LADDER = [
    ("$\\lambda$ = 0 (no penalty)", f"{DATA_DIR}/CDDM_std_g0/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
     ps.BASE),
    ("$\\lambda$ = 0.01", MET_CELL.format(lam="0.01"), ps.SLOTS[0]),
    ("$\\lambda$ = 0.1", MET_CELL.format(lam="0.1"), ps.SLOTS[1]),
    ("$\\lambda$ = 1", MET_CELL.format(lam="1.0"), ps.SLOTS[2]),
    ("$\\lambda$ = 10", MET_CELL.format(lam="10.0"), ps.SLOTS[3]),
]


def met_cell(cell_dir):
    """Task r2 and active-unit count, per seed, for one cell of the metabolic sweep.

    r2 is the run folder's own score prefix - the same field `F1.traces_of` filters on with its
    `min_r2` argument - and the active count is `common.active_count` under the scale-free rule at
    the last participation probe. Diverged runs (`nan_` prefix) are dropped, as everywhere else in
    this project.

    This cannot go through `F1.traces_of`: that loader keys on `participation_iters`, and the
    2026-07-28 metabolic traces store their probe iterations under `iters`, so it returns nothing
    for this sweep. Reading `participation[-1]` is the last probe either way.

    Args:
        cell_dir: path to one cell directory, holding one run folder per seed.
    Returns:
        (r2, active, iteration): r2 and active are (n_seeds,) float arrays, empty where the cell is
        missing; iteration is the last probe the cell was read at, or None if it is empty.
    """
    r2, active, last = [], [], None
    for f in sorted(glob.glob(os.path.join(cell_dir, "*", "*ParticipationTrace.pkl"))):
        head = os.path.basename(os.path.dirname(f)).split("_")[0]
        if head == "nan":
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        p = np.asarray(d["participation"], float)[-1]
        r2.append(float(head))
        active.append(active_count(p, "scalefree"))
        last = int(np.asarray(d["iters"])[-1])
    return np.array(r2, float), np.array(active, float), last


def metabolic_r2_slide(name="slide_x_metabolic_r2"):
    """Task r2 against active units for every net of the metabolic ladder, one colour per lambda.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if no cell of the ladder is on disk.
    """
    cells = [(lab, col) + met_cell(pat) for lab, pat, col in MET_LADDER]
    drawn = [c for c in cells if len(c[2])]
    if not drawn:
        print(f"  SKIP {name}: no metabolic cells on disk")
        return None
    probes = {c[4] for c in drawn}
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))

    # the dose order, encoded a second time: categorical colour cannot carry it
    ax.plot([c[3].mean() for c in drawn], [c[2].mean() for c in drawn], "-", lw=0.8,
            color=ps.FAINT, zorder=2, label="in order of $\\lambda$")
    for lab, col, r2, active, _ in drawn:
        ax.plot(active, r2, "o", ms=4.6, color=col, mec="white", mew=0.6, zorder=4,
                label=f"{lab}  ({len(r2)})")
        ax.plot(active.mean(), r2.mean(), "o", ms=9.0, color=col, mec="white", mew=1.0,
                alpha=0.45, zorder=3)

    ax.set_xlabel("active units of 1000  (scale-free rule, $p \\geq 0.05\\,q_{95}(p)$)")
    ax.set_ylabel("task $r^2$")
    every_r2 = np.concatenate([c[2] for c in drawn])
    probe = f"{sorted(probes)[0]:,}" if len(probes) == 1 else "the last probe"
    ax.set_title("No rung of the ladder trades performance for live units\n"
                 f"CDDM, N = 1000, read at 30,000 iterations (last probe {probe}). "
                 f"Every seed drawn; r$^2$ stays within {every_r2.min():.2f}–{every_r2.max():.2f} "
                 "across four decades of $\\lambda$.\nLarge pale dot is the cell mean; the grey "
                 "line joins them in order of $\\lambda$.",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ax.legend(loc="lower left", fontsize=5.8, handlelength=1.0, borderpad=0.2,
              borderaxespad=0.3, ncol=2)
    ps.ygrid(ax)
    return ps.save(fig, name)


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


# ---- the read-out rule, shown on two tasks -----------------------------------------------------
# ONE RULE, TWO TASK FAMILIES, LOGGED DIFFERENTLY. The flip-flop (N, k) grid lives in pr_matrix's own
# loader at one clean-loss sample per 10 iterations; CDDM lives in Figure 1's scaling table at one
# per iteration. Both go through excess_time_at, which takes the iterations rather than assuming a
# spacing. Handing CDDM to the PROBE-based excess_time would place its read-out 10x too late and
# smooth its loss over 21 iterations instead of 210, which on a jagged trace means the threshold is
# crossed on a noise dip.
READOUT_TASKS = ["3-bit flip-flop", "CDDM"]
# N = 1000 red and N = 2000 green, so the two middle sizes are told apart at a glance; the smallest
# stays neutral grey and the largest keeps the blue it had when this slide showed only two sizes.
READOUT_SIZE_COLOUR = {500: ps.BASE, 1000: ps.SLOTS[1], 2000: ps.SLOTS[2],
                       4000: ps.SLOTS[0], 5000: ps.SLOTS[0]}


def readout_runs(task, k=3):
    """Unpenalised runs of one task, each with its loss trace, fitted floor and read-out iteration.

    Args:
        task: "3-bit flip-flop", or any key of F1.SCALING; k: flip-flop complexity, unused otherwise.
    Returns:
        list of dicts with N, loss, it, budget, floor and T, keeping only the runs whose floor fitted
        and whose loss actually reaches (1 + EXCESS_DELTA) times it.
    """
    out = []
    if task == "3-bit flip-flop":
        for r in PR.load():
            if r["pen"] != "none" or r["k"] != k:
                continue
            L = np.asarray(r["loss"], float)
            out.append(dict(N=r["N"], loss=L, it=(np.arange(len(L)) + 1) * PR.PROBE,
                            budget=r["budget"]))
    else:
        Ns, pats, _cap, _col = F1.SCALING[task]
        for N in Ns:
            for r in F1.runs_with_loss(pats[N], min_r2=F1.TASK_MIN_R2.get(task)):
                it = np.asarray(r["it_loss"], float)
                L = np.asarray(r["loss"], float)
                n = min(len(it), len(L))
                if n < 100:
                    continue
                out.append(dict(N=N, loss=L[:n], it=it[:n], budget=float(it[n - 1])))
    keep = []
    for r in out:
        r["floor"] = PR.fit_floor_at(r["loss"], r["it"], r["budget"])
        r["T"] = PR.excess_time_at(r["loss"], r["it"], r["floor"], PR.EXCESS_DELTA)
        if r["floor"] is not None and np.isfinite(r["T"]):
            keep.append(r)
    return keep


def readout_time_slide(k=3):
    """How long each size takes to reach its own loss floor — one panel per task.

    `excess_time_matrix.py` draws this over the (N, k) grid for four penalty conditions, which is a
    2 x 4 sheet of heat maps. Three of those conditions have not been introduced at the point this
    slide appears, and k is not a dimension the talk needs, so this is the `none` column at one k,
    as a curve against N with every run drawn - and the same thing for CDDM beside it, because one
    task cannot show whether the pattern is a property of the rule or of the flip-flop.

    T is the iteration at which a run's noise-free loss first reaches (1 + EXCESS_DELTA) times its
    OWN fitted floor, so each network is judged against what it can do rather than against a shared
    target.

    ⚠️ THE BUDGETS WITHIN A TASK ARE NOT EQUAL, and each floor is fitted over its own run's whole
    trace. A longer trace pins the floor down better, hence slightly later crossings, so the panel
    prints the per-size budget rather than leaving that to be assumed away.

    Args:
        k: flip-flop complexity to show, the 3-bit task by default.
    Returns:
        the output path, or None if no task yields a finite time.
    """
    data = {t: readout_runs(t, k=k) for t in READOUT_TASKS}
    data = {t: rs for t, rs in data.items() if rs}
    if not data:
        print("  SKIP readout_time_slide: no finite read-out times")
        return None
    ps.setup()
    fig, axes = plt.subplots(1, len(data), figsize=(ps.W2, 62 * ps.MM))
    axes = np.atleast_1d(axes)
    for ax, (task, runs) in zip(axes, data.items()):
        Ns = sorted({r["N"] for r in runs})
        for r in runs:
            ax.plot(r["N"], r["T"], "o", ms=3.0, color=READOUT_SIZE_COLOUR.get(r["N"], ps.BASE),
                    alpha=0.55, mec="none", zorder=3)
        mu = [np.mean([r["T"] for r in runs if r["N"] == n]) for n in Ns]
        ax.plot(Ns, mu, "-", lw=1.2, color=ps.INK, alpha=0.5, zorder=4)
        allT = [r["T"] for r in runs]
        mid = 0.5 * (min(allT) + max(allT))
        for n, m in zip(Ns, mu):
            ax.plot([n], [m], "o", ms=4.4, color=READOUT_SIZE_COLOUR.get(n, ps.BASE),
                    mec="white", mew=0.6, zorder=5)
            # above the marker normally, below it in the top half, where a run's own point sits
            ax.annotate(f"{m/1000:.0f}k", (n, m), textcoords="offset points",
                        xytext=(0, -12 if m > mid else 8),
                        ha="center", fontsize=6.2, color=ps.INK)
        bud = ", ".join(f"{n}:{np.mean([r['budget'] for r in runs if r['N'] == n])/1000:.0f}k"
                        for n in Ns)
        ax.set(xscale="log", yscale="log", xlabel="network size $N$")
        ax.set_xticks(Ns, [str(n) for n in Ns])
        # a log axis also labels its minor ticks, which printed "6 x 10^2" on top of "500"
        ax.xaxis.set_minor_locator(NullLocator())
        ax.margins(x=0.10, y=0.16)       # no run sitting on the frame
        ax.set_title(f"{task}  ({len(runs)} runs)\n"
                     f"{min(mu)/1000:.0f}k to {max(mu)/1000:.0f}k over "
                     f"{max(Ns)/min(Ns):.0f}$\\times$ in $N$\n"
                     f"budget per size {bud}",
                     fontsize=6.8, color=ps.INK, linespacing=1.35, pad=5)
        ps.ygrid(ax)
    axes[0].set_ylabel("iterations to reach\n"
                       rf"${1 + PR.EXCESS_DELTA:.2f}\times$ its own loss floor")
    return ps.save(fig, "slide_05_readout_time")


def readout_rule_slide(k=3):
    """The comparison rule itself, on two tasks: read every network at 1.07x its OWN floor.

    This is the slide between "iteration count is the wrong clock" and the scaling result, and it has
    to show the rule rather than state it. Every size of each task, unpenalised: each run's smoothed
    loss against iteration, each size's own fitted floor as a dotted line, (1 + EXCESS_DELTA) times
    it as the threshold, and a marker where the size's mean crossing falls. Different networks have
    different floors, so the same rule lands at different iterations - which is the whole point of
    not fixing the iteration.

    THE READ-OUT TIMES ARE IN THE LEGEND, not annotated at the markers. On the flip-flop all four
    sizes reach floors within 10% of each other, so the markers nearly coincide and four labels at
    four nearly-identical points cannot be placed without colliding.

    ⚠️ The x axis is the ITERATION. It previously plotted np.arange(1, len(L) + 1), the sample index,
    while the markers were placed at a read-out time in iterations - so at PROBE = 10 the curves were
    compressed tenfold against their own markers, and the N = 4000 marker fell off the right-hand end
    of the data entirely.

    Args:
        k: flip-flop complexity to show.
    Returns:
        the output path, or None if no run is usable.
    """
    data = {t: readout_runs(t, k=k) for t in READOUT_TASKS}
    data = {t: rs for t, rs in data.items() if rs}
    if not data:
        print("  SKIP readout_rule_slide: no usable runs")
        return None
    ps.setup()
    fig, axes = plt.subplots(1, len(data), figsize=(ps.W2, 64 * ps.MM))
    axes = np.atleast_1d(axes)
    for ax, (task, runs) in zip(axes, data.items()):
        Ns = sorted({r["N"] for r in runs})
        # START THE AXIS AFTER THE INITIAL TRANSIENT. CDDM's clean loss is logged from iteration 0,
        # where an untrained network sits near 10^3; drawn from there, four decades of the panel are
        # the first few iterations and the approach to the floor - the thing the slide is about - is
        # a flat line at the bottom.
        x0 = 100.0
        top = 0.0
        for r in runs:
            c = READOUT_SIZE_COLOUR.get(r["N"], ps.MUTED)
            sm = F1.running_median(r["loss"])
            ax.plot(r["it"], sm, lw=0.7, color=c, alpha=0.5, zorder=3)
            j = np.searchsorted(r["it"], x0)
            if j < len(sm):
                top = max(top, float(np.nanmax(sm[j:j + 50])))
        handles = []
        for n in Ns:
            g = [r for r in runs if r["N"] == n]
            c = READOUT_SIZE_COLOUR.get(n, ps.MUTED)
            fl = float(np.mean([r["floor"] for r in g]))
            T = float(np.mean([r["T"] for r in g]))
            ax.axhline(fl, color=c, lw=0.5, ls=":", zorder=2)
            ax.axhline(fl * (1 + PR.EXCESS_DELTA), color=c, lw=0.7, ls="--", zorder=2)
            ax.plot([T], [fl * (1 + PR.EXCESS_DELTA)], "o", ms=5.0, color=c, mec="white", mew=0.8,
                    zorder=6)
            handles.append(Line2D([], [], color=c, lw=1.3,
                                  label=f"$N$ = {n}  ·  read at {T/1000:.0f}k"))
        lo = min(r["floor"] for r in runs)
        ax.set(xscale="log", yscale="log", xlabel="training iteration",
               xlim=(x0, max(r["it"][-1] for r in runs)), ylim=(lo * 0.85, top * 1.6))
        ax.set_title(task, fontsize=7.4, color=ps.INK, pad=5)
        # upper right: the curves fall left to right, so that corner is the empty one. At lower left
        # the legend lay across the floor lines and the read-out markers.
        ax.legend(handles=handles, loc="upper right", fontsize=5.8, handlelength=1.2,
                  borderaxespad=0.3, labelspacing=0.35, framealpha=0.9)
        ps.ygrid(ax)
    axes[0].set_ylabel("noise-free task loss")
    # the rule goes BELOW the panels: as a suptitle it printed across both panel titles
    fig.text(0.5, -0.02,
             "Read every network where ITS OWN loss stops falling   ·   "
             "dotted: that size's fitted floor   ·   "
             rf"dashed: ${1 + PR.EXCESS_DELTA:.2f}\times$ it   ·   dot: the read-out",
             ha="center", va="top", fontsize=6.6, color=ps.MUTED)
    return ps.save(fig, "slide_06_readout_rule")


def floor_fit_slide():
    """How the loss floor is fitted, one panel per task, before the read-out rule is used.

    The read-out rule rests on a fitted floor, so the fit has to be shown before the rule is. Each
    panel: the raw clean loss, the fitted stretched exponential L(t) = L_inf + A exp(-(t/tau)^beta),
    the fitted floor, 1.10x it, and where the loss crosses.

    ⚠️ IT FITS TWO OF THE THREE TASKS. On the flip-flop and CDDM beta comes out at 0.29 and 0.25 -
    strongly stretched, a broad spectrum of relaxation times - and forcing beta = 1, a plain
    exponential, makes the fit 1.5x and 2.9x worse. On DMTS the model fails outright: the RMS log
    residual is 0.59 against 0.004 and 0.043, and beta runs to the top of its range. DMTS does not
    descend smoothly, it sits near chance and then escapes, which no stretched exponential
    describes. The panel shows that rather than hiding it, and the read-out rule is not applied
    there.

    Returns:
        the output path.
    """
    from scipy.optimize import least_squares
    from common import stretched, logbin
    ps.setup()
    fig, axes = plt.subplots(1, len(F1.TRAJ), figsize=(ps.W2, 60 * ps.MM))
    for ax, (lab, pat, col, _) in zip(np.atleast_1d(axes), F1.TRAJ):
        runs = sorted(glob.glob(os.path.join(pat, "*")))
        if not runs:
            continue
        tr = pickle.load(open(glob.glob(os.path.join(runs[0], "*ParticipationTrace.pkl"))[0], "rb"))
        it, L = F1.clean_loss(runs[0], tr)
        it, L = np.asarray(it, float), np.asarray(L, float)
        m = it >= PR.T_START
        tb, yb = logbin(it[m], L[m])
        f = least_squares(lambda q: np.log(np.clip(stretched(tb, *q), 1e-12, None)) - np.log(yb),
                          [yb.min() * .9, float(yb.max()), 2e4, .4],
                          bounds=([1e-6, 1e-6, 1e2, .05], [1., 1e3, 1e8, 3.]), max_nfev=20000)
        floor, beta = float(f.x[0]), float(f.x[3])
        resid = float(np.sqrt(np.mean(f.fun ** 2)))
        ax.plot(it, L, lw=0.35, color=ps.FAINT, zorder=2)
        ax.plot(tb, stretched(tb, *f.x), lw=1.2, color=col, zorder=5)
        ax.axhline(floor, color=ps.MUTED, lw=0.7, ls=":", zorder=3)
        ax.axhline(floor * (1 + PR.EXCESS_DELTA), color=ps.MUTED, lw=0.8, ls="--", zorder=3)
        T = PR.excess_time(L, floor, PR.EXCESS_DELTA)
        if np.isfinite(T):
            # excess_time already returns an ITERATION, not a sample index - multiplying by the
            # probe step put the marker off the right-hand end of the data
            ax.plot([T], [floor * (1 + PR.EXCESS_DELTA)], "o", ms=5, color=col,
                    mec="white", mew=0.8, zorder=6)
        ok = resid < 0.1
        ax.set_title(f"{lab}\n" + r"$\beta$ = " + f"{beta:.2f}, residual {resid:.3f}"
                     + ("" if ok else "  — does not fit"),
                     fontsize=6.8, color=ps.INK if ok else ps.BAD, linespacing=1.3, pad=4)
        ax.set(xscale="log", yscale="log", xlabel="training iteration")
        ps.ygrid(ax)
    np.atleast_1d(axes)[0].set_ylabel("clean task loss")
    return ps.save(fig, "slide_05a_floor_fit")


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
        print(f"  {'slide_x_metabolic_r2':28s} "
              "No rung of the ladder trades performance for live units")
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
    got = floor_fit_slide()
    if got:
        out.append(got)
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

    got = metabolic_r2_slide()
    if got:
        out.append(got)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="name the slides without drawing them")
    args = ap.parse_args()
    main(list_only=args.list)
