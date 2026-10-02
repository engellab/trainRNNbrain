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
import re

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
from matplotlib.ticker import NullFormatter, NullLocator, ScalarFormatter
from matplotlib import transforms

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import hydra.utils
from omegaconf import OmegaConf

import torch

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.rnns.RNN_torch import drop_probabilities
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.utils import filter_kwargs

import paperstyle as ps
import common_r2 as common_r2_mod
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
        (reference array, [(label, values array, group), ...], iters) with empty arrays where a cell
        is missing, so a slide shows the gap rather than silently dropping a condition. `iters` is
        the set of iterations the family's cells were actually read at - one element when every cell
        shares a probe, more when they do not - and is EMPTY for the CSV and noise families, whose
        surviving summaries carry no iteration (see ARCHIVE_READOUT).
    """
    if trace:
        title, ref_pat, cap, items = family
        got = F1.live_matched(ref_pat, cap)
        iters = {got[1]} if got else set()
        ref = (got or (np.array([]),))[0]
        out = []
        for lab, pat, grp in items:
            got = None if pat is None else F1.live_matched(pat, cap)
            if got:
                iters.add(got[1])
            v = (got or (np.array([]),))[0]
            out.append((lab, np.asarray(v, float), grp))
        return np.asarray(ref, float), out, iters
    title, ref_spec, items = family
    if isinstance(ref_spec, tuple):
        ref = F1.csv_active(ref_spec[0], **ref_spec[1])
        out = [(lab, np.asarray(F1.csv_active(s[0], **s[1]), float), grp) for lab, s, grp in items]
    else:
        # THE NOISE SWEEP'S SEEDS ARE BACK. It saved no per-network rows and scored silence on peak
        # rate rather than participation, so its slide drew a mean and a 95% interval where every
        # other family draws its networks. cddm_noise_participation.py re-scores the trained weights
        # under the shared rule and records each net, so per-seed arrays are used when that CSV is
        # present and the old per-condition summary remains the fallback.
        if F1.noise_counts(ref_spec):
            ref = np.asarray(F1.noise_counts(ref_spec), float)
            out = [(lab, np.asarray([] if s is None else F1.noise_counts(s), float), grp)
                   for lab, s, grp in items]
        else:
            ref = F1.noise_active(ref_spec)
            out = [(lab, (float("nan"), float("nan"), 0) if s is None else F1.noise_active(s), grp)
                   for lab, s, grp in items]
    return ref, out, set()


def collect():
    """Every intervention in Figure 1d, keyed by (family title, group).

    Returns:
        dict {(family title, group): (reference array, [(label, values), ...], iters)}, with `iters`
        the set of iterations that family was read at - empty where the data do not record it.
    """
    out = {}
    fams = [(f, True) for f in F1.TRACE_FAMILIES]
    fams += [(F1.ARCHIVE_FAMILY, False), (F1.NOISE_FAMILY, False)]
    for fam, trace in fams:
        title = fam[0]
        ref, items, iters = _family_values(fam, trace)
        for lab, vals, grp in items:
            if grp == "reference":
                continue
            out.setdefault((title, grp), (ref, [], iters))[1].append((lab, vals))
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


# Both size panels are manuscript panels, which carry no title because Figure 2's caption names the
# task and the budget once for all seven panels. A slide has no caption, so the title has to.
SIZE_LINE = ("3-bit flip-flop, N = 500\u20134000, every arm read at 40,000 iterations.\n"
             "Matched: $\\gamma$ = 0, lr 10$^{-3}$, weight decay 10$^{-6}$, "
             "$\\sigma_{rec}$ = $\\sigma_{inp}$ = 0.05 \u2014 the intervention is the only difference.")


def size_active_titled(ax, c):
    """Figure 2's panel (f) with the slide's own title: the task and the read-out iteration.

    Args:
        ax: axes; c: the full cache dict, every size.
    Returns:
        whatever F2.panel_f returns, {N: {arm: (mean, n)}}.
    """
    out = F2.panel_f(ax, c)
    ax.set_title("Active units against network size\n" + SIZE_LINE,
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    return out


def size_r2_with_legend(ax, c):
    """Figure 2's panel (g), plus the legend and title it does not carry in the manuscript.

    In the paper (g) sits beside (f) and reads off its neighbour's legend. On a slide it stands
    alone, so an unlabelled marker - the single-size `frm + rws` diamond especially - names nothing.

    Args:
        ax: axes; c: the full cache dict, every size.
    Returns:
        whatever F2.panel_g returns, {N: {arm: mean}}.
    """
    out = F2.panel_g(ax, c)
    # ABOVE THE AXES, not inside. Every in-panel corner is occupied here: the control and
    # duplication run along the top, synaptic noise climbs through the lower left, and duplication's
    # N = 4000 crash sweeps the lower right. A legend outside cannot collide with any of them.
    ax.legend(loc="lower center", bbox_to_anchor=(0.5, 1.01), fontsize=5.5, handlelength=1.1,
              borderaxespad=0.0, ncol=3, columnspacing=1.0)
    ax.set_title("Held-out $r^2$ against network size\n" + SIZE_LINE, fontsize=7.4,
                 color=ps.INK, linespacing=1.35, pad=28)
    return out


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
# ⚠️ THE TWO ARCHIVED FAMILIES CANNOT DERIVE THEIR READ-OUT: their raw sweeps were deleted
# (Supplementary S6) and the surviving summaries do not all carry it. `silent_stats_all.csv`, which
# supplies the archive family's reference and its architecture rows, has no `iters` column at all,
# and the noise sweep's per-condition CSV has none either. Recorded here instead, each verified
# against something that does carry it:
#   - the archive reference's five rows (rel_5p95 = 0.569, 0.572, 0.575, 0.598, 0.614) are five of
#     the twenty `silent_stats_v2.csv` sweep=std rows at N = 1000, eq = h, every one at iters=30000;
#   - the metabolic rows come from `silent_stats_v2.csv` directly, all twelve at iters=30000;
#   - the architecture rows' sweeps still have their run folders - CDDM_ptrack_g0,
#     CDDM_ptrack_g0_nodale and CDDM_ptrack_g0_nodale_trainablebias are all MI=30000;
#   - the noise sweep's run folders survive too (CDDM_fb2792_g0_noise), max_iter 30000, and its nets
#     were scored from their FINAL weights, so the budget is the read-out.
ARCHIVE_READOUT = {"CDDM, 30k (archived)": 30_000, "CDDM, 30k": 30_000}


# The CDDM control's own trajectory. The panels in this section read at different budgets because
# the sweeps have different budgets, so the control they are each measured against is a different
# number - 414 at 30,000 iterations, 272 at 200,000. That is not an inconsistency between panels, it
# is the paper's own claim showing up in the reference, and the deck says so once rather than leaving
# a listener to notice three control numbers and distrust all three.
CONTROL_TRAJ_CELL = f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=*"
CONTROL_TRAJ_MARKS = [(30_000, "slides 12\u201314 read here"), (200_000, "slides 8, 10, 11 read here")]


def control_trajectory_slide(name="slide_07b_control_trajectory"):
    """The CDDM control's active count against training iteration, with the read-out points marked.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cell has no traces.
    """
    curves = []
    for f in sorted(glob.glob(os.path.join(CONTROL_TRAJ_CELL, "*", "*ParticipationTrace.pkl"))):
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        P = np.asarray(d["participation"], float)
        it = np.asarray(d.get("participation_iters", d.get("iters", [])), float)
        n = min(len(P), len(it))
        if n < 2:
            continue
        curves.append((it[:n], np.array([active_count(p, "scalefree") for p in P[:n]], float)))
    if not curves:
        print(f"  SKIP {name}: no traces under {CONTROL_TRAJ_CELL}")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    for it, a in curves:
        ax.plot(it, a, lw=0.7, color=ps.BASE, alpha=0.55, zorder=3)
    grid = curves[0][0]
    mean = np.mean([np.interp(grid, it, a) for it, a in curves], axis=0)
    ax.plot(grid, mean, lw=1.5, color=ps.SLOTS[0], zorder=5)
    for i, (x, lab) in enumerate(CONTROL_TRAJ_MARKS):
        # clamp rather than skip: the trace's last probe is 199,900, so a 200,000 mark would be
        # dropped and the panel would show only one of the two read-out points it exists to compare
        x = min(x, float(grid.max()))
        y = float(np.interp(x, grid, mean))
        ax.axvline(x, color=ps.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)
        ax.plot([x], [y], "o", ms=5.0, color=ps.SLOTS[0], mec="white", mew=0.8, zorder=6)
        # the label sits at the TOP of its rule, in blended (data x, axes y) coordinates, not beside
        # the marker: the curve descends across the panel, so a label offset from the 200,000 point
        # runs back across the data it is annotating
        # STAGGERED, and both hanging to the LEFT of their rule. 30,000 and 200,000 are close
        # together on a log axis, so two labels at one height collide with each other, and a label
        # offset rightward from the 200,000 rule leaves the panel.
        ax.annotate(f"{lab}\n{y:.0f} active", xy=(x, 0.98 - 0.16 * i),
                    xycoords=transforms.blended_transform_factory(ax.transData, ax.transAxes),
                    textcoords="offset points", xytext=(-5, -2), ha="right", va="top",
                    fontsize=6.2, color=ps.INK, linespacing=1.25)
    ax.set(xscale="log", xlabel="training iteration", ylabel="active units of 1000")
    ax.set_ylim(top=ax.get_ylim()[1] * 1.18)      # headroom for the two rule labels
    ax.set_title(f"CDDM, $N$ = 1000, unpenalised, {len(curves)} seeds\n"
                 "active units against training iteration",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, name)


def dropout_along_training_slide(name="slide_24c_dropout_along_training"):
    """Does dropout hold units open, or only delay the same silencing? Two panels, one condition.

    THE FIGURE THIS REPLACES DREW TWELVE PANELS and answered the question in one of them. Four
    penalty columns x (two silence criteria + loss): the frm and both columns sit pinned at ~1000
    live units, where the penalty saturates the count and dropout cannot be assessed at all; the
    second criterion repeats the first; and the loss row says only that every arm solves the task. It
    also drew a `dead` dropout arm whose cells are no longer on disk - that sweep was abandoned for
    mute (commit ae52c58, "dead is unstable under targeting") - so the panel could not be rebuilt
    from what exists.

    What is left is the question the script's own docstring asks: the 150k table shows dropout keeping
    more units alive, and a table cannot separate "holds them open" from "slows the same decline".
    The trace separates them, and the answer is the second: dropout buys a level, not a halt.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cells are missing.
    """
    import fig_dropout_live as DL
    root = DL.DATA_DIR
    arms = [("no dropout", ps.BASE,
             DL.cell(os.path.join(root, DL.REFS["none"]))
             + DL.cell(os.path.join(root, DL.DROP_SUB, "EqType=h_k=3_N=1000_pen=none_do=none"))),
            ("mute dropout", ps.SLOTS[0],
             DL.cell(os.path.join(root, DL.DROP_SUB, "EqType=h_k=3_N=1000_pen=none_do=mute")))]
    arms = [a for a in arms if a[2]]
    if len(arms) < 2:
        print(f"  SKIP {name}: dropout cells missing")
        return None
    ps.setup()
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(ps.W2, 62 * ps.MM))
    stats = {}
    for lab, col, traces in arms:
        ends, slopes = [], []
        for t in traces:
            pit, sf = np.asarray(t["pit"], float), np.asarray(t["sf"], float)
            ax.plot(pit, sf, lw=0.6, color=col, alpha=0.45, zorder=3)
            ends.append(sf[-1])
            m = pit >= pit.max() / 10.0            # the last decade, where the curves are straight
            if m.sum() > 3:
                slopes.append(np.polyfit(np.log10(pit[m]), sf[m], 1)[0])
            it, loss = np.asarray(t["it"], float), np.asarray(t["loss"], float)
            n = min(len(it), len(loss))
            ax2.plot(it[:n], loss[:n], lw=0.6, color=col, alpha=0.45, zorder=3)
        grid = np.asarray(traces[0]["pit"], float)
        mean = np.mean([np.interp(grid, np.asarray(t["pit"], float),
                                  np.asarray(t["sf"], float)) for t in traces], axis=0)
        ax.plot(grid, mean, lw=1.5, color=col, zorder=5, label=f"{lab} ({len(traces)})")
        stats[lab] = (float(np.mean(ends)), float(np.mean(slopes)))
    ax.set(xscale="log", xlabel="training iteration", ylabel="live units of 1000")
    ax.legend(loc="lower left", fontsize=6.0, handlelength=1.2)
    ps.ygrid(ax)
    # ⚠️ ONE no-dropout SEED SPIKES to ~1e7 near iteration 2,000 and, left on the axis, stretches it
    # over eight decades and flattens the comparison into a single line. The axis is clipped to the
    # band the two arms actually occupy and the excursion is named in the caption, rather than the
    # panel silently showing a flat pair of curves.
    spikes = sum(1 for _lab, _c, tr in arms for t in tr
                 if np.nanmax(np.asarray(t["loss"], float)) > 1.0)
    lo = min(float(np.nanmin(np.asarray(t["loss"], float))) for _l, _c, tr in arms for t in tr)
    ax2.set(xscale="log", yscale="log", xlabel="training iteration", ylabel="clean training loss",
            ylim=(lo * 0.7, 1.6))      # headroom so the 1e0 tick is not printed on the frame edge
    ps.ygrid(ax2)
    (e0, s0), (e1, s1) = stats["no dropout"], stats["mute dropout"]
    fig.suptitle("3-bit flip-flop, $N$ = 1000, unpenalised, every seed\n"
                 f"left: live units against iteration;  right: clean training loss\n"
                 f"at 150,000 iterations: no dropout {e0:.0f} units, mute dropout {e1:.0f}",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, y=1.02)
    # below the panels: a fourth title line ran into the loss axis's top tick label
    fig.text(0.5, -0.02,
             f"Both arms end at the same loss. The loss axis is clipped at 1; "
             f"{spikes} run{'s' if spikes != 1 else ''} spike above it early.",
             ha="center", va="top", fontsize=6.6, color=ps.MUTED)
    return ps.save(fig, name)


# --------------------------------------------------------------------------------------------
# DROPOUT: the design space (slide 23) and the rate x targeting sweep (slide 24)
# --------------------------------------------------------------------------------------------
#
# Dropout in this project is not one method but four independent choices, and the first sweep we
# ran was null because three of them were set wrongly rather than because dropout does not work.
# The post-fix grid in `NBitFlipFlop_std_bernoulli` varies two of them and is monotone in both, so
# the two slides are: what the choices are, and what the sweep says.
#
# Every cell below is the SAME launcher: 3-bit flip-flop, k = 3, N = 1000, no penalty, 40,000
# iterations, 3 seeds, standard architecture (bias fixed at 0, self-connections on). The control is
# the `do=none` cell of the matched launcher, so nothing is pooled across commits.
BERN = "NBitFlipFlop_std_bernoulli"
BERN_CTRL = "NBitFlipFlop_std_dropfix/EqType=h_k=3_N=1000_pen=none_do=none"
BERN_RATES = (0.05, 0.10, 0.175, 0.25)
BERN_BETAS = (1, 2, 4)
BERN_KINDS = (("mute", ps.COND_COL["mute"]), ("dead", ps.COND_COL["dead"]))
P_MAX = 0.9          # dropout_args.p_max in every one of these configs
ACTIVE_REL = 0.05    # dropout_args.active_rel - the project's own scale-free silence rule


def bern_cell(kind, rate, beta):
    """Path of one cell of the post-fix dropout grid.

    Args:
        kind: 'mute' or 'dead'; rate: drop rate as a float (printed as the launcher writes it);
        beta: targeting exponent, an int.
    Returns:
        absolute path of the cell folder (which may not exist).
    """
    r = f"{rate:.2f}" if rate != 0.175 else "0.175"
    return os.path.join(DATA_DIR, BERN,
                        f"EqType=h_k=3_N=1000_pen=none_do={kind}_rate={r}_beta={beta}")


def bern_read(path):
    """Per-seed active count and held-out R^2 in the common noise condition, at end of training.

    ⚠️ THE PERFORMANCE MEASURE IS NO LONGER THE NOISE-FREE PROBE. This returned the trace's
    dropout-off, noise-OFF loss, which is not a condition any of these networks ever operates in,
    and the project found that instrument reverses the ranking of arms: measured without noise
    frm+rws is the best arm and measured with it the worst. Every performance number now comes from
    common_r2 - a held-out batch, sigma_rec and sigma_inp at the values the network trained with,
    sigma_w = 0, averaged over nine draws - so arms differ in their inputs and in nothing else.
    Dropout is off in that evaluation, so a dropout net is still scored on its full network.

    The ACTIVE COUNT is unchanged and stays noise-free: a count of units above a participation
    threshold is not a performance measure, and rectified recurrent noise lifts every unit onto a
    floor that makes the count meaningless (see f2_remedies_cache.analyse).

    Args:
        path: cell folder holding one subfolder per network.
    Returns:
        (active, r2) float arrays, one entry per seed; both empty if the cell is absent.
    """
    act, r2 = [], []
    cache = common_r2_mod.load_cache()
    for d in sorted(glob.glob(os.path.join(path, "*", ""))):
        f = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not f:
            continue
        with open(f[0], "rb") as fh:
            tr = pickle.load(fh)
        act.append(active_count(np.asarray(tr["participation"][-1], float), "scalefree"))
        r2.append(common_r2_mod.common_r2(d.rstrip(os.sep), cache=cache))
    common_r2_mod.save_cache(cache)
    return np.asarray(act, float), np.asarray(r2, float)


def _unit_schematic(ax, y0, cut, title, note, label_arrows=False):
    """One row of panel (a): a unit, what reaches it, what it sends, and which of those are cut.

    Args:
        ax: a blank axes spanning 0..1 in both directions;
        y0: vertical centre of this row in axes coordinates;
        cut: which arrows carry a cross - any of 'drive', 'noise', 'rec', 'out';
        title: the variant's name, drawn at the left;
        note: one line under the glyph saying what survives;
        label_arrows: name the two recurrent arrows (top row only - see below).
    Returns:
        None.
    """
    col = ps.INK
    ax.text(0.0, y0 + 0.155, title, fontsize=7.4, fontweight="bold", color=col, ha="left")
    ax.text(0.0, y0 - 0.125, note, fontsize=6.2, color=ps.MUTED, ha="left", va="top",
            linespacing=1.3)
    ps.box(ax, 0.02, y0 - 0.065, 0.20, 0.13, "other units", col=ps.MUTED, fs=6.0)
    ps.box(ax, 0.44, y0 - 0.065, 0.14, 0.13, "unit $i$",
           col=ps.FAINT if "drive" in cut else col,
           text_col=ps.FAINT if "drive" in cut else col, fs=6.5, lw=1.0)
    ps.box(ax, 0.80, y0 - 0.065, 0.18, 0.13, "read-out $y$", col=ps.MUTED, fs=6.0)
    # incoming drive and the unit's own noise feed the unit; the unit feeds the others and the
    # read-out. `mute` cuts exactly one of the four, `dead` cuts all four.
    arrows = {"drive": ((0.22, y0 + 0.035), (0.44, y0 + 0.035), 0.0),
              "rec":   ((0.44, y0 - 0.035), (0.22, y0 - 0.035), 0.0),
              "out":   ((0.58, y0), (0.80, y0), 0.0),
              "noise": ((0.51, y0 + 0.150), (0.51, y0 + 0.066), 0.0)}
    for k, (a, b, rad) in arrows.items():
        gone = k in cut
        ps.arrow(ax, a, b, col=ps.FAINT if gone else ps.MUTED, rad=rad,
                 ls=(0, (1.6, 1.4)) if gone else "-", lw=0.9)
        if gone:
            m = ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)
            ax.plot([m[0]], [m[1]], marker="x", ms=4.6, mew=1.3, color=ps.BAD, zorder=6)
    ax.text(0.535, y0 + 0.155, r"$\eta_i$", fontsize=6.2, color=ps.MUTED, ha="left", va="center")
    # the two recurrent arrows are named on the top row only; repeating them underneath puts a word
    # on top of the cross that is the whole content of the bottom row.
    if label_arrows:
        ax.text(0.30, y0 + 0.050, "drive", fontsize=5.8, color=ps.MUTED, ha="center", va="bottom")
        ax.text(0.30, y0 - 0.050, "rate", fontsize=5.8, color=ps.MUTED, ha="center", va="top")


def dropout_selection_slide(name="slide_23_dropout_selection", beta=2, rho=0.25, seed=3):
    """One claim: every unit gets its own drop probability, then its own coin flip.

    23b and 23c each zoom into one step of this chain, so the chain has to be shown first. The
    numbers are not illustrative: a real control network's final participation vector goes through
    the production live-pool rule, the production softmax over ranks and the production
    `drop_probabilities`, and fourteen of its thousand units are printed.

    Args:
        name: output file stem; beta: targeting exponent to draw; rho: drop rate;
        seed: generator seed for the one Bernoulli draw shown in the last column.
    Returns:
        the output path, or None if the control trace is missing.
    """
    ctrl = sorted(glob.glob(os.path.join(DATA_DIR, BERN_CTRL, "*", "")))
    if not ctrl:
        print(f"  SKIP {name}: {BERN_CTRL} missing")
        return None
    with open(glob.glob(os.path.join(ctrl[0], "*ParticipationTrace.pkl"))[0], "rb") as fh:
        v = np.asarray(pickle.load(fh)["participation"][-1], float)
    live = v >= ACTIVE_REL * np.quantile(v, 0.95)
    M = int(live.sum())
    vp = torch.tensor(v[live], dtype=torch.float32)
    order = torch.argsort(vp)
    rank = torch.empty_like(vp)
    rank[order] = torch.arange(M, dtype=vp.dtype)
    w = torch.softmax(beta * rank / max(M - 1, 1), dim=0)
    p_live = drop_probabilities(w, rho * M, p_max=P_MAX)
    drawn = (torch.bernoulli(p_live, generator=torch.Generator().manual_seed(seed)) > 0).numpy()

    # fourteen units: four silent ones and ten spanning the live ranks, so the panel shows both the
    # exclusion and the gradient rather than fourteen units from the same part of the range.
    idx_live = np.flatnonzero(live)
    rk = rank.numpy()
    pick_live = idx_live[np.argsort(rk)][np.linspace(0, M - 1, 10).astype(int)]
    pick_dead = np.flatnonzero(~live)[np.linspace(0, int((~live).sum()) - 1, 4).astype(int)]
    rows = sorted(np.concatenate([pick_live, pick_dead]),
                  key=lambda i: -v[i])
    pos = {j: k for k, j in enumerate(idx_live)}

    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 70 * ps.MM))
    # a blank table fills its canvas; the default subplot margins would leave a quarter of the
    # slide empty on the left while the title ran the full width above it.
    fig.subplots_adjust(left=0.01, right=0.99, top=0.90, bottom=0.02)
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    X = dict(unit=0.015, v=0.175, live=0.275, rank=0.40, p=0.50, bar=0.56, drawn=0.965)
    head = [(X["unit"], "unit", "left"), (X["v"], "rate $v_i$", "right"),
            (X["live"], "live?", "center"), (X["rank"], "rank", "right"),
            (X["p"], "$p_i$", "right"), (X["bar"] + 0.10, "drop probability", "left"),
            (X["drawn"], "drawn", "center")]
    for x, t, ha in head:
        ax.text(x, 0.955, t, fontsize=6.6, color=ps.MUTED, ha=ha, va="bottom")
    ax.plot([0, 1], [0.935, 0.935], lw=0.6, color=ps.GRID, zorder=1)

    ys = np.linspace(0.875, 0.055, len(rows))
    for y, i in zip(ys, rows):
        on = bool(live[i])
        col = ps.INK if on else ps.FAINT
        ax.text(X["unit"], y, f"#{i}", fontsize=6.4, color=col, va="center")
        ax.text(X["v"], y, f"{v[i]:.2f}" if v[i] >= 0.01 else f"{v[i]:.0e}",
                fontsize=6.4, color=col, va="center", ha="right")
        ax.plot([X["live"]], [y], "o", ms=3.4, mew=0.8,
                color=ps.COND_COL["mute"] if on else "none",
                mec=ps.COND_COL["mute"] if on else ps.FAINT, zorder=3)
        if not on:
            ax.text(X["rank"] + 0.03, y, "never drawn", fontsize=6.2, color=ps.FAINT,
                    va="center", ha="left", style="italic")
            continue
        k = pos[i]
        pi = float(p_live[k])
        ax.text(X["rank"], y, f"{int(rk[k]) + 1}", fontsize=6.4, color=col, va="center", ha="right")
        ax.text(X["p"], y, f"{pi:.2f}", fontsize=6.4, color=col, va="center", ha="right")
        ax.add_patch(Rectangle((X["bar"], y - 0.016), 0.38 * pi, 0.032, lw=0,
                               color=ps.COND_COL["mute"], alpha=0.85, zorder=3))
        ax.plot([X["drawn"]], [y], marker="x" if drawn[k] else ".",
                ms=5.0 if drawn[k] else 3.0, mew=1.3,
                color=ps.BAD if drawn[k] else ps.FAINT, zorder=4)
    fig.suptitle("The dropout sampler, step by step\n"
                 f"rate $v_i$  $\\rightarrow$  live pool  $\\rightarrow$  rank  $\\rightarrow$  "
                 rf"$p_i \propto e^{{\beta\,\mathrm{{rank}}/M}}$, scaled so they sum to "
                 rf"{rho:g}$M$  $\rightarrow$  Bernoulli($p_i$), one draw per unit per step",
                 fontsize=8.0, color=ps.INK, linespacing=1.5, y=1.035)
    fig.text(0.5, -0.01,
             f"A real control network at iteration 40,000: {M} of its 1000 units are live, "
             rf"so $\rho$ = {rho:g} spends {rho * M:.0f} drops on them. "
             f"Fourteen units shown, at $\\beta$ = {beta}.",
             ha="center", va="top", fontsize=6.6, color=ps.MUTED)
    return ps.save(fig, name, w_mm=110)


def dropout_kinds_slide(name="slide_23d_dropout_kinds"):
    """One claim: a dropped unit can lose only its read-out, or everything.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 54 * ps.MM))
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    _unit_schematic(ax, 0.76, {"out"}, "mute",
                    "keeps running, keeps driving its neighbours,\n"
                    "disappears only from the output", label_arrows=True)
    _unit_schematic(ax, 0.26, {"drive", "noise", "rec", "out"}, "dead",
                    "loses its own drive and its own noise too,\nso it decays to zero and sends nothing")
    fig.suptitle("The two dropout kinds: mute and dead",
                 fontsize=8.6, color=ps.INK, y=1.02)
    return ps.save(fig, name, w_mm=110)


def dropout_targeting_slide(name="slide_23b_dropout_targeting"):
    """One claim: beta sets how hard the sampler aims at the busiest living units.

    The curves come from the production `drop_probabilities` on the production weight vector, so
    the panel cannot drift away from the sampler training actually uses.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 62 * ps.MM))
    M = 400
    rank = torch.arange(M, dtype=torch.float32)
    for beta, lw in zip((0, 1, 2, 4), (0.9, 1.1, 1.5, 2.0)):
        w = torch.softmax(beta * rank / (M - 1), dim=0)
        p = drop_probabilities(w, 0.25 * M, p_max=P_MAX).numpy()
        ax.plot(np.arange(M) / (M - 1), p, lw=lw, zorder=3,
                color=ps.MUTED if beta == 0 else ps.COND_COL["mute"],
                label=r"$\beta$ = " + (f"{beta}" if beta else "0  (uniform)"))
    ax.axhline(P_MAX, color=ps.BAD, lw=0.7, ls=(0, (2.2, 1.8)), zorder=2)
    ax.text(0.5, P_MAX + 0.015, r"ceiling $p_{\max}$ = 0.9", fontsize=6.2, color=ps.BAD,
            va="bottom", ha="center")
    ax.set(xlim=(0, 1), ylim=(0, 1),
           xlabel="rank among the live units (0 = quietest, 1 = busiest)",
           ylabel="probability of being dropped")
    ax.legend(loc="upper left", bbox_to_anchor=(0.0, 0.86), fontsize=6.8, handlelength=1.8)
    ps.ygrid(ax)
    fig.suptitle(r"Drop probability against activity rank, at three $\beta$"
                 "\nThe busiest living unit is $e^{\\beta}$ times likelier than the quietest: "
                 r"2.7, 7.4, 55 at $\beta$ = 1, 2, 4.",
                 fontsize=8.0, color=ps.INK, linespacing=1.4, y=1.03)
    return ps.save(fig, name, w_mm=110)


def dropout_dose_slide(name="slide_23c_dropout_dose", rho=0.25):
    """One claim: the rate is a share of the units still alive, not of N.

    THE EARLIER VERSION DREW THE DOSE AS A SECOND CURVE, rho * M beside M. On a log axis that is
    the same curve shifted down by a constant, so the panel carried one measurement twice and read
    as a comparison between two things. Only the reference matters, so only the reference is drawn.

    TWO OF THE THREE CONTROL SEEDS PASS THROUGH A TRANSIENT BLOW-UP, and the first version of this
    panel drew one of them alone with no sign of it. Participation reaches ~1e6 and the clean loss
    1e9 or worse for five consecutive snapshots, then the run recovers and finishes at its usual
    loss. The scale-free criterion is scale-INVARIANT - the bar is 5% of the 95th percentile, which
    rises with the excursion - so the active count walks straight through reporting an ordinary
    number. Every seed is drawn and the windows are shaded: a measure that cannot see a 1e6-fold
    rate excursion is worth knowing about wherever that measure is used, and it is used everywhere
    in this deck.

    Args:
        name: output file stem; rho: the drop rate whose dose is being compared.
    Returns:
        the output path, or None if the control cells are missing.
    """
    runs = []
    for d in sorted(glob.glob(os.path.join(DATA_DIR, BERN_CTRL, "*", ""))):
        fs = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not fs:
            continue
        with open(fs[0], "rb") as fh:
            tr = pickle.load(fh)
        P = [np.asarray(p, float) for p in tr["participation"]]
        runs.append((np.asarray(tr["participation_iters"], float),
                     np.array([active_count(p, "scalefree") for p in P], float),
                     np.array([np.quantile(p, 0.95) for p in P], float)))
    if not runs:
        print(f"  SKIP {name}: {BERN_CTRL} missing")
        return None

    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 62 * ps.MM))
    blew = 0
    for it, live, q95 in runs:
        pos = it > 0
        ax.plot(it[pos], live[pos], color=ps.BASE, lw=1.0, alpha=0.85, zorder=4)
        # a healthy net's q95 stays under 5 all through training; the excursions reach 1e5 to 1e7.
        hot = it[q95 > 100]
        if len(hot):
            blew += 1
            ax.axvspan(hot.min(), hot.max(), color=ps.BAD, alpha=0.12, lw=0, zorder=1)
    ax.axhline(rho * 1000, color=ps.BAD, lw=1.2, ls=(0, (2.4, 1.8)), zorder=3)
    ends = sorted(int(live[-1]) for _i, live, _q in runs)
    ax.text(115, 1080, f"units still alive, {len(runs)} control seeds", fontsize=6.8,
            color=ps.BASE, va="bottom")
    ax.text(115, rho * 1000 * 0.94, rf"$\rho N$ = {rho * 1000:.0f}, a share of the whole net",
            fontsize=6.8, color=ps.BAD, va="top")
    ax.set(xscale="log", yscale="log", xlim=(100, 5e4), ylim=(150, 1500),
           xlabel="training iteration", ylabel="units")
    # one decade of range, so the log locator fills the axis with 2x10^2-style minor labels
    ax.set_yticks([200, 300, 500, 1000])
    ax.set_yticklabels(["200", "300", "500", "1000"])
    ax.yaxis.set_minor_formatter(NullFormatter())
    ps.ygrid(ax)
    fig.suptitle("3-bit flip-flop, $N$ = 1000: the live pool, and the drops it is charged\n"
                 f"The pool falls to {ends[0]}-{ends[-1]} units by iteration 40,000, so "
                 rf"$\rho$ = {rho:g} spends {rho * ends[0]:.0f}-{rho * ends[-1]:.0f} drops a step, "
                 rf"where $\rho N$ would spend {rho * 1000:.0f}.",
                 fontsize=8.0, color=ps.INK, linespacing=1.4, y=1.03)
    fig.text(0.5, -0.015,
             f"Shaded: {blew} of the {len(runs)} seeds pass through a transient blow-up, "
             "participation reaching $10^6$ and the clean loss $10^9$ or worse, then recover.\n"
             "The active count is a fraction of the 95th percentile, so it rescales with the "
             "excursion and reports an ordinary number straight through it.",
             ha="center", va="top", fontsize=6.4, color=ps.MUTED, linespacing=1.4)
    return ps.save(fig, name, w_mm=110)


def _right_labels(ax, items, gap=0.050, x=1.012):
    """Label curves at the right edge of an axes, pushed apart so no two overlap.

    Six curves end within a few units of each other, so labels printed at their true heights sit on
    top of one another. They are converted to axes coordinates, separated by a minimum gap and
    drawn there; the displacement is at most a few per cent of the axis and each label keeps its
    curve's colour, so nothing is misattributed.

    Args:
        ax: a LINEARLY scaled axes; items: (y in data units, text, colour) per label;
        gap: minimum separation in axes fraction; x: label position in axes fraction.
    Returns:
        None.
    """
    items = sorted(items, key=lambda t: t[0])
    ys = [ax.transLimits.transform((0.0, y))[1] for y, _, _ in items]
    for i in range(1, len(ys)):
        ys[i] = max(ys[i], ys[i - 1] + gap)
    for (_, txt, col), ya in zip(items, ys):
        ax.text(x, ya, txt, transform=ax.transAxes, fontsize=6.2, color=col,
                va="center", ha="left")


def _sweep_panel(ax, grid, ref, j, rng):
    """Draw the rate x targeting grid of one measure: six curves, every seed, a control band.

    Args:
        ax: axes; grid: {(kind, rate, beta): (active, loss)}; ref: the control's per-seed values of
        the measure; j: 0 for active units, 1 for clean loss; rng: generator for the dot jitter.
    Returns:
        None.
    """
    ax.axhspan(ref.mean() - ref.std(ddof=1), ref.mean() + ref.std(ddof=1),
               color=ps.BASE, alpha=0.16, lw=0, zorder=1)
    ax.axhline(ref.mean(), color=ps.BASE, lw=1.1, zorder=2)
    ends = []
    for kind, col in BERN_KINDS:
        for beta, lw in zip(BERN_BETAS, (0.9, 1.4, 2.0)):
            m = np.array([grid[(kind, r, beta)][j].mean() for r in BERN_RATES])
            ax.plot(BERN_RATES, m, color=col, lw=lw, zorder=4,
                    ls="-" if kind == "mute" else (0, (2.6, 1.4)),
                    marker="o" if kind == "mute" else "s", ms=2.8, mec="none")
            for r in BERN_RATES:
                v = grid[(kind, r, beta)][j]
                ax.plot(r + rng.normal(0, 0.004, len(v)), v, ".", ms=2.0, color=col,
                        alpha=0.40, mec="none", zorder=3)
            ends.append((m[-1], rf"$\beta$ = {beta}", col))
    ax.set(xlim=(0.03, 0.27), xticks=BERN_RATES, xticklabels=[f"{r:g}" for r in BERN_RATES],
           xlabel=r"drop rate $\rho$")
    ps.ygrid(ax)
    _right_labels(ax, ends)
    ax.legend(handles=[Line2D([], [], color=c, lw=1.6,
                              ls="-" if k == "mute" else (0, (2.6, 1.4)),
                              marker="o" if k == "mute" else "s", ms=3.2, mec="none", label=k)
                       for k, c in BERN_KINDS],
              loc="upper left", fontsize=6.8, handlelength=2.4)


def bern_grid():
    """The post-fix dropout grid and its control, or (None, None, None) if a cell is missing.

    Returns:
        (control active, control loss, {(kind, rate, beta): (active, loss)}).
    """
    ca, cl = bern_read(os.path.join(DATA_DIR, BERN_CTRL))
    g = {(k, r, b): bern_read(bern_cell(k, r, b))
         for k, _ in BERN_KINDS for r in BERN_RATES for b in BERN_BETAS}
    if not len(ca) or any(not len(v[0]) for v in g.values()):
        return None, None, None
    return ca, cl, g


def dropout_rate_units_slide(name="slide_24_dropout_rate_units"):
    """One claim: a higher rate, and a sharper aim, keep more units alive.

    THE HEADLINE IS COMPUTED AND TESTED AGAINST THE SEED SCATTER, not typed. "Monotone in both
    knobs" is false for `dead` at beta = 1. "Sharper targeting adds units at every setting" then
    passed a monotonicity test but failed a cross-check: the Figure 2 cache, which rebuilds each
    network from its saved parameters instead of reading the trace, has mute at rho = 0.05 going
    386 -> 384 -> 406. The two counts differ by at most 17 units of 1000 and agree on every other
    ordering, so a 25-unit rise against a 59-unit seed sd was never resolvable. A series counts only
    if it rises at every step AND rises by more than its seeds scatter.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the grid is incomplete.
    """
    ctrl_a, _, grid = bern_grid()
    if grid is None:
        print(f"  SKIP {name}: dropout grid incomplete")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 64 * ps.MM))
    _sweep_panel(ax, grid, ctrl_a, 0, np.random.default_rng(0))
    ax.set_ylabel("active units of 1000")
    ax.text(0.036, ctrl_a.mean() - 14, f"no dropout, {ctrl_a.mean():.0f}", fontsize=6.6,
            color=ps.BASE, va="top")

    def _rises(series):
        means = [v[0].mean() for v in series]
        sd = float(np.sqrt(np.mean([v[0].var(ddof=1) for v in series])))
        return bool(np.all(np.diff(means) > 0) and means[-1] - means[0] > sd)
    rate_bad = [rf"{k} at $\beta$ = {b}" for k, _ in BERN_KINDS for b in BERN_BETAS
                if not _rises([grid[(k, r, b)] for r in BERN_RATES])]
    beta_bad = [rf"{k} at $\rho$ = {r:g}" for k, _ in BERN_KINDS for r in BERN_RATES
                if not _rises([grid[(k, r, b)] for b in BERN_BETAS])]
    exc = " Exceptions: " + ", ".join(rate_bad + beta_bad) + "." if (rate_bad or beta_bad) else ""
    fig.suptitle(r"3-bit flip-flop, $N$ = 1000: active units over the $\rho$ x $\beta$ grid"
                 "\n"
                 f"{ctrl_a.mean():.0f} of 1000 with no dropout, "
                 f"{grid[('mute', 0.25, 4)][0].mean():.0f} at mute "
                 rf"$\rho$ = 0.25, $\beta$ = 4, "
                 f"{grid[('dead', 0.25, 4)][0].mean():.0f} at dead.{exc}",
                 fontsize=8.0, color=ps.INK, linespacing=1.4, y=1.035)
    return ps.save(fig, name, w_mm=110)


def dropout_rate_cost_slide(name="slide_24b_dropout_rate_cost"):
    """One claim: mute recruits units for free, dead pays for them.

    The loss is the trainer's own noise-free, dropout-off probe on a fresh batch, so a dropout net
    is scored on the full network exactly as the control is.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the grid is incomplete.
    """
    _, ctrl_l, grid = bern_grid()
    if grid is None:
        print(f"  SKIP {name}: dropout grid incomplete")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 64 * ps.MM))
    _sweep_panel(ax, grid, ctrl_l, 1, np.random.default_rng(0))
    ax.set_ylabel("task loss, noise-free, dropout off")
    ax.text(0.036, ctrl_l.mean() + 0.008, f"no dropout, {ctrl_l.mean():.3f}", fontsize=6.6,
            color=ps.BASE, va="bottom")
    mute = [grid[("mute", r, b)][1].mean() for r in BERN_RATES for b in BERN_BETAS]
    dead = [grid[("dead", r, b)][1].mean() for r in BERN_RATES for b in BERN_BETAS]
    fig.suptitle(r"3-bit flip-flop, $N$ = 1000: clean loss over the $\rho$ x $\beta$ grid"
                 "\n"
                 f"mute stays between {min(mute):.3f} and {max(mute):.3f} against the control's "
                 f"{ctrl_l.mean():.3f}; dead reaches {max(dead):.3f}, about "
                 f"{max(dead) / ctrl_l.mean():.1f} times the control.",
                 fontsize=8.0, color=ps.INK, linespacing=1.4, y=1.035)
    fig.text(0.5, -0.015,
             "Scored with the training noise switched back on, mute instead gives up 2 points of "
             "$R^2$ (0.945 to 0.922).",
             ha="center", va="top", fontsize=6.6, color=ps.MUTED)
    return ps.save(fig, name, w_mm=110)


# The penalty arms, one figure per task. The 3-bit flip-flop panels cannot carry them as a series:
# its only frm+rws cells are N = 1000, and the other penalised flip-flop cells are k = 7 and k = 8, a
# different task rather than a bigger network.
#
# ⚠️ CDDM_std_g0_penalties IS LOCAL-ONLY - it is not on Della. A cluster-side scan says CDDM has
# penalties at N = 1000 alone; the Mac has all three arms at 500, 1000, 2000 and 5000.
PEN_KINDS = [("control", "no penalty", ps.BASE), ("frm", "frm", ps.SLOTS[0]),
             ("rws", "rws", ps.SLOTS[2]), ("both", "frm + rws", ps.SLOTS[1])]
PEN_TASKS = {
    # task: (nice label, sizes, {kind: glob template})
    "CDDM": ("CDDM", [500, 1000, 2000, 5000], {
        "control": f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N={{N}}_iters=*",
        "frm": f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N={{N}}_pen=frm",
        "rws": f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N={{N}}_pen=rws",
        "both": f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N={{N}}_pen=both"}),
    "DMTS": ("DMTS, 7$\\tau$ delay", [500, 1000, 2000], {
        "control": f"{DATA_DIR}/DMTS_d7_pen/EqType=h_N={{N}}_pen=none",
        "frm": f"{DATA_DIR}/DMTS_d7_pen/EqType=h_N={{N}}_pen=frm",
        "rws": f"{DATA_DIR}/DMTS_d7_pen/EqType=h_N={{N}}_pen=rws",
        "both": f"{DATA_DIR}/DMTS_d7_pen/EqType=h_N={{N}}_pen=both"}),
}


def _pen_cell_stats(pattern):
    """Active units and the rule's health ratio for one cell.

    Args:
        pattern: a glob matching the cell directory.
    Returns:
        (mean active, sd, n seeds, mean q50/q95) or None if the cell has no traces. q50/q95 is the
        scale-free rule's own check: above 0.05 the median unit counts as active and the number is a
        floor rather than a count.
    """
    vals, ratios = [], []
    for cell in sorted(glob.glob(pattern)):
        for f in sorted(glob.glob(os.path.join(cell, "*", "*ParticipationTrace.pkl"))):
            try:
                d = pickle.load(open(f, "rb"))
            except Exception:
                continue
            P = np.asarray(d.get("participation", []), float)
            if P.ndim != 2 or not len(P):
                continue
            vals.append(active_count(P[-1], "scalefree"))
            q50, q95 = np.quantile(P[-1], 0.5), np.quantile(P[-1], 0.95)
            ratios.append(float(q50 / max(q95, 1e-12)))
    if not vals:
        return None
    # the PER-RUN values come back too: these panels draw every network, like every other panel in
    # the deck, rather than a mean with an error bar that hides how many runs there were
    return (float(np.mean(vals)), float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0,
            len(vals), float(np.mean(ratios)), np.asarray(vals, float))


def penalty_size_slide(task, name=None):
    """Active units against network size for every penalty arm, one task.

    Args:
        task: a key of PEN_TASKS; name: output stem, defaulting to the task's own.
    Returns:
        the output path, or None if the task has no cells.
    """
    lab, sizes, pats = PEN_TASKS[task]
    name = name or f"slide_22b_penalty_{task.lower()}"
    rows = {k: {n: st for n in sizes if (st := _pen_cell_stats(pats[k].format(N=n)))}
            for k, _l, _c in PEN_KINDS}
    rows = {k: v for k, v in rows.items() if v}
    if not rows:
        print(f"  SKIP {name}: no cells for {task}")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    allN = sorted({n for r in rows.values() for n in r})
    ax.plot(allN, allN, lw=0.8, ls=":", color=ps.MUTED, zorder=6)
    ax.annotate("every unit active", (allN[-1], allN[-1]), textcoords="offset points",
                xytext=(-3, 4), ha="right", fontsize=5.8, color=ps.MUTED)
    sat = []
    # ARMS ARE OFFSET TOO, not just runs within an arm. frm and frm + rws give identical counts
    # wherever both saturate (500/1000/2000 on CDDM), so one line sits exactly under the other and
    # the hidden one reads as missing data.
    drawn_kinds = [k for k, _l, _c in PEN_KINDS if rows.get(k)]
    for kind, klab, col in PEN_KINDS:
        r = rows.get(kind)
        if not r:
            continue
        arm_i = drawn_kinds.index(kind) - (len(drawn_kinds) - 1) / 2.0
        Ns = sorted(r)
        # EVERY RUN IS DRAWN, FANNED OUT. Where the count saturates the seeds are identical - frm
        # is 500/500/500 at N = 500 and 1000/1000/1000 at N = 1000 - so three markers land on the
        # same pixel and the panel looks like it has one run per size. A small multiplicative offset
        # in N (the axis is logarithmic) separates them without moving them off their own size.
        for n in Ns:
            v = r[n][4]
            base = n * 1.075 ** arm_i
            off = base * 1.028 ** (np.arange(len(v)) - (len(v) - 1) / 2.0)
            ax.plot(off, v, "o", ms=3.2, color=col, alpha=0.9, mec="none", zorder=5)
        xs = [n * 1.075 ** arm_i for n in Ns]
        ax.plot(xs, [r[n][0] for n in Ns], "-", lw=1.1, color=col, zorder=3)
        # the mean is HOLLOW so the runs underneath it stay visible
        ax.plot(xs, [r[n][0] for n in Ns], "s", ms=6.5, mfc="none", mec=col, mew=1.2, zorder=6,
                label=f"{klab} ({r[Ns[0]][2]} runs)")
        if any(r[n][3] > 0.05 for n in Ns):
            sat.append(klab)
    ax.set(xscale="log", yscale="log", xlabel="network size $N$", ylabel="active units")
    ax.set_xticks(allN)
    ax.set_xticklabels([str(n) for n in allN])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_title(f"{lab}: active units against network size", fontsize=7.6,
                 color=ps.INK, pad=5)
    ax.legend(loc="upper left", fontsize=6.0, handlelength=1.1, borderaxespad=0.25)
    ps.ygrid(ax)
    if sat:
        fig.text(0.5, -0.02, "circles are runs, squares their mean; both are offset sideways so "
                 "identical values stay visible.  "
                 + ", ".join(sat) + " sit on the diagonal: there the participation "
                 "distribution is unimodal, so the count is a floor.",
                 ha="center", va="top", fontsize=6.2, color=ps.MUTED)
    return ps.save(fig, name)


PEN_CACHE = "data/cddm_penalty_cache.npz"
PEN_ARMS = [("control", "no penalty", ps.BASE), ("frm", "frm", ps.SLOTS[0]),
            ("rws", "rws", ps.SLOTS[2]), ("both", "frm + rws", ps.SLOTS[1])]
BURST = 0.05          # fig_S3_transients' own cut: tPR/n below this is a burst unit


def _pen_cache():
    """The CDDM N = 1000 penalty cache, or None if it has not been built."""
    if not os.path.exists(PEN_CACHE):
        print(f"  (build it with cddm_penalty_cache.py: {PEN_CACHE} missing)")
        return None
    return np.load(PEN_CACHE, allow_pickle=True)


def temporal_pr_slide(name="slide_30_temporal_pr"):
    """What frm's extra units do with their time, and what rws changes about it.

    frm puts every unit over the silence bar, so the active-unit count saturates and cannot tell a
    unit that fires throughout the trial from one that fires in a brief transient. tPR/n does: 1 for
    a constant rate, near 0 for a burst.

    FOUR STACKED ROWS, NOT FOUR OVERLAID CURVES. The distributions sit on top of one another when
    drawn in one axes and the lower tail - the only place the two penalised arms differ - is exactly
    where they overlap most. A shared x axis keeps them comparable.

    THE EFFECT IS IN THE LOWER TAIL. Adding rws to frm moves the median from 0.123 to 0.125 and the
    bottom quartile from 0.028 to 0.060, so the quartile is marked on every row.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cache is missing.
    """
    z = _pen_cache()
    if z is None:
        return None
    rows = [(a_, l_, c_) for a_, l_, c_ in PEN_ARMS
            if any(k.startswith(f"{a_}|") and k.endswith("|tpr") for k in z.files)]
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 74 * ps.MM))
    gs = GridSpec(len(rows), 2, figure=fig, width_ratios=[1.0, 0.85], hspace=0.22, wspace=0.3)
    bins = np.linspace(0, 0.4, 61)
    axes = []
    stats = {}
    for i, (arm, lab, col) in enumerate(rows):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        per_seed = [z[k] for k in z.files if k.startswith(f"{arm}|") and k.endswith("|tpr")]
        v = np.concatenate(per_seed)
        ax.hist(v, bins=bins, color=col, alpha=0.85, density=True, zorder=3)
        q25 = float(np.quantile(v, 0.25))
        ax.axvline(q25, color=ps.INK, lw=0.9, ls=(0, (3, 2)), zorder=5)
        ax.axvline(BURST, color=ps.MUTED, lw=0.7, ls=":", zorder=4)
        ax.annotate(f"{lab}   q25 = {q25:.3f}", (0.97, 0.78), xycoords="axes fraction",
                    ha="right", fontsize=6.2, color=ps.INK)
        stats[arm] = ([float(np.quantile(x, 0.25)) for x in per_seed],
                      [100.0 * float(np.mean(x < BURST)) for x in per_seed])
        ps.ygrid(ax)
        ax.set_yticks([])
        if i < len(rows) - 1:
            ax.tick_params(labelbottom=False)
    # the parenthetical ran off the left edge as an x label; it belongs in the title
    axes[-1].set_xlabel("temporal participation ratio, tPR / n")
    axes[len(rows) // 2].set_ylabel("density")

    # ---- what a burst unit and a sustained unit actually look like --------------------------------
    ex_arm = "frm" if f"frm|0|ex" in z.files else rows[0][0]
    ex = np.asarray(z[f"{ex_arm}|0|ex"], float)
    ex_tpr = np.asarray(z[f"{ex_arm}|0|ex_tpr"], float)
    for j, (ttl, col) in enumerate([("lowest tPR/n in this net", ps.SLOTS[1]),
                                    ("highest tPR/n", ps.SLOTS[2])]):
        ax = fig.add_subplot(gs[j * (len(rows) // 2) if len(rows) > 2 else j, 1])
        ax.plot(ex[j], lw=1.2, color=col, zorder=4)
        ax.set_title(f"{ttl}:  tPR/n = {ex_tpr[j]:.3f}", fontsize=6.6, color=ps.INK, pad=3)
        ax.set_ylabel("rate", fontsize=6.2)
        if j:
            ax.set_xlabel("time step")
        ps.ygrid(ax)
    fig.suptitle("CDDM, $N$ = 1000, 200,000 iterations, 3 seeds per condition\n"
                 "left: temporal participation ratio of every live unit, "
                 "dotted = burst cut, dashed = that row's quartile\n"
                 "right: two units of one frm net, trial-averaged",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, y=1.03)
    return ps.save(fig, name)


SEL_CACHE = "data/cddm_selectivity_cache.npz"
# Per condition: how many arms its configuration has, which seed to draw, and the viewing angle.
# ALL THREE ARE MEASURED, NOT CHOSEN BY EYE. The arm count is where the k-means clusters stay
# balanced - frm's four-cluster split degenerates (one seed produces a 1-unit cluster, median balance
# 0.07) while its three-cluster split is even (129/133/138), and frm+rws is the reverse (four-cluster
# balance 0.48-0.72). The seed is the MEDIAN of five by that balance, so no panel is a best case. The
# angle maximises the clusters' on-screen separation over a grid, because a 3-D structure hides arms
# behind one another at most views - which is why the original is an animation.
# (arms, seed, elev, azim). Elevation is capped at 60: above that a 3-D axes puts its z label
# where the panel title goes.
SEL_VIEW = {"control": (3, 4, 60, 76), "frm": (3, 1, 36, 124), "both": (4, 3, 36, 300)}


def selectivity_slide(name="slide_32_selectivity"):
    """CDDM selectivity configuration: every active unit a point in the top PCs of its response.

    ⚠️ THESE ARE THE 30,000-ITERATION NETWORKS (CDDM_std_g0), not the 200,000-iteration penalty sweep
    slides 28-31 read. By 200k the configuration has collapsed; at 30k it still has its full form,
    and that sweep is also the one carrying the animated_selectivity movies.

    Each panel is its own PCA, so PC1 of one condition has nothing to do with PC1 of another and the
    panels share neither scale nor orientation. Each therefore gets the seed and the angle measured
    for ITSELF (see SEL_VIEW); forcing one angle on all three hid arms in the two it was not fitted
    on.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cache is missing.
    """
    if not os.path.exists(SEL_CACHE):
        print(f"  SKIP {name}: {SEL_CACHE} missing (build it with cddm_selectivity_cache.py)")
        return None
    z = np.load(SEL_CACHE, allow_pickle=True)
    show = [(a_, l_, c_) for a_, l_, c_ in PEN_ARMS if a_ in SEL_VIEW
            and f"{a_}|{SEL_VIEW[a_][1]}|pcs" in z.files]
    if not show:
        print(f"  SKIP {name}: no coordinates")
        return None
    ps.setup()
    fig = plt.figure(figsize=(78 * ps.MM, 150 * ps.MM))
    for i, (arm, lab, col) in enumerate(show):
        k, seed, elev, azim = SEL_VIEW[arm]
        ax = fig.add_subplot(len(show), 1, i + 1, projection="3d")
        P = np.asarray(z[f"{arm}|{seed}|pcs"], float)
        var = np.asarray(z[f"{arm}|{seed}|var"], float)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=5.0, c=col, alpha=0.55, linewidths=0, zorder=3)
        ax.set_title(f"{lab}  ·  seed {seed}  ·  {len(P)} active units  ·  "
                     f"PC variance {', '.join(f'{v:.2f}' for v in var[:3])}",
                     fontsize=6.4, color=ps.INK, pad=0)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
            pane.line.set_color(ps.GRID)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        ax.set_xlabel("PC1", fontsize=6.0, labelpad=-10)
        ax.set_ylabel("PC2", fontsize=6.0, labelpad=-10)
        ax.set_zlabel("PC3", fontsize=6.0, labelpad=-10)
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1.0, 1.0, 0.75), zoom=1.22)
    fig.suptitle("CDDM, $N$ = 1000, 30,000 iterations\n"
                 "every active unit as a point in the top three PCs of its own response;\n"
                 "each panel its own PCA, so no scale or orientation is shared",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, y=1.0)
    fig.subplots_adjust(left=0.02, right=0.98, top=0.92, bottom=0.02, hspace=0.22)
    return ps.save(fig, name)


def readout_line(iters, recorded=None):
    """The 'read at ...' sentence a panel carries, from where its cells were actually read.

    Derived rather than typed, so it cannot drift from the data: a family read at one shared probe
    gives one number, a family whose cells land on different probes gives the range, and a family
    whose surviving data record no iteration says so rather than borrowing the sweep's nominal
    budget silently.

    Args:
        iters: set of iterations the family's cells were read at, possibly empty;
        recorded: the iteration from ARCHIVE_READOUT when the data cannot supply one.
    Returns:
        a one-sentence string, always ending in a full stop.
    """
    if iters:
        lo, hi = min(iters), max(iters)
        return f"Read at {lo:,} iterations." if lo == hi else \
               f"Read at {lo:,}\u2013{hi:,} iterations (the cells do not share a probe)."
    if recorded is not None:
        return f"Read at {recorded:,} iterations (recorded, not in the archived summary)."
    return "⚠️ Read-out iteration not recorded."


INTERVENTIONS = [
    ("slide_x_activation_cddm", "A different activation does not help",
     "CDDM, 200k", "activation", "ReLU (default)", 0,
     "CDDM, N = 1000. Every seed drawn."),
    ("slide_x_weightdecay", "Weight decay makes it monotonically worse",
     "CDDM, 200k", "weight decay", "W.D. 10$^{-6}$ (default)", 1,
     "CDDM, N = 1000. The default is a rung of this ladder, not a separate condition."),
    ("slide_x_activation_ff", "Nor on the other task",
     "3-bit flip-flop, 150k", "activation", "ReLU (default)", 0,
     "3-bit flip-flop, N = 1000. Every seed drawn."),
    # The reference is the DEFAULT DRAW, whose rows sit at norm 0.050 at N = 1000 - the lowest rung
    # of this ladder, not a middle one, which is why ref_at is 0. Labelling it "×1" and putting it
    # second (every version before 2026-10-01) was what made this panel look non-monotone.
    ("slide_x_inputscale", "Scaling the input weights up adds 40\u201375 units of 1000, peaking at row norm 2",
     "3-bit flip-flop, 150k", "input scale", "row norm 0.05 (default draw)", 0,
     "3-bit flip-flop, N = 1000. Every seed drawn.\n"
     "Rungs are the absolute L2 norm of each W$_{inp}$ row at init; the default draw is 0.050."),
    ("slide_x_metabolic", "The field-standard metabolic penalty moves nothing beyond seed scatter",
     "CDDM, 30k (archived)", "metabolic", "$\\lambda$ = 0 (default)", 0,
     "CDDM, N = 1000, four decades of $\\lambda$."),
    ("slide_x_architecture", "Nor does the equation form, nor a trainable bias",
     "CDDM, 30k (archived)", "architecture", "standard", 0,
     "CDDM, N = 1000."),
    ("slide_x_recnoise", "Removing recurrent noise is the largest effect we found — and it is negative",
     "CDDM, 30k", "noise", "$\\sigma$ = 0.05 (default)", 2,
     "CDDM, N = 1000. Re-scored from the trained weights onto the participation rule, so the seeds "
     "this sweep never saved are drawn."),
]


# ---- slide 11's companion: are the units the input scale keeps paid for? -----------------------
#
# R^2 denominator for the 3-bit flip-flop: mean((t - mean t)^2) over the masked target, which is what
# Trainer.r2_score divides by. Measured over six independent 1024-trial batches of the task as the
# runs' own configs instantiate it - 0.7158, 0.7206, 0.7232, 0.7184, 0.7265, 0.7221 - so 0.721 +-
# 0.003. One shared constant rather than each net's own batch variance, which moves R^2 in the 4th
# decimal.
VAR_TARGET_FF = 0.721

# WHY THIS PANEL DOES NOT USE THE FOLDER-NAME SCORE, the way met_cell above does. The input-scale
# reference is the ksweep cell, which has a 500,000-iteration budget while the four rungs stop at
# 150,000 - so its score prefix is the best over a run 3.3x longer than theirs, while the active
# count beside it is read at 150,000. R^2 here is computed instead from `loss_clean_train`, the
# noise-free probe the Trainer records beside the participation vector, at the SAME iteration as the
# count. It reads ~0.964 against the ~0.954 in the folder names because the folder score comes from a
# forward pass with noise on; that offset was measured per rung by re-scoring each cell offline with
# the noise on and off, and it is 0.0047 at all five rungs, so no rung is favoured by the choice.
#
# The clean-loss route is validated end to end against an independent re-score: load each net's
# LastParams into RNN_numpy, run a freshly drawn noise-free batch, take the masked MSE. Across the
# five cells the offline MSE and the trace's own clean loss agree to 0.00026-0.00064, against a
# threshold of 0.005 set before looking. ⚠️ That re-score needs `equation_type` passed explicitly -
# it is NOT in the saved npz and RNN_numpy defaults to "s" while these nets are "h", which scores
# them at MSE 2.08 instead of 0.026. penalty_matched.clean_loss has that bug.
#
# Colour carries the rung, as in the metabolic panel: four validated slots plus BASE for the
# unpenalised reference, and the dose order encoded a second time as a line through the cell means.
INPUTSCALE_LADDER = [
    ("row norm 0.05 (default)", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=*", ps.BASE),
    ("row norm 0.5", f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=0.5_iters=*", ps.SLOTS[0]),
    ("row norm 2",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=2_iters=*",   ps.SLOTS[2]),
    ("row norm 5",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=5_iters=*",   ps.SLOTS[3]),
    ("row norm 20",  f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=20_iters=*",  ps.SLOTS[1]),
]


def winp_cell(pattern, at_iter):
    """Noise-free R^2 and active-unit count, per seed, for one rung of the input-scale ladder.

    Both are read at the same iteration, each on its own probe grid: the participation vector is
    stored every 100 iterations and the clean loss every 10. Diverged runs (`nan_` prefix) are
    dropped, as everywhere else in this project.

    Args:
        pattern: glob matching the rung's run folders; at_iter: int, the iteration to read at.
    Returns:
        (r2, active, probes): r2 and active are (n_seeds,) float arrays, empty where the cell is
        missing; probes is the set of participation-probe iterations actually landed on, which is
        not always `at_iter` - the cells of this ladder have different budgets.
    """
    r2, active, probes = [], [], set()
    for f in sorted(glob.glob(os.path.join(pattern, "*", "*ParticipationTrace.pkl"))):
        if os.path.basename(os.path.dirname(f)).split("_")[0] == "nan":
            continue
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        pit = np.asarray(d["participation_iters"], float)
        j = int(np.argmin(np.abs(pit - at_iter)))
        p = np.asarray(d["participation"], float)[j]
        probes.add(int(pit[j]))
        it = np.asarray(d["iters"], float)
        L = np.asarray(d["metrics"]["loss_clean_train"], float)
        active.append(active_count(p, "scalefree"))
        r2.append(1.0 - L[int(np.argmin(np.abs(it - at_iter)))] / VAR_TARGET_FF)
    return np.array(r2, float), np.array(active, float), probes


def inputscale_r2_slide(name="slide_x_inputscale_r2", at_iter=150_000):
    """Task R^2 against active units for every net of the input-scale ladder, one colour per rung.

    Answers the question the ladder panel raises: its rungs differ by 95 active units of 1000, so
    did the rungs keeping more units pay for them? The y axis is held to a window ten times the
    spread of the data, because an axis zoomed to the data turns 0.002 of R^2 into a visible slope.

    Args:
        name: output file stem; at_iter: the iteration both axes are read at.
    Returns:
        the output path, or None if no cell of the ladder is on disk.
    """
    cells = [(lab, col) + winp_cell(pat, at_iter) for lab, pat, col in INPUTSCALE_LADDER]
    drawn = [c for c in cells if len(c[2])]
    probes = set().union(*[c[4] for c in drawn]) if drawn else set()
    if not drawn:
        print(f"  SKIP {name}: no input-scale cells on disk")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))

    # NO LINE JOINS THE MEANS, as on the metabolic and activation panels. An earlier version
    # drew one to carry the input-scale ordering, since categorical colour cannot. But neither
    # axis here is input scale, so a path between the means implies a trajectory through a plane
    # that has none - and it doubles back, because the ladder is single-peaked in active units
    # (263 -> 302 -> 339 -> 324 -> 306) while the R^2 it would thread spans 0.001. A line that
    # reverses inside the noise reads as noise, not as order. The legend names every rung.
    for lab, col, r2, active, _ in drawn:
        ax.plot(active, r2, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                label=f"{lab}  ({len(r2)})")
        # A SQUARE, not a large translucent circle. metabolic_r2_slide settled this: the mean
        # differs from a seed by SHAPE, not by opacity, because a translucent dot reads as blurred
        # or as less certain and a mean over three seeds is neither. One convention across the
        # scatter panels.
        ax.plot(active.mean(), r2.mean(), "s", ms=7.0, color=col, mec="white", mew=1.1, zorder=4)

    every_r2 = np.concatenate([c[2] for c in drawn])
    every_active = np.concatenate([c[3] for c in drawn])
    mid = 0.5 * (every_r2.min() + every_r2.max())
    ax.set_ylim(mid - 0.01, mid + 0.01)
    ax.set_xlabel("active units of 1000  (scale-free rule, $p \\geq 0.05\\,q_{95}(p)$)")
    ax.set_ylabel("task $R^2$, noise-free probe")
    ax.set_title("3-bit flip-flop, $N$ = 1000: $R^2$ against active units\n"
                 f"3-bit flip-flop, N = 1000. {readout_line(probes)} Every seed "
                 f"drawn.\n{len(every_r2)} networks spanning {every_active.min():.0f}\u2013"
                 f"{every_active.max():.0f} active units sit within "
                 f"{every_r2.max() - every_r2.min():.3f} of $R^2$ = {mid:.3f}. "
                 "Circles are networks, squares the cell means.",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.0, borderpad=0.2,
              borderaxespad=0.3, ncol=2)
    ps.ygrid(ax)
    return ps.save(fig, name)


# ---- slide 19b: two more ways to count dimensions, and they do not agree ------------------------
#
# Slide 19 plots the PARTICIPATION RATIO of the active units' rates, (sum L)^2 / sum L^2 over the
# eigenvalues L of their covariance. It is a soft count dominated by the top of the spectrum: equal
# variance in n directions gives exactly n, one dominant direction gives 1.
#
# Two other summaries of the SAME spectrum answer different questions, and 19b draws them against
# each other:
#   k99        the smallest number of principal components carrying 99% of the variance. A HARD
#              count, and the one sensitive to the tail - it asks how many directions are needed
#              before almost nothing is left.
#   stable rank  sum L / max L, the total variance in units of the single largest direction. The
#              softest of the three; it ignores the shape of the tail entirely.
# ⚠️ THIS PANEL WAS BUILT EXPECTING THEM TO DISAGREE, AND THEY DO NOT. The first version was titled
# "two other ways to count dimensions, and they disagree", on the strength of duplication doubling
# k99 (33 -> 65) while its stable rank sat on the control's (3.06 against 3.05). That dissociation
# does not survive a test: Welch p = 0.99 with n = 3 and fully overlapping ranges, which is "cannot
# tell", not "does not move". Measured properly, the three summaries AGREE - pairwise Pearson r from
# +0.82 to +0.93 over the 22 networks - and all three rank the control lowest and frm+rws highest.
# Dropping the penalty arm, which is extreme on every axis, weakens but does not reverse it
# (k99 vs stable rank r = +0.33, Spearman rho = +0.48).
#
# So the panel earns its place as a ROBUSTNESS CHECK rather than a dissociation: the slide-19 result
# does not depend on the participation ratio's particular weighting of the spectrum. What none of the
# three measures can do is separate the middle four arms - one of the six pairings is significant on
# k99 and none on the other two.
#
# The numbers come from `f2_dimensionality_extra.py` rather than the shared F2 cache, which stores
# `dims` and `dims95` as scalars and keeps no spectrum. That script re-derives every measure from one
# eigendecomposition per network and checks itself against the cache per network, with the agreement
# interval taken from each network's own batch-to-batch spread - 2.6% on control nets but 9-11% on
# duplication nets, whose near-identical unit pairs make the covariance near-degenerate.
DIMS_EXTRA = "data/f2_dimensionality_extra.npz"


def dims_extra_slide(name="slide_f2_srank_vs_pc99"):
    """Stable rank against the 99%-variance PC count, every slide-19 network, one colour per arm.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the companion cache has not been built.
    """
    if not os.path.exists(DIMS_EXTRA):
        print(f"  SKIP {name}: {DIMS_EXTRA} missing - build it with f2_dimensionality_extra.py")
        return None
    d = np.load(DIMS_EXTRA, allow_pickle=True)
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    xs, ys = [], []
    for arm, _, full, col in F2.ARMS:
        m = d["arm"] == arm
        if not m.any():
            continue
        x, y = np.asarray(d["k99"][m], float), np.asarray(d["srank"][m], float)
        xs.append(x), ys.append(y)
        ax.plot(x, y, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                label=f"{full} ({m.sum()})")
        ax.plot(x.mean(), y.mean(), "s", ms=7.0, color=col, mec="white", mew=1.1, zorder=4)
    # LOG x. The counts run 31 to 231, and every claim this panel makes is a RATIO - duplication
    # doubles the 99% count while leaving the stable rank where the control has it, the penalty pair
    # multiplies it by 6.6. On a linear axis the penalty arm sits alone on the right and squeezes the
    # other five into the left quarter, where a doubling is invisible.
    ax.set_xscale("log")
    ax.set_xticks([30, 50, 70, 100, 150, 220])
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlabel("principal components carrying 99% of the variance")
    ax.set_ylabel("stable rank,  $\\sum_i \\lambda_i \\,/\\, \\lambda_1$")
    ax.legend(loc="upper left", fontsize=5.4, handlelength=1.0, borderpad=0.25,
              borderaxespad=0.3, ncol=1)
    ps.ygrid(ax)
    return ps.save(fig, name)


# ---- slides 8b and 9b: are the activation panels' unit counts paid for in performance? ---------
#
# Slides 8 and 9 say a different activation does not raise the active count, and on CDDM two of the
# three LOWER it (272 active for ReLU against 249 for softplus and 205 for sigmoid, of 1000). The
# objection that leaves open is the same one slide 12b closes for the metabolic penalty: maybe the
# arms that shed units shed them because they were failing the task. These panels put R^2 on the
# other axis for the same networks.
#
# ⚠️ THE FOLDER-NAME SCORE CANNOT BE USED HERE, although met_cell uses it and all four CDDM arms
# share a 200,000-iteration budget so it looks comparable. It is a forward pass with the noise ON,
# and that noise penalty is ACTIVATION-DEPENDENT. Measured clean-minus-folder per arm: +0.0626
# (leaky ReLU), +0.0617 (softplus), +0.0738 (sigmoid) - a 0.0121 spread against a between-arm spread
# of only 0.007 in either measure alone. The consequence is not academic: on the folder score sigmoid
# is the WORST arm (0.8690 vs 0.8740 for ReLU) and on the clean score it is among the best. The
# ranking INVERTS with the choice, so the folder score is not a shared currency across activations.
# A bounded sigmoid saturates and a softplus has a nonzero floor; they do not absorb injected noise
# the way a ReLU net does. (For the input-scale ladder above, every arm IS a ReLU net, which is why
# a single uniform 0.0047 offset was enough there.)
#
# So R^2 is the NOISE-FREE score throughout. On the flip-flop every arm carries `loss_clean_train`
# in its trace and winp_cell already reads it. On CDDM the three activation arms carry it but the
# ReLU reference does not - CDDM_std_g0_drift predates that probe - so the reference is re-scored
# OFFLINE, by rebuilding each net from its saved parameters and running a noise-free batch. The same
# re-score is applied to all four arms, so no arm reaches the axis by a different route.
#
# THE OFFLINE RE-SCORE IS VALIDATED against the recorded probe on the three arms that have both, and
# the criterion comes from the measured scatter rather than being picked: `loss_clean_train` is
# noise-free in its forward pass but the PARAMETERS are still moving at 200k (slide 4's own point),
# so consecutive probes score different networks - per-probe sd 0.0035-0.0062 in R^2 over the last
# 10,000 iterations, with single-probe sigmoid excursions down to 0.465. One offline sample must
# therefore land inside the [min, max] the recorded probe actually spans over that window. All nine
# nets pass. On the flip-flop the same probe is far quieter (sd 0.0004-0.0033 near 150k, single probe
# within 0.001 of the window mean), which is why winp_cell's single-probe read is sound there.
#
# ⚠️ RNN_numpy's softplus is `log(1 + exp(beta*slope*x))/beta`, which OVERFLOWS np.exp and returns
# NaN for the whole run once beta*x > ~709 - at beta = 25 that is any unit above x ~ 28. torch's
# Softplus does not, because it switches to the linear branch above threshold=20. The instance is
# patched below with the stable form. The class itself is left alone, but the bug is live for anyone
# re-scoring the new flip-flop softplus nets offline.
ACTIVATION_CDDM = [
    ("ReLU (default)", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000", ps.BASE),
    ("leaky ReLU, leak 0.01", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=leakyrelu_iters=200000", ps.SLOTS[0]),
    ("softplus, $\\beta$ = 25", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=softplus25_iters=200000", ps.SLOTS[2]),
    ("sigmoid, 7.5(x$-$0.3)", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=sigmoid_iters=200000", ps.SLOTS[3]),
]

# The weight-decay ladder, in DOSE ORDER rather than with the default first. 1e-6 is the default in
# configs/trainer/trainer.yaml, so the family's reference cell IS a rung of this ladder - the same
# point F1's own table makes - and the ladder only reads as a dose-response when it sits in its
# place between 0 and 1e-5 rather than being drawn as a separate condition.
WEIGHTDECAY_CDDM = [
    ("W.D. 0", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=0_iters=200000", ps.SLOTS[2]),
    ("W.D. 10$^{-6}$ (default)", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000", ps.BASE),
    ("W.D. 10$^{-5}$", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-5_iters=200000",
     ps.SLOTS[1]),
    ("W.D. 10$^{-4}$", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-4_iters=200000",
     ps.SLOTS[3]),
]

ACTIVATION_FF = [
    ("ReLU (default)", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=*", ps.BASE),
    ("leaky ReLU, leak 0.01", f"{DATA_DIR}/NBitFlipFlop_std_activations/EqType=h_k=3_N=1000_act=leakyrelu_iters=150000", ps.SLOTS[0]),
    ("softplus, $\\beta$ = 25", f"{DATA_DIR}/NBitFlipFlop_std_activations/EqType=h_k=3_N=1000_act=softplus25_iters=150000", ps.SLOTS[2]),
    ("sigmoid, 7.5(x$-$0.3)", f"{DATA_DIR}/NBitFlipFlop_std_sigmoid/EqType=h_k=3_N=1000_iters=150000", ps.SLOTS[3]),
]


def cddm_batch_and_mask(folder):
    """The CDDM batch, scoring mask and R^2 denominator, built from one run's own config.

    CDDM is deterministic - the batch enumerates every coherence pair - so one call is the whole
    task and every net is scored on identical input.

    Args:
        folder: a net folder holding `*_config.yaml`.
    Returns:
        (inputs, target, mask, var): inputs (n_inp, T, B), target (n_out, T, B), mask a timepoint
        index array, var the masked target variance that R^2 divides by.
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, target, _ = task.get_batch()
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    tm = np.asarray(target, float)[:, mask, :]
    return (np.asarray(inputs, float), np.asarray(target, float), mask,
            float(np.mean((tm - tm.mean()) ** 2)))


def offline_clean_r2(folder, inputs, target, mask, var):
    """Noise-free R^2 of one net's final parameters, rebuilt and simulated offline.

    Args:
        folder: net folder holding `*LastParams*.npz` and `*_config.yaml`;
        inputs: (n_inp, T, B) input batch; target: (n_out, T, B); mask: scoring timepoint indices;
        var: the R^2 denominator.
    Returns:
        float R^2 = 1 - masked MSE / var.
    """
    d = np.load(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0], allow_pickle=True)
    p = {k: d[k] for k in d.files}
    # The activation and the equation form both come from the CONFIG, for every arm alike. The npz
    # cannot supply either: the drift sweep predates storable_ and stored activation_args as the
    # dict's KEYS, and equation_type was never saved at all while RNN_numpy defaults it to "s" -
    # these nets are "h", and simulating the wrong one scores a good net near zero.
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    p["activation_args"] = OmegaConf.to_container(cfg.model.activation_args, resolve=True)
    rnn = RNN_numpy(**filter_kwargs(RNN_numpy, p), equation_type=str(cfg.model.equation_type), seed=0)
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    o = rnn.get_output()
    return 1.0 - float(((o[:, mask, :] - target[:, mask, :]) ** 2).mean()) / var


def cddm_activation_cell(pattern, at_iter):
    """Noise-free R^2 and active-unit count, per seed, for one CDDM activation arm.

    R^2 is the offline re-score of each net's final parameters; the count is the scale-free rule at
    the participation probe nearest `at_iter`. Diverged runs (`nan_` prefix) are dropped.

    Args:
        pattern: glob matching the arm's run folders; at_iter: iteration to read the count at.
    Returns:
        (r2, active, probes): r2 and active are (n_seeds,) float arrays, empty where the cell is
        missing; probes is the set of participation-probe iterations actually landed on, which need
        not be `at_iter` - the panel's read-out sentence is derived from it rather than typed.
    """
    folders = [f for f in sorted(glob.glob(os.path.join(pattern, "*/")))
               if os.path.basename(f.rstrip("/")).split("_")[0] != "nan"]
    if not folders:
        return np.array([]), np.array([]), set()
    inputs, target, mask, var = cddm_batch_and_mask(folders[0])
    r2, active, probes = [], [], set()
    for folder in folders:
        tf = glob.glob(os.path.join(folder, "*ParticipationTrace.pkl"))
        if not tf:
            continue
        with open(tf[0], "rb") as fh:
            d = pickle.load(fh)
        pit = np.asarray(d["participation_iters"], float)
        j = int(np.argmin(np.abs(pit - at_iter)))
        p = np.asarray(d["participation"], float)[j]
        probes.add(int(pit[j]))
        active.append(active_count(p, "scalefree"))
        r2.append(offline_clean_r2(folder, inputs, target, mask, var))
    return np.array(r2, float), np.array(active, float), probes


def activation_r2_slide(name, ladder, cell_fn, at_iter, task_line, n_units=1000,
                        headline=None):
    """Task R^2 against active units for every net of one activation family, one colour per arm.

    Args:
        name: output file stem; ladder: list of (label, glob, colour);
        cell_fn: callable(pattern, at_iter) -> (r2, active, probes) per seed;
        at_iter: the iteration the count is read at; task_line: the subtitle naming the task and the
            conditions, WITHOUT a read-out iteration - that sentence is derived from the probes the
            cells actually landed on, so it cannot drift from the data;
        n_units: network size, for the x axis label;
        headline: the claim the panel makes, or None for the activation family's. It is a parameter
            because this function draws more than one family now, and the activation headline
            printed over a weight-decay panel is a caption describing a different figure.
    Returns:
        the output path, or None if no cell of the family is on disk.
    """
    cells = [(lab, col) + cell_fn(pat, at_iter) for lab, pat, col in ladder]
    drawn = [c for c in cells if len(c[2])]
    if not drawn:
        print(f"  SKIP {name}: no cells for this activation family")
        return None
    missing = [c[0] for c in cells if not len(c[2])]
    probes = set().union(*[c[4] for c in drawn])
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))
    for lab, col, r2, active, _ in drawn:
        ax.plot(active, r2, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                label=f"{lab}  ({len(r2)})")
        # A SQUARE, not a large translucent circle. metabolic_r2_slide settled this: the mean
        # differs from a seed by SHAPE, not by opacity, because a translucent dot reads as blurred
        # or as less certain and a mean over three seeds is neither. One convention across the
        # scatter panels.
        ax.plot(active.mean(), r2.mean(), "s", ms=7.0, color=col, mec="white", mew=1.1, zorder=4)
    every_r2 = np.concatenate([c[2] for c in drawn])
    every_active = np.concatenate([c[3] for c in drawn])
    # No connecting line here, unlike the dose panels: activation is CATEGORICAL, with no order for
    # a line to encode.
    #
    # The y window is several times the data spread, so a hair's-breadth range cannot read as a
    # slope - but it is CLAMPED AT R^2 = 1, which a pure multiple of the spread is not: the CDDM
    # arms span 0.016, and 10x that centred on 0.95 would run the axis up to 1.03 and spend a third
    # of the panel on scores no network can reach.
    mid = 0.5 * (every_r2.min() + every_r2.max())
    half = max(3.0 * (every_r2.max() - every_r2.min()), 0.025)
    lo, hi = mid - half, min(mid + half, 1.0)
    ax.set_ylim(min(lo, hi - 2 * half), hi)
    ax.set_xlabel(f"active units of {n_units}  (scale-free rule, $p \\geq 0.05\\,q_{{95}}(p)$)")
    ax.set_ylabel("task $R^2$, noise-free")
    note = ("\n" + "; ".join(missing) + (" is" if len(missing) == 1 else " are")
            + " still training") if missing else ""
    ax.set_title(f"{headline or '$R^2$ against active units'}\n"
                 f"{task_line} {readout_line(probes)}\n"
                 f"{len(every_r2)} networks spanning {every_active.min():.0f}–"
                 f"{every_active.max():.0f} active units sit within "
                 f"{every_r2.max() - every_r2.min():.3f} of $R^2$ = {mid:.3f}. "
                 f"Circles are networks, squares the cell means.{note}",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.0, borderpad=0.2,
              borderaxespad=0.3, ncol=2)
    ps.ygrid(ax)
    return ps.save(fig, name)


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
# baseline is not a series. Categorical hues carry no order, and the lambda ordering is left to the
# legend rather than drawn: see metabolic_r2_slide for why a line through the means was removed.
MET_CELL = f"{DATA_DIR}/CDDM_std_g0_metabolic/EqType=h_N=1000_LmbdMet={{lam}}"
MET_R2_CACHE = f"{DATA_DIR}/metabolic_clean_r2.npz"
MET_LADDER = [
    ("$\\lambda$ = 0 (no penalty)", f"{DATA_DIR}/CDDM_std_g0/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
     ps.BASE, "0"),
    ("$\\lambda$ = 0.01", MET_CELL.format(lam="0.01"), ps.SLOTS[0], "0.01"),
    ("$\\lambda$ = 0.1", MET_CELL.format(lam="0.1"), ps.SLOTS[1], "0.1"),
    ("$\\lambda$ = 1", MET_CELL.format(lam="1.0"), ps.SLOTS[2], "1.0"),
    ("$\\lambda$ = 10", MET_CELL.format(lam="10.0"), ps.SLOTS[3], "10.0"),
]


def met_cell(cell_dir, lam):
    """Noise-free task r2 and active-unit count, per seed, for one cell of the metabolic sweep.

    ⚠️ R2 IS THE NOISE-FREE PROBE, NOT THE FOLDER'S SCORE PREFIX. That prefix is
    `get_validation_score(...)` run at sigma_rec = sigma_inp = 0.05, i.e. r2_noisy. It is the wrong
    probe for a rate penalty: the penalty shrinks the rate scale ~6x while the injected noise stays
    at a fixed 0.05, so signal-to-noise falls with lambda for reasons unrelated to the task. Using
    it understated lambda = 10 by less than the truth and inflated the apparent tie between the
    other rungs. The clean values come from `metabolic_clean_r2.py`, which MUST run in a worktree
    pinned at 223c550f - RNN_torch's constructor has changed since, including the default of
    `self_connections`. Slide 8b hit the same defect independently on the activation arms.

    The active count still comes from the saved trace, whose final row the Trainer already wrote
    from a w_noise=False pass. Diverged runs (`nan_` prefix) are dropped, as everywhere else.

    This cannot go through `F1.traces_of`: that loader keys on `participation_iters`, and the
    2026-07-28 metabolic traces store their probe iterations under `iters`, so it returns nothing
    for this sweep. Reading `participation[-1]` is the last probe either way.

    Args:
        cell_dir: path to one cell directory, holding one run folder per seed;
        lam: the cell's lambda, as the cache spells it, used to pull its cached clean r2.
    Returns:
        (r2, active, iteration): r2 and active are (n_seeds,) float arrays, empty where the cell is
        missing; iteration is the last probe the cell was read at, or None if it is empty.
    """
    cache = np.load(MET_R2_CACHE)
    clean = cache["clean"][cache["lam"] == lam]
    active, last = [], None
    for f in sorted(glob.glob(os.path.join(cell_dir, "*", "*ParticipationTrace.pkl"))):
        head = os.path.basename(os.path.dirname(f)).split("_")[0]
        if head == "nan":
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        active.append(active_count(np.asarray(d["participation"], float)[-1], "scalefree"))
        last = int(np.asarray(d["iters"])[-1])
    assert len(clean) == len(active), f"{cell_dir}: {len(clean)} cached r2 vs {len(active)} nets"
    return np.asarray(clean, float), np.array(active, float), last


def metabolic_r2_slide(name="slide_x_metabolic_r2"):
    """Task r2 against active units for every net of the metabolic ladder, one colour per lambda.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if no cell of the ladder is on disk.
    """
    cells = [(lab, col) + met_cell(pat, lam) for lab, pat, col, lam in MET_LADDER]
    drawn = [c for c in cells if len(c[2])]
    if not drawn:
        print(f"  SKIP {name}: no metabolic cells on disk")
        return None
    probes = {c[4] for c in drawn}
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, H))

    # The mean differs from a seed by SHAPE, not by opacity. A large translucent dot reads as
    # blurred or as less certain, and a mean over 3-5 seeds is neither.
    #
    # NO LINE JOINS THE MEANS. An earlier version drew one to carry the lambda ordering, since
    # categorical colour cannot. But neither axis here is lambda, so a path between the means
    # implies a trajectory through a plane that has none - and it doubles back, which reads as
    # noise rather than as order. The legend names every lambda; that is enough.
    for lab, col, r2, active, _ in drawn:
        ax.plot(active, r2, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                label=f"{lab}  ({len(r2)})")
        ax.plot(active.mean(), r2.mean(), "s", ms=7.0, color=col, mec="white", mew=1.1, zorder=4)

    ax.set_xlabel("active units of 1000  (scale-free rule, $p \\geq 0.05\\,q_{95}(p)$)")
    ax.set_ylabel("task $r^2$, noise-free probe")
    every_r2 = np.concatenate([c[2] for c in drawn])
    probe = f"{sorted(probes)[0]:,}" if len(probes) == 1 else "the last probe"
    # The axis runs to r2 = 1 so the reader can see how close to perfect these all are; cropping
    # to the data would turn a 0.07 spread into the whole canvas.
    ax.set_ylim(0.88, 1.00)
    lo = np.mean([c[2] for c in drawn if c[0].endswith("= 10")][0])
    ref = np.mean([c[2] for c in drawn if "no penalty" in c[0]][0])
    ax.set_title("CDDM, $N$ = 1000: r$^2$ against active units\n"
                 f"read at 30,000 iterations (last probe {probe}). "
                 "Circles are networks, squares the cell means.",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ax.legend(loc="lower right", fontsize=5.8, handlelength=1.0, borderpad=0.2,
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


# ---- synaptic noise: the sigma_w ladder ---------------------------------------------------------
# Read from the Figure 2 cache rather than from the folder names, because the folder name is each
# net's score in ITS OWN noise and the claim here is the difference between that and a common
# condition. The rungs are DISCOVERED from the cache (any cell whose name carries sw=...), so a new
# rung joins this slide by finishing, not by being remembered.
SW_RE = re.compile(r"sw=([0-9.]+)")


def _synnoise_ladder(task="NBitFlipFlop", n_units=1000):
    """The sigma_w ladder at one task and size, control first.

    Args:
        task: task name as the cache records it; n_units: network size to restrict to.
    Returns:
        (rungs, keys) where rungs is [(label, {field: per-seed array}), ...] ordered
        sigma_w = 0 first then ascending, and keys is the list of fields carried. Empty list if the
        cache holds no synaptic-noise cell at this task and size.
    """
    c = F2.load()
    keys = ["n_active", "r2", "r2_common", "r2_clean", "dims"]
    sel = (c["task"] == task) & (c["N"] == n_units)
    def grab(mask):
        return {k: np.asarray(c[k][mask], float) for k in keys}
    sw = {}
    for i in np.flatnonzero(sel & (c["arm"] == "synnoise")):
        m = SW_RE.search(str(c["cell"][i]))
        if m:
            sw.setdefault(float(m.group(1)), []).append(i)
    if not sw:
        return [], keys
    ctrl = sel & (c["arm"] == "control")
    rungs = [("0", grab(ctrl))]
    for v in sorted(sw):
        idx = np.zeros(len(c["cell"]), bool)
        idx[sw[v]] = True
        rungs.append((f"{v:g}", grab(idx)))
    return rungs, keys


def synnoise_ladder_slide(name="slide_26_synnoise_ladder", task="NBitFlipFlop", n_units=1000):
    """One claim: the ladder buys units all the way up, and past sigma_w = 1 the net keeps them only
    while its synapses are still jittering.

    TWO R-SQUARED SERIES, NOT ONE. `r2` is each net scored in the condition it trained in, synaptic
    noise included -- flat at 0.92-0.95 across the whole ladder, which is why the sweep looked free
    from the folder names alone. `r2_common` is the project's single comparison read-out: sigma_w =
    0 with the recurrent and input noise every arm shares, nine draws averaged. The two agree up to
    sigma_w = 1 and separate above it, so the gap between the curves IS the dependence on the noise.

    Active units are counted on the noise-free pass under the scale-free rule, so the extra units
    are not units lit up by the injected noise at scoring time.

    Args:
        name: output file stem; task: task name as the cache records it; n_units: network size.
    Returns:
        the output path, or None if the cache holds no synaptic-noise cell at this task and size.
    """
    rungs, _ = _synnoise_ladder(task, n_units)
    if not rungs:
        print(f"  SKIP {name}: no synnoise cells at {task}, N = {n_units}")
        return None
    col = ps.COND_COL["synnoise"]
    xs = np.arange(len(rungs), dtype=float)
    cols = [ps.BASE] + [col] * (len(rungs) - 1)

    ps.setup()
    fig = plt.figure(figsize=(W, 92 * ps.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 1.0], hspace=0.17,
                  left=0.13, right=0.80, top=0.86, bottom=0.11)
    ax_u = fig.add_subplot(gs[0])
    ax_r = fig.add_subplot(gs[1], sharex=ax_u)

    # ---- (a) active units ----------------------------------------------------------------------
    ctrl_a = rungs[0][1]["n_active"]
    ax_u.axhspan(ctrl_a.mean() - ctrl_a.std(ddof=1), ctrl_a.mean() + ctrl_a.std(ddof=1),
                 color=ps.BASE, alpha=0.13, lw=0, zorder=1)
    act = ps.strip(ax_u, xs, [r[1]["n_active"] for r in rungs], cols,
                   rng=np.random.default_rng(0))
    ax_u.plot(xs, [a[0] for a in act], "-", lw=1.0, color=col, alpha=0.55, zorder=2)
    ax_u.set_ylabel(f"active units of {n_units}")
    ax_u.text(xs[-1], ctrl_a.mean(), f"  no synaptic noise, {ctrl_a.mean():.0f}", fontsize=6.4,
              color=ps.BASE, va="center", ha="left")
    ax_u.tick_params(labelbottom=False)
    ps.ygrid(ax_u)
    ps.despine(ax_u)

    # ---- (b) the two read-outs -----------------------------------------------------------------
    series = [("r2_common", f"scored at $\\sigma_w$ = 0\n(every arm's condition)", col, "o"),
              ("r2", "scored in the noise\nit trained in", ps.MUTED, "s")]
    means = {}
    for key, _, c_, mk in series:
        for x, (_, v) in zip(xs, rungs):
            g = v[key]
            ax_r.plot(x + np.random.default_rng(1).normal(0, 0.055, len(g)), g, mk, ms=2.8,
                      color=c_, alpha=0.8, mec="none", zorder=3, clip_on=False)
        means[key] = np.array([v[key].mean() for _, v in rungs])
        ax_r.plot(xs, means[key], "-", lw=1.3, color=c_, zorder=4)
    # the gap IS the claim, so it is filled rather than left for the eye to measure
    ax_r.fill_between(xs, means["r2"], means["r2_common"], color=col, alpha=0.14, lw=0, zorder=2)
    ax_r.set_ylabel("$R^2$ on a held-out batch")
    ax_r.set_xlabel("relative synaptic noise  $\\sigma_w$")
    ax_r.set_xticks(xs)
    ax_r.set_xticklabels([l for l, _ in rungs])
    ps.ygrid(ax_r)
    ps.despine(ax_r)
    _right_labels(ax_r, [(means[k][-1], lab, c_) for k, lab, c_, _ in series], gap=0.12)

    gap = means["r2"] - means["r2_common"]
    top = rungs[int(np.argmax([r[1]["n_active"].mean() for r in rungs]))]
    opens = [l for (l, _), gg in zip(rungs, gap) if gg > 0.05]
    fig.suptitle(
        f"3-bit flip-flop, $N$ = {n_units}, "
        f"{min(len(r[1]['n_active']) for r in rungs)}-{max(len(r[1]['n_active']) for r in rungs)}"
        " seeds per rung, every seed drawn\n"
        f"active units {ctrl_a.mean():.0f} at $\\sigma_w$ = 0 rising to "
        f"{top[1]['n_active'].mean():.0f} at $\\sigma_w$ = {top[0]}\n"
        f"$R^2$ at $\\sigma_w$ = 0 falls {means['r2_common'][0]:.3f} to "
        f"{means['r2_common'][-1]:.3f}; each net's score in its own noise holds "
        f"{means['r2'].min():.3f}-{means['r2'].max():.3f}",
        fontsize=7.6, color=ps.INK, linespacing=1.4, y=1.055)
    fig.text(0.5, 0.012,
             "Active units counted on the noise-free pass, scale-free rule "
             "$p_i \\geq 0.05\\,q_{95}(p)$. "
             + (f"The two $R^2$ curves separate by more than 0.05 at $\\sigma_w$ = "
                + ", ".join(opens) + "." if opens else "The two curves never separate."),
             ha="center", va="top", fontsize=6.4, color=ps.MUTED)
    return ps.save(fig, name, w_mm=110)


def main(list_only=False):
    """Write every slide. Returns the list of output paths."""
    if list_only:
        for stem, title, *_ in INTERVENTIONS:
            print(f"  {stem:28s} {title}")
        print(f"  {'slide_x_inputscale_r2':28s} "
              "No rung of the ladder trades performance for live units")
        print(f"  {'slide_x_metabolic_r2':28s} "
              "No rung breaks the task: r2 holds at 0.83-0.88 across four decades of lambda")
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
    # The deck draws r2 AGAINST active units rather than r2 on its own: the two categorical panels
    # ask one question together, and the scatter answers it in one picture. It carries every r2
    # value panel_c carries, with the per-arm cost in its legend. panel_c itself is untouched - the
    # manuscript figure still uses it, where it sits beside (b) and (d) in a row of three.
    out.append(panel_slide("slide_f2_r2_vs_active", F2.panel_r2_vs_active, c=at_1000))
    out.append(panel_slide("slide_f2_dims", F2.panel_d, c=at_1000))
    got = dims_extra_slide()
    if got:
        out.append(got)
    out.append(panel_slide("slide_f2_weights", F2.panel_e, c=at_1000))
    out.append(panel_slide("slide_f2_weight_shape", F2.panel_weight_shape, c=at_1000))
    got = weights_by_task(everything)
    if got:
        out.append(got)
    out.append(panel_slide("slide_f2_size_active", size_active_titled, c=everything))
    out.append(panel_slide("slide_f2_size_r2", size_r2_with_legend, c=everything))

    # ---- one figure per failed intervention ---------------------------------------------------
    fams = collect()
    for stem, title, fam_title, group, ref_label, ref_at, note in INTERVENTIONS:
        got = fams.get((fam_title, group))
        if got is None:
            print(f"  SKIP {stem}: no cells for ({fam_title}, {group})")
            continue
        ref, entries, iters = got
        # by what the data ARE, not by which family: the noise sweep now supplies per-seed
        # arrays and draws like the rest, and falls back to the interval only if the
        # re-score is missing
        per_seed = isinstance(ref, np.ndarray)
        draw = dose_slide if per_seed else summary_slide
        # the read-out goes on its own line, last, on every panel of this block
        full_note = f"{note}\n{readout_line(iters, ARCHIVE_READOUT.get(fam_title))}"
        out.append(draw(stem, title, ref_label, entries, ref,
                        F1.GROUP_COL.get(group, ps.MUTED), note=full_note, ref_at=ref_at))

    for _t in PEN_TASKS:
        got = penalty_size_slide(_t)
        if got:
            out.append(got)
    for _fn in (temporal_pr_slide, frm_vs_both_slide, selectivity_slide):
        got = _fn()
        if got:
            out.append(got)
    for _fn in (dropout_selection_slide, dropout_targeting_slide, dropout_dose_slide,
                dropout_kinds_slide, dropout_rate_units_slide, dropout_rate_cost_slide,
                dropout_along_training_slide):
        got = _fn()
        if got:
            out.append(got)
    got = control_trajectory_slide()
    if got:
        out.append(got)
    got = activation_r2_slide(
        "slide_x_activation_cddm_r2", ACTIVATION_CDDM, cddm_activation_cell, 199_900,
        "CDDM, N = 1000. $R^2$ is the noise-free offline re-score. Every seed drawn.")
    if got:
        out.append(got)
    got = activation_r2_slide(
        "slide_x_weightdecay_r2", WEIGHTDECAY_CDDM, cddm_activation_cell, 199_900,
        "CDDM, N = 1000. $R^2$ is the noise-free re-score of each net's final parameters. "
        "Every seed drawn.",
        headline="CDDM, $N$ = 1000: $R^2$ against active units")
    if got:
        out.append(got)
    got = activation_r2_slide(
        "slide_x_activation_ff_r2", ACTIVATION_FF, winp_cell, 150_000,
        "3-bit flip-flop, N = 1000. Every seed drawn.")
    if got:
        out.append(got)
    got = inputscale_r2_slide()
    if got:
        out.append(got)
    got = metabolic_r2_slide()
    if got:
        out.append(got)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="name the slides without drawing them")
    args = ap.parse_args()
    main(list_only=args.list)
