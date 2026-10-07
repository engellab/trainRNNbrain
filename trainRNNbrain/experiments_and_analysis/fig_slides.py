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
import heldout_r2 as heldout
import run_dims as RD
import fig_paper_F1 as F1
import fig_knobs_at_floor as KF
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


def summary_slide(name, title, ref_label, entries, ref, col, note=None, ref_at=0, n_total=1000):
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
    ax.set_xticklabels(say_knobs([r[0] for r in rows]), fontsize=6.4, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(rows) - 0.4)
    ax.set_ylabel(f"active units of {n_total:,}")
    # the same zero-to-N axis `dose_slide` uses, and for the same reason: these panels are read
    # against headings that say how far a knob moves the count
    ax.set_ylim(0, n_total * 1.08)
    ax.axhline(n_total, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
    ax.annotate(f"all {n_total:,}", (ax.get_xlim()[0], n_total), textcoords="offset points",
                xytext=(4, 3), fontsize=6.0, color=ps.MUTED)
    ax.set_title(title + ("\n" + note if note else ""), fontsize=7.4, color=ps.INK,
                 linespacing=1.35, pad=6)
    ps.ygrid(ax)
    return ps.save(fig, name)


def dose_slide(name, title, ref_label, entries, ref, col, ylabel="active units of 1000", note=None,
               ref_at=0, n_total=1000):
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
    ax.set_xticklabels(say_knobs(labels), fontsize=6.4, rotation=22, ha="right",
                       rotation_mode="anchor")
    ax.set_xlim(-0.6, len(groups) - 0.4)
    ax.set_ylabel(ylabel)
    # ⚠️ THE AXIS RUNS FROM ZERO TO N, AND IT HAS TO. These panels are read against headings that
    # say how much a knob moves the count, and an axis fitted to the data makes every knob look
    # decisive: the activation ladder spans 205 to 278 of 1000 and on a fitted axis sigmoid falls
    # off a cliff while the slide's heading says all four land in the same place. A newcomer
    # reading the deck said exactly that. On a 0-to-N axis the flat ladders are flat, weight decay's
    # collapse from 395 to 84 is still unmistakable, and the four panels of the section can be
    # compared with each other, which a per-panel axis makes impossible.
    ax.set_ylim(0, n_total * 1.08)
    ax.axhline(n_total, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
    ax.annotate(f"all {n_total:,}", (ax.get_xlim()[0], n_total), textcoords="offset points",
                xytext=(4, 3), fontsize=6.0, color=ps.MUTED)
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
# task and the budget once for all seven panels. A slide has no caption, so the title has to - but
# only with the task, the sizes and the read-out. The matched-hyperparameter sentence this line used
# to carry was 20 words of figure text that no listener reads off a projected slide; it is printed
# to stdout by `print_size_provenance` instead, so it stays auditable.
SIZE_LINE = "3-bit flip-flop, N = 500\u20134000, read at 40,000 iterations"
SIZE_MATCHED = ("matched across arms: gamma = 0, lr 1e-3, weight decay 1e-6, "
                "sigma_rec = sigma_inp = 0.05")

# The arm names Figure 2 uses internally, in the words this deck says out loud. `frm` and `rws` are
# the two penalties' code names and appear on the slides as what they do (slides 30 and 32 use the
# same two phrases); every other arm is already plain.
ARM_SAYS = {"none": "untouched", "duplicate": "duplication", "rescale": "rescaling",
            "syn. noise": "synaptic noise", "frm + rws": "rate + sparsity",
            "no intervention": "untouched", "dropout: mute": "dropout",
            "prune + duplicate": "duplication", "penalty: frm + rws": "rate + sparsity"}


# ---- slide-only colour: an ordered quantity is not a set of conditions --------------------------
# `fig_paper_F1.GROUP_COL` gives each knob family a categorical slot, which is right for Figure 1d -
# there the families sit side by side and have to be told apart - and wrong here, where each family
# gets its own slide showing ONE knob at several doses. The dose order is already on the x axis, so
# a ladder needs no hue at all; and every slot it could spend is spoken for elsewhere in this deck,
# where ps.COND_COL makes purple mean duplication, gold the sparsity penalty and green the rate +
# sparsity arm. Drawing the activation ladder in duplication purple and the weight-decay ladder in
# sparsity gold made one colour name two unrelated things, and green named three (recurrent noise,
# input scale, and the rate + sparsity arm on every what-works panel). So every knob ladder uses ONE
# accent that is in no COND_COL entry, and ps.BASE stays on the reference rung.
#
# The accent is a magenta the validated palette does not use. Checked against paperstyle's own
# requirement at the PAPER surface (#fcfcfb): contrast 6.0:1, twice the 3:1 floor, and dE76 at least
# 53 from every categorical slot and tint and 63 from ps.BASE, so it reads as neither a condition
# nor the reference.
KNOB_ACCENT = "#b02a73"

# The second accent, for the one panel drawing two series that are neither conditions nor doses:
# slide_x_metabolic_ruler's two ways of counting the same networks. Teal, contrast 5.8:1 at PAPER
# and dE76 78 from KNOB_ACCENT, so the pair stays apart at projector distance.
COUNT_ALT = "#0f6d7d"

# NETWORK SIZE IS ORDERED TOO, so it gets one hue in four values rather than four slots - and one of
# those slots was ps.BASE, which paperstyle reserves for the reference. This is the accent's own hue
# at L* 56, 45, 34, 24: dark means big, and the lightest value still clears the contrast floor at
# PAPER (3.5:1), which it has to - these curves are drawn at alpha 0.5.
SIZE_RAMP = ("#db5599", "#ba357c", "#9b0b61", "#7e0049")


# The knob ladders' rung names come from `fig_paper_F1`'s family tables, which are the MANUSCRIPT's
# labels and stay as they are: a printed page is read at reading distance, where "W.D. 10$^{-6}$" is
# legible. A projected slide is not. Worse, matplotlib renders that exponent's minus as a hair-thin
# superscript hyphen that vanishes on a projector, so the panel reads "ten to the five" and inverts
# the ladder it is drawing. Here the rungs are said as doses relative to the default, which needs no
# exponent at all. A label this map does not know passes through unchanged.
KNOB_SAYS = {
    "W.D. 0": "none",
    "W.D. 10\u207b\u2076*": "the default",
    "W.D. 10$^{-6}$ (default)": "the default",
    "W.D. 10\u207b\u2075": r"10$\times$ the default",
    "W.D. 10\u207b\u2074": r"100$\times$ the default",
}


def say_knobs(labels):
    """The knob ladders' rung names, in the words this deck says out loud.

    Args:
        labels: list of rung labels as the family table spells them.
    Returns:
        a new list, each label replaced by its plain form where one is known.
    """
    return [KNOB_SAYS.get(l, l) for l in labels]


def say_arms(ax, **kw):
    """Redraw an axes' legend with the arms' plain-English names.

    Figure 2's own labels are the cache's arm keys, which is right for a manuscript whose caption
    defines them and wrong for a slide: "frm + rws" names nothing to someone seeing it once. The
    mapping is `ARM_SAYS`; a label it does not know is passed through unchanged, and any trailing
    detail the panel appended to a label (a seed count, a percentage) is kept.

    Args:
        ax: the axes whose legend is rebuilt; kw: passed to ax.legend (loc, fontsize, ncol, ...).
    Returns:
        the new Legend, or None if the axes has no labelled artists.
    """
    handles, labels = ax.get_legend_handles_labels()
    if not handles:
        return None
    out = []
    for lab in labels:
        head = lab.split("  ")[0].strip()
        rest = lab[len(head):]
        for key, says in ARM_SAYS.items():
            if head == key or head.startswith(f"{key} ("):
                head = head.replace(key, says, 1)
                break
        out.append(head + rest)
    return ax.legend(handles, out, **kw)


def score_scale_key(ax, label="whole\nscore", min_slab=0.016):
    """Put a 0-to-1 reference bar beside an axes whose y window is a sliver of the score range.

    Held-out r2 panels are windowed to the arms' own spread - 0.915 to 0.947 on the size panel - so
    three percent of the score fills the frame and the eye reads a steep cost where the text says
    the cost is small. Rescaling the axis to 0-1 would hide the differences instead. This draws the
    whole 0-to-1 range as a bar outside the panel with the plotted window marked on it, so the
    reader sees how much of the score the panel is showing without the data being distorted.

    TWO FIXES OVER THE FIRST VERSION, both of which stopped it being read at all from a seat. It
    was 5.5% of the panel wide - about two millimetres printed - and its name ran up the side at
    5.8 pt rotated 90 degrees. It is now 13% wide, its name is horizontal above it, and two guide
    lines run from the panel's own top and bottom corners to the ends of the marked window, so the
    relation "the whole panel is this slab" is drawn rather than inferred.

    Args:
        ax: the axes, already given its final ylim; label: the two or three words naming the bar,
            newlines included where it should wrap; min_slab: the smallest fraction of the bar the
            marked window is drawn as, so a 3%-of-score window stays visible as a thick line.
    Returns:
        the inset axes holding the bar.
    """
    y0, y1 = ax.get_ylim()
    lo, hi = max(y0, 0.0), min(y1, 1.0)
    if hi - lo < min_slab:                      # keep it visible; the guide lines carry the zoom
        mid = 0.5 * (lo + hi)
        lo, hi = mid - min_slab / 2, mid + min_slab / 2
    key = ax.inset_axes([1.02, 0.0, 0.13, 1.0])
    key.set(xlim=(0, 1), ylim=(0, 1), xticks=[], yticks=[0, 1])
    key.add_patch(Rectangle((0.22, 0.0), 0.56, 1.0, facecolor=ps.PAPER, edgecolor=ps.MUTED,
                            lw=0.7, zorder=2))
    key.add_patch(Rectangle((0.22, lo), 0.56, hi - lo, facecolor=ps.INK, edgecolor="none",
                            zorder=3))
    key.set_yticklabels(["0", "1"], fontsize=6.2)
    key.tick_params(length=1.8, pad=1.6)
    for sp in key.spines.values():
        sp.set_visible(False)
    # THE ZOOM IS DRAWN. The key's own y runs 0-1 in score and its frame spans exactly the panel's
    # height, so the panel's top edge is y = 1 here and the window's top is y = hi: a line between
    # them is the zoom wedge, and it needs no words.
    for corner, edge in ((1.0, hi), (0.0, lo)):
        key.plot([0.0, 0.22], [corner, edge], lw=0.6, color=ps.FAINT, zorder=1, clip_on=False)
    key.annotate(label, (0.5, 1.015), xycoords="axes fraction", ha="center", va="bottom",
                 fontsize=6.2, color=ps.MUTED, linespacing=1.2)
    return key


def print_size_provenance():
    """Print the settings held equal across the size series, which the panels no longer print."""
    print(f"  size series: {SIZE_LINE}; {SIZE_MATCHED}")


def size_active_titled(ax, c):
    """Figure 2's panel (f) with the slide's own title: the task and the read-out iteration.

    Args:
        ax: axes; c: the full cache dict, every size.
    Returns:
        whatever F2.panel_f returns, {N: {arm: (mean, n)}}.
    """
    out = F2.panel_f(ax, c)
    say_arms(ax, loc="upper left", fontsize=5.8, handlelength=1.1, borderaxespad=0.2, ncol=2,
             columnspacing=0.9)
    ax.set_title(SIZE_LINE, fontsize=7.4, color=ps.INK, pad=6)
    print_size_provenance()
    return out


def name_series_in_place(ax, fontsize=6.4, single_note="one size only"):
    """Name every labelled line beside its own last point, in its own colour, and drop the legend.

    A legend of six entries against six crossing lines is a matching puzzle, and two of these six
    pairs are near-twins in hue (light against dark purple, light against dark blue), so the puzzle
    has no reliable answer from a seat. The name goes where the line ends instead: a series that
    runs to the right-hand edge is named to the right of its last point, one that stops earlier is
    named above its last point, and a series drawn as a single marker is named below it and told
    apart as the one point it is.

    The names come from `ARM_SAYS` applied to the label the panel gave each line, so they are the
    same words the rest of the deck uses.

    Args:
        ax: the axes, with its lines already drawn and labelled; fontsize: label size in points;
        single_note: the three or four words marking a one-point series, or None to leave it bare.
    Returns:
        list of (name, x, y) in data units, one per named series.
    """
    handles, labels = ax.get_legend_handles_labels()
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    series = []
    for h, lab in zip(handles, labels):
        xy = np.asarray(h.get_xydata(), float)
        xy = xy[np.isfinite(xy).all(axis=1)]
        if len(xy):
            series.append((ARM_SAYS.get(lab, lab), h.get_color(), xy))
    if not series:
        return []
    x_end = max(xy[-1, 0] for _, _, xy in series)
    out = []
    for says, col, xy in series:
        x, y = xy[-1]
        if len(xy) == 1:                      # a single marker, not a series
            txt, off, ha, va = (f"{says}\n{single_note}" if single_note else says), (0, -7), \
                               "center", "top"
        elif x >= x_end * 0.999:              # runs to the right-hand edge
            txt, off, ha, va = says, (5, 0), "left", "center"
        else:                                 # stops short: name it above where it stops
            txt, off, ha, va = says, (0, 8), "center", "bottom"
        ax.annotate(txt, (x, y), textcoords="offset points", xytext=off, ha=ha, va=va,
                    fontsize=fontsize, color=col, linespacing=1.25, zorder=8)
        out.append((says, float(x), float(y)))
    return out


def size_r2_with_legend(ax, c):
    """Figure 2's panel (g), with every series named at its own end and the scale key it needs.

    In the paper (g) sits beside (f) and reads off its neighbour's legend. On a slide it stands
    alone, and a six-entry legend over six crossing lines was the slowest thing on the slide, so
    `name_series_in_place` writes each arm's name where its own line ends.

    THE WINDOW IS DECLARED. The panel's y axis spans 0.915 to 0.947: three percent of the score,
    which without a reference reads as a cliff. `score_scale_key` draws the whole 0-to-1 range
    beside it with that window marked, and the control's level is drawn across the panel as the
    line every other arm is judged against.

    Args:
        ax: axes; c: the full cache dict, every size.
    Returns:
        whatever F2.panel_g returns, {N: {arm: mean}}.
    """
    out = F2.panel_g(ax, c)
    # THE REFERENCE IS A RULE ACROSS THE PANEL, not just a series among six: the control's own mean
    # over the sizes it was run at, in the neutral reference ink.
    ctrl = [v["none"] for v in out.values() if "none" in v]
    if ctrl:
        ax.axhline(float(np.mean(ctrl)), color=ps.BASE, lw=0.7, ls=(0, (4, 3)), zorder=2)
    # ROOM AT THE RIGHT for the four series that run to N = 4000, and room underneath for the
    # single-size arm's two-line label. The x ticks stay at the four sizes that were run.
    ax.set_xlim(430, 9200)
    lo, hi = ax.get_ylim()
    ax.set_ylim(lo - 0.12 * (hi - lo), hi)
    name_series_in_place(ax)
    ax.set_title(SIZE_LINE, fontsize=7.4, color=ps.INK, pad=6)
    score_scale_key(ax)
    print_size_provenance()
    return out


# The same names again, broken over two lines where a tick label would otherwise run into its
# neighbour. Six categories across a 110 mm panel leave about 18 mm each.
TICK_SAYS = dict(ARM_SAYS, **{"syn. noise": "synaptic\nnoise", "frm + rws": "rate +\nsparsity"})


def rules_without_gloss(ax, y_cut=0.2):
    """Figure 2's panel (a) with the bottom gloss row dropped - the deck's anchor slide.

    F2.panel_a writes each rule's consequence as a sentence along the bottom of the panel ("the loss
    cannot see it, but it still drives the rest", and its four siblings). In the manuscript those
    sentences do work the caption cannot; on the one frame where a newcomer meets all five rules at
    once they are 45 of the panel's 131 words, set at 5.1-5.4 pt, and the deck's per-mechanism slides
    now each make the same point with a picture. They are removed by POSITION after panel_a has
    drawn, so the manuscript figure is untouched.

    Args:
        ax: a blank axes spanning the figure's width; y_cut: drop texts whose y in panel_a's own
            data coordinates falls below this. panel_a puts its rows at y = 1.00 (rule name), 0.845
            (the operation), 0.34 (per-unit labels) and 0.06 (the gloss), so 0.2 cuts the gloss row
            and nothing else.
    Returns:
        (words before, words after).
    """
    F2.panel_a(ax)
    before = sum(len(t.get_text().split()) for t in ax.texts)
    for t in [t for t in ax.texts if t.get_position()[1] < y_cut]:
        t.remove()
    after = sum(len(t.get_text().split()) for t in ax.texts)
    print(f"  slide_rules: {before} words on the panel, {after} with the gloss row removed")
    return before, after


def active_untilted(ax, c):
    """Figure 2's per-arm active-unit panel, with horizontal tick labels and plain arm names.

    The manuscript panel rotates its six category labels 45 degrees, which is right when they are
    long and wrong here: they are six short words that fit horizontally, and a tilted label is one
    more thing an audience has to straighten out in their head.

    Args:
        ax: axes; c: the cache restricted to one N and one cell per grid arm.
    Returns:
        whatever F2.panel_b returns, per-arm (mean, sd, n).
    """
    out = F2.panel_b(ax, c)
    ax.set_xticklabels([TICK_SAYS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                       rotation=0, ha="center", fontsize=6.0, linespacing=1.25)
    # ONE PHRASE FOR ONE AXIS. Its sibling panels say "active units of 1000"; this one said
    # "active units", so the same count appeared to be two different quantities across the deck.
    ax.set_ylabel(f"active units of {F2.N_MAIN}")
    return out


def weight_shape_said(ax, c):
    """Figure 2's weight-shape panel with its legend in the arms' plain-English names.

    The manuscript panel's legend carries the cache's own arm names, one of which is "no
    intervention" - a fifth spelling of the unpenalised control in a deck that elsewhere calls it
    "untouched". `say_arms` rebuilds the legend and leaves the panel alone.

    Args:
        ax: axes; c: the cache restricted to one N and one cell per grid arm.
    Returns:
        whatever F2.panel_weight_shape returns, (arm, mean sd, mean excess kurtosis) per arm.
    """
    out = F2.panel_weight_shape(ax, c)
    say_arms(ax, loc="upper left", fontsize=5.4, handlelength=1.0, borderpad=0.25,
             borderaxespad=0.3, ncol=1)
    return out


# Where each arm's name sits relative to its own cluster of networks, in points, with the alignment
# that goes with it. Six clusters on one cloud, two of them (dropout and synaptic noise) overlapping
# within 25 units and 0.001 of score, so the offsets are chosen by looking at the drawn panel rather
# than by a rule; a cluster whose name cannot sit beside it gets a leader line to it.
R2VA_LABEL_AT = {
    "untouched":      ((-30, 9), "right", "center", True),
    "duplication":    ((9, 0), "left", "center", False),
    "rescaling":      ((-10, 0), "right", "center", False),
    "dropout":        ((-26, -16), "right", "center", True),
    "synaptic noise": ((26, 10), "left", "center", True),
    "rate + sparsity": ((-9, 0), "right", "center", False),
}


def dims_untilted(ax, c):
    """Figure 2's dimensionality panel, with horizontal tick labels and plain arm names.

    The companion of `active_untilted`, and for the same reason: the manuscript panel is left alone
    and only its tick labels are restated. Without it this panel printed "none", "duplicate",
    "syn. noise" and "frm + rws" at 45 degrees while the panel one slide earlier said "untouched",
    "duplication", "synaptic noise" and "rate + sparsity" horizontally -- eight names for four
    conditions, two slides apart.

    Args:
        ax: axes; c: the cache restricted to one N and one cell per grid arm.
    Returns:
        whatever F2.panel_d returns.
    """
    out = F2.panel_d(ax, c)
    ax.set_xticklabels([TICK_SAYS.get(t.get_text(), t.get_text()) for t in ax.get_xticklabels()],
                       rotation=0, ha="center", fontsize=6.0, linespacing=1.25)
    return out


def r2_vs_active_with_scale(ax, c):
    """Figure 2's r2-against-active-units scatter, with each arm named on its own cluster.

    The manuscript panel is `F2.panel_r2_vs_active` and is not edited here - it is a manuscript
    figure. What a slide needs on top of it is the arm names a newcomer can read and a declaration
    of how little of the score range the y window covers.

    THE LEGEND IS GONE. It carried twenty of the panel's words, and it mixed three things a reader
    cannot hold at once: the arm's name, how many seeds it has in brackets, and its score as a
    percentage change while the axis beside it shows the score itself. The names now sit on the
    clusters they belong to; the seed counts and the per-arm changes are printed to stdout, so the
    deck's prose can quote them and nothing is lost from the record.

    Args:
        ax: axes; c: the cache restricted to one N and one cell per grid arm.
    Returns:
        whatever F2.panel_r2_vs_active returns, per-arm (mean active, mean r2, n).
    """
    out = F2.panel_r2_vs_active(ax, c)
    # the manuscript panel spells the silence criterion as a formula in its x label; a slide says
    # what the axis counts and leaves the rule to the slide that defines it
    ax.set_xlabel(f"active units of {F2.N_MAIN}")
    if ax.get_legend() is not None:
        ax.get_legend().remove()
    ref = out[0][1]                           # the untouched arm's mean, ARMS' first entry
    for (_, _, full, col), (a_mean, r_mean, n) in zip(F2.ARMS, out):
        if not n:
            continue
        says = ARM_SAYS.get(full, full)
        off, ha, va, leader = R2VA_LABEL_AT.get(says, ((0, 8), "center", "bottom", False))
        ax.annotate(says, (a_mean, r_mean), textcoords="offset points", xytext=off, ha=ha, va=va,
                    fontsize=6.4, color=col, zorder=8,
                    arrowprops=dict(arrowstyle="-", lw=0.6, color=col,
                                    shrinkA=1.0, shrinkB=5.0) if leader else None)
        print(f"  slide_f2_r2_vs_active: {says} {n} seeds, {a_mean:.0f} active units, "
              f"r2 {r_mean:.4f} ({(r_mean - ref) / ref:+.1%} of untouched)")
    score_scale_key(ax)
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
    arms = [("untouched", ps.BASE,
             DL.cell(os.path.join(root, DL.REFS["none"]))
             + DL.cell(os.path.join(root, DL.DROP_SUB, "EqType=h_k=3_N=1000_pen=none_do=none"))),
            ("mute dropout", ps.SLOTS[0],
             DL.cell(os.path.join(root, DL.DROP_SUB, "EqType=h_k=3_N=1000_pen=none_do=mute")))]
    arms = [a for a in arms if a[2]]
    if len(arms) < 2:
        print(f"  SKIP {name}: dropout cells missing")
        return None
    ps.setup()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(ps.W2 * 0.56, 112 * ps.MM))
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
    (e0, s0), (e1, s1) = stats["untouched"], stats["mute dropout"]
    # the y labels say what each panel is and the endpoint counts are in the deck text
    fig.suptitle("3-bit flip-flop, $N$ = 1000, unpenalised, every seed",
                 fontsize=7.4, color=ps.INK, y=1.015)
    # below the panels: a fourth title line ran into the loss axis's top tick label
    fig.text(0.5, -0.055,
             f"loss axis clipped at 1; {spikes} run{'s' if spikes != 1 else ''} spike above it",
             ha="center", va="top", fontsize=6.4, color=ps.MUTED)
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
    """Per-seed active count and held-out R^2, at end of training.

    ⚠️ THE PERFORMANCE MEASURE IS NO LONGER THE NOISE-FREE PROBE. This returned the trace's
    dropout-off, noise-OFF loss, which is not a condition any of these networks ever operates in,
    and the project found that instrument reverses the ranking of arms: measured without noise
    frm+rws is the best arm and measured with it the worst. Every performance number now comes from
    heldout_r2 - a held-out batch, sigma_rec and sigma_inp at the values the network trained with,
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
    cache = heldout.load_cache()
    for d in sorted(glob.glob(os.path.join(path, "*", ""))):
        f = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not f:
            continue
        with open(f[0], "rb") as fh:
            tr = pickle.load(fh)
        act.append(active_count(np.asarray(tr["participation"][-1], float), "scalefree"))
        r2.append(heldout.heldout_r2(d.rstrip(os.sep), cache=cache))
    heldout.save_cache(cache)
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
    ax.text(0.036, ctrl_a.mean() - 14, f"untouched, {ctrl_a.mean():.0f}", fontsize=6.6,
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
                 f"{ctrl_a.mean():.0f} of 1000 untouched, "
                 f"{grid[('mute', 0.25, 4)][0].mean():.0f} at mute "
                 rf"$\rho$ = 0.25, $\beta$ = 4, "
                 f"{grid[('dead', 0.25, 4)][0].mean():.0f} at dead.{exc}",
                 fontsize=8.0, color=ps.INK, linespacing=1.4, y=1.035)
    return ps.save(fig, name, w_mm=110)


def dropout_rate_cost_slide(name="slide_24b_dropout_rate_cost"):
    """One claim: mute recruits units for free, dead pays for them.

    Performance is the project's single measure: a held-out batch at the sigma_rec and sigma_inp the
    network trained with, sigma_w = 0, nine draws, dropout off - so a dropout net is scored on its
    full network exactly as the control is, under the noise both of them operate in. The noise-free
    probe this used to plot measured a condition none of these networks runs in, and it reversed the
    ranking of arms rather than shifting it.

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
    ax.set_ylabel("held-out $R^2$")
    ax.text(0.036, ctrl_l.mean() - 0.004, f"untouched, {ctrl_l.mean():.3f}", fontsize=6.6,
            color=ps.BASE, va="top")
    mute = [grid[("mute", r, b)][1].mean() for r in BERN_RATES for b in BERN_BETAS]
    dead = [grid[("dead", r, b)][1].mean() for r in BERN_RATES for b in BERN_BETAS]
    fig.suptitle(r"3-bit flip-flop, $N$ = 1000: held-out $R^2$ over the $\rho$ x $\beta$ grid"
                 "\n"
                 f"mute holds {min(mute):.3f}-{max(mute):.3f} against the control's "
                 f"{ctrl_l.mean():.3f}; dead falls to {min(dead):.3f}, "
                 f"{ctrl_l.mean() - min(dead):.3f} below it.",
                 fontsize=8.0, color=ps.INK, linespacing=1.4, y=1.035)
    return ps.save(fig, name, w_mm=110)


# The penalty arms, one figure per task. The 3-bit flip-flop panels cannot carry them as a series:
# its only frm+rws cells are N = 1000, and the other penalised flip-flop cells are k = 7 and k = 8, a
# different task rather than a bigger network.
#
# ⚠️ CDDM_std_g0_penalties IS LOCAL-ONLY - it is not on Della. A cluster-side scan says CDDM has
# penalties at N = 1000 alone; the Mac has all three arms at 500, 1000, 2000 and 5000.
# ⚠️ THE COLOURS COME FROM `ps.COND_COL`, NOT FROM A LOCAL CHOICE. These slides used to assign
# frm = SLOTS[0], rws = SLOTS[2], both = SLOTS[1], which put "frm + rws" in orange here and in green
# on slides 17-19, and gave green to rws here and to frm + rws there. One name in two colours, and
# one colour on two conditions, inside one deck.
PEN_KINDS = [("control", "untouched", ps.BASE), ("frm", "frm", ps.COND_COL["frm"]),
             ("rws", "rws", ps.COND_COL["rws"]), ("both", "frm + rws", ps.COND_COL["both"])]
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
    fig, ax = plt.subplots(figsize=(ps.W2, 74 * ps.MM))
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
        # NO CAPTION UNDER THE PANEL. It was one long line, and ps.save trims to a tight
        # bounding box, so the caption set the saved width and left the axes occupying about
        # half of it. What the markers mean, and which arms saturate, is in the deck text.
        pass
    return ps.save(fig, name)


PEN_CACHE = "data/cddm_penalty_cache.npz"
# ⚠️ THE NAMES ARE WHAT THE TERMS DO, not what the config calls them: "frm" and "rws" are the two
# penalties' keys and name nothing to someone seeing them once. The deck spells them out on the
# overview slide and uses these words everywhere after it.
PEN_ARMS = [("control", "untouched", ps.BASE), ("frm", "rate", ps.COND_COL["frm"]),
            ("rws", "sparsity", ps.COND_COL["rws"]),
            ("both", "rate + sparsity", ps.COND_COL["both"])]
BURST = 0.05          # fig_S3_transients' own cut: tPR/n below this is a burst unit


def _pen_cache():
    """The CDDM N = 1000 penalty cache, or None if it has not been built."""
    if not os.path.exists(PEN_CACHE):
        print(f"  (build it with cddm_penalty_cache.py: {PEN_CACHE} missing)")
        return None
    return np.load(PEN_CACHE, allow_pickle=True)


# The two arms this figure compares, and the words it calls them by. The firing-rate-magnitude
# penalty pushes every unit's rate up; the row-wise-sparsity penalty caps how many inputs a unit
# listens to. The code names ride along in brackets once, which is the only place they appear.
TPR_ROWS = [("frm", "rate penalty (frm)"), ("both", "+ sparsity penalty (rws)")]


def temporal_pr_slide(name="slide_30_temporal_pr"):
    """How much of the trial a unit fires, under the rate penalty and under both penalties.

    THE AXIS IS THE PICTURE. The quantity used to be labelled "temporal participation ratio, tPR/n",
    an acronym defined only in the deck's prose, while the figure drew two example units - one
    bursting, one firing throughout - off to the right in a column of their own. Those two traces
    ARE the definition, so they now sit above the axis AT their own values, the bursting one at the
    left end and the sustained one at the right, and the axis is named in plain words.

    TWO ROWS, NOT FOUR. The unpenalised and sparsity-only arms were drawn too, and neither carries
    the claim: what the slide says is that adding the sparsity penalty to the rate penalty rescues
    the units that fire only in a brief transient. That is a statement about two distributions, and
    their difference is in the lower tail, where a third and fourth histogram only add ink.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cache is missing.
    """
    z = _pen_cache()
    if z is None:
        return None
    cols = {a_: c_ for a_, _l, c_ in PEN_ARMS}
    rows = [(a_, lab, cols[a_]) for a_, lab in TPR_ROWS
            if any(k.startswith(f"{a_}|") and k.endswith("|tpr") for k in z.files)]
    if not rows:
        print(f"  SKIP {name}: no temporal traces in the cache")
        return None
    ex_arm = "frm" if "frm|0|ex" in z.files else rows[0][0]
    ex = np.asarray(z[f"{ex_arm}|0|ex"], float)             # (2, timesteps): burst, sustained
    ex_tpr = np.asarray(z[f"{ex_arm}|0|ex_tpr"], float)
    x_hi = max(0.45, float(ex_tpr.max()) * 1.06)

    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 76 * ps.MM))
    gs = GridSpec(1 + len(rows), 1, figure=fig, height_ratios=[0.95] + [1.0] * len(rows),
                  hspace=0.30, left=0.055, right=0.985, top=0.93, bottom=0.11)
    # ---- the two example units, each above the value it has on the axis below ------------------
    ax_ex = ps.blank(fig.add_subplot(gs[0]))
    ax_ex.set(xlim=(0, x_hi), ylim=(0, 1))
    for j, says in enumerate(("brief burst", "fires throughout")):
        # width in DATA units of the axis below, so the inset sits over its own value; it is pulled
        # inside the frame at the edges rather than hanging off them
        w = 0.105 * x_hi
        x0 = min(max(float(ex_tpr[j]) - w / 2, 0.004 * x_hi), x_hi - w - 0.004 * x_hi)
        a_ = ax_ex.inset_axes([x0, 0.26, w, 0.62], transform=ax_ex.transData)
        a_.plot(ex[j], lw=1.2, color=ps.INK, zorder=4)
        a_.set(xticks=[], yticks=[], ylim=(-0.02, max(0.04, float(ex[j].max()) * 1.2)))
        for sp in a_.spines.values():
            sp.set_color(ps.GRID)
        a_.annotate(says, (0.5, 1.02), xycoords="axes fraction", ha="center", va="bottom",
                    fontsize=6.4, color=ps.INK)
        # the leader ties the trace to its place on the axis
        ax_ex.plot([float(ex_tpr[j])] * 2, [0.0, 0.24], ls=":", lw=0.8, color=ps.MUTED, zorder=2)

    # ---- the two distributions -----------------------------------------------------------------
    bins = np.linspace(0, x_hi, 68)
    axes, q25s = [], []
    for i, (arm, lab, col) in enumerate(rows):
        ax = fig.add_subplot(gs[i + 1], sharex=axes[0] if axes else None)
        axes.append(ax)
        per_seed = [z[k] for k in z.files if k.startswith(f"{arm}|") and k.endswith("|tpr")]
        v = np.concatenate(per_seed)
        ax.hist(v, bins=bins, color=col, alpha=0.85, density=True, zorder=3)
        q25 = float(np.quantile(v, 0.25))
        q25s.append(q25)
        # THE QUIETEST QUARTER IS AN AREA, NOT A NUMBER. The claim of this slide is that the
        # sparsity term moves the bottom of the distribution, and it used to be carried by two
        # printed numbers while the two histograms looked alike. Shading everything left of each
        # arm's own lower quartile turns the claim into a block that visibly widens between the
        # rows, and the arrow drawn below shows the boundary moving.
        ax.axvspan(0, q25, color=ps.GRID, alpha=0.85, lw=0, zorder=1)
        ax.axvline(q25, color=ps.INK, lw=1.0, ls=(0, (3, 2)), zorder=5)
        ax.annotate(f"{q25:.3f}", (q25, 0.80), xycoords=("data", "axes fraction"), ha="left",
                    va="top", fontsize=6.4, color=ps.INK, xytext=(3, 0),
                    textcoords="offset points")
        # the two example units' own values, marked where the data is so the traces above are
        # anchored to something inside the distribution
        for v_ex in ex_tpr:
            ax.axvline(float(v_ex), color=ps.FAINT, lw=0.7, ls=":", zorder=2)
        ax.annotate(lab, (0.985, 0.72), xycoords="axes fraction", ha="right", va="center",
                    fontsize=7.0, color=col)
        ax.set(xlim=(0, x_hi))
        ax.set_yticks([])
        ps.ygrid(ax)
        if i < len(rows) - 1:
            ax.tick_params(labelbottom=False)
        # the deck's burst share and per-seed quartiles stay derivable without being drawn
        print(f"  {name}: {arm} q25 per seed "
              f"{[round(float(np.quantile(x, 0.25)), 3) for x in per_seed]}, "
              f"units below {BURST} "
              f"{[round(100.0 * float(np.mean(x < BURST)), 1) for x in per_seed]}%")
    axes[-1].set_xlabel("share of the trial the unit fires")
    # THE SHIFT, DRAWN INSIDE THE LOWER ROW. The first row's quarter boundary is repeated there as
    # a faint line and an arrow runs from it to the row's own boundary, so the move is one picture
    # rather than two numbers in two panels that a listener has to subtract.
    for ax, q25 in zip(axes[1:], q25s[1:]):
        ax.axvline(q25s[0], color=ps.FAINT, lw=0.9, ls=(0, (3, 2)), zorder=4)
        ax.annotate("", xy=(q25, 0.52), xytext=(q25s[0], 0.52),
                    xycoords=("data", "axes fraction"),
                    arrowprops=dict(arrowstyle="-|>", lw=1.5, color=ps.INK, shrinkA=0, shrinkB=0,
                                    mutation_scale=9), zorder=6)
    # the dashed line is named ON the line, in the top row only
    axes[0].annotate("quietest quarter", (q25s[0], 0.99), xycoords=("data", "axes fraction"),
                     ha="left", va="top", fontsize=6.4, color=ps.INK, xytext=(3, 0),
                     textcoords="offset points")
    fig.suptitle("CDDM, $N$ = 1000", fontsize=7.4, color=ps.INK, y=0.985)
    return ps.save(fig, name)


def frm_vs_both_slide(name="slide_31_frm_vs_both", n_units=1000):
    """What each of the two penalties buys, on the four measures, CDDM at N = 1000.

    THE SLIDE THIS REPLACES HAD NO BUILDER. Its panel was drawn by code that was never committed, so
    it went on showing the noise-free score after the project moved to held-out r2 -- and the two
    disagree about which arm is best. This version reads the committed cache for the three measures
    that do not depend on the evaluation (active units, dimensionality, recurrent weight magnitudes)
    and calls `heldout_r2` for the one that does, so the figure and the talk track cannot drift
    apart again.

    FOUR PANELS, ONE MEASURE EACH, SAME FOUR ARMS ON EVERY x AXIS. The question is which of frm and
    rws moves which measure, and that is a question about a 4 x 4 table; drawing it as four strips
    with every seed shown lets a reader run their eye down a column.

    Args:
        name: output file stem; n_units: network size the arms are compared at.
    Returns:
        the output path, or None if the penalty cache is missing.
    """
    z = _pen_cache()
    if z is None:
        return None
    arm_of = np.asarray(z["arm"])
    rows = [(a, lab, col) for a, lab, col in PEN_ARMS if (arm_of == a).any()]

    # held-out r2 is NOT in the cache: the cache predates the measure. Score from the folders, which
    # is cheap because heldout_r2 memoises on the parameter file's path, size and mtime.
    heldout_by_arm = {}
    for a, _, _ in rows:
        vals = []
        for cell in sorted(glob.glob(PEN_TASKS["CDDM"][2][a].format(N=n_units))):
            for net in sorted(glob.glob(os.path.join(cell, "*"))):
                if not os.path.isdir(net):
                    continue
                v = heldout.heldout_r2(net)
                if v is not None and np.isfinite(v):
                    vals.append(float(v))
        heldout_by_arm[a] = np.asarray(vals, float)

    ps.setup()
    # ONE COLUMN: four panels in a row is four pictures competing for one glance, and these four
    # share a y axis meaning, so stacked they are read as one ladder.
    fig = plt.figure(figsize=(78 * ps.MM, 180 * ps.MM))
    gs = GridSpec(4, 1, figure=fig, hspace=0.52)
    xs = np.arange(len(rows))
    cols = [c for _, _, c in rows]
    labs = [lab for _, lab, _ in rows]

    panels = [("active units of 1000", [np.asarray(z["n_active"], float)[arm_of == a]
                                        for a, _, _ in rows]),
              ("held-out $r^2$", [heldout_by_arm[a] for a, _, _ in rows]),
              ("dimensions used", [np.asarray(z["dims"], float)[arm_of == a]
                                   for a, _, _ in rows])]
    for j, (ylab, groups) in enumerate(panels):
        ax = fig.add_subplot(gs[j])
        ps.strip(ax, xs, groups, cols, rng=np.random.default_rng(31))
        ax.set_xticks(xs)
        # ROTATED, and the rotation is not a style choice: four panels across 180 mm leaves about
        # 38 mm each, and horizontal labels ran "untouched" into "rate" and "sparsity" into
        # "rate + sparsity". The names are plain words now, which is the part that matters.
        ax.set_xticklabels(labs, rotation=28, ha="right", fontsize=6.2)
        ax.set_ylabel(ylab)
        ps.ygrid(ax)
        if j == 0:
            ax.set_ylim(0, 1050)
            ax.axhline(n_units, color=ps.FAINT, lw=0.7, ls=":", zorder=1)

    # ---- the weight distribution, where the two penalised arms part --------------------------
    ax = fig.add_subplot(gs[3])
    edges = np.asarray(z["log_bins"], float)
    centres = 0.5 * (edges[:-1] + edges[1:])
    for a, lab, col in rows:
        hs = [np.asarray(z[k], float) for k in z.files
              if k.startswith(f"{a}|") and k.endswith("|w_hist")]
        if not hs:
            continue
        h = np.mean(hs, axis=0)
        tot = h.sum()
        if tot <= 0:
            continue
        ax.plot(centres, h / tot, lw=1.1, color=col, zorder=3)
    ax.set(xlabel="recurrent weight", ylabel="fraction of synapses", xlim=(-8, 1))
    ax.set_xticks([-6, -4, -2, 0])
    ax.set_xticklabels(["$10^{-6}$", "$10^{-4}$", "$10^{-2}$", "1"])
    ps.ygrid(ax)

    fig.suptitle(f"CDDM, $N$ = {n_units}, 200,000 iterations, one dot per seed",
                 fontsize=7.4, color=ps.INK, y=1.04)
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


# The words each condition goes by on this slide, matching slide 30's.
SEL_SAYS = {"control": "untouched", "frm": "rate penalty (frm)",
            "both": "+ sparsity penalty (rws)"}
# ONE VIEWING ANGLE FOR ALL THREE. Each panel is still its own PCA - the coordinates are not
# comparable between conditions - but the ANGLE was per panel too, fitted to each condition's own
# clusters, and three clouds at three angles cannot be compared by eye at all: a reader cannot tell
# a shape difference from a camera difference. One angle, chosen to keep all three readable.
SEL_ANGLE = (24, 52)


def selectivity_slide(name="slide_32_selectivity"):
    """CDDM selectivity: every active unit as a point in the three directions of its own response.

    ⚠️ THESE ARE THE 30,000-ITERATION NETWORKS (CDDM_std_g0), not the 200,000-iteration penalty sweep
    slides 28-31 read. By 200k the configuration has collapsed; at 30k it still has its full form,
    and that sweep is also the one carrying the animated_selectivity movies.

    A ROW, NOT A COLUMN, AND ONE ANGLE. The three conditions used to be stacked vertically, each at
    its own viewing angle, which asked the audience to compare three shapes photographed from three
    camera positions. Side by side at one angle, the comparison is at least a fair one.

    WHAT IS NOT ON THE FIGURE. The deck's claim is an arm COUNT - a clump, three arms, four - and
    that count is a cluster-balance measurement that is not in this repository: the cache holds
    coordinates and explained variance only, and `SEL_VIEW`'s first field disagrees with the deck
    about the unpenalised networks (3 there, "a clump" in the prose). So each panel prints what the
    cache does measure, its active-unit count, and the arm count stays in the prose until the
    measurement that produced it is committed.

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
    fig = plt.figure(figsize=(ps.W2, 66 * ps.MM))
    gs = GridSpec(1, len(show), figure=fig, wspace=0.02, left=0.01, right=0.99,
                  top=0.86, bottom=0.02)
    elev, azim = SEL_ANGLE
    for i, (arm, _lab, col) in enumerate(show):
        _k, seed, _e, _a = SEL_VIEW[arm]
        ax = fig.add_subplot(gs[0, i], projection="3d")
        P = np.asarray(z[f"{arm}|{seed}|pcs"], float)
        ax.scatter(P[:, 0], P[:, 1], P[:, 2], s=4.5, c=col, alpha=0.55, linewidths=0, zorder=3)
        ax.set_title(f"{SEL_SAYS.get(arm, arm)}\n{len(P)} active units",
                     fontsize=7.0, color=col, pad=-2, linespacing=1.4)
        for pane in (ax.xaxis, ax.yaxis, ax.zaxis):
            pane.set_pane_color((1.0, 1.0, 1.0, 0.0))
            pane.line.set_color(ps.GRID)
        # NO PC1/PC2/PC3 LABELS. The axes are three directions of one condition's own response and
        # mean nothing across panels; what the panel is for is the SHAPE of the cloud, and three
        # acronyms per panel is nine pieces of text saying nothing a newcomer can decode.
        ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1.0, 1.0, 0.85), zoom=1.3)
        print(f"  {name}: {arm} seed {seed}, {len(P)} active units, "
              f"spread {np.round(P.std(axis=0), 2)}")
    # ABOVE the panel titles, not on them: a 3-D axes' title sits high in its own box
    fig.suptitle("CDDM, $N$ = 1000", fontsize=7.4, color=ps.INK, y=1.05)
    return ps.save(fig, name)


# Prune-and-duplicate: the two knobs the sweeps actually vary.
# JITTER: NBitFlipFlop_copy_perturb walks copy_noise 0 -> 3 at N = 1000, everything else fixed.
# MODE: NBitFlipFlop_replace_variants swaps what a pruned unit is replaced WITH. The alternatives
# exist only at max_replace_frac = 0.025 and maturity = 1000, so the comparison is made there -
# copy also has cells at other rates, and using them would compare the rate, not the mode.
PD_JITTER = f"{DATA_DIR}/NBitFlipFlop_copy_perturb"
PD_MODES = f"{DATA_DIR}/NBitFlipFlop_replace_variants"
# ⚠️ THE MATCHED CELL IS rate = 0.005, NOT 0.025. The other reinit modes exist only at 0.025, but
# every `copy` cell at 0.025 and N = 1000 carries the __DETUNED_SELFWEIGHT bug - duplication zeroed
# the 2x2 block spanning donor and copy, so the copy had no self-connection and the donor half of its
# own - and Figure 2 excludes those by name. The corrected copy at N = 1000 exists at rate 0.001 and
# 0.005 only. At 0.005 with maturity 1000 there is a control, a zero_out cell and a corrected copy
# cell, which is the comparison that isolates what the COPY contributes over merely pruning.
PD_MATCHED = [
    ("untouched", f"{DATA_DIR}/NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none"),
    ("prune, zero the unit",
     f"{DATA_DIR}/NBitFlipFlop_replace_variants/EqType=h_k=3_N=1000_mode=zero_out_rate=0.005_mat=1000"),
    ("prune, copy a donor",
     f"{DATA_DIR}/NBitFlipFlop_replace_variants_fix/EqType=h_k=3_N=1000_mode=copy_rate=0.005_mat=1000"),
]


def _pd_cell(cell):
    """Active units and held-out r2 per seed for one cell.

    ⚠️ R2 WAS THE FOLDER-NAME SCORE AND IS NOT ANY MORE. That prefix is one forward pass, one noise
    draw, on the TRAINING batch - noise-on, but neither held out nor averaged, so it is not the
    measure the rest of this deck reports. It is still read here to drop diverged runs (`nan_`),
    which is all a folder name is good for.

    Args:
        cell: a cell directory, or a glob matching one.
    Returns:
        (active list, r2 list), both empty if the cell has no usable traces.
    """
    act, r2 = [], []
    for base in sorted(glob.glob(cell)):
        for d in sorted(glob.glob(os.path.join(base, "*/"))):
            head = os.path.basename(d.rstrip("/")).split("_")[0]
            if head == "nan":
                continue
            tp = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
            if not tp:
                continue
            try:
                tr = pickle.load(open(tp[0], "rb"))
            except Exception:
                continue
            P = np.asarray(tr.get("participation", []), float)
            if P.ndim == 2 and len(P):
                act.append(active_count(P[-1], "scalefree"))
                r2.append(heldout.heldout_r2(d.rstrip(os.sep)))
    return act, r2


def prune_duplicate_slide(name="slide_25_prune_duplicate"):
    """Prune-and-duplicate: recruitment against the copy's noise, and against what replaces a unit.

    BOTH PANELS COUNT AGAINST 1000, THE WHOLE NETWORK. An earlier version windowed the left panel
    to 640-760, so a 70-unit difference filled the frame and read as a cliff. The axis now runs 0 to
    the network's size with the "all 1000" line drawn, which is the only scale the recruitment claim
    means anything on, and the two panels share it so the right one can be read against the left.

    ONE COLOUR, ONE MEANING. The duplication rule is `ps.COND_COL["duplicate"]` here and on every
    other slide; anything that does not duplicate - no pruning, and pruning that zeroes the unit
    instead of copying into it - is the neutral reference ink.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if neither sweep is on disk.
    """
    jit = []
    for cell in sorted(glob.glob(os.path.join(PD_JITTER, "*"))):
        cfgs = glob.glob(os.path.join(cell, "*", "*_config.yaml"))
        if not os.path.isdir(cell) or not cfgs:
            continue
        pa = OmegaConf.load(cfgs[0]).trainer.prune_args
        if bool(pa.get("copy_iid", False)) or bool(pa.get("copy_permute", False)):
            continue                      # those are separate controls, not rungs of this ladder
        a, r = _pd_cell(cell)
        if a:
            jit.append((float(pa.copy_noise), a, r))
    jit.sort()
    modes = []
    for lab, cell in PD_MATCHED:
        a, r = _pd_cell(cell)
        if a:
            modes.append((lab, a, r))
    if not jit and not modes:
        print(f"  SKIP {name}: no prune-and-duplicate cells on disk")
        return None
    n_units = 1000
    dup = ps.COND_COL["duplicate"]
    ps.setup()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(ps.W2 * 0.56, 108 * ps.MM), sharex=True)
    xs = np.arange(len(jit), dtype=float)
    for x, (noise, a, r) in zip(xs, jit):
        ax.plot([x] * len(a), a, "o", ms=3.4, color=dup, alpha=0.85, mec="none", zorder=4)
        ax.plot([x], [np.mean(a)], "s", ms=6.5, mfc="none", mec=dup, mew=1.2, zorder=5)
    ax.plot(xs, [np.mean(a) for _n, a, _r in jit], "-", lw=1.1, color=dup, zorder=3)
    ax.set_xticks(xs)
    # the anchor rides ON the leftmost tick: as a floating annotation it read as a stray label
    ax.set_xticklabels([("0\nexact copy" if n == 0 else f"{n:g}") for n, _a, _r in jit],
                       fontsize=6.4)
    # PLAIN WORDS, NOT THE CONFIG KEY. `copy_noise` sets how far the copy's incoming weight vector
    # is rotated off the donor's (Trainer holds the row's length fixed), so on the slide it is how
    # different the copy is, and x = 0 is marked as the exact copy it is.
    ax.set(xlabel="noise added to each copy", ylabel=f"active units of {n_units}")
    ax.set_xlim(-0.5, len(xs) - 0.5)
    ps.ygrid(ax)
    # The short label is what the audience reads; the long one in PD_MATCHED stays as the record of
    # which cell it is. Only the copying rule is a duplication arm, so only it takes the arm colour.
    short = {"untouched": "untouched", "prune, zero the unit": "pruned, zeroed",
             "prune, copy a donor": "pruned, copied"}
    xs2 = np.arange(len(modes), dtype=float)
    for x, (lab, a, r) in zip(xs2, modes):
        col = dup if "copy" in lab else ps.BASE
        ax2.plot([x] * len(a), a, "o", ms=3.4, color=col, alpha=0.85, mec="none", zorder=4)
        ax2.plot([x], [np.mean(a)], "s", ms=6.5, mfc="none", mec=col, mew=1.2, zorder=5)
    ax2.set_xticks(xs2)
    ax2.set_xticklabels([short.get(l, l) for l, _a, _r in modes], fontsize=6.4)
    ax2.set_xlim(-0.5, len(xs2) - 0.5)
    ps.ygrid(ax2)
    # THE REFERENCE CROSSES THE LEFT PANEL TOO. Without it the left panel is a flat trend with
    # nothing to be flat ABOVE: a reader cannot tell whether 700 units at every jitter is a result
    # or a null. The untouched networks' own level is the line, in the neutral reference ink.
    untouched = next((a for lab, a, _r in modes if lab == "untouched"), None)
    if untouched:
        ax.axhline(np.mean(untouched), color=ps.BASE, lw=0.9, ls=(0, (4, 3)), zorder=2)
        ax.annotate("untouched", (len(xs) - 0.55, np.mean(untouched)), textcoords="offset points",
                    xytext=(0, 4), ha="right", va="bottom", fontsize=6.4, color=ps.BASE)
    for a_ in (ax, ax2):
        a_.set_ylim(0, n_units * 1.06)
        a_.axhline(n_units, color=ps.FAINT, lw=0.7, ls=":", zorder=1)
    ax.annotate(f"all {n_units}", (-0.45, n_units), textcoords="offset points", xytext=(0, 3),
                ha="left", va="bottom", fontsize=6.0, color=ps.MUTED)
    fig.suptitle("3-bit flip-flop, $N$ = 1000", fontsize=7.4, color=ps.INK, y=1.015)
    return ps.save(fig, name)


# Prune-and-duplicate at the Figure 2 operating point (copy_noise = 1), everywhere it was run.
# The flip-flop's CONTROLS below N = 4000 are in ff_revive, not paper_grid, at the same 40,000
# iterations; CDDM and DMTS carry both arms in their own paper grids.
# NO PER-TASK COLOUR. These entries used to carry one, and the three tasks' figures drew the SAME
# duplication arm in three different hues - one of them the hue that means the firing-rate penalty
# everywhere else in the deck. The arm's colour is `ps.COND_COL["duplicate"]`, looked up once in
# the builder, so a task cannot acquire its own.
PD_FAIL = 0.5     # a DMTS run that never learned scores about 0.427; nothing solved lands near it
PD_ALL = [
    ("3-bit flip-flop", 40_000, {
        "control": {500: f"{DATA_DIR}/NBitFlipFlop_ff_revive/EqType=h_k=3_N=500_pen=none_arm=none",
                    1000: f"{DATA_DIR}/NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none",
                    2000: f"{DATA_DIR}/NBitFlipFlop_ff_revive/EqType=h_k=3_N=2000_pen=none_arm=none",
                    4000: f"{DATA_DIR}/NBitFlipFlop_paper_grid/EqType=h_N=4000_arm=control"},
        "duplicate": {n: f"{DATA_DIR}/NBitFlipFlop_paper_grid/EqType=h_N={n}_arm=duplication"
                      for n in (500, 1000, 2000, 4000)}}),
    ("CDDM", 100_000, {
        "control": {n: f"{DATA_DIR}/CDDM_paper_grid/EqType=h_N={n}_arm=control"
                    for n in (500, 1000, 2000, 4000)},
        "duplicate": {n: f"{DATA_DIR}/CDDM_paper_grid/EqType=h_N={n}_arm=duplication"
                      for n in (500, 1000, 2000, 4000)}}),
    ("DMTS, 7$\\tau$", 150_000, {
        "control": {n: f"{DATA_DIR}/DMTS_paper_grid/EqType=h_N={n}_arm=control"
                    for n in (500, 1000, 2000)},
        "duplicate": {n: f"{DATA_DIR}/DMTS_paper_grid/EqType=h_N={n}_arm=duplication"
                      for n in (500, 1000, 2000)}}),
]


def round_log_ticks(lo, hi):
    """Round tick values for a log axis that spans barely more than one decade.

    Matplotlib's log locator puts at most ONE major tick inside a range like 130 to 1800 and then
    falls back to labelling minors, which is where "6 x 10^2" comes from. No audience reads a value
    off an axis labelled that way. This returns values off the 1-2-5 ladder that bracket the data,
    densified to 1-2-3-5 and then to a finer ladder if the coarse one leaves fewer than three.

    Args:
        lo, hi: the smallest and largest value the axis has to show, both strictly positive.
    Returns:
        a sorted list of integer tick values covering [lo, hi] with a little slack at each end.
    """
    t = []
    for mults in ((1, 2, 5), (1, 2, 3, 5), (1, 1.5, 2, 3, 5, 7)):
        ladder = sorted(m * 10 ** k for k in range(6) for m in mults)
        t = [int(round(v)) for v in ladder if lo * 0.8 <= v <= hi * 1.25]
        if len(t) >= 3:
            break
    return t


def prune_duplicate_all_slide():
    """One figure per task: prune-and-duplicate against its control, in units and in performance.

    ONE TASK PER FIGURE. Three tasks on one pair of axes puts six series on each panel and the eye
    has nowhere to rest; the tasks also differ in budget and in whether their runs solve at all, so
    they are not really comparable within a panel anyway.

    Active units come from each run's participation trace, which the Trainer logs noise-free. r2 is
    the held-out score in the folder name: one forward pass with the network's own noise, on a batch
    regenerated after training.

    A CELL WHERE NO SEED SOLVED THE TASK IS STRUCK OUT. On DMTS every duplication run sits at the
    failure score, so its recruitment is counted in networks that never learned and means nothing.
    The figure now says that itself: the performance panel shades everything below the solved bar
    and names it, and each struck cell carries a cross in both panels. The audit's rule - a panel
    that shows a meaningless number without saying so is worse than no panel.

    Returns:
        list of output paths, one per task with data.
    """
    out = []
    col = ps.COND_COL["duplicate"]
    for task, iters, arms in PD_ALL:
        stem = "slide_25b_prune_duplicate_" + (task.split(",")[0].replace("-", "")
                                               .replace(" ", "").lower())
        ps.setup()
        fig, (ax, ax2) = plt.subplots(2, 1, figsize=(ps.W2 * 0.56, 112 * ps.MM))
        drew, failed_at, r2_all = False, [], []
        # the two arms get two COLOURS, not one colour and two line styles: the r2 panel draws
        # individual runs as dots, where a line style says nothing
        for arm, lab, acol, style in (("control", "untouched", ps.BASE, dict(ls=":")),
                                      ("duplicate", "prune + duplicate", col, dict(ls="-"))):
            Ns, act, r2, struck = [], [], [], []
            for n, cell in sorted(arms.get(arm, {}).items()):
                a_, r = _pd_cell(cell)
                if not a_:
                    continue
                drew = True
                solved = [v for v in r if v > PD_FAIL]
                dead_cell = bool(r) and not solved      # nothing in this cell learned the task
                Ns.append(n)
                act.append(np.mean(a_))
                struck.append(dead_cell)
                r2_all += list(r)
                # a mean over solved and failed runs is not a performance; see PD_FAIL
                r2.append(np.mean(r) if r and len(solved) == len(r) else np.nan)
                for v in a_:
                    ax.plot([n], [v], "o", ms=3.0, color=acol, alpha=0.3 if dead_cell else 0.55,
                            mec="none", zorder=3)
                for v in r:
                    ax2.plot([n], [v], "o", ms=3.6, color=acol, alpha=0.8, mec="none", zorder=4)
                if dead_cell:
                    failed_at.append((n, float(np.mean(a_)), float(np.mean(r))))
            if Ns:
                # NO LINE INTO OR OUT OF A CELL WHERE NOTHING SOLVED. The unit panel used to join
                # every cell, so on DMTS a trend was drawn straight through the two cells the figure
                # elsewhere strikes out - which invites the reader to read a recruitment curve off
                # networks that never learned the task. The squares stay, the segments do not.
                # (The performance panel already breaks, because a struck cell's r2 is nan there.)
                act_line = [np.nan if d else a for a, d in zip(act, struck)]
                ax.plot(Ns, act_line, ls=style["ls"], color=acol, lw=1.3, zorder=5)
                ax.plot(Ns, act, "s", ls="none", ms=5.2, color=acol, mew=1.2, mfc="none",
                        zorder=6, label=lab)
                ax2.plot(Ns, r2, "s" + style["ls"], ms=5.2, color=acol, mew=1.2,
                         mfc="none", lw=1.3, zorder=6)
        if not drew:
            plt.close(fig)
            continue
        Ns_all = sorted({n for d in arms.values() for n in d})
        for a_ in (ax, ax2):
            a_.set(xscale="log")
            a_.set_xticks(Ns_all)
            a_.set_xticklabels([str(n) for n in Ns_all])
            a_.xaxis.set_minor_locator(NullLocator())
            ps.ygrid(a_)
        # A LOG AXIS WITH TICKS THAT CAN BE READ. These ranges are barely over one decade, so the
        # default locator labelled a single tick and then fell back to minors like "6 x 10^2".
        ax.set(yscale="log", ylabel="active units")
        act_all = [v for ln in ax.get_lines() for v in np.asarray(ln.get_ydata(), float)
                   if np.isfinite(v) and v > 0]
        ticks = round_log_ticks(min(act_all), max(act_all))
        ax.set_yticks(ticks)
        ax.set_yticklabels([f"{t:,}" for t in ticks])
        ax.yaxis.set_minor_locator(NullLocator())
        ax.yaxis.set_minor_formatter(NullFormatter())
        ax.set_ylim(min(min(act_all), ticks[0]) * 0.92, max(max(act_all), ticks[-1]) * 1.10)
        ax2.set_ylabel("held-out $r^2$")
        ax2.margins(y=0.12)
        # THE SOLVED BAR IS DRAWN ONLY WHERE IT BITES. On the flip-flop and CDDM every run solves
        # and the scores live in 0.91-0.95, so stretching the axis down to 0.5 to show a bar nobody
        # crosses would flatten the one difference those panels carry.
        if failed_at:
            # the floor is dropped well under the failed runs so the band's name has clear space
            lo = min(min(r2_all), PD_FAIL) - 0.12
            ax2.set_ylim(lo, max(r2_all) + 0.04)
            ax2.axhspan(lo, PD_FAIL, color=ps.BAD, alpha=0.07, lw=0, zorder=1)
            ax2.axhline(PD_FAIL, color=ps.BAD, lw=0.8, ls=(0, (4, 2)), zorder=2)
            # the band's name sits at the RIGHT edge: at the left it printed over the smallest
            # network's seeds, which are the ones that did solve
            ax2.annotate("task not solved", (0.5, 0.02), xycoords="axes fraction", ha="center",
                         va="bottom", fontsize=6.2, color=ps.BAD)
            for n, a_mean, r_mean in failed_at:
                ax.plot([n], [a_mean], "x", ms=7.5, mew=1.7, color=ps.BAD, zorder=8)
                ax2.plot([n], [r_mean], "x", ms=7.5, mew=1.7, color=ps.BAD, zorder=8)
            n0, a0, _r0 = failed_at[0]
            ax.annotate("no seed solved", (n0, a0), textcoords="offset points", xytext=(6, 9),
                        ha="left", va="bottom", fontsize=6.2, color=ps.BAD)
        else:
            # NO FAILURES MEANS A SLIVER OF A WINDOW. CDDM's scores run 0.815 to 0.840, so a
            # difference of 0.01 fills the panel and reads as a collapse. The whole-score bar says
            # how much of the 0-to-1 range the frame is showing, as it does on the Figure 2 panels.
            score_scale_key(ax2)
        ax.legend(loc="upper left", fontsize=6.0, handlelength=1.5, borderaxespad=0.25)
        # ONE x label for the pair, under both panels - it is the same axis twice. Its row has to be
        # cleared first or it prints over the tick labels.
        fig.subplots_adjust(bottom=0.20)
        fig.supxlabel("network size $N$", fontsize=7.0, color=ps.INK, y=0.03)
        fig.suptitle(f"{task}, {iters:,} iterations", fontsize=7.4, color=ps.INK, y=1.015)
        out.append(ps.save(fig, stem))
    return out


# ---- synaptic rescaling -------------------------------------------------------------------------
# THE CELLS ARE DISCOVERED, NOT LISTED. The rule was developed over nine sweeps and the knobs are
# spelled differently in different folder names (`tgt=`, `arm=rescale_tgt10`, nothing at all), so
# every cell under these sweeps is read, its own config is asked what it was, and the slides select
# on the config. A hand-written cell list went stale three times while Figure 2 was being built.
RESC_SWEEPS = ("NBitFlipFlop_target_ladder", "NBitFlipFlop_rescale_revive",
               "NBitFlipFlop_rescale_active", "NBitFlipFlop_rescale_states",
               "NBitFlipFlop_norm_mode", "NBitFlipFlop_recgrow", "NBitFlipFlop_revive_ops2",
               "NBitFlipFlop_revive_refr", "NBitFlipFlop_size_series")
# the unpenalised networks these are all matched against: same task, size, budget, gamma = 0
RESC_CTRL = f"{DATA_DIR}/NBitFlipFlop_ff_revive/EqType=h_k=3_N={{n}}_pen=none_arm=none"
# THE LADDER'S OWN CELLS, defined by the knobs rather than by sweep: everything matched except the
# activity target. `rescale_active_only` is in here because it changes where the boost is spent, not
# merely how much of it there is - a synapse from a unit that emits nothing delivers nothing.
RESC_MATCHED = dict(rescale_alpha=1.002, rescale_normalize=True, rescale_norm_mode="self",
                    rescale_active_only=True, rescale_refractory=0, rescale_protect_frac=0.20,
                    rescale_cap=64.0)
RESC_UPV = 100            # Trainer.Penalties.UpV, the project's hard constant
RESC_CAP_FR = 0.3         # frm_args.cap_fr in every one of these configs


def rescale_target(frac, n_units):
    """The absolute activity a unit is boosted toward, in the units frm's own statistic uses.

    The rule's target is `rescale_target_frac` times frm's cap, cap_fr * log1p(UpV) / log1p(N) --
    the same expression the firing-rate-magnitude penalty drives activity toward -- so a target
    fraction above 1 asks a dormant unit for MORE activity than frm would.

    Args:
        frac: `rescale_target_frac` from the config; n_units: network size N.
    Returns:
        float target activity, in the units of Trainer.frm_activity_ (a soft maximum of the rate
        over time at tau = 0.1).
    """
    return RESC_CAP_FR * float(np.log1p(RESC_UPV) / np.log1p(n_units)) * float(frac)


def rescale_cells():
    """Every synaptic-rescaling cell on disk, with its knobs and its two measures.

    Walks RESC_SWEEPS, reads each cell's own saved config, and keeps the cells whose
    `reinit_mode` is `rescale` AND whose per-step operation is the rescale one -- the same
    machinery also drives synaptogenesis and disinhibition, which are different interventions and
    are left to the figures that cover them.

    Returns:
        list of dicts, one per cell, each carrying `sweep`, `cell` (the directory), `N`, `iters`,
        `active` (list of per-run active-unit counts from the run's own noise-free participation
        trace), `r2` (list of per-run held-out R^2) and every knob in RESC_MATCHED plus
        `rescale_target_frac`. Cells with no usable run are omitted.
    """
    want = ("rescale_alpha", "rescale_normalize", "rescale_norm_mode", "rescale_active_only",
            "rescale_refractory", "rescale_protect_frac", "rescale_cap", "rescale_target_frac")
    out = []
    for sweep in RESC_SWEEPS:
        for cell in sorted(glob.glob(os.path.join(DATA_DIR, sweep, "*"))):
            cfgs = sorted(glob.glob(os.path.join(cell, "*", "*_config.yaml")))
            if not os.path.isdir(cell) or not cfgs:
                continue
            cfg = OmegaConf.load(cfgs[0])
            pa = cfg.trainer.get("prune_args", {}) or {}
            if str(pa.get("reinit_mode", "")) != "rescale":
                continue
            if str(pa.get("revive_op", "rescale")) != "rescale":
                continue           # synaptogenesis / disinhibition ride the same three-state rule
            act, r2 = _pd_cell(cell)
            if not act:
                continue
            row = dict(sweep=sweep, cell=cell, N=int(cfg.model.N),
                       iters=int(cfg.trainer.max_iter), active=act, r2=r2)
            # the defaults are the ones Trainer.py reads when the key is absent, so a cell that
            # predates a knob is filed under the behaviour it actually had
            dflt = dict(rescale_alpha=1.0005, rescale_normalize=True, rescale_norm_mode="self",
                        rescale_active_only=False, rescale_refractory=0,
                        rescale_protect_frac=0.20, rescale_cap=8.0, rescale_target_frac=0.0)
            for k in want:
                v = pa.get(k, dflt[k])
                row[k] = (bool(v) if isinstance(dflt[k], bool)
                          else type(dflt[k])(v) if not isinstance(dflt[k], str) else str(v))
            out.append(row)
    return out


def _resc_ctrl(n_units):
    """Active units and held-out R^2 per seed for the unpenalised control at one size.

    Args:
        n_units: network size N.
    Returns:
        (active list, r2 list), both empty if that control cell is not on disk.
    """
    return _pd_cell(RESC_CTRL.format(n=n_units))


def rescale_rows_demo(net_dir):
    """What a dormant unit's incoming row is missing, measured on one trained control network.

    The rule has two parts, a multiplicative tilt from inhibition toward excitation and a choice of
    what the row's total length is held at, and the two answer two different deficits. This reports
    both deficits on real weights: the excitatory/inhibitory balance of the incoming recurrent
    weights that arrive FROM FIRING UNITS (the only ones that deliver current, which is what
    `rescale_active_only` restricts the boost to), and the length of that same part of the row.

    Args:
        net_dir: a trained-network folder holding `*LastParams*.npz` and a participation trace.
    Returns:
        (ratio, length, live) -- per-unit E/I ratio, per-unit row length over live columns, and the
        (N,) bool active mask -- or (None, None, None) if the folder is missing either file.
    """
    npz = glob.glob(os.path.join(net_dir, "*LastParams*.npz"))
    tp = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
    if not npz or not tp:
        return None, None, None
    W = np.asarray(np.load(npz[0], allow_pickle=True)["W_rec"], float)
    P = np.asarray(pickle.load(open(tp[0], "rb"))["participation"], float)
    live = np.asarray(P[-1] >= ACTIVE_REL * np.quantile(P[-1], 0.95))
    # the diagonal is excluded: the self-weight is redrawn at the reset and never boosted
    sub = W[:, live] * (~np.eye(W.shape[0], dtype=bool))[:, live]
    E = np.clip(sub, 0, None).sum(axis=1)
    I = -np.clip(sub, None, 0).sum(axis=1)
    return E / np.maximum(I, 1e-12), np.linalg.norm(sub, axis=1), live


def rescale_rule_slide(name="slide_27b_rescale_deficit", n_units=1000):
    """The two deficits a dormant incoming row has, which are the two things the rule sets.

    Args:
        name: output file stem; n_units: control size to measure on.
    Returns:
        the output path, or None if no control network is on disk at that size.
    """
    dirs = sorted(d.rstrip(os.sep) for d in glob.glob(RESC_CTRL.format(n=n_units) + "/*/")
                  if os.path.basename(d.rstrip(os.sep)).split("_")[0] != "nan")
    rows = [rescale_rows_demo(d) for d in dirs]
    rows = [r for r in rows if r[0] is not None]
    if not rows:
        print(f"  SKIP {name}: no control network at N = {n_units}")
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 70 * ps.MM))
    med = []
    for ratio, length, live in rows:
        for j, (m, col, lab) in enumerate(((live, ps.BASE, "active"),
                                           (~live, ps.COND_COL["rescale"], "dormant"))):
            ax.plot(ratio[m], length[m], "o", ms=1.9, color=col, alpha=0.30, mec="none", zorder=3,
                    label=lab if not med else None)
        med.append((np.median(ratio[live]), np.median(length[live]),
                    np.median(ratio[~live]), np.median(length[~live])))
    med = np.asarray(med)
    for j, (col, lab) in enumerate(((ps.BASE, "active"), (ps.COND_COL["rescale"], "dormant"))):
        ax.plot(med[:, 2 * j], med[:, 2 * j + 1], "s", ms=7.0, mfc="none", mec=col, mew=1.5,
                zorder=6)
    ax.set(xscale="log", yscale="log",
           xlabel="excitatory ÷ inhibitory incoming weight, firing sources only",
           ylabel="length of the incoming weight vector,\nfiring sources only")
    ps.ygrid(ax)
    ax.legend(loc="upper left", fontsize=6.2, handlelength=1.2, borderaxespad=0.3,
              markerscale=2.6)
    fig.suptitle(f"Dormant rows sit at balance {np.median(med[:, 2]):.2f} against "
                 f"{np.median(med[:, 0]):.2f}, and are "
                 f"{np.median(med[:, 1]) / np.median(med[:, 3]):.1f}x shorter",
                 fontsize=7.4, color=ps.INK, y=1.015)
    return ps.save(fig, name)


def rescale_target_slide(name="slide_27c_rescale_target", n_units=1000):
    """The dose-response of the activity target: what it recruits and what it costs.

    EVERY RUN IS DRAWN. The target cells carry 3 or 4 seeds each and the spread inside a cell is
    comparable to the spread across the ladder, so a mean with an error bar would claim a
    dose-response the runs do not show.

    The no-target form of the same rule -- boost while the unit is below the silence floor, stop the
    moment it clears it -- is marked separately because it deviates on two further knobs
    (`rescale_cap` 8 rather than 64, and the boost is spent on the whole row rather than on firing
    columns only), so it is not a rung of this ladder.

    Args:
        name: output file stem; n_units: network size to read.
    Returns:
        the output path, or None if no matched target cell is on disk.
    """
    cells = [c for c in rescale_cells() if c["N"] == n_units]
    lad, notgt = {}, {"active": [], "r2": []}
    for c in cells:
        matched = all(c[k] == v for k, v in RESC_MATCHED.items())
        if c["rescale_target_frac"] > 0 and matched:
            lad.setdefault(c["rescale_target_frac"], {"active": [], "r2": []})
            lad[c["rescale_target_frac"]]["active"] += c["active"]
            lad[c["rescale_target_frac"]]["r2"] += c["r2"]
        elif c["rescale_target_frac"] == 0:
            notgt["active"] += c["active"]
            notgt["r2"] += c["r2"]
    if not lad:
        print(f"  SKIP {name}: no matched target cells at N = {n_units}")
        return None
    c_act, c_r2 = _resc_ctrl(n_units)
    col = ps.COND_COL["rescale"]
    fracs = sorted(lad)
    ps.setup()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(ps.W2 * 0.56, 112 * ps.MM), sharex=True)
    for a_, key, ylab in ((ax, "active", f"active units of {n_units}"),
                          (ax2, "r2", "held-out $r^2$")):
        if c_act:
            ref = c_act if key == "active" else c_r2
            a_.axhspan(min(ref), max(ref), color=ps.BASE, alpha=0.16, lw=0, zorder=1)
            a_.axhline(np.mean(ref), color=ps.BASE, lw=1.0, ls=":", zorder=2)
            # named on the line rather than in the legend
            a_.annotate("untouched", (fracs[-1], np.mean(ref)), textcoords="offset points",
                        xytext=(0, 4), ha="right", va="bottom", fontsize=6.2, color=ps.BASE)
        for f in fracs:
            v = lad[f][key]
            a_.plot([f] * len(v), v, "o", ms=3.4, color=col, alpha=0.75, mec="none", zorder=4)
        a_.plot(fracs, [np.mean(lad[f][key]) for f in fracs], "s-", ms=5.2, mfc="none", mec=col,
                color=col, mew=1.2, lw=1.3, zorder=6)
        # the no-target form, off to the left of the ladder and deliberately not joined to it
        if notgt[key]:
            x0 = fracs[0] / 2.2
            a_.plot([x0] * len(notgt[key]), notgt[key], "^", ms=3.8, color=col, alpha=0.75,
                    mec="none", zorder=4)
            a_.plot([x0], [np.mean(notgt[key])], "^", ms=6.5, mfc="none", mec=col, mew=1.2,
                    zorder=6)
        # short here: the suptitle already spells out what the target IS
        # stacked panels share one x axis, so only the lower one is labelled; labelling
        # both put the upper label across the lower panel
        a_.set(xscale="log", ylabel=ylab)
        if a_ is ax2:
            a_.set_xlabel("how much activity the unit has to reach")
        a_.set_xticks([fracs[0] / 2.2] + list(fracs))
        a_.set_xticklabels(["no\ntarget"] + [f"{f:g}" for f in fracs], fontsize=6.2)
        a_.xaxis.set_minor_locator(NullLocator())
        ps.ygrid(a_)
    n_runs = sum(len(lad[f]["active"]) for f in fracs) + len(notgt["active"])
    # TWO ENTRIES, NOT FOUR. The triangle's own x tick already reads "no target", and the grey
    # reference is named on the line it draws, so the legend is down to the two marks that nothing
    # else can label: a run and its cell mean.
    ax.legend(handles=[
        Line2D([], [], ls="none", marker="o", ms=3.6, color=col, label="one run"),
        Line2D([], [], ls="-", marker="s", ms=5.0, mfc="none", mec=col, color=col,
               label="cell mean")],
        loc="lower right", fontsize=6.0, frameon=False, handlelength=1.4, labelspacing=0.3,
        borderaxespad=0.3)
    for a_, letter in ((ax, "(a)"), (ax2, "(b)")):
        a_.set_title(letter, fontsize=7.0, color=ps.INK, loc="left", pad=3)
    # WHAT THE X AXIS IS COMES FIRST. "Activity target" means nothing on its own: it is the level a
    # boosted unit has to reach before the rule lets go of it, and the panel sweeps it. The axis now
    # says that in words, so the two scatters are not of an unnamed quantity.
    # THE TITLE NAMES THE TASK AND THE SIZE AND STOPS. It used to restate the finding -- so many
    # units with no target, so many across a twelve-fold range -- which is what the deck's heading
    # says and what the left panel already draws, with every run shown.
    fig.suptitle("3-bit flip-flop, $N$ = 1000", fontsize=7.4, color=ps.INK, y=1.03)
    return ps.save(fig, name)


def rescale_cost_slide(name="slide_27d_rescale_cost", n_units=1000):
    """Every rescaling run: what it recruited against what it scored, and the same against size.

    THE EXCHANGE RATE IS DRAWN, NOT TITLED. The headline is a slope - how much held-out score each
    100 extra active units cost - and it used to be printed in the figure's title while the panel
    showed an undifferentiated cloud. The slope is now a line through the control's own point, so
    the claim and the picture are the same object.

    THE FAILURES ARE ON THE FIGURE. Five runs destroyed the network, two of them to r2 -114 and
    -23; they used to be mentioned in a sentence under a clipped axis. The left panel is now broken,
    with a strip underneath carrying those runs at their own unit counts, because "without the row's
    total weight held fixed the rule can blow the network up" is a result and belongs in the ink.

    Args:
        name: output file stem; n_units: the size the left panel reads.
    Returns:
        the output path, or None if no rescaling cell is on disk.
    """
    cells = rescale_cells()
    if not cells:
        print(f"  SKIP {name}: no rescaling cells on disk")
        return None
    col = ps.COND_COL["rescale"]
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 66 * ps.MM))
    # the left column is one axes broken in two: the runs that worked, and the strip of runs that
    # destroyed the network. The right column spans both rows.
    # A GAP WIDE ENOUGH TO READ AS A GAP. At hspace 0.14 the two halves of the left column sat a
    # hairline apart and the run at r2 = -114 read as being on the same scale as the run at 0.94.
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.26], hspace=0.42, wspace=0.30,
                  left=0.075, right=0.985, top=0.90, bottom=0.14)
    ax, axb, ax2 = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[:, 1])
    # LEFT: one point per run at N = 1000, marker by the setting that decides whether the rule
    # holds together at all - whether each boosted row's total synaptic weight is held fixed.
    # THE NAMES SAY WHAT HAPPENS TO THE SYNAPSES. "length held" and "length free" are this
    # project's own shorthand for the L2 norm of a unit's incoming weight row, which means nothing
    # to anyone hearing it once. In plain terms: when the rule strengthens a quiet unit's inputs,
    # either the row's total is held, so the boost only re-balances which inputs the unit listens
    # to, or the total is left alone and the weights simply get bigger.
    # The two cannot be labelled in place - the six working "grow" runs sit at 233-301 units,
    # inside the "rebalanced" cloud that spans 202-643 - so this pair stays a two-entry key on the
    # markers. The control is named in place instead.
    kinds = ((True, "o", "input weights rebalanced"), (False, "X", "input weights grow"))
    lo = 0.88
    below = []
    for norm, mk, lab in kinds:
        pts = [(a, r) for c in cells if c["N"] == n_units and c["rescale_normalize"] is norm
               for a, r in zip(c["active"], c["r2"])]
        if not pts:
            continue
        ax.plot([a for a, r in pts if r >= lo], [r for a, r in pts if r >= lo], mk, ms=4.0,
                color=col, alpha=0.8, mec="none", zorder=4, label=lab)
        under = sorted(p_ for p_ in pts if p_[1] < lo)
        # the broken strip, not the main axes' bottom edge: a run drawn at the edge it fell through
        # reads as a run that scored 0.88
        for a, r in under:
            axb.plot([a], [r], mk, ms=4.0, color=col, alpha=0.9, mec="none", zorder=4)
        below += [(a, r, lab) for a, r in under]
    c_act, c_r2 = _resc_ctrl(n_units)
    if c_act:
        # "untouched" in both panels, one word for one thing - and named ON its own markers here,
        # so the key below carries only the two rescaling variants
        ax.plot(c_act, c_r2, "s", ms=4.6, mfc="none", mec=ps.BASE, mew=1.3, zorder=6)
        ax.annotate("untouched", (float(np.max(c_act)), float(np.mean(c_r2))),
                    textcoords="offset points", xytext=(7, 4), ha="left", va="bottom",
                    fontsize=6.4, color=ps.BASE)
    ax.set(ylabel="held-out $r^2$", ylim=(lo, None))
    ax.tick_params(labelbottom=False)
    ax.legend(loc="lower right", fontsize=6.0, handlelength=1.2, borderaxespad=0.3)
    ps.ygrid(ax)
    # RIGHT: against size, at the one target setting that was run at more than one size
    sizes = {}
    for c in cells:
        if all(c[k] == v for k, v in RESC_MATCHED.items()) and c["rescale_target_frac"] == 10.0:
            sizes.setdefault(c["N"], {"active": [], "r2": []})
            sizes[c["N"]]["active"] += c["active"]
            sizes[c["N"]]["r2"] += c["r2"]
    Ns = sorted(sizes)
    for n in Ns:
        ax2.plot([n] * len(sizes[n]["active"]), sizes[n]["active"], "o", ms=3.4, color=col,
                 alpha=0.75, mec="none", zorder=4)
        a_c, _r = _resc_ctrl(n)
        if a_c:
            ax2.plot([n] * len(a_c), a_c, "o", ms=3.4, color=ps.BASE, alpha=0.75, mec="none",
                     zorder=3)
    ax2.plot(Ns, [np.mean(sizes[n]["active"]) for n in Ns], "s-", ms=5.2, mfc="none", mec=col,
             color=col, mew=1.2, lw=1.3, zorder=6)
    ctrl_mean = [np.mean(_resc_ctrl(n)[0]) if _resc_ctrl(n)[0] else np.nan for n in Ns]
    ax2.plot(Ns, ctrl_mean, "s:", ms=5.2, mfc="none", mec=ps.BASE, color=ps.BASE, mew=1.2, lw=1.3,
             zorder=6)
    ax2.set(xscale="log", yscale="log", xlabel="network size $N$", ylabel="active units")
    ax2.set_xticks(Ns)
    ax2.set_xticklabels([str(n) for n in Ns])
    ax2.xaxis.set_minor_locator(NullLocator())
    ax2.set_yticks([200, 300, 400, 600])
    ax2.yaxis.set_major_formatter(ScalarFormatter())
    ax2.yaxis.set_minor_locator(NullLocator())
    # NAMED IN PLACE, NOT IN A LEGEND: two series, two lines, and the names sit on them. The old
    # legend entry also spelled a config knob at the audience ("target = 10x frm's cap").
    if Ns:
        ax2.annotate("with rescaling", (Ns[-1], np.mean(sizes[Ns[-1]]["active"])),
                     textcoords="offset points", xytext=(-4, 7), ha="right", va="bottom",
                     fontsize=6.4, color=col)
        ax2.annotate("untouched", (Ns[-1], ctrl_mean[-1]), textcoords="offset points",
                     xytext=(-4, -8), ha="right", va="top", fontsize=6.4, color=ps.BASE)
    ps.ygrid(ax2)
    # THE BROKEN STRIP. Its scale is the failures' own, two ticks only, and the break is marked on
    # both axes so the gap cannot be read as empty space.
    print(f"  {name}: runs below the axis -> "
          + "; ".join(f"r2 {r:.2f} at {a:.0f} units ({lab.split()[-1]})"
                      for a, r, lab in sorted(below, key=lambda t: t[1])))
    worst = min((r for _a, r, _l in below), default=0.0)
    axb.set_ylim(min(worst * 1.15, -5.0), 1.0)
    axb.set_yticks([0, round(worst / 50.0) * 50 if worst < -50 else round(worst)])
    axb.set_xlabel(f"active units of {n_units}")
    axb.spines["top"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    # ONE x RANGE FOR BOTH HALVES, AND IT REACHES THE NETWORK'S SIZE. The runs that destroyed the
    # network are the ones that recruited almost everything, around 990 units, so a strip inheriting
    # only the working runs' range left three of the five failures outside the figure.
    x_all = [a for c in cells if c["N"] == n_units for a in c["active"]] + list(c_act)
    span = (min(x_all) - 40, max(max(x_all) + 40, n_units * 1.01))
    ax.set_xlim(*span)
    axb.set_xlim(*span)
    # THE BREAK IS THE CONVENTIONAL DOUBLE SLASH, DRAWN IN THE GAP ITSELF, in figure coordinates so
    # both marks are the same physical size whatever the two rows' height ratio is. Two hairline
    # strokes on the spines were what let a reader take the strip for more of the same axis.
    fig.canvas.draw()
    p_top, p_bot = ax.get_position(), axb.get_position()
    y_mid, half = 0.5 * (p_top.y0 + p_bot.y1), 0.5 * (p_top.y0 - p_bot.y1)
    for x_fig in (p_bot.x0, p_bot.x1):
        for d in (-0.005, 0.005):
            fig.add_artist(Line2D([x_fig + d - 0.007, x_fig + d + 0.007],
                                  [y_mid - 0.85 * half, y_mid + 0.85 * half],
                                  color=ps.INK, lw=1.4, solid_capstyle="round", zorder=12))
    if below:
        # the name sits AT the deepest failure, not centred in the strip: two of the five runs in
        # here merely scored badly (0.49, 0.81) and "blew up" is not true of them
        a_w, r_w, _l = min(below, key=lambda t: t[1])
        axb.annotate("network blew up", (a_w, r_w), textcoords="offset points", xytext=(-8, 2),
                     ha="right", va="bottom", fontsize=6.4, color=ps.BAD)
    # THE EXCHANGE RATE, COMPUTED AND THEN DRAWN. Over the matched targeted runs against the
    # control: how much held-out r2 each 100 extra active units costs. The line runs through the
    # control's own point, so the reader sees what the number is a slope of.
    tgt = [c for c in cells if c["N"] == n_units and c["rescale_target_frac"] > 0
           and all(c[k] == v for k, v in RESC_MATCHED.items())]
    t_a = np.mean([a for c in tgt for a in c["active"]])
    t_r = np.mean([r for c in tgt for r in c["r2"]])
    rate = ((np.mean(c_r2) - t_r) / ((t_a - np.mean(c_act)) / 100.0)) if c_act else np.nan
    if np.isfinite(rate):
        # the line stops where the runs that WORKED stop: drawn out to the 990-unit runs it would
        # cross the broken strip, and those runs are not on this axis
        x0 = np.mean(c_act)
        x1 = max((a for c in cells if c["N"] == n_units
                  for a, r in zip(c["active"], c["r2"]) if r >= lo), default=x0 + 100.0)
        xs_r = np.linspace(x0, x1, 2)
        ys_r = np.mean(c_r2) - rate * (xs_r - x0) / 100.0
        ax.plot(xs_r, ys_r, "-", lw=1.8, color=col, alpha=0.9, zorder=7)
        ax.annotate(f"{-rate:+.3f} per 100 units", (xs_r[-1], ys_r[-1]),
                    textcoords="offset points", xytext=(-3, -9), ha="right", va="top",
                    fontsize=6.6, color=col)
    for a_, letter in ((ax, "(a)"), (ax2, "(b)")):
        a_.set_title(letter, fontsize=7.0, color=ps.INK, loc="left", pad=3)
    fig.suptitle("3-bit flip-flop", fontsize=7.4, color=ps.INK, y=0.985)
    return ps.save(fig, name)


def readout_line(iters, recorded=None):
    """The 'read at ...' sentence a panel carries, from where its cells were actually read.

    Derived rather than typed, so it cannot drift from the data: a family read at one shared probe
    gives one number, a family whose cells land on different probes gives the range, and a family
    whose surviving data record no iteration says so rather than borrowing the sweep's nominal
    budget silently. The range is given bare - why the cells differ is a fact about the sweeps, not
    something a reader of this panel needs on the axes.

    Args:
        iters: set of iterations the family's cells were read at, possibly empty;
        recorded: the iteration from ARCHIVE_READOUT, used when the surviving summary carries no
            iteration column. It is a verified value, not a guess - see ARCHIVE_READOUT.
    Returns:
        a one-sentence string, always ending in a full stop.
    """
    if iters:
        lo, hi = min(iters), max(iters)
        return f"Read at {lo:,} iterations." if lo == hi else \
               f"Read at {lo:,}–{hi:,} iterations."
    if recorded is not None:
        # NO HEDGE ON THE NUMBER. It used to read "(recorded, not in the archived summary)", which
        # tells a reader the value is second-hand without telling them it is checked. ARCHIVE_READOUT
        # verifies every one of these against a source that does carry the iteration - the v2 CSV's
        # own iters column, and the surviving run folders' max_iter - so the figure states it plainly
        # and the provenance stays in the comment above ARCHIVE_READOUT.
        return f"Read at {recorded:,} iterations."
    return "⚠️ Read-out iteration not recorded."


# ⚠️ THE SEVENTH FIELD IS THE WHOLE FIGURE TITLE and it names the task and the size, nothing more;
# the read-out iteration is appended from the data. The CLAIM is the slide's heading in
# presentation.md, not a caption over the axes - a panel that argues with the reader before they
# have looked at it is a worse panel, and these titles had grown to three lines of prose restating
# what the axes already show.
INTERVENTIONS = [
    ("slide_x_activation_cddm", "A different activation does not help",
     "CDDM, 200k", "activation", "ReLU (default)", 0,
     "CDDM, $N$ = 1000"),
    ("slide_x_weightdecay", "Weight decay makes it monotonically worse",
     "CDDM, 200k", "weight decay", "the default", 1,
     "CDDM, $N$ = 1000"),
    ("slide_x_activation_ff", "Nor on the other task",
     "3-bit flip-flop, 150k", "activation", "ReLU (default)", 0,
     "3-bit flip-flop, $N$ = 1000"),
    # The reference is the DEFAULT DRAW, whose rows sit at norm 0.050 at N = 1000 - the lowest rung
    # of this ladder, not a middle one, which is why ref_at is 0. Labelling it "×1" and putting it
    # second (every version before 2026-10-01) was what made this panel look non-monotone.
    ("slide_x_inputscale", "Scaling the input weights up adds 40-75 units of 1000, peaking at row norm 2",
     "3-bit flip-flop, 150k", "input scale", "row norm 0.05 (default draw)", 0,
     "3-bit flip-flop, $N$ = 1000"),
    ("slide_x_metabolic", "The field-standard metabolic penalty moves nothing beyond seed scatter",
     "CDDM, 30k (archived)", "metabolic", "$\\lambda$ = 0 (default)", 0,
     "CDDM, $N$ = 1000"),
    ("slide_x_architecture", "Nor does the equation form, nor a trainable bias",
     "CDDM, 30k (archived)", "architecture", "standard", 0,
     "CDDM, $N$ = 1000"),
    ("slide_x_recnoise", "Removing recurrent noise is the largest effect, and it is negative",
     "CDDM, 30k", "noise", "$\\sigma$ = 0.05 (default)", 2,
     "CDDM, $N$ = 1000"),
]



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
    """Held-out R^2, and active-unit count, for one input-scale rung.

    ⚠️ R^2 IS NO LONGER THE NOISE-FREE PROBE AT `at_iter`. It is heldout_r2 on the run's saved
    parameters: a held-out batch at the sigma_rec and sigma_inp the network trained with,
    sigma_w = 0, nine draws. That is the project's one performance measure, so this panel can be set
    beside any other. The consequence worth knowing is that R^2 is now read at the END of training
    while the count is still read at `at_iter`; on this ladder the two coincide, since every rung
    runs to the same budget. The active count stays noise-free, as every count here does.

    Diverged runs (`nan_` prefix) are dropped, as everywhere else in this project.

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
        active.append(active_count(p, "scalefree"))
        r2.append(heldout.heldout_r2(os.path.dirname(f)))
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
    pats = [pat for (lab, pat, _c), c in zip(INPUTSCALE_LADDER, cells) if len(c[2])]
    sc, handles = _dims_scatter(ax, drawn, pats)
    if sc is None:
        for lab, col, r2, active, _ in drawn:
            ax.plot(active, r2, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                    label=f"{lab}  ({len(r2)})")
            ax.plot(active.mean(), r2.mean(), "s", ms=7.0, color=col, mec="white", mew=1.1,
                    zorder=4)

    every_r2 = np.concatenate([c[2] for c in drawn])
    every_active = np.concatenate([c[3] for c in drawn])
    mid = 0.5 * (every_r2.min() + every_r2.max())
    ax.set_ylim(mid - 0.01, mid + 0.01)
    ax.set_xlabel("active units of 1000  (scale-free rule, $p \\geq 0.05\\,q_{95}(p)$)")
    ax.set_ylabel("held-out task $R^2$")
    ax.set_title(f"3-bit flip-flop, $N$ = 1000. {readout_line(probes)}",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ax.legend(handles=handles or None, loc="lower right", fontsize=5.8, handlelength=1.0,
              borderpad=0.2, borderaxespad=0.3, ncol=2)
    ps.ygrid(ax)
    if sc is not None:
        _dims_colorbar(fig, ax, sc)
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
                label=f"{ARM_SAYS.get(full, full)} ({m.sum()})")
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
    ("none", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=0_iters=200000", ps.SLOTS[2]),
    ("the default", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000", ps.BASE),
    (r"10$\times$ it", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-5_iters=200000",
     ps.SLOTS[1]),
    (r"100$\times$ it", f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-4_iters=200000",
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


def cddm_activation_cell(pattern, at_iter):
    """Held-out R^2, and active-unit count, for one CDDM arm.

    R^2 is heldout_r2 on each net's saved parameters - a held-out batch (coherences interleaved with
    the trained grid), sigma_rec and sigma_inp as trained, sigma_w = 0, nine draws. It replaces an
    offline NOISE-FREE re-score, which mattered here more than anywhere: the noise penalty on CDDM
    is activation-dependent, so the two instruments do not merely shift this ladder, they reorder
    it. The count is the scale-free rule at the participation probe nearest `at_iter`, noise-free as
    always. Diverged runs (`nan_` prefix) are dropped.

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
        r2.append(heldout.heldout_r2(folder.rstrip(os.sep)))
    return np.array(r2, float), np.array(active, float), probes


# ---- condition as shape, dimensionality as colour ----------------------------------------------
# Slides 10b, 11b and 12b each ask whether a ladder's extra units were bought with performance, and
# the answer on all three is "no, within the seed scatter". That leaves the colour channel free, so
# it carries a third measure instead of repeating the legend: the population dimensionality of each
# network, the participation ratio of its noise-free rate covariance over its active units.
# Condition moves to marker SHAPE, which orders nothing - the ladders' order is on the legend, not
# in the hues.
DIMS_SHAPES = ("o", "s", "^", "D", "v", "P", "X")


def _cell_dims(pattern):
    """Population dimensionality per run of one cell, in the order the cell loaders return them.

    The three cell loaders behind these panels - `cddm_activation_cell`, `winp_cell` and `met_cell`
    - all walk `sorted(glob(<cell>/*/*ParticipationTrace.pkl))` and drop `nan_` runs, so repeating
    that walk here yields one dimensionality per run IN THE SAME ORDER as their r2 and active
    arrays. The caller checks the lengths rather than trusting that.

    Args:
        pattern: glob matching one cell's run folders, as the ladder spells it.
    Returns:
        (n_runs,) float array of participation-ratio dimensionality, nan where a run cannot be
        rebuilt.
    """
    out = []
    for f in sorted(glob.glob(os.path.join(pattern, "*", "*ParticipationTrace.pkl"))):
        if os.path.basename(os.path.dirname(f)).split("_")[0] == "nan":
            continue
        out.append(RD.dims(os.path.dirname(f)))
    return np.asarray(out, float)


def _dims_scatter(ax, drawn, patterns):
    """Draw the R2-against-active-units scatter with condition as shape and dimensionality as hue.

    Args:
        ax: the axes to draw on;
        drawn: list of (label, colour, r2, active, probes) - the colour is ignored here, since hue
            now carries dimensionality;
        patterns: the ladder's globs, parallel to `drawn`.
    Returns:
        (mappable, handles): the scatter to hang a colour bar on and the legend handles naming each
        shape, or (None, handles) if no run could be measured - in which case nothing is coloured
        and the caller should fall back.
    """
    dims = [_cell_dims(p) for p in patterns]
    for (lab, _c, r2, _a, _p), dv in zip(drawn, dims):
        if len(dv) != len(r2):
            print(f"  WARN {lab}: {len(dv)} dimensionalities for {len(r2)} runs - not coloured")
            return None, []
    allv = np.concatenate([d[np.isfinite(d)] for d in dims]) if dims else np.array([])
    if not len(allv):
        return None, []
    vmin, vmax = float(allv.min()), float(allv.max())
    sc, handles = None, []
    for (lab, _c, r2, active, _p), dv, mk in zip(drawn, dims, DIMS_SHAPES):
        sc = ax.scatter(active, r2, c=dv, cmap="viridis", vmin=vmin, vmax=vmax, marker=mk, s=30,
                        edgecolors="white", linewidths=0.5, zorder=3)
        # the cell mean keeps the cell's SHAPE and is drawn hollow in ink: a filled mean would take
        # a colour from the same scale and read as a network with that dimensionality
        ax.plot(active.mean(), r2.mean(), mk, ms=9.0, mfc="none", mec=ps.INK, mew=1.2, zorder=5)
        handles.append(Line2D([], [], ls="none", marker=mk, mfc=ps.FAINT, mec=ps.INK, mew=0.7,
                              ms=5.0, label=f"{lab}  ({len(r2)})"))
    return sc, handles


def _dims_colorbar(fig, ax, sc):
    """Hang a slim dimensionality colour bar on the right of one panel.

    Args:
        fig: the figure; ax: the panel; sc: the mappable `_dims_scatter` returned.
    Returns:
        the colour bar.
    """
    cb = fig.colorbar(sc, ax=ax, pad=0.015, fraction=0.045, aspect=26)
    cb.set_label("dimensionality  (participation ratio of the\nactive units' rate covariance)",
                 fontsize=6.0, color=ps.INK, linespacing=1.3)
    cb.ax.tick_params(labelsize=5.8)
    cb.outline.set_visible(False)
    return cb


def activation_r2_slide(name, ladder, cell_fn, at_iter, task_line, n_units=1000,
                        headline=None, by_dims=False):
    """Task R^2 against active units for every net of one activation family, one colour per arm.

    Args:
        name: output file stem; ladder: list of (label, glob, colour);
        cell_fn: callable(pattern, at_iter) -> (r2, active, probes) per seed;
        at_iter: the iteration the count is read at; task_line: the subtitle naming the task and the
            conditions, WITHOUT a read-out iteration - that sentence is derived from the probes the
            cells actually landed on, so it cannot drift from the data;
        n_units: network size, for the x axis label;
        headline: accepted and no longer drawn - the claim belongs to the slide's heading, not over
            the axes. Kept so existing callers need not change;
        by_dims: True draws condition as marker SHAPE and population dimensionality as hue, with a
            colour bar. Off by default: the activation panels this also draws are categorical with
            no third measure to carry, and four hues name four activations perfectly well.
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
    sc, handles = (None, [])
    if by_dims:
        pats = [pat for (lab, pat, _c), c in zip(ladder, cells) if len(c[2])]
        sc, handles = _dims_scatter(ax, drawn, pats)
    if sc is None:
        for lab, col, r2, active, _ in drawn:
            ax.plot(active, r2, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                    label=f"{lab}  ({len(r2)})")
            # A SQUARE, not a large translucent circle. metabolic_r2_slide settled this: the mean
            # differs from a seed by SHAPE, not by opacity, because a translucent dot reads as
            # blurred or as less certain and a mean over three seeds is neither. One convention
            # across the scatter panels.
            ax.plot(active.mean(), r2.mean(), "s", ms=7.0, color=col, mec="white", mew=1.1,
                    zorder=4)
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
    ax.set_ylabel("held-out task $R^2$")
    note = ("\n" + "; ".join(missing) + (" is" if len(missing) == 1 else " are")
            + " still training") if missing else ""
    # THE TITLE NAMES THE TASK, THE SIZE AND THE READ-OUT, and stops. The spread of the data is on
    # the axes, and the claim is the slide's heading in presentation.md.
    ax.set_title(f"{task_line} {readout_line(probes)}" + note,
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ax.legend(handles=handles or None, loc="lower right", fontsize=5.8, handlelength=1.0,
              borderpad=0.2, borderaxespad=0.3, ncol=2)
    ps.ygrid(ax)
    if sc is not None:
        _dims_colorbar(fig, ax, sc)
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
MET_LADDER = [
    ("$\\lambda$ = 0 (untouched)", f"{DATA_DIR}/CDDM_std_g0/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
     ps.BASE, "0"),
    ("$\\lambda$ = 0.01", MET_CELL.format(lam="0.01"), ps.SLOTS[0], "0.01"),
    ("$\\lambda$ = 0.1", MET_CELL.format(lam="0.1"), ps.SLOTS[1], "0.1"),
    ("$\\lambda$ = 1", MET_CELL.format(lam="1.0"), ps.SLOTS[2], "1.0"),
    ("$\\lambda$ = 10", MET_CELL.format(lam="10.0"), ps.SLOTS[3], "10.0"),
]


def met_cell(cell_dir, lam):
    """Held-out R^2, and active-unit count, for one metabolic cell.

    R^2 is heldout_r2: a held-out batch at the sigma_rec and sigma_inp the network trained with,
    sigma_w = 0, nine draws. It replaces a noise-free cache (`metabolic_clean_r2.npz`), which is no
    longer read.

    ⚠️ THE CAVEAT THE NOISE-FREE CACHE EXISTED FOR HAS NOT GONE AWAY, so read this ladder with it in
    mind: a metabolic penalty shrinks the rate scale about sixfold while the injected noise stays at
    a fixed 0.05, so signal-to-noise falls with lambda for reasons that have nothing to do with the
    task. A fixed-sigma evaluation therefore charges the high-lambda rungs for their own rate scale.
    The project reports one measure everywhere and this is it; the alternative was an instrument
    that reorders arms, which is worse. Where the lambda = 10 rung is quoted, quote this mechanism
    with it.

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
    r2, active, last = [], [], None
    for f in sorted(glob.glob(os.path.join(cell_dir, "*", "*ParticipationTrace.pkl"))):
        head = os.path.basename(os.path.dirname(f)).split("_")[0]
        if head == "nan":
            continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        active.append(active_count(np.asarray(d["participation"], float)[-1], "scalefree"))
        r2.append(heldout.heldout_r2(os.path.dirname(f)))
        last = int(np.asarray(d["iters"])[-1])
    return np.asarray(r2, float), np.array(active, float), last


def metabolic_ruler_slide(name="slide_x_metabolic_ruler"):
    """Why the metabolic penalty's active count is flat, drawn rather than asserted.

    THE COUNT IS FLAT AND THE EFFECT IS REAL, AND THOSE ARE NOT IN TENSION. Every active-unit count
    in this project is read against a bar set at 5% of that network's OWN 95th-percentile
    participation. The rule has no absolute scale, which is what makes a count at N = 500 comparable
    with one at N = 4000 and a flip-flop comparable with a CDDM. A rate penalty moves the scale
    itself: measured here, the 95th percentile falls sixfold across the ladder, the bar falls with
    it, and the count stays where it was. Hold the bar at the unpenalised control's value and the
    same networks lose two thirds of their units.

    TWO PANELS BECAUSE IT IS TWO FACTS. The top shows the two counts diverging; the bottom shows the
    ruler doing the shrinking. Read together they say the units were turned DOWN rather than killed,
    which is a different claim from "the penalty does nothing" - and the deck used to make the wrong
    one, because the moving bar divided the effect out before anyone saw it.

    ⚠️ THE CONTROL IS THE MATCHED UNPENALISED CELL of this same 30,000-iteration sweep, not the
    200,000-iteration drift control that other CDDM panels use. At 30,000 iterations it holds about
    414 active units; the same architecture at 200,000 holds about 272. Compare within this panel.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if fewer than three rungs are on disk.
    """
    rows = []
    for lab, cell, _, _ in MET_LADDER:
        pv = []
        for f in sorted(glob.glob(os.path.join(cell, "*", "*ParticipationTrace.pkl"))):
            if os.path.basename(os.path.dirname(f)).split("_")[0] == "nan":
                continue
            try:
                d = pickle.load(open(f, "rb"))
            except Exception:
                continue
            pv.append(np.asarray(d["participation"], float)[-1])
        if pv:
            rows.append((lab, pv))
    if len(rows) < 3:
        print(f"  SKIP {name}: only {len(rows)} metabolic cells on disk")
        return None

    # the fixed bar IS the control's own bar, so the two curves differ in exactly one thing
    bar = float(np.mean([0.05 * np.quantile(p, 0.95) for p in rows[0][1]]))
    xs = np.arange(len(rows))
    moving = [np.asarray([active_count(p, "scalefree") for p in pv], float) for _, pv in rows]
    fixed = [np.asarray([float((p >= bar).sum()) for p in pv], float) for _, pv in rows]
    q95 = [np.asarray([float(np.quantile(p, 0.95)) for p in pv], float) for _, pv in rows]

    ps.setup()
    fig = plt.figure(figsize=(W, 86 * ps.MM))
    gs = GridSpec(2, 1, figure=fig, height_ratios=[1.0, 0.5], hspace=0.16)
    ax, ax2 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1])

    # NEITHER CURVE IS THE REFERENCE. Both are the same networks counted two ways, so the moving
    # bar had no claim on ps.BASE (paperstyle: the reference is not a series) and the fixed bar no
    # claim on the red that means the rate penalty. KNOB_ACCENT is this deck's own scale-free count,
    # the quantity slide_x_metabolic plots; COUNT_ALT is the fixed-bar count, and the lower panel
    # keeps KNOB_ACCENT because the bar it draws is the one that moves.
    ps.strip(ax, xs, moving, [KNOB_ACCENT] * len(rows), rng=np.random.default_rng(12))
    ps.strip(ax, xs, fixed, [COUNT_ALT] * len(rows), rng=np.random.default_rng(13))
    ax.plot(xs, [v.mean() for v in moving], "-", lw=1.3, color=KNOB_ACCENT, zorder=2)
    ax.plot(xs, [v.mean() for v in fixed], "-", lw=1.3, color=COUNT_ALT, zorder=2)
    # LABELLED IN PLACE, not in a legend: two series, and the words belong on the lines they name
    ax.annotate("bar moves with the network", (xs[-1], moving[-1].mean()),
                textcoords="offset points", xytext=(-6, 26), ha="right", fontsize=7.2,
                color=KNOB_ACCENT)
    ax.annotate("bar held fixed", (xs[-1], fixed[-1].mean()), textcoords="offset points",
                xytext=(-6, -16), ha="right", fontsize=7.2, color=COUNT_ALT)
    ax.set(ylabel="active units of 1000", ylim=(0, 540))
    ax.set_xticks(xs)
    ax.tick_params(labelbottom=False)
    ps.ygrid(ax)

    ps.strip(ax2, xs, q95, [KNOB_ACCENT] * len(rows), rng=np.random.default_rng(14))
    ax2.plot(xs, [v.mean() for v in q95], "-", lw=1.3, color=KNOB_ACCENT, zorder=2)
    ax2.set(ylabel="where the bar sits", yscale="log")
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["none" if i == 0 else lab.split("= ")[-1]
                         for i, (lab, _) in enumerate(rows)])
    ax2.set_xlabel("metabolic penalty, dose")
    ps.ygrid(ax2)
    fig.suptitle("CDDM, $N$ = 1000, 30,000 iterations", fontsize=7.4, color=ps.INK, y=0.97)
    return ps.save(fig, name)


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
    pats = [pat for (lab, pat, _c, _l), c in zip(MET_LADDER, cells) if len(c[2])]
    sc, handles = _dims_scatter(ax, drawn, pats)
    if sc is None:
        for lab, col, r2, active, _ in drawn:
            ax.plot(active, r2, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                    label=f"{lab}  ({len(r2)})")
            ax.plot(active.mean(), r2.mean(), "s", ms=7.0, color=col, mec="white", mew=1.1,
                    zorder=4)

    ax.set_xlabel("active units of 1000  (scale-free rule, $p \\geq 0.05\\,q_{95}(p)$)")
    ax.set_ylabel("held-out task $r^2$")
    every_r2 = np.concatenate([c[2] for c in drawn])
    probe = f"{sorted(probes)[0]:,}" if len(probes) == 1 else "the last probe"
    # The axis keeps a fixed window ABOVE the data rather than cropping to it, so a 0.05 spread
    # does not fill the canvas and read as a catastrophe. It is anchored to the data, not typed: the
    # old hard-coded (0.88, 1.00) silently emptied this panel the moment the measure changed from the
    # noise-free probe to the held-out score, which sits ~0.12 lower on CDDM.
    ax.set_ylim(min(every_r2.min() - 0.01, every_r2.max() - 0.12), every_r2.max() + 0.015)
    lo = np.mean([c[2] for c in drawn if c[0].endswith("= 10")][0])
    ref = np.mean([c[2] for c in drawn if "untouched" in c[0]][0])
    ax.set_title(f"CDDM, $N$ = 1000. Read at 30,000 iterations (last probe {probe}).",
                 fontsize=7.4, color=ps.INK, linespacing=1.35, pad=6)
    ax.legend(handles=handles or None, loc="lower right", fontsize=5.8, handlelength=1.0,
              borderpad=0.2, borderaxespad=0.3, ncol=2)
    ps.ygrid(ax)
    if sc is not None:
        _dims_colorbar(fig, ax, sc)
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
    # ONE COLUMN, so the three tasks stack and their shared participation axis lines up; side
    # by side they were three pictures read left to right, each with its own y range.
    fig, axes = plt.subplots(len(have), 1, figsize=(96 * ps.MM, 44 * len(have) * ps.MM),
                             sharex=True)
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
    # so the gloss is dropped - the deck defines participation and the criterion once, in text,
    # under the figure. STACKED, the x label belongs to the BOTTOM panel: on the middle one it
    # landed under a panel that still has another below it, printing straight through that panel's
    # title. Every panel keeps its y label, since each has its own count range.
    for i, ax in enumerate(axes):
        for t in list(ax.texts):
            if "rate moves over a trial" in t.get_text():
                t.remove()
        if i != len(axes) - 1:
            ax.set_xlabel("")
    fig.subplots_adjust(left=0.165, right=0.98, top=0.90, bottom=0.085, hspace=0.62)
    return ps.save(fig, stem, w_mm=96)


# ---- the read-out rule, shown on two tasks -----------------------------------------------------
# ONE RULE, TWO TASK FAMILIES, LOGGED DIFFERENTLY. The flip-flop (N, k) grid lives in pr_matrix's own
# loader at one clean-loss sample per 10 iterations; CDDM lives in Figure 1's scaling table at one
# per iteration. Both go through excess_time_at, which takes the iterations rather than assuming a
# spacing. Handing CDDM to the PROBE-based excess_time would place its read-out 10x too late and
# smooth its loss over 21 iterations instead of 210, which on a jagged trace means the threshold is
# crossed on a noise dip.
READOUT_TASKS = ["3-bit flip-flop", "CDDM"]
# SIZE IS ORDERED, so it is one hue in four values, dark for big (see SIZE_RAMP). It used to be
# ps.BASE for N = 500 and three categorical slots for the rest, which said "four conditions" and
# spent the reference ink on the smallest network. N = 4000 (the flip-flop) and N = 5000 (CDDM) are
# each the largest size of their own task, so they share the darkest value.
READOUT_SIZE_COLOUR = {500: SIZE_RAMP[0], 1000: SIZE_RAMP[1], 2000: SIZE_RAMP[2],
                       4000: SIZE_RAMP[3], 5000: SIZE_RAMP[3]}


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

    THE SIZES ARE NAMED AT THE ENDS OF THEIR OWN CURVES, and the read-out iterations are not on this
    panel at all - slide_05 is the panel about them. On the flip-flop all four sizes reach floors
    within 2% of each other, so the markers nearly coincide and nothing can be labelled there; the
    curve ends are separated by budget instead, and the two sizes that share a budget are written
    one above and one below the line.

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
    # STACKED, NOT SIDE BY SIDE. The panels share a quantity on both axes (iteration, loss) but not
    # a range, and read one above the other the two tasks' crossings line up vertically against the
    # same iteration decade, which is what the slide is comparing.
    fig, axes = plt.subplots(len(data), 1, figsize=(W, 112 * ps.MM),
                             gridspec_kw=dict(hspace=0.42))
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
            sm = r["sm"] = F1.running_median(r["loss"])     # kept: the end labels read it back
            ax.plot(r["it"], sm, lw=0.7, color=c, alpha=0.5, zorder=3)
            j = np.searchsorted(r["it"], x0)
            if j < len(sm):
                top = max(top, float(np.nanmax(sm[j:j + 50])))
        for n in Ns:
            g = [r for r in runs if r["N"] == n]
            c = READOUT_SIZE_COLOUR.get(n, ps.MUTED)
            fl = float(np.mean([r["floor"] for r in g]))
            T = float(np.mean([r["T"] for r in g]))
            ax.axhline(fl, color=c, lw=0.5, ls=":", zorder=2)
            ax.axhline(fl * (1 + PR.EXCESS_DELTA), color=c, lw=0.7, ls="--", zorder=2)
            ax.plot([T], [fl * (1 + PR.EXCESS_DELTA)], "o", ms=5.0, color=c, mec="white", mew=0.8,
                    zorder=6)
        ends = []
        for n in Ns:
            r = max((q for q in runs if q["N"] == n), key=lambda q: q["it"][-1])
            ends.append((float(r["it"][-1]), float(r["sm"][-1]), n))
        lo = min(r["floor"] for r in runs)
        ax.set(xscale="log", yscale="log", xlabel="training iteration",
               xlim=(x0, max(r["it"][-1] for r in runs)), ylim=(lo * 0.85, top * 1.6))
        # EACH SIZE IS WRITTEN AT THE END OF ITS OWN CURVE. Two four-entry legends restating
        # "N = ... , read at ...k" asked the audience to hold a colour key while reading a loss
        # trace, and the read-out iteration is slide_05's whole subject. The ramp orders the sizes
        # and the label sits on the curve it names. Sizes trained to the same budget end at the same
        # place, so a label landing within a label's width of the one before it is stacked above it
        # - above, because the curves have converged to the floor by then and everything over them is
        # empty, while below them is the axis edge. The test is in display pixels, which is why this
        # runs after the axis has its limits.
        prev_px, stack = None, 0
        for xe, ye, n in sorted(ends):
            px = float(ax.transData.transform((xe, ye))[0])
            stack = stack + 1 if (prev_px is not None and px - prev_px < 50) else 0
            ax.annotate(f"$N$ = {n}", (xe, ye), textcoords="offset points",
                        xytext=(-3, 7 + 11 * min(stack, 3)), ha="right", va="bottom", fontsize=6.6,
                        color=READOUT_SIZE_COLOUR.get(n, ps.MUTED))
            prev_px = px
        ax.set_title(task, fontsize=7.4, color=ps.INK, pad=5)
        ps.ygrid(ax)
    for a_ in axes:
        a_.set_ylabel("noise-free task loss")
    for a_ in axes[:-1]:
        a_.set_xlabel("")
    # NO SENTENCE UNDER THE PANELS. The 20 words that ran edge to edge there - the rule, and what the
    # dotted line, the dashed line and the dot each are - are the words the speaker says while this
    # slide is up, and they are written in the deck beside it.
    return ps.save(fig, "slide_06_readout_rule")


def scaling_fraction_slide(name="slide_06_scaling_fraction"):
    """The scaling claim as the quantity it is about: the FRACTION of units that work, against N.

    ⚠️ WHY THIS PANEL EXISTS, AND WHY THE ABSOLUTE ONE WAS MISREAD. `F1.panel_c` plots the absolute
    active COUNT against N, with a red dash-dot rule at y = 1,000 labelled "1,000 active units" --
    an absolute milestone answering "how big must N be before 1,000 units work?". Every other panel
    in this deck is an N = 1000 network and draws its horizontal rule at 1,000 to mean THE CEILING,
    every unit active. A viewer taught that meaning by six panels reads this one's rule the same
    way, sees the CDDM series rise to meet it, and concludes CDDM nearly reaches every unit active.
    Pavel read it exactly that way. The series is correct -- CDDM is 327 / 477 / 621 / 945 active at
    N = 500 / 1000 / 2000 / 5000 -- but that is 65% / 48% / 31% / 19% of N, so the emptiest point on
    the curve is the one that looks closest to the ceiling. The colours make it worse: the rule is
    ps.BAD and the CDDM series ps.SLOTS[1], two reds a projector will not separate.

    THE CLAIM IS ABOUT THE FRACTION, SO THE PANEL PLOTS THE FRACTION. "Bigger networks are emptier"
    is then a falling curve with nothing to misread, and the ceiling is 100% by construction.
    `F1.panel_c` is untouched: the manuscript wants the absolute count and its exponent.

    THE NUMBERS ARE THE MANUSCRIPT'S OWN. Every count comes from `F1.at_own_floor` over
    `F1.SCALING`, the same table and the same read-out rule panel_c uses, so the two panels cannot
    disagree about anything but the axis.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if fewer than two sizes are readable.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 74 * ps.MM))
    drew = False
    x_right = max(max(v[0]) for v in F1.SCALING.values())
    for task, (Ns, pats, cap, col) in F1.SCALING.items():
        min_r2 = F1.TASK_MIN_R2.get(task)
        cells = {}
        for N in Ns:
            rs = F1.runs_with_loss(pats[N], min_r2=min_r2)
            if rs:
                cells[N] = rs
        if len(cells) < 2:
            continue
        read, _ = F1.at_own_floor(cells)
        pts = sorted((N, np.asarray(v[0], float)) for N, v in read.items())
        if len(pts) < 2:
            continue
        drew = True
        xs = [N for N, _ in pts]
        fr = [100.0 * c.mean() / N for N, c in pts]
        lo = [100.0 * c.min() / N for N, c in pts]
        hi = [100.0 * c.max() / N for N, c in pts]
        ax.plot(xs, fr, "-o", lw=1.4, ms=4.5, color=col, zorder=4)
        ax.fill_between(xs, lo, hi, color=col, alpha=0.16, lw=0, zorder=2)
        # Named at the end of its own line, so the panel needs no legend. A series that stops short
        # of the right edge is named BELOW its last point: labelled to the right, DMTS at N = 2000
        # ran its text across the two flip-flop lines.
        at_edge = xs[-1] >= x_right
        ax.annotate(task, (xs[-1], fr[-1]), textcoords="offset points",
                    xytext=(6, 0) if at_edge else (0, -11),
                    ha="left" if at_edge else "center",
                    va="center" if at_edge else "top", fontsize=6.6, color=col)
    if not drew:
        print(f"  SKIP {name}: no task had two readable sizes")
        plt.close(fig)
        return None
    ax.axhline(100, color=ps.FAINT, lw=0.8, ls=":", zorder=1)
    ax.annotate("every unit works", (ax.get_xlim()[0], 100), textcoords="offset points",
                xytext=(4, 3), fontsize=6.2, color=ps.MUTED)
    ax.set(xscale="log", yscale="log", xlabel="network size $N$",
           ylabel="units that work\n(% of $N$, read at 1.07$\\times$ each run's own loss floor)",
           ylim=(8, 150), xlim=(380, 9000))
    ax.set_xticks([500, 1000, 2000, 5000])
    ax.set_xticklabels(["500", "1,000", "2,000", "5,000"])
    ax.set_yticks([10, 20, 50, 100])
    ax.set_yticklabels(["10%", "20%", "50%", "100%"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.yaxis.set_minor_locator(NullLocator())
    ps.ygrid(ax)
    return ps.save(fig, name)


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
SW_RE = re.compile(r"sw=?([0-9.]+)")   # `sw=1.0` and `sw1.0` are both spelled on disk


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

    top = rungs[int(np.argmax([r[1]["n_active"].mean() for r in rungs]))]
    # The title carries the dataset; both series are drawn, so restating where they start and end
    # spends three lines on what the reader is already looking at.
    fig.suptitle(
        f"3-bit flip-flop, $N$ = {n_units}, "
        f"{min(len(r[1]['n_active']) for r in rungs)}-{max(len(r[1]['n_active']) for r in rungs)}"
        " seeds per rung, every seed drawn",
        fontsize=7.6, color=ps.INK, linespacing=1.4, y=1.02)
    # NO FOOTNOTE. The two lines that used to sit here - how an active unit is counted, and at which
    # rungs the two curves open up by more than 0.05 - were half this panel's words and neither is
    # something to look at: the counting rule is the deck's own and stated once when it is
    # introduced, and the gap is drawn, filled, and labelled on the curves it separates.
    return ps.save(fig, name, w_mm=110)


# Four measures against size for one rung of the ladder. Sizes where only ONE arm exists are
# dropped: the control has an N = 4000 cell and the synaptic-noise arm does not, and a lone control
# point at the right edge invites an extrapolation the data do not carry.
SW_MEASURES = [("n_active", "active units", True),
               ("r2_common", "$R^2$ at $\\sigma_w$ = 0", False),
               ("dims", "dimensionality\n$(\\sum_i\\lambda_i)^2 / \\sum_i\\lambda_i^2$", False),
               ("w_sigma_log", "sd of $\\ln|W_{rec}|$", False)]


def synnoise_size_slide(name="slide_26b_synnoise_size", sigma_w=1.0, task="NBitFlipFlop"):
    """One claim: at sigma_w = 1 every measure moves the same way at every size, and the unit
    multiplier is largest in the middle.

    Args:
        name: output file stem; sigma_w: the rung to follow; task: task name as the cache records it.
    Returns:
        the output path, or None if fewer than two sizes carry both arms.
    """
    c = F2.load()
    is_sw = np.array([bool(m) and float(m.group(1)) == sigma_w
                      for m in (SW_RE.search(str(x)) for x in c["cell"])])
    base = c["task"] == task
    sizes = sorted(set(c["N"][base & (c["arm"] == "synnoise") & is_sw].tolist())
                   & set(c["N"][base & (c["arm"] == "control")].tolist()))
    if len(sizes) < 2:
        print(f"  SKIP {name}: {len(sizes)} sizes carry both arms")
        return None
    col = ps.COND_COL["synnoise"]
    arms = [("control", f"no synaptic noise", ps.BASE),
            ("synnoise", rf"$\sigma_w$ = {sigma_w:g}", col)]

    def vals(a, n, key):
        m = base & (c["N"] == n) & (c["arm"] == a) & (is_sw if a == "synnoise" else True)
        return np.asarray(c[key][m], float)

    ps.setup()
    fig = plt.figure(figsize=(92 * ps.MM, 168 * ps.MM))
    gs = GridSpec(4, 1, figure=fig, hspace=0.12,
                  left=0.21, right=0.97, top=0.92, bottom=0.055)
    rng = np.random.default_rng(0)
    axes, ratios = [], []
    for i, (key, ylab, logy) in enumerate(SW_MEASURES):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        for a, _, c_ in arms:
            mu = []
            for n in sizes:
                g = vals(a, n, key)
                mu.append(g.mean())
                ax.plot(n * np.exp(rng.normal(0, 0.012, len(g))), g, "o", ms=2.8, color=c_,
                        alpha=0.8, mec="none", zorder=3, clip_on=False)
            ax.plot(sizes, mu, "-", lw=1.3, color=c_, zorder=4)
        ax.set_xscale("log")
        if logy:
            ax.set_yscale("log")
            ratios = [vals("synnoise", n, key).mean() / vals("control", n, key).mean()
                      for n in sizes]
            for n, r in zip(sizes, ratios):
                ax.annotate(f"{r:.2f}x", (n, vals("synnoise", n, key).mean()),
                            textcoords="offset points", xytext=(0, 6), ha="center",
                            fontsize=6.2, color=col)
        ax.set_xticks(sizes)
        ax.set_xticklabels([f"{n}" for n in sizes])
        ax.xaxis.set_minor_locator(NullLocator())
        if logy:
            ax.set_yticks([200, 300, 400, 600])
            ax.yaxis.set_minor_locator(NullLocator())
            for (_, lab, c_), yl in zip(arms, (0.16, 0.05)):
                ax.text(0.98, yl, lab, transform=ax.transAxes, ha="right", va="bottom",
                        fontsize=6.6, color=c_)
        ax.yaxis.set_major_formatter(ScalarFormatter())
        ax.set_ylabel(ylab, fontsize=7.0)
        if i == len(SW_MEASURES) - 1:
            ax.set_xlabel("$N$")
        else:
            ax.tick_params(labelbottom=False)
        axes.append(ax)
        ps.ygrid(ax)
        ps.despine(ax)
    drop = [100.0 * (vals("control", n, "r2_common").mean() - vals("synnoise", n, "r2_common").mean())
            for n in sizes]
    fig.suptitle(
        "3-bit flip-flop, 40,000 iterations, 3-6 seeds per point, every seed drawn",
        fontsize=7.4, color=ps.INK, linespacing=1.4, y=1.005)
    return ps.save(fig, name, w_mm=92)


def synnoise_scatter_slide(name="slide_26c_synnoise_scatter", sigma_w=1.0, task="NBitFlipFlop",
                           pr_range=None, task_label=None, min_sizes=1, cache=None,
                           source=None, control_cells=None):
    """One claim: synaptic noise moves every network up and to the right of its own control, at
    every size, and the colour says the extra units are spread over more directions.

    One point per network: active units on x, the common-condition R-squared on y, participation
    ratio as colour, arm as marker shape. Three measures in one picture, which the four panels of
    26b cannot do - there the same network appears four times and the reader has to re-pair it.

    The colour scale is CLIPPED to pr_range rather than fitted to the data: on the flip-flop two
    networks fall below 4 and one above 7, and letting them set the ends compresses the range the
    other 25 live in. The colourbar is drawn with both ends extended so the clipping is visible
    rather than silent. Left as None the range is the 10th-90th percentile of this task's own
    networks, rounded outward, because the two tasks' dimensionalities do not overlap - the
    flip-flop runs 4-7 and CDDM 1.2-1.5 - and one shared scale renders CDDM a single colour.

    Args:
        name: output file stem; sigma_w: the rung to draw; task: task name as the cache records it;
        pr_range: (vmin, vmax) for the colour scale, or None to take it from this task's data;
        task_label: how the title names the task, defaulting to `task`; min_sizes: the fewest sizes
        the task must carry in both arms - 1 for a task trained at one size only;
        cache: path to the npz of per-network rows, defaulting to the Figure 2 cache; source: a
        line naming where those rows came from, printed under the panel when the rows are not the
        Figure 2 cache's; control_cells: substrings a control cell must match to be drawn, or None
        for every control the cache files under this task and size.

    ⚠️ CONTROL_CELLS IS NOT OPTIONAL ON CDDM. The cache files anything without an intervention as
    `control`, and CDDM's controls include a weight-decay horizon sweep: at N = 1000 its wd = 1e-3
    and wd = 1e-2 cells hold 63 and 89 active units against 307 for the matched wd = 1e-6, so
    pooling them puts a different experiment in the reference cloud. The flip-flop's controls are
    one condition and need no filter.
    Returns:
        the output path, or None if the task carries fewer than min_sizes such sizes.
    """
    c = F2.load()
    is_sw = np.array([bool(m) and float(m.group(1)) == sigma_w
                      for m in (SW_RE.search(str(x)) for x in c["cell"])])
    base = c["task"] == task
    keep0 = (np.ones(len(c["cell"]), bool) if control_cells is None else
             np.array([any(t in str(x) for t in control_cells) for x in c["cell"]]))
    sizes = sorted(set(c["N"][base & (c["arm"] == "synnoise") & is_sw].tolist())
                   & set(c["N"][base & (c["arm"] == "control") & keep0].tolist()))
    if len(sizes) < min_sizes:
        print(f"  SKIP {name}: {len(sizes)} sizes carry both arms")
        return None
    arms = [("control", "no synaptic noise", "o"), ("synnoise", rf"$\sigma_w$ = {sigma_w:g}", "s")]

    keep_ctrl = (np.ones(len(c["cell"]), bool) if control_cells is None else
                 np.array([any(t in str(x) for t in control_cells) for x in c["cell"]]))

    def rows(a, n):
        m = base & (c["N"] == n) & (c["arm"] == a) & (is_sw if a == "synnoise" else keep_ctrl)
        return (np.asarray(c["n_active"][m], float), np.asarray(c["r2_common"][m], float),
                np.asarray(c["dims"][m], float))

    if pr_range is None:
        allpr = np.concatenate([rows(a, n)[2] for a, _, _ in arms for n in sizes])
        lo, hi = np.percentile(allpr, [10, 90])
        step = 0.5 if hi - lo > 1.0 else 0.1
        pr_range = (np.floor(lo / step) * step, np.ceil(hi / step) * step)

    ps.setup()
    fig = plt.figure(figsize=(118 * ps.MM, 78 * ps.MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.0, 0.035], wspace=0.06,
                  left=0.11, right=0.90, top=0.84, bottom=0.12)
    ax = fig.add_subplot(gs[0, 0])
    sc = None
    for a, _, mk in arms:
        for n in sizes:
            x, y, pr = rows(a, n)
            sc = ax.scatter(x, y, c=pr, cmap="viridis", vmin=pr_range[0], vmax=pr_range[1],
                            marker=mk, s=26, lw=0.5, edgecolors=ps.INK, zorder=3)
        # NOTHING JOINS THE POINTS. A path through the per-size means and a per-size label both
        # went in and both came out: the control's N = 500 and N = 1000 clusters overlap in x
        # (200-250 against 257-347), so a label sits on a neighbouring size's marker, and a mean
        # path asserts an ordering across three points that the clouds already show. The size
        # ordering is carried by x - active units rise with N in both arms - and stated in the
        # title.
    ax.set_xlabel("active units")
    ax.set_ylabel("$R^2$ at $\\sigma_w$ = 0")
    ps.ygrid(ax)
    ps.despine(ax)
    ax.legend(handles=[Line2D([], [], ls="none", marker=mk, ms=4.5, mfc=ps.FAINT, mec=ps.INK,
                              mew=0.5, label=lab) for _, lab, mk in arms],
              loc="lower left", fontsize=6.6, frameon=False, handletextpad=0.5,
              borderaxespad=0.2, labelcolor=ps.INK)

    cax = fig.add_subplot(gs[0, 1])
    cb = fig.colorbar(sc, cax=cax, extend="both")
    cb.set_label("dimensionality\n$(\\sum_i\\lambda_i)^2 / \\sum_i\\lambda_i^2$", fontsize=7.0, linespacing=1.8)
    cb.ax.tick_params(labelsize=6.4)
    cb.outline.set_visible(False)

    lab_n = ", ".join(str(n) for n in sizes)
    n_ctrl = sum(len(rows("control", n)[0]) for n in sizes)
    n_sw = sum(len(rows("synnoise", n)[0]) for n in sizes)
    order = (", rising left to right within each arm" if len(sizes) > 1 else "")
    fig.suptitle(f"{task_label or task}, one point per network; $N$ = {lab_n}{order}\n"
                 f"{n_ctrl} control networks against {n_sw} at $\\sigma_w$ = {sigma_w:g}",
                 fontsize=7.4, color=ps.INK, linespacing=1.4, y=1.02)
    if source:
        fig.text(0.5, 0.005, source, ha="center", va="top", fontsize=6.4, color=ps.MUTED)
    return ps.save(fig, name, w_mm=118)


# The sigma_w ladder's cells, with the control pooled from the two matched unpenalised cells. Named
# here rather than discovered because this slide reads PARTICIPATION TRACES off disk, which the
# Figure 2 cache does not carry - the cache holds one row per finished network, not its history.
SW_TRACE_CELLS = [
    ("0", ["NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none",
           "NBitFlipFlop_std_dropfix/EqType=h_k=3_N=1000_pen=none_do=none"]),
    ("0.1", ["NBitFlipFlop_synnoise/EqType=h_k=3_N=1000_sw=0.1"]),
    ("0.3", ["NBitFlipFlop_synnoise/EqType=h_k=3_N=1000_sw=0.3"]),
    ("1", ["NBitFlipFlop_synnoise/EqType=h_k=3_N=1000_sw=1.0"]),
    ("2", ["NBitFlipFlop_synnoise_ext/EqType=h_N=1000_sw=2.0_iters=40000"]),
    ("3", ["NBitFlipFlop_synnoise_ext/EqType=h_N=1000_sw=3.0_iters=40000"]),
]


def synnoise_along_training_slide(name="slide_26e_synnoise_along_training", rungs=("0", "1")):
    """Does synaptic noise delay the silencing, hold units open, or neither? The trace, per rung.

    THE PER-DECADE SLOPE THAT ANSWERS THIS FOR DROPOUT (slide 24c) DOES NOT TRANSFER. These curves
    are not straight: every arm, control included, falls to a minimum somewhere between iteration
    9,000 and 22,000 and then partly recovers. A line fitted over the last decade crosses the
    minimum and reports the recovery as growth - it returns +47 units per decade for sigma_w = 1,
    which would read as "silencing reversed" and is an artefact of fitting a line to a U. The floor
    and the endpoint are reported instead, both of which are read off the trace rather than fitted.

    Traces come from the trainer's own `w_noise=False` participation probe, which ignores sigma_w,
    so a synaptic-noise network is counted in the same noise-free condition as the control.

    Args:
        name: output file stem; rungs: which sigma_w labels to draw, control first. The default is
        the control against the operating point alone - all six rungs on one axes is the ladder,
        which slide 26 already draws, and twelve traces bury the one comparison this panel is for.
    Returns:
        the output path, or None if the cells are missing.
    """
    import fig_dropout_live as DL
    arms = []
    for lab, cells in SW_TRACE_CELLS:
        if lab not in rungs:
            continue
        tr = [t for c in cells for t in DL.cell(os.path.join(DL.DATA_DIR, c))]
        if tr:
            arms.append((lab, tr))
    if len(arms) < 2:
        print(f"  SKIP {name}: participation traces missing")
        return None
    # the deck's own two colours, so sigma_w = 1 is the same blue here as on slides 26b and 26c
    cols = ([ps.BASE] + [ps.COND_COL["synnoise"]] * (len(arms) - 1) if len(arms) == 2 else
            [ps.BASE] + [plt.get_cmap("viridis")(x)
                         for x in np.linspace(0.08, 0.88, len(arms) - 1)])

    ps.setup()
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(ps.W2 * 0.56, 116 * ps.MM))
    fig.subplots_adjust(wspace=0.26)
    floor, end = {}, {}
    for (lab, tr), col in zip(arms, cols):
        mins, ends, at = [], [], []
        for t in tr:
            pit, sf = np.asarray(t["pit"], float), np.asarray(t["sf"], float)
            ax.plot(pit, sf, lw=0.5, color=col, alpha=0.35, zorder=3)
            k = int(np.argmin(sf))
            mins.append(sf[k]); at.append(pit[k]); ends.append(sf[-1])
            it, loss = np.asarray(t["it"], float), np.asarray(t["loss"], float)
            n = min(len(it), len(loss))
            ax2.plot(it[:n], loss[:n], lw=0.5, color=col, alpha=0.35, zorder=3)
        grid = np.asarray(tr[0]["pit"], float)
        mean = np.mean([np.interp(grid, np.asarray(t["pit"], float),
                                  np.asarray(t["sf"], float)) for t in tr], axis=0)
        ax.plot(grid, mean, lw=1.6, color=col, zorder=5,
                label=(rf"$\sigma_w$ = {lab}" if lab != "0" else "no synaptic noise"))
        floor[lab] = (float(np.mean(mins)), float(np.mean(at)))
        end[lab] = float(np.mean(ends))
    ax.set(xscale="log", xlabel="training iteration", ylabel="live units of 1000")
    ps.ygrid(ax)
    ps.despine(ax)
    # a legend, not labels at the right edge: those land in the gutter between the two panels and
    # print over the loss axis's y label. The curves all start high, so the lower left is empty.
    ax.legend(loc="lower left", fontsize=6.4, frameon=False, handlelength=1.3,
              labelspacing=0.3, borderaxespad=0.3)
    lo = min(float(np.nanmin(np.asarray(t["loss"], float))) for _l, tr in arms for t in tr)
    ax2.set(xscale="log", yscale="log", xlabel="training iteration",
            ylabel="clean training loss", ylim=(lo * 0.7, 1.6))
    ps.ygrid(ax2)
    ps.despine(ax2)

    best = max(end, key=end.get)
    fig.suptitle(
        "3-bit flip-flop, $N$ = 1000, unpenalised, every seed drawn\n"
        "left: live units against iteration;  right: clean training loss",
        fontsize=7.4, color=ps.INK, linespacing=1.35, y=1.03)
    fig.text(0.5, -0.02,
             f"Minima fall between iteration {min(v[1] for v in floor.values()):.0f} and "
             f"{max(v[1] for v in floor.values()):.0f}.\nThe budget is 40,000 iterations, not the "
             "150,000 slide 24c reads, so nothing here says the silencing has stopped.",
             ha="center", va="top", fontsize=6.6, color=ps.MUTED, linespacing=1.4)
    return ps.save(fig, name)


# The bottom line: every knob and every mechanism on ONE axis, as a ratio to its OWN control.
# Absolute counts are not comparable across these rows - the one-knob families are CDDM and the
# flip-flop at four different budgets, the mechanisms are the flip-flop at N = 1000 and 40,000
# iterations - but each row's ratio is, which is the convention slide 15 already uses for panel d.
BOTTOM_LABEL = {"activation": "activation function", "architecture": "equation form, trainable bias",
                "metabolic": "metabolic penalty", "weight decay": "weight decay",
                "input scale": "input scale", "noise": "recurrent noise"}


def _bottom_knobs():
    """Each one-knob ladder's span of active-unit ratios against its own reference rung.

    ⚠️ READS THE BIAS-0 LADDERS, NOT `collect()`. Until 2026-10-04 this read the Figure-1 families:
    trainable-bias networks, each at one fixed iteration. The talk's knob section now draws the
    standard network with the bias fixed at 0, and the two disagree enough to change what the panel
    says -- the metabolic penalty spans 0.92-1.04x on the old data and 1.00-1.57x on the new, so it
    moves out of the block of knobs that do nothing. `fig_knobs_at_floor` owns those ladders and the
    read-out; this is a thin adapter so the bottom line cannot drift away from the section above it.

    Returns:
        [(label, lo, hi, worst_label, best_label), ...], one row per ladder. Input scale is left in:
        bottom_line_slide filters it out itself, because it belongs with the mechanisms.
    """
    rows = KF.knob_ratio_spans()
    # the bottom line's own row names, which are shorter than the ladder titles
    rename = {"input weight scale": "input scale", "activation function": "activation function"}
    return [(rename.get(lab, lab), lo, hi, w, b) for lab, lo, hi, w, b in rows]


def bottom_line_slide(name="slide_33_bottom_line"):
    """Two blocks: what the knobs a modeller already has do to the count, and what five added
    mechanisms do, each against its own control.

    ⚠️ THE KNOBS ARE NOT A BLOCK OF NULLS ANY MORE. On the trainable-bias runs this panel
    showed every knob inside 0.92-1.12x except weight decay, and the slide read "the knobs
    do nothing". On the standard network with the bias fixed at 0, weight decay spans
    0.32-1.55x and input scale 1.00-1.60x, as far as several of the mechanisms and in both
    directions. The metabolic penalty is NOT one of them: it spans 1.00-1.15x once its
    control is read at the same 75,000 iterations as its arms, against the 1.00-1.57x a
    200,000-iteration control produced.

    TWO BLOCKS ON ONE RATIO AXIS. Absolute counts cannot share an axis here - the one-knob families
    are CDDM and the flip-flop at four budgets - but every row is that row's own control, so the
    ratios can. 1.0 is drawn, and it is where the whole upper block sits.

    The R-squared cost is annotated on the mechanisms only. The knobs that leave the count alone
    have no trade to report, and the two that move it (weight decay, removing the recurrent noise)
    move it DOWNWARD, where a cost is beside the point.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if neither block can be built.
    """
    # input scale is a KNOB and sits in the knob block with the other four, all five read from
    # the same bias-0 ladders. It used to be moved down to the mechanism block with a ratio
    # from its own trainable-bias cells and an R^2 from a different budget; that put one arm
    # in two blocks of one panel on two different networks.
    knobs = sorted(_bottom_knobs(), key=lambda r: r[2])
    c = F2.load()
    at = F2.restrict(c, n_units=F2.N_MAIN)
    c1 = F2.restrict(c, n_units=F2.N_MAIN,
                     chosen={a: F2.pick_cell(at, a)[0] for a in F2.GRID_ARMS})
    ctrl_a = np.asarray(c1["n_active"][c1["arm"] == "control"], float)
    ctrl_r = np.asarray(c1[F2.R2_KEY][c1["arm"] == "control"], float)
    mech = []
    for key, _short, full, col in F2.ARMS:
        if key == "control":
            continue
        m = c1["arm"] == key
        if not m.sum():
            continue
        a = np.asarray(c1["n_active"][m], float)
        r = np.asarray(c1[F2.R2_KEY][m], float)
        # ARM_SAYS, not `full`: this panel used to be the one place in the deck still printing
        # "penalty: frm + rws" and "dropout: mute" while every other slide said "rate + sparsity"
        # and "dropout"
        mech.append((ARM_SAYS.get(full, full), a.mean() / ctrl_a.mean(),
                     100.0 * (r.mean() - ctrl_r.mean()) / ctrl_r.mean(), col))
    if not knobs or not mech:
        print(f"  SKIP {name}: nothing to draw")
        return None
    mech.sort(key=lambda t: t[1])

    ps.setup()
    fig, ax = plt.subplots(figsize=(150 * ps.MM, 92 * ps.MM))
    fig.subplots_adjust(left=0.30, right=0.90, top=0.84, bottom=0.11)
    y = 0.0
    yt, yl = [], []
    ax.axvline(1.0, color=ps.INK, lw=0.9, zorder=2)
    for lab, lo, hi, _wl, _bl in knobs:
        ax.plot([lo, hi], [y, y], "-", lw=3.0, color=ps.BASE, alpha=0.5,
                solid_capstyle="round", zorder=3)
        ax.plot([lo, hi], [y, y], "o", ms=3.4, color=ps.BASE, zorder=4)
        # THE GREY ROWS NOW CARRY THEIR NUMBERS, like the coloured ones below them. A bar with a dot
        # at each end and no figure said nothing about what it spanned, so a viewer could not tell
        # that a grey row is a knob's WHOLE sweep while a coloured marker is one operating point.
        ax.annotate(f"{lo:.2f}\u2013{hi:.2f}x", (hi, y), textcoords="offset points",
                    xytext=(9, 0), va="center", fontsize=6.4, color=ps.MUTED)
        yt.append(y); yl.append(lab); y -= 1.0
    y -= 0.7
    sep = y + 0.35
    for lab, mult, dr2, col in mech:
        ax.plot([1.0, mult], [y, y], "-", lw=1.0, color=col, alpha=0.45, zorder=3)
        ax.plot([mult], [y], "s", ms=5.0, color=col, zorder=4)
        ax.annotate(f"{mult:.2f}x,  {dr2:+.1f}% $R^2$", (mult, y), textcoords="offset points",
                    xytext=(9, 0), va="center", fontsize=6.4, color=col)
        yt.append(y); yl.append(lab); y -= 1.0
    ax.axhline(sep, color=ps.GRID, lw=0.8, zorder=1)
    ax.set_xscale("log")
    ax.set_xticks([0.3, 0.5, 1.0, 2.0, 3.0])
    ax.set_xticklabels(["0.3x", "0.5x", "1x", "2x", "3x"])
    ax.xaxis.set_minor_locator(NullLocator())
    ax.set_yticks(yt)
    ax.set_yticklabels(yl, fontsize=7.0)
    ax.set_ylim(y + 0.5, 0.8)
    ax.set_xlim(0.26, 5.6)
    ax.set_xlabel("active units, relative to that row's own control")
    ax.tick_params(axis="y", length=0)
    ps.despine(ax, keep=("bottom",))
    # the block headings sit ABOVE their first row, in the gutter: placed level with it they
    # printed over the "rescale" tick label
    # the heading says what the grey rows ARE, once, so the five of them need not each say it
    for txt, yy, col in [("one knob at a time, across its whole sweep", 0.55, ps.MUTED),
                         ("a mechanism added, at one setting", sep + 0.22, ps.INK)]:
        ax.text(0.27, yy, txt, fontsize=7.0, color=col, va="center", style="italic")

    # ONE LINE. Three earlier versions put a computed three-line summary here -- how many knobs
    # stayed near 1, which ones moved and how far, what the mechanisms cost -- and every figure on
    # the panel already says it: each bar carries its own ratio and its own cost, and the two
    # italic block headings label the two groups. Two of those summaries were also contradicted by
    # the panel beneath them (one read "no standard knob raises the count", which weight decay at
    # zero falsifies at 1.45x). The caveats that used to run under the figure -- ratios only,
    # because the rows differ in task and budget, and which row is read at which budget -- are
    # facts about the data rather than things to look at, so they live in the deck's appendix.
    fig.suptitle("every arm against its own matched control", fontsize=7.8, color=ps.INK, y=1.015)
    return ps.save(fig, name, w_mm=150)


# Synaptic rescaling, drawn rather than written out. The slide carried the rule as a 14-line
# pseudocode block listing four state transitions and four formulas; a talk cannot be read that way.
# These are the launcher's own swept settings, so the numbers on the panel are the rule's, not
# round ones chosen to look tidy.
RESCALE_ALPHA, RESCALE_CAP, RESCALE_PROTECT = 1.002, 64, 0.20


def rescale_mechanism_slide(name="slide_27_rescale_rule", seed=3, n_in=12):
    """The rescaling rule as a state diagram plus the arithmetic of one boost step.

    LEFT: the life of one dormant unit - silent, growing, and the two ways out. The conditions ride
    on the arrows, so the reader follows a path instead of matching four labelled blocks to four
    formulas.

    RIGHT: one incoming row before and after a boost, computed BY THE RULE rather than drawn: the
    positive entries are multiplied by alpha, the negative ones divided, and the row is rescaled to
    the length it started with. The bars whose source unit is silent are hatched, because that is
    where the measured 55% of a dormant unit's excitatory weight goes and it delivers nothing - no
    current flows from a unit that emits nothing, however large the synapse is made.

    The boost is drawn at an exaggerated alpha so one step is visible; the caption gives the real
    one, 1.002 per iteration, and says the panel is a schematic.

    Args:
        name: output file stem; seed: generator seed for the example row; n_in: synapses drawn.
    Returns:
        the output path.
    """
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 72 * ps.MM))
    gs = GridSpec(1, 2, figure=fig, width_ratios=[1.22, 1.0], wspace=0.17,
                  left=0.015, right=0.975, top=0.83, bottom=0.10)

    # ---- (a) the state machine -----------------------------------------------------------------
    # LAYOUT NOTE: the two exit arrows leave the GROWING box SIDEWAYS, not downward, because the
    # per-iteration block has to sit under that box and a pair of arrows through it is unreadable.
    ax = fig.add_subplot(gs[0, 0])
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    col = ps.COND_COL["rescale"]
    gx, gy, gw, gh = 0.50, 0.63, 0.34, 0.085           # the GROWING box
    for x, y, w, h, lab, c, face in [
            (0.50, 0.93, 0.30, 0.080, "dormant unit", ps.MUTED, "none"),
            (gx, gy, gw, gh, "GROWING", col, "#f2f1ec"),
            (0.15, 0.10, 0.28, 0.080, "fires again", col, "none"),
            (0.85, 0.10, 0.28, 0.080, "given up", ps.MUTED, "none")]:
        ps.box(ax, x - w / 2, y - h / 2, w, h, lab, col=c, face=face, lw=0.9, fs=6.6)

    # the arrow sits left of centre so the OPEN text can run beside it rather than across it
    ps.arrow(ax, (0.36, 0.93 - 0.040), (0.40, gy + gh / 2), col=ps.MUTED, lw=0.9)
    ax.text(0.46, 0.785, "OPEN, when it is silent, not already\n"
            f"growing, and fewer than {RESCALE_PROTECT:.0%} of $N$ grow:\n"
            "redraw its OUTGOING weights once,\nset gain = 1",
            ha="left", va="center", fontsize=5.8, color=ps.INK, linespacing=1.5)

    # the loop that does the work, drawn on the box's left shoulder
    ax.annotate("", xy=(gx - gw / 2 + 0.035, gy + gh / 2), xytext=(gx - gw / 2, gy - gh / 2 + 0.01),
                arrowprops=dict(arrowstyle="-|>", color=col, lw=1.0,
                                connectionstyle="arc3,rad=-1.5", shrinkA=0, shrinkB=0))
    ax.text(0.50, 0.495, "every iteration while it grows\n"
            r"$+$ synapses $\times\,\alpha$,  $-$ synapses $\div\,\alpha$"
            "\nrow rescaled to its old length\n"
            r"gradient held off,  gain $\times\!=\,\alpha$",
            ha="center", va="top", fontsize=5.9, color=col, linespacing=1.5)

    # exits leave sideways, then drop
    for x_end, lab, rad in [(0.15, "its activity reaches\nthe target $A^{*}$", 0.25),
                            (0.85, f"its gain reaches\nthe cap, {RESCALE_CAP}" + r"$\times$", -0.25)]:
        ps.arrow(ax, (gx + np.sign(x_end - gx) * gw / 2, gy), (x_end, 0.10 + 0.040),
                 col=ps.MUTED, lw=0.9, rad=rad)
        ax.text(x_end - np.sign(x_end - gx) * 0.055, 0.265, lab, ha="center", va="center",
                fontsize=5.8, color=ps.INK, linespacing=1.4)
    ax.text(0.50, 0.012, "either way the gradient is released and the unit is on its own",
            ha="center", va="center", fontsize=5.9, color=ps.MUTED)
    ax.set_title("(a)  the life of one dormant unit", fontsize=7.0, color=ps.INK, loc="left",
                 pad=4)

    # ---- (b) one boost step on one incoming row ------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    rng = np.random.default_rng(seed)
    w0 = rng.normal(0, 1.0, n_in)
    src_silent = rng.random(n_in) < 0.5           # which source units emit nothing
    show_alpha = 1.45                              # exaggerated so ONE step is visible
    # ⚠️ THE SHADED COLUMNS ARE LEFT ALONE, because the rule leaves them alone. `rescale_rows_`
    # takes a `live` mask over PREsynaptic units and the swept cells pass it (rescale_active_only),
    # so the recurrent boost reaches only columns of units that are firing. An earlier version of
    # this panel boosted every bar, which drew a rule the code does not implement.
    boost = (w0 > 0) & ~src_silent
    cut = (w0 <= 0) & ~src_silent
    w1 = np.where(boost, w0 * show_alpha, np.where(cut, w0 / show_alpha, w0))
    w1 *= np.linalg.norm(w0) / np.linalg.norm(w1)  # rescale_normalize: same row length
    x = np.arange(n_in)
    ax2.bar(x - 0.21, w0, width=0.40, color=ps.FAINT, edgecolor=ps.MUTED, lw=0.4, label="before")
    ax2.bar(x + 0.21, w1, width=0.40, color=col, edgecolor=col, lw=0.4, label="after one boost")
    for xi, sil in zip(x, src_silent):
        if sil:
            ax2.axvspan(xi - 0.5, xi + 0.5, color=ps.GRID, alpha=0.55, lw=0, zorder=0)
    ax2.axhline(0, color=ps.INK, lw=0.8, zorder=3)
    ax2.set_xticks([])
    ax2.set_xlabel("the unit's incoming synapses\n"
                   "shaded: the source unit is itself silent, so the rule skips that column",
                   fontsize=6.2, labelpad=4, linespacing=1.4)
    ax2.set_ylabel("weight")
    ax2.legend(loc="lower left", fontsize=6.0, frameon=False, handlelength=1.1,
               borderaxespad=0.2)
    ps.ygrid(ax2)
    ps.despine(ax2)
    ax2.set_title(f"(b)  one boost: $+$ up, $-$ down, same row length",
                  fontsize=7.0, color=ps.INK, loc="left", pad=4)

    # no banner: the diagram labels itself and the swept settings are in the deck text
    fig.text(0.5, 0.012, "(b) is a schematic: an example row, one step at an exaggerated factor",
             ha="center", va="top", fontsize=6.2, color=ps.MUTED)
    return ps.save(fig, name)


# Slide 20: the LOGNORMALITY OF FIRING RATES. The slide used to ask this of the recurrent weights,
# which is a different claim with a different reference: cortex's reported lognormal is the
# across-neuron firing-rate distribution (Buzsaki & Mizuseki 2014), while the synaptic-weight
# lognormal is a separate literature. `f2_rate_lognormal.py` builds the cache this reads.
RATE_CACHE = "data/f2_rate_lognormal.npz"


def _rate_cache(stat=""):
    """The rate cache restricted to one cell per arm, matching slides 17-19.

    Args:
        stat: "" for the mean rate, "sm_" for frm's soft maximum, "pt_" for the participation
            statistic - the three per-unit numbers the cache carries.
    Returns:
        (dict of arrays, field prefix, prefix for the count-matched fields), or (None, _, _) if the
        cache is absent. The arms are the SAME cells slides 17-19 draw, chosen by `F2.pick_cell` on
        the Figure 2 cache, so the two slides describe the same networks.
    """
    if not os.path.exists(RATE_CACHE):
        print(f"  (build it with f2_rate_lognormal.py: {RATE_CACHE} missing)")
        return None, "", ""
    z = np.load(RATE_CACHE, allow_pickle=True)
    c = {k: z[k] for k in z.files}
    main = F2.load()
    at = F2.restrict(main, n_units=F2.N_MAIN)
    chosen = {a: F2.pick_cell(at, a)[0] for a in F2.GRID_ARMS}
    keep = np.zeros(len(c["cell"]), bool)
    for i, (arm, cell) in enumerate(zip(c["arm"], c["cell"])):
        want = chosen.get(str(arm))
        keep[i] = (str(arm) == "control") or (want is not None and str(cell) == want)
    keep &= (c["task"] == F2.TASK_MAIN) & (c["N"].astype(int) == F2.N_MAIN)
    return {k: v[keep] if getattr(v, "shape", (0,))[:1] == keep.shape else v
            for k, v in c.items()}, f"r_{stat}", f"rk_{stat}"


def rate_dist_slide(name="slide_20_rate_dist", stat="", group=2):
    """The firing-rate distribution of the active units: one row per arm, sharing the rate axis.

    STACKED, NOT OVERLAID. Six curves in one axes cross each other four times and the thing the
    slide is about - how each arm's shape departs from a lognormal - is exactly where they overlap
    most. One row per arm, each with its OWN best-fit lognormal dashed over it, puts the comparison
    on the page instead of in the reader's head.

BINNING. The cache bins log10 r at 0.05, and ONE network with 300 active units puts about 15
    counts in a bin - a 25% Poisson wobble, which is what made the first version jagged. The counts
    are POOLED across an arm's networks before normalising, which is what buys the resolution back:
    at `group` = 2 the bins are 0.10 wide (1.26-fold in rate) and every arm still carries 68-188
    units per occupied bin, about a 10% wobble. Counts per bin were checked at group 2, 3 and 4
    before this default was set, rather than widening bins until the curve looked smooth. ⚠️ Pooling weights a network by
    its active count rather than equally; the per-network widths in the panel below are unweighted,
    and the two agree, so nothing here rests on the choice.

    Args:
        name: output file stem; stat: which per-unit statistic (see `_rate_cache`);
        group: cache bins merged per drawn bin; 2 gives 0.10 in log10, about 1.26-fold.
    Returns:
        the output path, or None if the cache is missing.
    """
    c, pre, _ = _rate_cache(stat)
    if c is None:
        return None
    bins = np.asarray(c["log_bins"], float)
    nb = (len(bins) - 1) // group * group
    edges = bins[:nb + 1:group]
    mid = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)[0]

    arms = [(k, full, col) for k, _s, full, col in F2.ARMS if (c["arm"] == k).any()]
    ps.setup()
    fig = plt.figure(figsize=(W, 128 * ps.MM))
    gs = GridSpec(len(arms), 1, figure=fig, hspace=0.20,
                  left=0.17, right=0.97, top=0.985, bottom=0.075)
    axes, occ = [], []
    for i, (kind, full, col) in enumerate(arms):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        m = c["arm"] == kind
        h = np.asarray(c[f"{pre}hist"][m], float)[:, :nb].reshape(-1, len(mid), group).sum(axis=2)
        pooled = h.sum(axis=0)
        dens = pooled / pooled.sum() / width
        ax.fill_between(mid, dens, step="mid", color=col, alpha=0.30, lw=0, zorder=3)
        ax.step(mid, dens, where="mid", lw=1.1, color=col, zorder=4)
        occ += list(mid[dens > dens.max() * 0.01])
        # that arm's OWN best-fit lognormal, which is a Gaussian on this axis
        mu = float(np.sum(mid * dens) / np.sum(dens))
        sg = float(np.nanmean(np.asarray(c[f"{pre}sigma_log"][m], float))) / np.log(10)
        xs = np.linspace(mid[0], mid[-1], 300)
        ax.plot(xs, np.exp(-0.5 * ((xs - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi)),
                lw=1.0, ls=(0, (4, 2)), color=ps.INK, zorder=5)
        ks = float(np.nanmean(np.asarray(c[f"{pre}ks"][m], float)))
        ref = float(np.nanmean(np.asarray(c[f"{pre}ks_ref"][m], float)))
        sk = float(np.nanmean(np.asarray(c[f"{pre}skew_log"][m], float)))
        # BOTH LABELS UPPER LEFT. At the right edge they printed over the distribution in four of
        # the six rows; every row's left shoulder is empty because the rates bottom out there.
        # ONE LINE ALONG THE TOP. Stacked at the left the second line crossed the fitted curve in
        # four rows; the strip above every distribution is empty in all six.
        ax.annotate(f"{ARM_SAYS.get(full, full)}  ({int(m.sum())})", (0.012, 0.99),
                    xycoords="axes fraction",
                    fontsize=6.6, color=col, va="top")
        ax.annotate(f"{ks / ref:.1f}$\\times$ the sampling scale,  skew {sk:+.2f}",
                    (0.40, 0.99), xycoords="axes fraction", va="top",
                    fontsize=6.2, color=ps.MUTED)
        # HEADROOM FOR THE LABEL ROW. Without it the tallest bin reaches the top of the axes and
        # the annotations print over the distribution in whichever rows happen to peak highest.
        ax.set_ylim(0, max(dens.max(), np.nanmax(
            np.exp(-0.5 * ((xs - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi)))) * 1.42)
        ax.set_yticks([])
        ps.ygrid(ax)
        ps.despine(ax, keep=("bottom",))
        if i < len(arms) - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(r"firing rate of an active unit,  $\log_{10}$")
    axes[len(arms) // 2].set_ylabel("density")
    axes[0].set_xlim(min(occ) - 0.25, max(occ) + 0.25)

    # NO SUPTITLE. The deck heading says what the slide is, each row is labelled with its arm and
    # its two numbers, and the x axis is labelled; a three-line banner repeating all of that is the
    # only thing left to cut. The dashed curve gets one inline label on the top row.
    axes[0].annotate("lognormal fit", (0.985, 0.60), xycoords="axes fraction", ha="right",
                     va="top", fontsize=6.0, color=ps.INK)
    return ps.save(fig, name, w_mm=110)


def rate_shape_slide(name="slide_20b_rate_shape", stat=""):
    """Width against shape of the rate distribution: the two ways it can fail to be lognormal.

    x is sd(ln r), the lognormal width parameter; y is the excess kurtosis of ln r, which is 0 for
    an exact lognormal. Both are invariant under scaling every rate by a constant, so neither
    responds to an arm simply firing harder.

    Args:
        name: output file stem; stat: which per-unit statistic (see `_rate_cache`).
    Returns:
        the output path, or None if the cache is missing.
    """
    c, pre, prek = _rate_cache(stat)
    if c is None:
        return None
    ps.setup()
    fig, ax = plt.subplots(figsize=(W, 72 * ps.MM))
    rows = []
    for kind, _short, full, col in F2.ARMS:
        m = c["arm"] == kind
        if not m.any():
            continue
        x = np.asarray(c[f"{pre}sigma_log"][m], float)
        y = np.asarray(c[f"{pre}kurt_log"][m], float)
        xk = np.asarray(c[f"{prek}sigma_log"][m], float)
        ax.plot(x, y, "o", ms=4.2, color=col, mec="white", mew=0.6, zorder=3,
                label=f"{ARM_SAYS.get(full, full)} ({int(m.sum())})")
        ax.plot(np.nanmean(x), np.nanmean(y), "s", ms=7.0, color=col, mec="white", mew=1.1,
                zorder=4)
        rows.append((kind, full, np.nanmean(x), np.nanmean(y), np.nanmean(xk),
                     np.nanmean(np.asarray(c[f"{pre}skew_log"][m], float)),
                     np.nanmean(np.asarray(c[f"{pre}ks"][m], float)),
                     np.nanmean(np.asarray(c[f"{pre}ks_ref"][m], float))))
    ax.axhline(0.0, color=ps.FAINT, lw=0.8, ls=":", zorder=1)
    ax.annotate("exact lognormal", (ax.get_xlim()[0], 0.0), textcoords="offset points",
                xytext=(3, 3), ha="left", va="bottom", fontsize=5.6, color=ps.MUTED)
    ax.set_xlabel(r"width of the rate distribution,  sd $\ln r$")
    ax.set_ylabel("departure from lognormal,\n" r"excess kurtosis of $\ln r$")
    ax.legend(loc="best", fontsize=5.6, handlelength=1.0, borderpad=0.25, borderaxespad=0.3)
    ps.ygrid(ax)
    ps.despine(ax)
    # NO BANNER AND NO FOOTNOTE. Both axes are labelled, the legend names the arms, and the
    # count-matched widths and the active-units caveat are in the deck text under the figure -
    # repeating them here only shrinks the panel.
    return ps.save(fig, name, w_mm=110)


PEN_RATE_CACHE = "data/cddm_penalty_rates.npz"
PEN_RATE_ARMS = [("control", "untouched", ps.BASE), ("frm", "frm", ps.SLOTS[0]),
                 ("rws", "rws", ps.SLOTS[2]), ("both", "frm + rws", ps.SLOTS[1])]


def penalty_rate_dist_slide(name="slide_31b_penalty_rate_dist", group=5):
    """The rate distribution of the four CDDM penalty arms, stacked, with each arm's lognormal fit.

    The companion of slide 20 on the other task, drawn the same way so the two can be set against
    each other. ⚠️ THEY DISAGREE ABOUT THE CONTROL, which is the point: the flip-flop's unpenalised
    rates are lognormal to within their sampling scale, and CDDM's at 200,000 iterations are not -
    they are right-skewed, a few loud units over a long quiet tail.

    Args:
        name: output file stem; group: cache bins merged per drawn bin.
    Returns:
        the output path, or None if the cache is missing.
    """
    if not os.path.exists(PEN_RATE_CACHE):
        print(f"  (build it with cddm_penalty_rates.py: {PEN_RATE_CACHE} missing)")
        return None
    z = np.load(PEN_RATE_CACHE, allow_pickle=True)
    c = {k: z[k] for k in z.files}
    bins = np.asarray(c["log_bins"], float)
    nb = (len(bins) - 1) // group * group
    edges = bins[:nb + 1:group]
    mid = 0.5 * (edges[1:] + edges[:-1])
    width = np.diff(edges)[0]

    ps.setup()
    fig = plt.figure(figsize=(W, 94 * ps.MM))
    gs = GridSpec(len(PEN_RATE_ARMS), 1, figure=fig, hspace=0.22,
                  left=0.17, right=0.97, top=0.985, bottom=0.095)
    axes, occ, stats = [], [], {}
    for i, (kind, lab, col) in enumerate(PEN_RATE_ARMS):
        ax = fig.add_subplot(gs[i, 0], sharex=axes[0] if axes else None)
        axes.append(ax)
        m = c["arm"] == kind
        h = np.asarray(c["r_hist"][m], float)[:, :nb].reshape(-1, len(mid), group).sum(axis=2)
        pooled = h.sum(axis=0)
        dens = pooled / pooled.sum() / width
        ax.fill_between(mid, dens, step="mid", color=col, alpha=0.30, lw=0, zorder=3)
        ax.step(mid, dens, where="mid", lw=1.1, color=col, zorder=4)
        occ += list(mid[dens > dens.max() * 0.01])
        mu = float(np.sum(mid * dens) / np.sum(dens))
        sg = float(np.nanmean(np.asarray(c["r_sigma_log"][m], float))) / np.log(10)
        xs = np.linspace(mid[0], mid[-1], 300)
        fit = np.exp(-0.5 * ((xs - mu) / sg) ** 2) / (sg * np.sqrt(2 * np.pi))
        ax.plot(xs, fit, lw=1.0, ls=(0, (4, 2)), color=ps.INK, zorder=5)
        g = lambda f: float(np.nanmean(np.asarray(c[f][m], float)))
        stats[kind] = (g("n_active"), g("r_sigma_log"), g("r_skew_log"),
                       g("r_ks") / g("r_ks_ref"))
        ax.set_ylim(0, max(dens.max(), fit.max()) * 1.42)
        ax.annotate(f"{lab}   {stats[kind][0]:.0f} active", (0.012, 0.99),
                    xycoords="axes fraction", fontsize=6.6, color=col, va="top")
        ax.annotate(f"{stats[kind][3]:.1f}$\\times$ the sampling scale,  "
                    f"skew {stats[kind][2]:+.2f}", (0.44, 0.99), xycoords="axes fraction",
                    va="top", fontsize=6.2, color=ps.MUTED)
        ax.set_yticks([])
        ps.ygrid(ax)
        ps.despine(ax, keep=("bottom",))
        if i < len(PEN_RATE_ARMS) - 1:
            ax.tick_params(labelbottom=False)
    axes[-1].set_xlabel(r"firing rate of an active unit,  $\log_{10}$")
    axes[len(PEN_RATE_ARMS) // 2].set_ylabel("density")
    axes[0].set_xlim(min(occ) - 0.25, max(occ) + 0.25)
    # same as slide 20: no banner, one inline label for the dashed curve
    axes[0].annotate("lognormal fit", (0.985, 0.60), xycoords="axes fraction", ha="right",
                     va="top", fontsize=6.0, color=ps.INK)
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
    got = scaling_fraction_slide()
    if got:
        out.append(got)

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

    out.append(panel_slide("slide_rules", rules_without_gloss, width=ps.W2, height=52 * ps.MM))
    out.append(panel_slide("slide_f2_active", active_untilted, c=at_1000))
    # The deck draws r2 AGAINST active units rather than r2 on its own: the two categorical panels
    # ask one question together, and the scatter answers it in one picture. It carries every r2
    # value panel_c carries, with the per-arm cost in its legend. panel_c itself is untouched - the
    # manuscript figure still uses it, where it sits beside (b) and (d) in a row of three.
    out.append(panel_slide("slide_f2_r2_vs_active", r2_vs_active_with_scale, c=at_1000))
    out.append(panel_slide("slide_f2_dims", dims_untilted, c=at_1000))
    got = dims_extra_slide()
    if got:
        out.append(got)
    out.append(panel_slide("slide_f2_weights", F2.panel_e, c=at_1000))
    out.append(panel_slide("slide_f2_weight_shape", weight_shape_said, c=at_1000))
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
        # THE TITLE IS THE TASK AND THE SIZE, plus the read-out iteration derived from the data.
        # `title` is the slide's claim; it stays out of the figure and lives in presentation.md.
        # KNOB_ACCENT, not F1.GROUP_COL: see the comment on that constant. One knob at several
        # doses is an ordered quantity and must not spend a hue that means a condition elsewhere.
        out.append(draw(stem, f"{note}. {readout_line(iters, ARCHIVE_READOUT.get(fam_title))}",
                        ref_label, entries, ref, KNOB_ACCENT, note=None, ref_at=ref_at))

    for _t in PEN_TASKS:
        got = penalty_size_slide(_t)
        if got:
            out.append(got)
    for _fn in (synnoise_ladder_slide, synnoise_size_slide,
                synnoise_along_training_slide):
        got = _fn()
        if got:
            out.append(got)
    # the scatter, once per task. CDDM needs its control pinned to the matched cells - see the
    # warning in synnoise_scatter_slide - and has one size, so min_sizes is 1.
    for _kw in ({"task_label": "3-bit flip-flop, 40,000 iterations"},
                {"name": "slide_26d_synnoise_cddm", "task": "CDDM", "min_sizes": 1,
                 "task_label": "CDDM, 100,000 iterations",
                 "control_cells": ("CDDM_paper_grid",
                                   "CDDM_std_g0_horizon/EqType=h_N=1000_wd=1e-6")}):
        got = synnoise_scatter_slide(**_kw)
        if got:
            out.append(got)
    for _fn in (rescale_rule_slide, rescale_target_slide, rescale_cost_slide):
        got = _fn()
        if got:
            out.append(got)
    # ⚠️ `frm_vs_both_slide` IS NOT IN THIS REPOSITORY. Slide 31's panel was built by code that was
    # never committed, so naming it here crashed `main()` with a NameError before any slide after
    # this point was written. The name is looked up rather than referenced, so a full run completes
    # and says what is missing.
    for _name in ("prune_duplicate_slide", "temporal_pr_slide", "frm_vs_both_slide",
                  "selectivity_slide"):
        _fn = globals().get(_name)
        if _fn is None:
            print(f"  SKIP {_name}: its builder is not in this repository")
            continue
        got = _fn()
        if got:
            out.append(got)
    out += prune_duplicate_all_slide() or []
    for _fn in (dropout_selection_slide, dropout_targeting_slide, dropout_dose_slide,
                dropout_kinds_slide, dropout_rate_units_slide, dropout_rate_cost_slide,
                dropout_along_training_slide):
        got = _fn()
        if got:
            out.append(got)
    for _fn in (rate_dist_slide, rate_shape_slide, penalty_rate_dist_slide):
        got = _fn()
        if got:
            out.append(got)
    for _fn in (rescale_rule_slide, rescale_mechanism_slide):
        got = _fn()
        if got:
            out.append(got)
    got = bottom_line_slide()
    if got:
        out.append(got)
    got = control_trajectory_slide()
    if got:
        out.append(got)
    got = activation_r2_slide(
        "slide_x_activation_cddm_r2", ACTIVATION_CDDM, cddm_activation_cell, 199_900,
        "CDDM, $N$ = 1000.")
    if got:
        out.append(got)
    got = activation_r2_slide(
        "slide_x_weightdecay_r2", WEIGHTDECAY_CDDM, cddm_activation_cell, 199_900,
        "CDDM, $N$ = 1000.", by_dims=True)
    if got:
        out.append(got)
    got = activation_r2_slide(
        "slide_x_activation_ff_r2", ACTIVATION_FF, winp_cell, 150_000,
        "3-bit flip-flop, $N$ = 1000.")
    if got:
        out.append(got)
    got = inputscale_r2_slide()
    if got:
        out.append(got)
    got = metabolic_r2_slide()
    if got:
        out.append(got)
    got = metabolic_ruler_slide()
    if got:
        out.append(got)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--list", action="store_true", help="name the slides without drawing them")
    args = ap.parse_args()
    main(list_only=args.list)
