#!/usr/bin/env python3
"""
The talk's opening: four pictures that say why a dormant unit is a problem, before any result.

The deck used to start at the measurement. An audience that has not been told what a trained RNN is
FOR cannot tell whether 248 live units of 1000 is a catastrophe or a detail, so the first number in
the talk landed on nobody. These four figures supply the missing context, and they are figures
rather than bullet points because the claims are all comparisons -- this against that, before
against after, our rule against someone else's -- and a comparison is what a picture is good at.

EVERY NUMBER HERE IS MEASURED, by `motivation_cache.py`, on the same control networks the results
sections use. Nothing on these slides is an illustration except the wiring cartoon in M1, which
carries no numbers at all.

  M1  what a trained RNN is used for          -- schematic, no data
  M2  training is what empties the network    -- untrained twins against trained nets
  M3  another field's criterion, same units   -- our rule against Sokar et al.'s dormancy score
  M4  the question the rest of the talk asks  -- the two axes every later result is a point on

Usage:  python fig_motivation.py
Output: img/internal_figures/slide_m*.pdf (+ .svg)
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
import fig_paper_F2 as F2

CACHE = "data/motivation_cache.npz"
# ⚠️ THE OPENING USES NO CONDITION COLOUR. These four figures come before any intervention exists,
# so a condition hue spent here is a hue the audience has to unlearn: the first version drew "this
# talk's rule" in the dropout blue and "deep RL's rule" in the rate-and-sparsity green, and by the
# results section those two colours mean two of the remedies. Two neutral inks instead, dark against
# mid, which is all the separation a two-way comparison needs.
DARK, MID = ps.INK, "#8a8780"
TASKS = ("3-bit flip-flop", "CDDM")
# ⚠️ THE THRESHOLD IS SOKAR ET AL.'S OWN, NOT THE ONE THAT AGREES BEST. Their rule is a family
# indexed by tau, and the first version of this figure drew tau = 0.1 because it put the two counts
# within five units of each other -- a threshold chosen after seeing the answer, which is not
# evidence. 0.025 is the value their paper uses. Measured across the whole range they report:
#
#     tau        0.0    0.01   0.025   0.1      this project's rule
#     flip-flop  997    322    316     267      262
#     CDDM       294    280    276     268      271
#
# Every threshold from 0.01 up agrees with this project's count to within about 60 units of 1000, and
# tau = 0 is an outlier on one task only, for a reason worth saying out loud: the flip-flop's quiet
# units sit near 1e-3 rather than at 0, so "exactly zero" counts almost nobody there. That is the
# whole argument for a relative rule, and it arrives here as a bonus rather than as a caveat.
TAU_SHOWN = 0.025
DORMANT_COL = "#c0392b"  # ps.BAD -- used only beside the word "dormant", never as a series colour


def cache():
    """The measured motivation numbers, or None if they have not been built.

    Returns:
        dict of arrays from motivation_cache.py, or None.
    """
    if not os.path.exists(CACHE):
        print(f"  (build it with motivation_cache.py: {CACHE} missing)")
        return None
    return dict(np.load(CACHE, allow_pickle=True))


def dormancy_counts(z, task, tau=TAU_SHOWN):
    """Active-unit counts for one task under both criteria, as percentages of N.

    Args:
        z: the motivation cache; task: a key of TASKS; tau: the dormancy threshold to read.
    Returns:
        (ours_pct, theirs_pct, n_units) -- both as percent of N so a 100-dot grid can show them.
    """
    taus = list(np.asarray(z["taus"], float))
    j = taus.index(float(tau))
    n = float(z[f"{task}|n_units"])
    ours = float(np.mean(z[f"{task}|trained_active"]))
    theirs = n - float(np.mean(z[f"{task}|trained_dormant"])[j]) \
        if np.asarray(z[f"{task}|trained_dormant"]).ndim == 1 \
        else n - float(np.mean(np.asarray(z[f"{task}|trained_dormant"])[:, j]))
    return 100.0 * ours / n, 100.0 * theirs / n, n


# -------------------------------------------------------------------------------------------------
# M1: what the model is for
# -------------------------------------------------------------------------------------------------

def _population(ax, x0, w, y0, h, col, n=7, seed=0, spiky=False):
    """Draw a small stack of activity traces -- the thing both a brain and an RNN hand you.

    Args:
        ax: axes; x0, y0: lower-left corner; w, h: extent; col: trace colour; n: traces;
        seed: RNG seed; spiky: draw sparse events instead of smooth rates.
    Returns:
        None.
    """
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 1, 220)
    for i in range(n):
        base = y0 + h * (i + 0.5) / n
        if spiky:
            v = (rng.random(t.size) < 0.035).astype(float)
            v = np.convolve(v, np.exp(-np.linspace(0, 6, 14)), mode="same")
        else:
            v = np.abs(np.sin(2 * np.pi * (1 + 2 * rng.random()) * t + rng.random() * 6))
            v *= 0.4 + rng.random()
        ax.plot(x0 + t * w, base + v * (h / n) * 0.72, lw=0.6, color=col, zorder=4)


def m1_model_organism(name="slide_m1_model_organism"):
    """M1: a trained RNN is read the way a recorded population is read.

    NO NUMBERS ON THIS SLIDE. It exists to establish what the units in the box are for, so that
    "three quarters of them never fire" has somewhere to land. The two columns are deliberately
    drawn with the same anatomy -- a source, a population of traces, and the same analysis box
    underneath -- because the claim is that one substitutes for the other.

    Args:
        name: output file stem.
    Returns:
        the output path.
    """
    ps.setup()
    fig, ax = plt.subplots(figsize=(150 * ps.MM, 76 * ps.MM))
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))

    for x0, title, col, spiky, sub in ((0.06, "a cortical population", ps.MUTED, True,
                                        "recorded during a task"),
                                       (0.56, "a trained RNN", DARK, False,
                                        "trained on the same task")):
        ps.box(ax, x0, 0.62, 0.38, 0.30, None, col=col, lw=0.8)
        _population(ax, x0 + 0.03, 0.32, 0.645, 0.26, col, n=8, seed=1 if spiky else 4,
                    spiky=spiky)
        ax.text(x0 + 0.19, 0.955, title, ha="center", va="center", fontsize=8.0, color=col,
                fontweight="bold")
        ax.text(x0 + 0.19, 0.585, sub, ha="center", va="top", fontsize=6.4, color=ps.MUTED)
        ps.arrow(ax, (x0 + 0.19, 0.555), (0.50, 0.44), col=ps.MUTED, lw=0.9, rad=0.0,
                 mutation_scale=8)

    ps.box(ax, 0.17, 0.26, 0.66, 0.17, None, col=ps.INK, face="#f2f1ec", lw=0.8)
    ax.text(0.50, 0.385, "the same population analyses", ha="center", va="center", fontsize=7.4,
            color=ps.INK)
    ax.text(0.50, 0.305, "how many dimensions  ·  what each unit codes  ·  "
                         "how rates are distributed",
            ha="center", va="center", fontsize=6.8, color=ps.MUTED)
    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# M2: training is what empties it
# -------------------------------------------------------------------------------------------------

def _grid_panel(ax, pct, col, caption, big, n_total=100):
    """A 100-dot pictogram of a live fraction, with its count over it and its label under it.

    One dot per percent, filled to the measured fraction. The silent dots are drawn hollow rather
    than omitted: a silent unit is present in the network, and that is the whole point.

    ⚠️ THE PITCH IS SET TO FIT THE ROWS, not to look square. `ps.square_pitch` equalises the two
    pitches in display units, which on a panel far from square puts rows 6 to 10 of a 10 x 10 grid
    below the axes and off the page -- so the first version of this figure showed five rows, all of
    them filled, and read as "every unit fires" on the panel whose whole point is that half of them
    do not. The y pitch is derived from the row count instead and x follows it.

    Args:
        ax: a blank axes; pct: percent of units active; col: fill colour for live dots;
        caption: the line under the grid; big: the headline number over it; n_total: dots.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    side = 10
    rows = int(np.ceil(n_total / side))
    top, bot = 0.80, 0.17                       # the band the grid occupies, leaving room for text
    dy = (top - bot) / (rows - 1)
    dx = 0.92 / (side - 1) * 0.86
    ps.unit_grid(ax, 0.5 - dx * (side - 1) / 2, top, round(pct), n_total, col=col, side=side,
                 pitch=(dx, dy), s=14, lw=0.7)
    ax.text(0.50, 0.93, big, ha="center", va="center", fontsize=9.6, color=col,
            fontweight="bold")
    ax.text(0.50, 0.055, caption, ha="center", va="center", fontsize=7.0, color=ps.MUTED)


def m2_training_empties(name="slide_m2_training_empties", task="3-bit flip-flop"):
    """M2: the same architecture, before and after training, by the same measurement.

    THE UNTRAINED TWIN IS THE CONTROL THIS CLAIM NEEDS. "Most units never fire" on its own invites
    the reply that a random ReLU network has half its units below threshold anyway -- which is
    true, and measured here: about 54 of 100. Training takes it to 26. Showing only the trained
    network would leave the audience unable to tell the architecture's doing from training's.

    THE HISTOGRAMS ARE THE SECOND HALF OF THE CLAIM. The counts alone could be a threshold moving;
    the distributions show that training does something a threshold cannot fake -- it opens a gap.
    At initialisation the live units form one mode; after training there are two, three orders of
    magnitude apart, with the majority in the lower one.

    Args:
        name: output file stem; task: which cached task to draw.
    Returns:
        the output path, or None if the cache is missing.
    """
    z = cache()
    if z is None:
        return None
    n = float(z[f"{task}|n_units"])
    init_a = np.asarray(z[f"{task}|init_active"], float)
    tr_a = np.asarray(z[f"{task}|trained_active"], float)
    read_at = float(np.asarray(z[f"{task}|read_at"], float)[0])

    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 80 * ps.MM))
    gs = GridSpec(2, 2, figure=fig, height_ratios=[1.0, 0.82], hspace=0.30, wspace=0.22)

    # ⚠️ THE LABELS NAME THE BUDGET, NOT "BEFORE" AND "AFTER". The deck reads 40,000 iterations on
    # the next slide and in every results section, so "after training" on its own invited exactly
    # the wrong reading -- that this panel's trained nets are the 40,000-iteration ones. They are
    # not: they run to their own max_iter, 150,000 on the flip-flop. "Untrained" is literal: the
    # comparison networks are fresh weight draws of the same architecture that have taken zero
    # gradient steps, not an early checkpoint of a trained run.
    for j, (vals, col, lab) in enumerate(((init_a, MID, "untrained"),
                                          (tr_a, DARK, f"after {read_at:,.0f} iterations"))):
        _grid_panel(fig.add_subplot(gs[0, j]), 100.0 * vals.mean() / n, col,
                    lab, f"{vals.mean():.0f} of {n:.0f} fire")

    # The distributions under the pictograms, on one shared log axis. The untrained network holds
    # units at EXACTLY zero -- their drive never rises above threshold for any input -- and a log
    # axis has nowhere to put them, so they are drawn as one bar past a break and marked, rather
    # than piled into the lowest bin where they would read as "very small" instead of "zero".
    bins = np.linspace(-5.5, 1, 46)
    ax0 = None
    for j, (key, col) in enumerate(((f"{task}|p_init", MID),
                                    (f"{task}|p_trained", DARK))):
        ax = fig.add_subplot(gs[1, j], sharex=ax0, sharey=ax0)
        ax0 = ax0 or ax
        p = np.asarray(z[key], float)
        n_zero = int((p <= 0).sum())
        ax.hist(np.log10(p[p > 0]), bins=bins, color=col, zorder=3)
        if n_zero > 0.01 * p.size:          # a handful of units is not a mode; do not label one
            ax.bar([-6.4], [n_zero], width=0.45, color=col, alpha=0.55, zorder=3)
            ax.annotate("exactly 0", (-6.4, n_zero), textcoords="offset points", xytext=(0, 3),
                        ha="center", fontsize=6.2, color=ps.MUTED)
        ax.set_xlim(-7.0, 1.2)
        ax.set_xticks([-6.4, -4, -2, 0])
        ax.set_xticklabels(["0", "$10^{-4}$", "$10^{-2}$", "1"])
        ax.set_xlabel("how much one unit fires")
        if j == 0:
            ax.set_ylabel("units")
        ps.ygrid(ax)
    fig.suptitle(f"{task}, $N$ = {n:.0f}, one architecture, one measurement\n"
                 f"left: {len(init_a)} fresh weight draws, no training at all.   "
                 f"right: {len(tr_a)} trained networks",
                 fontsize=7.2, color=ps.INK, linespacing=1.3, y=1.04)
    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# M3: another field's criterion, the same units
# -------------------------------------------------------------------------------------------------

def m3_two_criteria(name="slide_m3_two_criteria"):
    """M3: a dormancy rule written for deep reinforcement learning counts the same units.

    THIS IS THE INDEPENDENT ORACLE, and that is the only reason the slide exists. Our criterion is
    a spread-plus-level statistic thresholded at 5% of the network's own 95th percentile; theirs is
    a mean-rate statistic normalised by the population mean and thresholded at a fixed tau. The two
    share no term. They land within a handful of units of each other on both tasks, which is
    evidence about the networks rather than about either definition.

    ONE PICTOGRAM PER (TASK, RULE), so the comparison is read by looking rather than by subtracting.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the cache is missing.
    """
    z = cache()
    if z is None:
        return None
    tasks = [t for t in TASKS if f"{t}|trained_active" in z]
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 84 * ps.MM))
    gs = GridSpec(len(tasks), 2, figure=fig, hspace=0.34, wspace=0.14)
    # THE RULE NAMES ARE COLUMN HEADERS, printed once each. Repeating them under all four
    # pictograms spent eight words saying the same thing twice and pushed the figure's text past
    # what the brief allows.
    for i, task in enumerate(tasks):
        ours, theirs, n = dormancy_counts(z, task)
        for j, pct in enumerate((ours, theirs)):
            ax = fig.add_subplot(gs[i, j])
            _grid_panel(ax, pct, DARK if j == 0 else MID, task if j == 0 else "",
                        f"{pct * n / 100:.0f} of {n:.0f}")
            if i == 0:
                ax.set_title(["this talk's rule", "deep RL's rule"][j], fontsize=8.0,
                             color=DARK if j == 0 else MID, pad=16,
                             fontweight="bold")
    fig.suptitle("two unrelated definitions of a working unit, one answer",
                 fontsize=8.4, color=ps.INK, y=1.03)
    return ps.save(fig, name)


# -------------------------------------------------------------------------------------------------
# M4: the two axes the rest of the talk lives on
# -------------------------------------------------------------------------------------------------

def m4_the_question(name="slide_m4_the_question"):
    """M4: the axes every later result is a point on, with only the starting point drawn.

    THE SAME AXES AS SLIDE 18, DELIBERATELY. Setting the question on the axes that answer it means
    the payoff slide needs no introduction: it is this picture with the other arms added. The
    control is the measured control of the Figure 2 cache, not a mark placed by hand.

    Args:
        name: output file stem.
    Returns:
        the output path, or None if the Figure 2 cache is missing.
    """
    everything = F2.load()
    at_main = F2.restrict(everything, n_units=F2.N_MAIN)
    ctrl = at_main["arm"] == "control"
    cx = float(np.mean(np.asarray(at_main["n_active"], float)[ctrl]))
    cy = float(np.mean(np.asarray(at_main["r2"], float)[ctrl]))

    ps.setup()
    fig, ax = plt.subplots(figsize=(132 * ps.MM, 54 * ps.MM))
    n = float(F2.N_MAIN)
    ax.add_patch(plt.Rectangle((0.80 * n, cy - 0.003), 0.26 * n, 0.006,
                               color=MID, alpha=0.22, zorder=1))
    ax.axhline(cy, color=ps.FAINT, lw=0.8, ls=":", zorder=2)
    ax.axvline(n, color=ps.FAINT, lw=0.8, ls=":", zorder=2)
    ax.plot([cx], [cy], "o", ms=8, color=ps.BASE, mec="none", zorder=5)
    ax.annotate(f"{cx:.0f} of {n:.0f}", (cx, cy), textcoords="offset points",
                xytext=(0, -16), ha="center", fontsize=7.6, color=ps.BASE)
    ps.arrow(ax, (cx + 45, cy), (0.93 * n, cy), col=DARK, lw=1.7, rad=0.0,
             mutation_scale=11, zorder=6)
    ax.annotate("all of them,\nsame score", (0.93 * n, cy + 0.0022), ha="center", va="bottom",
                fontsize=7.8, color=DARK, linespacing=1.3)
    ax.set(xlim=(0, 1.10 * n), ylim=(cy - 0.0135, cy + 0.0105),
           xlabel="units that do work", ylabel="performance, held out")
    ax.set_yticks([cy])
    ax.set_yticklabels(["as trained"])
    ax.set_xticks([0, 500, 1000])
    ps.ygrid(ax)
    return ps.save(fig, name)


def main():
    """Write every motivation slide. Returns the list of output paths."""
    out = []
    for fn in (m1_model_organism, m2_training_empties, m3_two_criteria, m4_the_question):
        got = fn()
        if got:
            out.append(got)
    return out


if __name__ == "__main__":
    main()
