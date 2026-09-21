#!/usr/bin/env python3
"""
Supplementary Figure S0 - THE THREE TASKS. What the network is asked to do, on each of the tasks
the paper uses, as a pictogram of the trial and the actual input and target channels beside it.

This exists because Figure 1 used to try to explain a task and the silent-unit problem in the same
panel and explained neither. Silence is not a property of any one task - it happens on all three -
so the task definitions belong here, once, and the main text keeps its panels for the result.

  row 1  CDDM      context-dependent decision making (Mante et al. 2013). Two sensory features are
                   present on every trial and they can disagree; a context cue says which one to
                   report. The hard part is ignoring the other.
  row 2  k-bit     the flip-flop (Sussillo & Barak 2013). k independent memory bits, each driven by
         flip-flop its own Poisson-timed +-1 pulse train; each output holds the sign of the last
                   pulse that bit received, indefinitely, until the next one.
  row 3  DMTS      delayed match-to-sample. A sample stimulus, an empty delay, a test stimulus, and
                   a judgement of whether the two were the same, reported after a go cue.

NOTHING HERE IS DRAWN BY HAND. The right-hand channel stacks are the real input and target streams
produced by the task objects in `trainRNNbrain/tasks/`, instantiated from the same
`configs/task/*.yaml` the training runs used, so a reader can check the epoch times against the
config rather than against an illustrator. Only the left-hand pictograms are drawn.

Usage:  python fig_supp_tasks.py
Output: img/internal_figures/fig_supp_tasks.pdf (+ .svg; vector only - see paperstyle.save)
"""

import os
import sys

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from trainRNNbrain.training.training_utils import prepare_task_arguments

HERE = os.path.dirname(os.path.abspath(__file__))
CFG_DIR = os.path.join(HERE, "../../configs/task")

DT, TAU = 1.0, 10.0       # every run in the paper: dt = 1, tau = 10, so 300 steps = 30 tau
TASK_SEED = 0             # DMTS jitters its stimulus times, so the trial shown is seeded

# The two features of CDDM's stimulus. Motion is drawn as arrow DIRECTION and colour as arrow
# COLOUR, which is what the task is: one stimulus carrying two features that can point different
# ways. The two hues are categorical slots 2 and 3 of the validated palette, not red/green - the
# literal colours of Mante's task are the one pair a colour-blind reader cannot separate.
COL_A, COL_B = ps.SLOTS[1], ps.SLOTS[2]

# (stem, config, channel labels, target labels, epochs). Epochs are (start_step, end_step, label)
# and are only drawn where the task HAS epochs; the flip-flop's pulses are Poisson, so it has none.
TASKS = {
    "CDDM": dict(
        cfg="CDDM.yaml",
        title="CDDM  -  context-dependent decision making",
        inputs=["cue: attend motion", "cue: attend colour", "motion, right", "motion, left",
                "colour, right", "colour, left"],
        in_cols=[ps.INK, ps.INK, ps.INK, ps.INK, COL_A, COL_B],
        outputs=["choose right", "choose left"],
        epochs=[(0, 300, "context cue"), (100, 300, "stimulus"), (200, 300, "decision")],
        note="an attend-colour trial has the same stimulus and the opposite correct choice",
    ),
    "NBitFlipFlop": dict(
        cfg="NBitFlipFlop.yaml",
        title="$k$-bit flip-flop  -  $k$ independent memory bits",
        inputs=["bit 1 pulses", "bit 2 pulses", "bit 3 pulses"],
        in_cols=[ps.INK] * 3,
        outputs=["bit 1 state", "bit 2 state", "bit 3 state"],
        epochs=[],
        note="pulse times are Poisson, so a trial has no epochs and every trial differs",
    ),
    "DMTS": dict(
        cfg="DMTS.yaml",
        title="DMTS  -  delayed match-to-sample",
        inputs=["stimulus A", "stimulus B", "go cue"],
        in_cols=[COL_A, COL_B, ps.MUTED],
        outputs=["match"],
        epochs=[(10, 20, "sample"), (20, 80, "delay"), (80, 90, "test"), (100, 140, "decision")],
        note="a match trial; on a non-match trial the test lands on stimulus B and the target stays at 0",
    ),
}


def build(name):
    """Instantiate one task from its shipped config and return one representative trial.

    The trial is chosen, not drawn at random, so that each row shows the case that defines the
    task: CDDM with the two features DISAGREEING (the only trials that separate this task from a
    plain decision), and DMTS on a MATCH trial (the only trials whose target is non-zero).

    Args:
        name: key of TASKS, which is also the config stem.
    Returns:
        (inputs, targets): (n_inputs, n_steps) and (n_outputs, n_steps) float arrays.
    """
    cfg = OmegaConf.load(os.path.join(CFG_DIR, TASKS[name]["cfg"]))
    cfg.seed = TASK_SEED
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg, dt=DT))
    if name == "CDDM":
        # attend motion, motion favours right (+0.5), colour favours left (-0.5): a conflict trial
        inp, tgt = task.generate_input_target_stream("motion", 0.5, -0.5)
    elif name == "DMTS":
        inp, tgt, _ = task.generate_input_target_stream(0, 0)      # sample A, test A -> match
    else:
        inp, tgt, _ = task.generate_input_target_stream()
    return np.asarray(inp, float), np.asarray(tgt, float)


def channel_stack(ax, inp, tgt, spec):
    """Draw one task's input and target channels as a labelled stack over time.

    Args:
        ax: axes; inp: (n_inputs, n_steps); tgt: (n_outputs, n_steps); spec: the TASKS entry,
            supplying `inputs`, `in_cols`, `outputs`, `epochs` and `title`.
    Returns:
        None.
    """
    n_in, n_steps = inp.shape
    n_out = tgt.shape[0]
    t = np.arange(n_steps) * DT / TAU                       # time in membrane time constants
    rows = n_in + n_out
    amp = 0.40                                              # trace height, in row units
    # one shared vertical scale for the whole stack, so a tall step really is a larger input
    span = max(np.abs(np.concatenate([inp, tgt])).max(), 1e-9)

    bases, ticks, cols = [], [], []
    for k in range(rows):
        # inputs on top, targets below them, with one empty row of air between the two blocks
        bases.append(float(rows - 1 - k) + (0.0 if k < n_in else -0.55))
    for k, (lab, col) in enumerate(zip(spec["inputs"], spec["in_cols"])):
        ticks.append(lab)
        cols.append(col)
    for lab in spec["outputs"]:
        ticks.append(lab)
        cols.append(ps.SLOTS[0])

    for b, sig, col in zip(bases, np.vstack([inp, tgt]), cols):
        ax.plot([t[0], t[-1]], [b, b], lw=0.45, color="#b9b8b1", ls=(0, (1.2, 2)), zorder=1)
        ax.plot(t, b + amp * sig / span, lw=0.9, color=col, zorder=3, solid_joinstyle="miter")

    # the divider, and the two words that say which block is given to the network and which is
    # asked of it
    mid = (bases[n_in - 1] + bases[n_in]) / 2
    ax.axhline(mid, color=ps.FAINT, lw=0.5, ls=(0, (3, 3)), zorder=2)
    ax.text(t[-1] * 1.015, np.mean(bases[:n_in]), "input", rotation=-90, ha="left", va="center",
            fontsize=6.0, color=ps.MUTED)
    ax.text(t[-1] * 1.015, np.mean(bases[n_in:]), "target", rotation=-90, ha="left", va="center",
            fontsize=6.0, color=ps.SLOTS[0])

    # CDDM's epochs are NESTED (the cue is on for the whole trial, the stimulus for two thirds of
    # it), so they cannot share one line the way DMTS's sequential ones can. Greedy packing: an
    # epoch goes on the lowest level where it does not overlap one already there.
    top = bases[0] + 0.62
    levels = []
    for s0, s1, lab in spec["epochs"]:
        x0, x1 = s0 * DT / TAU, s1 * DT / TAU
        lv = next((k for k, occ in enumerate(levels) if x0 >= occ - 1e-9), len(levels))
        if lv == len(levels):
            levels.append(x1)
        else:
            levels[lv] = x1
        yb = top + 0.28 + 0.85 * lv
        ax.plot([x0, x1], [yb] * 2, lw=1.6, color=ps.FAINT, zorder=3, solid_capstyle="butt")
        ax.plot([x0, x0], [yb - 0.16, yb + 0.16], lw=0.5, color=ps.FAINT, zorder=3)
        ax.text((x0 + x1) / 2, yb + 0.06, lab, ha="center", va="bottom", fontsize=5.6,
                color=ps.MUTED)
    head = 0.28 + 0.85 * max(len(levels) - 1, 0) + 0.52 if spec["epochs"] else 0.2

    if spec.get("note"):
        ax.text(0.0, -0.25, spec["note"], transform=ax.transAxes, ha="left", va="top",
                fontsize=5.6, color=ps.MUTED)
    ax.set(yticks=bases, xlabel=r"time ($\tau$)",
           xlim=(0, t[-1] * 1.01), ylim=(bases[-1] - 0.62, top + head))
    ax.set_yticklabels(ticks, fontsize=6.0)
    for lab, col in zip(ax.get_yticklabels(), cols):
        lab.set_color(col)
    ps.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0)


def pic_cddm(ax):
    """Pictogram: one stimulus carrying two features that disagree, and a cue naming one of them.

    Args:
        ax: blank axes.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))

    ps.box(ax, 0.06, 0.800, 0.50, 0.120, "attend MOTION", col=ps.INK, face="#f2f1ec", lw=0.7,
           fs=6.0, text_col=ps.INK)
    ps.box(ax, 0.60, 0.800, 0.34, 0.120, "attend\nCOLOUR", col=ps.FAINT, lw=0.5, fs=5.4,
           text_col=ps.FAINT)
    ax.text(0.50, 0.995, "the cue names one of the two features", ha="center", va="top",
            fontsize=5.6, color=ps.MUTED)

    # the stimulus: a cloud of arrows. Direction is the motion feature, colour is the colour
    # feature, and on this trial they point opposite ways - which is the whole task.
    ps.box(ax, 0.13, 0.320, 0.74, 0.42, col=ps.MUTED, face=ps.PAPER, lw=0.7)
    rng = np.random.default_rng(4)
    xs = 0.19 + 0.62 * rng.random(26)
    ys = 0.375 + 0.31 * rng.random(26)
    right = rng.random(26) < 0.72                       # motion evidence: 72% rightward
    col_a = rng.random(26) < 0.28                       # colour evidence: 28% colour A
    for x, y, r, a in zip(xs, ys, right, col_a):
        d = 0.045 if r else -0.045
        ps.arrow(ax, (x - d / 2, y), (x + d / 2, y), col=COL_A if a else COL_B, lw=0.8,
                 mutation_scale=4.5, shrink=0, zorder=4)
    ax.text(0.50, 0.295, "most arrows point RIGHT;  most are colour B",
            ha="center", va="top", fontsize=5.6, color=ps.MUTED)

    ps.arrow(ax, (0.50, 0.245), (0.50, 0.195), col=ps.MUTED, lw=0.8)
    ps.box(ax, 0.16, 0.070, 0.31, 0.120, "choose RIGHT", col=ps.SLOTS[0], lw=0.9, fs=5.8,
           text_col=ps.SLOTS[0])
    ps.box(ax, 0.53, 0.070, 0.31, 0.120, "choose LEFT", col=ps.FAINT, lw=0.5, ls=(0, (2, 2)),
           fs=5.8, text_col=ps.FAINT)
    ax.text(0.50, 0.045, "the colour alone would have said LEFT,\nand the cue says to ignore it",
            ha="center", va="top", fontsize=5.2, color=COL_B, linespacing=1.3)


def pic_flipflop(ax):
    """Pictogram: one bit as a latch driven by signed pulses, times k.

    Args:
        ax: blank axes.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.50, 0.995, "each bit is a latch", ha="center", va="top", fontsize=5.6,
            color=ps.MUTED)

    # one bit, in full: the pulse train and the state it leaves behind
    pulses = [(0.33, +1), (0.52, -1), (0.78, +1)]
    y_in, y_out = 0.735, 0.445
    ax.plot([0.22, 0.97], [y_in] * 2, lw=0.5, color=ps.GRID, zorder=1)
    for x, s in pulses:
        ax.plot([x, x], [y_in, y_in + 0.085 * s], lw=1.1, color=ps.INK, zorder=3)
        ax.plot([x], [y_in + 0.085 * s], "o", ms=2.2, color=ps.INK, zorder=4)
    ax.text(0.20, y_in, "pulses in", ha="right", va="center", fontsize=5.8, color=ps.INK)

    # the held state: a step that changes only when a pulse arrives
    edges = [0.22] + [x for x, _ in pulses] + [0.97]
    state = [0] + [s for _, s in pulses]        # 0 before the first pulse, as the real target is
    for k in range(len(edges) - 1):
        ax.plot([edges[k], edges[k + 1]], [y_out + 0.11 * state[k]] * 2, lw=1.2,
                color=ps.SLOTS[0], zorder=3)
        if k:
            ax.plot([edges[k]] * 2, [y_out + 0.11 * state[k - 1], y_out + 0.11 * state[k]],
                    lw=1.2, color=ps.SLOTS[0], zorder=3)
    ax.text(0.20, y_out, "state\nheld", ha="right", va="center", fontsize=5.8,
            color=ps.SLOTS[0], linespacing=1.25)
    for lev, lab in [(0.11, "+1"), (0.0, "0"), (-0.11, "$-$1")]:
        ax.text(0.99, y_out + lev, lab, ha="left", va="center", fontsize=5.4, color=ps.SLOTS[0])
    ax.text(0.52, 0.285, "the state changes only when a pulse arrives,\n"
                         "and holds for as long as none does",
            ha="center", va="top", fontsize=5.6, color=ps.MUTED, linespacing=1.3)

    for k in range(3):
        ps.box(ax, 0.20 + 0.20 * k, 0.045, 0.16, 0.105, f"bit {k + 1}", col=ps.INK, lw=0.6,
               fs=5.6, face="#f2f1ec", text_col=ps.INK)
    ax.text(0.50, 0.020, "$k$ bits, independent, one channel each", ha="center", va="top",
            fontsize=5.6, color=ps.MUTED)


def pic_dmts(ax):
    """Pictogram: sample, empty delay, test, and the same/different judgement.

    Args:
        ax: blank axes.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.50, 0.995, "hold the sample across an empty delay", ha="center", va="top",
            fontsize=5.6, color=ps.MUTED)

    y = 0.60
    frames = [(0.07, "sample", COL_A), (0.37, "delay", None), (0.67, "test", COL_A)]
    for x, lab, col in frames:
        ps.box(ax, x, y, 0.25, 0.25, col=ps.MUTED, face=ps.PAPER, lw=0.7)
        if col is not None:
            ax.plot([x + 0.125], [y + 0.125], "s", ms=7.0, color=col, zorder=4)
        else:
            ax.text(x + 0.125, y + 0.125, "?", ha="center", va="center", fontsize=9.0,
                    color=ps.FAINT)
        ax.text(x + 0.125, y - 0.035, lab, ha="center", va="top", fontsize=5.8, color=ps.MUTED)
    for x in (0.325, 0.625):
        ps.arrow(ax, (x, y + 0.125), (x + 0.04, y + 0.125), col=ps.MUTED, lw=0.7,
                 mutation_scale=5)
    ps.arrow(ax, (0.07, 0.905), (0.78, 0.905), col=ps.FAINT, lw=0.6, mutation_scale=5)
    ax.text(0.80, 0.905, "time", ha="left", va="center", fontsize=5.6, color=ps.MUTED)

    ps.arrow(ax, (0.795, y - 0.10), (0.795, y - 0.175), col=ps.MUTED, lw=0.8)
    ps.box(ax, 0.58, 0.245, 0.42, 0.115, "same as the sample?", col=ps.INK, lw=0.6, fs=5.8,
           face="#f2f1ec", text_col=ps.INK)
    ax.text(0.545, 0.175, "match  $\\rightarrow$  output 1", ha="right", va="center", fontsize=5.8,
            color=ps.SLOTS[0])
    ax.text(0.545, 0.085, "non-match  $\\rightarrow$  output 0", ha="right", va="center",
            fontsize=5.8, color=ps.MUTED)
    ax.text(0.58, 0.130, "reported only\nafter the go cue", ha="left", va="center", fontsize=5.2,
            color=ps.MUTED, linespacing=1.25)


def main():
    """Assemble Supplementary Figure S0 and write it. Returns the output path."""
    ps.setup()
    fig = plt.figure(figsize=(ps.W2, 172 * ps.MM))
    gs = GridSpec(3, 2, figure=fig, width_ratios=[0.80, 1.30], hspace=0.62, wspace=0.34)

    for row, (name, pic, letter) in enumerate([("CDDM", pic_cddm, "a"),
                                               ("NBitFlipFlop", pic_flipflop, "b"),
                                               ("DMTS", pic_dmts, "c")]):
        inp, tgt = build(name)
        ax_p = fig.add_subplot(gs[row, 0])
        pic(ax_p)
        ax_p.set_title(TASKS[name]["title"], fontsize=7.2, color=ps.INK, loc="left", pad=15)
        ps.panel_letter(ax_p, letter, dx=-0.09, dy=1.10)
        channel_stack(fig.add_subplot(gs[row, 1]), inp, tgt, TASKS[name])
        print(f"  {name:13} {inp.shape[0]} inputs x {inp.shape[1]} steps -> "
              f"{tgt.shape[0]} outputs")

    return ps.save(fig, "fig_supp_tasks")


if __name__ == "__main__":
    main()
