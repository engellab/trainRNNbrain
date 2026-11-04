#!/usr/bin/env python3
"""
Supplementary Fig. 1 - THE FOUR TASKS. What the network is asked to do, on each of the tasks
the paper uses, as a pictogram of the trial and the actual input and target channels beside it.

This exists because Figure 1 used to try to explain a task and the silent-unit problem in the same
panel and explained neither. Silence is not a property of any one task - it happens on all four -
so the task definitions belong here, once, and the main text keeps its panels for the result.

  row 1  3BFF      the 3-bit flip-flop (Sussillo & Barak 2013). Three independent memory bits, each
                   driven by its own Poisson-timed +-1 pulse train; each output holds the sign of
                   the last pulse that bit received, indefinitely, until the next one.
  row 2  CDDM      context-dependent decision making (Mante et al. 2013). Two sensory features are
                   present on every trial and they can disagree; a context cue says which one to
                   report. The hard part is ignoring the other.
  row 3  DMTS      delayed match-to-sample. A sample stimulus, an empty delay, a test stimulus, and
                   a judgement of whether the two were the same, reported after a go cue.
  row 4  Maze      maze running, trained by PPO (trainRNNbrain/rl/maze_env.py). The network is the
                   whole agent: 8 egocentric range sensors and an efference copy of its last turn
                   and speed come in, a turn and a speed go out, one decision every 5 RNN steps
                   (0.5 tau), and reaching the cheese ends the episode. Nothing tells the agent
                   where the cheese is.

NOTHING HERE IS DRAWN BY HAND, AND NOTHING IS READ FROM `configs/task/`. The right-hand channel
stacks are the real input and target streams produced by the task objects in `trainRNNbrain/tasks/`,
instantiated from the config SAVED INSIDE A TRAINED RUN FOLDER - the one the manuscript's own
networks were trained with: NBitFlipFlop_ff_revive (the first 40k run of the 200k chain),
CDDM_paper_grid and DMTS_paper_grid_d5, all at N = 1000. The paper's DMTS runs are the 5-tau-delay
variant (configs/task/DMTS_d5.yaml): 2 stimuli, 2 outputs, a 19-tau trial. A figure of the tasks
is worth nothing if it is a figure of a different task, so
the config comes from the runs and the epoch bars are derived from it rather than typed in.

Row 4 has no target stream. Its stack shows the inputs and the executed outputs of ONE CLEAN
EPISODE (the RNN's noise off, the policy mean as the action, the training probe's drops) of the
first seed under MAZE_RUN, the final design's N = 1000 cell (Maze_suite, issue #3): of the probe
episodes that reach the cheese, the one of median length. The pictogram's maze is that run's
config, the path is that episode's. With no run folder yet, configs/rl/maze_ppo.yaml is composed,
an untrained network is played the same way and the row says so, as the other rows do.

Usage:  python fig_supp_tasks.py
Output: img/internal_figures/fig_supp_tasks.pdf (+ .svg; vector only - see paperstyle.save)
"""

import glob
import os
import sys

import hydra
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path
from matplotlib.ticker import MultipleLocator
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR
from trainRNNbrain.rl.maze_env import MazeEnv
from trainRNNbrain.rl.policy import MazePolicy
from trainRNNbrain.rl.ppo import rollout
from trainRNNbrain.rl.saved import load_run
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import filter_kwargs

TASK_SEED = 0             # DMTS jitters its stimulus times, so the trial shown is seeded
CFG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../configs/task")
DT_TAU = 1.0 / 10.0       # dt / tau, for the fallback path where no run config is available
BRACE_X = (-0.080, -0.205)   # right edge of each brace level, in axes fractions
BRACE_W = 0.017              # brace depth, likewise
H_MM = 176.0                 # printed height; the width is paperstyle.W_FULL_MM (165.1 mm). Four rows
                             # do not fit MAX_H_MM (141 mm): row d's 12 channels need 2.5 mm each for
                             # their tick labels and the supervised rows lose their text clearances
                             # below 26 mm. The paper has no height cap macro, so at width=\textwidth
                             # the figure (176 mm) and its caption still fit one float page.
MAZE_ROW = 1.3               # height of row d relative to a supervised row: 12 channels, not 6-8
# The maze runs of the final design (issue #3), the N = 1000 ReLU cell; seeds land as s1, s2, ...
MAZE_RUN = f"{DATA_DIR}_rl_maze/Maze_suite/N=1000_act=relu"
MAZE_CFG = "rl/maze_ppo"     # composed from configs/ for the pending fallback, no run folder yet
PENDING_DECISIONS = 60       # an untrained agent never reaches the cheese: its episode is cut here

# The two features of CDDM's stimulus. Motion is drawn as arrow DIRECTION and colour as arrow
# SHADE (dark or light grey), which is what the task is: one stimulus carrying two features that can
# point different ways. Shades, not hues: every hue of the house palette is reserved for a task, a
# per-unit rule or a training setting (paperstyle colour rules), and a colour-blind reader separates
# two greys of different lightness.
COL_A, COL_B = ps.INK, ps.FAINT   # dark and light grey, in the CDDM pictogram only

# Channel colour is SEMANTIC, not decorative: a channel is either a cue telling the network what to
# do (grey) or a sensory stimulus it must read (black); the constant input is light grey. Targets are
# drawn in the row's TASK colour (paperstyle.TASK_COLORS), the one meaning that hue has everywhere. Colouring CDDM's two colour channels red
# and green to match the pictogram implied that a channel carries a hue, which it does not - the
# colour feature is a pair of evidence channels like any other.
CUE_COL, SENS_COL, TONIC_COL = ps.MUTED, ps.INK, ps.FAINT

# `run` is a glob matching the run folders of the very cells the manuscript reports, so the task
# shown is the task trained. Channel labels are asserted against the config's channel counts, so a
# config change breaks the script instead of silently mislabelling a panel.
#
# `braces` group the channels: (first_row, last_row, label, level), rows counted over inputs then
# outputs, level 0 nearest the labels. A brace is drawn only where grouping adds something - the
# flip-flop's channels are one per bit and need none.
TASKS = {
    "CDDM": dict(
        run=f"{DATA_DIR}/CDDM_paper_grid/EqType=h_N=1000_arm=control",
        title="CDDM, context-dependent decision making", tcol=ps.TASK_CDDM,
        inputs=["motion", "colour", "right", "left", "right", "left"],
        in_cols=[CUE_COL, CUE_COL, SENS_COL, SENS_COL, SENS_COL, SENS_COL],
        outputs=["right", "left"],
        braces=[(2, 3, "motion", 0), (4, 5, "colour", 0),
                (0, 1, "cue", 1), (2, 5, "sensory", 1)],
    ),
    "NBitFlipFlop": dict(
        # the first 40k leg of the warm-start chain behind the 200k size series; every leg of
        # that chain saves the same task block, and this leg is the one kept on the Mac
        run=f"{DATA_DIR}/NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none",
        title="3BFF, 3-bit flip-flop", tcol=ps.TASK_3BFF,
        inputs=["bit 1", "bit 2", "bit 3"],
        in_cols=[SENS_COL] * 3,
        outputs=["bit 1", "bit 2", "bit 3"],
        braces=[(0, 2, "pulses", 0), (3, 5, "state held", 0)],
    ),
    # The 5-tau delay of the size series (DMTS_paper_grid_d5, 150k, finished): two stimulus
    # identities and a tonic channel, the 2026-09-21 design.
    "DMTS": dict(
        run=f"{DATA_DIR}/DMTS_paper_grid_d5/EqType=h_N=1000_arm=control",
        title="DMTS, delayed match-to-sample", tcol=ps.TASK_DMTS,
        inputs=["1", "2", "tonic", "go"],
        in_cols=[SENS_COL, SENS_COL, TONIC_COL, CUE_COL],
        outputs=["match", "non-match"],
        braces=[(0, 1, "stimuli", 0)],
    ),
    # The maze (issue #3). The whole agent is the RNN, so the stack shows what the environment gives
    # it and the two actions it reads out; there is no target, hence `out_word`. `inputs`, `in_cols`
    # and `braces` are filled in by build_maze from the run's own environment, which knows whether
    # the efference copy is on.
    "Maze": dict(
        run=MAZE_RUN,
        title="Maze running", tcol=ps.TASK_MAZE,
        outputs=["turn", "speed"], out_word="output",
    ),
}


def build(name):
    """Instantiate one task from a TRAINED RUN's own config and return one representative trial.

    The config comes from the run folder rather than from `configs/task/`, because those two can
    disagree - DMTS does - and a figure built from the wrong one is a figure of a different task.

    The trial is chosen, not drawn at random, so that each row shows the case that defines the
    task: CDDM with the two features DISAGREEING (the only trials that separate this task from a
    plain decision), and DMTS on a MATCH trial (the only trials whose match channel is non-zero).

    Args:
        name: key of TASKS, which is also the task name in the config.
    Returns:
        (inputs, targets, epochs, dt_over_tau, pending): the two streams, the epoch list (empty
        for the flip-flop), the step length in units of tau, and whether the config came from
        configs/task/ because no run of this task exists yet.
    """
    spec = TASKS[name]
    folders = sorted(glob.glob(os.path.join(spec["run"], "*", "")))
    if folders:
        cfg = OmegaConf.load(sorted(glob.glob(os.path.join(folders[0], "*_config.yaml")))[0])
        cfg_task = cfg.task
        dt_over_tau, pending = float(cfg.model.dt) / float(cfg.model.tau), False
    elif spec.get("cfg"):
        # a task whose runs have not landed yet: read the config they WILL use and mark the row, so
        # the panel is never quietly a picture of a superseded design
        cfg_task = OmegaConf.load(os.path.join(CFG_DIR, spec["cfg"]))
        dt_over_tau, pending = DT_TAU, True
    else:
        raise FileNotFoundError(f"no run folder under {spec['run']} and no fallback config named")
    cfg_task.seed = TASK_SEED
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg_task, dt=1.0))

    if name == "CDDM":
        # attend motion, motion favours right (+0.5), colour favours left (-0.5): a conflict trial
        inp, tgt = task.generate_input_target_stream("motion", 0.5, -0.5)
        epochs = [(task.cue_on, task.cue_off, "context cue"),
                  (task.stim_on, task.stim_off, "stimulus"),
                  (task.dec_on, task.dec_off, "decision")]
    elif name == "DMTS":
        inp, tgt, c = task.generate_input_target_stream(0, 0)      # sample 1, test 1 -> match
        # from the trial's OWN condition dict, so the bars sit on the jittered onsets actually drawn
        epochs = [(c["sample_on"], c["sample_off"], "sample"),
                  (c["sample_off"], c["match_on"], "delay"),
                  (c["match_on"], c["match_off"], "test"),
                  (c["dec_on"], c["dec_off"], "decision")]
    else:
        inp, tgt, _ = task.generate_input_target_stream()
        epochs = []

    inp, tgt = np.asarray(inp, float), np.asarray(tgt, float)
    assert len(spec["inputs"]) == inp.shape[0], \
        f"{name}: {inp.shape[0]} input channels in the config, {len(spec['inputs'])} labels"
    assert len(spec["outputs"]) == tgt.shape[0], \
        f"{name}: {tgt.shape[0]} output channels in the config, {len(spec['outputs'])} labels"
    return inp, tgt, epochs, dt_over_tau, pending


def build_maze():
    """One clean episode of a trained maze agent: its inputs, its outputs and the scene to draw.

    The run is the first seed under MAZE_RUN, rebuilt by trainRNNbrain.rl.saved.load_run and played
    noise-free with the policy mean as the action on the training probe's drops
    (cfg.ppo.probe_episodes episodes from cfg.ppo.probe_seed). Of the episodes that reach the
    cheese, the one with the median number of decisions is shown. With no run folder yet, the
    config the runs will use (configs/rl/maze_ppo.yaml) is composed, an untrained network is
    played the same way and the row is marked pending; an untrained agent does not reach the
    cheese, so its episode is cut at PENDING_DECISIONS.

    Returns:
        (inp, tgt, dt_over_tau, pending, spec, scene):
        inp (n_inputs, n_dec): the observation each decision answered, the range sensors in
            [0, 1] (1 at a wall, 0 at open space) and then, when the run has it, the efference
            copy, turn / turn_max in [-1, 1] and speed / v_max in [0, 1];
        tgt (2, n_dec): the executed turn / turn_max and speed / v_max, the two read-out rows after
            the environment's tanh and sigmoid, so that they share the inputs' scale and the
            efference copy repeats them one decision later;
        dt_over_tau: the length of one decision in membrane time constants, substeps * dt / tau;
        pending: bool, no run folder yet;
        spec: the TASKS["Maze"] entry with `inputs`, `in_cols` and `braces` filled in;
        scene: dict for pic_maze with wall (H, W) bool, goal (x, y) in cells, xy (n_dec + 1, 2)
            the path in cells, heading (n_dec + 1,) radians, rays (n_rays,) the distances the
            sensors measured at the drop in cells, ray_offsets (n_rays,) radians, and reached
            (bool, the episode ends at the cheese).
    """
    folders = sorted(glob.glob(os.path.join(MAZE_RUN, "s*", "")))
    if folders:
        cfg, env, policy = load_run(folders[0])
        pending = False
    else:
        config_dir = os.path.abspath(os.path.join(CFG_DIR, ".."))
        with hydra.initialize_config_dir(version_base="1.3", config_dir=config_dir):
            cfg = hydra.compose(config_name=MAZE_CFG)
        env = MazeEnv(**dict(cfg.env), device=torch.device("cpu"), seed=int(cfg.seed) + 1)
        args = filter_kwargs(RNN_torch, cfg.model)
        args.seed, args.n_inputs = int(cfg.seed), env.n_inputs
        args.n_outputs = env.n_actions + (1 if cfg.ppo.critic else 0)
        policy = MazePolicy(hydra.utils.instantiate(args), cfg.ppo.action_std, cfg.ppo.substeps)
        pending = True
    assert not env.goal_cue, \
        "the maze row labels sensors and the efference copy only; this run has a goal cue"
    g = torch.Generator()
    g.manual_seed(int(cfg.ppo.probe_seed))
    out = rollout(env, policy, int(cfg.ppo.probe_episodes), int(cfg.ppo.T), clean=True, generator=g)
    steps, success = out["steps"].int(), out["success"]
    if success.any():
        ok = torch.nonzero(success).flatten()
        ok = ok[torch.argsort(steps[ok])]
        b = int(ok[len(ok) // 2])                       # the successful episode of median length
        n, reached = int(steps[b]), True
    else:
        b, n, reached = 0, min(PENDING_DECISIONS, int(cfg.ppo.T)), False

    inp = out["u"][:, :n, b].numpy()
    act = out["actions"][:, :n, b]
    tgt = np.stack([torch.tanh(act[0]).numpy(), torch.sigmoid(act[1]).numpy()])
    dt_over_tau = policy.substeps * float(cfg.model.dt) / float(cfg.model.tau)
    angles = [f"{int(round(np.degrees(float(o))))}\u00b0" for o in env.ray_offsets]   # 0°, 45°, ...
    feedback = ["turn", "speed"] if env.action_feedback else []
    spec = dict(TASKS["Maze"],
                inputs=angles + feedback,
                in_cols=[SENS_COL] * env.n_rays + [CUE_COL] * len(feedback),
                braces=[(0, env.n_rays - 1, "sensors", 0)]
                       + ([(env.n_rays, env.n_rays + 1, "efference\ncopy", 0)] if feedback else []))
    assert len(spec["inputs"]) == inp.shape[0], \
        f"Maze: {inp.shape[0]} input channels in the run, {len(spec['inputs'])} labels"
    scene = dict(wall=env.wall.numpy(), goal=env.goal_xy.numpy(),
                 xy=out["xy"][:n + 1, b].numpy(), heading=out["heading"][:n + 1, b].numpy(),
                 rays=(1.0 - inp[:env.n_rays, 0]) * env.ray_max, ray_offsets=env.ray_offsets.numpy(),
                 reached=reached)
    return inp, tgt, dt_over_tau, pending, spec, scene


def brace(ax, x, y0, y1, label, col=ps.MUTED, w=BRACE_W, lw=0.6, fs=ps.TICK_PT, rot=0):
    """A curly brace left of the channel labels, opening right, with its label beyond the tip.

    Drawn in `ax.get_yaxis_transform()`, i.e. x in axes fractions (negative is left of the axes)
    and y in data units, so a brace spans exactly the channel rows it groups however the figure is
    resized.

    Args:
        ax: the channel-stack axes; x: the brace's right edge in axes fractions; y0, y1: the data
            y values of the outermost rows it spans, in either order; label: text beyond the tip;
            col: colour; w: brace depth in axes fractions; lw: line width; fs: label font size;
            rot: label rotation in degrees - the outer level is set upright so that its label
            costs almost no horizontal room, which is what keeps the braces out of the pictogram
            in the next column.
    Returns:
        None.
    """
    y0, y1 = (y0, y1) if y0 < y1 else (y1, y0)
    ym, q = 0.5 * (y0 + y1), 0.22 * (y1 - y0)
    verts = [(x, y0), (x - w, y0), (x - w, y0 + q),         # bottom hook
             (x - w, ym - q),                                # up the spine
             (x - w, ym), (x - 2 * w, ym),                   # out to the tip
             (x - w, ym), (x - w, ym + q),                   # back in
             (x - w, y1 - q),                                # up the spine
             (x - w, y1), (x, y1)]                           # top hook
    codes = [Path.MOVETO, Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3,
             Path.CURVE3, Path.CURVE3,
             Path.LINETO,
             Path.CURVE3, Path.CURVE3]
    ax.add_patch(PathPatch(Path(verts, codes), facecolor="none", edgecolor=col, lw=lw,
                           transform=ax.get_yaxis_transform(), clip_on=False, zorder=5))
    if rot:
        ax.text(x - 2 * w - 0.020, ym, label, transform=ax.get_yaxis_transform(), rotation=rot,
                ha="center", va="center", fontsize=fs, color=col, clip_on=False)
    else:
        ax.text(x - 2 * w - 0.012, ym, label, transform=ax.get_yaxis_transform(), ha="right",
                va="center", fontsize=fs, color=col, clip_on=False)


def channel_stack(ax, inp, tgt, epochs, dt_over_tau, spec, pending=False):
    """Draw one task's input and target channels as a labelled stack over time.

    Args:
        ax: axes; inp: (n_inputs, n_steps); tgt: (n_outputs, n_steps); epochs: list of
            (start_step, end_step, label) from the run config, possibly empty; dt_over_tau: step
            length in membrane time constants; spec: the TASKS entry, supplying `inputs`,
            `in_cols`, `outputs`, `braces` and, for a task without a target, `out_word` (the
            lower block's word, "target" unless given); pending: the config has not been trained
            yet, which the panel says out loud.
    Returns:
        None.
    """
    n_in, n_steps = inp.shape
    n_out = tgt.shape[0]
    t = np.arange(n_steps) * dt_over_tau                    # time in membrane time constants
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
        cols.append(spec["tcol"])

    for b, sig, col in zip(bases, np.vstack([inp, tgt]), cols):
        ax.plot([t[0], t[-1]], [b, b], lw=0.45, color="#b9b8b1", ls=(0, (1.2, 2)), zorder=1)
        ax.plot(t, b + amp * sig / span, lw=0.9, color=col, zorder=3, solid_joinstyle="miter")

    # the divider, and the two words that say which block is given to the network and which is
    # asked of it
    mid = (bases[n_in - 1] + bases[n_in]) / 2
    ax.axhline(mid, color=ps.FAINT, lw=0.5, ls=(0, (3, 3)), zorder=2)
    ax.text(t[-1] * 1.015, np.mean(bases[:n_in]), "input", rotation=-90, ha="left", va="center",
            fontsize=ps.TICK_PT, color=ps.MUTED)
    ax.text(t[-1] * 1.015, np.mean(bases[n_in:]), spec.get("out_word", "target"), rotation=-90,
            ha="left", va="center", fontsize=ps.TICK_PT, color=spec["tcol"])

    # CDDM's epochs are NESTED (the cue is on for the whole trial, the stimulus for two thirds of
    # it), so they cannot share one line the way DMTS's sequential ones can. Greedy packing: an
    # epoch goes on the lowest level where it does not overlap one already there.
    top = bases[0] + 0.62
    levels = []
    for s0, s1, lab in epochs:
        x0, x1 = s0 * dt_over_tau, s1 * dt_over_tau
        lv = next((k for k, occ in enumerate(levels) if x0 >= occ - 1e-9), len(levels))
        if lv == len(levels):
            levels.append(x1)
        else:
            levels[lv] = x1
        yb = top + 0.28 + 1.10 * lv
        ax.plot([x0, x1], [yb] * 2, lw=1.6, color=ps.FAINT, zorder=3, solid_capstyle="butt")
        ax.plot([x0, x0], [yb - 0.16, yb + 0.16], lw=0.5, color=ps.FAINT, zorder=3)
        ax.text((x0 + x1) / 2, yb + 0.06, lab, ha="center", va="bottom", fontsize=ps.TICK_PT,
                color=ps.MUTED)
    head = 0.28 + 1.10 * max(len(levels) - 1, 0) + 0.62 if epochs else 0.2

    # grouping braces. Level 0 sits next to the labels; level 1 encloses it, which is the only
    # order in which "sensory" can contain "motion" and "colour".
    for r0, r1, lab, level in spec.get("braces", []):
        brace(ax, BRACE_X[level], bases[r0], bases[r1], lab, rot=90 * (level > 0))

    if pending:
        ax.text(0.0, 1.0, "config not yet trained", transform=ax.transAxes, ha="left",
                va="bottom", fontsize=ps.TICK_PT, color=ps.INK)
    ax.set(yticks=bases, xlabel=r"time ($\tau$)",
           xlim=(0, t[-1] * 1.01), ylim=(bases[-1] - 0.62, top + head))
    # the same integer ticks (0, 5, 10, ...) on every row; a maze episode runs several times longer
    # than a supervised trial, so the step grows to keep at most eight intervals
    ax.xaxis.set_major_locator(MultipleLocator(next(s for s in (5, 10, 20, 50, 100) if t[-1] / s <= 8)))
    ax.set_yticklabels(ticks, fontsize=ps.TICK_PT)
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

    ax.text(0.03, 0.860, "attend", ha="left", va="center", fontsize=ps.TICK_PT, color=ps.INK)
    ps.box(ax, 0.29, 0.785, 0.31, 0.120, "MOTION", col=ps.INK, face="#f2f1ec", lw=0.7,
           fs=ps.TICK_PT, text_col=ps.INK)
    ps.box(ax, 0.64, 0.785, 0.31, 0.120, "COLOUR", col=ps.FAINT, lw=0.5, ls=(0, (2, 2)), fs=ps.TICK_PT,
           text_col=ps.FAINT)
    ax.text(0.50, 0.995, "the cue selects one of the two features", ha="center", va="top",
            fontsize=ps.TICK_PT, color=ps.MUTED)

    # the stimulus: a cloud of arrows. Direction is the motion feature, colour is the colour
    # feature, and on this trial they point opposite ways - which is the whole task.
    ps.box(ax, 0.13, 0.410, 0.74, 0.35, col=ps.MUTED, face=ps.PAPER, lw=0.7)
    # Arrows sit on a jittered 6 x 4 grid, so no two overlap and every direction can be read.
    rng = np.random.default_rng(4)
    gx, gy = np.meshgrid(np.linspace(0.21, 0.79, 6), np.linspace(0.47, 0.70, 4))
    xs = gx.ravel() + rng.uniform(-0.025, 0.025, gx.size)
    ys = gy.ravel() + rng.uniform(-0.015, 0.015, gy.size)
    right = rng.permutation(np.arange(gx.size) < 17)    # motion evidence: 17 of 24 rightward
    col_a = rng.permutation(np.arange(gx.size) < 7)     # colour evidence: 7 of 24 colour A
    for x, y, r, a in zip(xs, ys, right, col_a):
        d = 0.045 if r else -0.045
        ps.arrow(ax, (x - d / 2, y), (x + d / 2, y), col=COL_A if a else COL_B, lw=0.8,
                 mutation_scale=4.5, shrink=0, zorder=4)
    ax.text(0.50, 0.390, "most arrows point RIGHT (motion);\nmost are light grey (colour: LEFT)",
            ha="center", va="top", fontsize=ps.TICK_PT, color=ps.MUTED)

    # LEFT on the left and RIGHT on the right: the choice is a direction, so putting the winning
    # option on the wrong side of the panel fights the reader for no reason. Laid out like the cue
    # row above it - the verb outside, one word in each box.
    ax.text(0.03, 0.130, "choose", ha="left", va="center", fontsize=ps.TICK_PT, color=ps.INK)
    ps.box(ax, 0.29, 0.070, 0.31, 0.120, "LEFT", col=ps.FAINT, lw=0.5, ls=(0, (2, 2)),
           fs=ps.TICK_PT, text_col=ps.FAINT)
    ps.box(ax, 0.64, 0.070, 0.31, 0.120, "RIGHT", col=ps.TASK_CDDM, lw=0.9, fs=ps.TICK_PT,
           text_col=ps.TASK_CDDM)
    ax.text(0.50, 0.045, "colour alone points LEFT; the cue\nselects motion, so the target is RIGHT",
            ha="center", va="top", fontsize=ps.TICK_PT, color=ps.MUTED, linespacing=1.3)


def pic_flipflop(ax):
    """Pictogram: one bit as a latch driven by signed pulses, times three.

    Args:
        ax: blank axes.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.50, 0.995, "each bit is a latch", ha="center", va="top", fontsize=ps.TICK_PT,
            color=ps.MUTED)

    # one bit, in full: the pulse train and the state it leaves behind
    pulses = [(0.33, +1), (0.52, -1), (0.78, +1)]
    y_in, y_out = 0.775, 0.515
    ax.plot([0.22, 0.97], [y_in] * 2, lw=0.5, color=ps.GRID, zorder=1)
    for x, s in pulses:
        ax.plot([x, x], [y_in, y_in + 0.085 * s], lw=1.1, color=ps.INK, zorder=3)
        ax.plot([x], [y_in + 0.085 * s], "o", ms=2.2, color=ps.INK, zorder=4)
    ax.text(0.20, y_in, "pulses in", ha="right", va="center", fontsize=ps.TICK_PT, color=ps.INK)

    # the held state: a step that changes only when a pulse arrives
    edges = [0.22] + [x for x, _ in pulses] + [0.97]
    state = [0] + [s for _, s in pulses]        # 0 before the first pulse, as the real target is
    for k in range(len(edges) - 1):
        ax.plot([edges[k], edges[k + 1]], [y_out + 0.11 * state[k]] * 2, lw=1.2,
                color=ps.TASK_3BFF, zorder=3)
        if k:
            ax.plot([edges[k]] * 2, [y_out + 0.11 * state[k - 1], y_out + 0.11 * state[k]],
                    lw=1.2, color=ps.TASK_3BFF, zorder=3)
    ax.text(0.20, y_out, "state\nheld", ha="right", va="center", fontsize=ps.TICK_PT,
            color=ps.TASK_3BFF, linespacing=1.25)
    for lev, lab in [(0.11, "+1"), (0.0, "0"), (-0.11, "$-$1")]:
        ax.text(0.99, y_out + lev, lab, ha="left", va="center", fontsize=ps.TICK_PT, color=ps.TASK_3BFF)
    ax.text(0.52, 0.375, "the state changes only when a pulse arrives,\n"
                         "and holds for as long as none does",
            ha="center", va="top", fontsize=ps.TICK_PT, color=ps.MUTED, linespacing=1.3)

    for k in range(3):
        ps.box(ax, 0.20 + 0.20 * k, 0.090, 0.16, 0.100, f"bit {k + 1}", col=ps.INK, lw=0.6,
               fs=ps.TICK_PT, face="#f2f1ec", text_col=ps.INK)
    ax.text(0.50, 0.060, "3 bits, independent, one channel each", ha="center", va="top",
            fontsize=ps.TICK_PT, color=ps.MUTED)


def pic_dmts(ax):
    """Pictogram: sample, empty delay, test, and the same/different judgement on two channels.

    Two stimulus identities exist, so the four sample-test pairs are two matches and two
    non-matches - a balanced batch. With the four identities of the superseded design the split was
    4 of 16, and always answering "non-match" scored 75%.

    Args:
        ax: blank axes.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.50, 0.995, "hold the sample across an empty delay", ha="center", va="top",
            fontsize=ps.TICK_PT, color=ps.MUTED)

    y = 0.710
    for x, lab, filled in [(0.07, "sample", True), (0.37, "delay", False), (0.67, "test", True)]:
        ps.box(ax, x, y, 0.25, 0.19, col=ps.MUTED, face=ps.PAPER, lw=0.7)
        if filled:
            ax.plot([x + 0.125], [y + 0.095], "s", ms=6.5, color=SENS_COL, zorder=4)
        else:
            ax.text(x + 0.125, y + 0.095, "?", ha="center", va="center", fontsize=9.0,
                    color=ps.FAINT)
        ax.text(x + 0.125, y - 0.022, lab, ha="center", va="top", fontsize=ps.TICK_PT, color=ps.MUTED)
    for x in (0.325, 0.625):
        ps.arrow(ax, (x, y + 0.095), (x + 0.04, y + 0.095), col=ps.MUTED, lw=0.7, mutation_scale=5)

    # two identities, so the four sample-test pairs split evenly
    ax.text(0.14, 0.530, "one of two stimuli:", ha="left", va="center", fontsize=ps.TICK_PT,
            color=ps.MUTED)
    for k in range(2):
        ax.plot([0.66 + 0.095 * k], [0.530], "s", ms=5.0, zorder=4,
                color=SENS_COL if k == 0 else ps.FAINT)
    ax.text(0.50, 0.430, "4 sample-test pairs: 2 match, 2 non-match", ha="center", va="center",
            fontsize=ps.TICK_PT, color=ps.MUTED)

    ps.arrow(ax, (0.50, 0.375), (0.50, 0.320), col=ps.MUTED, lw=0.8, shrink=0)
    ps.box(ax, 0.14, 0.185, 0.72, 0.115, "same as the sample?", col=ps.INK, lw=0.6, fs=ps.TICK_PT,
           face="#f2f1ec", text_col=ps.INK)
    ax.text(0.455, 0.110, "match: channel 1", ha="right", va="center",
            fontsize=ps.TICK_PT, color=ps.TASK_DMTS)
    ax.text(0.545, 0.110, "non-match: channel 2", ha="left", va="center",
            fontsize=ps.TICK_PT, color=ps.TASK_DMTS)
    ax.text(0.50, 0.010, "both channels held at 0 until the go cue", ha="center", va="center",
            fontsize=ps.TICK_PT, color=ps.MUTED)


def pic_maze(ax, scene, tcol):
    """Pictogram: the maze, the cheese, one agent with its range sensors at its drop, and its path.

    Cells are drawn square whatever the panel's shape: a cell's width in axes fractions is its
    height divided by the panel's width-to-height ratio on the page. Rows run top down, as the
    config writes them, so the picture is the config's maze as typed. A key beside the maze names
    the four glyphs, since none of them is self-explanatory at 1.5 mm per cell.

    Args:
        ax: blank axes, already placed by the GridSpec; scene: dict from build_maze (wall, goal,
            xy, heading, rays, ray_offsets); tcol: the task colour, for the path and the cheese.
    Returns:
        None.
    """
    ps.blank(ax)
    ax.set(xlim=(0, 1), ylim=(0, 1))
    ax.text(0.50, 0.995, "find the cheese from the walls alone", ha="center", va="top",
            fontsize=ps.TICK_PT, color=ps.MUTED)

    wall, xy, heading = scene["wall"], scene["xy"], scene["heading"]
    n_rows, n_cols = wall.shape
    pos, (fig_w, fig_h) = ax.get_position(), ax.figure.get_size_inches()
    ratio = (pos.width * fig_w) / (pos.height * fig_h)       # the panel's width over its height
    y_top, y_bot, x_left = 0.84, 0.12, 0.02                  # the maze's extent, axes fractions
    dy = (y_top - y_bot) / n_rows
    dx = dy / ratio
    # the environment's (x, y) in cells -> axes fractions: x grows to the right, y (the row) downward
    for r, c in zip(*np.nonzero(wall)):
        ax.add_patch(Rectangle((x_left + c * dx, y_top - (r + 1) * dy), dx, dy, facecolor=ps.MUTED,
                               edgecolor=ps.MUTED, lw=0.3, zorder=1))
    ax.plot(x_left + xy[:, 0] * dx, y_top - xy[:, 1] * dy, lw=0.7, color=tcol, zorder=3,
            solid_joinstyle="round")
    gx, gy = scene["goal"]
    ax.plot([x_left + gx * dx], [y_top - gy * dy], "*", ms=6.0, color=tcol, zorder=5)
    # the agent at its drop: the sensor rays end where each one met a wall, the thicker tick is the
    # heading (ray 0 lies under it)
    x0, y0 = x_left + xy[0, 0] * dx, y_top - xy[0, 1] * dy
    for d, off in zip(scene["rays"], scene["ray_offsets"]):
        a = heading[0] + off
        ax.plot([x0, x0 + d * np.cos(a) * dx], [y0, y0 - d * np.sin(a) * dy], lw=0.4, color=ps.INK,
                zorder=4)
    ax.plot([x0, x0 + 0.5 * np.cos(heading[0]) * dx], [y0, y0 - 0.5 * np.sin(heading[0]) * dy],
            lw=0.9, color=ps.INK, zorder=6)
    ax.plot([x0], [y0], "o", ms=2.8, color=ps.INK, zorder=7)

    # the key, right of the maze: one glyph and one phrase per element. The panel is 48 mm wide
    # and the maze takes 25 of them, so each phrase is at most 17 mm at 6 pt (measured)
    kx = x_left + n_cols * dx + 0.04
    tx = kx + 0.052
    ky = [0.78, 0.63, 0.48, 0.33]
    ax.plot([kx], [ky[0]], "o", ms=2.8, color=ps.INK, zorder=7)
    ax.plot([kx, kx + 0.035], [ky[0]] * 2, lw=0.9, color=ps.INK, zorder=6)
    ax.text(tx, ky[0], "agent and heading", ha="left", va="center", fontsize=ps.TICK_PT, color=ps.INK)
    for a in (-0.5, 0.0, 0.5):
        ax.plot([kx, kx + 0.035 * np.cos(a)], [ky[1], ky[1] + 0.035 * np.sin(a) * ratio], lw=0.4,
                color=ps.INK)
    ax.text(tx, ky[1], "8 range sensors", ha="left", va="center", fontsize=ps.TICK_PT, color=ps.INK)
    ax.plot([kx - 0.02, kx + 0.025], [ky[2]] * 2, lw=0.7, color=tcol)
    ax.text(tx, ky[2], "path taken", ha="left", va="center", fontsize=ps.TICK_PT, color=tcol)
    ax.plot([kx], [ky[3]], "*", ms=6.0, color=tcol, zorder=5)
    ax.text(tx, ky[3], "cheese: the goal", ha="left", va="center", fontsize=ps.TICK_PT, color=tcol)


def check_text(fig):
    """Report text below the house minimum and pairs of overlapping text boxes.

    Args:
        fig: the drawn figure (drawn once so that every text has a window extent).
    Returns:
        (list of strings under ps.MIN_PT, list of overlapping text pairs as string tuples).
    """
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    texts = [t for t in fig.findobj(plt.Text) if t.get_visible() and t.get_text().strip()]
    small = [t.get_text() for t in texts if t.get_fontsize() < ps.MIN_PT]
    boxes = [(t.get_text(), t.get_window_extent(r)) for t in texts]
    over = [(a, b) for k, (a, ba) in enumerate(boxes) for b, bb in boxes[k + 1:]
            if ba.overlaps(bb) and ba.intersection(ba, bb).width * ba.intersection(ba, bb).height > 4]
    return small, over


def main():
    """Assemble Supplementary Fig. 1 at its printed size (165.1 x H_MM mm) and write it.

    Returns:
        the path of the PDF.
    """
    ps.setup()
    fig = ps.figure(H_MM)
    # explicit margins: the gutter holds both brace levels and their labels, which hang left of the
    # channel axes by about a quarter of its width; the right margin holds "input" / "target"
    gs = GridSpec(4, 2, figure=fig, width_ratios=[0.80, 1.30], height_ratios=[1, 1, 1, MAZE_ROW],
                  hspace=0.58, wspace=0.40, left=0.045, right=0.965, top=0.957, bottom=0.052)

    # rows in the order Methods and Results introduce the tasks: 3BFF, CDDM, DMTS, then the maze
    for row, (name, pic, letter) in enumerate([("NBitFlipFlop", pic_flipflop, "a"),
                                               ("CDDM", pic_cddm, "b"),
                                               ("DMTS", pic_dmts, "c"),
                                               ("Maze", pic_maze, "d")]):
        ax_p, ax_c = fig.add_subplot(gs[row, 0]), fig.add_subplot(gs[row, 1])
        if name == "Maze":
            inp, tgt, dt_over_tau, pending, spec, scene = build_maze()
            epochs = []
            pic(ax_p, scene, spec["tcol"])
        else:
            inp, tgt, epochs, dt_over_tau, pending = build(name)
            spec = TASKS[name]
            pic(ax_p)
        ax_p.set_title(spec["title"], fontsize=ps.LABEL_PT, fontweight="bold",
                       color=ps.INK, loc="left", pad=14)
        ps.panel_letter(ax_p, letter, x_mm=6.0, y_mm=4.9)
        channel_stack(ax_c, inp, tgt, epochs, dt_over_tau, spec, pending)
        if name == "Maze":
            # the episode's end: the cheese, or the cut of an untrained agent's wandering
            t_end = (inp.shape[1] - 1) * dt_over_tau
            ax_c.axvline(t_end, color=spec["tcol"], lw=ps.THIN_LW, ls=(0, (1.5, 1.5)), zorder=2)
            ax_c.text(t_end, 1.0, "cheese reached" if scene["reached"] else "episode cut",
                      transform=ax_c.get_xaxis_transform(), ha="right", va="bottom",
                      fontsize=ps.TICK_PT, color=spec["tcol"])
        print(f"  {name:13} {inp.shape[0]} inputs x {inp.shape[1]} steps "
              f"({inp.shape[1] * dt_over_tau:.0f} tau) -> {tgt.shape[0]} outputs; "
              f"epochs {[(a, b) for a, b, _ in epochs]}"
              + (f"; episode {'reaches' if scene['reached'] else 'does not reach'} the cheese"
                 if name == "Maze" else ""))
    small, over = check_text(fig)
    print(f"  text under {ps.MIN_PT} pt: {small or 'none'}; overlapping text: {over or 'none'}")
    return ps.save(fig, "fig_supp_tasks")


if __name__ == "__main__":
    main()
