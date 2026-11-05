#!/usr/bin/env python3
"""
Manuscript Figure 1 (round 2 of the paper push, 1 Nov 2026; dead-relu-paper#52). Training splits a
network's units into an active and an inactive group, and a network retrained without its inactive
units scores as the intact network retrained the same way.

  (a) one 3-bit flip-flop (3BFF) network (N = 1,000, 200,000 iterations): schematic and 8 random units
  (b) participation of every unit before and after training, with the criterion. The criterion is
      not claimed to sit in an empty valley: halving or doubling its factor moves the active count
      by 3-28% (dead-relu-paper#13, notebook 2026-10-31_1657).
  (c) active units at N = 1,000 on the three main-text tasks (MAIN_TASKS: 3BFF, CDDM and the maze;
      round 14), every supervised run at its own floor crossing with held-out R^2 there and every
      maze run at its "learned" probe with the probe success rate there (panel_tasks). All four 3BFF
      networks are drawn (decision dead-relu-paper#20).
  (d) the score as units are removed (held-out R^2 for 3BFF and CDDM, the success rate for the maze,
      on a y axis of its own), three networks per supervised task and every ablated maze network;
      one sub-panel per task, side by side, the legend above them and off the data. Round 3
      (dead-relu-paper#85, #83): the left
      spine sits 1.4 mm outward so the busiest-first curve is clear of it; the inactive units are a
      grey bar along the top, not a band behind the curves; the retrained networks are a strip right
      of each sub-panel, stars (inactive removed) and triangles (as many removed at random), every
      mark filled because fill means the floor crossing in every figure. Inactive units in (a) are
      light grey dots, no longer hollow.
  (e) round 7 (dead-relu-paper#127, plan risk R1): the fraction of units selective for a task
      variable in the 3BFF and CDDM networks of (d), computed over all units and over the active
      units only (panel_selective). Read from data/population_stats.json (population_stats.py;
      notebook 2026-10-31_2220), the file Extended Data Fig. 2 draws; nothing is recomputed here.
      Filled markers: the weights are the floor-crossing snapshots of (d). The maze column reads
      "no data" until that file holds maze networks. Panel (e) sits under (c), and (d) spans the
      height of both.

ROUND 14 (dead-relu-paper#182, Pavel's decision of 4 Nov). The maze-running task (an agent trained
by PPO to reach the cheese in a fixed maze; engellab/trainRNNbrain#3) replaces DMTS as the third
task of panels (c), (d) and (e); DMTS moves to Extended Data. MAIN_TASKS lists the three tasks
those panels draw; PANEL_TASKS and ABLATION keep DMTS because collect() and fig_cosyne2027 read
them. The maze is read when the figure is drawn, from this repo and not from COLLECT: its N = 1,000
ReLU runs (MAZE_ROOT, common.maze_load_runs; maze_rows) and the ablation JSONs of rl_maze_ablate.py
(MAZE_ABLATION_DIR; maze_ablation). Panel (c) draws each maze run at its "learned" probe, the first
probe from which the clean probe's success rate stays at or above 0.9 for 5 probes
(common.maze_learned_index), the maze counterpart of the floor crossing: 511, 434 and 410 active
units of 1,000 for seeds 1, 2 and 3 (at the end of training, update 2,000, the counts are 455, 423
and 404; those are not drawn). Its tick label gives the probe success rate at that probe (SCORE),
not an R^2. Panel (d)'s maze sub-panel draws the success rate under the training noise against
units removed, in the three orders of the other sub-panels, on a y axis of its own labelled
"success rate" (D_YLABEL_MM of room before it); the ablated weights are the final ones, because no
earlier snapshot of a maze run is saved. The maze's prune-and-retrain (rl_maze_retrain.py: the
inactive units removed, or as many at random, or none, and the agent trained 200 further PPO
updates; MAZE_RETRAIN_DIR, maze_retrain) fills its retrain strip in (d), and its selectivity
(rl_maze_population_stats.py, the partial-R^2 rule of population_stats.py against the turn and
speed commands, the distance to the cheese and the heading; data/population_stats_maze.json,
load_population_stats_all) fills its column in (e). Both are read at the end of training, a fixed
budget, so the maze's markers in the (d) strip and in (e) are OPEN (mark_fixed), where the
supervised networks' are filled (their floor-crossing snapshots; #83). Either reads "no data" until
its file exists.

ROUND 13 (dead-relu-paper#165). The 24 trials of (a) and (b) are drawn from a seeded task: the
flip-flop task draws its trials from its own generator, seeded from cfg.task.seed (null in every
training config), so np.random.seed alone left the batch different on every call. Panels (a) and (b)
are re-collected with `--collect-example` into COLLECT; the other panels keep the record of Della
job 16556248.

ROUND 2 STYLE (dead-relu-paper#31, #40; paperstyle.py). Drawn at the printed width (W_FULL_MM).
One meaning per colour: ACTIVE (near-black) and INACTIVE (light grey) for unit classes in (a), (b)
and the inactive band of (d); task colours only in (c) and (d); text in ink; the excitatory /
inhibitory colour code of (a) is gone (every sampled connection is drawn in one neutral grey).

TWO STAGES. `--collect` runs on Della (the networks, snapshots and ablation results live there) and
pickles every input the figure draws to COLLECT; without the flag the script draws from that pickle,
on the Mac, because Della's default font widens every legend. Copy the pickle back between them.

SIZE AND TRAINING TIME MOVED TO FIGURE 2. The panels that drew active units against N (panel_c) and
loss and inactive units over training (panel_e), with every helper they use, stay in this module
because the Figure 2 script imports them; Figure 1 no longer calls them. The text below this line
describes the pre-round-1 five-panel figure and is kept as the record of how those panels were built.

Manuscript Figure 1, until 31 Oct - THE PHENOMENON. Most units of a trained ReLU RNN never fire, the
ones that do not fire do no work, the bigger the network the larger the silent share, and the
silencing goes on after the loss has settled.

RESTRUCTURED 2026-10-12. Panel (d) used to be "no knob fixes it" - every standard knob as a change
from its own reference. That is a remedy result and moves to Figure 2; its builder is kept here as
`panel_knobs` (fig_slides imports this module's helpers). Panel (d) is now the ablation that makes
"silent" mean "does no work". Every panel reads STANDARD networks (ReLU, bias fixed at 0,
self-connections on, no Dale, gamma 0): the trainable-bias sweeps the old version drew are gone from
both clusters.

  (a) THE PROBLEM, AND ONLY THAT       the trained network as a circuit - inputs, a recurrent pool
                                       filled to its measured live fraction, outputs - beside eight
                                       units drawn at RANDOM out of that same network on one trial,
                                       on a shared rate scale. Real simulated traces, not cartoons.
                                       The arrows inside the pool are a SAMPLE of the connectivity:
                                       only pairs with an empty corridor between them are joined, so
                                       no arrow crosses a unit. Red arrowheads are excitatory and
                                       blue bars inhibitory connections, in the proportion MEASURED
                                       in this network's recurrent weights - these networks are not
                                       sign-constrained, so that is a property of connections.
  (b) WHY "SILENT" IS NOT A JUDGEMENT  the participation distribution of that same network on a log
                                       axis: bimodal with orders of magnitude of empty valley between
                                       the modes, so the threshold is read off the data.
  (c) SIZE MAKES IT WORSE              active units vs N on the three tasks, each run read where its
                                       own loss first reaches 1.05x its own fitted floor. The count
                                       grows far slower than N, so the active FRACTION falls.
  (d) THE SILENT UNITS DO NO WORK      held-out r2 as units are removed, quietest first, busiest
                                       first, and in random order (ablate_silent_units.py). Removing
                                       every silent unit leaves the score where it was; removing the
                                       same number at random, or the working few, destroys it.
  (e) IT DOES NOT STOP                 the loss and the silent count over training, every seed, one
                                       sub-panel per task, with each run's own 1.05x-floor crossing.

CRITERION. Everything here is the scale-free participation criterion (a unit is silent below 5% of
its own network's 95th-percentile participation), the only criterion that travels across tasks and
activation functions. Panel (b) shows why it lands in the valley rather than on a mode.

READ-OUT. 1.05x each run's OWN fitted floor of its NOISY training loss - the loss it was trained on,
under its training noise - smoothed by a running median over 2,001 iterations (common.noisy_floor;
the rule since 2026-10-12). The noise-free loss is not used: on the standard flip-flop it bottoms
out near 10k and climbs back 2-4x while the noisy loss keeps falling. A run with no usable crossing
(fit on its bound, loss never reaching the bar, loss climbing back > 1.5x) is dropped and counted.
The learning rate falls with size (lr = 1e-3 (100/N)^(1/3)); reading each network at its own floor
is how panel (c) handles that confound. The flip-flop's noisy loss is still falling at 75k in
every control, so its series is read on the controls continued to 200k. A task with a listed size
that has no floor read-out - including a size with no runs on disk yet, as the flip-flop's N = 4000
was until its 200k continuation landed - is drawn at the end of its budget, with open markers, and
the legend says so.

Usage:  python fig_paper_F1.py --collect [--refresh]   on Della, from the checkout root; writes
                                                       COLLECT (--refresh re-simulates the example
                                                       network)
        python fig_paper_F1.py --collect-example data/fig_paper_F1_collect.pkl
                                                       on Della: panels a and b only, the other
                                                       panels kept from that record; writes COLLECT
        python fig_paper_F1.py                         on the Mac, with that pickle in data/
Output: img/internal_figures/fig_paper_F1.pdf (+ .svg; vector only - see paperstyle.save)
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys

import numpy as np
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Circle, Patch, Rectangle
from matplotlib.path import Path
from matplotlib.transforms import offset_copy

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
import heldout_r2 as heldout
from pr_matrix import fit_floor_at, excess_time_at
from common import (DATA_DIR, SILENT_REL, participation, logbin, noisy_loss_series,
                    FLOOR_DELTA, warm_start_chain, active_mask, runs_with_loss, floor_crossing,
                    at_own_floor, read_run, maze_load_runs, maze_learned_index,
                    MAZE_LEARNED_SUCCESS, MAZE_LEARNED_HOLD)
from flipflop_diversity import rates_and_targets
from flipflop_fixedpoints import load_net
import hydra
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import filter_kwargs
from gradient_participation import initial_params
from fig_ed_population import load as load_population_stats

# v3: the batch is drawn from a seeded task (#165); v2 (the untrained twin's participation added)
# was drawn from an unseeded one
CACHE = "data/fig_paper_F1_cache_v3.npz"
# The example network is a STANDARD unpenalised 3-bit flip-flop control at N = 1000, trained for
# 200,000 iterations (40k + a 35k top-up + a 125k top-up, one continuous chain): the flip-flop's
# budget since its noisy loss was found still falling at 75k. Panel (b) draws it twice: as it was
# before its first update - its untrained twin, rebuilt from the chain's first saved seed exactly
# as run_experiment built it (gradient_participation.initial_params) - and after training, both
# simulated on the same trials. Until 2026-10-12 this was a trainable-bias network
# (NBitFlipFlop_std_dropout) and panel (b) showed the trained network only.
EXAMPLE_NET = f"{DATA_DIR}/NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none"
EXAMPLE_TASK = "3-bit flip-flop"
N_TRIALS = 24
N_UNITS = 1000            # every intervention family is measured at this size

# Panel (a). The pool is drawn as N_GLYPH units standing for all N of them, filled to the MEASURED
# live fraction, and N_SHOWN units are drawn at random from the network - at random, so that the
# proportion of them that turns out to be silent is itself the result rather than a choice. Both
# seeds are fixed so the panel is reproducible; neither was searched over.
# The pool is drawn at N_GLYPH units, not 100. The panel is ~36 mm wide, so 100 glyphs inside the
# boundary sit 4.2 pt apart with a 3.7 pt dot in each: the dots touch, and an arrow between two of
# them has no visible length at all. At 48 the gap is 4.3 pt, a third wider than a dot, and an
# arrow keeps 4-7 pt of shaft after clearing both glyphs. The count is arbitrary either way - the
# FRACTION filled is the measurement, and it survives any count.
N_GLYPH, N_SHOWN = 48, 8
N_EDGES = 65              # connections wanted; the layout keeps as many as fit without crossing
ARC_SAMPLES = 26          # points per drawn arc, for the clearance and crossing tests
BEND = -0.20              # arc3 curvature. Negative bows LEFT of travel, so an arrow running left
                          # to right bows up and one running right to left bows down; flip the sign
                          # to swap that.
R_POOL = 0.90             # pool radius inside the unit-radius boundary circle
DOT_S = 10.0              # unit glyph area in pt^2 (3.6 pt across), on a 7.5 pt lattice
GLYPH_SEED, TRACE_SEED = 3, 11
SILENT_GREY = ps.INACTIVE  # one grey for an inactive unit, in the drawing, traces and histograms
ACTIVE_COL = ps.ACTIVE    # near-black: active units, their traces and their histogram
CONN_COL = ps.FAINT       # the sampled recurrent connections of panel (a), one neutral grey

# A series listed here is drawn, with every seed, but NOT fitted (task -> reason). The exponent
# would be quoted, and an exponent one seed can move by 0.3 is not a measurement. None at present.
NO_FIT = {}

# (label, {N: glob}, budget, colour). STANDARD controls only, re-sourced 2026-10-12: the trainable-bias
# k-sweep, big-N and drift sweeps this panel used are on neither cluster any more. Each series is one
# experiment at every size - same task, same model, same trainer, only N (and the N-scaled learning
# rate) differ:
#   3-bit flip-flop  the unpenalised control of the remedy grid, 40k + a 35k top-up + a 125k top-up
#                    = 200k, one warm-start chain (NBitFlipFlop_ff_revive_c200000 arm=none at
#                    N <= 2000; the paper grid's control at N = 4000, the same configuration). At 75k
#                    the noisy loss was still falling in every control.
#   CDDM             the paper grid's control, 100k.
#   DMTS             the paper grid's control at a 5 tau delay, 150k. At 7 tau the unpenalised
#                    network solved 2 of 3 seeds at two sizes; at 5 tau it solves 8 of 9.
# The 6-bit flip-flop series is gone: it was never re-run on the standard network.
FF200 = f"{DATA_DIR}/NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N={{N}}_pen=none_arm=none"
DMTS_LABEL = "DMTS, 5$\\tau$ delay"
SCALING = {
    "3-bit flip-flop": ([500, 1000, 2000, 4000], {
        500:  FF200.format(N=500),
        1000: FF200.format(N=1000),
        2000: FF200.format(N=2000),
        4000: f"{DATA_DIR}/NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control",
    }, 200_000, ps.SLOTS[0]),
    "CDDM": ([500, 1000, 2000, 4000], {
        N: f"{DATA_DIR}/CDDM_paper_grid/EqType=h_N={N}_arm=control" for N in (500, 1000, 2000, 4000)
    }, 100_000, ps.SLOTS[1]),
    DMTS_LABEL: ([500, 1000, 2000], {
        N: f"{DATA_DIR}/DMTS_paper_grid_d5/EqType=h_N={N}_arm=control" for N in (500, 1000, 2000)
    }, 150_000, ps.SLOTS[2]),
}

# ⚠️ A NETWORK THAT NEVER LEARNED THE TASK IS NOT EVIDENCE ABOUT HOW MANY UNITS THE TASK NEEDS.
# Unpenalised DMTS is bimodal per seed: it either solves the task or sits at the constant-output
# solution, which scores r2 ~ 0.42-0.43. At 5 tau one seed of nine does (N = 2000, 0.418) and every
# other lands at 0.994-0.9995, so 0.8 sits in an empty valley. The flip-flop and CDDM have no
# chance-level runs; diverged runs are dropped by name in traces_of.
TASK_MIN_R2 = {DMTS_LABEL: 0.8}

# The maze-running task (PPO; engellab/trainRNNbrain#3; in the main text since dead-relu-paper#182).
# Its runs and ablations live in this repo, not on Della, and are read when the figure is drawn.
MAZE_LABEL = "Maze"
REPO_ROOT = os.path.normpath(os.path.join(DATA_DIR, "..", ".."))
MAZE_ROOT = os.path.join(REPO_ROOT, "data", "trained_RNNs_rl_maze", "Maze_suite")
MAZE_ABLATION_DIR = os.path.join(REPO_ROOT, "data", "rl_maze_ablation")
MAZE_RETRAIN_DIR = os.path.join(REPO_ROOT, "data", "rl_maze_retrain")          # rl_maze_retrain.py
MAZE_POP_STATS = os.path.join(REPO_ROOT, "data", "population_stats_maze.json")  # rl_maze_population_stats.py


def load_population_stats_all():
    """The per-network results of population_stats.py plus the maze's, when its file exists.

    The maze results (rl_maze_population_stats.py, task "Maze") live in their own file because
    fig_ed_population.load sorts data/population_stats.json by the three supervised tasks.

    Returns:
        list of result dicts, the supervised ones first; the maze's are left out when
        MAZE_POP_STATS does not exist, and panel (e) then reads "no data" for the maze.
    """
    res = load_population_stats()
    if os.path.exists(MAZE_POP_STATS):
        with open(MAZE_POP_STATS) as fh:
            res = res + json.load(fh)["results"]
    return res


def maze_retrain(folder=MAZE_RETRAIN_DIR, N=N_UNITS):
    """Panel (d)'s maze retrain strip: the JSONs of rl_maze_retrain.py, one dict per network.

    rl_maze_retrain.py writes one file per (network, arm), <cell>__s<seed>__<arm>.json, each with a
    `curve` of (update, success score) pairs, the score of the ablation curves (success under the
    training noise); the arms are intact, silent_removed and random_removed. They are joined here
    into the shape the supervised retrain dicts have (ablate_silent_units.retrain_test): label, N,
    n_silent and one curve per arm, so panel_d draws them unchanged. A network is listed once its
    silent_removed file exists; a missing arm is left out of its dict.

    Args:
        folder: str; N: int, the network size.
    Returns:
        list of dicts, one per network, in seed order.
    """
    out = []
    for path in sorted(glob.glob(os.path.join(folder, f"N={N}_act=relu__s*__silent_removed.json"))):
        d = {}
        for arm in ("intact", "silent_removed", "random_removed"):
            f = path.replace("__silent_removed.json", f"__{arm}.json")
            if os.path.exists(f):
                with open(f) as fh:
                    a = json.load(fh)
                d[arm] = [tuple(pt) for pt in a["curve"]]
                d.update(label=a["label"], N=a["N"], n_silent=a["n_silent"], n_active=a["n_active"],
                         retrain_updates=a["retrain_updates"], run_dir=a["run_dir"])
        out.append(d)
    return out

# Figure 1 names each task in full once (panel a, and the caption) and then by the abbreviation the
# paper defines; the series labels above stay as they are, because the Figure 2 legends use them.
TASK_SHORT = {EXAMPLE_TASK: "3BFF", "CDDM": "CDDM", DMTS_LABEL: "DMTS", MAZE_LABEL: "Maze"}
# The three tasks of the main text, in the order panels (c), (d) and (e) draw them (#182: the maze
# in place of DMTS, which stays in PANEL_TASKS and ABLATION for collect() and fig_cosyne2027).
MAIN_TASKS = [EXAMPLE_TASK, "CDDM", MAZE_LABEL]
# How each task is scored: the name in panel (c)'s tick label and the number's format (tick, fmt),
# panel (d)'s y-axis label (axis) and the name in the printed numbers (text). The supervised tasks
# are scored by held-out R^2 at the floor crossing, to three decimals because the seeds agree to
# 0.001. The maze is scored by the clean probe's success rate over 128 episodes, to two decimals
# because one probe's rate has a standard error near 0.02.
SCORE_R2 = dict(tick="$R^2$", fmt=".3f", axis=ps.LABELS["r2"], text="held-out R2")
SCORE = {MAZE_LABEL: dict(tick="success", fmt=".2f", axis="success rate", text="success rate")}

# Panel (c): the N = 1000 cell of each task's size series (the same runs as Figure 2's N = 1000
# point), each run read at 1.05x its own noisy-loss floor (common.read_run) with held-out R^2 at
# that crossing (heldout_r2.floor_r2: measured on a kept snapshot, else estimated from the end).
# (label, cell, colour); the colours are the ones panel (d) gives the same tasks. These are the
# supervised cells, read by collect() on Della; the maze runs are read on the Mac (maze_rows).
PANEL_TASKS = [
    (EXAMPLE_TASK, FF200.format(N=1000), ps.TASK_3BFF),
    ("CDDM", SCALING["CDDM"][1][1000], ps.TASK_CDDM),
    (DMTS_LABEL, SCALING[DMTS_LABEL][1][1000], ps.TASK_DMTS),
]
# Runs left out of panel (c) on purpose, by a substring of the run folder's name -> the reason.
# Empty since decision dead-relu-paper#20 (option 2, 31 Oct): the N = 1000 3BFF network of seed
# 3374117298, once excluded as a straggler and replaced by FFseed4, has a floor crossing under
# read-out 79932ef (t* 58,968, 277 active, R^2 0.949; entry 2026-10-31_1702), so all four networks
# are drawn: 277.5 +- 26.6 (n = 4). A collect record written under the old list is re-flagged by
# apply_decisions, so the change needs no new Della read.
EXCLUDED_RUNS = {}
# everything the drawing stage reads. v2 (#165): panels (a) and (b) on the seeded 24-trial batch,
# the other panels copied from v1 (data/fig_paper_F1_collect.pkl, Della job 16556248)
COLLECT = "data/fig_paper_F1_collect_v2.pkl"

# Every intervention we ran, grouped into families that share a task, an architecture, a read-out
# iteration AND a silence criterion. The panel plots a CHANGE from each family's OWN reference,
# which is what makes it legitimate to show families side by side: no number is ever compared
# across a criterion boundary, only against a reference measured the same way.
#
# Families 3 and 4 come from summary CSVs rather than participation traces. Their raw sweeps were
# deleted (Supplementary S6) and family 4 uses a peak-rate rather than a participation criterion,
# which is exactly why they are their own blocks with their own references.
TRACE_FAMILIES = [
    ("CDDM, 200k", f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=*", None, [
        # Each activation carries its own shape parameter in the label: "a different activation" is
        # not one condition, and a reader cannot tell a leak of 0.01 from one of 0.3, or softplus at
        # beta=25 (a floor of log(2)/25 = 0.028, nearly a ReLU) from beta=1 (a floor of 0.69, nothing
        # like one). The values are read from the trained nets' own saved configs.
        ("leaky ReLU, leak 0.01", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=leakyrelu_iters=*",  "activation"),
        ("softplus, $\\beta$ = 25", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=softplus25_iters=*", "activation"),
        ("sigmoid, 7.5(x$-$0.3)", f"{DATA_DIR}/CDDM_std_g0_activations/EqType=h_N=1000_act=sigmoid_iters=*",    "activation"),
        ("W.D. 0",            f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=0_iters=*",           "weight decay"),
        # 1e-6 is the default in configs/trainer/trainer.yaml, so every run in this family's
        # reference carries WD=1e-06 - the reference IS this rung, and saying so turns three
        # scattered points into a monotone dose-response
        ("W.D. 10⁻⁶*", None,                                                             "reference"),
        ("W.D. 10⁻⁵",         f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-5_iters=*",        "weight decay"),
        ("W.D. 10⁻⁴",         f"{DATA_DIR}/CDDM_std_g0_weightdecay/EqType=h_N=1000_wd=1e-4_iters=*",        "weight decay"),
    ]),
    ("3-bit flip-flop, 150k", f"{DATA_DIR}/NBitFlipFlop_std_ksweep/EqType=h_k=3_N=1000_iters=*", 150_000, [
        ("leaky ReLU, leak 0.01", f"{DATA_DIR}/NBitFlipFlop_std_activations/EqType=h_k=3_N=1000_act=leakyrelu_iters=150000",  "activation"),
        ("softplus, $\\beta$ = 25", f"{DATA_DIR}/NBitFlipFlop_std_activations/EqType=h_k=3_N=1000_act=softplus25_iters=150000", "activation"),
        ("sigmoid, 7.5(x$-$0.3)", f"{DATA_DIR}/NBitFlipFlop_std_sigmoid/EqType=h_k=3_N=1000_iters=*",   "activation"),
        # ⚠️ THESE WERE LABELLED "input w. ×0.5 … ×20" until 2026-10-01, which read as multiples of
        # the default. They are not. `model.input_row_norm=s` sets every W_inp row to the ABSOLUTE
        # norm s at init, and the default draw's rows sit at √(n_inputs/N) = 0.050 at N = 1000, so
        # the four rungs are 10×, 40×, 100× and 400× the default and the reference is BELOW all of
        # them, not between 0.5 and 2. The old labels put the reference in the middle of its own
        # ladder and made a single-peaked curve look non-monotone.
        ("row norm 0.5", f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=0.5_iters=*", "input scale"),
        ("row norm 2",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=2_iters=*",   "input scale"),
        ("row norm 5",   f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=5_iters=*",   "input scale"),
        ("row norm 20",  f"{DATA_DIR}/NBitFlipFlop_std_winp/EqType=h_k=3_N=1000_s=20_iters=*",  "input scale"),
    ]),
]

# (label, csv, row filter, group). All at CDDM N=1000, eq=h, 30k.
ARCHIVE_FAMILY = ("CDDM, 30k (archived)",
                  ("silent_stats_all.csv", dict(sweep="std", penalty="none")), [
    ("eq. s instead of h",       ("silent_stats_all.csv", dict(sweep="std", penalty="none", eq="s")), "architecture"),
    # The `dale` sweep (Dale's law with non-negative input and output weights) was a row here until
    # 2026-09-30. Its label was the longest on the panel and cost every other panel width, and
    # Dale-constrained networks have a supplementary section of their own (S4), so it was dropped
    # from the figure rather than shortened into something unreadable.
    #
    # ⚠️ THESE ROWS WERE MISLABELLED until 2026-09-21. The CSV's sweep names were read as
    # "self-connections off" (nodale_bias) and "bias fixed at 0" (nodale). Neither is what they are.
    # Matching each sweep's per-net counts against the participation traces still on disk identifies
    # them exactly: `dale` is CDDM_ptrack_g0 (Dale + non-negative I/O), `nodale` is
    # CDDM_ptrack_g0_nodale (both constraints off, bias fixed at 0) and `nodale_bias` is
    # CDDM_ptrack_g0_nodale_trainablebias (both off, bias TRAINABLE) - the five counts agree
    # element for element in all three cases.
    #   - self-connections were never varied: `self_connections=False` is the model default on
    #     every path, including the reference, so no row can be about them;
    #   - "bias fixed at 0" is the reference's own setting, so it cannot be an intervention. The
    #     intervention is making the bias trainable, which is what `nodale_bias` does;
    #   - `nodale` differs from the `std` reference only in being a different sweep of the same
    #     unconstrained architecture, so it is a sweep-to-sweep replicate, not a knob. Dropped.
    ("trainable bias",           ("silent_stats_all.csv", dict(sweep="nodale_bias", penalty="none")), "architecture"),
    ("metabolic λ = 0.01",       ("silent_stats_v2.csv", dict(sweep="metabolic", met="0.01")), "metabolic"),
    ("metabolic λ = 0.1",        ("silent_stats_v2.csv", dict(sweep="metabolic", met="0.1")), "metabolic"),
    ("metabolic λ = 1",          ("silent_stats_v2.csv", dict(sweep="metabolic", met="1.0")), "metabolic"),
    ("metabolic λ = 10",         ("silent_stats_v2.csv", dict(sweep="metabolic", met="10.0")), "metabolic"),
])

# The noise sweep saved no participation traces, so it was the one family under a peak-rate rather
# than a participation criterion; cddm_noise_participation.py re-scores it from the trained weights
# and it now shares the rule with every other family (see noise_active).
# sigma_rec = 0.05 is the default in every model config, so as with weight decay the reference is a
# rung of this ladder rather than something outside it. Listed in ascending order with the rest.
NOISE_FAMILY = ("CDDM, 30k", "0.05", [
    ("rec. noise σ = 0",    "0.0",  "noise"),
    ("rec. noise σ = 0.01", "0.01", "noise"),
    ("rec. noise σ = 0.05*", None, "reference"),
    ("rec. noise σ = 0.1",  "0.1",  "noise"),
])

# Tried, but with no read-out that can sit on this axis. These sweeps saved no participation traces
# and have no row under the scale-free rule, so there is no honest way to put a number on them - but
# leaving them off entirely would let the panel read as "this is everything", which it is not. Their
# trained weights are still on disk, so each could be re-read with a re-simulation pass.
TRIED_NOT_PLOTTED = [
    ("cubic term $\\gamma$", "CDDM_4a031e (on) vs CDDM_4a031e_g0 (off)"),
    ("weight boundary, sticky vs reflective", "CDDM_2bc3c1_g0_reflective"),
]
# I/O positivity is a special case: it is never varied ALONE in any sweep on disk - every `nodale`
# arm switches `dale` and `io_nonnegativity` off together - so the Dale row below is a joint
# contrast and there is no I/O-positivity-only number to report.

# Knobs we did NOT vary. Panel (d) shows everything we tried; these are the obvious candidates it
# does not cover, so the text says "we did not vary" rather than implying a measured null.
NEVER_SWEPT = ["spectral radius of the initial recurrent weights",
               "connectivity density (every network here is dense)",
               "learning rate (fixed by the rule lr = 1e-3 (100/N)^(1/3))",
               "batch size"]

GROUP_COL = {"activation": ps.SLOTS[3], "weight decay": ps.SLOTS[4],
             "input scale": ps.SLOTS[2], "metabolic": ps.SLOTS[1],
             "architecture": ps.SLOTS[0], "noise": ps.COND_COL["both"]}


# Sizes that exist on disk for a scaling series but are deliberately NOT plotted. Every entry needs
# a reason, because the audit below will otherwise shout about it on every run. An empty reason is
# not allowed: if a size is excluded, that is a decision and it gets written down.
SCALING_EXCLUSIONS = {
    ("CDDM", 100):   "near the ceiling (76% of units active) and only one condition has it; "
                     "excluded from the fit in Methods",
    ("CDDM", 10000): "only reaches 80,000 iterations, so it cannot join a 100,000-iteration "
                     "matched-compute read-out; quoted separately in Methods",
}
COVERAGE_CACHE = "data/fig_paper_F1_coverage_v2.pkl"   # v2: the signature gained T, bias, remedies


def _signature(cfg):
    """The experiment identity of a run, everything except network size.

    Two cells with the same signature are the same experiment at different N and belong on the same
    scaling curve, whichever sweep folder they happen to live in.

    Args:
        cfg: an OmegaConf config loaded from a run's saved *_config.yaml.
    Returns:
        a hashable tuple, or None if the config is missing a field we need.
    """
    # The trial length, the bias range and the remedy switches are part of the identity: without
    # them a trainable-bias 30k sweep, and DMTS at a 7 tau delay, were reported as missing sizes of
    # the standard series (2026-10-12).
    try:
        t, m, tr = cfg.task, cfg.model, cfg.trainer
        return (str(t.taskname), int(t.n_inputs), int(t.n_outputs), int(t.T),
                str(m.equation_type), str(m.activation_args.name), bool(m.dale), float(m.gamma),
                float(m.spectral_rad), tuple(float(b) for b in m.bias_range),
                float(tr.lambda_frm), float(tr.lambda_rws), float(tr.lambda_met),
                bool(tr.dropout), bool(tr.get("prune_reinit", False)),
                bool(tr.get("synaptic_scaling", False)), float(m.get("sigma_w", 0.0) or 0.0))
    except Exception:
        return None


def _cell_index(refresh=False):
    """Index every trained cell on disk by (signature -> {N: cell path}).

    Reads ONE saved config per cell, not per run, and caches the result. ~2 s cold over ~550 cells.

    Args:
        refresh: rebuild the cache even if it exists.
    Returns:
        dict mapping signature tuple -> {int N: cell directory path}.
    """
    if os.path.exists(COVERAGE_CACHE) and not refresh:
        with open(COVERAGE_CACHE, "rb") as fh:
            return pickle.load(fh)
    index = {}
    for cell in sorted(glob.glob(os.path.join(DATA_DIR, "*", "*", ""))):
        cfgs = glob.glob(os.path.join(cell, "*", "*_config.yaml"))
        if not cfgs:
            continue
        try:
            cfg = OmegaConf.load(cfgs[0])
        except Exception:
            continue
        sig = _signature(cfg)
        if sig is None:
            continue
        index.setdefault(sig, {})[int(cfg.model.N)] = cell
    with open(COVERAGE_CACHE, "wb") as fh:
        pickle.dump(index, fh)
    return index


def audit_scaling_coverage(refresh=False):
    """Shout if a scaling series has usable data on disk that SCALING does not plot.

    THE BUG THIS EXISTS TO CATCH. SCALING hard-codes each series' sizes, so a cell living in a
    different sweep folder is silently ignored. The 6-bit flip-flop was fitted on three points for
    weeks while NBitFlipFlop_std_bigN/EqType=h_k=6_N=4000_pen=none sat on disk with three seeds --
    the k-sweep glob could never have matched it, because the folder name is different. Matching on
    the configured path would therefore not have found it either. This audit instead reads every
    cell's OWN saved config and groups by experiment signature, so where a run was filed is
    irrelevant.

    Args:
        refresh: rebuild the on-disk cell index.
    Returns:
        dict series -> sorted list of unplotted sizes (empty when a series is fully covered).
    """
    index = _cell_index(refresh=refresh)
    missed = {}
    print("\n--- scaling coverage audit ---")
    for task, (sizes, cells, _it, _col) in SCALING.items():
        ref = None
        for N in sizes:
            cfgs = glob.glob(os.path.join(cells[N], "*", "*_config.yaml")) or \
                   glob.glob(os.path.join(cells[N], "*", "*", "*_config.yaml"))
            if cfgs:
                ref = _signature(OmegaConf.load(cfgs[0]))
                break
        if ref is None:
            print(f"  {task:18} no data on disk - cannot audit")
            continue
        on_disk = set(index.get(ref, {}))
        extra = sorted(n for n in on_disk - set(sizes)
                       if (task, n) not in SCALING_EXCLUSIONS)
        known = sorted(n for n in on_disk - set(sizes) if (task, n) in SCALING_EXCLUSIONS)
        note = f"  (excluded on purpose: {known})" if known else ""
        if extra:
            missed[task] = extra
            print(f"  {task:18} !! UNPLOTTED DATA AT N = {extra} -- {index[ref][extra[0]]}{note}")
        else:
            print(f"  {task:18} {len(sizes)} sizes plotted, none missed{note}")
    if missed:
        print("  ^^ add these to SCALING, or give each a reason in SCALING_EXCLUSIONS.")
    return missed


def traces_of(pattern, min_r2=None):
    """Every (participation matrix, iteration vector) pair under a run-folder glob.

    DIVERGED RUNS ARE DROPPED. A run folder is named `<score>_<task>;...`, and a run whose loss went
    to NaN is saved as `nan_...`. Its participation trace is all zeros from the divergence onward, so
    it reports ZERO active units and drags the cell mean down as if the network had silenced
    completely - which is the opposite of what happened. There are 59 such folders on disk; one of
    them sits in the 8-bit flip-flop cell at N = 2000 and was pulling its mean down by ~160 units.

    RUNS THAT NEVER LEARNED THE TASK ARE DROPPED TOO, when `min_r2` is given. A network sitting at
    the constant-output solution has whatever activity its initialisation left it, which says
    nothing about how many units the task needs -- see TASK_MIN_R2 for the one series that needs
    this and why its threshold lands where it does.

    Args:
        pattern: glob matching run folders (not the pickles themselves).
        min_r2: float, drop any run whose score prefix is below this, or None to keep every run
            that did not diverge.
    Returns:
        list of (P, iters): P is (n_probes, N) participation, iters is (n_probes,).
    """
    out = []
    for f in sorted(glob.glob(os.path.join(pattern, "*", "*ParticipationTrace.pkl"))):
        head = os.path.basename(os.path.dirname(f)).split("_")[0]
        if head == "nan":
            continue
        if min_r2 is not None:
            try:
                if float(head) < min_r2:
                    continue
            except ValueError:
                continue
        try:
            d = pickle.load(open(f, "rb"))
        except Exception:
            continue
        P, it = np.asarray(d.get("participation", [])), np.asarray(d.get("participation_iters", []))
        if len(it) and P.ndim == 2:
            out.append((P, it))
    return out


def live_matched(pattern, cap=None, min_r2=None):
    """Active units per seed at the largest iteration every seed in the cell reaches.

    Matched compute, not each run's own endpoint: a big network read at its end and a small one read
    at its end differ in convergence depth as well as size, which confounds the very comparison the
    scaling panel makes.

    Args:
        pattern: glob matching run folders; cap: read no later than this iteration, or None for the
            deepest shared probe; min_r2: drop runs scoring below this, passed to traces_of.
    Returns:
        (counts, iteration) with counts an (n_seeds,) int array, or None if the cell is empty.
    """
    tr = traces_of(pattern, min_r2=min_r2)
    if not tr:
        return None
    shared = min(int(it[-1]) for _, it in tr)
    it_read = shared if cap is None else min(shared, cap)
    counts = []
    for P, it in tr:
        p = P[int(np.argmin(np.abs(it - it_read)))]
        counts.append(int(active_mask(p).sum()))
    return np.array(counts), it_read


PENALTY_KEYS = ("lambda_frm", "lambda_rws", "lambda_met", "lambda_orth")
CONV_MARGIN = 1.0 + FLOOR_DELTA   # 1.05: read where the noisy loss first reaches this x its floor


def clean_loss_series(run_dir, trace):
    """The dropout-off training loss of one run, and the iterations it was sampled at.

    Prefers metrics["loss_clean_train"] inside the participation trace, which is recorded beside
    the participation probes and is the clean loss by construction. Falls back to TrainLosses.json
    ONLY where the run's own config shows dropout off and every penalty at zero, since the
    training-pass loss is then the same quantity; otherwise returns nothing rather than mixing two
    definitions. CDDM's runs predate the metric and come through the fallback.

    Args:
        run_dir: path to one trained-network folder; trace: its loaded ParticipationTrace dict.
    Returns:
        (iterations, loss) float arrays, or (None, None).
    """
    m = trace.get("metrics", {})
    if "loss_clean_train" in m:
        return np.asarray(trace["iters"], float), np.asarray(m["loss_clean_train"], float)
    cfg = glob.glob(os.path.join(run_dir, "*config.yaml"))
    jf = glob.glob(os.path.join(run_dir, "*TrainLosses.json"))
    if not (cfg and jf):
        return None, None
    t = OmegaConf.load(cfg[0]).trainer
    if bool(t.get("dropout")) or any(float(t.get(k, 0) or 0) for k in PENALTY_KEYS):
        return None, None
    L = np.asarray(json.load(open(jf[0])).get("train_losses", []), float)
    return (np.arange(len(L), dtype=float), L) if len(L) else (None, None)


def at_end_of_budget(cells):
    """Active units per size at each run's last stored probe - the secondary, end-of-training read-out.

    Every size of a series shares one budget, so this is matched compute; it is used only for a task
    that has no usable floor read-out, and is labelled as such wherever it is drawn.

    Args:
        cells: dict N -> list of run dicts from runs_with_loss.
    Returns:
        dict N -> (counts array, last iteration).
    """
    return {N: (np.array([r["active"][-1] for r in v], float),
                float(np.mean([r["it_act"][-1] for r in v]))) for N, v in cells.items() if v}


def matched_after_convergence(cells):
    """Active units per size, read the same number of iterations after each network CONVERGED.

    ⚠️ WHY NOT A FIXED ITERATION CAP. Reading every size at the same iteration assumes every size
    reaches its final performance at the same point, and on DMTS that is false: the mean crossing
    runs 650, 31,620 and 83,040 iterations at N = 500, 1000 and 2000, so at a 150,000-iteration
    read-out the largest networks have spent a fifth as long in the phase where units go silent.
    That alone put the DMTS exponent at 0.87 against 0.31-0.47 for every other task. On the
    flip-flops and CDDM the crossing does NOT move with size (24,307/26,340/25,777 and
    705/946/1,007), which is why they were unaffected and why the artefact went unnoticed.

    THE BAR IS ABSOLUTE PER TASK, not per run: CONV_MARGIN times the worst final clean loss among
    that task's runs, so every run reaches it and none sets its own. A per-run floor is
    backward-looking -- a network ending at a higher loss gets a looser bar and crosses earlier,
    and training the same network longer moves its bar and so its crossing.

    K IS BOUNDED BY THE WORST INDIVIDUAL RUN, not by a cell mean. Bounding it by the mean lets a
    late-converging seed's target fall past its last probe, where a nearest-probe lookup silently
    clamps it to the endpoint -- reading it exactly the way this function exists to avoid. Measured
    on DMTS, that mistake reported an exponent of 0.56 where the correct value is 0.33.

    Args:
        cells: dict N -> list of run dicts from runs_with_loss.
    Returns:
        (dict N -> (counts array, mean crossing), bar, K).
    """
    runs = [r for v in cells.values() for r in v]
    bar = max(r["loss"][-1] for r in runs) * CONV_MARGIN
    for r in runs:
        r["cross"] = float(r["it_loss"][int(np.argmax(r["loss"] <= bar))])
    K = min(r["it_act"][-1] - r["cross"] for r in runs)
    out = {}
    for N, v in cells.items():
        counts = []
        for r in v:
            want = r["cross"] + K
            if want > r["it_act"][-1] + 50:            # never read past where it was measured
                raise AssertionError(f"N={N} {r['name'][:14]}: target {want:,.0f} past last probe")
            counts.append(r["active"][int(np.argmin(np.abs(r["it_act"] - want)))])
        out[N] = (np.array(counts, float), float(np.mean([r["cross"] for r in v])))
    return out, bar, K


def example_network_pair(refresh=False):
    """Rates, targets and participation of one trained unpenalised flip-flop network, and its twin.

    Both the trained network and its untrained twin are simulated noise-free on the SAME batch of
    trials, so the two participation distributions of panel (b) differ only by training. Cached,
    because panels (a) and (b) must show a real network rather than an illustration.

    Args:
        refresh: re-simulate even if the cache exists.
    Returns:
        (rates, targets, p, p_init, iterations): trained (N, T, B) rates, (k, T, B) target bits,
        (N,) participation trained and untrained, and the iterations the network was trained for.
    """
    if os.path.exists(CACHE) and not refresh:
        z = np.load(CACHE)
        return z["rates"], z["targets"], z["p"], z["p_init"], int(z["iterations"])
    folder = sorted(glob.glob(os.path.join(EXAMPLE_NET, "*/")))[0]
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    cfg.task.batch_size = N_TRIALS
    # The task draws its trials from its own generator (self.rng, seeded from cfg.task.seed, which
    # the training configs leave null), so np.random.seed alone does not fix the batch (#165). Seed
    # the task itself, as active_before_after.py does.
    cfg.task.seed = TRACE_SEED
    np.random.seed(TRACE_SEED)
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, targets, _ = task.get_batch()

    def run(net):
        """Simulate `net` noise-free on the batch `inputs`; return its (N, T, B) ReLU rates."""
        net.clear_history()
        net.y = net.y_init
        net.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
        return np.maximum(np.array(net.get_history()), 0.0)

    rates = run(load_net(folder)[0])
    # the untrained twin is the chain's FIRST run as it stood before its first update; the top-ups
    # each drew a fresh seed, so only the first config carries the seed the weights were drawn from
    chain = warm_start_chain(folder)
    cfgs = [OmegaConf.load(glob.glob(os.path.join(d, "*_config.yaml"))[0]) for d in chain]
    p0 = dict(initial_params(cfgs[0]))
    p0["activation_name"] = "relu"
    p0.pop("activation_args", None)
    p0.pop("equation_type", None)            # given explicitly below, from the config
    rates0 = run(RNN_numpy(**filter_kwargs(RNN_numpy, p0),
                           equation_type=str(cfgs[0].model.equation_type), seed=0))
    iterations = sum(int(c.trainer.max_iter) for c in cfgs)
    p, p_init = participation(rates), participation(rates0)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, rates=rates.astype(np.float32),
                        targets=np.asarray(targets, np.float32), p=p, p_init=p_init,
                        iterations=iterations)
    return rates.astype(np.float32), np.asarray(targets, np.float32), p, p_init, iterations


def example_network(refresh=False):
    """The trained example network alone - (rates, targets, p) - for the deck's single panels."""
    return example_network_pair(refresh)[:3]


def excitatory_fraction():
    """Fraction of the example network's off-diagonal recurrent weights that are positive.

    The drawn connections are a schematic, but the MIX of excitatory and inhibitory ones does not
    have to be invented: it is read from the trained weight matrix, so a reader counting arrowheads
    on the panel is counting the right proportion. These networks are not sign-constrained, so this
    is a property of connections, not of units - a unit both excites and inhibits.

    Returns:
        float in [0, 1].
    """
    folder = sorted(glob.glob(os.path.join(EXAMPLE_NET, "*/")))[0]
    net, _ = load_net(folder)
    W = np.asarray(net.W_rec)
    return float((W[~np.eye(W.shape[0], dtype=bool)] > 0).mean())


def arc_points(p0, p1, rad, n=ARC_SAMPLES):
    """Sample matplotlib's `arc3` connector as a polyline.

    The collision tests below have to run on the curve that is actually DRAWN, not on the chord, so
    this reproduces `matplotlib.patches.ConnectionStyle.Arc3` exactly: a quadratic Bezier whose
    control point is the midpoint displaced by `rad` times the chord rotated -90 degrees. That
    rotation is why a positive `rad` bows to the RIGHT of travel.

    Args:
        p0, p1: (x, y) endpoints; rad: arc3 curvature; n: samples along the curve.
    Returns:
        (n, 2) array of points from p0 to p1.
    """
    p0, p1 = np.asarray(p0, float), np.asarray(p1, float)
    d = p1 - p0
    c = 0.5 * (p0 + p1) + rad * np.array([d[1], -d[0]])
    t = np.linspace(0.0, 1.0, n)[:, None]
    return (1 - t) ** 2 * p0 + 2 * (1 - t) * t * c + t ** 2 * p1


def inked_span(pts, trim):
    """The part of an arc that is actually drawn: the samples clear of BOTH of its own glyphs.

    Two arrows leaving the same unit share that endpoint exactly, so a crossing test run on the
    full curves calls every such pair a crossing, and the panel ends up with at most one connection
    per unit - which is what capped the first version at twenty arrows for forty-eight units. The
    drawn arrow is shrunk clear of its glyphs anyway, so the test belongs on the shrunk curve.

    Args:
        pts: (n, 2) sampled arc; trim: distance from each endpoint to drop.
    Returns:
        (m, 2) array, or None if nothing survives.
    """
    keep = ((np.linalg.norm(pts - pts[0], axis=1) > trim)
            & (np.linalg.norm(pts - pts[-1], axis=1) > trim))
    return pts[keep] if keep.sum() >= 2 else None


def plan_connections(gx, gy, rng, n_want, lo, hi, clear, bend, trim, r_max=0.99):
    """Choose as many drawable connections as fit, greedily, and return them with their geometry.

    Three things have to hold at once, and they interact, which is why this is a search rather than
    a formula: an arrow may not pass within `clear` of a unit that is not one of its endpoints, no
    two arrows may cross, and every arrow must stay inside the boundary circle. All three are tested
    on the sampled CURVE, so bending is not cosmetic - a bowed arrow can go around a unit that
    blocks the straight chord, which is what lets the panel carry several times more connections
    than the straight-line version could.

    The bend follows the direction of travel: every arrow bows to the same side of its own
    direction, so left-to-right arrows bow one way and right-to-left arrows the other. `bend` sets
    the magnitude and which way.

    Args:
        gx, gy: (n,) unit positions; rng: seeded Generator, for which candidates are tried first;
        n_want: stop once this many are accepted; lo, hi: allowed endpoint separation; clear: the
            corridor half-width that must be free of other units; bend: signed arc3 curvature;
        trim: the un-inked length at each end, used for the crossing test; r_max: arrows must stay
            within this radius of the origin.
    Returns:
        list of (src, dst, rad, points) - points being the (n, 2) sampled curve.
    """
    P = np.column_stack([gx, gy])
    cand = [(a, b) for a in range(len(P)) for b in range(len(P)) if a != b
            and lo <= np.hypot(*(P[b] - P[a])) <= hi]
    rng.shuffle(cand)

    out, taken, used = [], [], set()

    def try_add(a, b):
        """Accept the arrow a -> b if it passes every test; returns True when it was added."""
        if (a, b) in used or (b, a) in used:
            return False                               # one arrow per pair of units
        pts = arc_points(P[a], P[b], bend)
        if np.hypot(pts[:, 0], pts[:, 1]).max() > r_max:
            return False                               # would leave the boundary
        d = np.linalg.norm(pts[:, None, :] - P[None, :, :], axis=2)   # (samples, units)
        d[:, [a, b]] = np.inf
        if d.min() < clear:
            return False                               # passes over a unit
        seg = inked_span(pts, trim)
        if seg is None:
            return False
        path = Path(seg)
        if any(path.intersects_path(q, filled=False) for q in taken):
            return False                               # crosses an arrow already placed
        out.append((a, b, bend, pts))
        taken.append(path)
        used.add((a, b))
        return True

    # Every unit takes part in the circuit (Pavel, 4 Nov 2026: an unconnected unit in the drawing
    # reads as a unit outside the network). The cover pass runs FIRST, on the empty canvas, so no
    # unit is boxed in by arrows placed for others: each unit tries its partners nearest first,
    # within the normal separation range and then within twice it.
    order = list(range(len(P)))
    rng.shuffle(order)
    for u in order:
        if any(u in pair for pair in used):
            continue
        for reach in (hi, 2.0 * hi):
            near = sorted(((a, b) for a in range(len(P)) for b in range(len(P)) if u in (a, b)
                           and a != b and lo <= np.hypot(*(P[b] - P[a])) <= reach),
                          key=lambda pair: np.hypot(*(P[pair[1]] - P[pair[0]])))
            if any(try_add(a, b) for a, b in near):
                break
    for a, b in cand:
        if len(out) >= n_want:
            break
        try_add(a, b)
    return out


def check_connections(conns, gx, gy, clear, trim):
    """Assert the invariants `plan_connections` promises, including that every unit has at least
    one connection. Raises AssertionError on failure.

    A drawing whose whole point is "no arrow crosses anything" should fail loudly rather than ship
    a scribble, so this runs on every build.

    Args:
        conns: the output of `plan_connections`; gx, gy: unit positions; clear: corridor
            half-width; trim: the un-inked length at each end, as passed to the planner.
    Returns:
        None.
    """
    touched = {i for src, dst, _, _ in conns for i in (src, dst)}
    assert touched == set(range(len(gx))), f"units without a connection: {sorted(set(range(len(gx))) - touched)}"
    P = np.column_stack([gx, gy])
    paths = [Path(inked_span(pts, trim)) for _, _, _, pts in conns]
    for k, (a, b, _, pts) in enumerate(conns):
        d = np.linalg.norm(pts[:, None, :] - P[None, :, :], axis=2)
        d[:, [a, b]] = np.inf
        assert d.min() >= clear, f"connection {a}->{b} passes within {d.min():.4f} of a unit"
        for q in paths[k + 1:]:
            assert not paths[k].intersects_path(q, filled=False), f"connection {a}->{b} crosses"


def panel_a(ax_net, ax_tr, rates, p):
    """Panel (a): the problem, and nothing else. The recurrent pool with most of it inactive, and
    eight units drawn at random out of that same network.

    THE TASK IS DELIBERATELY ABSENT. The previous version of this panel drew the input and output
    ports and laid the target bits over the traces, so it was half a task diagram and half a
    problem statement and did neither: a reader cannot learn the flip-flop from three grey steps,
    and the silence is not about the flip-flop anyway - it happens on all three tasks. The trial
    structure of all three now has its own supplementary figure (`fig_supp_tasks.py`), which leaves
    this panel one job: most units of a trained ReLU RNN never leave zero.

    TWO COLOURS, AND THEY ARE THE SAME TWO ON BOTH SIDES. Filled black = active, hollow grey =
    inactive in the drawing; the traces repeat exactly that, so a grey trace and a grey dot are the same
    statement. Giving each trace its own hue, as this panel used to, encodes identity - which is
    not a variable the reader needs.

    RATES SHARE ONE SCALE. Every trace is drawn against the same rate axis and the same scale bar.
    The earlier version normalised each unit by its own maximum, which blew a silent unit's
    numerical dust up to the height of a driven unit's response and made the panel argue against
    itself.

    Args:
        ax_net: blank axes for the network drawing; ax_tr: blank axes for the traces;
        rates: (N, T, B) firing rates; p: (N,) participation.
    Returns:
        (n_live, n_shown_live, n_conn): active units in the network, how many of the drawn units
        were active, and how many connections the layout managed to place.
    """
    N = len(p)
    thr = SILENT_REL * np.quantile(p, 0.95)
    live = p >= thr
    n_live = int(live.sum())

    # --- left: the recurrent pool as a circuit -------------------------------------------------
    ps.blank(ax_net)
    ax_net.set(xlim=(-1.78, 1.78), ylim=(-1.50, 1.16))
    ax_net.set_aspect("equal", adjustable="box")

    rng = np.random.default_rng(GLYPH_SEED)
    n_on = int(round(N_GLYPH * n_live / N))
    # Vogel's sunflower inside the boundary: an equal-area packing of the disc, so a quarter of the
    # dots covers a quarter of it and reads as a quarter of the POOL. A square lattice reads as a
    # layer, which a recurrent pool is not.
    i = np.arange(N_GLYPH)
    rad = R_POOL * np.sqrt((i + 0.5) / N_GLYPH)
    ang = i * np.pi * (3.0 - np.sqrt(5.0))
    gx, gy = rad * np.cos(ang), rad * np.sin(ang)
    on = np.zeros(N_GLYPH, bool)
    on[rng.choice(N_GLYPH, n_on, replace=False)] = True   # silence is not spatially organised

    ax_net.add_patch(Circle((0, 0), 1.0, facecolor="none", edgecolor=ps.MUTED, lw=0.8, zorder=1))

    # A sample of the recurrent connectivity - the trained networks are dense, so any subset is a
    # sample. The pairs are chosen to be near each other AND to have an empty corridor between
    # them, so no arrow crosses a unit; the excitatory fraction is read off the trained weights.
    nn = 2.0 * R_POOL / np.sqrt(N_GLYPH)                  # typical nearest-neighbour separation
    clear, trim = 0.40 * nn, 0.30 * nn        # clearance 3.0 pt against a 1.8 pt glyph radius
    conns = plan_connections(gx, gy, rng, N_EDGES, 1.02 * nn, 2.4 * nn, clear, BEND, trim)
    check_connections(conns, gx, gy, clear, trim)
    # one neutral grey for every connection: the sign of a weight is not a variable of this figure,
    # and a red/blue sign code reused the task and unit-class hues (dead-relu-paper#31 item 2)
    for src, dst, rad, _ in conns:
        ps.arrow(ax_net, (gx[src], gy[src]), (gx[dst], gy[dst]), rad=rad, col=CONN_COL, lw=0.5,
                 zorder=2, mutation_scale=4.0, shrink=1.9, style="-|>")

    # both classes are filled and told apart by colour: fill marks the floor crossing in every
    # figure and means nothing else (dead-relu-paper#83)
    ax_net.scatter(gx[~on], gy[~on], s=DOT_S, color=SILENT_GREY, edgecolors=ps.INK,
                   linewidths=ps.MARKER_EW, zorder=3)
    ax_net.scatter(gx[on], gy[on], s=DOT_S, color=ACTIVE_COL, edgecolors=ps.INK,
                   linewidths=ps.MARKER_EW, zorder=4)

    # inputs and outputs: what makes it a circuit rather than a bag of units. The task is named in
    # the panel title, not explained - its trial structure is Supplementary Fig. 1.
    for y in (0.30, 0.0, -0.30):
        xc = np.sqrt(max(1.0 - y * y, 0.0))
        ps.arrow(ax_net, (-1.60, y), (-xc - 0.03, y), col=ps.INK, lw=0.7, mutation_scale=5)
        ps.arrow(ax_net, (xc + 0.03, y), (1.60, y), col=ps.INK, lw=0.7, mutation_scale=5)
    ax_net.text(-1.30, 0.44, "inputs", ha="center", va="bottom", fontsize=ps.TICK_PT, color=ps.INK)
    ax_net.text(1.30, 0.44, "outputs", ha="center", va="bottom", fontsize=ps.TICK_PT, color=ps.INK)

    # the key: a near-black dot is an active unit, a light grey one an inactive unit, with the counts
    ax_net.scatter([-1.62], [-1.32], s=DOT_S, color=ACTIVE_COL, edgecolors=ps.INK,
                   linewidths=ps.MARKER_EW, zorder=4,
                   clip_on=False)
    ax_net.text(-1.50, -1.32, f"{ps.fmt_count(n_live)} active", ha="left", va="center",
                fontsize=ps.TICK_PT, color=ps.INK)
    ax_net.scatter([0.16], [-1.32], s=DOT_S, color=SILENT_GREY, edgecolors=ps.INK,
                   linewidths=ps.MARKER_EW, zorder=4,
                   clip_on=False)
    ax_net.text(0.28, -1.32, f"{ps.fmt_count(N - n_live)} inactive", ha="left", va="center",
                fontsize=ps.TICK_PT, color=ps.INK)

    # --- right: units drawn at random out of that same network ---------------------------------
    ps.blank(ax_tr)
    pick = np.random.default_rng(TRACE_SEED).choice(N, N_SHOWN, replace=False)
    pick = pick[np.argsort(-p[pick])]          # active on top, so the block itself shows a ratio
    trial = int(np.argmax(rates[int(np.argmax(p))].max(axis=0)))
    R = np.asarray(rates[pick][:, :, trial], float)
    T = R.shape[1]
    tt = np.arange(T)
    scale = float(R.max())
    amp = 0.80 / max(scale, 1e-9)              # one common rate scale for every trace

    for j, (u, r) in enumerate(zip(pick, R)):
        base = float(N_SHOWN - 1 - j)
        ax_tr.plot([0, T - 1], [base, base], lw=0.4, color=ps.GRID, zorder=1)
        ax_tr.plot(tt, base + amp * r, lw=0.75, zorder=3,
                   color=ACTIVE_COL if p[u] >= thr else SILENT_GREY)

    n_shown_live = int(live[pick].sum())
    for lo, hi, lab, col in [(N_SHOWN - n_shown_live, N_SHOWN - 1, "active", ACTIVE_COL),
                             (0, N_SHOWN - n_shown_live - 1, "inactive", SILENT_GREY)]:
        if hi < lo:
            continue
        ax_tr.plot([T + 16] * 2, [lo - 0.18, hi + 0.86], lw=0.8, color=col, zorder=4,
                   solid_capstyle="round")
        ax_tr.text(T + 26, (lo + hi + 0.68) / 2, lab, ha="left", va="center", fontsize=ps.TICK_PT,
                   color=ps.INK)

    # scale bars instead of axes: a schematic panel should not spend two spines on a quantity
    # whose absolute value carries no meaning (ReLU rates are in arbitrary units)
    v = float(f"{scale / 2:.0g}")
    y0 = float(N_SHOWN - 1)                    # beside the tallest trace, not in a corner
    ax_tr.plot([-20, -20], [y0, y0 + amp * v], lw=1.0, color=ps.INK, zorder=4,
               solid_capstyle="butt")
    ax_tr.text(-27, y0 + amp * v / 2, f"{v:g} a.u.\nrate", ha="right", va="center",
               fontsize=ps.TICK_PT, color=ps.INK, linespacing=1.2)
    ax_tr.plot([0, 100], [-0.72] * 2, lw=1.0, color=ps.INK, zorder=4, solid_capstyle="butt")
    ax_tr.text(50, -0.95, r"10 $\tau$", ha="center", va="top", fontsize=ps.TICK_PT, color=ps.INK)

    ax_tr.text(T / 2, N_SHOWN + 0.02, f"{N_SHOWN} units drawn at random, one trial",
               ha="center", va="bottom", fontsize=ps.LABEL_PT, color=ps.INK)
    ax_tr.set(xlim=(-130, T + 95), ylim=(-1.55, N_SHOWN + 0.55))
    return n_live, n_shown_live, len(conns)


def panel_b_pair(axes, p_init, p, iterations):
    """Panel (b): participation of the example network before and after training, on a log axis.

    Two sub-panels on one x axis: the untrained twin (top) and the same network after training
    (bottom), the dashed line at the criterion 0.05 q95(p) of each. Active units are drawn in ACTIVE,
    inactive ones in INACTIVE, and each class is labelled directly with its count, above its own
    bars, so no legend sits on the data. The figure makes no claim that the criterion falls in an
    empty valley (dead-relu-paper#13: halving or doubling the factor moves the count by 3-28%).

    Args:
        axes: (top, bottom) axes; p_init, p: (N,) participation untrained and trained;
        iterations: training iterations, for the bottom title.
    Returns:
        (active untrained, active trained).
    """
    allp = np.maximum(np.r_[p_init, p], 1e-6)
    bins = np.logspace(np.log10(allp.min() * 0.7), np.log10(allp.max() * 1.4), 46)
    counts = []
    for ax, q, title in ((axes[0], p_init, "untrained"),
                         (axes[1], p, f"after {ps.fmt_count(iterations)} training iterations")):
        thr = SILENT_REL * np.quantile(q, 0.95)
        qq = np.maximum(q, 1e-6)
        live = qq >= thr
        h_in, _, _ = ax.hist(qq[~live], bins=bins, color=SILENT_GREY, edgecolor="none")
        h_on, _, _ = ax.hist(qq[live], bins=bins, color=ACTIVE_COL, edgecolor="none")
        ax.axvline(thr, color=ps.INK, lw=0.8, ls=(0, (3, 2)), zorder=5)
        top = max(h_in.max(), h_on.max())
        ax.set_ylim(0, top * 1.32)
        ax.set_xscale("log")
        ax.set_xlim(bins[0], bins[-1])
        ax.set_title(title, fontsize=ps.LABEL_PT, color=ps.INK, pad=2, loc="left")
        # direct labels: each class's count just above its own tallest bar, centred on its bars
        # (the inactive label of the untrained network sits left of the criterion, where its few
        # bars are)
        for mask, h, word, ha in ((~live, h_in, "inactive", "right"), (live, h_on, "active", "center")):
            if not mask.any():
                continue
            k = int(np.argmax(h))
            xc = np.sqrt(bins[k] * bins[k + 1])
            if ha == "right":
                xc = min(xc * 3.0, thr / 1.25) if q is p_init else xc * 1.6
            ax.text(xc, h[k] + top * 0.06, f"{ps.fmt_count(int(mask.sum()))} {word}",
                    ha=ha if q is p_init else ("left" if word == "inactive" else "center"),
                    va="bottom", fontsize=ps.TICK_PT,
                    color=ps.INK)
        ax.set_ylabel("units")
        ax.set_yticks([0, 100, 200])
        counts.append(int(live.sum()))
    axes[0].text(SILENT_REL * np.quantile(p_init, 0.95) / 1.25, axes[0].get_ylim()[1] * 0.98,
                 "criterion", fontsize=ps.TICK_PT, color=ps.INK, ha="right", va="top")
    axes[0].tick_params(labelbottom=False)
    axes[1].set_xlabel("participation $p_i$")
    return counts


def panel_b(ax, p):
    """One participation histogram on a log axis, silent and active units split at the criterion.

    The single-network panel the talk uses (fig_slides); Figure 1 draws panel_b_pair.

    Args:
        ax: axes; p: (N,) participation values.
    """
    thr = SILENT_REL * np.quantile(p, 0.95)
    pp = np.maximum(p, 1e-6)
    bins = np.logspace(np.log10(pp.min() * 0.7), np.log10(pp.max() * 1.4), 46)
    live = pp >= thr
    ax.hist(pp[~live], bins=bins, color=ps.FAINT, edgecolor="none", label=f"inactive ({(~live).sum()})")
    ax.hist(pp[live], bins=bins, color=ACTIVE_COL, edgecolor="none", label=f"active ({live.sum()})")
    ax.axvline(thr, color=ps.INK, lw=0.9, ls="--", zorder=5)
    top = ax.get_ylim()[1]
    ax.set_ylim(0, top * 1.28)
    ax.text(thr * 1.35, top * 1.14, "criterion", fontsize=5.8, color=ps.INK, ha="left",
            va="center")
    ax.set_xscale("log")
    ax.set(xlabel="participation  $p_i=\\mathrm{std}(r_i)+q_{0.9}(|r_i|)$", ylabel="units")
    ax.text(0.5, -0.36, "how much unit $i$'s rate moves over a trial, and how high it gets",
            transform=ax.transAxes, ha="center", va="top", fontsize=5.8, color=ps.MUTED)
    ax.legend(loc="upper left", fontsize=5.8, bbox_to_anchor=(0.0, 1.0))
    ps.ygrid(ax)


def panel_c(ax):
    """Panel (c): active units vs network size, each run read at 1.05x its own fitted loss floor.

    Not at a fixed iteration: the learning rate falls with N (lr = 1e-3 (100/N)^(1/3)), so a fixed
    iteration compares networks at different depths of convergence. Reading each one where its own
    loss settles is how this panel handles that confound. The claim it carries is that the count
    grows far slower than N, which is what makes the active FRACTION fall with size.

    A task with a size at which no run has a usable floor read-out (see floor_crossing) is drawn at
    the end of its budget instead - open markers, dashed line, and the legend says so - rather than
    dropped, because the reader is entitled to see the data the rule could not read.

    Returns:
        dict task -> (b, A, N_needed_for_1000, n_sizes, largest N, read-out) of the fit.
    """
    fits, handles = {}, []
    for task, (Ns, pats, cap, col) in SCALING.items():
        min_r2 = TASK_MIN_R2.get(task)
        cells = {}
        for N in Ns:
            rs = runs_with_loss(pats[N], min_r2=min_r2)
            total = len(runs_with_loss(pats[N]))
            if min_r2 is not None and len(rs) < total:
                print(f"  .. {task} N={N}: {total - len(rs)} of {total} seed(s) below "
                      f"r2 {min_r2} dropped, {len(rs)} kept")
            if rs:
                cells[N] = rs
        if len(cells) < 2:
            print(f"  !! {task}: {len(cells)} size(s) with a usable clean loss - series omitted")
            continue
        why = []
        read, n_dropped = at_own_floor(cells, why=why)
        for N, name, reason in why:
            print(f"  .. {task} N={N} {name[:12]}: no floor read-out - {reason}")
        readout = "floor"
        missing = sorted(set(Ns) - set(cells))
        if missing:
            print(f"  !! {task}: no runs on disk at N = {missing}")
        if len(read) < len(Ns):
            # a series is read at its floor only when EVERY size has a floor read-out; a series
            # read at its floor at some sizes and not others would compare the sizes at different
            # depths of convergence, which is what the floor rule exists to prevent
            read = at_end_of_budget(cells)
            readout = "end"
            print(f"  !! {task}: no floor read-out at N = "
                  f"{sorted(set(Ns) - set(at_own_floor(cells)[0]))} - the series is drawn at the "
                  f"END of its budget instead")
        else:
            print(f"  .. {task}: read AT {CONV_MARGIN:.2f}x each run's own fitted floor "
                  f"({n_dropped} run(s) without one); mean crossing per size "
                  + ", ".join(f"N={N}:{read[N][1]:,.0f}" for N in sorted(read)))
        xs, ys, sds = [], [], []
        mfc = col if readout == "floor" else "none"
        for N in sorted(read):
            c = read[N][0]
            xs.append(N)
            ys.append(c.mean())
            sds.append(c.std(ddof=1) if len(c) > 1 else 0.0)
            ax.plot([N] * len(c), c, "o", ms=2.4, color=col, mfc=mfc, alpha=0.55,
                    mec=col if readout == "end" else "none", mew=0.5, zorder=4)
        xs, ys, sds = np.array(xs, float), np.array(ys, float), np.array(sds, float)
        ax.errorbar(xs, ys, yerr=sds, fmt="o-" if readout == "floor" else "o--", color=col,
                    mfc=mfc, ms=3.4, lw=1.1, zorder=5, capsize=1.6, capthick=ps.DATA_LW)
        tag = "" if readout == "floor" else ", end of training"
        if task in NO_FIT:
            print(f"  !! {task}: not fitted - {NO_FIT[task]}")
            handles.append(Line2D([], [], color=col, marker="o", mfc=mfc, ms=3.0, lw=1.1,
                                  label=task + tag))
            continue
        b, loga = np.polyfit(np.log(xs), np.log(ys), 1)
        fits[task] = (b, np.exp(loga), np.exp((np.log(1000) - loga) / b), len(xs),
                      float(xs.max()), readout)
        # extrapolate only the tasks with four sizes; a three-size fit with a wide seed spread is
        # not something to project a decade beyond the data
        hi = 2.4e4 if len(xs) >= 4 else xs.max() * 1.25
        xf = np.logspace(np.log10(xs.min() * 0.85), np.log10(hi), 50)
        ax.plot(xf, np.exp(loga) * xf ** b, ls=":", lw=0.8, color=col, zorder=3)
        # built by hand: an errorbar's legend handle is a container, and letting matplotlib collect
        # handles here silently produced two entries for the same task
        handles.append(Line2D([], [], color=col, marker="o", mfc=mfc, ms=3.0, lw=1.1,
                              ls="-" if readout == "floor" else "--",
                              label=f"{task}{tag}  $\\propto N^{{{b:.2f}}}$"))

    # The y axis stops just above the 1,000-unit line rather than at the top of the "every unit
    # active" diagonal: three of the four decades the full diagonal spans hold no data at all and
    # squashed the points into the bottom fifth. The FLOOR is 100, not the 150 that fitted the
    # flip-flop and CDDM series alone -- unpenalised DMTS at N = 500 sits at 114 and 134 active
    # units, and at a floor of 150 both seeds and their whole cell were clipped off the panel
    # without any warning that they had been.
    nn = np.array([3e2, 2.6e4])
    ax.plot(nn, nn, "-", lw=0.7, color=ps.MUTED, zorder=2)
    # the 10% label sits at the lower left, where no series or extrapolation passes
    for frac, lab, xl in [(0.5, "50% active", 3.0e3), (0.1, "10% active", 7.0e2)]:
        ax.plot(nn, frac * nn, ls=(0, (4, 3)), lw=0.55, color=ps.FAINT, zorder=1)
        ax.text(xl, frac * xl * 1.12, lab, fontsize=5.2, color=ps.FAINT, ha="center", va="bottom")
    ax.axhline(1000, color=ps.BAD, lw=0.7, ls="-.", zorder=2)
    ax.set(xscale="log", yscale="log", xlabel="network size N",
           ylabel="active units", xlim=(3.2e2, 2.7e4), ylim=(20, 2.3e3))
    # The 1,000-unit line and the diagonal are named in the legend: written on the panel, both
    # labels were crossed by the diagonal or by the extrapolated fits.
    handles += [Line2D([], [], color=ps.BAD, lw=0.7, ls="-.", label="1,000 active units"),
                Line2D([], [], color=ps.MUTED, lw=0.7, ls="-", label="every unit active")]
    # The y floor sits at 20, well below every series, so the five-entry legend in the lower right
    # stays under the N = 4000 points (at 40 its top reached the flip-flop's). The
    # legend, not the axis label, says how each series was read: floor or end of training.
    ax.legend(handles=handles, loc="lower right", fontsize=5.6, handlelength=2.2)
    ps.ygrid(ax)
    return fits


# (label, glob, colour, N) for the trajectory panels, one per task. Two sources of the clean
# (dropout-off) loss are accepted, because the loss is the same quantity in both: metrics
# ["loss_clean_train"] inside the participation trace, recorded every 10 iterations, and
# TrainLosses.json for runs that predate that metric. The second is only used for a run with
# dropout off and every penalty at zero, where the training-pass loss IS the clean loss; the
# loader checks that against the run's own config rather than assuming it.
# STANDARD controls at N = 1000, re-sourced 2026-10-12. CDDM is the 200,000-iteration drift run, the
# longest standard CDDM control, because the point of the panel is what happens long after the floor.
# The flip-flop is its controls continued to 200,000 (40k + 35k + 125k, one joined trace): at 75k
# its noisy loss had not reached its floor.
TRAJ = [
    ("3-bit flip-flop", f"{DATA_DIR}/NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none",
     ps.SLOTS[0], 1000),
    ("CDDM", f"{DATA_DIR}/CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000", ps.SLOTS[1], 1000),
    (DMTS_LABEL, f"{DATA_DIR}/DMTS_paper_grid_d5/EqType=h_N=1000_arm=control", ps.SLOTS[2], 1000),
]
# how far left of the e stack the panel letter of that column sits, as a fraction of the stack's
# width; panel a's letter is placed to match it
E_LETTER_DX = 0.13
PENALTY_KEYS = ("lambda_frm", "lambda_rws", "lambda_met", "lambda_orth")


def clean_loss(run_dir, trace):
    """The dropout-off training loss of one run, and the iterations it was sampled at.

    Prefers metrics["loss_clean_train"] in the participation trace, which is recorded beside the
    participation probes and is the clean loss by construction. Falls back to TrainLosses.json
    ONLY when the run's own config shows dropout off and every penalty at zero, since the
    training-pass loss is then the same quantity; otherwise it is not, and the run is skipped
    rather than silently plotted against a different definition.

    Args:
        run_dir: path to one trained-network folder; trace: its loaded ParticipationTrace dict.
    Returns:
        (iterations, loss) arrays, or (None, None).
    """
    m = trace.get("metrics", {})
    if "loss_clean_train" in m:
        return np.asarray(trace["iters"], float), np.asarray(m["loss_clean_train"], float)
    cfg = glob.glob(os.path.join(run_dir, "*config.yaml"))
    jf = glob.glob(os.path.join(run_dir, "*TrainLosses.json"))
    if not (cfg and jf):
        return None, None
    t = OmegaConf.load(cfg[0]).trainer
    if bool(t.get("dropout")) or any(float(t.get(k, 0) or 0) for k in PENALTY_KEYS):
        return None, None
    L = np.asarray(json.load(open(jf[0])).get("train_losses", []), float)
    return (np.arange(len(L), dtype=float), L) if len(L) else (None, None)


def trajectory(run_dir):
    """Silent-unit count and NOISY training loss over training, for one run.

    Args:
        run_dir: path to one trained-network folder.
    Returns:
        dict with the loss series, the silent-unit series, the network size, the held-out r2, and
        the run's own 1.05x-floor crossing (nan when floor_crossing finds none, with the reason);
        or None if the run carries no noisy loss.
    """
    tp = glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))
    if not tp:
        return None
    t = pickle.load(open(tp[0], "rb"))
    it_L, L = noisy_loss_series(run_dir)
    if L is None or not len(L):
        return None
    P, it_P = np.asarray(t["participation"]), np.asarray(t["participation_iters"], float)
    live = np.array([active_mask(p).sum() for p in P], float)
    # ⚠️ R2 IS THE HELD-OUT SCORE, NOT THE FOLDER-NAME PREFIX. That prefix is one forward pass, one
    # noise draw, on the TRAINING batch, and panel (e) prints it in every sub-panel title. The score
    # this project reports is a held-out batch at the sigma_rec and sigma_inp the network trained
    # with, sigma_w = 0, averaged over nine draws.
    T, reason = floor_crossing(it_L, L)
    return dict(it_loss=it_L, loss=L, it_act=it_P, silent=P.shape[1] - live, N=P.shape[1],
                r2=heldout.heldout_r2(run_dir), plateau=T, no_floor=reason)


def running_median(y, frac=0.03, w_max=401):
    """Median over a sliding window whose width GROWS with the index, for display only.

    A fixed window is wrong on a log x axis. The clean loss falls by most of its total inside the
    first ~2,000 iterations, and a window 301 samples wide flattens exactly that part: normalised
    by its own smoothed start, a 28-fold drop is drawn as a 5-fold one. A window proportional to
    the index is narrow where the curve is steep and wide where it is only noisy, which is what a
    log axis asks for.

    The PLATEAU ITERATION is computed on the raw series, never on this.

    Args:
        y: 1-D array; frac: window half-width as a fraction of the index; w_max: cap in samples.

    ⚠️ CHANGING frac DOES ALMOST NOTHING, AND NOT BECAUSE THE DATA IS SMOOTH. frac only sets the
    window before w_max binds - at 0.03 that is the first 6,667 of a 50,000-sample series - so 87%
    of the curve is identical at any frac. What that hides is how much smoothing happens at all: the
    raw clean loss steps by 2.1% of its mean from one sample to the next, and the smoothed curve by
    0.4%, so 99.6% of the jaggedness is gone. A narrower cap keeps more (w_max 21 keeps 7.9%) and an
    uncapped proportional window keeps less (0.08%), but nothing in this range makes the drawn line
    look like the data. That is why panel (e) draws the raw series underneath it.
    Returns: array of the same length.
    """
    n = len(y)
    out = np.empty(n)
    for i in range(n):
        h = min(w_max // 2, int(frac * (i + 1)))
        out[i] = np.median(y[max(0, i - h):min(n, i + h + 1)])
    return out


def panel_e(axes):
    """Panel (e): every task trains to a good solution, and units keep going silent afterwards.

    One sub-panel per task, each showing EVERY seed rather than a mean. Left axis, dark: the noisy
    training loss - the loss the network is trained on - as log-binned medians, which is what
    "trained to a good level" has to be read off. Right
    axis, in the task's colour from panel c: the number of silent units, which keeps climbing after
    the loss has stopped moving. The vertical rule marks the mean of the runs' own 1.05x-floor
    crossings; a task whose loss climbs back after its minimum has no such crossing and no rule.

    The sub-panels are stacked vertically, so they share one x range and one x label.

    Args:
        axes: a list of one Axes per entry of TRAJ, top to bottom.
    Returns:
        list of (task, N, plateau iteration, silent there, silent at end, mean final r2) rows.
    """
    rows, drawn, twins, x_end = [], [], [], []
    for ax, (label, pat, col, _) in zip(axes, TRAJ):
        runs = [trajectory(d) for d in sorted(glob.glob(os.path.join(pat, "*"))) if os.path.isdir(d)]
        runs = [r for r in runs if r]
        if not runs:
            print(f"  !! {label}: no run carries a noisy loss - sub-panel left empty")
            continue
        axr = ax.twinx()
        for r in runs:
            # the loss is drawn RELATIVE TO ITS OWN START so all three sub-panels share one axis:
            # the three tasks finish near 0.035, 0.022 and 1e-4. It is the per-iteration noisy loss
            # reduced to log-binned medians - 120 bins over the run, each the median of every step
            # in it - so a 200,000-step series becomes a curve without a per-sample smoother, and
            # DMTS's brief spikes back to chance do not drag it.
            tb, yb = logbin(r["it_loss"], r["loss"], nbins=120)
            ax.plot(tb, yb / yb[0], "-", lw=0.85, color=ps.INK, alpha=0.75, zorder=4)
            axr.plot(r["it_act"][1:], r["silent"][1:], "-", lw=0.9, color=col, alpha=0.85, zorder=5)
        at_floor = [r for r in runs if np.isfinite(r["plateau"])]
        for r in runs:
            if r["no_floor"]:
                print(f"  .. panel e, {label}: a run has no floor crossing - {r['no_floor']}")
        plateau = float(np.mean([r["plateau"] for r in at_floor])) if at_floor else float("nan")
        s_plat = (float(np.mean([r["silent"][np.argmin(np.abs(r["it_act"] - r["plateau"]))]
                                 for r in at_floor])) if at_floor else float("nan"))
        s_end = float(np.mean([r["silent"][-1] for r in at_floor or runs]))
        r2 = float(np.mean([r["r2"] for r in runs]))
        N = runs[0]["N"]
        if at_floor:
            ax.axvline(plateau, color=ps.MUTED, lw=0.7, ls=(0, (3, 2)), zorder=2)

        # x starts at 8, not 90: most of the loss drop happens inside the first hundred iterations
        # and a panel that begins at 90 shows a curve already a third of the way down while its
        # axis label says "relative to its start".
        ax.set(xscale="log", yscale="log", ylim=(4e-4, 2.2))
        axr.set(ylim=(0, N * 1.04))
        axr.spines[["top"]].set_visible(False)
        # the house style hides every right spine; this axis needs its own, or the silent-units
        # ticks hang off nothing and the reader cannot tell which panel edge they belong to
        axr.spines[["right"]].set_visible(True)
        # each sub-panel now carries its own right-hand ticks, so each set is coloured by its own
        # task rather than all three by the last one
        axr.tick_params(axis="y", colors=col)
        ax.set_title(f"{label},  $r^2 = {r2:.3f}$, $N = {N}$", fontsize=5.8, color=ps.INK, pad=3)
        drawn.append(ax)
        twins.append(axr)
        x_end.append(max(r["it_loss"][-1] for r in runs))
        rows.append((label, N, plateau, s_plat, s_end, r2))

    if not drawn:
        return rows
    # one x range for the stack, so a vertical read across the three sub-panels compares the same
    # iteration; the label and its tick labels go under the bottom sub-panel only
    for ax in drawn:
        ax.set_xlim(8, max(x_end) * 1.6)
    for ax in drawn[:-1]:
        ax.tick_params(labelbottom=False)
    drawn[-1].set_xlabel("training iteration")
    # the axis labels sit on the middle sub-panel, where each one labels all three
    mid = len(drawn) // 2
    drawn[mid].set_ylabel("training loss (under noise),\nrelative to its start")
    twins[mid].set_ylabel("inactive units")
    return rows


# Panel (d). Ablation results written by ablate_silent_units.py, one JSON per network and read-out,
# in ABLATION_DIR on Della; prune-and-retrain results in RETRAIN_DIRS. (label, [(file prefix,
# read-out), ...] in order of preference, colour), top to bottom in the order of panel (e). The
# networks are the claim-1 snapshot runs, read where their noisy loss reaches 1.05x its floor; the
# flip-flop falls back to its end-of-training controls until its 200k continuations have a floor.
# Earlier results: the *_seeded folders hold the re-run of 2026-10-14 (held-out and ordering batches
# seeded); the unseeded results stay in c1_ablation and c1_retrain.
ABLATION_DIR = "/scratch/gpfs/TENGEL/pt1290/c1_ablation_79932ef"
# Prune-and-retrain folders, searched in order; per cell the first one holding results is used.
# 2026-10-31: ABLATION_DIR and this folder are the re-run at each network's crossing under read-out
# 79932ef (Della 16555113 and 16555114, entry 2026-10-31_1702), with the fine-tune's training batches
# seeded (FINETUNE_SEED). The ablations of 14 Oct (c1_ablation_seeded, c1_retrain_seeded2/rep1,
# c1_retrain_seeded) sat on snapshots chosen under the read-out of that day; one DMTS network was
# ablated 24k iterations before its current crossing.
RETRAIN_DIRS = ("/scratch/gpfs/TENGEL/pt1290/c1_retrain_79932ef",)
ABLATION = [
    ("3-bit flip-flop", [("NBitFlipFlop_c1_snapshots_c200000__EqType=h_N=1000_arm=control", "floor"),
                         ("NBitFlipFlop_ff_revive_c75000__EqType=h_k=3_N=1000_pen=none_arm=none",
                          "end")], ps.TASK_3BFF),
    ("CDDM", [("CDDM_c1_snapshots__EqType=h_N=1000_arm=control", "floor")], ps.TASK_CDDM),
    (DMTS_LABEL, [("DMTS_c1_snapshots__EqType=h_N=1000_arm=control", "floor")], ps.TASK_DMTS),
]
READOUT_TITLE = {"floor": "each at its loss floor", "end": "end of training"}
# Panel (d) of the main figure: the ABLATION entries of MAIN_TASKS and the maze, whose results are
# read from this repo (maze_ablation) and so have no Della source.
ABLATION_MAIN = [t for t in ABLATION if t[0] in MAIN_TASKS] + [(MAZE_LABEL, [], ps.TASK_MAZE)]
ABL_GRID = np.linspace(0.0, 1.0, 201)        # fraction of units removed, the common x of every curve
# panel (d) marks (round 3, dead-relu-paper#85 and #83)
D_YLIM = (0.0, 1.10)           # room above R^2 = 1 for the retrained stars and the inactive bar
D_BAR_Y = (1.055, 1.09)        # inactive-units bar, in R^2 units, above every curve and star
D_TITLE_PAD_PT = 6.0           # task title above the axes
D_SPINE_OUT_PT = 4.0           # left spine offset: 1.4 mm between the spine and the busiest-first drop
D_STAR_MS, D_TRI_MS = 5.5, 3.8 # retrained marks: star (inactive removed), triangle (random removed)
D_STRIP_JITTER = 0.06          # half-spread of the networks within one column of the retrain strip
D_STRIP_XLIM = (-0.42, 1.42)   # strip x range: columns at 0 and 1, 3.5 mm apart on a 6.5 mm strip,
                               # which leaves 1 mm of white between the star and triangle columns (#85)


def ablation_results(prefix, readout):
    """Every ablation result for one cell and read-out, as loaded dicts.

    Args:
        prefix: "<sweep>__<cell>" as ablate_silent_units names its files; readout: "end" or "floor".
    Returns:
        (ablation dicts, prune-and-retrain dicts) - see ablate_silent_units for the keys.
    """
    load = lambda pat: [json.load(open(f)) for f in sorted(glob.glob(pat))]
    tag = "retrain_floor" if readout == "floor" else "retrain"
    retrain = []
    for d in RETRAIN_DIRS:
        retrain = load(os.path.join(d, f"{prefix}__*__{tag}.json"))
        if retrain:
            break
    return load(os.path.join(ABLATION_DIR, f"{prefix}__*__{readout}.json")), retrain


def _on_grid(res, key):
    """One result's curve(s) under `key`, interpolated onto ABL_GRID (fraction removed)."""
    x = np.asarray(res["ks"], float) / res["N"]
    y = np.asarray(res[key], float)
    return (np.interp(ABL_GRID, x, y) if y.ndim == 1
            else np.array([np.interp(ABL_GRID, x, yy) for yy in y]))


def ablation_for(sources):
    """The ablation and prune-and-retrain results of the first source that has any.

    Args:
        sources: [(file prefix, read-out), ...] in order of preference, as in ABLATION.
    Returns:
        (ablation dicts, retrain dicts, read-out) - ([], [], None) when no source has results.
    """
    for prefix, ro in sources:
        res, ret = ablation_results(prefix, ro)
        if res:
            return res, ret, ro
    return [], [], None


def maze_ablation(folder=MAZE_ABLATION_DIR, N=N_UNITS):
    """Panel (d)'s maze entry: the JSONs of rl_maze_ablate.py, in the shape panel_d reads (#182).

    Every JSON of a ReLU network of size N under `folder` is read, so the sub-panel grows as seeds
    land. The maze ablation removes units from the network at the end of training (update 2,000,
    the weights in the run's *Params_Maze.npz; no earlier snapshot of a maze run is saved), not at the learned
    probe of panel (c). Its score is the success rate over fixed drops under the training noise,
    the counterpart of the supervised held-out R^2; it is stored under the r2_* keys because
    panel_d reads those, and SCORE labels the axis "success rate". The scores with exactly the
    inactive units removed sit on the k grid (rl_maze_ablate.py puts the inactive count on it); the
    score with the active units removed is interpolated between grid points. The retrain list is
    maze_retrain()'s (empty until rl_maze_retrain.py has run, when the strip reads "no data").

    Args:
        folder: str, the folder of the JSONs; N: int, the network size.
    Returns:
        (ablation dicts, maze_retrain(), "end") as ablation_for returns them; each dict holds N, n_active,
        n_silent, ks, quiet_first, busy_first, random (one list per random order), r2_full,
        r2_no_silent, r2_random_same_count, r2_no_active, iteration (the last probe's PPO update),
        label and run_dir.
    """
    res = []
    for path in sorted(glob.glob(os.path.join(folder, f"N={N}_act=relu__s*.json"))):
        with open(path) as fh:
            a = json.load(fh)
        ks, n_silent = np.asarray(a["ks"], float), a["N"] - a["n_active"]
        probes = np.load(sorted(glob.glob(os.path.join(REPO_ROOT, a["run_dir"], "*Participation.npz")))[0])
        res.append(dict(N=a["N"], n_active=a["n_active"], n_silent=n_silent, ks=a["ks"],
                        quiet_first=a["score_quiet"], busy_first=a["score_busy"],
                        random=a["score_random"], r2_full=a["score_full"],
                        r2_no_silent=float(np.interp(n_silent, ks, a["score_quiet"])),
                        r2_random_same_count=[float(np.interp(n_silent, ks, r))
                                              for r in a["score_random"]],
                        r2_no_active=float(np.interp(a["n_active"], ks, a["score_busy"])),
                        iteration=int(probes["update"][-1]), label=a["label"],
                        run_dir=a["run_dir"]))
    return res, maze_retrain(), "end"


def panel_d(axes, loaded=None, ax_key=None, strips=None, tasks=ABLATION):
    """Panel (d): the score as units are removed, one sub-panel per task, side by side.

    Every network is drawn. Units are removed by zeroing their outgoing weights and the network is
    scored on held-out trials under its training noise (ablate_silent_units.py; for the maze,
    rl_maze_ablate.py scores the success rate the same way, see maze_ablation). Three orders, all in
    the task's colour and told apart by line style: quietest first (solid; thick, mean; thin, each
    network), random order (dashed, mean over five orders and the networks; band, range) and
    busiest first (dotted, mean). The left spine is set D_SPINE_OUT_PT outward, so the busiest-first
    curve, which falls to R^2 = 0 within the first 1-2% of units, runs clear of the spine
    (dead-relu-paper#85 item 1). The inactive units (mean share of the networks) are a solid INACTIVE
    bar along the top of the axes, not a band behind the curves, so the grey never mixes with the
    random-order band (#85 item 3). The retrained networks are drawn in a strip to the right of each
    sub-panel, sharing its y axis: a star per network for every inactive unit removed and the rest
    retrained for 3,000 iterations, a triangle for as many units removed at random and retrained the
    same way (#85 item 2). Every marker is filled: the weights are the floor-crossing snapshots, and
    fill means the floor crossing in every figure (#83); stars and triangles differ by shape only.
    Scores below 0 are drawn at 0. The y axis is shared; the first sub-panel carries its score's
    label (SCORE), and a sub-panel scored by a different quantity than the one before it keeps its
    tick labels and carries its own (the maze's "success rate", #182). A task with no retrained
    networks gets "no data" in its strip.

    Args:
        axes: one Axes per entry of `tasks`, left to right, sharing the y axis;
        loaded: dict task label -> (ablation dicts, retrain dicts, read-out) as ablation_for returns
            them (the collect stage's record), or None to read the result files on disk;
        ax_key: a blank Axes above the sub-panels that holds the legend (off the data), or None to
            draw no legend;
        strips: one narrow Axes per entry of `tasks` for the retrained networks, or None to draw
            them inside `axes` at each network's inactive share (the pre-round-3 placement, kept for
            fig_cosyne2027);
        tasks: list of (label, sources, colour) as ABLATION holds them (the default; the main
            figure passes ABLATION_MAIN).
    Returns:
        list of (task, n nets, inactive, r2 intact, r2 all inactive removed, r2 same number random,
        r2 all active removed, read-out, iteration, r2 after pruning and retraining) rows, means
        over nets.
    """
    rows = []
    strips = strips if strips is not None else [None] * len(axes)
    for ax, ax_s, (label, sources, col) in zip(axes, strips, tasks):
        res, ret, readout = (ablation_for(sources) if loaded is None
                             else loaded.get(label, ([], [], None)))
        floor_r2 = TASK_MIN_R2.get(label)
        if floor_r2 is not None:
            res = [r for r in res if r["r2_full"] >= floor_r2]
        if not res:
            print(f"  !! panel d, {label}: no ablation results")
            continue
        x = 100 * ABL_GRID
        silent = np.array([r["n_silent"] / r["N"] for r in res])
        # the inactive units: a solid bar just above the data, from 0 to the mean inactive share
        ax.add_patch(Rectangle((0, D_BAR_Y[0]), 100 * silent.mean(), D_BAR_Y[1] - D_BAR_Y[0],
                               color=SILENT_GREY, lw=0, zorder=1))
        quiet = np.array([_on_grid(r, "quiet_first") for r in res])
        busy = np.array([_on_grid(r, "busy_first") for r in res])
        rand = np.concatenate([_on_grid(r, "random") for r in res])
        for q in quiet:
            ax.plot(x, q, "-", lw=ps.THIN_LW, color=col, alpha=0.45, zorder=3)
        ax.plot(x, quiet.mean(0), "-", lw=1.2, color=col, zorder=5)
        ax.fill_between(x, rand.min(0), rand.max(0), color=col, alpha=0.13, lw=0, zorder=2)
        ax.plot(x, rand.mean(0), "--", lw=0.9, color=col, zorder=4)
        ax.plot(x, busy.mean(0), ":", lw=1.1, color=col, zorder=4)
        ax.spines["left"].set_position(("outward", D_SPINE_OUT_PT))
        # fill means the floor crossing (#83): a task read at the end of a fixed budget (the maze,
        # read-out "end") gets open markers
        mark = ps.mark_fixed(col) if readout == "end" else ps.mark_floor(col)
        if ax_s is None:
            for t in ret:
                xs = 100 * t["n_silent"] / t["N"]
                ax.plot(xs, t["silent_removed"][-1][1], "*", ms=D_STAR_MS, **mark,
                        zorder=7, clip_on=False)
                if t.get("random_removed"):
                    ax.plot(xs, t["random_removed"][-1][1], "^", ms=D_TRI_MS, **mark,
                            zorder=6, clip_on=False)
        else:
            # one column per retraining, the networks spread a little within it
            dx = np.linspace(-D_STRIP_JITTER, D_STRIP_JITTER, len(ret)) if len(ret) > 1 else [0.0]
            for j, t in zip(dx, ret):
                ax_s.plot(j, max(t["silent_removed"][-1][1], 0), "*", ms=D_STAR_MS,
                          **mark, zorder=7, clip_on=False)
                if t.get("random_removed"):
                    ax_s.plot(1 + j, max(t["random_removed"][-1][1], 0), "^", ms=D_TRI_MS,
                              **mark, zorder=6, clip_on=False)
            ax_s.set(xlim=D_STRIP_XLIM, ylim=D_YLIM)
            ax_s.set_title("trained\nfurther", fontsize=ps.TICK_PT, color=ps.MUTED, pad=D_TITLE_PAD_PT)
            ax_s.set_xticks([])
            ax_s.tick_params(labelleft=False)
            ax_s.set_yticks([0, 0.5, 1.0])
            ax_s.spines["left"].set_visible(False)
            ax_s.tick_params(axis="y", length=0)
            ps.ygrid(ax_s)
            if not ret:
                ax_s.text(0.5, 0.5, "no\ndata", transform=ax_s.transAxes, ha="center",
                          va="center", fontsize=ps.TICK_PT, color=ps.MUTED)
        ax.set(xlim=(0, 100), ylim=D_YLIM)
        ax.set_xticks([0, 50, 100])
        ax.set_yticks([0, 0.5, 1.0])
        ps.ygrid(ax)
        ax.set_title(TASK_SHORT.get(label, label), fontsize=ps.LABEL_PT, color=ps.INK,
                     pad=D_TITLE_PAD_PT)
        rows.append((label, len(res), float(np.mean([r["n_silent"] for r in res])),
                     float(np.mean([r["r2_full"] for r in res])),
                     float(np.mean([r["r2_no_silent"] for r in res])),
                     float(np.mean([np.mean(r["r2_random_same_count"]) for r in res])),
                     float(np.mean([r["r2_no_active"] for r in res])),
                     readout, int(np.mean([r["iteration"] for r in res])),
                     float(np.mean([t["silent_removed"][-1][1] for t in ret])) if ret else np.nan))
    drawn = [(ax, label) for ax, (label, _s, _c) in zip(axes, tasks) if ax.has_data()]
    if not drawn:
        return rows
    for i, (ax, label) in enumerate(drawn):
        if i == 0 or score_of(label)["axis"] != score_of(drawn[i - 1][1])["axis"]:
            ax.set_ylabel(score_of(label)["axis"])
        else:
            ax.tick_params(labelleft=False)
    drawn[len(drawn) // 2][0].set_xlabel("units removed (% of the network)")
    if ax_key is not None:
        ps.blank(ax_key)
        ax_key.legend(handles=[
            Line2D([], [], color=ps.MUTED, lw=1.2, label="increasing participation"),
            Line2D([], [], color=ps.MUTED, lw=0.9, ls="--", label="random order"),
            Line2D([], [], color=ps.MUTED, lw=1.1, ls=":", label="decreasing participation"),
            Patch(color=SILENT_GREY, lw=0, label="inactive units"),
            Line2D([], [], marker="*", ls="", ms=D_STAR_MS, **ps.mark_floor(ps.MUTED),
                   label="inactive units removed,\nthen trained further"),
            Line2D([], [], marker="^", ls="", ms=D_TRI_MS, **ps.mark_floor(ps.MUTED),
                   label="as many removed at random,\nthen trained further")],
            loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=3, fontsize=ps.TICK_PT,
            handlelength=2.0, columnspacing=1.2, borderaxespad=0.0, borderpad=0.0, frameon=False)
    return rows


def csv_active(fname, **match):
    """Active-unit counts from an archived summary CSV, at CDDM N = 1000, eq = h.

    The CSVs record the SILENT fraction under the scale-free participation rule (`rel_5p95`), so
    the active count is N(1 - rel_5p95). These sweeps' raw networks were deleted; the rows are all
    that survives, which is why they are shown as their own block against their own reference.

    Args:
        fname: CSV under DATA_DIR; match: column -> value, compared as floats where possible.
    Returns:
        (n_nets,) array of active counts.
    """
    eq = match.pop("eq", "h")
    out = []
    for r in csv.DictReader(open(os.path.join(DATA_DIR, fname))):
        if r.get("eq") != eq or int(r["N"]) != N_UNITS:
            continue
        ok = True
        for k, v in match.items():
            try:
                ok &= float(r[k]) == float(v)
            except ValueError:
                ok &= r[k] == v
        if ok:
            out.append(N_UNITS * (1.0 - float(r["rel_5p95"])))
    return np.array(out)


def noise_active(sigma):
    """Active units at one recurrent-noise level, under the participation rule where available.

    THIS SWEEP SAVED NO PARTICIPATION TRACES, so its original number came from a peak-rate rule - a
    unit silent below 5% of the 95th-percentile peak rate - which put its sigma = 0.05 reference at
    524 active where the metabolic reference, same task and size at the same 30,000-iteration budget,
    reads 414. That gap was the measuring stick, not the networks.

    The trained weights are on disk, so `cddm_noise_participation.py` rebuilds each net, runs it
    noise-free and scores it with the same scale-free participation rule as every other family. When
    that CSV is present it is used and the reference reads 443, beside the metabolic sweep's 414.
    The peak-rate CSV remains the fallback, so the panel still draws if the re-score has not been run.

    Args:
        sigma: sigma_rec as it appears in the CSV.
    Returns:
        (mean active, sd, n_nets).
    """
    root = os.path.join(DATA_DIR, "CDDM_fb2792_g0_noise")
    rescored = os.path.join(root, "silent_units_per_condition_participation.csv")
    if os.path.exists(rescored):
        for r in csv.DictReader(open(rescored)):
            if r["eq"] == "h" and float(r["sigma_rec"]) == float(sigma):
                return (float(r["active_mean"]), float(r["active_std"]), int(r["n_nets"]))
    for r in csv.DictReader(open(os.path.join(root, "silent_units_per_condition.csv"))):
        if r["eq"] == "h" and float(r["sigma_rec"]) == float(sigma):
            return (N_UNITS - float(r["silent_rel_mean"]), float(r["silent_rel_std"]),
                    int(r["n_nets"]))
    return (float("nan"), float("nan"), 0)


def noise_counts(sigma):
    """Per-seed active-unit counts at one recurrent-noise level, where the re-score supplies them.

    The original sweep kept only a per-condition mean, sd and n, which is why its slide drew an
    interval where every other family draws its individual networks. `cddm_noise_participation.py`
    re-scores the trained weights and records each net, so the seeds are available again.

    Args:
        sigma: sigma_rec as it appears in the CSV.
    Returns:
        list of ints, one per net; empty when the re-score has not been run.
    """
    path = os.path.join(DATA_DIR, "CDDM_fb2792_g0_noise",
                        "silent_units_per_condition_participation.csv")
    if not os.path.exists(path):
        return []
    for r in csv.DictReader(open(path)):
        if r["eq"] == "h" and float(r["sigma_rec"]) == float(sigma):
            return [int(x) for x in r.get("active_counts", "").split(";") if x]
    return []


def panel_knobs(ax):
    """Panel (d): every intervention we ran, as a change from its own family's reference.

    Four families, each internally consistent in task, architecture, read-out iteration and
    silence criterion. Plotting changes rather than counts is what lets them share an axis.

    Returns:
        list of (family, label, delta, se, n) rows.
    """
    rows, ticks, labels, refs, y = [], [], [], [], 0.0
    bands = []

    def block(title, ref, items):
        """Draw one family: a shaded band, a title, and one interval per intervention."""
        nonlocal y
        start = y
        for label, vals, group in items:
            # The reference condition is a rung of the weight-decay ladder, not just its baseline:
            # without it the panel shows 0 and 10^-5 and 10^-4 with nothing between them and no
            # marker for where "no change" sits inside the ladder itself. Drawn as a point at zero
            # with no interval - a condition compared with itself has no uncertainty to show - in
            # the neutral reference ink rather than a categorical slot.
            if group == "reference":
                ax.plot(0, y, "o", ms=3.4, color=ps.BASE, zorder=5, mec="none")
                refs.append(len(labels))
                ticks.append(y)
                labels.append(label)
                y += 1
                continue
            if vals is None or len(vals) < 2 or len(ref) < 2:
                y += 1
                continue
            d = vals.mean() - ref.mean()
            se = np.sqrt(vals.var(ddof=1) / len(vals) + ref.var(ddof=1) / len(ref))
            col = GROUP_COL.get(group, ps.MUTED)
            ax.plot([d - 1.96 * se, d + 1.96 * se], [y, y], lw=1.0, color=col, zorder=4,
                    solid_capstyle="round")
            ax.plot(d, y, "o", ms=3.4, color=col, zorder=5, mec="none")
            tip = d + 1.96 * se if d >= 0 else d - 1.96 * se
            ax.text(tip + (14 if d >= 0 else -14), y, f"{d:+.0f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=5.5, color=col)
            rows.append((title, label, float(d), float(se), len(vals)))
            ticks.append(y)
            labels.append(label)
            y += 1
        bands.append((start - 0.6, y - 0.4, title, ref.mean(), len(ref)))
        y += 1.3

    for title, ref_pat, cap, items in TRACE_FAMILIES:
        got = live_matched(ref_pat, cap)
        if got is None:
            continue
        ref, _ = got
        block(title, ref, [(lab, None if pat is None else (live_matched(pat, cap) or (np.array([]),))[0], grp)
                           for lab, pat, grp in items])

    title, (rf, rm), items = ARCHIVE_FAMILY
    block(title, csv_active(rf, **rm),
          [(lab, csv_active(f, **m), grp) for lab, (f, m), grp in items])

    title, ref_sigma, items = NOISE_FAMILY
    rmean, rsd, rn = noise_active(ref_sigma)
    start = y
    for label, sigma, group in items:
        if group == "reference":
            ax.plot(0, y, "o", ms=3.4, color=ps.BASE, zorder=5, mec="none")
            refs.append(len(labels))
            ticks.append(y)
            labels.append(label)
            y += 1
            continue
        m, sd, n = noise_active(sigma)
        d = m - rmean
        se = np.sqrt(sd ** 2 / max(n, 1) + rsd ** 2 / max(rn, 1))
        col = GROUP_COL.get(group, ps.MUTED)
        ax.plot([d - 1.96 * se, d + 1.96 * se], [y, y], lw=1.0, color=col, zorder=4,
                solid_capstyle="round")
        ax.plot(d, y, "o", ms=3.4, color=col, zorder=5, mec="none")
        # a very negative value would put its label under the row label, so flip it inside
        if d < -250:
            ax.text(d + 1.96 * se + 16, y, f"{d:+.0f}", va="center", ha="left",
                    fontsize=5.5, color=col)
        else:
            tip = d + 1.96 * se if d >= 0 else d - 1.96 * se
            ax.text(tip + (14 if d >= 0 else -14), y, f"{d:+.0f}", va="center",
                    ha="left" if d >= 0 else "right", fontsize=5.5, color=col)
        rows.append((title, label, float(d), float(se), n))
        ticks.append(y)
        labels.append(label)
        y += 1
    bands.append((start - 0.6, y - 0.4, title, rmean, rn))

    for i, (lo, hi, title, refm, refn) in enumerate(bands):
        if i % 2 == 0:
            ax.axhspan(lo, hi, color="#f6f5f0", zorder=0)
        ax.text(0.012, lo + 0.04, f"{title}   ({refm:.0f} active)",
                transform=ax.get_yaxis_transform(), ha="left", va="bottom", fontsize=5.7,
                color=ps.INK, zorder=6)

    ax.axvline(0, color=ps.INK, lw=0.8, zorder=3)
    top = y + 0.2
    ax.annotate("", xy=(708, top), xytext=(0, top),
                arrowprops=dict(arrowstyle="-|>", lw=1.0, color=ps.SLOTS[1], mutation_scale=7))
    ax.text(354, top + 0.45, "rate penalty (Fig. 3)", ha="center", fontsize=6.0,
            color=ps.SLOTS[1])
    ax.set(yticks=ticks, yticklabels=labels, ylim=(top + 1.0, -1.0), xlim=(-430, 800),
           xlabel="change in active units")
    ps.despine(ax, keep=("bottom",))
    ax.tick_params(axis="y", length=0, labelsize=5.8)
    # Each block's reference row is drawn in the same neutral ink as its marker dot, so the label
    # and the point it names carry one colour and the asterisk is not the only thing marking it.
    # NOT bold: this style's font stack resolves both weights to the same Helvetica.ttc face, so
    # set_fontweight("bold") sets the property and changes nothing on the page.
    for i in refs:
        ax.get_yticklabels()[i].set_color(ps.BASE)
    ax.xaxis.grid(True, alpha=0.2, lw=0.5, color=ps.GRID)
    ax.set_axisbelow(True)
    return rows



# Figure 1 layout (drawing stage), in millimetres on the printed page: the canvas is W_FULL_MM
# (165.1 mm, \\textwidth) by FIG_H_MM, and every axes is placed by its left edge, top edge, width and
# height measured from the canvas's top-left corner, so the printed sizes are exact.
FIG_H_MM = 140.0               # round 7: 136 -> 140 mm for panel (e); the cap is ps.MAX_H_MM
LAYOUT_MM = {                  # name: (left, top, width, height)
    "net":   (1.0, 9.0, 52.0, 40.0),      # a, the circuit drawing
    "trace": (55.0, 10.0, 46.0, 42.0),    # a, eight random units
    "b_top": (111.0, 10.0, 51.0, 18.0),   # b, untrained
    "b_bot": (111.0, 36.5, 51.0, 18.0),   # b, trained
    "c":     (14.0, 73.0, 40.0, 25.0),    # c, active units per task
    "e":     (14.0, 114.0, 40.0, 17.5),   # e, fraction of units selective, under c
    "d_key": (69.0, 68.0, 93.0, 7.0),     # d, legend strip, above the sub-panels and off the data
    "d":     (70.5, 83.0, 18.0, 49.0),    # d, first of three side-by-side sub-panels (removal curves);
                                          # 20.5 until round 14, when the maze's own y label took 7.5 mm
}
D_GAP_MM = 3.5                 # gap between one task's retrain strip and the next task's sub-panel
D_YLABEL_MM = 7.5              # extra gap before a sub-panel with its own tick labels and y label (maze)
D_STRIP_MM = (1.5, 6.5)        # retrain strip right of each sub-panel: (gap, width)
LETTER_X_MM = {"a": 1.0, "c": 1.0, "e": 1.0, "b": 100.0, "d": 59.0}  # letters: left edges
LETTER_TOP_MM = {"a": 4.5, "b": 4.5, "c": 68.0, "d": 68.0, "e": 110.0}  # letter baseline, from the top
E_DX = 0.2                     # panel (e): the two sets of units sit this far either side of a task


def ax_mm(fig, left, top, width, height):
    """Add an axes placed in millimetres from the canvas's top-left corner.

    Args:
        fig: the figure (its size fixes the conversion); left, top, width, height: in mm.
    Returns:
        the new Axes.
    """
    W, H = fig.get_figwidth() / ps.MM, fig.get_figheight() / ps.MM
    return fig.add_axes([left / W, 1.0 - (top + height) / H, width / W, height / H])


def letter_mm(ax, letter):
    """Put a panel letter at LETTER_X_MM / LETTER_TOP_MM, through ps.panel_letter's mm offsets.

    Args:
        ax: the panel's top-left axes (its position fixes the offsets); letter: 'a' to 'd'.
    Returns:
        the Text artist.
    """
    fig = ax.figure
    H = fig.get_figheight() / ps.MM
    bb = ax.get_position()
    left_mm = bb.x0 * fig.get_figwidth() / ps.MM
    top_mm = H * (1.0 - bb.y1)
    return ps.panel_letter(ax, letter, x_mm=left_mm - LETTER_X_MM[letter],
                           y_mm=top_mm - LETTER_TOP_MM[letter])


def excluded_by_decision(name):
    """The recorded reason a panel (c) run is left out (EXCLUDED_RUNS), or None."""
    return next((why for key, why in EXCLUDED_RUNS.items() if key in name), None)


def apply_decisions(rec):
    """Re-apply EXCLUDED_RUNS to the panel (c) rows of a collect record (as fig_paper_F2_size does).

    The record of Della job 16556248 was written while seed 3374117298 was excluded; decision
    dead-relu-paper#20 draws it, and this re-flags every row so that no new Della read is needed.

    Args:
        rec: the unpickled COLLECT record (modified in place).
    Returns:
        the same record.
    """
    for r in rec["tasks"]:
        r["excluded"] = excluded_by_decision(r["name"])
    return rec


def score_of(label):
    """How a task is scored: the SCORE entry of its label, SCORE_R2 for the supervised tasks.

    Args:
        label: str, a task label as PANEL_TASKS, ABLATION and MAIN_TASKS name it.
    Returns:
        dict with tick (name in a tick label), fmt (format of the number), axis (y-axis label) and
        text (name in the printed numbers).
    """
    return SCORE.get(label, SCORE_R2)


def task_color(label):
    """The paper's colour of a task (paperstyle.TASK_COLORS), looked up by its short name.

    Args:
        label: str, a task label with an entry in TASK_SHORT.
    Returns:
        str, a hex colour.
    """
    return ps.TASK_COLORS[TASK_SHORT[label]]


def maze_rows(root=MAZE_ROOT, N=N_UNITS):
    """Panel (c)'s rows for the maze networks, in the shape of the collect record's rows (#182).

    Every ReLU run of size N without a goal cue under `root` (common.maze_load_runs) is read at its
    "learned" probe: the first probe from which the clean probe's success rate stays at or above
    MAZE_LEARNED_SUCCESS for MAZE_LEARNED_HOLD probes in a row (common.maze_learned_index), the
    maze counterpart of the supervised runs' floor crossing. The count drawn is therefore the one
    at the point where the task is first solved, not the one at the end of training (511, 434 and
    410 of 1,000 at the learned probe for seeds 1-3; 455, 423 and 404 at update 2,000). The run's
    score is the probe success rate at that probe, stored under "r2" because panel_tasks reads that
    key; SCORE names it "success" in the tick label. A run that never meets the learned rule gets
    a reason and is listed among the runs not drawn.

    Args:
        root: str, the suite folder (cell/s<seed>/ run folders); N: int, the network size.
    Returns:
        list of dicts with the collect record's keys: task (MAZE_LABEL), name, active, t_star (the
        PPO update of the learned probe), reason, covered, excluded, r2 (the success rate),
        measured, active_end, it_end (the last probe's update).
    """
    rows = []
    for r in maze_load_runs(root):
        if r["N"] != N or r["act"] != "relu" or r["cue"]:
            continue
        pr = r["probes"]
        li = maze_learned_index(pr["probe_success"])
        ok = li is not None
        rows.append(dict(task=MAZE_LABEL, name=os.path.relpath(r["folder"], root),
                         active=int(pr["active"][li]) if ok else None,
                         t_star=float(pr["update"][li]) if ok else float("nan"),
                         reason=None if ok else (f"probe success never stayed at or above "
                                                 f"{MAZE_LEARNED_SUCCESS} for {MAZE_LEARNED_HOLD} "
                                                 f"probes"),
                         covered=True, excluded=None,
                         r2=float(pr["probe_success"][li]) if ok else float("nan"), measured=ok,
                         active_end=int(pr["active"][-1]), it_end=float(pr["update"][-1])))
    return rows


def panel_tasks(ax, rows, tasks=MAIN_TASKS):
    """Panel (c): active units at N = 1000 on each main-text task, every run at its own read-out point.

    One column per task: one point per run in the task's colour, a bar at the mean, and under the
    task's name the mean score at the read-out point and the number of runs drawn. The score is the
    task's SCORE: held-out R^2 at the floor crossing for the supervised tasks, the probe success rate
    at the "learned" probe for the maze (maze_rows; the maze's end-of-training count is not drawn).
    A supervised run is drawn when it has a floor read-out (read_run's reason is None), its
    participation probes cover the crossing, and it is not in EXCLUDED_RUNS; a maze run when it
    has a learned probe. The dashed line marks half of the network. A strip and not a scatter
    against the score: the tasks sit at R^2 0.83 and 0.95 and at success 0.93, so a shared score
    axis spends most of its width between the tasks and none within one.

    Args:
        ax: axes; rows: the collect stage's panel (c) records and the maze's (maze_rows): dicts
            with task, name, active, t_star, reason, covered, excluded, r2, measured;
        tasks: the task labels to draw, left to right (default MAIN_TASKS).
    Returns:
        list of (task, active counts, scores, measured flags, read-out points, runs on disk, list
        of "name: reason" strings for the runs not drawn), one per task.
    """
    out, ticks = [], []
    for k, label in enumerate(tasks):
        col = task_color(label)
        mine = [r for r in rows if r["task"] == label]
        drawn = [r for r in mine if r["reason"] is None and r["covered"] and not r["excluded"]]
        why = [f"{r['name'][:40]}: {r['excluded'] or r['reason'] or 'trace misses the crossing'}"
               + (f" (would read {r['active']} active, R2 {r['r2']:.3f})"
                  if r["active"] is not None and np.isfinite(r["r2"]) else "")
               for r in mine if r not in drawn]
        y = np.array([r["active"] for r in drawn], float)
        r2 = np.array([r["r2"] for r in drawn], float)
        xs = k + np.linspace(-0.12, 0.12, len(y)) if len(y) > 1 else np.full(len(y), float(k))
        ax.plot([k - 0.26, k + 0.26], [y.mean()] * 2, lw=1.2, color=col, zorder=3,
                solid_capstyle="butt")
        ax.plot(xs, y, "o", ms=ps.MARKER_PT + 0.6, color=col, zorder=4)
        ax.text(k + 0.31, y.mean(), f"{y.mean():.0f}", ha="left", va="center", fontsize=ps.TICK_PT,
                color=ps.INK)
        sc = score_of(label)
        ticks.append(f"{TASK_SHORT.get(label, label)}\n{sc['tick']} = {r2.mean():{sc['fmt']}}"
                     f"\n$n$ = {len(y)}")
        out.append((label, y, r2, [r["measured"] for r in drawn], [r["t_star"] for r in drawn],
                    len(mine), why))
    ax.axhline(500, color=ps.FAINT, lw=0.6, ls=(0, (4, 3)), zorder=1)
    # the label sits at the left end of the line, over the 3BFF column (about 280 active); at the
    # right end the maze's seed-1 network (511 active) sits on the line (#182)
    ax.text(-0.45, 515, "half of the units", fontsize=ps.TICK_PT, color=ps.MUTED, ha="left",
            va="bottom")
    ax.set(xlim=(-0.5, len(tasks) - 0.45), ylim=(0, 1000),
           ylabel=ps.of_n(ps.LABELS["active"], N_UNITS))
    ax.set_xticks(range(len(tasks)))
    ax.set_xticklabels(ticks, fontsize=ps.TICK_PT, linespacing=1.25)
    ax.tick_params(axis="x", length=0, pad=3)
    ax.set_yticks([0, 250, 500, 750, 1000])
    ps.count_ticks(ax.yaxis)
    return out


def panel_selective(ax, res, seeds_d, tasks=MAIN_TASKS):
    """Panel (e): fraction of units selective for a task variable, over all units and over active units.

    One column per task, in the order of panel (c). Each network is a pair of filled markers in the
    task's colour, joined by a thin line: left, the number of selective units divided by all N
    units; right, the number of selective active units divided by the number of active units. Tick
    labels "all" and "active" name the two sets under every pair, and the task names sit on a second
    line below them, so no legend is needed (PR #147 review, row 10). A task with no network in
    `res` (the maze, until rl_maze_population_stats.py has run; #182) gets "no data" in its column.
    The maze networks are read at the end of training, a fixed budget, so their markers are open
    (mark_fixed), where the supervised ones, the floor-crossing snapshots of (d), are filled (#83).

    Args:
        ax: axes; res: the per-network results of population_stats.py (fig_ed_population.load()),
            each with "task", "seed", "N" and "S3a" = {"all", "act", "n_all", "n_act"};
        seeds_d: set of int, the seeds of the supervised panel (d) networks; the supervised
            results must be those networks (the maze's seeds are 1-3 and are not checked here);
        tasks: the task labels to draw, left to right (default MAIN_TASKS).
    Returns:
        list of (task, shares over all units, shares over active units), one per task; both arrays
        empty for a task with no data.
    """
    seeds_e = {int(str(s["seed"]).lstrip("s")) for s in res if s["task"] != TASK_SHORT[MAZE_LABEL]}
    assert seeds_e == seeds_d, ("panel e must draw the panel d networks", sorted(seeds_e), sorted(seeds_d))
    out = []
    offs = (-0.05, 0.0, 0.05)                 # the three networks of a task, side by side
    for k, label in enumerate(tasks):
        col = task_color(label)
        rows = [s for s in res if s["task"] == TASK_SHORT[label]]
        if not rows:
            ax.text(k, 0.5, "no data", ha="center", va="center", fontsize=ps.TICK_PT, color=ps.MUTED)
            out.append((TASK_SHORT[label], np.array([]), np.array([])))
            continue
        y_all = np.array([s["S3a"]["n_all"] / s["N"] for s in rows])
        y_act = np.array([s["S3a"]["act"] for s in rows])
        for dx, ya, yb in zip(offs, y_all, y_act):
            ax.plot([k - E_DX + dx, k + E_DX + dx], [ya, yb], color=col, lw=ps.THIN_LW, zorder=2)
        mark = ps.mark_fixed(col) if label == MAZE_LABEL else ps.mark_floor(col)
        for side, y in ((-E_DX, y_all), (E_DX, y_act)):
            ax.plot(k + side + np.array(offs), y, "o", ms=ps.MARKER_PT, ls="none", **mark, zorder=4,
                    clip_on=False)
        out.append((TASK_SHORT[label], y_all, y_act))
    below = offset_copy(ax.get_xaxis_transform(), fig=ax.figure, y=-11.5, units="points")
    for k, label in enumerate(tasks):
        ax.text(k, 0, TASK_SHORT[label], transform=below, ha="center", va="top", fontsize=ps.TICK_PT,
                color=ps.INK)
    ax.set(xlim=(-0.5, len(tasks) - 0.45), ylim=(0, 1))
    ax.set_ylabel("fraction of units\nselective")
    ax.set_xticks([k + side for k in range(len(tasks)) for side in (-E_DX, E_DX)])
    ax.set_xticklabels(["all", "active"] * len(tasks), fontsize=ps.TICK_PT)
    ax.tick_params(axis="x", length=0, pad=2)
    ax.set_yticks([0, 0.5, 1])
    ax.set_yticklabels(["0", "0.5", "1"])
    return out


def check_overlaps(fig):
    """Print every pair of drawn text artists whose rendered boxes overlap.

    Run on the finished canvas before saving, so a label collision is reported from the actual
    rendered extents rather than judged by eye. Collected per axes: free text, titles, axis labels,
    legend entries, and the tick labels whose tick lies inside the view (matplotlib keeps Text
    objects for ticks outside the limits, which are never drawn); plus the figure's own texts.

    Args:
        fig: the figure.
    Returns:
        list of (text a, text b) pairs that overlap; empty when none do.
    """
    fig.canvas.draw()
    rend = fig.canvas.get_renderer()
    texts = list(fig.texts)
    for ax in fig.axes:
        texts += list(ax.texts) + [ax.title, ax._left_title, ax._right_title,
                                   ax.xaxis.label, ax.yaxis.label]
        leg = ax.get_legend()
        if leg is not None:
            texts += list(leg.get_texts())
        for axis, (lo, hi) in ((ax.xaxis, sorted(ax.get_xlim())), (ax.yaxis, sorted(ax.get_ylim()))):
            for tick in axis.get_major_ticks():
                if lo <= tick.get_loc() <= hi and tick.label1.get_visible():
                    texts.append(tick.label1)
    texts = [t for t in texts if t.get_visible() and t.get_text().strip()]
    boxes = [(t, t.get_window_extent(rend)) for t in texts]
    hits = []
    for i, (ta, ba) in enumerate(boxes):
        for tb, bb in boxes[i + 1:]:
            if ba.overlaps(bb) and ba.width > 0 and bb.width > 0:
                hits.append((ta.get_text(), tb.get_text()))
    for a, b in hits:
        print(f"  !! text overlap: {a[:40]!r} <-> {b[:40]!r}")
    if not hits:
        print(f"  .. no overlapping text ({len(boxes)} text artists checked)")
    return hits


def collect(refresh=False):
    """Read every input Figure 1 draws, on Della, and pickle it to COLLECT.

    Panel (c): every run of each PANEL_TASKS cell, read with common.read_run at 1.05x its own
    noisy-loss floor, with held-out R^2 at the crossing (heldout_r2.floor_r2, R2_DRAWS draws).
    The shared held-out R^2 cache is read but never written: other jobs write it without a lock, so
    the scores this run adds go to a private copy beside COLLECT. Panel (d): the ablation and
    prune-and-retrain result files (ablation_for). Panels (a) and (b): the example network and its
    untrained twin (example_network_pair), and its excitatory fraction.

    Args:
        refresh: re-simulate the example network instead of reading CACHE.
    Returns:
        the path written.
    """
    cache = heldout.load_cache()
    n_cached = len(cache)
    rows = []
    for label, pat, _col in PANEL_TASKS:
        min_r2 = TASK_MIN_R2.get(label)
        runs = runs_with_loss(pat, min_r2=min_r2)
        n_all = len(runs_with_loss(pat))
        if n_all > len(runs):
            print(f"  .. {label}: {n_all - len(runs)} of {n_all} run(s) below R2 {min_r2} dropped")
        for r in runs:
            rd = read_run(r["run_dir"], run=r)
            excluded = next((why for key, why in EXCLUDED_RUNS.items() if key in r["name"]), None)
            row = dict(task=label, name=r["name"], active=rd["active"], t_star=float(rd["t_star"]),
                       reason=rd["reason"], covered=bool(rd["covered"]), excluded=excluded,
                       r2=float("nan"), measured=False, active_end=rd["active_end"],
                       it_end=rd["it_end"])
            if rd["reason"] is None and rd["covered"]:
                row["r2"], row["measured"] = heldout.floor_r2(rd, cache)
            rows.append(row)
            print(f"  .. {label:18} {r['name'][:60]}: active {row['active']} at t* "
                  f"{row['t_star']:,.0f}, R2 {row['r2']:.4f} "
                  f"({'measured' if row['measured'] else 'estimated'}); reason {row['reason']}; "
                  f"excluded: {excluded}")
    ablation = {label: ablation_for(sources) for label, sources, _col in ABLATION}
    # The ablations were run on 14 Oct at the snapshot nearest each network's crossing under the
    # read-out of that day. Re-read each network under the current read-out, so a stale read-out
    # point is reported rather than inherited: |current t* - ablated iteration| per network.
    floor_now = {}
    for label, (res, _ret, _ro) in ablation.items():
        for r in res:
            rd = read_run(r["run_dir"])
            floor_now[r["label"]] = dict(t_star=float(rd["t_star"]), reason=rd["reason"],
                                         active=rd["active"], iteration=int(r["iteration"]))
            print(f"  .. panel d, {label:18} {r['label'][-24:]}: ablated at {r['iteration']:,}, "
                  f"current t* {rd['t_star']:,.0f} ({rd['reason']}), active there {rd['active']} "
                  f"vs {r['N'] - r['n_silent']} in the ablation file")
    rec = dict(tasks=rows, ablation=ablation, floor_now=floor_now, example=example_record(refresh))
    os.makedirs(os.path.dirname(COLLECT), exist_ok=True)
    with open(COLLECT, "wb") as fh:
        pickle.dump(rec, fh)
    private = COLLECT.replace(".pkl", "_r2_cache.json")
    with open(private, "w") as fh:
        json.dump(cache, fh, indent=0, sort_keys=True)
    print(f"wrote {COLLECT}; {len(cache) - n_cached} new held-out R2 score(s) in {private}")
    return COLLECT


def example_record(refresh=False):
    """Panels (a) and (b)'s entry of the collect record: the example network and its untrained twin.

    Args:
        refresh: re-simulate the example network instead of reading CACHE.
    Returns:
        dict with rates (N, T, B), participation p and p_init (N,), iterations (int) and p_exc
        (float, excitatory_fraction).
    """
    rates, _targets, p, p_init, iterations = example_network_pair(refresh=refresh)
    return dict(rates=rates, p=p, p_init=p_init, iterations=iterations, p_exc=excitatory_fraction())


def collect_example(source):
    """Re-collect panels (a) and (b) only, keeping every other panel of an existing record (#165).

    Re-simulates the example network (refresh) and writes the record to COLLECT.

    Args:
        source: path of the collect record whose other panels are kept (v1: Della job 16556248).
    Returns:
        the path written.
    """
    with open(source, "rb") as fh:
        rec = pickle.load(fh)
    rec["example"] = example_record(refresh=True)
    with open(COLLECT, "wb") as fh:
        pickle.dump(rec, fh)
    ex = rec["example"]
    print(f"wrote {COLLECT} (other panels from {source}): {int(active_mask(ex['p_init']).sum())} "
          f"active untrained, {int(active_mask(ex['p']).sum())} after {ex['iterations']:,} iterations")
    return COLLECT


def main():
    """Collect Figure 1's inputs (--collect, on Della) or draw it from them. Returns the path written."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--collect", action="store_true",
                    help="on Della: read every input and write COLLECT, then stop")
    ap.add_argument("--refresh", action="store_true", help="re-simulate the example network")
    ap.add_argument("--collect-example", metavar="SOURCE",
                    help="re-simulate panels a and b only; keep the other panels of record SOURCE")
    args = ap.parse_args()
    if args.collect_example:
        return collect_example(args.collect_example)
    if args.collect:
        return collect(refresh=args.refresh)
    with open(COLLECT, "rb") as fh:
        rec = apply_decisions(pickle.load(fh))
    ex = rec["example"]
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    L = LAYOUT_MM
    ax_net, ax_tr = ax_mm(fig, *L["net"]), ax_mm(fig, *L["trace"])
    n_live, n_shown_live, n_conn = panel_a(ax_net, ax_tr, ex["rates"], ex["p"])
    ax_net.set_title(f"trained 3BFF network, $N$ = {ps.fmt_count(N_UNITS)}",
                     fontsize=ps.LABEL_PT, color=ps.INK, pad=3)
    letter_mm(ax_net, "a")

    axes_b = [ax_mm(fig, *L["b_top"]), ax_mm(fig, *L["b_bot"])]
    axes_b[1].sharex(axes_b[0])
    n_init, n_trained = panel_b_pair(axes_b, ex["p_init"], ex["p"], ex["iterations"])
    letter_mm(axes_b[0], "b")

    ax_c = ax_mm(fig, *L["c"])
    rows_c = panel_tasks(ax_c, rec["tasks"] + maze_rows())
    letter_mm(ax_c, "c")

    left, top, w, h = L["d"]
    axes_d, strips_d, x = [], [], left
    for i, (label, _src, _col) in enumerate(ABLATION_MAIN):
        if i > 0:
            x += w + sum(D_STRIP_MM) + D_GAP_MM
            if score_of(label)["axis"] != score_of(ABLATION_MAIN[i - 1][0])["axis"]:
                x += D_YLABEL_MM        # this sub-panel keeps its tick labels and has its own y label
        axes_d.append(ax_mm(fig, x, top, w, h))
        strips_d.append(ax_mm(fig, x + w + D_STRIP_MM[0], top, D_STRIP_MM[1], h))
    for ax in axes_d[1:] + strips_d:
        ax.sharey(axes_d[0])
    ax_key = ax_mm(fig, *L["d_key"])
    ablation = dict(rec["ablation"], **{MAZE_LABEL: maze_ablation()})
    rows_d = panel_d(axes_d, loaded=ablation, ax_key=ax_key, strips=strips_d, tasks=ABLATION_MAIN)
    letter_mm(ax_key, "d")

    ax_e = ax_mm(fig, *L["e"])
    seeds_d = {int(re.search(r"__s(\d+)__", lab).group(1)) for lab in rec["floor_now"]}
    rows_e = panel_selective(ax_e, load_population_stats_all(), seeds_d)
    letter_mm(ax_e, "e")

    check_overlaps(fig)
    out = ps.save(fig, "fig_paper_F1")

    print("\n--- numbers quoted in the caption and the text ---")
    print(f"  panel a: {n_live} of {N_UNITS} units active; {n_shown_live} of {N_SHOWN} randomly "
          f"drawn units active; {n_conn} connections drawn over {N_GLYPH} glyphs")
    print(f"  panel b: {n_init} of {N_UNITS} active untrained, {n_trained} after "
          f"{ex['iterations']:,} iterations")
    for task, counts, r2, measured, t_star, n_all, why in rows_c:
        print(f"  panel c: {task:18} {len(counts)} of {n_all} runs drawn; active "
              f"{', '.join(str(int(c)) for c in counts)} (mean {np.mean(counts):.1f} +- "
              f"{np.std(counts, ddof=1) if len(counts) > 1 else 0:.1f} SD); "
              f"{score_of(task)['text']} {', '.join(f'{v:.3f}' for v in r2)} ({sum(measured)} "
              f"measured on a snapshot); read-out points {', '.join(f'{t:,.0f}' for t in t_star)}")
        for w in why:
            print(f"           not drawn: {w}")
    for task, n, sil, r2f, r2s, r2r, r2a, readout, it, r2t in rows_d:
        print(f"  panel d: {task:20} {n} nets, read-out {readout} (iteration {it:,}): "
              f"{sil:.0f} inactive; {score_of(task)['text']} {r2f:.3f} intact, {r2s:.3f} with every inactive unit "
              f"removed, {r2r:.3f} with as many removed at random, {r2a:.3f} with the active ones "
              f"removed; {r2t:.3f} after pruning the inactive units and retraining")
    for label, _src, _col in ABLATION_MAIN:
        res, ret, _ro = ablation[label]
        n_in = [r["n_silent"] for r in res]
        print(f"  panel d: {label:20} inactive per network {n_in} of {res[0]['N'] if res else '?'}; "
              f"{score_of(label)['text']} with every inactive unit removed "
              + ", ".join(f"{r['r2_no_silent']:.3f}" for r in res) + "; same number at random "
              + ", ".join(f"{np.mean(r['r2_random_same_count']):.3f}" for r in res))
        for key in ("intact", "silent_removed", "random_removed"):
            v = [t[key][-1][1] for t in ret if t.get(key)]
            if v:
                print(f"           retrain 3,000 it, {key:14}: "
                      f"{', '.join(f'{x:.3f}' for x in v)} (start "
                      f"{', '.join(f'{t[key][0][1]:.3f}' for t in ret if t.get(key))})")
    for lab, d in rec["floor_now"].items():
        lag = d["t_star"] - d["iteration"]
        flag = "  !! STALE" if not np.isfinite(lag) or abs(lag) > 5000 else ""
        print(f"  panel d read-out check: {lab[-40:]} ablated at {d['iteration']:,}, current t* "
              f"{d['t_star']:,.0f} ({d['reason']}){flag}")
    for task, y_all, y_act in rows_e:
        if len(y_all) == 0:
            print(f"  panel e: {task:5} no data")
            continue
        print(f"  panel e: {task:5} fraction selective over all units "
              f"{', '.join(f'{v:.3f}' for v in y_all)} ({y_all.min():.2f}-{y_all.max():.2f}); over active "
              f"units {', '.join(f'{v:.3f}' for v in y_act)} ({y_act.min():.2f}-{y_act.max():.2f})")
    return out


if __name__ == "__main__":
    main()
