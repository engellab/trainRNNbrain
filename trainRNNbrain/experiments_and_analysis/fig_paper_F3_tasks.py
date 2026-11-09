#!/usr/bin/env python3
"""Manuscript Figure 3, cross-task rebuild: the standard training settings on 3BFF, CDDM and the maze at N = 1,000.

QUESTION (Pavel, 5 Nov 2026; dead-relu-paper#183, #184; ROUND15_BRIEF "WP cross-task"). Does a
standard training setting change the number of active units in the same direction on every task?
The rule first written in entries 2026-11-05_0903, _2259 and _2310 (the matrix file holds no rule),
restricted to networks that learned after the data were seen (entry 2026-11-06_2009), applied
per (setting value, task) cell at N = 1,000: a value INCREASES or DECREASES the count when every seed
THAT LEARNED THE TASK (the maze: a learned point, common.maze_learned_index; 3BFF and CDDM:
held-out R^2 at or above the read-out file's bar, 0.8) lies outside the control's range of seeds on
the same side; otherwise NO CHANGE; a cell with no seed that learned has NO DIRECTION (the review
of dead-relu-paper PR #188, item 1: a direction comes only from networks that reached the task's
criterion, as the maze reference comes from the learned controls). A cell with fewer finished
seeds than expected gets a provisional direction. The printout ends with the tallies the text
quotes: the values with a direction on both supervised tasks and on all three, and how many of
them agree. Every cell is read at the same-size
read-out point, 75,000 iterations on 3BFF and CDDM (the CDDM input-weight norm ladder at its
50,000-iteration budget) and update 2,000 on the maze, so every marker is OPEN (paperstyle,
dead-relu-paper#83).

PANELS (165.1 x FIG_H_MM mm, bold letters). One block per task (a 3BFF, b CDDM, c maze), each of
two axes that share the columns: the upper one the number of active units, the lower one the
performance (held-out R^2 on 3BFF and CDDM; the clean probe's success on the maze, the "learned"
level 0.9 dashed). One column per setting, one slot per value, the slots in the ladder's order, so
a ladder that spans decades (weight decay, the metabolic penalty, SGD's learning rate) is evenly
spaced in the log of its value; the first column is the task's control. Inside a slot, the seeds that
learned the task are one open marker at their mean with a thin bar over their range (the paper's
convention for open markers that would overlap, dead-relu-paper#104; the number of seeds beside
the marker when it is not 3), and behind it the band of the cell's OWN control (each ladder is read against the control its read-out file names; the bands
differ between columns where the controls do). A maze run that did not learn the task
(common.maze_learned_index) and a supervised run with R^2 below the read-out file's "learned" bar
(0.8) are crosses, in the performance row at their value or at the axis' bottom edge when below
it; a run the maze's rollback guard stopped is a cross at 0. A slot reads "unfinished" when no run
of the cell has finished; a partly finished cell carries the number of its learned runs beside the
marker, as every cell does when that number is not 3. A short dash marks a value the programme has
no cell for on the task (the maze's upper spectral-radius rung is 1.5 where the supervised tasks'
is 2.0; the input-weight cap was run at 0.1, 0.5 and 2.0 on 3BFF only), with the words "not run"
written once per block in the performance row of its first such slot (dead-relu-paper#190); "no
read at 75k" marks CDDM's tanh cell, which was read at the end of its 150,000 iterations only. The
task label of each block is the paper's name ("Maze"). Adam without weight decay is the
weight-decay column's 0 and is not repeated in the optimiser column.

WHY THE PERFORMANCE IS A SECOND ROW. Figure 3a of the current paper writes R^2 minus the default
beside the counts, and fig_ed_maze_settings draws a success row under the count row; a caption
cannot carry 34 cells x 3 tasks of performance, and a marker's fill is reserved for the read-out
point, so the second row is the one encoding that keeps every number on the page.

READ-OUT FILES ONLY. Nothing is recomputed here. 3BFF: data/read_uniform_cells_2.json
(read_uniform_cells_2.py: weight decay, activity noise, spectral radius, the recurrent-weight,
metabolic and sparsity penalties, the activations and the cap 0.3, each against the B5 Adam
control; the optimiser arms of the B5 cache, as that file's "other" table of the CDDM optimiser
group holds them) and data/read_uniform_cells.json (the caps 0.1, 0.5 and 2.0 against the
input-norm ladder's default), the Figure 3 cache data/fig_paper_F2_fails_cache.pkl (the input-weight
norm ladder). CDDM: the Figure 3 cache (weight decay, noise, the metabolic penalty, the activations,
the sparsity penalty alone, the cap 0.3 and the input-weight norm at 50,000), read_uniform_cells_2
(the spectral radius and recurrent-weight penalty with their third seeds; the optimiser). Maze (round 19,
dead-relu-paper#201): maze_arms_r19.py, the reader the Extended Data Maze settings figure shares, every
record cut at update 2,000 (the stated read; continued arms and controls hold records to 4,000, which are
not drawn here). Its settings reference is the five N = 1,000 controls learned at update 2,000 (uniform
seeds 1, 2 and the G12 seeds 4-6: 394-449 active; entry 2026-11-08_0859); uniform seed 3 (415, success
0.73) is the cross in the control slot. The metabolic-penalty arms are the reruns with
trainer.freeze_after_end=true (data/trained_RNNs_rl_maze_fz; entries 2026-11-08_0940, 2026-11-09_1221),
which replace the withdrawn runs; the other arms keep their first runs. The activations are the suite's
N = 1,000 runs cut at update 2,000 as well (since round 19 those records run to 4,000, so reading their
last probe would read 4,000). The reader's cross-check against data/round19/round19_results.json and
fz_reruns.json is printed. data/read_maze_arms.json is no longer read here.

TALLIES. Two rules, both printed: the rule of the paper (a direction from the networks that learned
the task; a value counts on a task when its cell is finished and has a direction) and the rule as
first written (every finished network gives the direction, against every finished control; a value
counts when every one of its networks finished on every task). For each, the values with a direction
on both supervised tasks and on all three tasks, and which of them change the number in the same
direction.

OUTPUT. img/internal_figures/fig_paper_F3_tasks.{pdf,svg} (paperstyle.save, vector only), and on
stdout every number drawn (per cell: the counts, performances, the control's values, the status,
the direction computed here and the one the read-out file stores), a settings x tasks table of
directions, and the text-overlap check.

Run:  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/fig_paper_F3_tasks.py
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")

import paperstyle as ps
from common import MAZE_LEARNED_SUCCESS
from fig_ed_maze import ax_mm, check_overlaps
from fig_paper_F2_fails import CACHE as F2_CACHE
from fig_paper_F5_tasks import cross_positions, draw_group
import maze_arms_r19 as M19

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUC1 = os.path.join(REPO, "data", "read_uniform_cells.json")
RUC2 = os.path.join(REPO, "data", "read_uniform_cells_2.json")
MAZE_READ = 2000                  # the stated Maze read (update); the reads at 4,000 are the Extended Data figure's
N = 1000
TASKS = ("3BFF", "CDDM", "maze")
TASK_COL = {"3BFF": ps.TASK_3BFF, "CDDM": ps.TASK_CDDM, "maze": ps.TASK_MAZE}
TASK_NAME = {"3BFF": "3BFF", "CDDM": "CDDM", "maze": "Maze"}       # the tasks as the paper names them
PERF_NAME = {"3BFF": "held-out $R^2$", "CDDM": "held-out $R^2$", "maze": "success"}
PERF_LIM = {"3BFF": (0.80, 0.975), "CDDM": (0.775, 0.85), "maze": (0.0, 1.04)}
PERF_TICKS = {"3BFF": [0.8, 0.9], "CDDM": [0.78, 0.82], "maze": [0, 0.5, 1.0]}
# the columns: (setting key, column title, [(value key, slot label), ...]) in drawing order
COLUMNS = [
    ("control", "control", [("control", "control")]),
    ("wd", "weight decay\n(default $10^{-6}$)", [("0", "0"), ("1e-5", "$10^{-5}$"), ("1e-4", "$10^{-4}$")]),
    ("inorm", "input-weight\nrow norm", [("0.5", "0.5"), ("2", "2"), ("5", "5"), ("20", "20")]),
    ("cap", "input-\nweight cap", [("0.1", "0.1"), ("0.3", "0.3"), ("0.5", "0.5"), ("2.0", "2")]),
    ("srec", "activity noise\n(default 0.05)", [("0", "0"), ("0.01", "0.01"), ("0.1", "0.1")]),
    ("srad", "spectral radius\n(default 1.2)", [("0.5", "0.5"), ("1.5", "1.5"), ("2.0", "2")]),
    ("rws", "sparsity\npenalty", [("0.05", "0.05")]),
    ("met", "metabolic\npenalty", [("0.01", "0.01"), ("0.1", "0.1"), ("1", "1"), ("10", "10")]),
    ("rwm", "recurrent-\nweight penalty", [("0.05", "0.05"), ("0.2", "0.2")]),
    ("opt", "optimizer\n(default Adam)", [("adamw", "AdamW"), ("sgd0.03", "SGD 0.03"), ("sgd0.1", "SGD 0.1"),
                                          ("sgd0.3", "SGD 0.3"), ("sgd1", "SGD 1")]),
    ("act", "activation\n(default ReLU)", [("leaky", "leaky ReLU"), ("softplus", "softplus"),
                                           ("sigmoid", "sigmoid"), ("tanh", "tanh")]),
]
# the maze arm keys of read_maze_arms.json per (setting, value)
MAZE_KEY = {("wd", "0"): "wd0", ("wd", "1e-5"): "wd1e-5", ("wd", "1e-4"): "wd1e-4",
            ("inorm", "0.5"): "inorm0.5", ("inorm", "2"): "inorm2", ("inorm", "5"): "inorm5", ("inorm", "20"): "inorm20",
            ("cap", "0.3"): "cap0.3", ("srec", "0"): "srec0", ("srec", "0.01"): "srec0.01", ("srec", "0.1"): "srec0.1",
            ("srad", "0.5"): "srad0.5", ("srad", "1.5"): "srad1.5", ("rws", "0.05"): "rws0.05",
            ("met", "0.01"): "met0.01", ("met", "0.1"): "met0.1", ("met", "1"): "met1", ("met", "10"): "met10",
            ("rwm", "0.05"): "rwm0.05", ("rwm", "0.2"): "rwm0.2", ("opt", "adamw"): "adamw",
            ("opt", "sgd0.03"): "sgd0.03", ("opt", "sgd0.1"): "sgd0.1", ("opt", "sgd0.3"): "sgd0.3", ("opt", "sgd1"): "sgd1"}
MAZE_ACT = {"leaky": "leaky_relu", "softplus": "softplus", "sigmoid": "sigmoid", "tanh": "tanh"}
# the 3BFF arms of read_uniform_cells_2.json: (setting, value) -> (group name, arm label)
FF_ARM = {("wd", "0"): ("3BFF weight decay", "wd 0"), ("wd", "1e-5"): ("3BFF weight decay", "wd 1e-5"),
          ("wd", "1e-4"): ("3BFF weight decay", "wd 1e-4"),
          ("srec", "0"): ("3BFF activity noise", "sigma_rec 0"), ("srec", "0.01"): ("3BFF activity noise", "sigma_rec 0.01"),
          ("srec", "0.1"): ("3BFF activity noise", "sigma_rec 0.1"),
          ("srad", "0.5"): ("3BFF spectral radius and recurrent-weight penalty", "spectral radius 0.5"),
          ("srad", "2.0"): ("3BFF spectral radius and recurrent-weight penalty", "spectral radius 2.0"),
          ("rwm", "0.05"): ("3BFF spectral radius and recurrent-weight penalty", "lambda_rwm 0.05"),
          ("rwm", "0.2"): ("3BFF spectral radius and recurrent-weight penalty", "lambda_rwm 0.2"),
          ("met", "0.01"): ("3BFF metabolic penalty", "lambda_met 0.01"), ("met", "0.1"): ("3BFF metabolic penalty", "lambda_met 0.1"),
          ("met", "1"): ("3BFF metabolic penalty", "lambda_met 1"), ("met", "10"): ("3BFF metabolic penalty", "lambda_met 10"),
          ("rws", "0.05"): ("3BFF sparsity penalty alone", "lambda_rws 0.05"),
          ("act", "softplus"): ("3BFF activations", "softplus beta 25"), ("act", "leaky"): ("3BFF activations", "leaky ReLU 0.01"),
          ("act", "sigmoid"): ("3BFF activations", "shifted sigmoid"), ("act", "tanh"): ("3BFF activations", "tanh"),
          ("cap", "0.3"): ("3BFF input-weight cap 0.3", "cap 0.3")}
# the CDDM arms of read_uniform_cells_2.json
CDDM_ARM = {("srad", "0.5"): ("CDDM spectral radius and recurrent-weight penalty, third seeds", "spectral radius 0.5"),
            ("srad", "2.0"): ("CDDM spectral radius and recurrent-weight penalty, third seeds", "spectral radius 2.0"),
            ("rwm", "0.05"): ("CDDM spectral radius and recurrent-weight penalty, third seeds", "lambda_rwm 0.05"),
            ("rwm", "0.2"): ("CDDM spectral radius and recurrent-weight penalty, third seeds", "lambda_rwm 0.2"),
            ("opt", "adamw"): ("CDDM optimiser", "AdamW"), ("opt", "sgd0.03"): ("CDDM optimiser", "SGD lr 0.03"),
            ("opt", "sgd0.1"): ("CDDM optimiser", "SGD lr 0.1"), ("opt", "sgd0.3"): ("CDDM optimiser", "SGD lr 0.3"),
            ("opt", "sgd1"): ("CDDM optimiser", "SGD lr 1")}
OPT_LABEL = {"adamw": "AdamW", "sgd0.03": "SGD lr 0.03", "sgd0.1": "SGD lr 0.1", "sgd0.3": "SGD lr 0.3", "sgd1": "SGD lr 1"}
# the CDDM ladders of the Figure 3 cache: (setting, value) -> (stem, rung), with the control rung per stem
CDDM_F2 = {("wd", "0"): ("slide_k_weightdecay", 0), ("wd", "1e-5"): ("slide_k_weightdecay", 2),
           ("wd", "1e-4"): ("slide_k_weightdecay", 3),
           ("inorm", "0.5"): ("slide_k_inputscale_cddm", 1), ("inorm", "2"): ("slide_k_inputscale_cddm", 2),
           ("inorm", "5"): ("slide_k_inputscale_cddm", 3), ("inorm", "20"): ("slide_k_inputscale_cddm", 4),
           ("cap", "0.3"): ("revival_cddm", 1),
           ("srec", "0"): ("slide_k_recnoise", 0), ("srec", "0.01"): ("slide_k_recnoise", 1), ("srec", "0.1"): ("slide_k_recnoise", 3),
           ("rws", "0.05"): ("slide_k_frm_rws", 2),
           ("met", "0.01"): ("slide_k_metabolic", 1), ("met", "0.1"): ("slide_k_metabolic", 2),
           ("met", "1"): ("slide_k_metabolic", 3), ("met", "10"): ("slide_k_metabolic", 4),
           ("act", "leaky"): ("slide_k_activation", 1), ("act", "softplus"): ("slide_k_activation", 2),
           ("act", "sigmoid"): ("slide_k_activation", 3)}
FF_F2 = {("inorm", "0.5"): ("slide_k_inputscale", 1), ("inorm", "2"): ("slide_k_inputscale", 2),
         ("inorm", "5"): ("slide_k_inputscale", 3), ("inorm", "20"): ("slide_k_inputscale", 4)}
F2_CONTROL_RUNG = {"slide_k_weightdecay": 1, "slide_k_inputscale_cddm": 0, "revival_cddm": 0, "slide_k_recnoise": 2,
                   "slide_k_frm_rws": 0, "slide_k_metabolic": 0, "slide_k_activation": 0, "slide_k_inputscale": 0}
N_SEEDS = 3

# layout, mm from the canvas' top-left
FIG_H_MM = 122.0                  # 133.5 -> 122 (round 17): the float with its longer caption fits the page
LEFT_MM = 15.5
PITCH_MM = 3.6                    # one value's slot
GAP_MM = 2.0                      # between two settings
SEED_DX_MM = 0.8                  # a cell's crosses sit this far apart
DASH_MM = 1.6                     # the dash of a value not run on a task
BLOCK_TOP_MM = (15.0, 48.0, 81.0)
COUNT_H_MM, PERF_H_MM, ROW_GAP_MM = 15.0, 9.5, 2.5   # the performance row keeps 9.5 mm: its y label clears the tick labels
TITLE_ROWS_MM = (2.0, 7.5)        # the column titles' baselines above the first block, staggered
TITLE_RULE_MM = 0.9               # a thin rule this far under each title, spanning the title's slots


def slot_positions():
    """The x position (mm from the axes' left edge) of every slot and of every column's centre.

    Returns:
        (slots, columns, width): slots a list of (setting, value key, label, x); columns a list of
        (setting, title, x_centre, x_left, x_right), the last two the edges of the column's slots;
        width the axes width in mm.
    """
    slots, columns, x = [], [], PITCH_MM / 2
    for g, (setting, title, values) in enumerate(COLUMNS):
        if g:
            x += GAP_MM
        xs = []
        for key, label in values:
            slots.append((setting, key, label, x))
            xs.append(x)
            x += PITCH_MM
        columns.append((setting, title, float(np.mean(xs)), xs[0] - PITCH_MM / 2, xs[-1] + PITCH_MM / 2))
    return slots, columns, x - PITCH_MM / 2


def cell(active, perf, learned, ctrl_active, ctrl_perf, n_expected, source, status=None, n_running=0,
         n_stopped=0, stored=None, note=""):
    """One (task, setting, value) cell.

    Args:
        active: list of int, active units per finished seed; perf: list of float, held-out R^2 or
            success per finished seed; learned: list of bool per finished seed (False: a cross);
            ctrl_active, ctrl_perf: the cell's own control's lists; n_expected: int; source: str;
            status: None (derived from the counts) or one of "not run", "no read"; n_running,
            n_stopped: int; stored: the direction or verdict the read-out file holds, or None;
            note: str, printed.
    Returns:
        dict with the inputs and the derived status ("finished", "incomplete", "unfinished",
        "stopped", "not run", "no read").
    """
    if status is None:
        if n_expected and len(active) == n_expected:
            status = "finished"
        elif active:
            status = "incomplete"
        elif n_running:
            status = "unfinished"
        else:
            status = "stopped"
    return dict(active=list(active), perf=list(perf), learned=list(learned), ctrl_active=list(ctrl_active),
                ctrl_perf=list(ctrl_perf), n_expected=n_expected, source=source, status=status,
                n_running=n_running, n_stopped=n_stopped, stored=stored, note=note)


def absent(ctrl_active, ctrl_perf, status, note):
    """A cell the programme has no run for ("not run") or no read at the read-out point ("no read")."""
    return cell([], [], [], ctrl_active, ctrl_perf, 0, "-", status=status, note=note)


def direction(active, ctrl_active, learned=None):
    """The direction of a setting value against its control's range of seeds.

    Only the seeds that learned the task give a direction (the review of dead-relu-paper PR #188,
    item 1).

    Args:
        active: list of int, the cell's counts; ctrl_active: the control's; learned: list of bool
            per seed (None: every seed learned).
    Returns:
        str: "increase", "decrease" or "no change"; "none learned" for a finished cell whose seeds
        all failed to learn; "-" for an empty cell.
    """
    if not active:
        return "-"
    ok = [True] * len(active) if learned is None else list(learned)
    active = [a for a, o in zip(active, ok) if o]
    if not active:
        return "none learned"
    if all(a > max(ctrl_active) for a in active):
        return "increase"
    if all(a < min(ctrl_active) for a in active):
        return "decrease"
    return "no change"


def groups_of(path):
    """The groups of a read_uniform_cells*.json file by name, with the file's thresholds.

    Args:
        path: str.
    Returns:
        (dict name -> group, dict of the file's top-level scalars).
    """
    with open(path) as fh:
        d = json.load(fh)
    return {g["name"]: g for g in d["groups"]}, {k: v for k, v in d.items() if not isinstance(v, (list, dict))}


def runs_of(arm_or_ctrl, r2_learned):
    """Counts, R^2 and learned flags of a read_uniform_cells* arm or control.

    Args:
        arm_or_ctrl: a dict with "runs"; r2_learned: float, the R^2 below which a run did not learn.
    Returns:
        (active, r2, learned) lists.
    """
    runs = arm_or_ctrl["runs"]
    return ([r["active"] for r in runs], [r["r2"] for r in runs], [r["r2"] >= r2_learned for r in runs])


def ruc_cell(group, label, source, r2_learned):
    """One cell from an arm of a read_uniform_cells*.json group, against the group's control.

    Args:
        group: the group dict; label: the arm's exact label; source: str, the file; r2_learned:
            the "learned" bar.
    Returns:
        a `cell` dict.
    """
    arm = next(a for a in group["arms"] if a["label"] == label)
    a, r2, ok = runs_of(arm, r2_learned)
    ca, cr2, _ = runs_of(group["control"], r2_learned)
    v = arm.get("verdict", {})
    return cell(a, r2, ok, ca, cr2, v.get("n_expected", len(a) or N_SEEDS), f"{source} '{label}'",
                n_running=len(arm.get("pending", [])), stored=v.get("direction") or v.get("text"))


def f2_cell(f2, stem, rung, r2_learned, note=""):
    """One cell from a rung of the Figure 3 cache against its ladder's control rung.

    Args:
        f2: the loaded cache; stem: the ladder's stem; rung: the value's rung index; r2_learned:
            the "learned" bar; note: str.
    Returns:
        a `cell` dict.
    """
    lad = next(l for l in f2["ladders"] + f2["revival"] if l["spec"]["stem"] == stem)
    fx = [r["fixed"] for r in lad["rungs"][rung]["runs"]]
    cx = [r["fixed"] for r in lad["rungs"][F2_CONTROL_RUNG[stem]]["runs"]]
    return cell([int(f["active"]) for f in fx], [f["r2"] for f in fx], [f["r2"] >= r2_learned for f in fx],
                [int(f["active"]) for f in cx], [f["r2"] for f in cx], len(fx),
                f"fig_paper_F2_fails_cache.pkl {stem} rung {rung} vs {F2_CONTROL_RUNG[stem]}", note=note)


def load_supervised():
    """Every 3BFF and CDDM cell.

    Returns:
        dict (task, setting, value) -> `cell` dict, plus (task, "control", "control") for the
        control slot.
    """
    g1, _ = groups_of(RUC1)
    g2, thr = groups_of(RUC2)
    r2_learned = thr["r2_learned"]
    with open(F2_CACHE, "rb") as fh:
        f2 = pickle.load(fh)
    out = {}
    # 3BFF
    ctrl = g2["3BFF weight decay"]["control"]
    ca, cr2, ok = runs_of(ctrl, r2_learned)
    out[("3BFF", "control", "control")] = cell(ca, cr2, ok, ca, cr2, len(ca), f"read_uniform_cells_2.json {ctrl['cell']}")
    for (setting, value), (gname, label) in FF_ARM.items():
        out[("3BFF", setting, value)] = ruc_cell(g2[gname], label, "read_uniform_cells_2.json", r2_learned)
    for (setting, value), (stem, rung) in FF_F2.items():
        out[("3BFF", setting, value)] = f2_cell(f2, stem, rung, r2_learned)
    for value in ("0.1", "0.5", "2.0"):
        out[("3BFF", "cap", value)] = ruc_cell(g1["3BFF input-weight cap"], f"cap {value}", "read_uniform_cells.json", r2_learned)
    other = g2["CDDM optimiser"]["other"]
    oca, ocr2, _ = runs_of(other["control"], r2_learned)
    for value, label in OPT_LABEL.items():
        arm = next(a for a in other["arms"] if a["label"] == label)
        a, r2, ok = runs_of(arm, r2_learned)
        out[("3BFF", "opt", value)] = cell(a, r2, ok, oca, ocr2, len(a), f"read_uniform_cells_2.json B5 '{label}'",
                                           stored=(arm.get("verdict") or {}).get("text"))
    # CDDM
    ctrl = g2["CDDM optimiser"]["control"]
    ca, cr2, ok = runs_of(ctrl, r2_learned)
    out[("CDDM", "control", "control")] = cell(ca, cr2, ok, ca, cr2, len(ca), f"read_uniform_cells_2.json {ctrl['cell']}")
    for (setting, value), (stem, rung) in CDDM_F2.items():
        note = "read at 50,000 iterations, the ladder's budget" if stem == "slide_k_inputscale_cddm" else ""
        out[("CDDM", setting, value)] = f2_cell(f2, stem, rung, r2_learned, note=note)
    for (setting, value), (gname, label) in CDDM_ARM.items():
        out[("CDDM", setting, value)] = ruc_cell(g2[gname], label, "read_uniform_cells_2.json", r2_learned)
    return out


def maze_cell(seeds, ca, cs, source, ctrl_all, stored=None):
    """One maze cell at MAZE_READ from maze_arms_r19 seed rows, against the learned controls.

    Args:
        seeds: list of maze_arms_r19.seed_row dicts; ca, cs: the learned controls' counts and success
            rates at MAZE_READ; source: str, printed; ctrl_all: every control's count at MAZE_READ (the
            reference of the every-finished-network tally); stored: the direction maze_arms_r19 computes, printed.
    Returns:
        a `cell` dict: the seeds read at MAZE_READ are its runs (a cross when not learned there), the
        seeds whose record ends before MAZE_READ are its stopped runs (a cross at 0).
    """
    pts = [s[f"at{MAZE_READ}"] for s in seeds]
    read = [p for p in pts if p["reached"]]
    n_stopped = sum(not p["reached"] for p in pts)
    c = cell([p["active"] for p in read], [p["success"] for p in read], [p["learned"] for p in read],
             ca, cs, len(seeds), source, n_stopped=n_stopped, stored=stored)
    c["ctrl_all"] = list(ctrl_all)
    return c


def load_maze():
    """Every maze cell at update MAZE_READ, from maze_arms_r19 (round 19).

    Returns:
        (cells, reference): dict (task, setting, value) -> `cell` dict, and the settings reference at
        MAZE_READ (maze_arms_r19.settings_reference: the controls learned there).
    """
    controls = M19.load_controls()
    arms = M19.load_arms()
    M19.crosscheck(controls, arms)
    ref = M19.settings_reference(controls, MAZE_READ)
    learned = [c[f"at{MAZE_READ}"] for c in controls if c[f"at{MAZE_READ}"]["learned"]]
    ca, cs = [p["active"] for p in learned], [p["success"] for p in learned]
    call = [c[f"at{MAZE_READ}"]["active"] for c in controls if c[f"at{MAZE_READ}"]["reached"]]
    out = {("maze", "control", "control"): maze_cell(controls, ca, cs,
                                                     f"Maze_suite N = 1,000 ReLU seeds 1-6 at update {MAZE_READ:,} "
                                                     "(the learned ones are the reference)", call)}
    for (setting, value), key in MAZE_KEY.items():
        a = arms[key]
        out[("maze", setting, value)] = maze_cell(
            a["seeds"], ca, cs, f"maze_arms_r19 '{key}' ({a['source']} tree)", call,
            stored=M19.setting_direction(a["seeds"], ref, MAZE_READ))
    suite = M19.load_suite_activations(MAZE_ACT.values())
    for value, act in MAZE_ACT.items():
        out[("maze", "act", value)] = maze_cell(suite[act], ca, cs,
                                                f"Maze_suite/N={N}_act={act}, update {MAZE_READ:,}", call)
    return out, ref


def fill_absent(cells):
    """Add the cells the programme has no run for, so every (task, slot) is defined.

    Args:
        cells: dict from load_supervised and load_maze (changed in place).
    Returns:
        the same dict.
    """
    for task in TASKS:
        ctrl = cells[(task, "control", "control")]
        for setting, _, values in COLUMNS:
            for key, _ in values:
                if (task, setting, key) in cells:
                    continue
                if task == "CDDM" and (setting, key) == ("act", "tanh"):
                    cells[(task, setting, key)] = absent(ctrl["ctrl_active"], ctrl["ctrl_perf"], "no read",
                                                         "CDDM_std_g0_tanh is read at the end of 150,000 iterations only")
                else:
                    cells[(task, setting, key)] = absent(ctrl["ctrl_active"], ctrl["ctrl_perf"], "not run",
                                                         "no such cell in the programme on this task")
    return cells


def draw_cell(ax_n, ax_p, x, c, col, perf_lim, note=False):
    """One cell's marks in both rows: a marker per finished seed, crosses, the mean bars, the band.

    Args:
        ax_n, ax_p: the count and performance axes (x in mm); x: the slot's centre; c: a `cell`
            dict; col: the task's colour; perf_lim: the performance axis' limits; note: bool, write
            the words "not run" in this slot (the block's first slot the programme did not run).
    Returns:
        str, what was drawn, for the printout.
    """
    for ax, lo, hi in ((ax_n, min(c["ctrl_active"]), max(c["ctrl_active"])),
                       (ax_p, min(c["ctrl_perf"]), max(c["ctrl_perf"]))):
        ax.fill_between([x - PITCH_MM / 2, x + PITCH_MM / 2], [lo] * 2, [hi] * 2, color=ps.FAINT, alpha=0.3,
                        lw=0, zorder=1)
    active, perf = np.asarray(c["active"], float), np.asarray(c["perf"], float)
    ok = np.asarray(c["learned"], bool)
    n_group = int(ok.sum())
    # the number of networks behind the marker when it is not 3, "1" included (PR #188 review, 9a)
    label = str(n_group) if c["status"] == "finished" and n_group >= 1 and n_group != N_SEEDS else None
    # the runs that did not learn, and the stopped ones, are crosses set apart along x
    crosses = [(a, p) for a, p, o in zip(active, perf, ok) if not o] + [(0.0, None)] * c["n_stopped"]
    # with crosses beside the marker the number goes above its bar, in the count row only: above the
    # performance row's marker it would meet the count axis (review of PR #193, item 11)
    draw_group(ax_n, x, active[ok], col, "o", label, label_above=bool(crosses))
    draw_group(ax_p, x, perf[ok], col, "o", None if crosses else label)
    what = [f"{a:.0f} {p:.3f}" for a, p in zip(active[ok], perf[ok])]
    for xi, (a, p) in zip(cross_positions(x, len(crosses), n_group > 0), crosses):
        ax_n.plot([xi], [a], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
        if p is not None:
            y_p = max(p, perf_lim[0] + 0.03 * (perf_lim[1] - perf_lim[0]))
            ax_p.plot([xi], [y_p], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
            what.append(f"{a:.0f} {p:.3f} (not learned)")
    if c["n_stopped"]:
        what.append(f"{c['n_stopped']} stopped")
    if c["status"] == "not run":
        # a short dash just above the count row's axis line marks a value the programme did not run
        # on this task (at mid-height it read as a value, review of PR #193); the words "not run"
        # are written once per block, in the performance row of the block's first such slot
        # (dead-relu-paper#190, Reviewer W, figures item 2)
        ax_n.plot([x - DASH_MM / 2, x + DASH_MM / 2], [0.03, 0.03], transform=ax_n.get_xaxis_transform(),
                  color=ps.MUTED, lw=ps.DATA_LW, solid_capstyle="butt", clip_on=False, zorder=2)
        if note:
            ax_p.text(x, 0.5, "not run", rotation=90, transform=ax_p.get_xaxis_transform(), fontsize=ps.TICK_PT,
                      color=ps.MUTED, ha="center", va="center")
        what.append("not run")
    elif c["status"] in ("unfinished", "no read"):
        text = {"unfinished": "unfinished", "no read": "no read at 75k"}[c["status"]]
        ax_n.text(x, 0.5, text, rotation=90, transform=ax_n.get_xaxis_transform(), fontsize=ps.TICK_PT,
                  color=ps.MUTED, ha="center", va="center")
        what.append(text + (f" ({c['n_running']} training)" if c["n_running"] else ""))
    elif c["status"] == "stopped":
        ax_n.text(x, 0.55, "stopped", rotation=90, transform=ax_n.get_xaxis_transform(), fontsize=ps.TICK_PT,
                  color=ps.MUTED, ha="center", va="center")
    elif c["status"] == "incomplete":
        # no "k/n" text: the crosses at 0 show the stopped runs, and the number beside a marker
        # gives the networks behind it (dead-relu-paper#190, Reviewer S, figures item 3)
        what.append(f"{len(c['active'])} of {c['n_expected']}" + (f", {c['n_running']} training" if c["n_running"] else ""))
    return "; ".join(what)


def draw_block(fig, task, top, cells, slots, columns, width, letter, titles):
    """One task's block: the count row and the performance row over the shared slots.

    Args:
        fig: the figure; task: the task key; top: mm, the top of the count axes; cells: every
            cell; slots, columns, width: slot_positions output; letter: the panel letter; titles:
            bool, write the column titles above this block.
    Returns:
        list of (setting, value, what was drawn) per slot.
    """
    col = TASK_COL[task]
    ax_n = ax_mm(fig, LEFT_MM, top, width, COUNT_H_MM)
    ax_p = ax_mm(fig, LEFT_MM, top + COUNT_H_MM + ROW_GAP_MM, width, PERF_H_MM)
    out = []
    first_absent = next(((s, k) for s, k, _, _ in slots if cells[(task, s, k)]["status"] == "not run"), None)
    for setting, key, label, x in slots:
        out.append((setting, key, draw_cell(ax_n, ax_p, x, cells[(task, setting, key)], col, PERF_LIM[task],
                                            note=(setting, key) == first_absent)))
    if titles:
        # the titles are staggered on two rows (34 slots in 146 mm leave no room for one row), so a
        # thin rule under each title spans that title's slots and ties it to its column
        for i, (setting, title, x, x0, x1) in enumerate(columns):
            y_mm = TITLE_ROWS_MM[i % 2]
            ax_n.text(x, 1.0 + y_mm / COUNT_H_MM, title, transform=ax_n.get_xaxis_transform(), fontsize=ps.LABEL_PT,
                      color=ps.INK, ha="center", va="bottom", linespacing=1.1)
            ax_n.plot([x0, x1], [1.0 + (y_mm - TITLE_RULE_MM) / COUNT_H_MM] * 2, transform=ax_n.get_xaxis_transform(),
                      color=ps.MUTED, lw=ps.THIN_LW, clip_on=False, zorder=1)
    for ax in (ax_n, ax_p):
        ax.set_xlim(0, width)
        ax.set_xticks([x for _, _, _, x in slots])
        ps.ygrid(ax)
    ax_n.set_ylim(0, 1040)
    ax_n.set_yticks([0, 500, 1000])
    ps.count_ticks(ax_n.yaxis)
    ax_n.set_ylabel(f"active units\n(of {ps.fmt_count(N)})")
    ax_n.tick_params(axis="x", length=0, labelbottom=False)
    ax_n.text(0.004, 0.97, TASK_NAME[task], transform=ax_n.transAxes,
              fontsize=ps.LABEL_PT, color=ps.INK, ha="left", va="top", fontweight="bold")
    ax_p.set_ylim(*PERF_LIM[task])
    ax_p.set_yticks(PERF_TICKS[task])
    ax_p.set_ylabel(PERF_NAME[task])
    if task == "maze":
        ax_p.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    if task == TASKS[-1]:
        ax_p.set_xticklabels([label for _, _, label, _ in slots], fontsize=ps.TICK_PT, rotation=90)
        ax_p.tick_params(axis="x", length=0, pad=2.5)
    else:
        ax_p.tick_params(axis="x", length=0, labelbottom=False)
    ps.panel_letter(ax_n, letter)
    return out


def report(cells, rows, ref):
    """Print every number drawn, the direction per cell and the settings x tasks table.

    Args:
        cells: every cell; rows: dict task -> draw_block output; ref: the maze reference.
    Returns:
        None.
    """
    print(f"maze reference at update {MAZE_READ:,}: {ref['n']} learned controls (seeds {ref['seeds']}), "
          f"{ref['active']:.1f} active ({ref['active_min']}-{ref['active_max']}), success {ref['success']:.3f}")
    table = {}
    for task in TASKS:
        print(f"\n--- {task}: per slot, what was drawn; direction by the rule (every seed outside the control's range on "
              "one side) and the read-out file's own ---")
        for setting, key, what in rows[task]:
            c = cells[(task, setting, key)]
            d = direction(c["active"], c["ctrl_active"], c["learned"])
            prov = "" if c["status"] == "finished" else f" ({c['status']})"
            dperf = (np.mean(c["perf"]) - np.mean(c["ctrl_perf"])) if c["perf"] else float("nan")
            table[(task, setting, key)] = (d, c["status"])
            print(f"  {setting:>7} {key:>8}: {what or '-'}; control {', '.join(map(str, c['ctrl_active']))} "
                  f"perf {', '.join(f'{p:.3f}' for p in c['ctrl_perf'])}; dperf {dperf:+.3f}; {d}{prov}"
                  + (f" | stored: {c['stored']}" if c["stored"] else "") + f" | {c['source']}"
                  + (f"; {c['note']}" if c["note"] else ""))
    print("\n--- settings x tasks: the direction per value (f = finished, i = incomplete, u = unfinished, "
          "s = stopped, n = not run / no read; 'none' = finished, no network learned) ---")
    short = {"increase": "up", "decrease": "down", "no change": "same", "none learned": "none", "-": "-"}
    print(f"  {'setting':>8} {'value':>9} | " + " | ".join(f"{t:^10}" for t in TASKS))
    for setting, _, values in COLUMNS:
        if setting == "control":
            continue
        for key, _ in values:
            print(f"  {setting:>8} {key:>9} | " + " | ".join(
                f"{short[table[(t, setting, key)][0]] + ' ' + table[(t, setting, key)][1][0]:^10}" for t in TASKS))
    # the tallies the text quotes, under the paper's rule (learned networks only): a value counts on a
    # task when its cell is finished and has a direction
    def has(t, setting, key):
        """Whether the (task, setting, value) cell is finished with a direction (not 'none learned')."""
        d, status = table[(t, setting, key)]
        return status == "finished" and d in ("increase", "decrease", "no change")
    vals = [(s, k) for s, _, values in COLUMNS if s != "control" for k, _ in values]
    print(f"\n--- tallies, the paper's rule (directions from the networks that learned; finished cells with a direction) ---")
    tally(vals, has, lambda t, s, k: table[(t, s, k)][0], short)
    # the rule as first written: every finished network gives the direction, against every finished
    # control; a value counts only where every one of its networks finished
    def dir_all(t, setting, key):
        """The direction of a cell from every finished network against every finished control."""
        c = cells[(t, setting, key)]
        return direction(c["active"], c.get("ctrl_all", c["ctrl_active"]), None)

    def has_all(t, setting, key):
        """Whether every network of the cell finished (the rule as first written)."""
        c = cells[(t, setting, key)]
        return c["status"] == "finished" and c["n_stopped"] == 0 and len(c["active"]) == c["n_expected"]
    print(f"\n--- tallies, the rule as first written (every finished network; values all of whose networks finished) ---")
    tally(vals, has_all, dir_all, short)


def tally(vals, has, dir_of, short):
    """Print how many values have a direction on both supervised tasks and on all three, and which agree.

    Args:
        vals: list of (setting, value key); has: callable (task, setting, key) -> bool, the value counts on
            that task; dir_of: callable (task, setting, key) -> direction string; short: dict direction -> word.
    Returns:
        None.
    """
    both = [(s, k) for s, k in vals if has("3BFF", s, k) and has("CDDM", s, k)]
    same2 = [(s, k) for s, k in both if dir_of("3BFF", s, k) == dir_of("CDDM", s, k)]
    all3 = [(s, k) for s, k in vals if all(has(t, s, k) for t in TASKS)]
    same3 = [(s, k) for s, k in all3 if len({dir_of(t, s, k) for t in TASKS}) == 1]
    print(f"  with a direction on both supervised tasks: {len(both)}; the same direction on both: {len(same2)}: "
          + ", ".join(f"{s} {k} ({short[dir_of('3BFF', s, k)]})" for s, k in same2))
    print(f"  with a direction on all three tasks: {len(all3)}: "
          + ", ".join(f"{s} {k} ({'/'.join(short[dir_of(t, s, k)] for t in TASKS)})" for s, k in all3))
    print(f"  the same direction on all three: {len(same3)}: "
          + ", ".join(f"{s} {k} ({short[dir_of('3BFF', s, k)]})" for s, k in same3))


def main():
    """Load every cell, draw the three blocks, print every number and the table, save the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cells = load_supervised()
    maze_cells, ref = load_maze()
    cells.update(maze_cells)
    fill_absent(cells)
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    slots, columns, width = slot_positions()
    assert LEFT_MM + width <= ps.W_FULL_MM - 2, f"the slots are {width:.1f} mm wide; the canvas is too narrow"
    rows = {}
    for task, top, letter in zip(TASKS, BLOCK_TOP_MM, "abc"):
        rows[task] = draw_block(fig, task, top, cells, slots, columns, width, letter, titles=task == TASKS[0])
    report(cells, rows, ref)
    check_overlaps(fig)
    ps.save(fig, "fig_paper_F3_tasks")


if __name__ == "__main__":
    main()
