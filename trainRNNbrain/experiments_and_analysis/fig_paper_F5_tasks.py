#!/usr/bin/env python3
"""Manuscript Figure 5, cross-task rebuild: the training techniques on 3BFF, CDDM and the maze at N = 1,000.

QUESTION (Pavel, 5 Nov 2026; dead-relu-paper#183, #184; ROUND15_BRIEF "WP cross-task"). Which
training techniques increase the number of active units on every task, and which on some tasks
only? The criterion chosen for this comparison (when each part was written: entry 2026-11-06_2259;
the matrix file holds no criterion), applied per (technique, task) cell at N = 1,000: the technique WORKS when every seed has more active units than the largest
control and the arm's mean performance is within the tolerance of the controls' mean (held-out
R^2 within R2_TOL = 0.01 on the supervised tasks; the clean probe's success within SUCCESS_TOL =
0.05 on the maze); otherwise it FAILS, with the condition that failed. Every cell is read at the
same-size read-out point: 75,000 iterations on 3BFF and CDDM, update 2,000 on the maze (the end of
its budget), so every marker is OPEN (paperstyle, dead-relu-paper#83).

PANELS (165.1 x FIG_H_MM mm, bold letters, a key row on top).
  (a) fold change in the number of active units over the task's own control (the control's mean
      at the same read-out point), one column per technique, inside it one group per task
      (3BFF, CDDM, maze: the task colours), one open marker per group at the mean over every
      finished seed, the mean the criterion judges (dead-relu-paper#190, Reviewer R item 4), with a
      thin bar over their range (the paper's convention for open markers that would overlap,
      dead-relu-paper#104; the number of seeds beside it when it is not 3) and, where a
      technique was run at several values on a task, one group per value (the value written
      under the column); the band behind a group is the control's own range of seeds. A run the maze's rollback guard stopped
      is a cross at 0; a maze run that did not learn the task (common.maze_learned_index) is
      inside its group's mean and also a cross at its value; a group with no finished run says
      "unfinished", a partly finished one carries the number of its finished seeds beside the
      marker. The task labels are the paper's ("Maze"). The columns are ORDERED by the number of tasks on which the technique works (ties: by
      the number of tasks on which the count condition passes, then the name).
      Round 19 (dead-relu-paper#201, r19-text): every value run on all three tasks is drawn, so the round 18
      parity values join (noisy synapses sigma_w 0.1 and 0.3, read-out dropout rate 0.05 with beta 4, rescaling
      target 4.0: PARITY), inside each technique's slots in increasing order of value (value_key); and on Maze
      the rate penalty, rate + sparsity and prune-and-duplicate are the trainer.freeze_after_end=true reruns
      (entries 2026-11-09_0136 and _1221; the earlier runs averaged the penalties over post-episode steps and
      their verdicts are withdrawn). The other Maze techniques keep their original runs. The printout adds the
      verdicts under the one-value and the matched-values rules and the tolerance range over which
      prune-and-duplicate alone passes on all three tasks (rule_tables).
  (b) the performance difference from the controls' mean: top, held-out R^2 (3BFF and CDDM), the
      tolerance 0.01 dashed; bottom, the maze's success at the end, the tolerance 0.05 dashed.
  (c) the techniques across network sizes, two sub-columns. 3BFF, the current Figure 5c:
      prune-and-duplicate's fold change (top) and R^2 change (bottom) per network against N, read
      at the floor crossing (filled), the other techniques as context lines through their means
      (rescaling and read-out dropout at 75,000 iterations, dashed, as Figure 5c draws them). CDDM
      (round 18, G6; dead-relu-paper#201 decision 4): prune-and-duplicate per network and read-out
      dropout as a context line, both at the floor crossing (data/round18/g6_reads.json, the
      CDDM_paper_grid networks at N = 500 to 4,000 against the same-size control; R^2 estimated
      from the final weights, as the G6 read-out does). One technique key for both sub-columns.
  (d) removing the added units, one marker per network, by ONE procedure on all three tasks since
      round 18 (dead-relu-paper#201 decision 3): the k added units removed, the inactive units left
      in place, against k active units removed at random (median over 5 draws). k is a network's
      number of active units minus the controls' mean; its k added units ("recruited" in the read-out
      files) are its k least participating active units. 3BFF: the removal analysis A3
      (data/remedy_reads_31oct/a3_N1000.json, held-out R^2 at 75,000 iterations, 19 networks). CDDM:
      G3 (data/round18/g3_cddm_removal.json, held-out R^2, 9 draws; prune-and-duplicate and read-out
      dropout at their 100,000-iteration weights against the paper-grid control, the rate penalty 0.1
      and 0.2 at 75,000 against the Figure 3a control; 14 networks). Maze (round 19): G4's procedure on
      the freeze_after_end=true reruns at update 2,000 (data/round19/fz_maze_added_summary.json,
      round19_fz_removal.py over rl_maze_ablate.py --added 430: success under the training noise, 256
      drops x 4 draws, k from all three controls' mean of 430; prune-and-duplicate and the rate penalty
      at 0.03, 0.1 and 0.3, 12 networks). Round 18's G4 on the runs before the fix
      (data/round18/g4_maze_added_summary.json) is no longer drawn.

READ-OUT FILES ONLY. Nothing is recomputed here. 3BFF: the remedies catalogue
(data/remedy_reads_31oct/remedies_catalogue.json, read_remedies.py; the Figure 5 size series,
fig_paper_F5_drive.size_series) at 75,000 iterations (active_75k, r2_75k: 36 noise draws); the
rate penalty's 75,000-iteration weights are not in the catalogue's chain, so its R^2 is the
removal analysis's intact score (a3_N1000.json, 9 draws) against the same analysis's controls,
and the printout says so. Its floor-crossing reads are printed beside the 75,000 ones. CDDM:
data/read_uniform_cells.json (read-out dropout, prune-and-duplicate, noisy synapses against the
paper-grid control), data/read_uniform_cells_2.json (rescaling, the rate penalty 0.1 alone and with
the sparsity penalty) and the Figure 3 cache data/fig_paper_F2_fails_cache.pkl (the rate penalty
0.2 alone and with the sparsity penalty, against the Figure 3a control). Maze:
data/read_maze_arms.json (read_maze_arms.py; all three controls, dead-relu-paper#194) for the controls and
the original arms; data/round19/fz_reruns.json (round19_read_fz.py) for the reruns, each seed's record cut
at update 2,000. Parity values (round 18 training package, entry 2026-11-08_0859): 3BFF from G2
(data/round18/g1g2.json, 36 draws at 75,000, against G2's control, the four default networks of Fig. 1),
CDDM from data/round18_training/round18_cddm.json (against its CDDM_paper_grid control at 75,000), Maze
from data/round18_training/round18_maze.json. The one-value rule's 3BFF verdicts in the printout use every
G2 value (load_3bff_all_values), as G2 (a) does; the drawn 3BFF values give the same verdict at 0.01.
Panels c (CDDM) and d (CDDM, Maze): the round-18 JSONs named above (entry
2026-11-07_0235_round18-results, G3, G4, G6).

OUTPUT. img/internal_figures/fig_paper_F5_tasks.{pdf,svg} (paperstyle.save, vector only), and on
stdout every number drawn (per cell: the runs' counts, fold changes, performances and differences,
the control's values, the status), the floor reads of 3BFF, the verdict table (technique x task,
computed here by the one rule and checked against the verdict each read-out file stores), the
column order and the text-overlap check.

Run:  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/fig_paper_F5_tasks.py
"""
import argparse
import json
import os
import pickle

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D

import paperstyle as ps
import fig_paper_F5_drive as F5
from common import DATA_DIR, MAZE_LEARNED_SUCCESS
from fig_ed_maze import ax_mm, check_overlaps
from fig_paper_F2_fails import CACHE as F2_CACHE

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUC1 = os.path.join(REPO, "data", "read_uniform_cells.json")
RUC2 = os.path.join(REPO, "data", "read_uniform_cells_2.json")
MAZE_JSON = os.path.join(REPO, "data", "read_maze_arms.json")
R18 = os.path.join(REPO, "data", "round18")
G3_JSON = os.path.join(R18, "g3_cddm_removal.json")          # CDDM removal of the added units (G3)
G4_JSON = os.path.join(R18, "g4_maze_added_summary.json")    # Maze removal of the added units (G4; old runs, not drawn since round 19)
R18T = os.path.join(REPO, "data", "round18_training")
R18_CDDM = os.path.join(R18T, "round18_cddm.json")           # CDDM parity values at 75,000 (round 18 training package)
R18_MAZE = os.path.join(R18T, "round18_maze.json")           # Maze parity values at update 2,000 (round 18 training package)
R19 = os.path.join(REPO, "data", "round19")
FZ_JSON = os.path.join(R19, "fz_reruns.json")                # Maze freeze_after_end=true reruns (round 19)
FZ_REMOVAL_JSON = os.path.join(R19, "fz_maze_added_summary.json")   # G4's removal on the rerun networks (round19_fz_removal.py)
G6_READS = os.path.join(R18, "g6_reads.json")                # CDDM techniques across sizes (G6)
G1G2_JSON = os.path.join(R18, "g1g2.json")                   # 3BFF rate penalty at 0.01 and 0.03 at 75,000 (G1; #204)
CDDM_SIZE_ARMS = (("duplicate", "duplication"), ("dropout", "mute"))   # panel c, CDDM: (technique, arm)
G3_TECH = {"prune-and-duplicate": "duplicate", "read-out dropout": "dropout",
           "rate penalty 0.1": "frm", "rate penalty 0.2": "frm"}
N = 1000
N_SEEDS = 3                      # the groups' usual size; another size is written beside the marker
TASKS = ("3BFF", "CDDM", "maze")
TASK_COL = {"3BFF": ps.TASK_3BFF, "CDDM": ps.TASK_CDDM, "maze": ps.TASK_MAZE}
TASK_NAME = {"3BFF": "3BFF", "CDDM": "CDDM", "maze": "Maze"}       # the tasks as the paper names them
# technique key -> (column title, marker; the shapes are Figure 5's, fig_paper_F5_drive.MARK)
TECH = {
    "duplicate": ("prune-and-\nduplicate", F5.MARK["duplicate"]),
    "frm": ("rate penalty\n$\\lambda_{\\mathrm{frm}}$", F5.MARK["frm"]),
    "dropout": ("read-out\ndropout", F5.MARK["dropout"]),
    "frm+rws": ("rate penalty $\\lambda_{\\mathrm{frm}}$\n+ sparsity 0.05", "<"),
    "rescale": ("rescaling", F5.MARK["rescale"]),
    "synnoise": ("noisy\nsynapses", F5.MARK["synnoise"]),
}
# rate + sparsity takes a left triangle: Figure 5's filled plus ("P") has no white inside left once
# it is drawn open with the coloured ring and the ink rim (tests/scan_open_markers.py, rule 1)
TRIANGLES = ("^", "v", "<")       # drawn TRI_SCALE larger: a triangle has 64% of a circle's area
TECH_NAME = {"duplicate": "prune-and-duplicate", "frm": "rate penalty", "dropout": "read-out dropout",
             "frm+rws": "rate + sparsity", "rescale": "rescaling", "synnoise": "noisy synapses"}
# the maze arms per technique: (value label, arm key, source). Sources: "arms" read_maze_arms.json (the
# original runs); "fz" data/round19/fz_reruns.json (the freeze_after_end=true reruns, which replace the
# withdrawn penalty runs and prune-and-duplicate; entries 2026-11-09_0136 and _1221); "parity"
# data/round18_training/round18_maze.json (the round 18 parity values)
MAZE_ARMS = {
    "duplicate": [("", "duplicate", "fz")],
    "frm": [("0.03", "frm0.03", "fz"), ("0.1", "frm0.1", "fz"), ("0.3", "frm0.3", "fz")],
    "dropout": [("0.05", "dropout0.05_beta4", "parity"), ("0.2", "dropout", "arms")],
    "frm+rws": [("0.1", "frm0.1_rws0.05", "fz")],
    "rescale": [("4.0", "rescale_tgt4", "parity"), ("10\u00d7", "rescale", "arms"),
                ("slow tilt", "rescale_mild", "arms"), ("5% grow", "rescale_few", "arms")],
    "synnoise": [("0.1", "sigmaw0.1", "parity"), ("0.3", "sigmaw0.3", "parity"), ("1.0", "synnoise", "arms")],
}
# the round 18 parity values (entry 2026-11-07_0136): the 3BFF values that meet the criterion, run on CDDM and
# Maze. (technique, label, the start of the 3BFF G2 name in data/round18/g1g2.json, the CDDM key of
# round18_cddm.json's "parity")
PARITY = [("synnoise", "0.1", "EqType=h_k=3_N=1000_sw=0.1", "sigma_w 0.1"),
          ("synnoise", "0.3", "EqType=h_k=3_N=1000_sw=0.3", "sigma_w 0.3"),
          ("dropout", "0.05", "EqType=h_k=3_N=1000_pen=none_do=mute_rate=0.05_beta=4", "dropout 0.05, beta 4"),
          ("rescale", "4.0", "EqType=h_k=3_N=1000_op=rescale_novel=false_tgt=4.0_step=0.0005_prot=0.20",
           "rescaling, target 4.0")]
# the order of values inside one technique's group of slots
LABEL_ORDER = {"": 0.0, "10\u00d7": 10.0, "slow tilt": 11.0, "5% grow": 12.0}
# the value labels as printed under panel b (round 20, dead-relu-paper #212, Reviewer W item 14): the rescaling
# value m, and the two Maze variants of m = 10 by the names Methods: Training techniques and Supplementary
# Table 2 give them in round 20 (dead-relu-paper PR #215): "smaller factor" (alpha = 1.0005, the cumulative
# factor capped at 8) and "fewer units" (at most 0.05 N units rescaled at a time). The labels
# inside the cells stay as they are, because round18_g1g2.py matches cells by them.
LABEL_SHOWN = {"10\u00d7": "10", "4.0": "4", "slow tilt": "smaller factor", "5% grow": "fewer units"}
# the maze's rescaling variants (entry 2026-11-05_0832), named as fig_ed_maze_interventions names them:
# "slow tilt" is alpha 1.0005 and cap 8 (the paper's arm has 1.002 and 64), "5% grow" protects 0.05 N
# units (the paper's 0.2 N); "10x" is the paper's arm
MAZE_RESCALE_NOTE = {"rescale": "the paper's values (target 10x, alpha 1.002, cap 64, protect 0.2 N)",
                     "rescale_mild": "slow tilt: alpha 1.0005, cap 8", "rescale_few": "5% grow: protect 0.05 N"}

# layout, mm from the canvas' top-left: one slot per (technique, task, value) group
FIG_H_MM = 130.0                  # 136 -> 128 mm (round 15) -> 112.5 (round 17) -> 130 (round 20: printed at 100%, caption cut to 350 words, panel c enlarged)
LEFT_MM = 16.0
PITCH_MM = 3.6                    # one group's slot (5.4 until #204; 5.1 until the parity values joined in round 19)
GAP_MM = 2.0                      # between two techniques (3.2 until #204; 2.8 until round 19)
SEED_DX_MM = 0.85                 # a cell's crosses sit this far apart
CROSS_DX_MM = 1.3                 # a cross beside a group marker sits this far from it
KEY_TOP_MM = 0.5
A_TOP_MM, A_H_MM = 14.5, 16.0
# round 20 (dead-relu-paper #212, Reviewers S and W): 6 mm between panel a's axis and panel b's top, so the
# letter b no longer sits beside panel a's "0" tick
B1_TOP_MM, B1_H_MM = 36.5, 13.5
B2_TOP_MM, B2_H_MM = 52.0, 15.0
ROW2_TOP_MM, ROW2_H_MM = 84.0, 36.0      # panel c's height; both panels end at ROW2_TOP_MM + ROW2_H_MM
D_H_MM = 30.0                     # panel d: 29 to 30 mm; its open markers overlap in 27 and from 32 (tests/scan_open_markers.py, a see-through pair in 3BFF)
C_LEFT_MM, C_W_MM, C_GAP_MM = 16.0, 31.0, 2.5     # panel c: two sub-columns (3BFF, CDDM) of C_W_MM each, sharing y (26 mm until round 20)
D_LEFT_MM, D_W_MM, D_GAP_MM = 93.5, 22.0, 2.5     # panel d: three sub-panels (3BFF, CDDM, Maze) of D_W_MM each (at least 22 mm, tests/scan_open_markers.py), sharing y
C_DODGE = 1.07                   # panel c: the techniques' markers at one size sit this factor apart along the log x axis
FOLD_LIM = (0.0, 4.3)
R2_LIM = (-0.038, 0.013)
SUCC_LIM = (-0.04, 1.06)          # round 20: panel b's Maze row draws each value's success rate itself (0 to 1), not its difference from the control mean
D_LO = -0.05                      # panel d: values below this are drawn at the edge with an arrow
EDGE_JOIN = 0.03                  # panel d: edge markers of one series closer than this (data units) are one marker
EDGE_ARROW_PT = 4.0               # panel d: arrow length (points) of a merged edge marker, its tail short of the circle at (0.02, 0.54) (PR #202 third review); fig_paper_F5_drive's is 9


def rule(active, perf, ctrl_active, ctrl_perf, tol, floor=None):
    """The criterion's verdict on one cell: works, or fails with the condition that failed.

    Args:
        active: list of int, the arm's active counts (one per seed); perf: list of float, the
            arm's performance per seed (held-out R^2 or success); ctrl_active, ctrl_perf: the
            same for the control; tol: the tolerance on the mean performance; floor: float or
            None, the level no network may end below (the maze's 0.9, issue #5:
            dead-relu-paper#194), or None for no such condition.
    Returns:
        dict: count_ok (bool), perf_ok (bool), floor_ok (bool), works (bool), fold (float, mean count over the
        control's mean), dperf (float, mean performance minus the control's mean), text (str).
    """
    fold = float(np.mean(active)) / float(np.mean(ctrl_active))
    dperf = float(np.mean(perf)) - float(np.mean(ctrl_perf)) if len(perf) else float("nan")
    count_ok = bool(len(active)) and all(a > max(ctrl_active) for a in active)
    perf_ok = bool(len(perf)) and dperf >= -tol
    floor_ok = floor is None or all(p >= floor for p in perf)
    why = []
    if not count_ok:
        why.append(f"count not above every control's ({max(ctrl_active)})")
    if not perf_ok:
        why.append(f"performance {dperf:+.3f} beyond {tol}")
    if not floor_ok:
        why.append(f"a network ends below {floor}")
    works = count_ok and perf_ok and floor_ok
    return dict(count_ok=count_ok, perf_ok=perf_ok, floor_ok=floor_ok, works=works, fold=fold, dperf=dperf,
                text="works" if works else "fails: " + "; ".join(why))


def cell(tech, task, label, active, perf, ctrl_active, ctrl_perf, n_expected, source, learned=None,
         n_stopped=0, n_running=0, stored=None, note=""):
    """One (technique, task, value) cell of panels a and b.

    Args:
        tech, task: keys of TECH and TASKS; label: str, the value written under the column ("" for
            a technique with one value on this task); active: list of int, active units at the
            read-out point, one per finished seed; perf: list of float, held-out R^2 or success per
            finished seed (may be shorter than `active` where a run has no performance read);
            ctrl_active, ctrl_perf: the control's lists; n_expected: int, seeds the arm was run
            with; source: str, the read-out file; learned: list of bool per finished seed (the
            maze), or None; n_stopped: int, runs the guard stopped (crosses at 0); n_running: int,
            runs still training; stored: str or None, the verdict the read-out file holds; note:
            str, printed.
    Returns:
        dict with the inputs plus status ("finished", "incomplete", "unfinished", "stopped"), fold
        (per seed), dperf (per seed) and the band (the control's range in fold, dperf and perf units).
    """
    ctrl_mean, perf_mean = float(np.mean(ctrl_active)), float(np.mean(ctrl_perf))
    if len(active) == n_expected:
        status = "finished"
    elif active:
        status = "incomplete"
    elif n_running:
        status = "unfinished"
    else:
        status = "stopped"
    return dict(tech=tech, task=task, label=label, active=list(active), perf=list(perf),
                ctrl_active=list(ctrl_active), ctrl_perf=list(ctrl_perf), n_expected=n_expected,
                source=source, learned=list(learned) if learned is not None else [True] * len(active),
                n_stopped=n_stopped, n_running=n_running, stored=stored, note=note, status=status,
                fold=[a / ctrl_mean for a in active], dperf=[p - perf_mean for p in perf],
                band_fold=(min(ctrl_active) / ctrl_mean, max(ctrl_active) / ctrl_mean),
                band_dperf=(min(ctrl_perf) - perf_mean, max(ctrl_perf) - perf_mean),
                band_perf=(min(ctrl_perf), max(ctrl_perf)))


def load_3bff():
    """The 3BFF cells at 75,000 iterations from the remedies catalogue (Figure 5's size series).

    Returns:
        (cells, floor): cells is a list of `cell` dicts, one per technique; floor a list of dicts
        per technique with the floor-crossing read (active, r2, the control's, n_read of n_runs)
        for the printout.
    """
    cat = F5.load_catalogue()
    series = F5.size_series(cat)
    a3 = F5.load_a3()[N]
    ctrl = series["control"][N]
    a3_ctrl_r2 = [r["r2_full"] for r in a3 if r["family"] == "control"]
    labels = {"frm": "0.1", "frm+rws": "0.1", "rescale": "10\u00d7", "synnoise": "1.0", "dropout": "0.2"}
    out, floor = [], []
    for fam in TECH:
        c = series[fam][N]
        perf, ctrl_perf, note = c["r2_75k"], ctrl["r2_75k"], "R^2: catalogue, 36 draws"
        if not perf:
            perf = [r["r2_full"] for r in a3 if r["family"] == fam]
            ctrl_perf = a3_ctrl_r2
            note = "R^2: no 75,000-iteration weights in the catalogue's chain; the removal analysis's intact score (9 draws) against its controls"
        if fam == "frm":
            out += load_3bff_rate_low()
        out.append(cell(fam, "3BFF", labels.get(fam, ""), c["active_75k"], perf, ctrl["active_75k"],
                        ctrl_perf, c["n_runs"], f"remedies_catalogue.json {c['cell'].split('/')[0]}", note=note))
        floor.append(dict(tech=fam, active=c["active"], r2=c["r2"], n_read=c["n_read"], n_runs=c["n_runs"],
                          ctrl_active=ctrl["active"], ctrl_r2=ctrl["r2"], excluded=c["excluded"]))
    out += load_3bff_parity()
    return out, floor


def g2_rows():
    """Every 3BFF value of round 18's G2 (data/round18/g1g2.json) with G2's control.

    Returns:
        (rows, control): rows is the list of G2 dicts (name, family, active, r2, ...), control the
        dict of the four default networks (active) and their R^2 (three with 75,000-iteration weights).
    """
    with open(G1G2_JSON) as fh:
        g = json.load(fh)
    return g["g2"], g["control"]


def load_3bff_parity():
    """The round 18 parity values on 3BFF, from G2 (36 draws on the 75,000-iteration weights) against G2's control.

    Returns:
        list of `cell` dicts, one per PARITY entry.
    """
    rows, ctrl = g2_rows()
    out = []
    for tech, label, name, _ in PARITY:
        r = next(r for r in rows if r["name"] == name)
        out.append(cell(tech, "3BFF", label, r["active"], r["r2"], ctrl["active"], ctrl["r2"], r["n_runs"],
                        f"round18 g1g2.json G2 {r['source']}", note="parity value; R^2: G2, 36 draws at 75,000"))
    return out


def load_3bff_all_values():
    """Every 3BFF value of G2 (57 values plus the rate penalty) as `cell` dicts, for the one-value rule's range.

    Not drawn: the figure draws the values run on all three tasks and the rate penalty's three values.
    Returns:
        list of `cell` dicts (label = the G2 name).
    """
    rows, ctrl = g2_rows()
    return [cell(r["family"], "3BFF", r["name"], r["active"], r["r2"], ctrl["active"], ctrl["r2"], r["n_runs"],
                 f"round18 g1g2.json G2 {r['source']}") for r in rows]


def load_3bff_rate_low():
    """The 3BFF rate penalty at lambda_frm = 0.01 and 0.03, read at 75,000 iterations by round 18's G1 (#204).

    Reads data/round18/g1g2.json: per network the active count (trace probe at 75,000) and held-out
    R^2 (36 noise draws on the 75,000-iteration snapshot), against G1's controls (the four default
    networks of Fig. 1, R^2 in the three with 75,000-iteration weights). The 0.01 value sets the upper end
    of the one-value tolerance range in the text.

    Returns:
        list of two `cell` dicts (labels "0.01" and "0.03"), in that order.
    """
    with open(G1G2_JSON) as fh:
        g = json.load(fh)
    ctrl = g["control"]
    out = []
    for r in g["g1"]:
        lab = r["name"].split("= ")[-1]
        if lab not in ("0.01", "0.03"):
            continue
        out.append(cell("frm", "3BFF", lab, r["active"], r["r2"], ctrl["active"], ctrl["r2"], r["n_runs"],
                        f"round18 g1g2.json {r['source']}", note="R^2: G1, 36 draws on the 75,000-iteration snapshot"))
    assert [c["label"] for c in out] == ["0.01", "0.03"], out
    return out


def group_by_name(path):
    """The groups of a read_uniform_cells*.json file by name.

    Args:
        path: str, the JSON file.
    Returns:
        (dict name -> group, the file's r2_tol).
    """
    with open(path) as fh:
        d = json.load(fh)
    return {g["name"]: g for g in d["groups"]}, d["r2_tol"]


def ruc_cell(tech, label, group, arm_label, source):
    """One CDDM cell from an arm of a read_uniform_cells*.json group.

    Args:
        tech: technique key; label: the value label; group: the group dict; arm_label: str, the
            start of the arm's label in the file; source: str, the file's name.
    Returns:
        a `cell` dict.
    """
    # the shortest label that starts with `arm_label`: "rate 0.1" is not "rate 0.1 + sparsity 0.05"
    arm = min((a for a in group["arms"] if a["label"].startswith(arm_label)), key=lambda a: len(a["label"]))
    ctrl = group["control"]
    v = arm["verdict"]
    return cell(tech, "CDDM", label, [r["active"] for r in arm["runs"]], [r["r2"] for r in arm["runs"]],
                [r["active"] for r in ctrl["runs"]], [r["r2"] for r in ctrl["runs"]],
                v.get("n_expected", len(arm["runs"])), f"{source} '{arm['label']}'",
                n_running=len(arm.get("pending", [])), stored=v["text"])


def load_cddm():
    """The CDDM cells at 75,000 iterations from the two uniform read-outs and the Figure 3 cache.

    Returns:
        (cells, r2_tol): the list of `cell` dicts and the tolerance the read-out files hold.
    """
    g1, tol1 = group_by_name(RUC1)
    g2, tol2 = group_by_name(RUC2)
    assert tol1 == tol2, f"the two read-out files hold different tolerances ({tol1}, {tol2})"
    out = [ruc_cell("duplicate", "", g1["CDDM per-unit rules"], "prune-and-duplicate", "read_uniform_cells.json"),
           ruc_cell("frm", "0.1", g2["CDDM rate penalty 0.1"], "rate 0.1", "read_uniform_cells_2.json")]
    with open(F2_CACHE, "rb") as fh:
        f2 = pickle.load(fh)
    lad = next(l for l in f2["ladders"] if l["spec"]["stem"] == "slide_k_frm_rws")
    rung = {i: ([r["fixed"]["active"] for r in rg["runs"]], [r["fixed"]["r2"] for r in rg["runs"]])
            for i, rg in enumerate(lad["rungs"])}
    for i, (tech, label) in ((1, ("frm", "0.2")), (3, ("frm+rws", "0.2"))):
        out.append(cell(tech, "CDDM", label, rung[i][0], rung[i][1], rung[0][0], rung[0][1], len(rung[i][0]),
                        f"fig_paper_F2_fails_cache.pkl slide_k_frm_rws rung {i} against rung 0 "
                        f"({lad['rungs'][0]['path'].split('trained_RNNs/')[-1]})"))
    out += [ruc_cell("dropout", "0.2", g1["CDDM per-unit rules"], "read-out dropout", "read_uniform_cells.json"),
            ruc_cell("frm+rws", "0.1", g2["CDDM rate penalty 0.1"], "rate 0.1 + sparsity", "read_uniform_cells_2.json"),
            ruc_cell("rescale", "10\u00d7", g2["CDDM rescaling"], "rescaling", "read_uniform_cells_2.json"),
            ruc_cell("synnoise", "1.0", g1["CDDM noisy synapses"], "sigma_w 1.0", "read_uniform_cells.json")]
    out += load_cddm_parity()
    return out, tol1


def load_cddm_parity():
    """The round 18 parity values on CDDM at 75,000 iterations, against the CDDM_paper_grid control of that read.

    Reads data/round18_training/round18_cddm.json (round18_training_read.py cddm; entry 2026-11-08_0859).
    Returns:
        list of `cell` dicts, one per PARITY entry.
    """
    with open(R18_CDDM) as fh:
        d = json.load(fh)
    ctrl = d["control"]
    out = []
    for tech, label, _, key in PARITY:
        p = d["parity"][key]
        stored = p["tol=0.01"]["text"]
        out.append(cell(tech, "CDDM", label, [r["active"] for r in p["runs"]], [r["r2"] for r in p["runs"]],
                        [r["active"] for r in ctrl], [r["r2"] for r in ctrl], len(p["runs"]),
                        f"round18_cddm.json parity '{key}' ({p['cell']})", stored=stored, note="parity value"))
    return out


def load_maze():
    """The maze cells at update 2,000: original arms, the freeze_after_end=true reruns and the parity values.

    The controls are all three uniform-series networks (issue #5, dead-relu-paper#194): the reference_techniques
    of read_maze_arms.json (426, 449, 415 active; mean success 0.898). Per MAZE_ARMS source: "arms" reads
    read_maze_arms.json (the original runs, read at update 2,000); "fz" reads data/round19/fz_reruns.json (the
    rate penalty, rate + sparsity and prune-and-duplicate rerun with trainer.freeze_after_end=true; the record of
    each seed cut at update 2,000); "parity" reads round18_maze.json's parity block.

    Returns:
        (cells, success_tol, reference): the `cell` dicts in MAZE_ARMS order, the tolerance read_maze_arms.json
        holds and its reference_techniques dict.
    """
    with open(MAZE_JSON) as fh:
        d = json.load(fh)
    with open(FZ_JSON) as fh:
        fz = {a["key"]: a for a in json.load(fh)["arms"]}
    with open(R18_MAZE) as fh:
        par = json.load(fh)["parity"]
    ref = d["reference_techniques"]
    arms = {a["key"]: a for a in d["arms"]}
    ctrl = [r for r in d["runs"] if r["key"] == "control"]
    ctrl_active = [r["active_end"] for r in ctrl]
    ctrl_succ = [r["success_end"] for r in ctrl]
    assert float(np.mean(ctrl_active)) == ref["active"], "the controls do not give the file's reference"
    learned_of = {(r["key"], r["seed"]): r["learned"] is not None for r in d["runs"]}
    out = []
    for tech, values in MAZE_ARMS.items():
        for label, key, src in values:
            if src == "arms":
                a = arms[key]
                out.append(cell(tech, "maze", label, a["active_end"], a["success_end"], ctrl_active, ctrl_succ,
                                len(a["active_end"]) + a["n_running"] + a["n_stopped"],
                                f"read_maze_arms.json '{key}'",
                                learned=[learned_of[(key, s)] for s in a["seeds_finished"]],
                                n_stopped=a["n_stopped"], n_running=a["n_running"], stored=a["verdict"],
                                note=MAZE_RESCALE_NOTE.get(key, "")))
            elif src == "fz":
                seeds = fz[key]["seeds"]
                fin = [x["at2000"] for x in seeds if x["at2000"]["reached"]]
                out.append(cell(tech, "maze", label, [x["active"] for x in fin], [x["success"] for x in fin],
                                ctrl_active, ctrl_succ, len(seeds), f"fz_reruns.json '{key}' at update 2,000",
                                learned=[x["learned"] for x in fin], n_stopped=len(seeds) - len(fin),
                                stored=fz[key].get("verdict_2000"), note="freeze_after_end=true rerun"))
            else:
                runs = par[key]["runs"]
                fin = [r for r in runs if r["status"] == "finished"]
                out.append(cell(tech, "maze", label, [r["active_end"] for r in fin], [r["success_end"] for r in fin],
                                ctrl_active, ctrl_succ, len(runs), f"round18_maze.json parity '{key}'",
                                learned=[r["learned"] is not None for r in fin], n_stopped=len(runs) - len(fin),
                                stored="works" if par[key]["passes"] else "fails: " + "; ".join(par[key]["why"]),
                                note="parity value"))
    return out, d["success_tol"], ref


def value_key(c):
    """Sort key placing a technique's values in increasing order inside its slots (LABEL_ORDER for words)."""
    lab = c["label"]
    if lab in LABEL_ORDER:
        return LABEL_ORDER[lab]
    try:
        return float(lab)
    except ValueError:
        return 99.0


def tolerance_sensitivity(cells, tol, r2_tols=(0.005, 0.01, 0.02), success_tols=(0.025, 0.05, 0.10)):
    """Print each technique's verdict per task at other tolerances, and the tolerance at which it flips.

    The 0.01 tolerance in R^2 was written after the 3BFF networks and most CDDM networks existed
    (entry 2026-11-06_2259), so the paper reports how the verdicts depend on it. A technique works
    on a task if it works at one finished value, as in `task_verdict`.

    Args:
        cells: the `cell` dicts; tol: dict task -> tolerance used in the figure; r2_tols: tolerances
            tried on 3BFF and CDDM; success_tols: tolerances tried on the maze.
    Returns:
        dict (technique, task) -> minimum tolerance at which the technique works on that task, or
        None if no tolerance would make it work (the count or the maze's 0.9 condition fails at
        every value).
    """
    out = {}
    print("\n--- tolerance sensitivity: technique x task, works (W) or fails (-), full-precision differences ---")
    for task in TASKS:
        tols = success_tols if task == "maze" else r2_tols
        print(f"  {task}: tolerances {tols}")
        for tech in TECH:
            mine = [c for c in cells if c["tech"] == tech and c["task"] == task and c["status"] == "finished"]
            floor = MAZE_LEARNED_SUCCESS if task == "maze" else None
            marks = [any(rule(c["active"], c["perf"], c["ctrl_active"], c["ctrl_perf"], t, floor)["works"] for c in mine)
                     for t in tols]
            need = []
            for c in mine:
                r = rule(c["active"], c["perf"], c["ctrl_active"], c["ctrl_perf"], 1e9, floor)
                if r["works"]:
                    need.append(max(0.0, -r["dperf"]))
            out[(tech, task)] = min(need) if need else None
            diffs = ", ".join(f"{c['label'] or '-'}: {rule(c['active'], c['perf'], c['ctrl_active'], c['ctrl_perf'], 1e9)['dperf']:+.5f}"
                              for c in mine)
            print(f"    {TECH_NAME[tech]:>20} {' '.join('W' if m else '-' for m in marks)}   works from tolerance "
                  f"{'never' if out[(tech, task)] is None else format(out[(tech, task)], '.5f')}   ({diffs})")
    r2_values = sorted({v for (t, k), v in out.items() if k != "maze" and v is not None})
    print("  techniques that work on all three tasks, as the R^2 tolerance varies (maze at the figure's):")
    for t in sorted(set(r2_tols) | set(r2_values)):
        both = []
        for tech in TECH:
            ok = all(out[(tech, k)] is not None and out[(tech, k)] <= t for k in ("3BFF", "CDDM"))
            m = out[(tech, "maze")]
            if ok and m is not None and m <= tol["maze"]:
                both.append(TECH_NAME[tech])
        print(f"    {t:.5f}: {both}")
    return out


def r2_need(c, maze_tol):
    """The smallest R^2 tolerance at which a cell meets the criterion (0 for the maze if it passes at maze_tol).

    Args:
        c: a finished `cell` dict; maze_tol: the fixed success tolerance on the maze.
    Returns:
        float or None: max(0, -mean R^2 difference) when the count condition holds (3BFF, CDDM); 0.0 for a maze
        cell that meets the whole criterion at maze_tol; None when no tolerance makes the cell pass.
    """
    if c["task"] == "maze":
        return 0.0 if rule(c["active"], c["perf"], c["ctrl_active"], c["ctrl_perf"], maze_tol,
                           MAZE_LEARNED_SUCCESS)["works"] else None
    r = rule(c["active"], c["perf"], c["ctrl_active"], c["ctrl_perf"], 1e9)
    return max(0.0, -r["dperf"]) if r["works"] else None


def matched_labels(cells, tech):
    """The value labels of a technique run on all three tasks (the matched-values rule)."""
    per = [{c["label"] for c in cells if c["tech"] == tech and c["task"] == k and c["status"] == "finished"}
           for k in TASKS]
    return sorted(per[0] & per[1] & per[2], key=lambda lab: value_key(dict(label=lab)))


def rule_tables(cells, extra_3bff, maze_tol, r2_tols=(0.005, 0.01, 0.02)):
    """The verdicts under the one-value and the matched-values rules, and the tolerance ranges.

    One-value rule: a technique meets the criterion on a task if one of its finished values does; on 3BFF every
    G2 value counts (`extra_3bff`, as round 18's G2 (a) counts them). Matched-values rule: only the values run
    on all three tasks (`matched_labels`); a value counts on all three tasks only if it passes on each.

    Args:
        cells: the drawn `cell` dicts; extra_3bff: load_3bff_all_values(); maze_tol: the success tolerance.
    Returns:
        dict with "one" and "matched": per technique the tolerance from which it passes on all three tasks
        (None: never), and the per-task needs.
    """
    fin = [c for c in cells if c["status"] == "finished"]
    pool = fin + [c for c in extra_3bff if c["status"] == "finished"]
    out = dict(one={}, matched={})
    print("\n--- the two rules: the smallest R^2 tolerance at which each technique passes on all three tasks "
          f"(Maze at its fixed success tolerance {maze_tol}) ---")
    for tech in TECH:
        need = {}
        for k in TASKS:
            src = pool if k == "3BFF" else fin
            ns = [(r2_need(c, maze_tol), c["label"]) for c in src if c["tech"] == tech and c["task"] == k]
            ns = [n for n in ns if n[0] is not None]
            need[k] = min(ns) if ns else None
        one = None if any(v is None for v in need.values()) else max(v[0] for v in need.values())
        out["one"][tech] = dict(need={k: v for k, v in need.items()}, from_tol=one)
        per_value = {}
        for lab in matched_labels(cells, tech):
            ns = [r2_need(next(c for c in fin if c["tech"] == tech and c["task"] == k and c["label"] == lab),
                          maze_tol) for k in TASKS]
            per_value[lab] = dict(need=ns, from_tol=None if None in ns else max(ns))
        cand = [v["from_tol"] for v in per_value.values() if v["from_tol"] is not None]
        out["matched"][tech] = dict(values=per_value, from_tol=min(cand) if cand else None)
        fmt_n = lambda v: "never" if v is None else f"{v:.4f}"
        print(f"  {TECH_NAME[tech]:>20}  one-value: " + ", ".join(
            f"{k} {fmt_n(need[k][0] if need[k] else None)}" + (f" ({need[k][1]})" if need[k] else "") for k in TASKS)
              + f" -> all three from {fmt_n(one)}")
        for lab, v in per_value.items():
            print(f"  {'':>20}  matched value {lab or '(one value)'}: "
                  + ", ".join(f"{k} {fmt_n(n)}" for k, n in zip(TASKS, v["need"])) + f" -> all three from {fmt_n(v['from_tol'])}")
    for name in ("one", "matched"):
        print(f"  {name}-value rule: techniques on all three tasks at R^2 tolerance " + "; ".join(
            f"{t}: " + (", ".join(TECH_NAME[x] for x in TECH if out[name][x]["from_tol"] is not None
                                  and out[name][x]["from_tol"] <= t) or "none") for t in r2_tols))
        dup = out[name]["duplicate"]["from_tol"]
        others = sorted((v["from_tol"], TECH_NAME[x]) for x, v in out[name].items()
                        if x != "duplicate" and v["from_tol"] is not None)
        print(f"  {name}-value rule: prune-and-duplicate alone passes on all three tasks from {dup:.4f}"
              + (f" up to {others[0][0]:.4f} (exclusive), where {others[0][1]} joins it" if others else ", with no upper limit"))
    return out


def verdict_word(r):
    """Return "works" or "fails" for a `rule` dict r."""
    return "works" if r["works"] else "fails"


def learned_control_sensitivity(cells, tol):
    """Print the Maze verdicts under the two learned controls beside those under all three.

    The first draft of the analysis used the two learned controls (entry 2026-11-05_0812); the
    paper reports them once as a sensitivity check (dead-relu-paper#194). Printed for each cell:
    the criterion under both references, and under each without the 0.9 condition, because that
    condition alone decides two cells under all three controls.

    Args:
        cells: the `cell` dicts; tol: dict task -> tolerance.
    Returns:
        dict with mean_all, mean_learned, active_all, active_learned (the two references) and
        changed (list of (technique, value) whose full verdict differs between them).
    """
    with open(MAZE_JSON) as fh:
        d = json.load(fh)
    ctrl = [r for r in d["runs"] if r["key"] == "control"]
    a_all, s_all = [r["active_end"] for r in ctrl], [r["success_end"] for r in ctrl]
    a_l = [r["active_end"] for r in ctrl if r["learned"] is not None]
    s_l = [r["success_end"] for r in ctrl if r["learned"] is not None]
    out = dict(mean_all=float(np.mean(s_all)), mean_learned=float(np.mean(s_l)), active_all=float(np.mean(a_all)),
               active_learned=float(np.mean(a_l)), changed=[])
    print("\n--- Maze verdicts: all three controls (drawn) against the two learned controls ---")
    print(f"  control success {[round(x, 3) for x in s_all]}, active {a_all}; all three: {out['mean_all']:.3f}, "
          f"{out['active_all']:.1f}; two learned: {out['mean_learned']:.3f}, {out['active_learned']:.1f}")
    for c in cells:
        if c["task"] != "maze" or not c["active"]:
            continue
        b3 = rule(c["active"], c["perf"], a_all, s_all, tol["maze"], MAZE_LEARNED_SUCCESS)
        b2 = rule(c["active"], c["perf"], a_l, s_l, tol["maze"], MAZE_LEARNED_SUCCESS)
        m3 = rule(c["active"], c["perf"], a_all, s_all, tol["maze"])
        m2 = rule(c["active"], c["perf"], a_l, s_l, tol["maze"])
        if b3["works"] != b2["works"]:
            out["changed"].append((c["tech"], c["label"]))
        print(f"  {TECH_NAME[c['tech']]:>20} {c['label']:>9}: all three {b3['dperf']:+.3f} fold {b3['fold']:.2f} {verdict_word(b3)} "
              f"(without the 0.9 condition {verdict_word(m3)}); two learned {b2['dperf']:+.3f} fold {b2['fold']:.2f} "
              f"{verdict_word(b2)} (without it {verdict_word(m2)})")
    print(f"  verdicts that differ between the two references: {out['changed']}")
    return out


def verdicts(cells, tol):
    """The one rule applied to every cell, checked against the stored verdict where there is one.

    Args:
        cells: list of `cell` dicts; tol: dict task -> tolerance.
    Returns:
        list of dicts per cell: tech, task, label, status, rule (the `rule` dict, or None for a
        cell with no finished run), stored, agrees (None where no stored verdict or no verdict).
    """
    out = []
    for c in cells:
        floor = MAZE_LEARNED_SUCCESS if c["task"] == "maze" else None
        r = rule(c["active"], c["perf"], c["ctrl_active"], c["ctrl_perf"], tol[c["task"]], floor) if c["active"] else None
        agrees = None
        if r is not None and c["stored"] is not None and c["status"] == "finished":
            agrees = c["stored"].startswith("works") == r["works"]
        out.append(dict(tech=c["tech"], task=c["task"], label=c["label"], status=c["status"], rule=r,
                        stored=c["stored"], agrees=agrees))
    return out


def task_verdict(vs, tech, task):
    """Whether a technique works on a task: it works at at least one finished value.

    Args:
        vs: `verdicts` output; tech, task: keys.
    Returns:
        (works, count_ok, text): works and count_ok are bools over the task's finished cells
        (count_ok: the count condition passes at some finished value); text names the values.
    """
    mine = [v for v in vs if v["tech"] == tech and v["task"] == task]
    fin = [v for v in mine if v["status"] == "finished"]
    works = [v for v in fin if v["rule"]["works"]]
    counts = [v for v in fin if v["rule"]["count_ok"]]
    if not fin:
        return False, False, "no finished cell (" + ", ".join(v["status"] for v in mine) + ")"
    parts = []
    for v in mine:
        tag = f" at {v['label']}" if v["label"] else ""
        if v["status"] != "finished":
            parts.append(f"{v['status']}{tag}")
        else:
            parts.append(("works" if v["rule"]["works"] else v["rule"]["text"]) + tag)
    return bool(works), bool(counts), "; ".join(parts)


def column_order(vs):
    """The techniques ordered by the number of tasks on which they work.

    Ties: the number of tasks on which the count condition passes, then the name.

    Args:
        vs: `verdicts` output.
    Returns:
        (order, table): order is the list of technique keys; table a dict (tech, task) ->
        (works, count_ok, text).
    """
    table = {(t, k): task_verdict(vs, t, k) for t in TECH for k in TASKS}
    keyed = sorted(TECH, key=lambda t: (-sum(table[(t, k)][0] for k in TASKS),
                                        -sum(table[(t, k)][1] for k in TASKS), TECH_NAME[t]))
    return keyed, table


def slot_layout(order, cells):
    """Where every cell's group sits along the shared x axis of panels a and b.

    Args:
        order: list of technique keys in column order; cells: every `cell` dict.
    Returns:
        (slots, columns, width): slots is a list of (cell, x) in drawing order; columns a list of
        (tech, x_left, x_right); width the axes width in mm.
    """
    slots, columns, x = [], [], 0.0
    for tech in order:
        x0 = x
        for task in TASKS:
            for c in cells:
                if c["tech"] == tech and c["task"] == task:
                    slots.append((c, x + PITCH_MM / 2))
                    x += PITCH_MM
        columns.append((tech, x0, x))
        x += GAP_MM
    return slots, columns, x - GAP_MM


def draw_group(ax, x, ys, col, marker, label=None, ms=ps.MARKER_PT, label_above=False, label_below=False,
               guide=None):
    """One group of open markers the paper's way: a marker at the mean, a bar over the range.

    Open markers of one group would overlap at any spacing that fits the slots of these figures,
    and an open marker drawn over another's ink rim no longer reads as open (dead-relu-paper#104,
    tests/scan_open_markers.py), so a group is one open marker at its mean with a thin bar from
    its smallest to its largest value and, when `label` is given, that text beside it.

    Args:
        ax: axes whose x is in mm; x: the group's x; ys: the values (array-like); col: the
            colour; marker: the shape; label: str or None, written 1 mm right of the marker;
            ms: marker size in pt (triangles are drawn TRI_SCALE larger); label_above: bool, write
            the label above the bar instead, for a slot whose crosses sit beside the marker
            (review of PR #193, item 11: check_overlaps tests text against text only); label_below:
            bool, write it below the bar instead (third review, item 6: above the slow-tilt bar in
            panel b it lay on the tolerance line); guide: float or None, the y of a dashed line that a
            label written beside the marker would sit on; the label is then written just below that
            line when the mean is above it, and just above it otherwise (round 20: the "4" and "5" of
            panel b's held-out R^2 row lay on the tolerance line).
    Returns:
        None (nothing drawn for an empty group).
    """
    ys = np.asarray(ys, float)
    if not len(ys):
        return
    if np.ptp(ys) > 0:
        ax.plot([x, x], [ys.min(), ys.max()], color=col, lw=ps.THIN_LW, zorder=4, solid_capstyle="butt",
                clip_on=False)
    size = ms * (F5.TRI_SCALE if marker in TRIANGLES else 1.0)
    ax.plot([x], [ys.mean()], marker, ms=size, ls="none", zorder=5, clip_on=False, **ps.mark_fixed(col))
    if label and label_above:
        ax.annotate(label, xy=(x, ys.max()), xytext=(0, 2.5), textcoords="offset points", fontsize=ps.TICK_PT,
                    color=ps.INK, ha="center", va="bottom", zorder=6, annotation_clip=False)
    elif label and label_below:
        ax.annotate(label, xy=(x, ys.min()), xytext=(0, -2.5), textcoords="offset points", fontsize=ps.TICK_PT,
                    color=ps.INK, ha="center", va="top", zorder=6, annotation_clip=False)
    elif label and guide is not None:
        above = ys.mean() < guide
        ax.annotate(label, xy=(x + 1.0, guide), xytext=(0, 1.0 if above else -1.0), textcoords="offset points",
                    fontsize=ps.TICK_PT, color=ps.INK, ha="left", va="bottom" if above else "top", zorder=6,
                    annotation_clip=False)
    elif label:
        ax.text(x + 1.0, ys.mean(), label, fontsize=ps.TICK_PT, color=ps.INK, ha="left", va="center", zorder=6)


def draw_marks(ax, x, c, key, zero=None, guide=None):
    """One cell's marks in one axes: every finished seed as one group, the not-learned ones also as crosses.

    The group marker is the mean over every finished seed, the mean the criterion
    judges (`rule`; dead-relu-paper#190, Reviewer R item 4), so a marker inside the tolerance band
    passes the performance condition and one outside it fails. A seed that did not learn the task
    is inside that mean and is also drawn as a cross at its value, beside the marker; a seed the
    guard stopped is a cross at `zero`.

    Args:
        ax: axes (x in mm); x: the slot's centre; c: a `cell` dict; key: "fold", "dperf" or "perf",
            the values drawn; zero: the y of a stopped run's cross, or None to draw none; guide: the y
            of a dashed line in this axes, or None; a number written beside a marker that would sit on
            it is moved off it.
    Returns:
        None.
    """
    col, marker = TASK_COL[c["task"]], TECH[c["tech"]][1]
    vals = np.asarray(c[key], float)
    ok = np.asarray(c["learned"][:len(vals)], bool)
    n_group = len(vals)
    # the number of networks behind the marker when it is not 3, "1" included (PR #188 review, 9a)
    label = str(n_group) if n_group >= 1 and n_group != N_SEEDS else None
    crosses = list(vals[~ok]) + ([zero] * c["n_stopped"] if zero is not None else [])
    # below the bar only while the label stays inside the axes: below the axes it meets the value labels
    # (round 19: the Maze parity rescaling, whose bar reaches -0.89); otherwise beside the marker
    lo, hi = ax.get_ylim()
    room_below = len(vals) > 0 and vals.min() - lo > 0.12 * (hi - lo)
    near = label and guide is not None and len(vals) and abs(vals.mean() - guide) < 0.09 * (hi - lo)
    draw_group(ax, x, vals, col, marker, label, label_above=bool(crosses) and key == "fold",
               label_below=bool(crosses) and key != "fold" and room_below, guide=guide if near else None)
    for xi, v in zip(cross_positions(x, len(crosses), n_group > 0), crosses):
        ax.plot([xi], [v], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)


def cross_positions(x, n, beside):
    """Where a cell's crosses go along x: spread about the slot's centre, or beside its marker.

    Args:
        x: the slot's centre (mm); n: how many crosses; beside: bool, True when the slot also
            holds a group marker, which the crosses must clear (CROSS_DX_MM to either side).
    Returns:
        array of n x positions.
    """
    if not beside:
        return x + (np.arange(n) - (n - 1) / 2) * SEED_DX_MM
    return x + np.array([CROSS_DX_MM * (k // 2 + 1) * (1 if k % 2 == 0 else -1) for k in range(n)])


def panel_ab(fig, slots, columns, width, tol):
    """Panels a and b: the fold change, the held-out R^2 difference and the Maze success rate over the shared slots.

    Round 20 (dead-relu-paper #212, Reviewer W, figures item 1): panel b's Maze row draws each value's
    success rate of the noise-free probe itself, with the learned level 0.9 dashed, in place of the
    success rate minus the control mean against the tolerance 0.05. The 0.9 condition decides every Maze
    verdict the success rate decides (Methods: Training techniques), and every network that ends below
    0.9 has no learned point, so it is a cross: a value with a cross under the dashed line fails.

    Args:
        fig: the figure; slots, columns, width: slot_layout output; tol: dict task -> tolerance.
    Returns:
        (ax_a, ax_b1, ax_b2).
    """
    ax_a = ax_mm(fig, LEFT_MM, A_TOP_MM, width, A_H_MM)
    ax_b1 = ax_mm(fig, LEFT_MM, B1_TOP_MM, width, B1_H_MM)
    ax_b2 = ax_mm(fig, LEFT_MM, B2_TOP_MM, width, B2_H_MM)
    for ax, lim in ((ax_a, FOLD_LIM), (ax_b1, R2_LIM), (ax_b2, SUCC_LIM)):
        ax.set_ylim(*lim)                 # before drawing: draw_marks places labels by the limits
    for c, x in slots:
        maze = c["task"] == "maze"
        band_ax = ax_b2 if maze else ax_b1
        for ax, key in ((ax_a, "band_fold"), (band_ax, "band_perf" if maze else "band_dperf")):
            lo, hi = c[key]
            ax.fill_between([x - PITCH_MM / 2, x + PITCH_MM / 2], [lo] * 2, [hi] * 2, color=ps.FAINT,
                            alpha=0.3, lw=0, zorder=1)
        draw_marks(ax_a, x, c, "fold", zero=0.0)
        if maze:
            draw_marks(ax_b2, x, c, "perf", guide=MAZE_LEARNED_SUCCESS)
        else:
            draw_marks(ax_b1, x, c, "dperf", guide=-tol[c["task"]])
        if c["status"] == "unfinished":
            ax_a.text(x, 0.5, "unfinished", rotation=90, transform=ax_a.get_xaxis_transform(),
                      fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="center")
        elif c["status"] == "stopped":
            ax_a.text(x, 0.55, "stopped", rotation=90, transform=ax_a.get_xaxis_transform(),
                      fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="center")
        # a partly finished cell carries no "k/n": the number beside its marker gives the finished
        # networks, as in every other cell (dead-relu-paper#190, Reviewer S, figures item 3)
        if c["label"]:
            ax_b2.text(x, -0.06, LABEL_SHOWN.get(c["label"], c["label"]), rotation=90,
                       transform=ax_b2.get_xaxis_transform(), fontsize=ps.TICK_PT, color=ps.INK, ha="center",
                       va="top")
    for tech, x0, x1 in columns:
        ax_a.text((x0 + x1) / 2, 1.04, TECH[tech][0], transform=ax_a.get_xaxis_transform(),
                  fontsize=ps.LABEL_PT, color=ps.INK, ha="center", va="bottom", linespacing=1.15)
    ax_a.axhline(1.0, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_a.set_ylim(*FOLD_LIM)
    ax_a.set_yticks([0, 1, 2, 3, 4])
    ax_a.set_ylabel("fold change")      # one line: two lines overrun this 16 mm axis and meet the letter b (the caption names the quantity)
    ax_b1.axhline(0.0, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_b1.axhline(-tol["3BFF"], lw=ps.THIN_LW, ls=(0, (4, 3)), color=ps.INK, zorder=2)
    ax_b1.text(width - 0.3, -tol["3BFF"] - 0.0015, f"tolerance {tol['3BFF']}", fontsize=ps.TICK_PT,
               color=ps.INK, ha="right", va="top")
    ax_b1.set_ylim(*R2_LIM)
    ax_b1.set_yticks([-0.03, -0.02, -0.01, 0])
    ax_b1.set_yticklabels(["−0.03", "−0.02", "−0.01", "0"])
    ax_b1.set_ylabel("held-out $R^2$\nminus control")
    # the learned level, named by its own tick label (no in-panel text to overprint)
    ax_b2.axhline(MAZE_LEARNED_SUCCESS, lw=ps.THIN_LW, ls=(0, (4, 3)), color=ps.INK, zorder=2)
    ax_b2.set_ylim(*SUCC_LIM)
    ax_b2.set_yticks([0, 0.5, MAZE_LEARNED_SUCCESS])
    ax_b2.set_yticklabels(["0", "0.5", f"{MAZE_LEARNED_SUCCESS:g}"])
    ax_b2.set_ylabel("success rate")
    for ax in (ax_a, ax_b1, ax_b2):
        ax.set_xlim(0, width)
        ax.set_xticks([])
        ax.tick_params(axis="x", length=0, labelbottom=False)
        ps.ygrid(ax)
    ps.panel_letter(ax_a, "a")
    ps.panel_letter(ax_b1, "b")
    return ax_a, ax_b1, ax_b2


def panel_key(fig):
    """The key row: the technique shapes (ink), the task colours and the open-marker meaning.

    Args:
        fig: the figure.
    Returns:
        the key axes.
    """
    ax = ax_mm(fig, 0.0, KEY_TOP_MM, ps.W_FULL_MM, 8.0)
    ps.blank(ax)
    h1 = [Line2D([], [], marker=TECH[t][1], ls="none", ms=3.2 * (F5.TRI_SCALE if TECH[t][1] in TRIANGLES else 1.0),
                 label=TECH_NAME[t], **ps.mark_fixed(ps.INK)) for t in TECH]
    leg1 = ax.legend(handles=h1, loc="upper center", ncol=len(h1), frameon=False, fontsize=ps.TICK_PT,
                     handlelength=1.0, columnspacing=1.2, handletextpad=0.35, borderaxespad=0)
    ax.add_artist(leg1)
    h2 = [Line2D([], [], marker="o", ls="none", ms=3.2, label=TASK_NAME[t], **ps.mark_fixed(TASK_COL[t])) for t in TASKS]
    h2.append(Line2D([], [], ls="none", label="open: 75,000 iterations (3BFF, CDDM), update 2,000 (Maze); "
                                              "filled: floor crossing"))
    ax.legend(handles=h2, loc="lower center", ncol=len(h2), frameon=False, fontsize=ps.TICK_PT,
              handlelength=1.0, columnspacing=1.2, handletextpad=0.35, borderaxespad=0)
    return ax


def cddm_size_rows():
    """Panel c's CDDM sub-column: per size and technique, the networks' fold changes and R^2 changes.

    Reads data/round18/g6_reads.json (G6): every CDDM_paper_grid network at N = 500 to 4,000 at its
    floor crossing (1.05x; the record's "active" and "r2", R^2 estimated from the final weights). A
    network without a crossing (non-null "reason") is left out, as G6 leaves it out.

    Returns:
        list of dicts: tech (CDDM_SIZE_ARMS key), N, fold (array, count over the same-size control
        mean), dr2 (array, R^2 minus the same-size control mean), n_runs (networks in the cell).
    """
    with open(G6_READS) as fh:
        reads = json.load(fh)
    rows = []
    for N in F5.SIZES:
        ctrl = [r for r in reads[f"{N}|control"] if r["reason"] is None]
        c_act = np.mean([r["active"] for r in ctrl])
        c_r2 = np.mean([r["r2"] for r in ctrl])
        for tech, arm in CDDM_SIZE_ARMS:
            runs = reads[f"{N}|{arm}"]
            ok = [r for r in runs if r["reason"] is None]
            rows.append(dict(tech=tech, N=N, fold=np.array([r["active"] / c_act for r in ok]),
                             dr2=np.array([r["r2"] - c_r2 for r in ok]), n_runs=len(runs)))
    return rows


def panel_sizes(fig):
    """Panel c: the techniques across sizes, 3BFF (left) and CDDM (right) sub-columns sharing y.

    3BFF reuses fig_paper_F5_drive's loaders and read-outs (floor crossing; rescaling and read-out
    dropout at 75,000 iterations, dashed). CDDM draws prune-and-duplicate per network and read-out
    dropout, both at the floor crossing (cddm_size_rows). Round 20 (dead-relu-paper #212, Reviewer S,
    figures items 5 and 6): one code for the whole figure, colour the task and marker shape the
    technique (TECH, the shapes of panels a, b and d), in place of one colour per technique. Each
    technique is a line through its means (top) and medians (bottom) with its marker at every size,
    filled at the floor crossing and open at 75,000 iterations; prune-and-duplicate also draws one
    small filled marker per network and the thicker line. A technique's markers at one size are
    spread along x by C_DODGE so that markers of two techniques at one size do not cover each other.

    Args:
        fig: the figure.
    Returns:
        (axes, rows): rows is a list of (task, technique, N, n networks, mean fold, median dR2).
    """
    series = F5.size_series(F5.load_catalogue())
    marks = F5.load_marks()
    h_top = ROW2_H_MM * 0.55
    axes = {}
    for j, task in enumerate(("3BFF", "CDDM")):
        left = C_LEFT_MM + j * (C_W_MM + C_GAP_MM)
        axes[task] = (ax_mm(fig, left, ROW2_TOP_MM, C_W_MM, h_top),
                      ax_mm(fig, left, ROW2_TOP_MM + h_top + 1.5, C_W_MM, ROW2_H_MM - h_top - 1.5))
    rows = []
    order = list(F5.MAIN)

    def draw_family(ax_m, ax_r, fam, xs, folds, dr2s, at_75k, col):
        """Draw one technique's line through its means (top) and medians (bottom), with its marker at each
        size, and one small filled marker per network for prune-and-duplicate.

        Args:
            ax_m, ax_r: the fold-change and R^2 axes; fam: technique key; xs: sizes; folds, dr2s: per-size
                arrays; at_75k: bool, read at 75,000 iterations (dashed line, open markers); col: task colour.
        Returns:
            (means, medians): lists per size.
        """
        mm = [float(f.mean()) for f in folds]
        rr = [float(np.median(d)) for d in dr2s]
        xd = [n * C_DODGE ** (order.index(fam) - (len(order) - 1) / 2) for n in xs]
        marker = TECH[fam][1]
        dup = fam == "duplicate"
        if dup:
            for n, f, d in zip(xd, folds, dr2s):
                for ax, v in ((ax_m, f), (ax_r, d)):
                    ax.scatter(np.full(len(v), n), v, s=5, marker=marker, zorder=4,
                               **ps.mark_floor(col, scatter=True))
        ls = "--" if at_75k else "-"
        lw = ps.DATA_LW if dup else ps.THIN_LW
        size = ps.MARKER_PT * (F5.TRI_SCALE if marker in TRIANGLES else 1.0)
        for ax, ys in ((ax_m, mm), (ax_r, rr)):
            ax.plot(xd, ys, ls=ls, color=col, lw=lw, zorder=3 if dup else 2)
            if not dup:
                kw = ps.mark_fixed(col) if at_75k else ps.mark_floor(col)
                ax.plot(xd, ys, marker, ms=size, ls="none", zorder=5, clip_on=False, **kw)
        return mm, rr

    ax_m, ax_r = axes["3BFF"]
    for fam in F5.MAIN:
        ro = "75k" if fam in F5.AT_75K else "floor"
        xs, folds, dr2s = [], [], []
        for n in F5.SIZES:
            d = F5.per_run(series, fam, n, ro, marks)
            if not len(d["mult"]):
                continue
            xs.append(n)
            folds.append(np.asarray(d["mult"], float))
            dr2s.append(np.asarray(d["dr2"], float))
        mm, rr = draw_family(ax_m, ax_r, fam, xs, folds, dr2s, fam in F5.AT_75K, TASK_COL["3BFF"])
        rows += [("3BFF", fam, n, len(f), m, r) for n, f, m, r in zip(xs, folds, mm, rr)]
    ax_m, ax_r = axes["CDDM"]
    crows = cddm_size_rows()
    for fam, _ in CDDM_SIZE_ARMS:
        cr = [r for r in crows if r["tech"] == fam and len(r["fold"])]
        mm, rr = draw_family(ax_m, ax_r, fam, [r["N"] for r in cr], [r["fold"] for r in cr],
                             [r["dr2"] for r in cr], False, TASK_COL["CDDM"])
        rows += [("CDDM", fam, r["N"], f"{len(r['fold'])}/{r['n_runs']}", m, q) for r, m, q in zip(cr, mm, rr)]
    for j, task in enumerate(("3BFF", "CDDM")):
        ax_m, ax_r = axes[task]
        for ax in (ax_m, ax_r):
            ax.set_xscale("log")
            ax.set_xlim(380, 5300)
            ax.set_xticks(F5.SIZES)
            ax.set_xticklabels([ps.fmt_count(n) for n in F5.SIZES])
            ax.minorticks_off()
            ps.ygrid(ax)
        ax_m.axhline(1, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
        ax_r.axhline(0, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
        ax_m.set_ylim(0.6, 6.2)           # the 3BFF rate penalty ends at 5.6
        ax_m.set_yticks([1, 3, 5])
        ax_m.tick_params(labelbottom=False)
        ax_r.set_ylim(-0.048, 0.006)      # the 3BFF noisy-synapse median at N = 500 is -0.044
        ax_r.set_yticks([-0.04, -0.02, 0])
        ax_r.set_yticklabels(["−0.04", "−0.02", "0"])
        ax_m.set_title(task, fontsize=ps.LABEL_PT, pad=3, loc="left")
        if j:
            ax_m.tick_params(labelleft=False)
            ax_r.tick_params(labelleft=False)
    axes["3BFF"][0].set_ylabel("fold change")
    axes["3BFF"][1].set_ylabel("$R^2$ minus\ncontrol")
    fig.text((C_LEFT_MM + C_W_MM + C_GAP_MM / 2) / ps.W_FULL_MM, 1.0 - (ROW2_TOP_MM + ROW2_H_MM + 6.0) / FIG_H_MM,
             ps.LABELS["size"], ha="center", va="top", fontsize=ps.LABEL_PT, color=ps.INK)
    ps.panel_letter(axes["3BFF"][0], "c")
    return axes, rows


def removal_rows():
    """Panel d's networks: one removal on all three tasks (the k added units, against k random active units).

    Returns:
        list of dicts per network: task, tech (TECH key), name, x (median score after removing k
        random active units), y (score after removing the k added units, the inactive units left
        in place), intact, k, where (3BFF: the loss-burst mark of the R^2 weights; else "calm").
    """
    rows = []
    marks = F5.load_marks()
    for r in F5.load_a3()[N]:
        if r["family"] not in TECH:
            continue
        rows.append(dict(task="3BFF", tech=r["family"], name=os.path.basename(r["run_dir"].rstrip("/"))[:28],
                         x=float(np.median(r["r2_random_active"])), y=float(r["r2_recruited"]),
                         intact=float(r["r2_full"]), k=int(r["surplus"]),
                         where=marks["a3"].get(r["run_dir"], "calm")))
    with open(G3_JSON) as fh:
        g3 = json.load(fh)
    for block in g3.values():
        for arm, a in block["arms"].items():
            for i, r in enumerate(a["runs"], 1):
                rows.append(dict(task="CDDM", tech=G3_TECH[arm], name=f"{arm} network {i}",
                                 x=float(np.median(r["r2_random_active"])), y=float(r["r2_recruited"]),
                                 intact=float(r["r2_full"]), k=int(r["surplus"]), where="calm"))
    with open(FZ_REMOVAL_JSON) as fh:      # G4's procedure on the freeze_after_end=true reruns (round 19)
        g4 = json.load(fh)
    for r in g4["rows"]:
        tech = "duplicate" if r["cell"] == "duplicate" else "frm"
        rows.append(dict(task="maze", tech=tech, name=r["file"].replace(".json", ""), x=float(r["random_active"]),
                         y=float(r["added"]), intact=float(r["intact"]), k=int(r["k"]), where="calm"))
    return rows


def panel_removal(fig):
    """Panel d: performance after removing the added units against performance after a random removal.

    Three sub-axes sharing both axes, 3BFF and CDDM (held-out R^2) and the maze (success), one removal
    in all three (removal_rows): the k added units removed, the inactive units left in place, against
    k active units removed at random.

    Args:
        fig: the figure.
    Returns:
        (axes, rows): rows as removal_rows.
    """
    rows = removal_rows()
    axes = []
    for j, (task, title) in enumerate((("3BFF", "3BFF, $R^2$"), ("CDDM", "CDDM, $R^2$"), ("maze", "Maze, success"))):
        ax = ax_mm(fig, D_LEFT_MM + j * (D_W_MM + D_GAP_MM), ROW2_TOP_MM + ROW2_H_MM - D_H_MM, D_W_MM, D_H_MM)
        axes.append(ax)
        ax.set_xlim(D_LO - 0.012, 0.5)
        ax.set_ylim(D_LO - 0.015, 0.85)
        ax.plot([D_LO, 0.5], [D_LO, 0.5], color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
        groups = []
        for tech in TECH:
            pts = [(r["x"], r["y"]) for r in rows if r["task"] == task and r["tech"] == tech]
            groups += edge_or_group(ax, pts, TASK_COL[task], TECH[tech][1])
        for g in groups:
            if g["x"] < 0.0:      # a label on the left would sit on the y axis (PR #202 review, item 12)
                g["side"] = "below"
        F5.draw_groups(ax, F5.see_through(ax, groups))
        ax.set_xticks([0, 0.2, 0.4])
        ax.set_yticks([0, 0.4, 0.8])
        ax.set_title(title, fontsize=ps.LABEL_PT, pad=3, loc="left")
        ps.ygrid(ax)
        if j:
            ax.tick_params(labelleft=False)
    # round 20 (dead-relu-paper #212, Reviewer W item 6): the units by what they are, not "added units"
    axes[0].set_ylabel("$k$ least participating\nactive units removed")
    fig.text((D_LEFT_MM + 1.5 * D_W_MM + D_GAP_MM) / ps.W_FULL_MM, 1.0 - (ROW2_TOP_MM + ROW2_H_MM + 6.0) / FIG_H_MM,
             "$k$ random active units removed (median of 5)", ha="center", va="top", fontsize=ps.LABEL_PT,
             color=ps.INK)
    axes[0].text(0.49, 0.50, "equal", fontsize=ps.TICK_PT, color=ps.MUTED, ha="right", va="bottom")
    ps.panel_letter(axes[0], "d")
    return axes, rows


def edge_or_group(ax, pts, col, marker):
    """Open-marker groups of one series in panel d; a value below the axis goes to the edge.

    Args:
        ax: the axes, limits set; pts: list of (x, y); col: colour; marker: shape.
    Returns:
        list of fig_paper_F5_drive.open_groups dicts for the points inside the axes.
    """
    inside, edge = [], []
    for x, y in pts:
        if x < D_LO or y < D_LO:
            edge.append(((D_LO + 0.01, y) if x < D_LO else (x, D_LO + 0.012), "<" if x < D_LO else "v"))
        else:
            inside.append((x, y))
    # edge markers of one series closer than EDGE_JOIN (data units along the edge) are drawn as one
    # marker at their mean with their number beside it (round 18: two CDDM rate-penalty networks)
    edge.sort(key=lambda e: (e[1], e[0][1] if e[1] == "<" else e[0][0]))
    merged = []
    for (xy, d) in edge:
        along = xy[1] if d == "<" else xy[0]
        if merged and merged[-1][1] == d and abs(along - merged[-1][2][-1]) < EDGE_JOIN:
            merged[-1][0].append(xy)
            merged[-1][2].append(along)
        else:
            merged.append(([xy], d, [along]))
    s_mark = 11 * (F5.TRI_SCALE ** 2 if marker == "<" else 1.0)
    for xys, d, _ in merged:
        xd, yd = np.mean([q[0] for q in xys]), np.mean([q[1] for q in xys])
        if len(xys) == 1:
            F5.edge_arrow(ax, xd, yd, d, col, False, marker, s=s_mark)
            continue
        # a merged marker (PR #202 second review, item 1): its arrow is shortened to EDGE_ARROW_PT so the
        # tail stops short of the neighbouring markers, and its count sits outside the axes, beside the
        # marker it counts (as fig_paper_F5_drive.draw_groups labels a group at the edge)
        ax.scatter([xd], [yd], s=F5.open_size(marker, s_mark), marker=marker, zorder=F5.Z_OPEN, clip_on=False,
                   **F5.open_kw(col))
        off = (EDGE_ARROW_PT, 0) if d == "<" else (0, EDGE_ARROW_PT)
        ax.annotate("", xy=(xd, yd), xytext=off, textcoords="offset points", zorder=4,
                    arrowprops=dict(arrowstyle="-|>,head_length=0.15,head_width=0.12", lw=0.5, color=col,
                                    shrinkA=0, shrinkB=2.2))
        if d == "<":
            ax.annotate(str(len(xys)), xy=(D_LO - 0.012, yd), xytext=(-1.5, 0), textcoords="offset points",
                        fontsize=ps.TICK_PT, color=ps.INK, ha="right", va="center", annotation_clip=False)
        else:
            ax.annotate(str(len(xys)), xy=(xd, D_LO - 0.015), xytext=(0, -1.5), textcoords="offset points",
                        fontsize=ps.TICK_PT, color=ps.INK, ha="center", va="top", annotation_clip=False)
    if not inside:
        return []
    x, y = (np.array(v) for v in zip(*inside))
    return F5.open_groups(ax, x, y, col, marker, 11 * (F5.TRI_SCALE ** 2 if marker == "<" else 1.0))


def fmt(vals, nd=3):
    """Comma-separated values rounded to nd decimals ('-' for an empty list)."""
    return ", ".join(f"{v:.{nd}f}" for v in vals) if len(vals) else "-"


def report(cells, vs, order, table, floor, rows_c, rows_d, tol):
    """Print every number drawn and the verdict table.

    Args:
        cells: the `cell` dicts; vs: `verdicts` output; order, table: `column_order` output;
        floor: load_3bff's floor list; rows_c, rows_d: the panel c and d rows; tol: dict task ->
        tolerance.
    Returns:
        None.
    """
    print("\n--- panels a, b: one row per (technique, task, value) cell; fold = count / control mean; "
          "dperf = performance - control mean ---")
    for c, v in zip(cells, vs):
        r = v["rule"]
        print(f"  {TECH_NAME[c['tech']]:>20} {c['task']:>5} {c['label'] or '-':>8} [{c['status']}] "
              f"active {', '.join(map(str, c['active'])) or '-'} (fold {fmt(c['fold'], 2)}; mean "
              f"{np.mean(c['fold']) if c['fold'] else float('nan'):.2f}) perf {fmt(c['perf'])} (dperf "
              f"{fmt(c['dperf'])}; mean {np.mean(c['dperf']) if c['dperf'] else float('nan'):+.4f}); control "
              f"active {', '.join(map(str, c['ctrl_active']))} perf {fmt(c['ctrl_perf'])}; "
              f"learned {sum(c['learned'])}/{len(c['learned'])}; stopped {c['n_stopped']}; training "
              f"{c['n_running']}; {c['source']}" + (f"; {c['note']}" if c["note"] else ""))
        print(f"  {'':>36} rule: {r['text'] if r else 'no finished run'}"
              + (f" | stored: {c['stored']}" if c["stored"] else "")
              + ("" if v["agrees"] in (None, True) else " | DISAGREES WITH THE STORED VERDICT"))
    print("\n--- 3BFF at the floor crossing (not drawn; the catalogue's 1.05x read) ---")
    for f in floor:
        r = rule(f["active"], f["r2"], f["ctrl_active"], f["ctrl_r2"], tol["3BFF"]) if f["active"] else None
        print(f"  {TECH_NAME[f['tech']]:>20}: {f['n_read']} of {f['n_runs']} read; active "
              f"{', '.join(map(str, f['active'])) or '-'} R2 {fmt(f['r2'])}; control active "
              f"{', '.join(map(str, f['ctrl_active']))} R2 {fmt(f['ctrl_r2'])}; "
              f"{r['text'] if r else 'no floor read'}" + (f"; excluded: {f['excluded']}" if f["excluded"] else ""))
    print("\n--- verdict table (technique x task), the drawn read-out ---")
    print(f"  {'technique':>20} | " + " | ".join(f"{t:^9}" for t in TASKS) + " | works on")
    for t in order:
        n_works = sum(table[(t, k)][0] for k in TASKS)
        print(f"  {TECH_NAME[t]:>20} | " + " | ".join(f"{'works' if table[(t, k)][0] else 'fails':^9}" for k in TASKS)
              + f" | {n_works} task(s)")
        for k in TASKS:
            print(f"  {'':>20}   {k}: {table[(t, k)][2]}")
    print(f"\n  column order: {', '.join(TECH_NAME[t] for t in order)}")
    print("\n--- panel c (across sizes; 3BFF read as Figure 5c: floor crossing, rescaling and read-out "
          "dropout at 75,000; CDDM at the floor crossing, G6) ---")
    for task, fam, n, k, m, r in rows_c:
        print(f"  {task:>5} {F5.NAME[fam]:>20} N={n:<5} n={k} fold {m:.2f} dR2 median {r:+.4f}")
    print("\n--- panel d (every task: x = after k random active units, median of 5; y = after the k added "
          "units, the inactive units left in place) ---")
    for r in rows_d:
        print(f"  {r['task']:>5} {TECH_NAME[r['tech']]:>20} {r['name']:<32} k={r['k']:<4} intact {r['intact']:.3f} "
              f"x {r['x']:+.3f} y {r['y']:+.3f} drop {r['intact'] - r['y']:.3f} ({r['where']})")
    for task in TASKS:
        sel = [r for r in rows_d if r["task"] == task]
        print(f"  {task}: {len(sel)} networks; random lower than added in {sum(r['x'] < r['y'] for r in sel)}; "
              f"y {min(r['y'] for r in sel):.3f} to {max(r['y'] for r in sel):.3f}; "
              f"x {min(r['x'] for r in sel):+.3f} to {max(r['x'] for r in sel):+.3f}")


def main():
    """Load the three tasks' read-outs, order the techniques, draw the figure, print every number."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cells_ff, floor = load_3bff()
    cells_cddm, r2_tol = load_cddm()
    cells_maze, success_tol, _ = load_maze()
    tol = {"3BFF": r2_tol, "CDDM": r2_tol, "maze": success_tol}
    cells = sorted(cells_ff + cells_cddm + cells_maze, key=value_key)    # stable: values in order within a slot group
    vs = verdicts(cells, tol)
    order, table = column_order(vs)
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    panel_key(fig)
    slots, columns, width = slot_layout(order, cells)
    assert LEFT_MM + width <= ps.W_FULL_MM - 2, f"the slots are {width:.1f} mm wide; the canvas is too narrow"
    panel_ab(fig, slots, columns, width, tol)
    _, rows_c = panel_sizes(fig)
    _, rows_d = panel_removal(fig)
    report(cells, vs, order, table, floor, rows_c, rows_d, tol)
    learned_control_sensitivity(cells, tol)
    tolerance_sensitivity(cells, tol)
    rule_tables(cells, load_3bff_all_values(), tol["maze"])
    check_overlaps(fig)
    ps.save(fig, "fig_paper_F5_tasks")


if __name__ == "__main__":
    main()
