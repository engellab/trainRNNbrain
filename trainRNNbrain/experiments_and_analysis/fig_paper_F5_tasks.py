#!/usr/bin/env python3
"""Manuscript Figure 5, cross-task rebuild: the training techniques on 3BFF, CDDM and the maze at N = 1,000.

QUESTION (Pavel, 5 Nov 2026; dead-relu-paper#183, #184; ROUND15_BRIEF "WP cross-task"). Which
training techniques increase the number of active units on every task, and which on some tasks
only? The pre-registered criterion of docs/interventions_matrix.md, applied per (technique, task)
cell at N = 1,000: the technique WORKS when every seed has more active units than the largest
control and the arm's mean performance is within the tolerance of the controls' mean (held-out
R^2 within R2_TOL = 0.01 on the supervised tasks; the clean probe's success within SUCCESS_TOL =
0.05 on the maze); otherwise it FAILS, with the condition that failed. Every cell is read at the
same-size read-out point: 75,000 iterations on 3BFF and CDDM, update 2,000 on the maze (the end of
its budget), so every marker is OPEN (paperstyle, dead-relu-paper#83).

PANELS (165.1 x FIG_H_MM mm, bold letters, a key row on top).
  (a) fold change in the number of active units over the task's own control (the control's mean
      at the same read-out point), one column per technique, inside it one group per task
      (3BFF, CDDM, maze: the task colours), one open marker per group at the seeds' mean with a
      thin bar over their range (the paper's convention for open markers that would overlap,
      dead-relu-paper#104; the number of seeds beside it when it is not 3) and, where a
      technique was run at several values on a task, one group per value (the value written
      under the column); the band behind a group is the control's own range of seeds. A run the maze's rollback guard stopped
      is a cross at 0; a maze run that did not learn the task (common.maze_learned_index) is a
      cross at its value; a group with no finished run says "unfinished", a partly finished one
      "k/n". The columns are ORDERED by the number of tasks on which the technique works (ties: by
      the number of tasks on which the count condition passes, then the name).
  (b) the performance difference from the controls' mean: top, held-out R^2 (3BFF and CDDM), the
      tolerance 0.01 dashed; bottom, the maze's success at the end, the tolerance 0.05 dashed.
  (c) the technique that works on every task across network sizes on 3BFF, the current Figure 5c:
      prune-and-duplicate's fold change (top) and R^2 change (bottom) per network against N, read
      at the floor crossing (filled), the other techniques as context lines through their means
      (rescaling and read-out dropout at 75,000 iterations, dashed, as Figure 5c draws them).
  (d) removing the added units: performance after removing a network's k added units (the paper's
      name for them; "recruited" in the read-out files)
      (its number of active units minus the controls' mean; the k least participating active
      units, after the inactive ones) against performance after removing as many active units
      at random (median over 5 draws), one marker per network. 3BFF: the removal analysis A3
      (data/remedy_reads_31oct/a3_N1000.json, held-out R^2 at 75,000 iterations); the maze: the
      rate-penalty networks ablated at the end of training (data/rl_maze_ablation_interventions,
      success under the training noise; the random order there draws from all units, and its
      value at k is the median over the five orders, interpolated on the k grid). CDDM: no
      technique network has been ablated, and the panel says so.

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
data/read_maze_arms.json (read_maze_arms.py; the learned controls' reference). A cell that is
still training is drawn as the JSON holds it and picked up on a rerun once the JSON is rebuilt.

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
from common import DATA_DIR
from fig_ed_maze import ax_mm, check_overlaps, load_ablation
from fig_paper_F2_fails import CACHE as F2_CACHE

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
RUC1 = os.path.join(REPO, "data", "read_uniform_cells.json")
RUC2 = os.path.join(REPO, "data", "read_uniform_cells_2.json")
MAZE_JSON = os.path.join(REPO, "data", "read_maze_arms.json")
MAZE_ABLATION_DIR = os.path.join(REPO, "data", "rl_maze_ablation_interventions")
N = 1000
N_SEEDS = 3                      # the groups' usual size; another size is written beside the marker
TASKS = ("3BFF", "CDDM", "maze")
TASK_COL = {"3BFF": ps.TASK_3BFF, "CDDM": ps.TASK_CDDM, "maze": ps.TASK_MAZE}
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
# the maze arms per technique: (value label, arm key in read_maze_arms.json)
MAZE_ARMS = {
    "duplicate": [("", "duplicate")],
    "frm": [("0.03", "frm0.03"), ("0.1", "frm0.1"), ("0.3", "frm0.3")],
    "dropout": [("", "dropout")],
    "frm+rws": [("0.1", "frm0.1+rws0.05")],
    "rescale": [("10\u00d7", "rescale"), ("slow tilt", "rescale_mild"), ("5% grow", "rescale_few")],
    "synnoise": [("", "synnoise")],
}
# the maze's rescaling variants (entry 2026-11-05_0832), named as fig_ed_maze_interventions names them:
# "slow tilt" is alpha 1.0005 and cap 8 (the paper's arm has 1.002 and 64), "5% grow" protects 0.05 N
# units (the paper's 0.2 N); "10x" is the paper's arm
MAZE_RESCALE_NOTE = {"rescale": "the paper's values (target 10x, alpha 1.002, cap 64, protect 0.2 N)",
                     "rescale_mild": "slow tilt: alpha 1.0005, cap 8", "rescale_few": "5% grow: protect 0.05 N"}

# layout, mm from the canvas' top-left: one slot per (technique, task, value) group
FIG_H_MM = 132.0                  # 136 -> 132 mm (round 15): the float with its caption fits the page
LEFT_MM = 16.0
PITCH_MM = 5.4                    # one group's slot
GAP_MM = 3.2                      # between two techniques
SEED_DX_MM = 0.85                 # a cell's crosses sit this far apart
CROSS_DX_MM = 1.3                 # a cross beside a group marker sits this far from it
KEY_TOP_MM = 1.5
A_TOP_MM, A_H_MM = 16.0, 27.0
B1_TOP_MM, B1_H_MM = 45.5, 15.0
B2_TOP_MM, B2_H_MM = 62.5, 14.0
ROW2_TOP_MM, ROW2_H_MM = 92.0, 30.0
C_LEFT_MM, C_W_MM = 16.0, 50.0
D_LEFT_MM, D_W_MM, D_GAP_MM = 90.0, 48.0, 4.0
FOLD_LIM = (0.0, 4.3)
R2_LIM = (-0.038, 0.013)
SUCC_LIM = (-0.60, 0.05)
D_LO = -0.05                      # panel d: values below this are drawn at the edge with an arrow


def rule(active, perf, ctrl_active, ctrl_perf, tol):
    """The pre-registered verdict on one cell: works, or fails with the condition that failed.

    Args:
        active: list of int, the arm's active counts (one per seed); perf: list of float, the
            arm's performance per seed (held-out R^2 or success); ctrl_active, ctrl_perf: the
            same for the control; tol: the tolerance on the mean performance.
    Returns:
        dict: count_ok (bool), perf_ok (bool), works (bool), fold (float, mean count over the
        control's mean), dperf (float, mean performance minus the control's mean), text (str).
    """
    fold = float(np.mean(active)) / float(np.mean(ctrl_active))
    dperf = float(np.mean(perf)) - float(np.mean(ctrl_perf)) if len(perf) else float("nan")
    count_ok = bool(len(active)) and all(a > max(ctrl_active) for a in active)
    perf_ok = bool(len(perf)) and dperf >= -tol
    why = []
    if not count_ok:
        why.append(f"count not above every control's ({max(ctrl_active)})")
    if not perf_ok:
        why.append(f"performance {dperf:+.3f} beyond {tol}")
    return dict(count_ok=count_ok, perf_ok=perf_ok, works=count_ok and perf_ok, fold=fold, dperf=dperf,
                text="works" if count_ok and perf_ok else "fails: " + "; ".join(why))


def cell(tech, task, label, active, perf, ctrl_active, ctrl_perf, n_expected, source, learned=None,
         n_stopped=0, n_running=0, stored=None, note=""):
    """One (technique, task, value) cell of panels a and b.

    Args:
        tech, task: keys of TECH and TASKS; label: str, the value written under the column ("" for
            a technique with one value on this task); active: list of int, active units at the
            read-out point, one per finished seed; perf: list of float, held-out R^2 or success per
            finished seed (may be shorter than `active` where a run has no performance read);
            ctrl_active, ctrl_perf: the control's lists; n_expected: int, seeds the pre-registration
            expects; source: str, the read-out file; learned: list of bool per finished seed (the
            maze), or None; n_stopped: int, runs the guard stopped (crosses at 0); n_running: int,
            runs still training; stored: str or None, the verdict the read-out file holds; note:
            str, printed.
    Returns:
        dict with the inputs plus status ("finished", "incomplete", "unfinished", "stopped"), fold
        (per seed), dperf (per seed) and the band (the control's range in fold and dperf units).
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
                band_dperf=(min(ctrl_perf) - perf_mean, max(ctrl_perf) - perf_mean))


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
    labels = {"frm": "0.1", "frm+rws": "0.1", "rescale": "10\u00d7"}
    out, floor = [], []
    for fam in TECH:
        c = series[fam][N]
        perf, ctrl_perf, note = c["r2_75k"], ctrl["r2_75k"], "R^2: catalogue, 36 draws"
        if not perf:
            perf = [r["r2_full"] for r in a3 if r["family"] == fam]
            ctrl_perf = a3_ctrl_r2
            note = "R^2: no 75,000-iteration weights in the catalogue's chain; the removal analysis's intact score (9 draws) against its controls"
        out.append(cell(fam, "3BFF", labels.get(fam, ""), c["active_75k"], perf, ctrl["active_75k"],
                        ctrl_perf, c["n_runs"], f"remedies_catalogue.json {c['cell'].split('/')[0]}", note=note))
        floor.append(dict(tech=fam, active=c["active"], r2=c["r2"], n_read=c["n_read"], n_runs=c["n_runs"],
                          ctrl_active=ctrl["active"], ctrl_r2=ctrl["r2"], excluded=c["excluded"]))
    return out, floor


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
    out += [ruc_cell("dropout", "", g1["CDDM per-unit rules"], "read-out dropout", "read_uniform_cells.json"),
            ruc_cell("frm+rws", "0.1", g2["CDDM rate penalty 0.1"], "rate 0.1 + sparsity", "read_uniform_cells_2.json"),
            ruc_cell("rescale", "10\u00d7", g2["CDDM rescaling"], "rescaling", "read_uniform_cells_2.json"),
            ruc_cell("synnoise", "", g1["CDDM noisy synapses"], "sigma_w 1.0", "read_uniform_cells.json")]
    return out, tol1


def load_maze():
    """The maze cells at update 2,000 from read_maze_arms.json.

    Returns:
        (cells, success_tol, reference): the `cell` dicts in MAZE_ARMS order, the tolerance the
        file holds and its controls' reference dict.
    """
    with open(MAZE_JSON) as fh:
        d = json.load(fh)
    ref = d["reference"]
    arms = {a["key"]: a for a in d["arms"]}
    ctrl = [r for r in d["runs"] if r["key"] == "control" and r["learned"] is not None]
    ctrl_active = [r["active_end"] for r in ctrl]
    ctrl_succ = [r["success_end"] for r in ctrl]
    assert float(np.mean(ctrl_active)) == ref["active"], "the learned controls do not give the file's reference"
    learned_of = {(r["key"], r["seed"]): r["learned"] is not None for r in d["runs"]}
    out = []
    for tech, values in MAZE_ARMS.items():
        for label, key in values:
            a = arms[key]
            out.append(cell(tech, "maze", label, a["active_end"], a["success_end"], ctrl_active, ctrl_succ,
                            len(a["active_end"]) + a["n_running"] + a["n_stopped"],
                            f"read_maze_arms.json '{key}'",
                            learned=[learned_of[(key, s)] for s in a["seeds_finished"]],
                            n_stopped=a["n_stopped"], n_running=a["n_running"], stored=a["verdict"],
                            note=MAZE_RESCALE_NOTE.get(key, "")))
    return out, d["success_tol"], ref


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
        r = rule(c["active"], c["perf"], c["ctrl_active"], c["ctrl_perf"], tol[c["task"]]) if c["active"] else None
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


def draw_group(ax, x, ys, col, marker, label=None, ms=ps.MARKER_PT):
    """One group of open markers the paper's way: a marker at the mean, a bar over the range.

    Open markers of one group would overlap at any spacing that fits the slots of these figures,
    and an open marker drawn over another's ink rim no longer reads as open (dead-relu-paper#104,
    tests/scan_open_markers.py), so a group is one open marker at its mean with a thin bar from
    its smallest to its largest value and, when `label` is given, that text beside it.

    Args:
        ax: axes whose x is in mm; x: the group's x; ys: the values (array-like); col: the
            colour; marker: the shape; label: str or None, written 1 mm right of the marker;
            ms: marker size in pt (triangles are drawn TRI_SCALE larger).
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
    if label:
        ax.text(x + 1.0, ys.mean(), label, fontsize=ps.TICK_PT, color=ps.INK, ha="left", va="center", zorder=6)


def draw_marks(ax, x, c, key, zero=None):
    """One cell's marks in one axes: the learned seeds as one group, the others as crosses.

    Args:
        ax: axes (x in mm); x: the slot's centre; c: a `cell` dict; key: "fold" or "dperf", the
            values drawn; zero: the y of a stopped run's cross, or None to draw none.
    Returns:
        None.
    """
    col, marker = TASK_COL[c["task"]], TECH[c["tech"]][1]
    vals = np.asarray(c[key], float)
    ok = np.asarray(c["learned"][:len(vals)], bool)
    n_group = int(ok.sum())
    label = str(n_group) if c["status"] == "finished" and n_group > 1 and n_group != N_SEEDS else None
    draw_group(ax, x, vals[ok], col, marker, label)
    crosses = list(vals[~ok]) + ([zero] * c["n_stopped"] if zero is not None else [])
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
    """Panels a and b: the fold change and the two performance differences over the shared slots.

    Args:
        fig: the figure; slots, columns, width: slot_layout output; tol: dict task -> tolerance.
    Returns:
        (ax_a, ax_b1, ax_b2).
    """
    ax_a = ax_mm(fig, LEFT_MM, A_TOP_MM, width, A_H_MM)
    ax_b1 = ax_mm(fig, LEFT_MM, B1_TOP_MM, width, B1_H_MM)
    ax_b2 = ax_mm(fig, LEFT_MM, B2_TOP_MM, width, B2_H_MM)
    for c, x in slots:
        band_ax = ax_b2 if c["task"] == "maze" else ax_b1
        for ax, key in ((ax_a, "band_fold"), (band_ax, "band_dperf")):
            lo, hi = c[key]
            ax.fill_between([x - PITCH_MM / 2, x + PITCH_MM / 2], [lo] * 2, [hi] * 2, color=ps.FAINT,
                            alpha=0.3, lw=0, zorder=1)
        draw_marks(ax_a, x, c, "fold", zero=0.0)
        draw_marks(band_ax, x, c, "dperf")
        if c["status"] == "unfinished":
            ax_a.text(x, 0.5, "unfinished", rotation=90, transform=ax_a.get_xaxis_transform(),
                      fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="center")
        elif c["status"] == "stopped":
            ax_a.text(x, 0.55, "stopped", rotation=90, transform=ax_a.get_xaxis_transform(),
                      fontsize=ps.TICK_PT, color=ps.MUTED, ha="center", va="center")
        elif c["status"] == "incomplete":
            ax_a.text(x, 0.97, f"{len(c['active'])}/{c['n_expected']}", transform=ax_a.get_xaxis_transform(),
                      fontsize=ps.MIN_PT, color=ps.MUTED, ha="center", va="top")
        if c["label"]:
            ax_b2.text(x, -0.06, c["label"], rotation=90, transform=ax_b2.get_xaxis_transform(),
                       fontsize=ps.TICK_PT, color=ps.INK, ha="center", va="top")
    for tech, x0, x1 in columns:
        ax_a.text((x0 + x1) / 2, 1.04, TECH[tech][0], transform=ax_a.get_xaxis_transform(),
                  fontsize=ps.LABEL_PT, color=ps.INK, ha="center", va="bottom", linespacing=1.15)
    ax_a.axhline(1.0, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_a.set_ylim(*FOLD_LIM)
    ax_a.set_yticks([0, 1, 2, 3, 4])
    ax_a.set_ylabel("active units,\nfold over control")
    ax_b1.axhline(0.0, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_b1.axhline(-tol["3BFF"], lw=ps.THIN_LW, ls=(0, (4, 3)), color=ps.INK, zorder=2)
    ax_b1.text(width - 0.3, -tol["3BFF"] - 0.0015, f"tolerance {tol['3BFF']}", fontsize=ps.TICK_PT,
               color=ps.INK, ha="right", va="top")
    ax_b1.set_ylim(*R2_LIM)
    ax_b1.set_yticks([-0.03, -0.02, -0.01, 0])
    ax_b1.set_yticklabels(["−0.03", "−0.02", "−0.01", "0"])
    ax_b1.set_ylabel("held-out $R^2$\nminus control")
    ax_b2.axhline(0.0, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_b2.axhline(-tol["maze"], lw=ps.THIN_LW, ls=(0, (4, 3)), color=ps.INK, zorder=2)
    ax_b2.text(width - 0.3, -tol["maze"] - 0.012, f"tolerance {tol['maze']}", fontsize=ps.TICK_PT,
               color=ps.INK, ha="right", va="top")
    ax_b2.set_ylim(*SUCC_LIM)
    ax_b2.set_yticks([-0.4, -0.2, 0])
    ax_b2.set_yticklabels(["−0.4", "−0.2", "0"])
    ax_b2.set_ylabel("success\nminus control")
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
    h2 = [Line2D([], [], marker="o", ls="none", ms=3.2, label=t, **ps.mark_fixed(TASK_COL[t])) for t in TASKS]
    h2.append(Line2D([], [], ls="none", label="open: 75,000 iterations (3BFF, CDDM); update 2,000 (maze)"))
    ax.legend(handles=h2, loc="lower center", ncol=len(h2), frameon=False, fontsize=ps.TICK_PT,
              handlelength=1.0, columnspacing=1.2, handletextpad=0.35, borderaxespad=0)
    return ax


def panel_sizes(fig):
    """Panel c: prune-and-duplicate across sizes on 3BFF, the other techniques as context lines.

    Reuses fig_paper_F5_drive's loaders and read-outs (floor crossing, filled; rescaling and
    read-out dropout at 75,000 iterations, dashed).

    Args:
        fig: the figure.
    Returns:
        (axes pair, rows): rows is a list of (technique, N, n networks, mean fold, median dR2).
    """
    series = F5.size_series(F5.load_catalogue())
    marks = F5.load_marks()
    h_top = ROW2_H_MM * 0.58
    ax_m = ax_mm(fig, C_LEFT_MM, ROW2_TOP_MM, C_W_MM, h_top)
    ax_r = ax_mm(fig, C_LEFT_MM, ROW2_TOP_MM + h_top + 1.5, C_W_MM, ROW2_H_MM - h_top - 1.5)
    rows, handles = [], []
    for fam in F5.MAIN:
        ro = "75k" if fam in F5.AT_75K else "floor"
        xs, mm, rr = [], [], []
        for n in F5.SIZES:
            d = F5.per_run(series, fam, n, ro, marks)
            if not len(d["mult"]):
                continue
            xs.append(n)
            mm.append(float(d["mult"].mean()))
            rr.append(float(np.median(d["dr2"])))
            rows.append((fam, n, int(len(d["mult"])), mm[-1], rr[-1]))
            if fam == "duplicate":
                ax_m.scatter(np.full(len(d["mult"]), n), d["mult"], s=9, marker=F5.MARK[fam], zorder=4,
                             **ps.mark_floor(F5.COL[fam], scatter=True))
                ax_r.scatter(np.full(len(d["dr2"]), n), d["dr2"], s=9, marker=F5.MARK[fam], zorder=4,
                             **ps.mark_floor(F5.COL[fam], scatter=True))
        ls = "--" if fam in F5.AT_75K else "-"
        alpha, lw, z = (1.0, ps.DATA_LW, 3) if fam == "duplicate" else (0.45, 0.8, 2)
        ax_m.plot(xs, mm, ls=ls, color=F5.COL[fam], lw=lw, alpha=alpha, zorder=z)
        ax_r.plot(xs, rr, ls=ls, color=F5.COL[fam], lw=lw, alpha=alpha, zorder=z)
        handles.append(Line2D([], [], ls=ls, color=F5.COL[fam], lw=lw, alpha=alpha, label=F5.NAME[fam]))
    for ax in (ax_m, ax_r):
        ax.set_xscale("log")
        ax.set_xlim(380, 5300)
        ax.set_xticks(F5.SIZES)
        ax.set_xticklabels([ps.fmt_count(n) for n in F5.SIZES])
        ax.minorticks_off()
        ps.ygrid(ax)
    ax_m.axhline(1, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
    ax_r.axhline(0, color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
    ax_m.set_ylim(0.6, 8.6)               # the legend sits above the lines (the rate penalty ends at 5.6)
    ax_m.set_yticks([1, 3, 5])
    ax_m.set_ylabel("fold change")
    ax_m.tick_params(labelbottom=False)
    ax_r.set_ylim(-0.045, 0.006)
    ax_r.set_yticks([-0.04, -0.02, 0])
    ax_r.set_yticklabels(["−0.04", "−0.02", "0"])
    ax_r.set_ylabel("$R^2$ minus\ncontrol")
    ax_r.set_xlabel(ps.LABELS["size"])
    ax_m.set_title("3BFF, four sizes", fontsize=ps.LABEL_PT, pad=3, loc="left")
    ax_m.legend(handles=handles, loc="upper left", fontsize=ps.MIN_PT, handlelength=1.6, ncol=2,
                columnspacing=0.8, labelspacing=0.2, borderaxespad=0.2, frameon=False)
    ps.panel_letter(ax_m, "c")
    return (ax_m, ax_r), rows


def maze_removal(abl, ref_active):
    """The maze's "recruited units removed" read of one ablation JSON, interpolated on its k grid.

    Args:
        abl: one dict from fig_ed_maze.load_ablation (N, n_active, ks, score_quiet, score_random,
            score_full); ref_active: float, the learned controls' mean count.
    Returns:
        dict: k_added (int), k_total (int, the inactive units plus the recruited ones), quiet
        (success with k_total units removed quietest first), random_median (the median over the
        random orders at k_total), random_mean, full (intact success under the training noise).
    """
    ks = np.asarray(abl["ks"], float)
    n_in = abl["N"] - abl["n_active"]
    k_added = abl["n_active"] - ref_active
    k_total = n_in + k_added
    quiet = float(np.interp(k_total, ks, abl["score_quiet"]))
    rand = np.array([np.interp(k_total, ks, row) for row in abl["score_random"]])
    return dict(k_added=int(round(k_added)), k_total=int(round(k_total)), quiet=quiet,
                random_median=float(np.median(rand)), random_mean=float(rand.mean()), full=abl["score_full"])


def panel_removal(fig, maze_ref):
    """Panel d: performance after removing the recruited units against as many random active units.

    Two sub-axes sharing y, 3BFF (held-out R^2) on the left and the maze (success) on the right, so
    a 3BFF network and a maze network with the same two values do not print on top of each other.

    Args:
        fig: the figure; maze_ref: the maze controls' reference dict (active: the mean count).
    Returns:
        (axes, rows): rows is a list of dicts per network (task, tech, name, x, y, where).
    """
    w = (D_W_MM - D_GAP_MM) / 2
    ax_ff = ax_mm(fig, D_LEFT_MM, ROW2_TOP_MM, w, ROW2_H_MM)
    ax_mz = ax_mm(fig, D_LEFT_MM + w + D_GAP_MM, ROW2_TOP_MM, w, ROW2_H_MM)
    rows = []
    a3 = F5.load_a3()[N]
    marks = F5.load_marks()
    for r in a3:
        if r["family"] not in TECH:
            continue
        rows.append(dict(task="3BFF", tech=r["family"], name=os.path.basename(r["run_dir"].rstrip("/"))[:28],
                         x=float(np.median(r["r2_random_active"])), y=float(r["r2_recruited"]),
                         where=marks["a3"].get(r["run_dir"], "calm"), k=r["surplus"]))
    for abl in load_ablation(MAZE_ABLATION_DIR) if os.path.isdir(MAZE_ABLATION_DIR) else []:
        tech = "frm" if "/frm" in abl["run_dir"] else "other"
        if tech not in TECH:
            continue
        m = maze_removal(abl, maze_ref["active"])
        rows.append(dict(task="maze", tech=tech, name=abl["run_dir"].split("Maze_interventions/")[-1],
                         x=m["random_median"], y=m["quiet"], where="end", k=m["k_added"], full=m["full"],
                         random_mean=m["random_mean"]))
    for ax, task, title in ((ax_ff, "3BFF", "3BFF, held-out $R^2$"), (ax_mz, "maze", "maze, success")):
        ax.set_xlim(D_LO - 0.012, 0.5)
        ax.set_ylim(D_LO - 0.015, 0.85)
        ax.plot([D_LO, 0.5], [D_LO, 0.5], color=ps.FAINT, lw=ps.THIN_LW, zorder=1)
        groups = []
        for tech in TECH:
            pts = [(r["x"], r["y"]) for r in rows if r["task"] == task and r["tech"] == tech]
            groups += edge_or_group(ax, pts, TASK_COL[task], TECH[tech][1])
        F5.draw_groups(ax, F5.see_through(ax, groups))
        ax.set_xticks([0, 0.2, 0.4])
        ax.set_yticks([0, 0.4, 0.8])
        ax.set_title(title, fontsize=ps.LABEL_PT, pad=3, loc="left")
        ps.ygrid(ax)
    ax_mz.tick_params(labelleft=False)
    ax_ff.set_ylabel("after removing the\n$k$ added units")
    ax_ff.text(0.49, 0.50, "equal", fontsize=ps.TICK_PT, color=ps.MUTED, ha="right", va="bottom")
    ax_mz.text(0.49, D_LO + 0.02, "CDDM:\nnot measured", fontsize=ps.TICK_PT, color=ps.MUTED, ha="right",
               va="bottom", linespacing=1.15)
    x0, x1 = ax_ff.get_position().x0, ax_mz.get_position().x1
    fig.text((x0 + x1) / 2, ax_ff.get_position().y0 - 6.5 / FIG_H_MM, "after removing $k$ random active units (median of 5)",
             fontsize=ps.LABEL_PT, color=ps.INK, ha="center", va="top")
    ps.panel_letter(ax_ff, "d")
    return (ax_ff, ax_mz), rows


def edge_or_group(ax, pts, col, marker):
    """Open-marker groups of one series in panel d; a value below the axis goes to the edge.

    Args:
        ax: the axes, limits set; pts: list of (x, y); col: colour; marker: shape.
    Returns:
        list of fig_paper_F5_drive.open_groups dicts for the points inside the axes.
    """
    inside = []
    for x, y in pts:
        if x < D_LO or y < D_LO:
            xd, yd = (D_LO + 0.01, y) if x < D_LO else (x, D_LO + 0.012)
            F5.edge_arrow(ax, xd, yd, "<" if x < D_LO else "v", col, False, marker,
                          s=11 * (F5.TRI_SCALE ** 2 if marker == "<" else 1.0))
        else:
            inside.append((x, y))
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
    print("\n--- panel c (3BFF across sizes; fold change read as Figure 5c: floor crossing, rescaling and "
          "read-out dropout at 75,000) ---")
    for fam, n, k, m, r in rows_c:
        print(f"  {F5.NAME[fam]:>20} N={n:<5} n={k} fold {m:.2f} dR2 median {r:+.3f}")
    print("\n--- panel d (added units removed; x = after k random active units, median; y = after "
          "the k added units) ---")
    for r in rows_d:
        extra = f"; intact {r['full']:.3f}; random mean {r['random_mean']:.3f}" if r["task"] == "maze" else ""
        print(f"  {r['task']:>5} {TECH_NAME[r['tech']]:>20} {r['name']:<32} k={r['k']:<4} x {r['x']:+.3f} "
              f"y {r['y']:+.3f} ({r['where']}){extra}")


def main():
    """Load the three tasks' read-outs, order the techniques, draw the figure, print every number."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.parse_args()
    cells_ff, floor = load_3bff()
    cells_cddm, r2_tol = load_cddm()
    cells_maze, success_tol, maze_ref = load_maze()
    tol = {"3BFF": r2_tol, "CDDM": r2_tol, "maze": success_tol}
    cells = cells_ff + cells_cddm + cells_maze
    vs = verdicts(cells, tol)
    order, table = column_order(vs)
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    panel_key(fig)
    slots, columns, width = slot_layout(order, cells)
    assert LEFT_MM + width <= ps.W_FULL_MM - 2, f"the slots are {width:.1f} mm wide; the canvas is too narrow"
    panel_ab(fig, slots, columns, width, tol)
    _, rows_c = panel_sizes(fig)
    _, rows_d = panel_removal(fig, maze_ref)
    report(cells, vs, order, table, floor, rows_c, rows_d, tol)
    check_overlaps(fig)
    ps.save(fig, "fig_paper_F5_tasks")


if __name__ == "__main__":
    main()
