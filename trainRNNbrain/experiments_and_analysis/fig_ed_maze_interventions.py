#!/usr/bin/env python3
"""Extended Data figure: the paper's training techniques on the maze task (PPO) at N = 1,000.

QUESTION. On 3BFF, each training technique of Figure 5 increases the number of active units and
lowers held-out R^2. Does the same hold when the task is the maze and the optimiser is PPO (issue
engellab/trainRNNbrain#5)? The test was set before any run (issue #5): a technique "works" at a
value when, at the end of training, the arm's number of active units exceeds every learned
control's in all three seeds AND the arm's mean probe success is within SUCCESS_TOL = 0.05 of the
controls' mean; an arm that does not learn counts against the technique at that value.

PANELS (165.1 x FIG_H_MM mm, bold letters; a key row above the panels names each arm's colour and
marker, as Figure 5's does; the controls are BASELINE grey, Figure 5's convention):
  (a) active units against PPO updates, one thin line per finished run; each technique in its
      remedy colour, the three rate-penalty strengths as a light-to-dark ramp of RATE_PENALTY
      (the three shades of Figure 5d, fig_paper_F5_drive.RATE_SHADES, ordered by lambda); a filled
      marker at each run's "learned" probe (common.maze_learned_index: the first probe from which
      the clean probe's success stays at or above 0.9 for 5 probes, with the last probe at or above
      0.9 as well).
  (b) probe success at the end of training against active units at the end, one marker per run.
      Every value in this panel is read at the end of the fixed budget (update 2,000), so every
      marker is OPEN (paperstyle, dead-relu-paper#83: a filled marker means a run's own convergence
      point, an open marker a fixed iteration); a run that did not learn is a cross in its arm's
      colour, as in fig_ed_maze. A thin line marks the learned controls' mean count.
  (c) top: the fold change in active units at the end, each run's count over the learned controls'
      mean, one open marker per seed (a cross for a run that did not learn) and a bar at the arm's
      mean over its finished runs; bottom: each run's end success minus the learned controls' mean
      success, the same marks, with the pre-registered tolerance as a dashed line. One column per
      arm of the ladder; a column with no finished run says "unfinished", "stopped" or "no runs",
      so the layout does not change as runs finish.
  (d) the ablation (rl_maze_ablate.py) of every technique network with a JSON under --ablation,
      overlaid with one learned control's (the lowest seed with a JSON under --control-ablation):
      success rate under the training noise as units are removed quietest first (solid), busiest
      first (dotted) and in random order (dashed; band: range over the random orders), a thin
      vertical line at each network's inactive count. Without a technique JSON the panel says so.

ARMS ARE DERIVED FROM EACH RUN'S SAVED CONFIG, not from its folder name (arm_of): trainer.lambda_frm,
trainer.lambda_rws, model.sigma_w, trainer.dropout (with its drop_rate and dropout_beta) and
trainer.prune_reinit. The folder name is printed beside the derived arm; a folder whose name does not
match its config's arm is reported. A run whose config switches on a combination outside the
ladder, or that is not an N = 1,000 ReLU run, is listed and drawn nowhere.

READ-OUT. A run is finished when its last logged update reaches the config's n_updates; an unfinished
run is listed with its last update and drawn nowhere. A run folder with no run files is listed as
stopped and drawn nowhere (its arm can then only come from the folder name, and the printout says
so). The controls' reference is the mean over the LEARNED controls among the uniform series'
N = 1,000 ReLU runs: the pre-registered exclusion of a run that does not learn (issue #3) applies to
the controls as to the arms; the printout gives the means over all controls beside it.

DATA. --root (default data/trained_RNNs_rl_maze/Maze_interventions): the ladder of issue #5,
<arm>/s<seed>/ run folders written by run_rl.py (Spock arrays 7383696, 7383697 and 7383755).
--control-root (default data/trained_RNNs_rl_maze/Maze_suite): the uniform series (Spock 7383494),
of which only the N = 1,000 ReLU runs without the goal cue are used. --ablation (default
data/rl_maze_ablation_interventions) and --control-ablation (default data/rl_maze_ablation): JSONs
of rl_maze_ablate.py, matched to runs by the run folder each records.

OUTPUT. img/internal_figures/fig_ed_maze_interventions.{pdf,svg} (paperstyle.save, vector only)
and, on stdout: one row per run (derived arm, folder, seed, status, last update, success at the
end, learned update, active at learned, active at the end, fold change); the controls' reference;
one row per arm (mean fold change, success cost, the pre-registered verdict); the ablation numbers;
the markers of panel b that print closer than one marker diameter; the text-overlap check.

Run:  python trainRNNbrain/experiments_and_analysis/fig_ed_maze_interventions.py [--root DIR]
      [--control-root DIR] [--ablation DIR] [--control-ablation DIR]
"""
import argparse
import glob
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import yaml

import paperstyle as ps
from common import MAZE_LEARNED_SUCCESS, maze_load_runs
from fig_ed_maze import ax_mm, check_overlaps, load_ablation, read_points, time_axis

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_interventions")
CONTROL_ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite")
ABLATION_DIR = os.path.join(REPO, "data", "rl_maze_ablation_interventions")
CONTROL_ABLATION_DIR = os.path.join(REPO, "data", "rl_maze_ablation")

N = 1000                          # the ladder's one network size
SEEDS = (1, 2, 3)
SUCCESS_TOL = 0.05                # pre-registered: an arm's mean end success within this of the controls'
TRI_SCALE = 1.25                  # an open triangle is drawn this much larger (a circle's area; Figure 5)

# The three rate-penalty strengths, light to dark: Figure 5d's three shades of the rate violet
# (fig_paper_F5_drive.RATE_SHADES, which that panel assigns to networks), here ordered by lambda.
RATE_RAMP = {0.03: "#c3b0e8", 0.1: "#9d7fd3", 0.3: ps.RATE_PENALTY}

# The ladder of issue #5 in drawing order. key (as arm_of derives it from a saved config) ->
# (key-row label, colour, marker, the launcher's folder name). The folder name is used only to
# report a mismatch with the derived arm and to name the arm of a folder that holds no config.
ARMS = {
    "control":        ("control",                   ps.BASELINE,       "o", "control"),
    "frm0.03":        ("rate 0.03",                 RATE_RAMP[0.03],   "o", "frm0.03"),
    "frm0.1":         ("rate 0.1",                  RATE_RAMP[0.1],    "o", "frm0.1"),
    "frm0.3":         ("rate 0.3",                  RATE_RAMP[0.3],    "o", "frm0.3"),
    "frm0.1+rws0.05": ("rate 0.1 + sparsity 0.05",  ps.RATE_SPARSITY,  "P", "frm0.1_rws0.05"),
    "dropout":        ("read-out dropout",          ps.DROPOUT,        "v", "dropout0.2_beta4"),
    "synnoise":       ("noisy synapses",            ps.SYNAPTIC_NOISE, "s", "sigmaw1.0"),
    "rescale":        ("rescaling",                 ps.RESCALE,        "D", "rescale"),
}
LADDER = [k for k in ARMS if k != "control"]
# panel c's column labels: at most two lines, under 10 mm wide at TICK_PT
SHORT = {"frm0.03": "rate\n0.03", "frm0.1": "rate\n0.1", "frm0.3": "rate\n0.3",
         "frm0.1+rws0.05": "rate 0.1 +\nsparsity", "dropout": "read-out\ndropout",
         "synnoise": "noisy\nsynapses", "rescale": "rescaling"}

# layout in millimetres from the canvas's top-left corner: (left, top, width, height)
FIG_H_MM = 106.0
LAYOUT_MM = {
    "key":   (0.0, 0.8, 165.1, 7.0),
    "a":     (12.5, 13.0, 66.0, 33.0),
    "b":     (97.0, 13.0, 63.0, 33.0),
    "c_top": (15.5, 60.0, 72.0, 19.0),    # 15.5: the lower axes' two-line y label fits left of it
    "c_bot": (15.5, 81.5, 72.0, 15.0),
    "d_key": (101.0, 52.5, 60.0, 7.0),    # the line-style key of (d), above it and off the data
    "d":     (103.0, 60.0, 57.0, 36.5),
}


# --------------------------------------------------------------------------------------------
# loading
# --------------------------------------------------------------------------------------------

def load_config(folder):
    """The saved config of one run folder as a plain dict.

    Args:
        folder: str, a run folder holding a *config.yaml (a continued run's last folder holds the
            config of the run that is read).
    Returns:
        dict.
    """
    with open(sorted(glob.glob(os.path.join(folder, "*config.yaml")))[0]) as fh:
        return yaml.safe_load(fh)


def arm_of(cfg):
    """The technique a saved config switches on, read from the config and not from the folder name.

    Args:
        cfg: dict, a saved run config: run_rl.py's layout (the trainer block under `trainer`) or
            the earlier one (under `ppo`); the model block under `model` in both.
    Returns:
        (key, note): key is "control" when no option is on, "frm<lambda>" for the rate penalty
        alone, "frm<lambda>+rws<lambda>" for the rate and sparsity penalties together, "synnoise",
        "dropout" or "rescale" for noisy synapses, read-out dropout or rescaling alone, and "other"
        for any other combination; note is the option values as text, for the printout.
    """
    tr = cfg.get("trainer") or cfg.get("ppo") or {}
    frm = float(tr.get("lambda_frm") or 0.0)
    rws = float(tr.get("lambda_rws") or 0.0)
    sigma_w = float((cfg.get("model") or {}).get("sigma_w") or 0.0)
    dropout = bool(tr.get("dropout", False))
    rescale = bool(tr.get("prune_reinit", False))
    parts = []
    if frm > 0:
        parts.append(f"lambda_frm {frm:g}")
    if rws > 0:
        parts.append(f"lambda_rws {rws:g}")
    if sigma_w > 0:
        parts.append(f"sigma_w {sigma_w:g}")
    if dropout:
        da = tr.get("dropout_args") or {}
        parts.append(f"dropout {da.get('dropout_kind', '?')} rate {da.get('drop_rate', '?')} "
                     f"beta {da.get('dropout_beta', '?')}")
    if rescale:
        parts.append("prune_reinit (rescaling)")
    flags = (frm > 0, rws > 0, sigma_w > 0, dropout, rescale)
    if not any(flags):
        key = "control"
    elif flags == (True, False, False, False, False):
        key = f"frm{frm:g}"
    elif flags == (True, True, False, False, False):
        key = f"frm{frm:g}+rws{rws:g}"
    elif flags == (False, False, True, False, False):
        key = "synnoise"
    elif flags == (False, False, False, True, False):
        key = "dropout"
    elif flags == (False, False, False, False, True):
        key = "rescale"
    else:
        key = "other"
    return key, ", ".join(parts) or "no option on"


def load_records(root, control_root):
    """Every run under the two roots, each with its derived arm, status and read-out points.

    The control root contributes only the runs whose config switches no option on and that are
    N = 1,000 ReLU runs without the goal cue (the rest of the uniform series is fig_ed_maze's). The
    ladder root contributes every run; a run there is drawn only if its derived arm is in the
    ladder and it is an N = 1,000 ReLU run without the goal cue.

    Args:
        root: str, the ladder's folder of <arm>/s<seed>/ run folders;
        control_root: str, the uniform series' folder.
    Returns:
        (records, stopped): records is a list of dicts, each a common.maze_load_runs run plus key
        (arm_of), note, folder_arm (the folder's parent name), rel (folder relative to the repo),
        last_update, n_updates, finished (bool), drawn (bool), why (why not drawn, or ""), and pt
        (fig_ed_maze.read_points) when finished; stopped is a list of dicts (folder_arm, seed, rel)
        for the run folders under `root` that hold no config.
    """
    records = []
    for source, root_dir in (("control", control_root), ("ladder", root)):
        if not os.path.isdir(root_dir):
            print(f"no folder {root_dir}")
            continue
        for run in maze_load_runs(root_dir):
            cfg = load_config(run["folder"])
            key, note = arm_of(cfg)
            tr = cfg.get("trainer") or cfg.get("ppo") or {}
            rec = dict(run, key=key, note=note, cfg=cfg,
                       folder_arm=os.path.basename(os.path.dirname(run["folder"])),
                       rel=os.path.relpath(run["folder"], REPO),
                       last_update=int(run["progress"][-1]["update"]),
                       n_updates=int(tr.get("n_updates") or 0), source=source)
            rec["finished"] = 0 < rec["n_updates"] <= rec["last_update"]
            standard = run["N"] == N and run["act"] == "relu" and not run["cue"]
            if source == "control":
                if key != "control" or not standard:
                    continue                      # the rest of the uniform series
                rec["why"] = ""
            elif key == "control":
                rec["why"] = "no option on under the ladder root (the controls are the series')"
            elif key not in ARMS:
                rec["why"] = f"not in the ladder ({note})"
            elif not standard:
                rec["why"] = f"not an N = {N:,} ReLU run without the goal cue"
            else:
                rec["why"] = ""
            rec["drawn"] = rec["why"] == "" and rec["finished"]
            if rec["finished"]:
                rec["pt"] = read_points(run)
            folder_of_key = ARMS.get(key, (None,) * 4)[3]
            if source == "ladder" and folder_of_key is not None and rec["folder_arm"] != folder_of_key:
                print(f"NOTE: {rec['rel']} is in folder '{rec['folder_arm']}' but its config says "
                      f"'{key}' ({note}); the config decides")
            records.append(rec)
    stopped = []
    for folder in sorted(glob.glob(os.path.join(root, "*", "s*", ""))):
        if not glob.glob(os.path.join(folder, "*config.yaml")):
            folder = folder.rstrip("/")
            seed = os.path.basename(folder)[1:]
            stopped.append(dict(folder_arm=os.path.basename(os.path.dirname(folder)),
                                seed=int(seed) if seed.isdigit() else None,
                                rel=os.path.relpath(folder, REPO)))
    return records, stopped


def control_reference(records):
    """The controls' reference: means over the learned controls, with the means over all controls.

    Args:
        records: list from load_records.
    Returns:
        dict with active (mean count at the end over the learned controls), active_max (the largest
        learned control's count), success (their mean end success), n (how many), all_active,
        all_success, n_all (the same over every finished control); or None without a learned
        control.
    """
    ctrl = [r for r in records if r["key"] == "control" and r["drawn"]]
    learned = [r for r in ctrl if r["pt"]["learned_i"] is not None]
    if not learned:
        return None
    act = np.array([r["pt"]["active_end"] for r in learned], float)
    suc = np.array([r["pt"]["success_end"] for r in learned], float)
    return dict(active=float(act.mean()), active_max=int(act.max()), success=float(suc.mean()),
                n=len(learned),
                all_active=float(np.mean([r["pt"]["active_end"] for r in ctrl])),
                all_success=float(np.mean([r["pt"]["success_end"] for r in ctrl])), n_all=len(ctrl))


def arm_summaries(records, stopped, ref):
    """Per ladder arm: its finished runs, fold changes, success cost, status and verdict.

    Args:
        records: list from load_records; stopped: list from load_records; ref: control_reference.
    Returns:
        dict key -> dict with runs (the finished, drawn records), fold (array, one per run: count at
        the end over the controls' mean; empty without a reference), cost (array: end success
        minus the controls' mean success), mean_fold, mean_cost (floats or None), status
        ("finished", "unfinished", "stopped" or "no runs": the state of the arm's column) and
        verdict (str: "works", "fails: ...", "incomplete ..." or "no verdict ...").
    """
    out = {}
    for key in LADDER:
        mine = [r for r in records if r["key"] == key and r["why"] == ""]
        fin = [r for r in mine if r["finished"]]
        n_stopped = sum(s["folder_arm"] == ARMS[key][3] for s in stopped)
        status = ("finished" if fin else "unfinished" if mine else "stopped" if n_stopped
                  else "no runs")
        s = dict(runs=fin, status=status, fold=np.array([]), cost=np.array([]), mean_fold=None,
                 mean_cost=None)
        if fin and ref is not None:
            s["fold"] = np.array([r["pt"]["active_end"] / ref["active"] for r in fin])
            s["cost"] = np.array([r["pt"]["success_end"] - ref["success"] for r in fin])
            s["mean_fold"], s["mean_cost"] = float(s["fold"].mean()), float(s["cost"].mean())
        if ref is None:
            s["verdict"] = "no verdict (no learned control)"
        elif not fin:
            s["verdict"] = f"no verdict ({status})"
        elif len(fin) < len(SEEDS):
            s["verdict"] = f"incomplete ({len(fin)} of {len(SEEDS)} runs finished)"
        else:
            low = [r["seed"] for r in fin if r["pt"]["active_end"] <= ref["active_max"]]
            reasons = []
            if low:
                reasons.append(f"count not above every learned control's ({ref['active_max']}) in "
                               f"seed{'s' if len(low) > 1 else ''} {', '.join(map(str, low))}")
            if s["mean_cost"] < -SUCCESS_TOL:
                reasons.append(f"success cost {s['mean_cost']:+.3f} beyond {SUCCESS_TOL}")
            s["verdict"] = "works" if not reasons else "fails: " + "; ".join(reasons)
        out[key] = s
    return out


def match_ablations(abl, records):
    """Pair every ablation JSON with the run record whose folder it names.

    Args:
        abl: list from fig_ed_maze.load_ablation; records: list from load_records.
    Returns:
        list of (json dict, record) for the JSONs whose run folder is a drawn record; the others
        are left out (the control folder also holds the other sizes of the series).
    """
    by_folder = {r["folder"]: r for r in records if r["drawn"]}
    out = []
    for a in abl:
        folder = a["run_dir"] if os.path.isabs(a["run_dir"]) else os.path.join(REPO, a["run_dir"])
        rec = by_folder.get(os.path.normpath(folder))
        if rec is not None:
            out.append((a, rec))
    return out


# --------------------------------------------------------------------------------------------
# printout
# --------------------------------------------------------------------------------------------

def print_runs(records, stopped, ref):
    """One row per run and per stopped folder: every number the panels draw.

    Args:
        records: list from load_records; stopped: list from load_records; ref: control_reference.
    Returns:
        None.
    """
    fmt = lambda v: "-" if v is None else str(v)
    print(f"{'arm (from config)':>17} {'folder':>36} {'seed':>4} {'status':>10} {'update':>6} "
          f"{'succ@end':>8} {'learned':>7} {'act@learn':>9} {'act@end':>7} {'fold':>5}  note")
    for r in sorted(records, key=lambda r: (r["source"] != "control", list(ARMS).index(r["key"])
                                            if r["key"] in ARMS else 99, r["seed"])):
        p = r.get("pt")
        status = "finished" if r["finished"] else "unfinished"
        fold = "-" if (p is None or ref is None) else f"{p['active_end'] / ref['active']:.2f}"
        why = "" if not r["why"] else f"NOT DRAWN: {r['why']}"
        print(f"{r['key']:>17} {r['rel'][-36:]:>36} {r['seed']:>4} {status:>10} {r['last_update']:>6} "
              + (f"{p['success_end']:>8.2f} {fmt(p['learned']):>7} {fmt(p['active_learned']):>9} "
                 f"{p['active_end']:>7} {fold:>5}" if p else f"{'-':>8} {'-':>7} {'-':>9} {'-':>7} {'-':>5}")
              + f"  {r['note']} {why}".rstrip())
    for s in stopped:
        print(f"{'?':>17} {s['rel'][-36:]:>36} {fmt(s['seed']):>4} {'stopped':>10} {'-':>6} "
              f"{'-':>8} {'-':>7} {'-':>9} {'-':>7} {'-':>5}  no run files on disk; the arm "
              f"'{s['folder_arm']}' is the folder's name, not a config's")
    have = {(r["key"], r["seed"]) for r in records} | {
        (k, s["seed"]) for s in stopped for k in ARMS if ARMS[k][3] == s["folder_arm"]}
    miss = [(k, s) for k in LADDER for s in SEEDS if (k, s) not in have]
    print("missing ladder cells (no folder): " + (", ".join(f"{k} s{s}" for k, s in miss) or "none"))
    if ref is None:
        print("controls: no learned control, so no reference, no fold change and no verdict")
    else:
        print(f"controls' reference: {ref['n']} learned controls, active at the end mean "
              f"{ref['active']:.1f} (largest {ref['active_max']}), end success mean {ref['success']:.3f}; "
              f"over all {ref['n_all']} controls: active {ref['all_active']:.1f}, success "
              f"{ref['all_success']:.3f}")


def print_arms(summ):
    """One row per ladder arm: mean fold change, success cost and the pre-registered verdict.

    Args:
        summ: dict from arm_summaries.
    Returns:
        None.
    """
    print(f"\n{'arm':>15} {'status':>10} {'n':>2} {'fold per seed':>22} {'mean fold':>9} "
          f"{'succ cost':>9}  verdict (count above every learned control's in all seeds AND "
          f"mean success within {SUCCESS_TOL} of the controls')")
    for key, s in summ.items():
        folds = ", ".join(f"{v:.2f}" for v in s["fold"]) or "-"
        mf = "-" if s["mean_fold"] is None else f"{s['mean_fold']:.2f}"
        mc = "-" if s["mean_cost"] is None else f"{s['mean_cost']:+.3f}"
        print(f"{key:>15} {s['status']:>10} {len(s['runs']):>2} {folds:>22} {mf:>9} {mc:>9}  {s['verdict']}")


# --------------------------------------------------------------------------------------------
# panels
# --------------------------------------------------------------------------------------------

def marker_size(marker):
    """Marker size (pt) of an open marker of a shape: triangles TRI_SCALE times larger (Figure 5).

    Args:
        marker: str, a matplotlib marker.
    Returns:
        float.
    """
    return ps.MARKER_PT * (TRI_SCALE if marker in ("v", "^") else 1.0)


def run_marker(ax, x, y, rec):
    """One run's marker at (x, y): open in its arm's shape if learned, a cross if not.

    Both read-out points of this figure are the end of the fixed budget, so a learned run's
    marker is open (dead-relu-paper#83); the cross is fig_ed_maze's mark of a run that did not
    learn.

    Args:
        ax: axes; x, y: data coordinates; rec: a load_records record (finished).
    Returns:
        None.
    """
    _, col, marker, _ = ARMS[rec["key"]]
    if rec["pt"]["learned_i"] is None:
        ax.plot([x], [y], "x", ms=ps.MARKER_PT, mec=col, mew=0.8, ls="none", zorder=5, clip_on=False)
    else:
        ax.plot([x], [y], marker, ms=marker_size(marker), ls="none", zorder=5, clip_on=False,
                **ps.mark_fixed(col))


def panel_key(ax, keys):
    """The key row: one entry per arm drawn (colour, marker) and one row of marker meanings.

    Args:
        ax: a blank axes spanning the figure width; keys: the ARMS keys with a finished run.
    Returns:
        None.
    """
    ps.blank(ax)
    h = [Line2D([], [], marker=ARMS[k][2], color=ARMS[k][1], ls="-", lw=ps.DATA_LW, ms=3.4,
                mec=ps.INK, mew=ps.MARKER_EW, label=ARMS[k][0]) for k in keys]
    if h:
        leg = ax.legend(handles=h, loc="upper center", ncol=len(h), frameon=False,
                        fontsize=ps.TICK_PT, handlelength=1.5, columnspacing=1.2,
                        handletextpad=0.35, borderaxespad=0)
        ax.add_artist(leg)
    h2 = [Line2D([], [], marker="o", ls="none", ms=ps.MARKER_PT, **ps.mark_floor(ps.INK),
                 label=f"learned (first of 5 probes at or above {MAZE_LEARNED_SUCCESS:g})"),
          Line2D([], [], marker="o", ls="none", ms=ps.MARKER_PT, **ps.mark_fixed(ps.INK),
                 label="end of training (update 2,000)"),
          Line2D([], [], marker="x", ls="none", ms=ps.MARKER_PT, mec=ps.INK, mew=0.8,
                 label="not learned")]
    ax.legend(handles=h2, loc="lower center", ncol=len(h2), frameon=False, fontsize=ps.TICK_PT,
              handlelength=1.0, columnspacing=1.6, handletextpad=0.35, borderaxespad=0)


def panel_traces(ax, records, x_max):
    """Panel (a): active units against PPO updates, one line per finished run, learned probes marked.

    Args:
        ax: axes; records: list from load_records; x_max: int, the x limit.
    Returns:
        int, the number of runs drawn.
    """
    drawn = [r for r in records if r["drawn"]]
    for r in sorted(drawn, key=lambda r: r["key"] != "control"):     # the controls go under
        _, col, marker, _ = ARMS[r["key"]]
        pr, li = r["probes"], r["pt"]["learned_i"]
        ax.plot(pr["update"], pr["active"], "-", lw=ps.THIN_LW, color=col,
                zorder=2 if r["key"] == "control" else 3)
        if li is not None:
            ax.plot([pr["update"][li]], [pr["active"][li]], marker, ms=ps.MARKER_PT, ls="none",
                    zorder=5, **ps.mark_floor(col))
    if not drawn:
        ax.text(0.5, 0.5, "no finished runs", transform=ax.transAxes, ha="center", va="center",
                fontsize=ps.TICK_PT, color=ps.MUTED)
    time_axis(ax, x_max)
    ax.set(ylim=(0, N), xlabel="PPO update", ylabel=ps.of_n(ps.LABELS["active"], N))
    ax.set_yticks([0, N / 4, N / 2, 3 * N / 4, N])
    ps.count_ticks(ax.yaxis)
    ps.ygrid(ax)
    return len(drawn)


def panel_end(ax, records, ref):
    """Panel (b): probe success at the end against active units at the end, one marker per run.

    Args:
        ax: axes; records: list from load_records; ref: control_reference (a thin line at its
            mean count, or nothing).
    Returns:
        list of (record, x, y) drawn, for the proximity check.
    """
    pts = []
    for r in records:
        if r["drawn"]:
            x, y = r["pt"]["active_end"], r["pt"]["success_end"]
            run_marker(ax, x, y, r)
            pts.append((r, x, y))
    if not pts:
        ax.text(0.5, 0.5, "no finished runs", transform=ax.transAxes, ha="center", va="center",
                fontsize=ps.TICK_PT, color=ps.MUTED)
    ax.set_xlim(0, N * 1.04)
    y_lo = min(0.3, np.floor(min([y for _, _, y in pts] + [1.0]) * 10) / 10 - 0.05)
    ax.set_ylim(y_lo, 1.03)
    ax.axhline(MAZE_LEARNED_SUCCESS, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax.text(0.01 * N, MAZE_LEARNED_SUCCESS - 0.015, f"{MAZE_LEARNED_SUCCESS:g}", fontsize=ps.TICK_PT,
            color=ps.MUTED, ha="left", va="top")
    if ref is not None:
        ax.axvline(ref["active"], lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
        ax.text(ref["active"] - 0.01 * N, y_lo + 0.02, "control\nmean", fontsize=ps.TICK_PT,
                color=ps.MUTED, ha="right", va="bottom", linespacing=1.2)
    ax.set_xticks([0, N / 4, N / 2, 3 * N / 4, N])
    ps.count_ticks(ax.xaxis)
    ax.set_yticks([t for t in (0.2, 0.4, 0.6, 0.8, 1.0) if t >= y_lo])
    ax.set(xlabel=ps.of_n("active units at the end", N), ylabel="success rate at the end\n(clean probe)")
    ps.ygrid(ax)
    return pts


def panel_fold(ax_t, ax_b, summ, ref):
    """Panel (c): fold change in active units (top) and the success cost (bottom), one column per arm.

    Args:
        ax_t, ax_b: the two axes (the bottom one carries the column labels); summ: arm_summaries;
            ref: control_reference.
    Returns:
        None.
    """
    for i, key in enumerate(LADDER):
        s, col = summ[key], ARMS[key][1]
        if not s["runs"] or ref is None:
            ax_t.text(i, 0.5, s["status"] if ref is not None else "no learned\ncontrol", rotation=90,
                      transform=ax_t.get_xaxis_transform(), ha="center", va="center",
                      fontsize=ps.TICK_PT, color=ps.MUTED)
            continue
        n = len(s["runs"])
        for j, r in enumerate(s["runs"]):
            x = i + (j - (n - 1) / 2) * 0.2
            run_marker(ax_t, x, s["fold"][j], r)
            run_marker(ax_b, x, s["cost"][j], r)
        for ax, v in ((ax_t, s["mean_fold"]), (ax_b, s["mean_cost"])):
            ax.plot([i - 0.32, i + 0.32], [v] * 2, "-", lw=1.4, color=col, zorder=4,
                    solid_capstyle="butt")
    folds = np.concatenate([s["fold"] for s in summ.values()] + [np.array([1.0])])
    costs = np.concatenate([s["cost"] for s in summ.values()] + [np.array([-SUCCESS_TOL])])
    ax_t.set_ylim(0, np.ceil(folds.max() * 1.15 * 2) / 2)
    ax_t.axhline(1, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_t.set_yticks(np.arange(0, ax_t.get_ylim()[1] + 1e-9, 1.0))
    ax_t.set_ylabel("fold change in\nactive units")
    ax_t.tick_params(axis="x", length=0, labelbottom=False)
    lo = np.floor(costs.min() * 10) / 10 - 0.05
    ax_b.set_ylim(lo, 0.08)
    ax_b.axhline(0, lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_b.axhline(-SUCCESS_TOL, ls=(0, (4, 3)), lw=ps.THIN_LW, color=ps.FAINT, zorder=1)
    ax_b.text(len(LADDER) - 0.5, -SUCCESS_TOL - 0.02, f"$-${SUCCESS_TOL:g}", fontsize=ps.TICK_PT,
              color=ps.MUTED, ha="right", va="top")
    ax_b.set_yticks(np.arange(0.0, lo, -0.2))
    ax_b.set_ylabel("success at the end\nminus control mean")
    ax_b.set_xticks(range(len(LADDER)))
    ax_b.set_xticklabels([SHORT[k] for k in LADDER], fontsize=ps.TICK_PT, linespacing=1.25)
    ax_b.tick_params(axis="x", length=0, pad=3)
    for ax in (ax_t, ax_b):
        ax.set_xlim(-0.6, len(LADDER) - 0.4)
        ps.ygrid(ax)


def panel_ablation(ax, ax_key, tech, ctrl, ref):
    """Panel (d): success rate as units are removed, technique networks over one control.

    Per network, in its arm's colour: quietest first (solid), busiest first (dotted), the mean over
    the random orders (dashed) with their range as a band; a thin vertical line at its inactive
    count. The legend of line styles sits in `ax_key`, above the panel and off the data.

    Args:
        ax: axes; ax_key: a blank axes above it; tech: list of (json, record) from
            match_ablations for the technique networks; ctrl: one (json, record) or None, the
            control drawn; ref: control_reference (sets the "added units" of each technique
            network).
    Returns:
        list of dicts per drawn network: key, seed, n_inactive, full, quiet_at_inactive,
        random_at_inactive, busy_at_inactive (success with 0 and n_inactive units removed), and for
        a technique network k_added (its count minus the controls' mean), k_grid (the grid point
        nearest n_inactive + k_added) and quiet_at_added (success there, quietest first).
    """
    rows = []
    for a, rec in ([ctrl] if ctrl else []) + tech:
        col = ARMS[rec["key"]][1]
        x = np.asarray(a["ks"], float)
        rand = np.asarray(a["score_random"], float)
        n_in = a["N"] - a["n_active"]
        ax.fill_between(x, rand.min(0), rand.max(0), color=col, alpha=0.15, lw=0, zorder=2)
        ax.plot(x, a["score_quiet"], "-", lw=ps.DATA_LW, color=col, zorder=5)
        ax.plot(x, rand.mean(0), "--", lw=0.9, color=col, zorder=4)
        ax.plot(x, a["score_busy"], ":", lw=1.1, color=col, zorder=4)
        ax.plot([n_in] * 2, [0, 1.0], "-", lw=ps.THIN_LW, color=col, zorder=3)
        i = int(np.argmin(np.abs(x - n_in)))
        row = dict(key=rec["key"], seed=rec["seed"], n_inactive=n_in, full=a["score_full"],
                   quiet_at_inactive=a["score_quiet"][i], random_at_inactive=float(rand.mean(0)[i]),
                   busy_at_inactive=a["score_busy"][i])
        if rec["key"] != "control" and ref is not None:
            k_added = a["n_active"] - ref["active"]
            j = int(np.argmin(np.abs(x - (n_in + k_added))))
            row.update(k_added=k_added, k_grid=int(x[j]), quiet_at_added=a["score_quiet"][j])
        rows.append(row)
    if not tech:
        ax.text(0.5, 0.5, "no ablation of a\ntechnique network yet", transform=ax.transAxes,
                ha="center", va="center", fontsize=ps.TICK_PT, color=ps.MUTED)
    ax.set(xlim=(0, N), ylim=(0, 1.02), xlabel=ps.of_n("units removed", N),
           ylabel="success rate (training noise)")
    ax.set_xticks([0, N / 2, N])
    ps.count_ticks(ax.xaxis)
    ax.set_yticks([0, 0.5, 1.0])
    ps.blank(ax_key)
    ax_key.legend(handles=[Line2D([], [], color=ps.MUTED, lw=ps.DATA_LW),
                           (Patch(color=ps.MUTED, alpha=0.2, lw=0),
                            Line2D([], [], color=ps.MUTED, lw=0.9, ls="--")),
                           Line2D([], [], color=ps.MUTED, lw=1.1, ls=":"),
                           Line2D([], [], color=ps.MUTED, lw=ps.THIN_LW)],
                  labels=["quietest first", "random order (band: range)", "busiest first",
                          "inactive units"],
                  loc="lower left", bbox_to_anchor=(0.0, 0.0), ncol=2, handlelength=1.6,
                  columnspacing=1.0, borderaxespad=0.0, borderpad=0.0)
    ps.ygrid(ax)
    return rows


def close_markers(ax, pts):
    """Print the pairs of panel b's markers whose printed centres are closer than one marker diameter.

    Args:
        ax: the axes, with its limits set; pts: list of (record, x, y) from panel_end.
    Returns:
        list of (record a, record b, distance in pt).
    """
    if len(pts) < 2:
        return []
    xy = ax.transData.transform([(x, y) for _, x, y in pts]) * 72 / ax.figure.dpi
    out = []
    for i in range(len(pts)):
        for j in range(i + 1, len(pts)):
            d = float(np.hypot(*(xy[i] - xy[j])))
            if d < ps.MARKER_PT:
                out.append((pts[i][0], pts[j][0], d))
                print(f"  !! panel b: {pts[i][0]['key']} s{pts[i][0]['seed']} and {pts[j][0]['key']} "
                      f"s{pts[j][0]['seed']} print {d:.1f} pt apart (marker {ps.MARKER_PT} pt)")
    if not out:
        print(f"  .. panel b: no two markers closer than {ps.MARKER_PT} pt ({len(pts)} markers)")
    return out


def draw(records, summ, ref, tech, ctrl):
    """Lay out the panels, draw them, print the ablation numbers and the checks, save the figure.

    Args:
        records: list from load_records; summ: arm_summaries; ref: control_reference;
        tech, ctrl: match_ablations output for the techniques and the one control drawn.
    Returns:
        str, the path of the PDF.
    """
    ps.setup()
    fig = ps.figure(FIG_H_MM)
    ax_key, ax_a, ax_b, ax_ct, ax_cb, ax_dk, ax_d = (ax_mm(fig, *LAYOUT_MM[k]) for k in
                                                    ("key", "a", "b", "c_top", "c_bot", "d_key", "d"))
    keys = [k for k in ARMS if any(r["key"] == k and r["drawn"] for r in records)]
    panel_key(ax_key, keys)
    x_max = int(max((r["probes"]["update"][-1] for r in records if r["drawn"]), default=2000))
    x_max = int(np.ceil(x_max / 500.0) * 500)
    n_a = panel_traces(ax_a, records, x_max)
    pts = panel_end(ax_b, records, ref)
    panel_fold(ax_ct, ax_cb, summ, ref)
    rows = panel_ablation(ax_d, ax_dk, tech, ctrl, ref)
    for ax, letter in ((ax_a, "a"), (ax_b, "b"), (ax_ct, "c"), (ax_d, "d")):
        ps.panel_letter(ax, letter)

    print(f"\nPanel a: {n_a} runs drawn; panel b: {len(pts)} markers")
    print("Panel d:", "no technique network ablated" if not tech else "")
    for w in rows:
        line = (f"  {w['key']} s{w['seed']}: {w['n_inactive']} of {N} inactive; success intact {w['full']:.2f}; "
                f"with the inactive units removed: quietest first {w['quiet_at_inactive']:.2f}, random "
                f"{w['random_at_inactive']:.2f}, busiest first {w['busy_at_inactive']:.2f}")
        if "k_added" in w:
            line += (f"; added units {w['k_added']:.0f} (count minus the controls' mean): quietest-first "
                     f"success {w['quiet_at_added']:.2f} with {w['k_grid']} units removed (the nearest grid point)")
        print(line)
    close_markers(ax_b, pts)
    check_overlaps(fig)
    return ps.save(fig, "fig_ed_maze_interventions")


def main():
    """Parse arguments, load the runs and the ablations, print every number, draw the figure."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="the ladder: <arm>/s<seed>/ maze run folders")
    ap.add_argument("--control-root", default=CONTROL_ROOT, help="the uniform series' run folders")
    ap.add_argument("--ablation", default=ABLATION_DIR, help="rl_maze_ablate.py JSONs of the ladder")
    ap.add_argument("--control-ablation", default=CONTROL_ABLATION_DIR,
                    help="rl_maze_ablate.py JSONs of the uniform series")
    args = ap.parse_args()
    records, stopped = load_records(args.root, args.control_root)
    ref = control_reference(records)
    print_runs(records, stopped, ref)
    summ = arm_summaries(records, stopped, ref)
    print_arms(summ)
    tech = match_ablations(load_ablation(args.ablation) if os.path.isdir(args.ablation) else [], records)
    tech = [(a, r) for a, r in tech if r["key"] != "control"]
    ctrls = [(a, r) for a, r in match_ablations(load_ablation(args.control_ablation)
                                                if os.path.isdir(args.control_ablation) else [], records)
             if r["key"] == "control" and r["pt"]["learned_i"] is not None]
    ctrl = min(ctrls, key=lambda ar: ar[1]["seed"]) if ctrls else None
    print(f"{len(tech)} technique ablation(s) under {args.ablation}; control drawn: "
          + (f"s{ctrl[1]['seed']}" if ctrl else "none"))
    draw(records, summ, ref, tech, ctrl)


if __name__ == "__main__":
    main()
