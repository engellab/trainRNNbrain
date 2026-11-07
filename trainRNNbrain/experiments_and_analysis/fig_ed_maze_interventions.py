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
      colour, as in fig_ed_maze. A thin line marks the controls' mean count (all three).
  (c) top: the fold change in active units at the end, each run's count over all three controls'
      mean, one open marker per seed (a cross for a run that did not learn) and a bar at the arm's
      mean over its finished runs; bottom: each run's end success minus all three controls' mean
      success, the same marks, with the pre-registered tolerance as a dashed line. One column per
      arm of the ladder; a column with no finished run says "unfinished", "stopped" or "no runs",
      so the layout does not change as runs finish.
  (d) the ablation (rl_maze_ablate.py) of every technique network with a JSON under --ablation,
      overlaid with one learned control's (the lowest seed with a JSON under --control-ablation):
      success rate under the training noise as units are removed quietest first (solid), busiest
      first (dotted) and in random order (dashed; band: range over the random orders), a thin
      vertical line at each network's inactive count. Without a technique JSON the panel says so.

ARMS ARE DERIVED FROM EACH RUN'S SAVED CONFIG, not from its folder name (arm_of). The per-unit
rules: trainer.lambda_frm, trainer.lambda_rws, model.sigma_w, trainer.dropout (with its drop_rate
and dropout_beta) and trainer.prune_reinit with its prune_args.reinit_mode: "rescale" keyed by the
rescaling values (the paper's arm, alpha 1.002, cap 64, protect 0.20; "rescale_mild", alpha 1.0005
and cap 8; "rescale_few", protect 0.05; any other values are "rescale_other"), "copy" is
"duplicate" (the paper's prune-and-duplicate arm), "random" is "redraw" and "zero_out" is
"zeroout" (the paper's re-initialisation arms). The settings of the uniform-interventions programme
(issue #6; read_maze_arms.py and fig_ed_maze_settings.py): trainer.weight_decay (default 1e-6,
"wd<v>"), model.input_row_norm ("inorm<v>"), model.sigma_rec (default 0.05, "srec<v>"),
model.spectral_rad (default 1.2, "srad<v>"), lambda_rws alone ("rws<v>"), model.inp_weight_cap
("cap<v>"), trainer.lambda_rwm ("rwm<v>"), trainer.lambda_met ("met<v>") and trainer.optimizer
("adamw"; "sgd<lr>" with trainer.lr). A config with more than one option on gets the options'
keys joined by "+" ("frm0.1+rws0.05", the paper's pair, is the one combination in the ladder). The
folder name is printed beside the derived arm; a folder whose name does not match its config's arm
(FOLDER_OF, the launcher's ARM_TAG) is reported. A run whose arm is not in ARMS (every setting), or
that is not an N = 1,000 ReLU run, is listed and drawn nowhere in THIS figure.

READ-OUT. A run is finished when its last logged update reaches the config's n_updates; an unfinished
run is listed with its last update and drawn nowhere. A run folder with no run files is listed as
stopped and drawn nowhere (its arm can then only come from the folder name, and the printout says
so). A run the rollback guard stopped (RuntimeError after six non-finite rollouts in a row) keeps
its last checkpoint on disk and looks unfinished; stopped_by_guard tells it apart by its records
(the last probe counts 0 active units and the last update reports a restore), and it is listed as
"stopped@<update>". The technique criterion's reference is the mean over ALL THREE of the uniform series' N = 1,000 ReLU
runs, as issue #5 wrote it before any run (technique_reference; dead-relu-paper#194 reverses entry
2026-11-05_0812, which had moved it to the two learned controls after 15 of 18 runs were read). The
settings rule keeps the learned controls (control_reference).

DATA. --root (default data/trained_RNNs_rl_maze/Maze_interventions): the ladder of issue #5 and
the settings arms of issue #6, <arm>/s<seed>/ run folders written by run_rl.py (Spock arrays
7383696, 7383697, 7383755, 7384816, 7389335 and 7389398).
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
# the three rescaling arms: the paper's values in the remedy colour, the two gentler variants of
# 5 Nov (issue #5) as lighter shades of it, mixed with white
RESCALE_RAMP = {k: matplotlib.colors.to_hex(tuple(1 - (1 - c) * w for c in matplotlib.colors.to_rgb(ps.RESCALE)))
                for k, w in (("rescale", 1.0), ("rescale_few", 0.62), ("rescale_mild", 0.36))}
# The defaults of the settings arm_of reads (configs/trainer/trainer_ppo.yaml,
# configs/model/rnn_relu_standard.yaml, the uniform series' controls): a config at these values has
# that setting "off".
DEFAULTS = dict(weight_decay=1e-6, sigma_rec=0.05, spectral_rad=1.2, optimizer="adam")

# The ladder of issue #5 (the per-unit rules of Figure 5 on the maze) in drawing order. key (as
# arm_of derives it from a saved config) -> (key-row label, colour, marker). The settings arms of
# issue #6 are not here: they are read by read_maze_arms.py and drawn by fig_ed_maze_settings.py.
ARMS = {
    "control":        ("control",                   ps.BASELINE,       "o"),
    "frm0.03":        ("rate 0.03",                 RATE_RAMP[0.03],   "o"),
    "frm0.1":         ("rate 0.1",                  RATE_RAMP[0.1],    "o"),
    "frm0.3":         ("rate 0.3",                  RATE_RAMP[0.3],    "o"),
    "frm0.1+rws0.05": ("rate 0.1 + sparsity 0.05",  ps.RATE_SPARSITY,  "P"),
    "dropout":        ("read-out dropout",          ps.DROPOUT,        "v"),
    "synnoise":       ("noisy synapses",            ps.SYNAPTIC_NOISE, "s"),
    "rescale":        ("rescaling",                 RESCALE_RAMP["rescale"],      "D"),
    "rescale_mild":   ("rescaling, slow tilt",      RESCALE_RAMP["rescale_mild"], "D"),   # Figure 5's names and order
    "rescale_few":    ("rescaling, 5% grow",        RESCALE_RAMP["rescale_few"],  "D"),
    "duplicate":      ("prune-and-duplicate",       ps.DUPLICATE,      "^"),    # Figure 5's colour and marker
}
LADDER = [k for k in ARMS if k != "control"]
# panel c's column labels: at most two lines, under 9 mm wide at TICK_PT (the column pitch)
SHORT = {"frm0.03": "rate\n0.03", "frm0.1": "rate\n0.1", "frm0.3": "rate\n0.3",
         "frm0.1+rws0.05": "rate 0.1\n+ sparsity", "dropout": "read-out\ndropout",
         "synnoise": "noisy\nsynapses", "rescale": "3BFF\nvalues", "rescale_few": "5%\ngrow",
         "rescale_mild": "slow\ntilt", "duplicate": "prune-and-\nduplicate"}
RESCALE_COLUMNS = ("rescale", "rescale_mild", "rescale_few")    # share the heading "rescaling" under their labels
# Every arm of the launcher (slurm/rl_maze_interventions_spock.slurm, ARM_TAG, in its order): the
# derived key -> the folder name. Used to report a folder whose name does not match its config's
# arm, to name the arm of a folder that holds no config, and as the arms' order in the printouts.
FOLDER_OF = {
    "control": "control", "frm0.03": "frm0.03", "frm0.1": "frm0.1", "frm0.3": "frm0.3",
    "frm0.1+rws0.05": "frm0.1_rws0.05", "synnoise": "sigmaw1.0", "dropout": "dropout0.2_beta4",
    "rescale": "rescale", "rescale_mild": "rescale_mild", "rescale_few": "rescale_few",
    "wd0": "wd0", "wd1e-5": "wd1e-5", "wd1e-4": "wd1e-4",
    "inorm0.5": "inorm0.5", "inorm2": "inorm2", "inorm5": "inorm5", "inorm20": "inorm20",
    "duplicate": "duplicate",
    "srec0": "srec0", "srec0.01": "srec0.01", "srec0.1": "srec0.1",
    "srad0.5": "srad0.5", "srad1.5": "srad1.5", "rws0.05": "rws0.05",
    "cap0.3": "cap0.3", "rwm0.05": "rwm0.05", "rwm0.2": "rwm0.2",
    "met0.01": "met0.01", "met0.1": "met0.1", "met1": "met1", "met10": "met10",
    "adamw": "adamw", "sgd0.03": "sgd0.03", "sgd0.1": "sgd0.1", "sgd0.3": "sgd0.3", "sgd1": "sgd1",
    "redraw": "redraw", "zeroout": "zeroout",
}

# layout in millimetres from the canvas's top-left corner: (left, top, width, height)
FIG_H_MM = 113.5                          # 110 + 3.5 mm for the key's third row of arms
LAYOUT_MM = {
    "key":   (0.0, 0.8, 165.1, 14.5),     # three rows of arms (four per row) and the marker row
    "a":     (12.5, 20.5, 66.0, 33.0),
    "b":     (97.0, 20.5, 63.0, 33.0),
    "c_top": (15.5, 67.5, 90.0, 19.0),    # 15.5: the lower axes' two-line y label fits left of it
    "c_bot": (15.5, 89.0, 90.0, 15.0),    # 90 mm: ten columns of 9 mm (the duplicate arm added)
    "d_key": (109.5, 60.0, 52.0, 7.0),    # the line-style key of (d), above it and off the data
    "d":     (111.5, 67.5, 48.5, 36.5),   # 111.5: its "0.0" tick label clears the last column label of (c)
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


def fmt_value(v):
    """A setting's value as the launcher's folder names write it: 1e-05 -> '1e-5', 0.0001 -> '1e-4', 2.0 -> '2'.

    Args:
        v: float.
    Returns:
        str.
    """
    if 0 < abs(v) < 1e-3:
        mant, exp = f"{v:.0e}".split("e")
        return f"{mant}e{int(exp)}"
    return f"{v:g}"


def arm_of(cfg):
    """The option(s) a saved config switches on, read from the config and not from the folder name.

    Args:
        cfg: dict, a saved run config: run_rl.py's layout (the trainer block under `trainer`) or
            the earlier one (under `ppo`); the model block under `model` in both.
    Returns:
        (key, note): key is "control" when no option is on; with one option on, the option's key
        (the module docstring lists them: "frm<lambda>", "rws<lambda>", "synnoise", "dropout",
        "rescale" / "rescale_mild" / "rescale_few" / "rescale_other" by the rescaling values,
        "duplicate", "redraw", "zeroout" or "reinit_other" by prune_args.reinit_mode, "wd<v>",
        "inorm<v>", "srec<v>", "srad<v>", "cap<v>", "rwm<v>", "met<v>", "adamw", "sgd<lr>"); with
        several on, their keys joined by "+" in that order ("frm0.1+rws0.05"); note is the option
        values as text, for the printout.
    """
    tr = cfg.get("trainer") or cfg.get("ppo") or {}
    md = cfg.get("model") or {}
    keys, parts = [], []
    frm = float(tr.get("lambda_frm") or 0.0)
    if frm > 0:
        keys.append(f"frm{fmt_value(frm)}")
        parts.append(f"lambda_frm {frm:g}")
    rws = float(tr.get("lambda_rws") or 0.0)
    if rws > 0:
        keys.append(f"rws{fmt_value(rws)}")
        parts.append(f"lambda_rws {rws:g}")
    sigma_w = float(md.get("sigma_w") or 0.0)
    if sigma_w > 0:
        keys.append("synnoise")
        parts.append(f"sigma_w {sigma_w:g}")
    if bool(tr.get("dropout", False)):
        da = tr.get("dropout_args") or {}
        keys.append("dropout")
        parts.append(f"dropout {da.get('dropout_kind', '?')} rate {da.get('drop_rate', '?')} "
                     f"beta {da.get('dropout_beta', '?')}")
    if bool(tr.get("prune_reinit", False)):
        pa = tr.get("prune_args") or {}
        mode = str(pa.get("reinit_mode", "rescale"))
        if mode == "rescale":
            alpha, cap = float(pa.get("rescale_alpha", 0)), float(pa.get("rescale_cap", 0))
            protect = float(pa.get("rescale_protect_frac", 0))
            keys.append({(1.002, 64.0, 0.2): "rescale", (1.0005, 8.0, 0.2): "rescale_mild",
                         (1.002, 64.0, 0.05): "rescale_few"}.get((alpha, cap, protect), "rescale_other"))
            parts.append(f"rescaling alpha {alpha:g} cap {cap:g} protect {protect:g}")
        else:
            keys.append({"copy": "duplicate", "random": "redraw", "zero_out": "zeroout"}.get(mode, "reinit_other"))
            parts.append(f"prune_reinit {mode} rate {pa.get('max_replace_frac', '?')} maturity "
                         f"{pa.get('maturity', '?')} utility {pa.get('utility', '?')} every "
                         f"{pa.get('check_every', '?')} patience {pa.get('patience', '?')}")
    wd = float(tr.get("weight_decay", DEFAULTS["weight_decay"]) or 0.0)
    if wd != DEFAULTS["weight_decay"]:
        keys.append(f"wd{fmt_value(wd)}")
        parts.append(f"weight_decay {wd:g}")
    inorm = md.get("input_row_norm")
    if inorm is not None:
        keys.append(f"inorm{fmt_value(float(inorm))}")
        parts.append(f"input_row_norm {float(inorm):g}")
    srec = float(md.get("sigma_rec", DEFAULTS["sigma_rec"]) or 0.0)
    if srec != DEFAULTS["sigma_rec"]:
        keys.append(f"srec{fmt_value(srec)}")
        parts.append(f"sigma_rec {srec:g}")
    srad = float(md.get("spectral_rad", DEFAULTS["spectral_rad"]))
    if srad != DEFAULTS["spectral_rad"]:
        keys.append(f"srad{fmt_value(srad)}")
        parts.append(f"spectral_rad {srad:g}")
    cap = md.get("inp_weight_cap")
    if cap is not None:
        keys.append(f"cap{fmt_value(float(cap))}")
        parts.append(f"inp_weight_cap {float(cap):g}")
    rwm = float(tr.get("lambda_rwm") or 0.0)
    if rwm > 0:
        keys.append(f"rwm{fmt_value(rwm)}")
        parts.append(f"lambda_rwm {rwm:g}")
    met = float(tr.get("lambda_met") or 0.0)
    if met > 0:
        keys.append(f"met{fmt_value(met)}")
        parts.append(f"lambda_met {met:g}")
    opt = str(tr.get("optimizer") or DEFAULTS["optimizer"]).lower()
    if opt != DEFAULTS["optimizer"]:
        lr = float(tr.get("lr", 0.0))
        keys.append(opt if opt != "sgd" else f"sgd{fmt_value(lr)}")
        parts.append(f"optimizer {opt} lr {lr:g}" + (f" momentum {tr.get('momentum', 0)}" if opt == "sgd" else ""))
    return "+".join(keys) or "control", ", ".join(parts) or "no option on"


def stopped_by_guard(run):
    """Whether the rollback guard stopped a run before its budget, read from the run's own records.

    The guard restores the last checkpoint when a rollout is not finite and raises after six
    restores in a row (TrainerPPO); the folder then keeps the records up to the last update. Such a
    run's last probe counts 0 active units, because the network's rates are not finite; a run
    still training has a finite network at its last probe. Checked against the Spock logs of the
    six runs stopped so far (MazeInterv.7384816_25, 7389335_46-48, 7389335_50-51): all six end
    on a probe of 0 active units (five of them with the restore count in their last update
    record, one, rescale_mild seed 1, with the restore between its last record and the raise).
    The caller applies it to unfinished runs only.

    Args:
        run: dict from common.maze_load_runs.
    Returns:
        bool.
    """
    return int(run["probes"]["active"][-1]) == 0


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
            rec["stopped_at"] = rec["last_update"] if (not rec["finished"] and stopped_by_guard(run)) else None
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
            if source == "ladder" and rec["folder_arm"] != FOLDER_OF.get(key, key):
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
        dict with active (mean count at the end over the learned controls), active_min and
        active_max (the smallest and largest learned control's count), success (their mean end
        success), n (how many), all_active, all_active_min, all_active_max, all_success, n_all (the
        same over every finished control); or None without a learned control.
    """
    ctrl = [r for r in records if r["key"] == "control" and r["drawn"]]
    learned = [r for r in ctrl if r["pt"]["learned_i"] is not None]
    if not learned:
        return None
    act = np.array([r["pt"]["active_end"] for r in learned], float)
    suc = np.array([r["pt"]["success_end"] for r in learned], float)
    return dict(active=float(act.mean()), active_min=int(act.min()), active_max=int(act.max()),
                success=float(suc.mean()), n=len(learned),
                all_active=float(np.mean([r["pt"]["active_end"] for r in ctrl])),
                all_active_min=int(min(r["pt"]["active_end"] for r in ctrl)),
                all_active_max=int(max(r["pt"]["active_end"] for r in ctrl)),
                all_success=float(np.mean([r["pt"]["success_end"] for r in ctrl])), n_all=len(ctrl))


def technique_reference(ref):
    """The reference of the technique criterion: all three controls, as issue #5 wrote it before any run.

    Entry 2026-11-05_0812 moved this reference to the two learned controls after 15 of 18 runs had
    been read; dead-relu-paper#194 reversed that. The settings rule keeps the learned controls
    (control_reference), and read_maze_arms and the technique panels take this dict instead.

    Args:
        ref: control_reference output, or None.
    Returns:
        dict with control_reference's keys active, active_min, active_max, success and n, computed
        over every finished control; or None when ref is None.
    """
    if ref is None:
        return None
    return dict(active=ref["all_active"], active_min=ref["all_active_min"], active_max=ref["all_active_max"],
                success=ref["all_success"], n=ref["n_all"])


def arm_status(records, stopped, key):
    """One arm's runs and the state of its column.

    Args:
        records: list from load_records; stopped: list from load_records; key: an arm key.
    Returns:
        (fin, status): fin is the arm's finished records whose `why` is empty; status is
        "finished" (at least one finished run), "unfinished" (none finished, one still training),
        "stopped" (none finished or training, one stopped by the guard or a folder with no run
        files) or "no runs".
    """
    mine = [r for r in records if r["key"] == key and r["why"] == ""]
    fin = [r for r in mine if r["finished"]]
    n_stopped = (sum(s["folder_arm"] == FOLDER_OF.get(key, key) for s in stopped)
                 + sum(r["stopped_at"] is not None for r in mine))
    running = [r for r in mine if not r["finished"] and r["stopped_at"] is None]
    return fin, ("finished" if fin else "unfinished" if running else "stopped" if n_stopped
                 else "no runs")


def rule_verdict(fin, status, ref):
    """The pre-registered verdict on a per-unit rule (issue #5): "works", "fails: ...", or none yet.

    A rule works at a value when every one of its SEEDS runs finished, each run's count of active
    units at the end exceeds every control's, the arm's mean end success is within SUCCESS_TOL of
    the mean of all three controls, and no run ends below MAZE_LEARNED_SUCCESS (issue #5: such an
    arm "counts against the technique at that value"; dead-relu-paper#194).

    Args:
        fin: the arm's finished records; status: str from arm_status; ref: technique_reference.
    Returns:
        str.
    """
    if ref is None:
        return "no verdict (no learned control)"
    if not fin:
        return f"no verdict ({status})"
    if len(fin) < len(SEEDS):
        return f"incomplete ({len(fin)} of {len(SEEDS)} runs finished)"
    low = [r["seed"] for r in fin if r["pt"]["active_end"] <= ref["active_max"]]
    cost = float(np.mean([r["pt"]["success_end"] for r in fin])) - ref["success"]
    reasons = []
    if low:
        reasons.append(f"count not above every control's ({ref['active_max']}) in "
                       f"seed{'s' if len(low) > 1 else ''} {', '.join(map(str, low))}")
    if cost < -SUCCESS_TOL:
        reasons.append(f"success cost {cost:+.3f} beyond {SUCCESS_TOL}")
    below = [r["seed"] for r in fin if r["pt"]["success_end"] < MAZE_LEARNED_SUCCESS]
    if below:
        reasons.append(f"seed{'s' if len(below) > 1 else ''} {', '.join(map(str, below))} "
                       f"ending below {MAZE_LEARNED_SUCCESS:g}")
    return "works" if not reasons else "fails: " + "; ".join(reasons)


def arm_summaries(records, stopped, ref):
    """Per ladder arm: its finished runs, fold changes, success cost, status and verdict.

    Args:
        records: list from load_records; stopped: list from load_records; ref: technique_reference.
    Returns:
        dict key -> dict with runs (the finished, drawn records), fold (array, one per run: count at
        the end over the controls' mean; empty without a reference), cost (array: end success
        minus the controls' mean success), mean_fold, mean_cost (floats or None), status
        ("finished", "unfinished", "stopped" or "no runs": the state of the arm's column) and
        verdict (str: "works", "fails: ...", "incomplete ..." or "no verdict ...").
    """
    out = {}
    for key in LADDER:
        fin, status = arm_status(records, stopped, key)
        s = dict(runs=fin, status=status, fold=np.array([]), cost=np.array([]), mean_fold=None,
                 mean_cost=None, verdict=rule_verdict(fin, status, ref))
        if fin and ref is not None:
            s["fold"] = np.array([r["pt"]["active_end"] / ref["active"] for r in fin])
            s["cost"] = np.array([r["pt"]["success_end"] - ref["success"] for r in fin])
            s["mean_fold"], s["mean_cost"] = float(s["fold"].mean()), float(s["cost"].mean())
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
    order = list(FOLDER_OF)
    for r in sorted(records, key=lambda r: (r["source"] != "control", order.index(r["key"])
                                            if r["key"] in order else 99, r["key"], r["seed"])):
        p = r.get("pt")
        status = ("finished" if r["finished"] else f"stopped@{r['stopped_at']}"
                  if r["stopped_at"] is not None else "unfinished")
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
        (k, s["seed"]) for s in stopped for k in ARMS if FOLDER_OF[k] == s["folder_arm"]}
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
          f"{'succ cost':>9}  verdict (count above every control's in all seeds, no seed below 0.9, AND "
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
    _, col, marker = ARMS[rec["key"]]
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
        # at most four entries per row (three rows for the eleven arms): the long labels would
        # overrun the page width at five per row
        leg = ax.legend(handles=h, loc="upper center", ncol=min(len(h), 4), frameon=False,
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
        _, col, marker = ARMS[r["key"]]
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
        ax: axes; records: list from load_records; ref: technique_reference (a thin line at its
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
            ref: technique_reference.
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
    ax_b.text(-0.55, -SUCCESS_TOL - 0.02, f"$-${SUCCESS_TOL:g}", fontsize=ps.TICK_PT,
              color=ps.MUTED, ha="left", va="top")
    ax_b.set_yticks(np.arange(0.0, lo, -0.2))
    ax_b.set_ylabel("success minus\ncontrol mean")
    ax_b.set_xticks(range(len(LADDER)))
    ax_b.set_xticklabels([SHORT[k] for k in LADDER], fontsize=ps.TICK_PT, linespacing=1.25)
    ax_b.tick_params(axis="x", length=0, pad=3)
    # one heading under the three rescaling columns (review of PR #193, item 12)
    cols = [LADDER.index(k) for k in RESCALE_COLUMNS]
    head_y_mm = -7.0                                    # below the two-line tick labels
    height_mm = ax_b.get_position().height * ax_b.figure.get_figheight() / ps.MM
    y_rule = head_y_mm / height_mm
    ax_b.plot([min(cols) - 0.4, max(cols) + 0.4], [y_rule, y_rule], transform=ax_b.get_xaxis_transform(),
              color=ps.MUTED, lw=ps.THIN_LW, clip_on=False, solid_capstyle="butt")
    ax_b.text(float(np.mean(cols)), y_rule - 0.4 / height_mm, "rescaling", transform=ax_b.get_xaxis_transform(),
              fontsize=ps.TICK_PT, color=ps.INK, ha="center", va="top")
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
            control drawn; ref: technique_reference (sets the "added units" of each technique
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
           ylabel="success rate\n(training noise)")
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
        records: list from load_records; summ: arm_summaries; ref: technique_reference;
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
    tref = technique_reference(ref)
    summ = arm_summaries(records, stopped, tref)
    print_arms(summ)
    tech = match_ablations(load_ablation(args.ablation) if os.path.isdir(args.ablation) else [], records)
    tech = [(a, r) for a, r in tech if r["key"] != "control"]
    ctrls = [(a, r) for a, r in match_ablations(load_ablation(args.control_ablation)
                                                if os.path.isdir(args.control_ablation) else [], records)
             if r["key"] == "control" and r["pt"]["learned_i"] is not None]
    ctrl = min(ctrls, key=lambda ar: ar[1]["seed"]) if ctrls else None
    print(f"{len(tech)} technique ablation(s) under {args.ablation}; control drawn: "
          + (f"s{ctrl[1]['seed']}" if ctrl else "none"))
    draw(records, summ, tref, tech, ctrl)


if __name__ == "__main__":
    main()
