#!/usr/bin/env python3
"""Read-out of every maze arm of the uniform-interventions programme (issue engellab/trainRNNbrain#6).

QUESTION. Pavel's instruction of 5 Nov: report the interventions that work on every task first. On
the maze (PPO, N = 1,000, 2,000 updates) every per-unit rule of Figure 5 and every training setting
of Figure 3 was run at the values the two supervised tasks used (docs/interventions_matrix.md; the
launcher slurm/rl_maze_interventions_spock.slurm). This script reads all of them against the
uniform series' N = 1,000 ReLU controls and gives each arm its pre-registered verdict.

THE TWO VERDICT CLASSES (pre-registered: issue #5 for the rules; the notebook entries of 5 Nov,
2026-11-05_2310 and docs/interventions_matrix.md, for the settings), both read at the END OF
TRAINING, update 2,000:
  rule (section B of the matrix: the rate penalty, rate + sparsity, read-out dropout, noisy
      synapses, rescaling, prune-and-duplicate, re-initialisation by redraw or zero-out), against
      ALL THREE controls as issue #5 wrote it (dead-relu-paper#194, reversing entry
      2026-11-05_0812; fig_ed_maze_interventions.technique_reference): "works" when every seed's
      count of active units at the end exceeds every control's (449), the arm's mean end success
      is within SUCCESS_TOL = 0.05 of the controls' mean (0.898), and no seed ends below 0.9;
      otherwise "fails", with the reason (fig_ed_maze_interventions.rule_verdict);
  setting, against the LEARNED controls (two of the three: seed 3 ended at a success of 0.73),
      (section A: weight decay, input row norm, activity noise, spectral radius, the sparsity
      penalty alone, the input-weight cap, the recurrent-weight magnitude and metabolic penalties,
      the optimiser): "increases the count" when the counts of every seed THAT LEARNED THE TASK
      (common.maze_learned_index) lie above the learned controls' range (426 to 449), "decreases
      the count" when they all lie below it, "no change" otherwise, and "no direction" when no
      seed learned (dead-relu-paper PR #188 review, item 1: a direction comes only from networks
      that reached the task's criterion, as the controls' reference does); reported with the arm's
      mean end success over all finished seeds minus the controls' mean and how many seeds
      learned.
A verdict needs all SEEDS runs finished; an arm with fewer is "incomplete", with the number
finished, still training and stopped. The "learned" point of every run is reported beside the
end point (the count there), as the maze entries do.

ARMS ARE DERIVED FROM EACH RUN'S SAVED CONFIG (fig_ed_maze_interventions.arm_of), never from the
folder name; the folder is printed beside the derived arm and a mismatch is reported by
load_records. A run the rollback guard stopped is "stopped@<update>" (stopped_by_guard: its last
probe counts 0 active units); a run below its budget otherwise is "unfinished" at its last update
and is picked up on a rerun once its folder is copied again; an arm of the launcher with no folder
is "absent".

OUTPUT. On stdout, table 1: one row per run (class, arm, folder, seed, status, last update,
success at the end, the learned update, active units at the learned point and at the end, the
fold change over the class's controls' mean count, the option values); the controls' rows first;
then the folders that hold no run files. Table 2: one row per arm of the launcher (class, arm,
status, runs finished / training / stopped, active units at the end per seed, mean fold change,
success minus the controls' mean, seeds learned, verdict), the controls first. The same numbers
go to --json (default data/read_maze_arms.json): reference (learned controls), reference_techniques (all three), runs, arms, roots, written.

Run:  python trainRNNbrain/experiments_and_analysis/read_maze_arms.py [--root DIR]
      [--control-root DIR] [--json PATH]
"""
import argparse
import json
import os
from datetime import datetime

import numpy as np

from fig_ed_maze_interventions import (CONTROL_ROOT, FOLDER_OF, REPO, ROOT, SEEDS, SUCCESS_TOL,
                                       control_reference, load_records, rule_verdict, technique_reference)

JSON_OUT = os.path.join(REPO, "data", "read_maze_arms.json")
# the per-unit rules (section B of docs/interventions_matrix.md), by the prefix of their derived key;
# every other key is a setting (section A)
RULE_PREFIXES = ("frm", "dropout", "synnoise", "rescale", "duplicate", "redraw", "zeroout")


def arm_class(key):
    """The verdict class of an arm key: "control", "rule" or "setting" (the module docstring).

    Args:
        key: str, a key from fig_ed_maze_interventions.arm_of.
    Returns:
        str.
    """
    if key == "control":
        return "control"
    return "rule" if key.startswith(RULE_PREFIXES) else "setting"


def run_status(rec):
    """A run's status word: "finished", "stopped@<update>" or "unfinished".

    Args:
        rec: a load_records record.
    Returns:
        str.
    """
    if rec["finished"]:
        return "finished"
    if rec["stopped_at"] is not None:
        return f"stopped@{rec['stopped_at']}"
    return "unfinished"


def setting_verdict(fin, ref):
    """The pre-registered verdict on a setting: where the counts of the seeds that learned lie against the learned controls' range.

    Only the seeds that learned the task (a learned point) give a direction; a seed that did not
    learn is listed but enters no direction, and an arm with no learned seed has none.

    Args:
        fin: the arm's finished records (all SEEDS of them); ref: control_reference.
    Returns:
        str: "increases the count", "decreases the count", "no change in the count" or "no
        direction", followed by the learned seeds' counts against the range (the other seeds'
        counts marked "not learned"), the success difference over all finished seeds and how many
        seeds learned.
    """
    learned_counts = [r["pt"]["active_end"] for r in fin if r["pt"]["learned_i"] is not None]
    shown = [f"{r['pt']['active_end']}" + ("" if r["pt"]["learned_i"] is not None else " not learned") for r in fin]
    if not learned_counts:
        word = "no direction"
    elif all(c > ref["active_max"] for c in learned_counts):
        word = "increases the count"
    elif all(c < ref["active_min"] for c in learned_counts):
        word = "decreases the count"
    else:
        word = "no change in the count"
    diff = float(np.mean([r["pt"]["success_end"] for r in fin])) - ref["success"]
    n_learned = len(learned_counts)
    learned = ("all seeds learned" if n_learned == len(fin) else "no seed learned" if n_learned == 0
               else f"{n_learned} of {len(fin)} seeds learned")
    return (f"{word} ({', '.join(shown)} against the controls' {ref['active_min']}-"
            f"{ref['active_max']}); success {diff:+.3f} from the controls' mean; {learned}")


def arm_rows(records, stopped, ref):
    """One summary per arm of the launcher (FOLDER_OF order), plus any arm found only in a config.

    Args:
        records: list from load_records; stopped: list from load_records (folders with no run
            files); ref: control_reference or None.
    Returns:
        list of dicts: key, cls, folder, status ("finished", "incomplete", "unfinished",
        "stopped", "absent"), n_finished, n_running, n_stopped, seeds_finished, active_end (per
        finished run, by seed), learned (per finished run), folds, mean_fold, success_end (per
        finished run), success_diff, n_learned, verdict.
    """
    keys = list(FOLDER_OF) + sorted({r["key"] for r in records} - set(FOLDER_OF))
    out = []
    for key in keys:
        if key == "control":
            mine = [r for r in records if r["key"] == key and r["source"] == "control"]
        else:
            mine = [r for r in records if r["key"] == key and r["source"] == "ladder"]
        fin = sorted([r for r in mine if r["finished"]], key=lambda r: r["seed"])
        running = [r for r in mine if not r["finished"] and r["stopped_at"] is None]
        n_stopped = (sum(r["stopped_at"] is not None for r in mine)
                     + sum(s["folder_arm"] == FOLDER_OF.get(key, key) for s in stopped))
        if not mine and n_stopped == 0:
            status = "absent"
        elif len(fin) == len(SEEDS):
            status = "finished"
        elif fin:
            status = "incomplete"
        elif running:
            status = "unfinished"
        else:
            status = "stopped"
        row = dict(key=key, cls=arm_class(key), folder=FOLDER_OF.get(key, key), status=status,
                   n_finished=len(fin), n_running=len(running), n_stopped=n_stopped,
                   seeds_finished=[r["seed"] for r in fin],
                   active_end=[r["pt"]["active_end"] for r in fin],
                   learned=[r["pt"]["learned"] for r in fin],
                   success_end=[r["pt"]["success_end"] for r in fin],
                   n_learned=sum(r["pt"]["learned_i"] is not None for r in fin),
                   folds=[], mean_fold=None, success_diff=None)
        own = technique_reference(ref) if row["cls"] == "rule" else ref    # the class's controls
        if fin and own is not None:
            row["folds"] = [r["pt"]["active_end"] / own["active"] for r in fin]
            row["mean_fold"] = float(np.mean(row["folds"]))
            row["success_diff"] = float(np.mean(row["success_end"])) - own["success"]
        if key == "control":
            row["verdict"] = ("reference" if ref is None else
                              f"reference: {ref['n']} learned controls, mean {ref['active']:.1f} active "
                              f"({ref['active_min']}-{ref['active_max']}), success {ref['success']:.3f}")
        elif ref is None:
            row["verdict"] = "no verdict (no learned control)"
        elif status in ("absent", "unfinished", "stopped"):
            row["verdict"] = f"no verdict ({status})"
        elif status == "incomplete":
            parts = [f"{len(fin)} of {len(SEEDS)} finished"]
            if running:
                parts.append(f"{len(running)} training")
            if n_stopped:
                parts.append(f"{n_stopped} stopped")
            row["verdict"] = "incomplete (" + ", ".join(parts) + ")"
        elif row["cls"] == "rule":
            row["verdict"] = rule_verdict(fin, status, own)
        else:
            row["verdict"] = setting_verdict(fin, ref)
        out.append(row)
    return out


def print_runs(records, stopped, ref):
    """Table 1: one row per run, the controls first, then the folders with no run files.

    Args:
        records: list from load_records; stopped: list from load_records; ref: control_reference.
    Returns:
        None.
    """
    fmt = lambda v: "-" if v is None else str(v)
    order = list(FOLDER_OF)
    print(f"{'class':>7} {'arm (from config)':>17} {'folder':>17} {'seed':>4} {'status':>12} {'update':>6} "
          f"{'succ@end':>8} {'learned':>7} {'act@learn':>9} {'act@end':>7} {'fold':>5}  note")
    for r in sorted(records, key=lambda r: (r["source"] != "control",
                                            order.index(r["key"]) if r["key"] in order else 99,
                                            r["key"], r["seed"])):
        p = r.get("pt")
        fold = "-" if (p is None or ref is None) else f"{p['active_end'] / ref['active']:.2f}"
        folder = "series" if r["source"] == "control" else r["folder_arm"]
        print(f"{arm_class(r['key']):>7} {r['key']:>17} {folder:>17} {r['seed']:>4} {run_status(r):>12} "
              f"{r['last_update']:>6} "
              + (f"{p['success_end']:>8.2f} {fmt(p['learned']):>7} {fmt(p['active_learned']):>9} "
                 f"{p['active_end']:>7} {fold:>5}" if p else f"{'-':>8} {'-':>7} {'-':>9} {'-':>7} {'-':>5}")
              + f"  {r['note']}")
    for s in stopped:
        print(f"{'?':>7} {'?':>17} {s['folder_arm']:>17} {fmt(s['seed']):>4} {'stopped':>12} {'-':>6} "
              f"{'-':>8} {'-':>7} {'-':>9} {'-':>7} {'-':>5}  no run files on disk; the arm "
              f"'{s['folder_arm']}' is the folder's name, not a config's")
    if ref is None:
        print("controls: no learned control, so no reference, no fold change and no verdict")
    else:
        print(f"controls' reference: {ref['n']} learned controls, active at the end mean "
              f"{ref['active']:.1f} (range {ref['active_min']}-{ref['active_max']}), end success mean "
              f"{ref['success']:.3f}; over all {ref['n_all']} controls: active {ref['all_active']:.1f}, "
              f"success {ref['all_success']:.3f}")


def print_arms(rows):
    """Table 2: one row per arm with its verdict.

    Args:
        rows: list from arm_rows.
    Returns:
        None.
    """
    print(f"\n{'class':>7} {'arm':>15} {'status':>10} {'fin/run/stop':>12} {'active at the end':>18} "
          f"{'mean fold':>9} {'succ diff':>9} {'learned':>7}  verdict (rule, all three controls: count above every "
          f"control's in all seeds, no seed below 0.9 AND mean success within {SUCCESS_TOL} of the controls'; setting: all "
          f"seeds outside the learned controls' range on one side)")
    for w in rows:
        counts = ", ".join(map(str, w["active_end"])) or "-"
        mf = "-" if w["mean_fold"] is None else f"{w['mean_fold']:.2f}"
        sd = "-" if w["success_diff"] is None else f"{w['success_diff']:+.3f}"
        learned = f"{w['n_learned']}/{w['n_finished']}" if w["n_finished"] else "-"
        print(f"{w['cls']:>7} {w['key']:>15} {w['status']:>10} "
              f"{w['n_finished']}/{w['n_running']}/{w['n_stopped']:<8} {counts:>18} {mf:>9} {sd:>9} "
              f"{learned:>7}  {w['verdict']}")


def main():
    """Parse arguments, load every run, print both tables, write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=ROOT, help="the arms: <arm>/s<seed>/ maze run folders")
    ap.add_argument("--control-root", default=CONTROL_ROOT, help="the uniform series' run folders")
    ap.add_argument("--json", default=JSON_OUT, help="where the tables are written")
    args = ap.parse_args()
    records, stopped = load_records(args.root, args.control_root)
    ref = control_reference(records)
    print_runs(records, stopped, ref)
    rows = arm_rows(records, stopped, ref)
    print_arms(rows)
    runs = []
    for r in records:
        p = r.get("pt") or {}
        runs.append(dict(key=r["key"], cls=arm_class(r["key"]), folder=r["rel"], seed=r["seed"],
                         status=run_status(r), last_update=r["last_update"], n_updates=r["n_updates"],
                         success_end=p.get("success_end"), learned=p.get("learned"),
                         active_learned=p.get("active_learned"), active_end=p.get("active_end"),
                         fold=(None if not p or ref is None else p["active_end"]
                               / (technique_reference(ref) if arm_class(r["key"]) == "rule" else ref)["active"]),
                         note=r["note"]))
    out = dict(written=datetime.now().isoformat(timespec="minutes"), root=os.path.relpath(args.root, REPO),
               control_root=os.path.relpath(args.control_root, REPO), success_tol=SUCCESS_TOL,
               reference=ref, reference_techniques=technique_reference(ref), runs=runs, arms=rows,
               no_run_files=[dict(folder=s["rel"], seed=s["seed"]) for s in stopped])
    os.makedirs(os.path.dirname(args.json), exist_ok=True)
    with open(args.json, "w") as fh:
        json.dump(out, fh, indent=1)
    print(f"\nwrote {os.path.relpath(args.json, REPO)}")


if __name__ == "__main__":
    main()
