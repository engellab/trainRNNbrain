#!/usr/bin/env python3
"""Round 19 read-out: every Maze network trained in 2,000-update stages, read by the committed rules.

RULES (docs/project_trajectory/): 2026-11-07_1008_round19-train-all-rules.md (the stage protocol, the size
series and the N = 1,000 group, the pilot at 2,000 updates), 2026-11-07_1111_round19-settings-stage2-rules.md
(the settings and techniques at 2,000 and 4,000 updates) and 2026-11-07_1723_round19-pilot-stage2-rules.md (the
two pilot conditions at 4,000 updates). Tracking: ptolmachev/dead-relu-paper#201.

LEARNED AT UPDATE U. common.maze_learned_index applied to a network's joined record (common.maze_load_runs)
cut at update U, provided the record reaches U; a record that ends before U (a run stopped by the rollback
guard) is not learned at U. The learned update is the probe update of that index.

PARTS (all from the rsynced trees under --root):
  suite     every Maze_suite run: stages trained, the last update, learned at each stage end, the learned
            update, the active count at the learned probe and at the end, success at the end.
  settings  every arm of Maze_interventions (the arm from the saved config, fig_ed_maze_interventions.arm_of):
            per seed, success, active count and learned or not at updates 2,000 and 4,000. At 4,000, for every
            arm with a continued seed: the settings direction rule (learned seeds against the range of the
            controls learned at 4,000) or the technique criterion (every seed above every one of the six
            controls, mean success within 0.05 of theirs, no seed below 0.9); an arm with a seed stopped
            before 4,000 is "incomplete". The labels of the part 2 rules ("learns more slowly", ...). The
            2,000 verdicts are quoted from data/round18_training/round18_settings.json (settings) and
            data/read_maze_arms.json (techniques, by arm folder; None for the round 18 parity arms, which that
            file predates), not recomputed.
  pilot     every Maze_r19pilot run: learned at 2,000 and, for the continued ones, at 4,000; the pass rule.

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/round19_read.py [--root data/trained_RNNs_rl_maze] [--out FILE]
Output: data/round19/round19_results.json and the printed tables.
"""
import argparse
import json
import os
import sys

import numpy as np
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
sys.path.insert(0, HERE)
from trainRNNbrain.experiments_and_analysis.common import maze_load_runs, maze_learned_index   # noqa: E402
from fig_ed_maze_interventions import arm_of                                                    # noqa: E402
from read_maze_arms import arm_class                                                            # noqa: E402

OUT = os.path.join(REPO, "data", "round19", "round19_results.json")
SUCCESS_TOL = 0.05


def last_update(run):
    """The last PPO update a joined record holds.

    Args:
        run: dict from common.maze_load_runs.
    Returns:
        int.
    """
    return int(run["progress"][-1]["update"])


def budget(run):
    """The update at which the chain's last run was due to end: its iter_offset plus its n_updates.

    Args:
        run: dict from common.maze_load_runs.
    Returns:
        int.
    """
    cfg = yaml.safe_load(open([os.path.join(run["folder"], f) for f in os.listdir(run["folder"])
                               if f.endswith("config.yaml")][0]))
    tr = cfg.get("trainer") or cfg.get("ppo")
    off = tr.get("iter_offset", 0)
    return int(tr["n_updates"]) + (int(off) if isinstance(off, (int, float)) else 0)


def at(run, U):
    """A network's state at update U from its joined record.

    Args:
        run: dict from common.maze_load_runs; U: int, an update.
    Returns:
        dict: reached (bool, the record extends to U), learned (bool), learned_update (int or None),
        active_learned (int or None), success and active at the last probe at or before U (None if not
        reached).
    """
    pu = np.asarray(run["probes"]["update"])
    if last_update(run) < U or pu.max() < U:
        return dict(reached=False, learned=False, learned_update=None, active_learned=None,
                    success=None, active=None)
    m = pu <= U
    ps = np.asarray(run["probes"]["probe_success"])[m]
    act = np.asarray(run["probes"]["active"])[m]
    li = maze_learned_index(ps)
    return dict(reached=True, learned=li is not None,
                learned_update=None if li is None else int(pu[m][li]),
                active_learned=None if li is None else int(act[li]),
                success=float(ps[-1]), active=int(act[-1]))


def read_suite(root):
    """Every Maze_suite network: stages trained and learned state at each stage end.

    Args:
        root: str, the Maze_suite folder.
    Returns:
        list of dicts, one per network.
    """
    rows = []
    for r in maze_load_runs(root):
        end = last_update(r)
        stage_ends = [u for u in (2000, 4000, 6000, 8000) if u <= end]
        full = at(r, end)
        rows.append(dict(N=int(r["N"]), act=r["act"], seed=int(r["seed"]), n_runs=len(r["chain"]), end=end,
                         budget=budget(r), learned_at={u: at(r, u)["learned"] for u in stage_ends},
                         learned_update=full["learned_update"], active_learned=full["active_learned"],
                         success_end=full["success"], active_end=full["active"]))
    return sorted(rows, key=lambda x: (x["act"], x["N"], x["seed"]))


def label(n_learned_2000, n_learned_4000, n_seeds):
    """The part 2 label of an arm from its learned counts at 2,000 and 4,000 updates.

    Args:
        n_learned_2000, n_learned_4000: int, seeds learned at each point; n_seeds: int.
    Returns:
        str.
    """
    if n_learned_4000 == 0:
        return "does not learn within 4,000 updates"
    if n_learned_4000 < n_seeds:
        return f"{n_learned_4000} of {n_seeds} seeds learn within 4,000 updates"
    late = n_seeds - n_learned_2000
    if late >= 2:
        return "learns more slowly"
    if late == 1:
        return "learns within 4,000 updates; one seed after update 2,000, as one of the six controls"
    return "all seeds learned by update 2,000"


def read_settings(root, suite_root):
    """Every arm at updates 2,000 and 4,000, and the 4,000 verdicts and labels.

    Args:
        root: str, the Maze_interventions folder; suite_root: str, the Maze_suite folder (the controls).
    Returns:
        dict: controls (rows and the 4,000 references), arms (one dict per arm).
    """
    ctrl = [r for r in maze_load_runs(suite_root) if r["N"] == 1000 and r["act"] == "relu" and not r["cue"]]
    crow = [dict(seed=int(r["seed"]), at2000=at(r, 2000), at4000=at(r, 4000)) for r in ctrl]
    learned4 = [c["at4000"]["active"] for c in crow if c["at4000"]["learned"]]
    ref = dict(n=len(crow), n_learned_4000=len(learned4),
               learned_range_4000=[min(learned4), max(learned4)] if learned4 else None,
               all_max_4000=max(c["at4000"]["active"] for c in crow if c["at4000"]["reached"]),
               mean_success_4000=float(np.mean([c["at4000"]["success"] for c in crow if c["at4000"]["reached"]])))
    s2000 = {x["key"]: x["new_text"] for x in json.load(open(os.path.join(REPO, "data", "round18_training",
                                                                          "round18_settings.json")))["rows"]}
    # technique verdicts keyed on the arm (the launcher's ARM_TAG, read_maze_arms' "folder"), never on the class
    # key: a class holds several arms (dropout0.2_beta4 and dropout0.05_beta4 are both "dropout"), and arms added
    # after read_maze_arms.json was written (the round 18 parity arms) have no verdict there (entry 2026-11-10_2018)
    tarms = json.load(open(os.path.join(REPO, "data", "read_maze_arms.json")))["arms"]
    t2000 = {x["folder"]: x["verdict"] for x in tarms}
    assert len(t2000) == len(tarms), "two read_maze_arms.json entries share one arm folder"
    arms = {}
    for r in maze_load_runs(root):
        cfg = yaml.safe_load(open([os.path.join(r["folder"], f) for f in os.listdir(r["folder"])
                                   if f.endswith("config.yaml")][0]))
        # the arm is the stage-1 folder's name (the launcher's ARM_TAG): arm_of folds every dropout rate into
        # "dropout" and every sigma_w into "synnoise", so it gives the class, not the arm
        key = os.path.basename(os.path.dirname(r["chain"][0]))
        cls_key, _ = arm_of(cfg)
        cont = len(r["chain"]) > 1
        arms.setdefault((key, cls_key), []).append(dict(seed=int(r["seed"]), continued=cont, end=last_update(r),
                                             budget=budget(r), at2000=at(r, 2000), at4000=at(r, 4000)))
    out = []
    for (key, cls_key), seeds in sorted(arms.items()):
        seeds = sorted(seeds, key=lambda x: x["seed"])
        cls = arm_class(cls_key)
        row = dict(key=key, cls=cls, seeds=seeds,
                   verdict_2000=s2000.get(key) if cls == "setting" else t2000.get(key))
        if any(s["continued"] for s in seeds):
            n2 = sum(s["at2000"]["learned"] for s in seeds)
            n4 = sum(s["at4000"]["learned"] for s in seeds)
            stopped = [s["seed"] for s in seeds if not s["at4000"]["reached"]]
            row["label"] = label(n2, n4, len(seeds))
            row["stopped_before_4000"] = stopped
            if stopped and len(stopped) == len(seeds):
                row["verdict_4000"] = "no verdict (every seed stopped)"
            elif stopped:
                row["verdict_4000"] = f"incomplete ({len(seeds) - len(stopped)} of {len(seeds)} reach 4,000; seeds {stopped} stopped)"
            elif cls == "setting":
                lr = ref["learned_range_4000"]
                acts = [s["at4000"]["active"] for s in seeds if s["at4000"]["learned"]]
                if not acts:
                    row["verdict_4000"] = "no direction (no seed learned)"
                elif all(a > lr[1] for a in acts):
                    row["verdict_4000"] = "increases the count"
                elif all(a < lr[0] for a in acts):
                    row["verdict_4000"] = "decreases the count"
                else:
                    row["verdict_4000"] = "no change in the count"
            else:
                acts = [s["at4000"]["active"] for s in seeds]
                succ = [s["at4000"]["success"] for s in seeds]
                reasons = []
                if not all(a > ref["all_max_4000"] for a in acts):
                    reasons.append(f"not every seed above every control ({ref['all_max_4000']})")
                if abs(np.mean(succ) - ref["mean_success_4000"]) > SUCCESS_TOL:
                    reasons.append(f"mean success {np.mean(succ):.3f} against the controls' {ref['mean_success_4000']:.3f}")
                if min(succ) < 0.9:
                    reasons.append(f"a seed ends at {min(succ):.2f}")
                row["verdict_4000"] = "works" if not reasons else "fails: " + "; ".join(reasons)
        out.append(row)
    return dict(controls=crow, reference_4000=ref, arms=out)


def read_pilot(root):
    """Every pilot run: learned at 2,000 and, when continued, at 4,000; the pass rule per condition.

    Args:
        root: str, the Maze_r19pilot folder.
    Returns:
        list of dicts, one per run.
    """
    rows = []
    for r in maze_load_runs(root):
        cell = os.path.basename(os.path.dirname(r["chain"][0]))
        rows.append(dict(cell=cell, seed=int(r["seed"]), end=last_update(r), at2000=at(r, 2000),
                         at4000=at(r, 4000) if last_update(r) >= 4000 else None))
    return sorted(rows, key=lambda x: (x["cell"], x["seed"]))


def main():
    """Read the three parts, print the tables and write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.join(REPO, "data", "trained_RNNs_rl_maze"))
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    suite = read_suite(os.path.join(a.root, "Maze_suite"))
    print("SUITE: N act seed | runs end budget | learned at stage ends | learned update, active there | end success, active")
    for s in suite:
        print(f"{s['N']:5d} {s['act']:16s} s{s['seed']} | {s['n_runs']} {s['end']:5d} {s['budget']:5d} | "
              f"{s['learned_at']} | {s['learned_update']}, {s['active_learned']} | {s['success_end']:.2f}, {s['active_end']}")
    st = read_settings(os.path.join(a.root, "Maze_interventions"), os.path.join(a.root, "Maze_suite"))
    print("\nCONTROLS (ReLU N = 1,000): seed | 2,000: learned, success, active | 4,000: learned, success, active")
    for c in st["controls"]:
        x, y = c["at2000"], c["at4000"]
        print(f"  s{c['seed']} | {x['learned']}, {x['success']:.2f}, {x['active']} | {y['learned']}, "
              f"{y['success']:.2f}, {y['active']}")
    print("  reference at 4,000:", st["reference_4000"])
    print("\nSETTINGS AND TECHNIQUES")
    for r in st["arms"]:
        if "label" not in r:
            continue
        print(f"\n{r['key']} ({r['cls']})")
        print(f"  2,000: {r['verdict_2000']}")
        print(f"  4,000: {r['verdict_4000']}; {r['label']}")
        for s in r["seeds"]:
            x, y = s["at2000"], s["at4000"]
            ys = (f"{'L' if y['learned'] else '-'} {y['success']:.2f} {y['active']} (learned at {y['learned_update']})"
                  if y["reached"] else f"stopped at {s['end']}")
            xs = f"{'L' if x['learned'] else '-'} {x['success']:.2f} {x['active']}" if x["reached"] else "stopped"
            print(f"    s{s['seed']}: 2,000 {xs} | 4,000 {ys}")
    pilot = read_pilot(os.path.join(a.root, "Maze_r19pilot"))
    print("\nPILOT: cell seed | 2,000 learned, success, active | 4,000 learned, success, active, learned update")
    for p in pilot:
        x, y = p["at2000"], p["at4000"]
        ys = "" if y is None else f"| {y['learned']}, {y['success']:.2f}, {y['active']}, {y['learned_update']}"
        print(f"  {p['cell']:36s} s{p['seed']} | {x['learned']}, {x['success']:.2f}, {x['active']} {ys}")
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump(dict(suite=suite, settings=st, pilot=pilot), fh, indent=1, default=str)
    print(f"\n-> {a.out}")


if __name__ == "__main__":
    main()
