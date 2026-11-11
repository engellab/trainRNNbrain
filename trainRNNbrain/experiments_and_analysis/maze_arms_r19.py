#!/usr/bin/env python3
"""The Maze settings and technique networks at N = 1,000, read at updates 2,000 and 4,000 (round 19).

QUESTION. Figure 3c and the Extended Data figure of the Maze settings and techniques draw every arm of the
uniform-interventions programme (issues engellab/trainRNNbrain#5, #6). After rounds 18 and 19 three things
changed (dead-relu-paper#201):
  - three more learned N = 1,000 controls (G12, uniform seeds 4-6), so the settings reference at update
    2,000 is the five learned controls (seeds 1, 2, 4, 5, 6: 394-449 active units; entry
    2026-11-08_0859_round18-training-final-results);
  - the arms not learned at update 2,000, and the other seeds of their arm, were continued to update 4,000
    (entries 2026-11-07_1111 and 2026-11-08_0915); update 2,000 stays the stated read and update 4,000 is
    reported beside it, against ReLU N = 1,000 seeds 1-6 at update 4,000 (314-404 active, all learned);
  - the rate-penalty, rate + sparsity and metabolic-penalty arms are read from their reruns with
    trainer.freeze_after_end=true (entries 2026-11-08_0940, 2026-11-09_1221), which replace the withdrawn runs,
    and prune-and-duplicate from its rerun with the same switch (entry 2026-11-09_0136);
  - round 22 (dead-relu-paper#228): read-out dropout (rates 0.2 and 0.05, beta 4) and the four rescaling arms are
    read from their reruns with the same switch (entries 2026-11-09_1513 rules, 2026-11-10_2014 update 2,000,
    2026-11-11_0306 update 4,000), which replace the runs made before the trainer correction. Noisy synapses and
    re-initialisation (redraw, zeroout) were not rerun and keep their first runs.
This module is the one reader both figures use, so the two draw the same numbers.

SOURCES (the Mac's copies of the cluster trees):
  controls   data/trained_RNNs_rl_maze/Maze_suite, the N = 1,000 ReLU runs without the goal cue (seeds 1-6),
             joined along their continuations by common.maze_load_runs;
  arms       data/trained_RNNs_rl_maze/Maze_interventions, keyed by the stage-1 folder name (the launcher's
             ARM_TAG: arm_of folds every sigma_w into "synnoise" and every dropout rate into "dropout", so it
             gives the class, not the arm), joined along continuations (<arm>_cont);
  fz arms    data/trained_RNNs_rl_maze_fz/Maze_interventions for FZ_ARMS: those keys are taken from this tree
             only, and their runs in the first tree are not read.
A run folder with no saved config (stopped by the rollback guard before its first checkpoint) is a seed with
no record, "stopped before a checkpoint".

READ-OUT AT UPDATE U (point_at, the same rule as round19_read.at): common.maze_learned_index on the joined
record cut at U, provided the record reaches U; success and active units at the last probe at or before U. A
record that ends before U is not read there; it is "stopped" when it ended before its budget (iter_offset +
n_updates of its last run) and "not trained to U" otherwise.

CHECK. crosscheck() compares every seed's two reads with data/round19/round19_results.json (round19_read.py)
and data/round19/fz_reruns.json (round19_read_fz.py) and prints every difference; the figures print its result.

Run alone:  PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/maze_arms_r19.py
            (prints every arm at both points and the cross-check)
"""
import glob
import json
import os

import numpy as np
import yaml

from common import maze_learned_index, maze_load_runs

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
SUITE = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite")
ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_interventions")
ROOT_FZ = os.path.join(REPO, "data", "trained_RNNs_rl_maze_fz", "Maze_interventions")
R19_JSON = os.path.join(REPO, "data", "round19", "round19_results.json")
FZ_JSON = os.path.join(REPO, "data", "round19", "fz_reruns.json")
N = 1000
SEEDS = (1, 2, 3)
READS = (2000, 4000)              # the stated read and the read beside it
TECH_CONTROL_SEEDS = (1, 2, 3)    # the technique criterion's controls at 2,000 (issue #5, decision #194)
SUCCESS_TOL = 0.05                # the technique criterion's tolerance in success
LEARNED_LEVEL = 0.9               # no technique network may end below this
# the arms read from the freeze_after_end=true reruns only (stage-1 folder names)
FZ_ARMS = ("frm0.03", "frm0.1", "frm0.3", "frm0.1_rws0.05", "met0.01", "met0.1", "met1", "met10", "duplicate",
           "dropout0.2_beta4", "dropout0.05_beta4", "rescale", "rescale_mild", "rescale_few", "rescale_tgt4")


def run_config(folder):
    """The saved config of one run folder.

    Args:
        folder: str, a run folder holding a *config.yaml.
    Returns:
        dict.
    """
    with open(sorted(glob.glob(os.path.join(folder, "*config.yaml")))[0]) as fh:
        return yaml.safe_load(fh)


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
    tr = run_config(run["folder"])
    tr = tr.get("trainer") or tr.get("ppo")
    off = tr.get("iter_offset", 0)
    return int(tr["n_updates"]) + (int(off) if isinstance(off, (int, float)) else 0)


def point_at(run, U):
    """A network's state at update U from its joined record (round19_read.at's rule).

    Args:
        run: dict from common.maze_load_runs, or None (a seed with no record); U: int, an update.
    Returns:
        dict: reached (bool, the record extends to U), learned (bool), learned_update (int or None),
        active_learned (int or None), success and active at the last probe at or before U (None if not
        reached).
    """
    none = dict(reached=False, learned=False, learned_update=None, active_learned=None, success=None, active=None)
    if run is None:
        return none
    pu = np.asarray(run["probes"]["update"])
    if last_update(run) < U or pu.max() < U:
        return none
    m = pu <= U
    succ = np.asarray(run["probes"]["probe_success"])[m]
    act = np.asarray(run["probes"]["active"])[m]
    li = maze_learned_index(succ)
    return dict(reached=True, learned=li is not None, learned_update=None if li is None else int(pu[m][li]),
                active_learned=None if li is None else int(act[li]), success=float(succ[-1]), active=int(act[-1]))


def seed_row(run, seed):
    """One network's record and its reads at READS.

    Args:
        run: dict from common.maze_load_runs, or None for a folder with no saved config; seed: int.
    Returns:
        dict: seed, run, end (last update or None), budget, continued (bool), stopped_at (the last update of a
        record that ended before its budget; 0 for a folder with no record; None otherwise), and one entry
        per read, "at2000" and "at4000" (point_at).
    """
    if run is None:
        return dict(seed=seed, run=None, end=None, budget=None, continued=False, stopped_at=0,
                    **{f"at{U}": point_at(None, U) for U in READS})
    end, bud = last_update(run), budget(run)
    return dict(seed=seed, run=run, end=end, budget=bud, continued=len(run["chain"]) > 1,
                stopped_at=end if end < bud else None, **{f"at{U}": point_at(run, U) for U in READS})


def load_controls(suite=SUITE):
    """The N = 1,000 ReLU controls of the suite (seeds 1-6), each read at READS.

    Args:
        suite: str, the Maze_suite folder.
    Returns:
        list of seed_row dicts, by seed.
    """
    runs = [r for r in maze_load_runs(suite) if r["N"] == N and r["act"] == "relu" and not r["cue"]]
    return [seed_row(r, int(r["seed"])) for r in sorted(runs, key=lambda r: r["seed"])]


def load_suite_activations(acts, suite=SUITE):
    """The N = 1,000 suite runs of other activation functions, each read at READS.

    Args:
        acts: iterable of the saved activation names (leaky_relu, softplus, sigmoid, tanh); suite: str.
    Returns:
        dict act -> list of seed_row dicts, by seed.
    """
    runs = [r for r in maze_load_runs(suite) if r["N"] == N and not r["cue"] and r["act"] in set(acts)]
    return {a: [seed_row(r, int(r["seed"])) for r in sorted(runs, key=lambda r: r["seed"]) if r["act"] == a]
            for a in acts}


def _arms_of_tree(root, keep):
    """Every arm under one Maze_interventions tree, keyed by its stage-1 folder name.

    Args:
        root: str; keep: callable key -> bool, which arm keys to read from this tree.
    Returns:
        dict key -> dict seed -> seed_row (a folder with no saved config gives a row with run None).
    """
    out = {}
    if not os.path.isdir(root):
        print(f"maze_arms_r19: no folder {root}")
        return out
    for r in maze_load_runs(root):
        key = os.path.basename(os.path.dirname(r["chain"][0]))
        if keep(key):
            out.setdefault(key, {})[int(r["seed"])] = seed_row(r, int(r["seed"]))
    for folder in sorted(glob.glob(os.path.join(root, "*", "s*", ""))):
        folder = folder.rstrip("/")
        key, s = os.path.basename(os.path.dirname(folder)), os.path.basename(folder)[1:]
        if key.endswith("_cont") or "_cont" in key or not s.isdigit() or not keep(key):
            continue
        if not glob.glob(os.path.join(folder, "*config.yaml")) and int(s) not in out.get(key, {}):
            out.setdefault(key, {})[int(s)] = seed_row(None, int(s))
    return out


def load_arms(root=ROOT, root_fz=ROOT_FZ):
    """Every arm: FZ_ARMS from the freeze_after_end=true tree, every other arm from the first tree.

    Args:
        root, root_fz: str, the two Maze_interventions trees.
    Returns:
        dict key -> dict(key, source ("fz" or "orig"), seeds: list of seed_row dicts by seed).
    """
    orig = _arms_of_tree(root, lambda k: k not in FZ_ARMS)
    fz = _arms_of_tree(root_fz, lambda k: k in FZ_ARMS)
    out = {}
    for src, tree in (("orig", orig), ("fz", fz)):
        for key, seeds in tree.items():
            out[key] = dict(key=key, source=src, seeds=[seeds[s] for s in sorted(seeds)])
    return out


def settings_reference(controls, U):
    """The settings rule's reference at update U: the controls learned at U.

    Args:
        controls: load_controls output; U: int, a read.
    Returns:
        dict: n (learned controls), seeds, active_min, active_max, active (mean), success (their mean success).
    """
    pts = [(c["seed"], c[f"at{U}"]) for c in controls if c[f"at{U}"]["learned"]]
    act = [p["active"] for _, p in pts]
    return dict(n=len(pts), seeds=[s for s, _ in pts], active_min=min(act), active_max=max(act),
                active=float(np.mean(act)), success=float(np.mean([p["success"] for _, p in pts])))


def technique_reference(controls, U):
    """The technique criterion's reference at update U.

    At 2,000 it is the three uniform-series controls (seeds 1-3), as issue #5 wrote it before any run
    (decision dead-relu-paper#194); at 4,000 it is every one of the six controls (the part 2 rules of
    2026-11-07_1111).

    Args:
        controls: load_controls output; U: int, a read.
    Returns:
        dict: n, seeds, active (mean), active_min, active_max (the bar), success (mean success).
    """
    use = [c for c in controls if (U != 2000 or c["seed"] in TECH_CONTROL_SEEDS) and c[f"at{U}"]["reached"]]
    act = [c[f"at{U}"]["active"] for c in use]
    return dict(n=len(use), seeds=[c["seed"] for c in use], active=float(np.mean(act)), active_min=min(act),
                active_max=max(act), success=float(np.mean([c[f"at{U}"]["success"] for c in use])))


def setting_direction(seeds, ref, U):
    """The settings direction rule at update U: learned seeds only, against the learned controls' range.

    Args:
        seeds: an arm's seed_row list; ref: settings_reference at U; U: int.
    Returns:
        str: "increases", "decreases", "no change", "no direction" (every seed read, none learned),
        "incomplete" (a seed not read at U, some read) or "no verdict" (no seed read at U).
    """
    pts = [s[f"at{U}"] for s in seeds]
    if not any(p["reached"] for p in pts):
        return "no verdict"
    if not all(p["reached"] for p in pts) or len(pts) < len(SEEDS):
        return "incomplete"
    act = [p["active"] for p in pts if p["learned"]]
    if not act:
        return "no direction"
    if all(a > ref["active_max"] for a in act):
        return "increases"
    if all(a < ref["active_min"] for a in act):
        return "decreases"
    return "no change"


def technique_verdict(seeds, tref, U):
    """The technique criterion at update U (issue #5; decision #194).

    A value works when every seed is read at U, every seed's count exceeds every control's, the mean
    success is no more than SUCCESS_TOL below the controls' mean (one-sided, as rule_verdict), and no seed
    ends below LEARNED_LEVEL.

    Args:
        seeds: an arm's seed_row list; tref: technique_reference at U; U: int.
    Returns:
        (verdict, clauses): verdict "works", "fails", "incomplete" or "no verdict"; clauses a list of the
        failing clauses ("a" count, "p" performance, "l" a seed below 0.9).
    """
    pts = [s[f"at{U}"] for s in seeds]
    if not any(p["reached"] for p in pts):
        return "no verdict", []
    if not all(p["reached"] for p in pts) or len(pts) < len(SEEDS):
        return "incomplete", []
    clauses = []
    if not all(p["active"] > tref["active_max"] for p in pts):
        clauses.append("a")
    if np.mean([p["success"] for p in pts]) - tref["success"] < -SUCCESS_TOL:
        clauses.append("p")
    if min(p["success"] for p in pts) < LEARNED_LEVEL:
        clauses.append("l")
    return ("works" if not clauses else "fails"), clauses


def crosscheck(controls, arms):
    """Compare every read with the round 19 read-out files; print and return the differences.

    Args:
        controls: load_controls output; arms: load_arms output.
    Returns:
        list of str, one per difference (empty when every read matches).
    """
    diffs = []
    keys = ("reached", "learned", "learned_update", "active_learned", "active")
    with open(R19_JSON) as fh:
        r19 = json.load(fh)["settings"]
    with open(FZ_JSON) as fh:
        fz = json.load(fh)

    def cmp(tag, mine, theirs):
        """Append a difference for each read key that does not match (success to 1e-9)."""
        for U in READS:
            a, b = mine[f"at{U}"], theirs[f"at{U}"]
            for k in keys:
                if a[k] != b[k]:
                    diffs.append(f"{tag} at {U}: {k} {a[k]} here, {b[k]} in the file")
            if (a["success"] is None) != (b["success"] is None) or (
                    a["success"] is not None and abs(a["success"] - b["success"]) > 1e-9):
                diffs.append(f"{tag} at {U}: success {a['success']} here, {b['success']} in the file")

    ctrl = {c["seed"]: c for c in controls}
    for c in r19["controls"]:
        cmp(f"control s{c['seed']}", ctrl[c["seed"]], c)
    for src, file_arms in (("orig", r19["arms"]), ("fz", fz["arms"])):
        for fa in file_arms:
            if (src == "orig") == (fa["key"] in FZ_ARMS):
                continue
            mine = {s["seed"]: s for s in arms.get(fa["key"], {"seeds": []})["seeds"]}
            for s in fa["seeds"]:
                if s["seed"] not in mine:
                    diffs.append(f"{fa['key']} s{s['seed']}: in the file, not read here")
                    continue
                cmp(f"{fa['key']} s{s['seed']} ({src})", mine[s["seed"]], s)
    for d in diffs:
        print("  CROSS-CHECK DIFFERENCE:", d)
    print(f"maze_arms_r19 cross-check against round19_results.json and fz_reruns.json: "
          f"{'every read matches' if not diffs else f'{len(diffs)} differences'}")
    return diffs


def fmt_seed(s, U):
    """One seed's read at U as text: 'L 1.00, 998 (770)', '0.87, 997', 'stopped at 1,900' or 'not trained'.

    Args:
        s: a seed_row; U: int.
    Returns:
        str.
    """
    p = s[f"at{U}"]
    if p["reached"]:
        return (f"{'L ' if p['learned'] else ''}{p['success']:.2f}, {p['active']}"
                + (f" ({p['learned_update']})" if p["learned"] else ""))
    if s["stopped_at"] is not None:
        return "stopped before a checkpoint" if s["stopped_at"] == 0 else f"stopped at {s['stopped_at']:,}"
    return "not trained"


def main():
    """Print every control and arm at both reads, the references, and the cross-check."""
    controls = load_controls()
    arms = load_arms()
    for U in READS:
        print(f"settings reference at {U}: {settings_reference(controls, U)}")
        print(f"technique reference at {U}: {technique_reference(controls, U)}")
    for c in controls:
        print(f"  control s{c['seed']}: " + " | ".join(fmt_seed(c, U) for U in READS))
    for key in sorted(arms):
        a = arms[key]
        print(f"{key} ({a['source']}): " + "; ".join(f"s{s['seed']} " + " | ".join(fmt_seed(s, U) for U in READS)
                                                  for s in a["seeds"]))
    crosscheck(controls, arms)


if __name__ == "__main__":
    main()
