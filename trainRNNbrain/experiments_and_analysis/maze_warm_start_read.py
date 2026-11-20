"""Read the Maze warm-start test (rules: notebook entry 2026-11-17_2249_rules-maze-warm-start-test.md; written and
committed while the jobs were running, before any of their records was pulled).

Nine networks continue from update 20,000 in two arms: warm (Adam state reloaded, no warm-up; stages 11 and 12,
updates 20,001-24,000) and cold (Adam restarted with the warm-up; stage 11, 20,001-22,000). The records are pulled
into their own root, data/trained_RNNs_rl_maze_warmtest/Maze_suite, with the nine update-20,000 parent folders linked
in, so common.maze_load_runs returns per network one warm chain (18,001-24,000) and one cold chain (18,001-22,000).

Readings (thresholds from the rules):
  jump J(U)  median active count of the first 5 probes after U minus the median of the last 5 probes up to U;
  noise      the largest |J| at the non-boundary points 18,250, 18,500, ..., 19,750 of the parent stage;
  R1  warm not disruptive: |J_warm(20,000)| <= noise in >= 7 of 9 AND |J_warm(22,000)| <= noise in >= 7 of 9;
  R2  cold disruptive: J_cold(20,000) > noise in >= 5 of 9;
  R3  paired: J_warm(20,000) < J_cold(20,000) in >= 7 of 9;
  R4  warm probe success >= 0.9 at every probe of updates 20,001-20,500 in every network;
  R5  descriptive: warm arm, updates 20,001-24,000: units lost per 100,000 Adam steps (least squares; 8 Adam steps per
      update) against the supervised bar of 5, and the return change per 1,000 updates over the network's rise
      (two-block medians of the first and last 100 updates, as maze_flat_return_full.measure) against 3%. The rise is
      taken from data/maze_long_read_20000.json (reading B), because the rise needs the start of the chain, which
      this root does not hold.

Usage (from trainRNNbrain/experiments_and_analysis):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_warm_start_read.py
Writes data/maze_warm_start_read.json and prints one table.
"""
import json
import os

import numpy as np

from common import maze_load_runs
from round18_maze import REPO

ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze_warmtest", "Maze_suite")
RISE_JSON = os.path.join(REPO, "data", "maze_long_read_20000.json")
OUT = os.path.join(REPO, "data", "maze_warm_start_read.json")
NOISE_POINTS = [18250.0 + 250.0 * k for k in range(7)]     # 18,250 ... 19,750
N_SIDE = 5                    # probes on each side of a jump
R4_WINDOW = (20000.0, 20500.0)
R5_WINDOW = (20000.0, 24000.0)
STEPS_PER_UPDATE = 8          # n_epochs 4 x n_minibatches 2
BLOCK = 100                   # updates per block of the return change
FLAT_BAR, SUP_BAR = 0.03, 5.0


def jump(upd, act, u):
    """Median of the first N_SIDE probes after update u minus the median of the last N_SIDE probes up to u.

    Args:
        upd, act: arrays (probes) of update index and active count; u: float, the boundary.
    Returns:
        float, the jump in units.
    """
    return float(np.median(act[upd > u][:N_SIDE]) - np.median(act[upd <= u][-N_SIDE:]))


def noise(upd, act):
    """The largest |jump| over NOISE_POINTS (non-boundary points of the parent stage).

    Args:
        upd, act: probe arrays of a chain that contains the parent stage.
    Returns:
        float, units.
    """
    return max(abs(jump(upd, act, u)) for u in NOISE_POINTS)


def r5(run, rise):
    """Units lost per 100,000 Adam steps and return change over rise, warm arm, R5_WINDOW.

    Args:
        run: dict from maze_load_runs (warm chain); rise: float, the network's rise from the 20,000 reading.
    Returns:
        dict lost_per_100k_steps (float), return_change_over_rise (per 1,000 updates), flat (bool), supervised_decr (bool).
    """
    pr = run["probes"]
    upd, act = np.asarray(pr["update"], float), np.asarray(pr["active"], float)
    m = (upd > R5_WINDOW[0]) & (upd <= R5_WINDOW[1])
    lost_per_update = -float(np.polyfit(upd[m], act[m], 1)[0])
    lost = lost_per_update * 100000.0 / STEPS_PER_UPDATE
    u = np.array([p["update"] for p in run["progress"]], float)
    r = np.array([p["discounted_return"] for p in run["progress"]], float)
    first = (u > R5_WINDOW[0]) & (u <= R5_WINDOW[0] + BLOCK)
    last = (u > R5_WINDOW[1] - BLOCK) & (u <= R5_WINDOW[1])
    dist = (np.mean(u[last]) - np.mean(u[first])) / 1000.0
    d = float((np.median(r[last]) - np.median(r[first])) / dist / rise)
    return dict(lost_per_100k_steps=lost, return_change_over_rise=d, flat=bool(abs(d) < FLAT_BAR),
                supervised_decr=bool(lost >= SUP_BAR))


def load_arms(root=ROOT):
    """Pair each network's warm and cold chains.

    Args:
        root: str, the warm-test root.
    Returns:
        dict (N, seed) -> {"warm": run or None, "cold": run or None}.
    """
    nets = {}
    for run in maze_load_runs(root):
        name = os.path.relpath(run["folder"], root)
        arm = "warm" if "_warm_" in name else "cold" if "_cold_" in name else None
        if arm is None:
            continue
        nets.setdefault((int(run["N"]), int(run["seed"])), {"warm": None, "cold": None})[arm] = run
    return nets


def main():
    """Apply R1-R5, print the per-network table and the verdicts, and write the JSON; returns None."""
    rises = {}
    for w in json.load(open(RISE_JSON))["networks"]:
        key = w["label"].replace(" (resumed)", "")
        rises[key] = w["B"]["rise"]
    rows = []
    for (n, seed), arms in sorted(load_arms().items()):
        wr, cr = arms["warm"], arms["cold"]
        row = dict(N=n, seed=seed, complete=bool(wr is not None and cr is not None))
        if row["complete"]:
            wu, wa = np.asarray(wr["probes"]["update"], float), np.asarray(wr["probes"]["active"], float)
            cu, ca = np.asarray(cr["probes"]["update"], float), np.asarray(cr["probes"]["active"], float)
            ws = np.asarray(wr["probes"]["probe_success"], float)
            row.update(complete=bool(wu[-1] >= 24000 and cu[-1] >= 22000))
        if row["complete"]:
            row.update(noise=noise(wu, wa), J_warm_20k=jump(wu, wa, 20000.0), J_warm_22k=jump(wu, wa, 22000.0),
                       J_cold_20k=jump(cu, ca, 20000.0),
                       min_success_r4=float(ws[(wu > R4_WINDOW[0]) & (wu <= R4_WINDOW[1])].min()))
            row.update(r5(wr, rises[f"N={n} s{seed}"]))
        rows.append(row)
    ok = [w for w in rows if w["complete"]]
    r1a = sum(abs(w["J_warm_20k"]) <= w["noise"] for w in ok)
    r1b = sum(abs(w["J_warm_22k"]) <= w["noise"] for w in ok)
    r2 = sum(w["J_cold_20k"] > w["noise"] for w in ok)
    r3 = sum(w["J_warm_20k"] < w["J_cold_20k"] for w in ok)
    r4 = all(w["min_success_r4"] >= 0.9 for w in ok)
    print(f"{'network':<12} {'noise':>5} {'J warm 20k':>10} {'J warm 22k':>10} {'J cold 20k':>10} {'min succ':>8} "
          f"{'lost/100k steps':>15} {'ret/rise %':>10} {'flat':>5}")
    for w in rows:
        if not w["complete"]:
            print(f"N={w['N']} s{w['seed']}: incomplete (a stage is missing or short)")
            continue
        print(f"{'N=%d s%d' % (w['N'], w['seed']):<12} {w['noise']:>5.0f} {w['J_warm_20k']:>+10.0f} {w['J_warm_22k']:>+10.0f} "
              f"{w['J_cold_20k']:>+10.0f} {w['min_success_r4']:>8.2f} {w['lost_per_100k_steps']:>15.0f} "
              f"{100 * w['return_change_over_rise']:>+10.1f} {str(w['flat']):>5}")
    v = dict(n=len(ok),
             R1=("warm start not disruptive" if (r1a >= 7 and r1b >= 7) else "warm start disruptive"),
             R1_counts=[int(r1a), int(r1b)],
             R2=("cold restart disruptive" if r2 >= 5 else "cold restart not shown disruptive"), R2_count=int(r2),
             R3=("warm below cold" if r3 >= 7 else "not shown"), R3_count=int(r3),
             R4=("passes" if r4 else "fails"),
             R5_flat=int(sum(w["flat"] for w in ok)),
             R5_supervised_decreasing=int(sum(w["supervised_decr"] for w in ok)),
             R5_flat_and_decreasing=int(sum(w["flat"] and w["supervised_decr"] for w in ok)))
    print(f"\n{len(ok)} complete networks of {len(rows)}")
    print(f"R1: |J warm 20k| <= noise in {r1a}, |J warm 22k| <= noise in {r1b} -> {v['R1']}")
    print(f"R2: J cold 20k > noise in {r2} -> {v['R2']}")
    print(f"R3: J warm < J cold at 20k in {r3} -> {v['R3']}")
    print(f"R4: minimum warm probe success over 20,001-20,500 >= 0.9 in every network -> {v['R4']}")
    print(f"R5: flat return {v['R5_flat']}; >= {SUP_BAR:.0f} units lost per 100k Adam steps {v['R5_supervised_decreasing']}; "
          f"both {v['R5_flat_and_decreasing']}")
    with open(OUT, "w") as f:
        json.dump(dict(rows=rows, verdicts=v), f, indent=1)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
