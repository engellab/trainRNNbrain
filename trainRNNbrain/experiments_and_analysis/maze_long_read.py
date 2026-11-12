"""Read the Maze networks continued to 12,000 updates (rule: notebook entry
2026-11-11_2322_rules-maze-long-training-and-retrain-at-250.md, section 1; written before the jobs ran,
and this script was written before any continued record existed).

Reading A, the decrease as training progresses: per network, the least-squares slope of the noise-free
probe's number of active units over updates 4,000 to 12,000, and the medians of the last 10 probes up
to update 4,000 and up to update 12,000. A network decreases if it loses at least 5 active units per
1,000 updates AND its later median is below its earlier one.

Reading B, the decrease under a flat return: maze_flat_return_full.measure with the last quarter of the
record (updates 9,000 to 12,000) as the window; fewer than 4 flat networks give no verdict.

Usage (from trainRNNbrain/experiments_and_analysis):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_long_read.py
Writes data/maze_long_read.json and prints one table per reading.
"""
import json
import os

import numpy as np

from common import maze_load_runs
from fig_paper_F2_size import MAZE_ROOT, maze_series
from maze_flat_return_full import DECREASE_BAR, measure
from round18_maze import REPO, three_way

START, END = 4000.0, 12000.0     # reading A: the continued stretch
N_MEDIAN = 10                    # probes in each median of reading A
MIN_FLAT = 4                     # reading B: fewer flat networks than this give no verdict
SIZES = (500, 1000)              # the continued networks
OUT = os.path.join(REPO, "data", "maze_long_read.json")


def continued_rows():
    """The continued networks whose record reaches update 12,000.

    Returns:
        list of (row, run): row from fig_paper_F2_size.maze_series (every default ReLU network of the
        sizes in SIZES with it_end >= END, learned or not), run from common.maze_load_runs.
    """
    runs = {os.path.relpath(r["folder"], MAZE_ROOT): r for r in maze_load_runs(MAZE_ROOT)}
    rows = [r for r in maze_series() if r["N"] in SIZES and r["it_end"] >= END]
    return [(r, runs[r["name"]]) for r in rows]


def reading_a(row):
    """The change of the number of active units over updates 4,000 to 12,000 in one network.

    Args:
        row: dict from maze_series (it_act: probe updates; active: number of active units per probe;
            success: probe success rate per probe; learned: bool).
    Returns:
        dict with N, seed, learned, success_end, active_4000, active_12000 (medians of N_MEDIAN probes),
        lost_per_1000 (minus the least-squares slope, units per 1,000 updates), decreasing (bool).
    """
    upd, act = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    early = float(np.median(act[upd <= START][-N_MEDIAN:]))
    late = float(np.median(act[upd <= END][-N_MEDIAN:]))
    m = (upd >= START) & (upd <= END)
    slope = float(np.polyfit(upd[m] / 1000.0, act[m], 1)[0])
    return dict(N=int(row["N"]), seed=int(row["seed"]), learned=bool(row["learned"]),
                success_end=float(row["success"][-1]), active_4000=early, active_12000=late,
                lost_per_1000=-slope, decreasing=bool(-slope >= DECREASE_BAR and late < early))


def main():
    """Print both readings with their verdicts and write the JSON; returns None."""
    pairs = continued_rows()
    print(f"{len(pairs)} continued networks with a record to update {END:.0f}")

    a_rows = [reading_a(row) for row, _ in pairs]
    print(f"\n[A decrease over 4,000-12,000] {'N':>5} {'seed':>4} {'learned':>7} {'success':>7} "
          f"{'active 4,000':>12} {'12,000':>7} {'lost/1000':>9} {'decr':>5}")
    for w in a_rows:
        print(f"  {w['N']:>33} {w['seed']:>4} {str(w['learned']):>7} {w['success_end']:>7.2f} "
              f"{w['active_4000']:>12.0f} {w['active_12000']:>7.0f} {w['lost_per_1000']:>9.1f} {str(w['decreasing']):>5}")
    kept = [w for w in a_rows if w["learned"]]
    v_a = three_way([w["decreasing"] for w in kept], "keeps decreasing", "does not keep decreasing")
    print(f"  [A] counted {len(kept)} of {len(a_rows)} (last probe at or above 0.9); decreasing "
          f"{sum(w['decreasing'] for w in kept)} -> {v_a}")

    b_rows = [measure(row, run, "B") for row, run in pairs if row["learned"]]
    print(f"\n[B flat return, updates 9,000-12,000] {'N':>5} {'seed':>4} {'rise':>7} {'D/rise %':>9} {'flat':>5} "
          f"{'lost/1000':>9} {'decr':>5}")
    for w in b_rows:
        print(f"  {w['N']:>40} {w['seed']:>4} {w['rise']:>7.3f} {100 * w['D_over_rise']:>+9.2f} {str(w['flat']):>5} "
              f"{w['lost_per_1000']:>9.1f} {str(w['decreasing']):>5}")
    flat = [w for w in b_rows if w["flat"]]
    if len(flat) < MIN_FLAT:
        v_b = "return still not flat at 12,000 updates (fewer than 4 flat networks): no verdict"
    else:
        v_b = three_way([w["decreasing"] for w in flat], "decreasing while the return is flat", "not decreasing")
    print(f"  [B] tested {len(b_rows)}; flat {len(flat)}; decreasing among flat "
          f"{sum(w['decreasing'] for w in flat)} -> {v_b}")

    with open(OUT, "w") as f:
        json.dump(dict(A=dict(rows=a_rows, verdict=v_a), B=dict(rows=b_rows, verdict=v_b)), f, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
