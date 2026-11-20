"""Descriptive, written AFTER Phase A's R1-R4 were read (no verdict, no threshold): the active count of the warm and
the cold arm of the Maze warm-start test (stage 11, updates 20,001-22,000) in windows after update 20,000, relative to
the median of the last 5 probes up to 20,000.

Why: R2 (cold restart disruptive) read J(20,000), the first 5 probes after the boundary (updates 20,001-20,040),
which all fall inside the cold arm's 50-update learning-rate ramp; this table shows the arms after the ramp.

Usage (from trainRNNbrain/experiments_and_analysis), on the same root as maze_warm_start_readings.py:
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_warm_cold_after.py <suite root>
"""
import sys

import numpy as np

from maze_warm_start_readings import B0, N_SIDE, arm_records

WINDOWS = [(20001, 20040), (20050, 20100), (20100, 20200), (20200, 20500), (20500, 21000), (21000, 21500),
           (21500, 22000)]


def window_medians(row):
    """Active-count medians of one arm's record in WINDOWS, minus its median over the last N_SIDE probes up to B0.

    Args:
        row: dict from fig_paper_F2_size.maze_series (it_act, active).
    Returns:
        (pre, [delta per window]): float, list of float (nan when a window has no probe).
    """
    upd, act = np.asarray(row["it_act"], float), np.asarray(row["active"], float)
    pre = float(np.median(act[upd <= B0][-N_SIDE:]))
    deltas = [float(np.median(act[(upd >= lo) & (upd <= hi)])) - pre if ((upd >= lo) & (upd <= hi)).any()
              else float("nan") for lo, hi in WINDOWS]
    return pre, deltas


def main(root):
    """Print warm/cold deltas per network and window, and per window how many networks have cold above warm."""
    recs = arm_records(root)
    print("network      pre  " + " ".join(f"{lo:>6}-{hi:<6}" for lo, hi in WINDOWS) + "  (warm / cold, minus pre)")
    above = np.zeros(len(WINDOWS), int)
    for key in sorted(recs):
        arms = recs[key]
        if arms["warm"] is None or arms["cold"] is None:
            print(f"N={key[0]} s{key[1]}: an arm is missing")
            continue
        pre, w = window_medians(arms["warm"][0])
        _, c = window_medians(arms["cold"][0])
        above += np.array(c) > np.array(w)
        print(f"N={key[0]:<4} s{key[1]} {pre:5.0f}  " + " ".join(f"{a:+6.0f}/{b:+6.0f}" for a, b in zip(w, c)))
    print("cold > warm  " + "      " + " ".join(f"{int(k):>13d}" for k in above))


if __name__ == "__main__":
    main(sys.argv[1])
