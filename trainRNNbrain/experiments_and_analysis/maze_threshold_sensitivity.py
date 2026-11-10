#!/usr/bin/env python3
"""Activity-threshold sensitivity of the Maze size exponent (dead-relu-paper #218, committee #217 item R2).

RULE. docs/project_trajectory/2026-11-09_1824_maze-threshold-sensitivity-rule.md, written before the run: the
learned Maze size-series networks of N >= 500 (fig_paper_F2_size.maze_series), the stored participation vector
nearest each network's learned point, the active count with p >= f * q95(p) at f = 0.025, 0.05, 0.10, and the
exponent of Fig. 2a (power_fit, boot_exponent) refitted at each factor. Check: at every stored vector whose update
is also a probe update, the count at f = 0.05 equals the probe's stored count.

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/maze_threshold_sensitivity.py [--root DATA/trained_RNNs_rl_maze/Maze_suite]
"""
import argparse
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from common import active_mask, maze_load_runs                                                # noqa: E402
from fig_paper_F2_size import BOOT_SEED, FIT_MIN_N, boot_exponent, maze_series, power_fit      # noqa: E402

FACTORS = (0.025, 0.05, 0.10)


def check_same_quantity(probes):
    """Compare the f = 0.05 count of every stored participation vector with the probe count at the same update.

    Args:
        probes: dict of probe arrays of one run (common.maze_load_runs): update, active, p_update, p (n, N).
    Returns:
        (n_compared, n_mismatch): ints.
    """
    at = {int(u): int(a) for u, a in zip(probes["update"], probes["active"])}
    n, bad = 0, 0
    for u, p in zip(probes["p_update"], probes["p"]):
        if int(u) in at:
            n += 1
            bad += int(active_mask(p).sum()) != at[int(u)]
    return n, bad


def main():
    """Read the learned Maze networks, print each network's counts per factor, the check and the refitted exponents."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite"))
    a = ap.parse_args()
    rows = [r for r in maze_series(a.root) if r["learned"] and r["N"] >= FIT_MIN_N]
    runs = {(r["N"], str(r["seed"])): r for r in maze_load_runs(a.root) if r["act"] == "relu" and not r["cue"]}
    counts = {f: {} for f in FACTORS}
    n_cmp, n_bad = 0, 0
    print("N     seed  t*     p_update  fig2a  " + "  ".join(f"f={f}" for f in FACTORS))
    for row in rows:
        pr = runs[(row["N"], row["seed"])]["probes"]
        c, b = check_same_quantity(pr)
        n_cmp, n_bad = n_cmp + c, n_bad + b
        k = int(np.argmin(np.abs(pr["p_update"] - row["t_star"])))
        p = pr["p"][k]
        line = f"{row['N']:5d} s{row['seed']:<4s} {row['t_star']:6.0f} {int(pr['p_update'][k]):8d}  {row['active_learned']:5.0f}"
        for f in FACTORS:
            n_act = int(active_mask(p, rel=f).sum())
            counts[f].setdefault(row["N"], []).append(n_act)
            line += f"  {n_act:7d}"
        print(line)
    print(f"\ncheck: {n_cmp} stored vectors at probe updates, {n_bad} with a count different from the probe's")
    for f in FACTORS:
        cells = {N: np.array(v, float) for N, v in sorted(counts[f].items())}
        b, _ = power_fit(cells, min_n=FIT_MIN_N)
        lo, hi = boot_exponent(cells, np.random.default_rng(BOOT_SEED), FIT_MIN_N)
        print(f"f = {f:<6}: b = {b:.3f} (bootstrap 95% {lo:.3f}-{hi:.3f}); n per size "
              + ", ".join(f"{N}: {len(v)}" for N, v in cells.items()))
    for f in (0.025, 0.10):
        rel = [100 * (x / y - 1) for N in counts[0.05] for x, y in zip(counts[f][N], counts[0.05][N])]
        print(f"count at f = {f} relative to 0.05: {min(rel):+.0f}% to {max(rel):+.0f}%")


if __name__ == "__main__":
    main()
