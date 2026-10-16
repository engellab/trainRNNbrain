#!/usr/bin/env python3
"""How often each training method, size and hyperparameter setting bursts: one row per cell.

A burst is a stretch where the noisy training loss climbs above 2x its local level (common.find_bursts,
called inside common.noisy_floor) - a network briefly going unstable and recovering. The read-out
removes bursts from the floor and the crossing; this table keeps them as a property of the cell.

Bursts are counted on each run's OWN iterations, from the floor fit window on (2,000 iterations, or
200 after the escape from a stall): a warm-started run does not inherit its parent's bursts.
Rates are pooled over a cell's runs: total bursts / total counted iterations.

Output: data/burst_stats.csv, one row per <sweep>/<cell>, and the cells with bursts printed.
Usage:  python trainRNNbrain/experiments_and_analysis/burst_stats.py [--workers 8]
"""

import argparse
import csv
import glob
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DATA_DIR, noisy_floor, noisy_loss_series

OUT = os.path.join(DATA_DIR, "..", "burst_stats.csv")


def run_bursts(run_dir):
    """Bursts of one run on its own iterations.

    Args:
        run_dir: a trained-network folder (holds a *TrainLosses.json).
    Returns:
        dict with run, counted (iterations bursts were looked for in) and bursts (list of
        (first, last, peak ratio)), or None when the run has no loss series.
    """
    it, L = noisy_loss_series(run_dir)
    if L is None:
        return None
    with open(sorted(glob.glob(os.path.join(run_dir, "*TrainLosses.json")))[0]) as fh:
        own = len(json.load(fh).get("train_losses", []))
    f = noisy_floor(it, L)
    t_from = max(it[-1] - own + 1, f["burst_from"])
    return dict(run=run_dir, counted=max(it[-1] - t_from + 1, 0.0),
                bursts=[b for b in f["bursts"] if b[0] >= t_from])


def cell_row(cell, runs):
    """Pool one cell's runs into a table row.

    Args:
        cell: "<sweep>/<cell>"; runs: run_bursts results for that cell.
    Returns:
        dict: cell, runs, runs_bursting, bursts, per_100k (bursts per 100,000 counted iterations),
        pct_iters (percent of counted iterations inside a burst), median_len (iterations), max_peak.
    """
    bs = [b for r in runs for b in r["bursts"]]
    counted = sum(r["counted"] for r in runs)
    lens = [b[1] - b[0] + 1 for b in bs]
    return dict(cell=cell, runs=len(runs), runs_bursting=sum(bool(r["bursts"]) for r in runs),
                bursts=len(bs), per_100k=1e5 * len(bs) / counted if counted else float("nan"),
                pct_iters=100 * sum(lens) / counted if counted else float("nan"),
                median_len=float(np.median(lens)) if lens else float("nan"),
                max_peak=max(b[2] for b in bs) if bs else float("nan"))


def main():
    """Count bursts in every run under DATA_DIR and write the per-cell table."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    run_dirs = sorted({os.path.dirname(p) for p in glob.glob(os.path.join(DATA_DIR, "*", "*", "*",
                                                                          "*TrainLosses.json"))})
    with Pool(args.workers) as pool:
        res = [r for r in pool.map(run_bursts, run_dirs, chunksize=4) if r]
    cells = {}
    for r in res:
        cells.setdefault(os.path.relpath(os.path.dirname(r["run"]), DATA_DIR), []).append(r)
    rows = [cell_row(c, rs) for c, rs in sorted(cells.items())]
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    print(f"{len(res)} runs in {len(rows)} cells -> {os.path.normpath(OUT)}")
    print(f"{'cell':80s} {'runs':>4s} {'burst':>5s} {'n':>4s} {'/100k':>6s} {'%iter':>6s} "
          f"{'medlen':>6s} {'peak':>8s}")
    for r in sorted(rows, key=lambda r: -r["per_100k"] if r["bursts"] else 0):
        if r["bursts"]:
            print(f"{r['cell'][:80]:80s} {r['runs']:4d} {r['runs_bursting']:5d} {r['bursts']:4d} "
                  f"{r['per_100k']:6.1f} {r['pct_iters']:6.2f} {r['median_len']:6.0f} {r['max_peak']:8.2g}")


if __name__ == "__main__":
    main()
