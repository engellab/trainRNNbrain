#!/usr/bin/env python3
"""
Pull the weight-displacement trajectories AND the final participation vector, one unpenalised
cell per task, for the talk.

RUNS ON THE CLUSTER, where the participation traces live; writes a small npz the slide script reads.

THE QUESTION is whether the parameters are still moving at the end of training, and the honest way
to answer it is a trajectory rather than a summary. The Trainer logs, every 100 iterations,
|W(t+L) - W(t)| / |W(t)| at lags L = 100, 1,000 and 10,000. Plotted against t, a curve that falls to
zero means the weights have stopped; one that flattens at a non-zero value means they are still
moving, just not going anywhere new. The (N, k) heat maps this replaces reported only the final
lag-scaling EXPONENT, which answers a different question and only for the flip-flop.

ONE CELL PER TASK, unpenalised, N = 1000, every seed. The budgets differ because the tasks do:
DMTS needs 150,000 iterations to be solved and CDDM 100,000. The flip-flop's long runs (150k and
400k) were deleted before drift logging existed, so 40,000 is the longest flip-flop trace that
carries these metrics - the panel says so rather than implying the task was only ever run that far.

The participation vector is the first seed's, at its last probe: the distribution panel shows one
real network per task rather than a pooled histogram, which would blur the bimodality that is the
whole point of it.

Usage (on the cluster):  python f2_drift_traces.py [OUT.npz]
Output: ~/f2_drift_traces.npz -> data/f2_drift_traces.npz beside the slide script
"""

import glob
import os
import pickle
import sys

import numpy as np

D = os.environ.get("F2_DATA", "/home/pt1290/trainRNNbrain/data/trained_RNNs")
VARS = ("W_inp", "W_rec", "W_out")
LAG = 10_000                 # the longest logged lag: the least noisy read of "still moving"

# (task key, nice label, cell). Unpenalised, N = 1000, the longest drift-logged run each task has.
CELLS = [
    ("CDDM", "CDDM", "CDDM_paper_grid/EqType=h_N=1000_arm=control"),
    ("NBitFlipFlop", "3-bit flip-flop", "NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none"),
    ("DMTS", "DMTS, 7 tau delay", "DMTS_d7_pen/EqType=h_N=1000_pen=none"),
]


def series(trace, key):
    """One logged metric as (iterations, values), dropping probes where it was not recorded.

    Args:
        trace: a trace dict with 'iters' and a 'metrics' sub-dict; key: the metric name.
    Returns:
        (iters, values) float arrays, empty if the metric is absent.
    """
    m = (trace.get("metrics") or {}).get(key)
    it = trace.get("iters")
    if m is None or it is None:
        return np.array([]), np.array([])
    it, m = np.asarray(it, float), np.asarray(m, float)
    n = min(len(it), len(m))
    it, m = it[:n], m[:n]
    ok = np.isfinite(m) & (m > 0)
    return it[ok], m[ok]


def main(out_path):
    """Write the trajectories. Returns the output path."""
    store = {}
    for task, label, cell in CELLS:
        files = sorted(glob.glob(os.path.join(D, cell, "*", "*ParticipationTrace.pkl")))
        if not files:
            print(f"  SKIP {task}: no traces under {cell}")
            continue
        kept = 0
        for s, f in enumerate(files):
            try:
                tr = pickle.load(open(f, "rb"))
            except Exception as e:
                print(f"  SKIP {task} seed {s}: {type(e).__name__}")
                continue
            if s == 0:
                # the final participation vector of the first seed, for the per-task distribution
                # panel: one real network per task rather than a pooled histogram
                P = np.asarray(tr.get("participation", []), float)
                pit = np.asarray(tr.get("participation_iters", []), float)
                if P.ndim == 2 and len(P):
                    store[f"{task}|participation"] = P[-1]
                    # the iteration the vector was probed at, so the panel can say how long this
                    # network trained rather than leaving the budget to a caption
                    store[f"{task}|participation_iter"] = np.array(pit[-1] if len(pit) else np.nan)
            for var in VARS:
                it, v = series(tr, f"drift_{var}_lag{LAG}")
                if not len(it):
                    continue
                store[f"{task}|{var}|{s}|it"] = it
                store[f"{task}|{var}|{s}|v"] = v
            kept += 1
        print(f"  {task:14s} {kept} seeds, to {store.get(f'{task}|W_rec|0|it', [0]).max():.0f} "
              f"iterations, cell {cell}")
        store[f"{task}|label"] = np.array(label)
        store[f"{task}|cell"] = np.array(cell)
    if not store:
        raise SystemExit("no trajectories found")
    np.savez_compressed(out_path, lag=np.array(LAG), **store)
    print(f"\nwrote {out_path}")
    return out_path


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else os.path.expanduser("~/f2_drift_traces.npz"))
