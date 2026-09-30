"""Is 40,000 iterations enough, and is it enough at N = 4000?

Two questions, each answered from records already on disk rather than from an argument.

  1. HAS THE LOSS STOPPED FALLING at the read-out? For each cell, the iteration at which the
     training loss first comes within 10% of its final value, and how much further it falls over the
     last quarter of training. A cell still improving at 40,000 is being read mid-descent.

  2. HAS THE ACTIVE COUNT STOPPED FALLING? This is the one that matters for this paper. Units go
     silent as training proceeds, so a network read before its count settles is reported with MORE
     active units than it would have at convergence - the confound runs against the paper's claim
     at small N and in favour of it at large N. The participation trace gives the count at every
     probe; the slope over the last fifth of training says whether it has settled.

Both are reported per network size, so the question "is 40k enough at N = 4000" is answered with the
same statistic at every size rather than by eye.
"""
import glob
import json
import os
import pickle
import sys

import numpy as np

D = os.environ.get("F2_DATA", "/home/pt1290/trainRNNbrain/data/trained_RNNs")
SILENT_REL = 0.05

CELLS = [
    ("control", 500, "NBitFlipFlop_ff_revive/EqType=h_k=3_N=500_pen=none_arm=none"),
    ("control", 1000, "NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none"),
    ("control", 2000, "NBitFlipFlop_ff_revive/EqType=h_k=3_N=2000_pen=none_arm=none"),
    ("control", 4000, "NBitFlipFlop_paper_grid/EqType=h_N=4000_arm=control"),
    ("duplicate", 500, "NBitFlipFlop_paper_grid/EqType=h_N=500_arm=duplication"),
    ("duplicate", 1000, "NBitFlipFlop_paper_grid/EqType=h_N=1000_arm=duplication"),
    ("duplicate", 2000, "NBitFlipFlop_paper_grid/EqType=h_N=2000_arm=duplication"),
    ("duplicate", 4000, "NBitFlipFlop_paper_grid/EqType=h_N=4000_arm=duplication"),
    ("mute", 500, "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=500_pen=none_do=mute_rate=0.20_beta=4"),
    ("mute", 1000, "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=1000_pen=none_do=mute_rate=0.20_beta=4"),
    ("mute", 2000, "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=2000_pen=none_do=mute_rate=0.20_beta=4"),
    ("mute", 4000, "NBitFlipFlop_dropout_sizes/EqType=h_k=3_N=4000_pen=none_do=mute_rate=0.20_beta=4"),
    # the direct test of the budget: the same two arms at N = 1000, trained 150,000 iterations
    ("duplicate 150k", 1000, "NBitFlipFlop_paper_grid_150k/EqType=h_N=1000_arm=duplication"),
    ("mute 150k", 1000, "NBitFlipFlop_paper_grid_150k/EqType=h_N=1000_arm=mute"),
]


def loss_profile(net_dir, smooth=500):
    """When the training loss settles, and how much it still falls afterwards.

    Args:
        net_dir: one trained-network folder; smooth: boxcar width in iterations.
    Returns:
        (n_iter, frac_to_settle, last_quarter_drop) - the iteration count, the fraction of training
        at which the smoothed loss first comes within 10% of its final value, and the relative fall
        of the smoothed loss over the last quarter, or None if the record is missing.
    """
    fs = glob.glob(os.path.join(net_dir, "*TrainLosses.json"))
    if not fs:
        return None
    L = np.asarray(json.load(open(fs[0]))["train_losses"], float)
    if L.size < 100:
        return None
    k = max(1, min(smooth, L.size // 20))
    sm = np.convolve(L, np.ones(k) / k, mode="valid")
    final = sm[-1]
    hit = np.argmax(sm <= final * 1.10)
    q = len(sm) // 4
    return int(L.size), float(hit / len(sm)), float((sm[-q] - final) / max(final, 1e-12))


def active_profile(net_dir, tail_frac=0.2):
    """The active-unit count along training, and whether it has stopped falling.

    Args:
        net_dir: one trained-network folder; tail_frac: fraction of training used for the slope.
    Returns:
        (count at the end, count at the start of the tail, percent change over the tail), or None.
    """
    fs = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
    if not fs:
        return None
    d = pickle.load(open(fs[0], "rb"))
    P = np.asarray(d.get("participation", []), float)
    if P.ndim != 2 or len(P) < 5:
        return None
    counts = np.array([(p >= SILENT_REL * np.quantile(p, 0.95)).sum() for p in P], float)
    j = int(len(counts) * (1 - tail_frac))
    return float(counts[-1]), float(counts[j]), float(100.0 * (counts[-1] - counts[j]) / counts[j])


if __name__ == "__main__":
    print(f"{'arm':>15s} {'N':>5s} {'n':>2s} {'iters':>7s} {'settles at':>11s} "
          f"{'loss still':>11s} {'active at':>10s} {'active':>8s} {'change over':>12s}")
    print(f"{'':>15s} {'':>5s} {'':>2s} {'':>7s} {'(% of run)':>11s} "
          f"{'falling':>11s} {'80% of run':>10s} {'at end':>8s} {'last 20%':>12s}")
    print("-" * 100)
    for arm, n_units, pat in CELLS:
        rows_l, rows_a, iters = [], [], []
        for nd in sorted(glob.glob(os.path.join(D, pat, "*"))):
            if not os.path.isdir(nd):
                continue
            lp, ap = loss_profile(nd), active_profile(nd)
            if lp:
                iters.append(lp[0])
                rows_l.append(lp[1:])
            if ap:
                rows_a.append(ap)
        if not rows_l and not rows_a:
            print(f"{arm:>15s} {n_units:5d}  - no records on disk")
            continue
        n = max(len(rows_l), len(rows_a))
        it = int(np.mean(iters)) if iters else 0
        settle = f"{100 * np.mean([r[0] for r in rows_l]):9.0f}%" if rows_l else "        -"
        drop = f"{100 * np.mean([r[1] for r in rows_l]):9.1f}%" if rows_l else "        -"
        if rows_a:
            end = np.mean([r[0] for r in rows_a])
            mid = np.mean([r[1] for r in rows_a])
            chg = np.mean([r[2] for r in rows_a])
            a = f"{mid:10.0f} {end:8.0f} {chg:+11.1f}%"
        else:
            a = f"{'-':>10s} {'-':>8s} {'-':>12s}"
        print(f"{arm:>15s} {n_units:5d} {n:2d} {it:7d} {settle:>11s} {drop:>11s} {a}")
