#!/usr/bin/env python3
"""
Elaboration claim CS4, the part the switch experiment never measured: does turning the sparsity
penalty on or off in a trained network change the units' SELECTIVITY, not only their persistence?

For each of the four arms (A1 frm -> frm+rws, A2 frm -> frm, A3 both -> frm, A4 both -> both) and each
of the three paired seeds, the parent network (the 400k penlong net the arm was warm-started from) and
the arm's endpoint (50k iterations later) are simulated once, and unit_stats.flipflop_one gives, per
network, the number of mixed-selective units, the number of burst units, and the median Hoyer of the
tuning coefficients over tuned units. Parents are matched to arms exactly as the launcher did: the
r2 >= 0.5 run folders of the penlong cell sorted by name, index = rep.

Output: img/internal_figures/fig_CS4_selectivity.png (start -> end per arm, three statistics)
Usage:  python flipflop_switch_selectivity.py
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from unit_stats import flipflop_one

CACHE = "data/switch_selectivity_cache.pkl"
PENLONG = "data/trained_RNNs/NBitFlipFlop_std_penlong/EqType=h_k=3_N=2000_pen={pen}_iters=400000"
SWITCH = "data/trained_RNNs/NBitFlipFlop_std_switch/warm_EqType=h_k=3_N=2000_arm={arm}_from={pen}_rep={rep}_iters=50000"
ARMS = [("A1", "frm", "participation → participation + sparsity"), ("A2", "frm", "participation → participation (control)"),
        ("A3", "both", "both → participation (sparsity removed)"), ("A4", "both", "both → both (control)")]
STATS = [("mixed", "mixed-selective units"), ("burst", "burst units"), ("hoyer_med", "median Hoyer of tuning coefficients")]


def scored_runs(cell):
    """Run folders of a cell with a numeric r2 prefix >= 0.5, sorted by name (the launcher's rule)."""
    out = []
    for d in sorted(glob.glob(os.path.join(cell, "*"))):
        try:
            if float(os.path.basename(d).split("_")[0]) >= 0.5 and glob.glob(os.path.join(d, "*LastParams*.npz")):
                out.append(d)
        except ValueError:
            continue
    return out


def stats_for(folder, cache):
    """Cached per-network statistics for one folder."""
    if folder not in cache:
        st, _ = flipflop_one(folder)
        hy = np.asarray(st["hoyer"])[st["tuned_mask"]]
        cache[folder] = dict(mixed=st["mixed"], burst=st["burst"], live=st["live"], hoyer_med=float(np.median(hy[np.isfinite(hy)])))
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"  {os.path.basename(os.path.dirname(folder))[:60]} / {os.path.basename(folder)[:12]}: {cache[folder]}", flush=True)
    return cache[folder]


def main():
    """Compute start/end statistics per arm and draw fig_CS4_selectivity.png."""
    ps.setup()
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    res = {}
    for arm, pen, _ in ARMS:
        parents = scored_runs(PENLONG.format(pen=pen))
        for rep in range(3):
            ends = scored_runs(SWITCH.format(arm=arm, pen=pen, rep=rep))
            if rep >= len(parents) or not ends:
                continue
            res.setdefault(arm, []).append((stats_for(parents[rep], cache), stats_for(ends[0], cache)))
    fig, ax = plt.subplots(1, 3, figsize=(13, 4.2))
    for j, (key, lab) in enumerate(STATS):
        for i, (arm, pen, desc) in enumerate(ARMS):
            pairs = res.get(arm, [])
            s = np.array([p[0][key] for p in pairs]); e = np.array([p[1][key] for p in pairs])
            ax[j].plot([i - 0.18, i + 0.18], [s.mean(), e.mean()], "o-", color=plt.cm.tab10(i), lw=2)
            for a, b in zip(s, e):
                ax[j].plot([i - 0.18, i + 0.18], [a, b], "-", color=plt.cm.tab10(i), alpha=.3, lw=1)
            print(f"{arm} {desc:42s} {lab:36s} {s.mean():8.2f} -> {e.mean():8.2f}  (n={len(pairs)})")
        ax[j].set(xticks=range(len(ARMS)), xticklabels=[a[0] for a in ARMS], ylabel=lab, title=f"{lab}: start → end")
    fig.suptitle("CS4 — selectivity across the penalty switch (N=2000, k=3; A1 frm→both, A2 frm→frm, A3 both→frm, A4 both→both; 3 paired seeds)",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, "fig_CS4_selectivity", tight=False)


if __name__ == "__main__":
    main()
