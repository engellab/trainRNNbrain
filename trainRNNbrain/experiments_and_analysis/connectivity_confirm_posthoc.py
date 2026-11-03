#!/usr/bin/env python3
"""
Descriptive numbers for the paper text of the r12-weights confirmatory test, computed after its
verdict (review of dead-relu-paper PR #178). Nothing here changes the rule's verdict
(notebook 2026-11-02_2218, result 2026-11-02_2232).

(1) rel = (S_all - S_act) / S_act of W1 (s.d. of the recurrent weights), grouped by task and by the
    iteration of the weights used, with the gap to the floor crossing.
(2) W1 over all units and over active units in the weight-decay-0 CDDM networks (#40-45).
(3) In every CDDM network of the main test (weight decay 1e-6): the slope of the inactive-inactive
    weights on their initial values, the s.d. of those weights over their initial s.d., and the s.d.
    over all units over the initial s.d.

Usage:  python connectivity_confirm_posthoc.py   (reads data/connectivity_confirm.json and the
        network folders of data/connectivity_confirm/)
"""

import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import connectivity_confirm as K          # noqa: E402


def by_read_point(res):
    """Print -rel of W1 per task and read iteration, with the gap read iteration - t*."""
    print("(1) W1: -rel by task and read iteration")
    for task in ("3BFF", "CDDM", "DMTS"):
        rows = [r for r in res if r["task"] == task]
        for it in sorted({r["read_iter"] for r in rows}):
            g = [r for r in rows if r["read_iter"] == it]
            rel = [-r["rel"]["W1"] for r in g]
            gaps = [r["read_iter"] - r["t_star"] for r in g if r["t_star"] == r["t_star"]]
            gap = f"gap {min(gaps):+.0f}..{max(gaps):+.0f}" if gaps else "no floor crossing"
            print(f"  {task} {it:>7}: n={len(g):>2}  -rel {min(rel):.3f}-{max(rel):.3f}  {gap}")


def block_stats(d, act):
    """Slope and s.d. ratios of one network folder.

    Args:
        d: network folder; act: (k,) int active indices.
    Returns:
        dict slope_II, sd_II_ratio (s.d. of W[I,I] / s.d. of W0[I,I]), sd_all_ratio (s.d. of W / s.d. of W0),
        all, act (W1 over all units and over the active units).
    """
    net = K.load_network(d)
    W, W0 = net["W"], net["W0"]
    I = np.setdiff1d(np.arange(W.shape[0]), act)
    w, w0 = W[np.ix_(I, I)].ravel(), W0[np.ix_(I, I)].ravel()
    return dict(slope_II=float(w @ w0 / (w0 @ w0)), sd_II_ratio=float(w.std() / w0.std()),
                sd_all_ratio=float(W.std() / W0.std()), all=float(W.std()),
                act=float(W[np.ix_(act, act)].std()))


def main():
    """Print the three sets of numbers."""
    with open(K.OUT) as fh:
        out = json.load(fh)
    res = out["results"]
    by_read_point(res)
    folders = {idx: d for idx, d, _ in K.folders()}
    print("(2) weight decay 0, CDDM: W1 over all units / over active units")
    for x in out["slopes"]:
        if x["arm"] != "wd 0":
            continue
        d = folders[x["id"]]
        b = block_stats(d, K.active_set(d, K.load_network(d)["params"]))
        print(f"  {x['id']}: all {b['all']:.4f}  active {b['act']:.4f}  -rel {1 - b['all'] / b['act']:.3f}"
              f"  all / initial {b['sd_all_ratio']:.2f}")
    print("(3) weight decay 1e-6, all CDDM networks of the main test")
    rows = []
    for r in res:
        if r["task"] != "CDDM":
            continue
        b = block_stats(folders[r["id"]], np.asarray(r["active"]))
        rows.append((r["id"], r["read_iter"], b))
        print(f"  {r['id']} ({r['read_iter']}): slope {b['slope_II']:+.4f}  sd II / initial {b['sd_II_ratio']:.3f}"
              f"  sd all / initial {b['sd_all_ratio']:.2f}")
    sl = [b["slope_II"] for _, _, b in rows]
    print(f"  slope range {min(sl):+.4f}..{max(sl):+.4f}; sd II / initial "
          f"{min(b['sd_II_ratio'] for _, _, b in rows):.3f}-{max(b['sd_II_ratio'] for _, _, b in rows):.3f}; "
          f"sd all / initial {min(b['sd_all_ratio'] for _, _, b in rows):.2f}-{max(b['sd_all_ratio'] for _, _, b in rows):.2f}")


if __name__ == "__main__":
    main()
