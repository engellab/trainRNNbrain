#!/usr/bin/env python3
"""
Is the 3-bit flip-flop still silencing at 75,000 iterations? Live units read along each network's
joined 0 -> 75,000 trace, at the operating point ff75k_readout.py draws.

ff75k_readout.py compares two snapshots per network, 40k and 75k. A drop between them does not say
whether the count is still falling at 75k or fell early and levelled off, and that decides whether
75k is a read-out or just a later point on the same slide. Every continuation's
*ParticipationTrace.pkl starts at iteration 0 (Trainer.join_parent_trace prepends the parent), so
the whole path is in the 75k folder.

Values are READ OFF the trace, never fitted: these curves fall to a floor and partly recover, and a
line through a U reports the recovery as growth (the synaptic-noise slide 26e retraction). Each read
is the mean of the trace's snapshots within +-WIN iterations of the target, so one noisy snapshot
cannot set it. The count is fig_dropout_live.trace's scale-free live count, the deck's along-training
measure; it is not the cache's n_active, so compare rows with each other, not with the readout.

Usage:
    python ff75k_traces.py SHARD_DIR [TRAINED_RNNS_ROOT]
SHARD_DIR holds the f2_remedies_cache shard_*.npz built with F2_FF_ITERS=75000 F2_PARENT=1; the
root defaults to the local mirror, data/trained_RNNs.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_paper_F2 as F
from common import DATA_DIR
from fig_dropout_live import trace
from ff75k_readout import SOLVED, merge, view

AT = (40000, 60000, 75000)
WIN = 1500


def read(t, it):
    """Mean scale-free live count of the snapshots within WIN of iteration `it` (nan if none)."""
    k = np.abs(t["pit"] - it) <= WIN
    return float(t["sf"][k].mean()) if k.any() else np.nan


def main(shard_dir, root):
    c = merge(shard_dir)
    bad = (c["r2_common"] < SOLVED) | (c["p40_r2_common"] < SOLVED)
    c = {k: (v[~bad] if isinstance(v, np.ndarray) and len(v) == len(bad) else v) for k, v in c.items()}
    print(f"{len(c['arm'])} solved networks; live units (scale-free) along the joined trace\n")
    print(f"  {'':16s}  {'40k':>7s} {'60k':>7s} {'75k':>7s}   {'40->60':>7s} {'60->75':>7s}"
          f"   floor (at iter)   still falling at 75k")
    for N in F.SIZES:
        at = F.restrict(view(c, "75"), n_units=N)
        print(f"  N = {N}")
        for arm, _, _, _ in F.ARMS:
            if not (at["arm"] == arm).any():
                continue
            cell = F.pick_cell(at, arm)[0] if arm in F.GRID_ARMS else None
            m = (c["arm"] == arm) & (c["N"] == N) & ((c["cell"] == cell) if cell else True)
            rows, floors, falling = [], [], 0
            for cl, net in zip(c["cell"][m], c["net"][m]):
                rt, sub = cl.split("/", 1)
                t = trace(os.path.join(root, rt + "_c75000", sub, net))
                if t is None or t["pit"].max() < AT[-1] - WIN:
                    continue
                v = [read(t, it) for it in AT]
                rows.append(v)
                i = int(np.argmin(t["sf"]))
                floors.append((t["sf"][i], t["pit"][i]))
                falling += v[2] < v[1]
            if not rows:
                print(f"    {arm:9s} no trace found under {root}")
                continue
            r = np.array(rows)
            fl = np.array(floors)
            print(f"    {arm:9s} n={len(r):2d}  {r[:, 0].mean():7.0f} {r[:, 1].mean():7.0f} {r[:, 2].mean():7.0f}"
                  f"   {np.mean(r[:, 1] - r[:, 0]):+7.0f} {np.mean(r[:, 2] - r[:, 1]):+7.0f}"
                  f"   {fl[:, 0].mean():5.0f} ({np.median(fl[:, 1]):6.0f})   {falling}/{len(r)}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else os.path.join(DATA_DIR, "trained_RNNs"))
