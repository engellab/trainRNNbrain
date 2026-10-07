#!/usr/bin/env python3
"""
The 3-bit flip-flop read at 75,000 iterations against the same networks at 40,000.

Every network in the 75k cache is a continuation of a 40k one, and the cache scores both
(f2_remedies_cache.py with F2_PARENT=1: the parent's fields carry a p40_ prefix). So every
comparison here is PAIRED: the same network, read twice, never two different sets of seeds.

The cell each arm is drawn at is chosen by fig_paper_F2.pick_cell, the rule the deck uses, applied
to the 75k cell names with the "_c75000" suffix removed so the operating-point patterns match.

Usage (on the cluster):
    python ff75k_readout.py SHARD_DIR [OLD_40K_CACHE]
SHARD_DIR holds shard_*.npz; OLD_40K_CACHE is the cache behind the current deck, read only to check
that this run's 40k numbers reproduce it.
"""
import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import fig_paper_F2 as F

KEYS = ("n_active", "r2_common", "r2_clean", "dims", "w_sigma_log")
SOLVED = 0.8   # r2_common below this is a network that lost the task, reported and set aside


def merge(shard_dir):
    """Concatenate the shard caches into one dict of arrays, one row per network."""
    parts = [dict(np.load(f, allow_pickle=True)) for f in sorted(glob.glob(os.path.join(shard_dir, "shard_*.npz")))]
    keys = sorted(set.intersection(*[set(p) for p in parts]) - {"log_bins"})
    c = {k: np.concatenate([p[k] for p in parts]) for k in keys}
    c["cell"] = np.array([s.replace("_c75000", "") for s in c["cell"]])
    return c


def view(c, at):
    """The cache as one read-out: '75' as stored, '40' with every p40_ field moved into place."""
    if at == "75":
        return dict(c)
    v = dict(c)
    for k in KEYS:
        v[k] = c["p40_" + k]
    return v


def fmt(x):
    x = np.asarray(x, float)
    if not len(x):
        return "      -      "
    return f"{x.mean():6.1f} ± {x.std(ddof=1) if len(x) > 1 else 0:4.1f}"


def main(shard_dir, old=None):
    c = merge(shard_dir)
    n = len(c["arm"])
    print(f"{n} networks, {len(set(c['cell']))} cells, {len(glob.glob(os.path.join(shard_dir, 'shard_*.npz')))} shards\n")

    # 1. networks that lost the task, at either read-out
    bad = (c["r2_common"] < SOLVED) | (c["p40_r2_common"] < SOLVED)
    print(f"== {bad.sum()} network(s) below r2_common {SOLVED} at 40k or 75k (set aside below)")
    for i in np.where(bad)[0]:
        print(f"   {c['arm'][i]:9s} N={c['N'][i]:4d} {c['cell'][i].split('/', 1)[1][:60]:60s} "
              f"r2 40k {c['p40_r2_common'][i]:9.3f} -> 75k {c['r2_common'][i]:9.3f}  "
              f"active {c['p40_n_active'][i]:4.0f} -> {c['n_active'][i]:4.0f}")
    ok = ~bad
    cc = {k: (v[ok] if isinstance(v, np.ndarray) and len(v) == n else v) for k, v in c.items()}

    # 2. reproduction check against the deck's 40k cache, cell by cell
    if old:
        o = dict(np.load(old, allow_pickle=True))
        o_ff = o["task"] == "NBitFlipFlop"
        print("\n== this run's 40k vs the deck's cache (same cells; different seed sets where a cell"
              " had networks that were not continued)")
        for arm, _, _, _ in F.ARMS:
            for N in F.SIZES:
                m_new = (cc["arm"] == arm) & (cc["N"] == N)
                if not m_new.any():
                    continue
                at_new = F.restrict(view(cc, "40"), n_units=N)
                cell = F.pick_cell(at_new, arm)[0] if arm in F.GRID_ARMS else None
                sel = m_new & ((cc["cell"] == cell) if cell else True)
                m_old = o_ff & (o["arm"] == arm) & (o["N"] == N) & ((o["cell"] == cell) if cell else True)
                print(f"   {arm:9s} N={N:4d} active: deck {fmt(o['n_active'][m_old])} (n={m_old.sum()})"
                      f"   this run {fmt(cc['p40_n_active'][sel])} (n={sel.sum()})")

    # 3. each arm at its operating point, every size, 40k vs 75k on the same networks
    print("\n== operating point per arm and size: active units / r2_common / dims, 40k -> 75k")
    for N in F.SIZES:
        print(f"  N = {N}")
        at = F.restrict(view(cc, "75"), n_units=N)
        for arm, _, _, _ in F.ARMS:
            if not (at["arm"] == arm).any():
                continue
            cell, how = (F.pick_cell(at, arm) if arm in F.GRID_ARMS else (None, "all cells"))
            m = (cc["arm"] == arm) & (cc["N"] == N) & ((cc["cell"] == cell) if cell else True)
            d = cc["n_active"][m] - cc["p40_n_active"][m]
            print(f"    {arm:9s} n={m.sum():2d}  active {fmt(cc['p40_n_active'][m])} -> {fmt(cc['n_active'][m])}"
                  f"  (paired change {d.mean():+6.1f}, range {d.min():+.0f}..{d.max():+.0f})"
                  f"  r2 {cc['p40_r2_common'][m].mean():.3f} -> {cc['r2_common'][m].mean():.3f}"
                  f"  dims {np.nanmean(cc['p40_dims'][m]):5.2f} -> {np.nanmean(cc['dims'][m]):5.2f}"
                  f"  [{(cell or how).split('/')[0]}]")

    # 4. every grid cell at N = 1000
    print("\n== every cell at N = 1000, 40k -> 75k (mean over networks)")
    m1 = cc["N"] == 1000
    for arm, _, _, _ in F.ARMS:
        cells = sorted(set(cc["cell"][m1 & (cc["arm"] == arm)]))
        if not cells:
            continue
        print(f"  {arm}")
        for cell in cells:
            m = m1 & (cc["cell"] == cell)
            print(f"    {cell[13:][:78]:78s} n={m.sum()}  active {cc['p40_n_active'][m].mean():5.0f} -> "
                  f"{cc['n_active'][m].mean():5.0f}  r2 {cc['p40_r2_common'][m].mean():.3f} -> "
                  f"{cc['r2_common'][m].mean():.3f}")

    # 5. one number for the whole set
    d = cc["n_active"] - cc["p40_n_active"]
    print(f"\n== all {ok.sum()} solved networks: active changes by {d.mean():+.1f} on average "
          f"(median {np.median(d):+.0f}); {np.mean(d < 0):.0%} lose units, {np.mean(d > 0):.0%} gain;"
          f" r2_common {cc['p40_r2_common'].mean():.4f} -> {cc['r2_common'].mean():.4f}")
    for arm, _, _, _ in F.ARMS:
        m = cc["arm"] == arm
        print(f"   {arm:9s} n={m.sum():3d}  mean change {d[m].mean():+6.1f}  "
              f"as a fraction of the 40k count {np.mean(d[m] / cc['p40_n_active'][m]):+.1%}")


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else None)
