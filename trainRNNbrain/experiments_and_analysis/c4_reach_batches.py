#!/usr/bin/env python3
"""3BFF gradient reach on several seeded trial batches (dead-relu-paper #112; Methods, Gradient reach).

Results 4 measures, on the Figure 1d weights of the three 3BFF networks of Fig. 4a-c, which inactive
units the incoming-weight gradient reaches and how many of those are active 500 and 5,500 iterations
later. fig_paper_F4_noreturn.fig1d does it on one noise-free batch (task seed 12345). This script
repeats the same measurement on other seeded batches of the same size, so the paper can state the
range across batches. Rule and pass conditions: notebook entry
2026-11-01_1143_c4-reach-across-3bff-batches-rule-set-before-the-run.md.

Per network and batch (the definitions of fig_paper_F4_noreturn.load_c4):
  active      p >= 0.05 q95(p), p = std(r) + q0.9(|r|) over every step and trial of the batch
  silent      inactive, x <= 0 on every step of every trial of the batch
  low-active  every other inactive unit
  reached     g_in >= 0.05 q95(g_in), g_in the norm of the loss gradient on the unit's rows of W_rec
              and W_inp (c4_gradient_mechanism.backward, strict ReLU gate)
  later       participation of the weights 500 and 5,500 iterations later on the same batch

Usage (Mac, checkout root, CPU only):
  python trainRNNbrain/experiments_and_analysis/c4_reach_batches.py [--seeds 12345 101 ...]
Output: data/fig4_noreturn/c4_reach_batches.json and a printed table.
"""

import argparse
import glob
import json
import os
import sys
import time

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import c4_gradient_mechanism as C4
from common import DATA_DIR, active_mask
from fig_paper_F4_noreturn import CELL, C4_RUNS, LATER, POP_DIR, SNAP_LOCAL, participation

OUT = os.path.join(DATA_DIR, "..", "fig4_noreturn", "c4_reach_batches.json")
SEEDS = (12345, 101, 102, 103, 104, 105)    # 12345 is the paper's batch (reproduction check)
LATER_FILES = {LATER[0]: "d0", LATER[1]: "d5000"}   # snapshot copies: d0 = +500, d5000 = +5,500


def measure(folder, seed, task_seed):
    """Classes, reach and later activity of one 3BFF network on one seeded noise-free batch.

    Args:
        folder: the network's data/pop_stats_nets folder (config and Figure 1d weights);
        seed: the network's seed (int), naming its later snapshots in SNAP_LOCAL;
        task_seed: the task seed of the batch (int).
    Returns:
        dict with B (trials), active, silent, low, reached (counts of low-active units reached), and
        share_<dt> for dt in LATER: the share of the reached low-active units active dt iterations
        later.
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    P = C4.load_params(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0])
    u, y, mask = C4.batch(cfg, task_seed)
    H, U, no, out = C4.forward(P, u)
    act = active_mask(participation(H))
    sil = ~act & ((H > 0).sum((0, 2)) == 0)
    low = ~act & ~sil
    G = C4.backward(P, H, U, no, out, y, mask, 0.0)
    del H, U
    reach = active_mask(C4.g_in_out(G)[0])
    low_r = low & reach
    res = dict(B=int(u.shape[2]), active=int(act.sum()), silent=int(sil.sum()), low=int(low.sum()),
               reached=int(low_r.sum()))
    for dt, tag in LATER_FILES.items():
        Hl, _, _, _ = C4.forward(C4.load_params(os.path.join(SNAP_LOCAL, f"{seed}_{tag}.npz")), u)
        res[f"share_{dt}"] = float(active_mask(participation(Hl))[low_r].mean())
        del Hl
    return res


def main(seeds):
    """Run measure() on every 3BFF network of C4_RUNS and every task seed; write OUT and print.

    Args:
        seeds: iterable of task seeds (int).
    Returns:
        dict network seed (str) -> task seed (str) -> measure() dict.
    """
    rec = {}
    for task, sweep, net_seeds in C4_RUNS:
        if task != "3BFF":
            continue
        for s in net_seeds:
            folder = os.path.join(POP_DIR, f"{sweep}__{CELL}__s{s}__floor")
            for ts in seeds:
                t0 = time.time()
                r = measure(folder, s, ts)
                rec.setdefault(str(s), {})[str(ts)] = r
                print(f"3BFF s{s} batch {ts} (B = {r['B']}): active {r['active']}, silent {r['silent']}, "
                      f"low-active {r['low']}, reached {r['reached']} ({r['reached'] / r['low']:.1%}); "
                      f"active at +{LATER[0]} {r[f'share_{LATER[0]}']:.1%}, +{LATER[1]} "
                      f"{r[f'share_{LATER[1]}']:.1%}  [{time.time() - t0:.0f} s]", flush=True)
    with open(OUT, "w") as fh:
        json.dump(rec, fh, indent=1)
    print(f"\nwrote {OUT}\n\nRange over {len(seeds)} batches per network:")
    for s, by in rec.items():
        v = list(by.values())
        rng = lambda k: (min(x[k] for x in v), max(x[k] for x in v))
        sh = [x["reached"] / x["low"] for x in v]
        print(f"  s{s}: low-active {rng('low')}, reached {rng('reached')}, reached share "
              f"{min(sh):.1%}-{max(sh):.1%}, active at +{LATER[1]} "
              f"{rng(f'share_{LATER[1]}')[0]:.1%}-{rng(f'share_{LATER[1]}')[1]:.1%}")
    allv = [x for by in rec.values() for x in by.values()]
    sh = [x["reached"] / x["low"] for x in allv]
    late = [x[f"share_{LATER[1]}"] for x in allv]
    print(f"  all networks and batches: reached share {min(sh):.1%}-{max(sh):.1%}; active at "
          f"+{LATER[1]} {min(late):.1%}-{max(late):.1%}; bound 'fewer than a tenth' "
          f"{'holds' if max(late) < 0.10 else 'FAILS'}")
    return rec


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=list(SEEDS), help="task seeds of the batches")
    main(ap.parse_args().seeds)
