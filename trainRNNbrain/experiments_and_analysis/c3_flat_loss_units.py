#!/usr/bin/env python3
"""C3 on the flip-flop: does the active-unit count keep falling while the loss is flat?

The flip-flop's noisy loss is flat from ~150k (agent-c4b). This reads the standard 3-bit flip-flop
networks trained to 200k or 300k, each along its whole warm-start chain:
  - claim-1 lineages at N = 1000: NBitFlipFlop_c1_snapshots (75k) -> _c200000 -> _c300000
    (two of the three were continued to 300k; a 200k run that was continued is read once, as the
    300k run);
  - NBitFlipFlop_ff_revive (40k) -> _c75000 -> _c200000, arm `none`, at N = 500, 1000, 2000.

Per run it reports:
  - the noisy training loss over 150k -> end, as medians of 5k blocks: the last block over the first,
    as a % change and as % per 100k;
  - the active count (fig_paper_F1.runs_with_loss: noise-free participation >= 5% of its q95) at the
    run's 1.05x noisy-loss floor (common.noisy_floor), at 150k, 200k and 300k (where the run gets that
    far; the probe nearest each iteration);
  - units lost per 100k after 150k: minus the least-squares slope of the active count on every
    participation probe at >= 150k, with its standard error.
The test, written before any number was computed, is in docs/notes/agent-c3.md.

Usage:
    python trainRNNbrain/experiments_and_analysis/c3_flat_loss_units.py [ROOT]
ROOT is the folder holding the sweeps (default: common.DATA_DIR). On a local copy whose
`init_from` paths do not exist, each run's parent must sit inside it as a `parent_*` folder (the
layout common.warm_start_chain falls back to).
"""

import glob
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_paper_F1 as F1
from common import DATA_DIR, noisy_floor, warm_start_chain

CELLS = [("N=1000 claim-1", "NBitFlipFlop_c1_snapshots_c300000/EqType=h_N=1000_arm=control"),
         ("N=1000 claim-1", "NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control"),
         ("N=500", "NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=500_pen=none_arm=none"),
         ("N=1000 ff_revive", "NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none"),
         ("N=2000", "NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=2000_pen=none_arm=none")]
T_FLAT = 150_000        # start of the flat-loss window
BLOCK = 5_000           # loss block length for the medians
READS = (150_000, 200_000, 300_000)
FLAT_PCT = 3.0          # |loss change| below this % per 100k counts as flat (notes, written first)
DIE_BAR = 5.0           # units lost per 100k: below this, "stopped dying"


def loss_change(it, L, t0=T_FLAT, block=BLOCK):
    """Change of the noisy loss from t0 to the end of the run, on 5k-block medians.

    Args:
        it, L: per-iteration noisy loss and its (absolute, 1-based) iterations.
        t0: start of the window; block: block length in iterations.
    Returns:
        (first-block median, last-block median, % change, % change per 100k).
    """
    edges = np.arange(t0, it[-1] + 1, block)
    med = [np.median(L[(it > a) & (it <= a + block)]) for a in edges[:-1]]
    span = (edges[-2] - edges[0]) / 1e5          # centre of the first block to centre of the last
    pct = 100.0 * (med[-1] / med[0] - 1.0)
    return med[0], med[-1], pct, pct / span


def loss_rate(ita, act, t0=T_FLAT):
    """Units lost per 100k iterations after t0: minus the least-squares slope of the active count.

    Args:
        ita: participation probe iterations; act: active count at each probe; t0: window start.
    Returns:
        (units lost per 100k, its standard error).
    """
    m = ita >= t0
    x, y = ita[m] / 1e5, act[m]
    A = np.vstack([x, np.ones_like(x)]).T
    coef, res, _, _ = np.linalg.lstsq(A, y, rcond=None)
    resid = y - A @ coef
    se = np.sqrt(resid @ resid / (len(y) - 2) / np.sum((x - x.mean()) ** 2))
    return -coef[0], se


def at(ita, act, t):
    """Active count at the probe nearest iteration t, or nan if the run ends before t.

    Args:
        ita, act: probe iterations and active counts; t: iteration.
    Returns:
        float active count or nan.
    """
    if t > ita[-1] + 100:
        return float("nan")
    return float(act[np.argmin(np.abs(ita - t))])


def main(root):
    """Read every run, print the per-run table and the per-size verdict of the written test.

    Args:
        root: folder holding the sweeps.
    """
    seen, rows = set(), []
    for size, rel in CELLS:
        for r in F1.runs_with_loss(os.path.join(root, rel)):
            run_dir = os.path.join(root, rel, r["name"])
            chain = [os.path.basename(d.rstrip("/")).replace("parent_", "", 1)
                     for d in warm_start_chain(run_dir)]
            if r["name"] in seen:          # a 200k run already read as part of a 300k lineage
                continue
            seen.update(chain)
            it, L = np.asarray(r["it_loss"], float), np.asarray(r["loss"], float)
            ita, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
            f = noisy_floor(it, L)
            l0, l1, pct, pct100 = loss_change(it, L)
            lost, se = loss_rate(ita, act)
            rows.append(dict(size=size.split()[0], label=size, seed=r["name"].split("seed=")[1].split(";")[0],
                             end=it[-1], links=len(chain), t_star=f["t_star"], kind=f["reason"] or f["floor_kind"],
                             a_floor=at(ita, act, f["t_star"]) if np.isfinite(f["t_star"]) else float("nan"),
                             a150=at(ita, act, 150_000), a200=at(ita, act, 200_000), a300=at(ita, act, 300_000),
                             l0=l0, l1=l1, pct=pct, pct100=pct100, lost=lost, se=se))

    print(f"{'size':<18}{'seed':<12}{'end k':>6}{'chain':>6}{'floor k':>8} {'kind':<13}{'@floor':>7}"
          f"{'@150k':>6}{'@200k':>6}{'@300k':>6}{'L150':>8}{'Lend':>8}{'dL%':>7}{'dL%/100k':>9}"
          f"{'lost/100k':>10}{'se':>5}")
    for r in rows:
        print(f"{r['label']:<18}{r['seed']:<12}{r['end'] / 1e3:>6.0f}{r['links']:>6}{r['t_star'] / 1e3:>8.0f} "
              f"{str(r['kind'])[:12]:<13}{r['a_floor']:>7.0f}{r['a150']:>6.0f}{r['a200']:>6.0f}{r['a300']:>6.0f}"
              f"{r['l0']:>8.4f}{r['l1']:>8.4f}{r['pct']:>+7.1f}{r['pct100']:>+9.1f}{r['lost']:>10.1f}{r['se']:>5.1f}")

    print(f"\nTest (docs/notes/agent-c3.md): flat = |loss change| < {FLAT_PCT}% per 100k; "
          f"'stopped' = loses < {DIE_BAR} units per 100k")
    for size in dict.fromkeys(r["size"] for r in rows):
        sub = [r for r in rows if r["size"] == size]
        flat = [r for r in sub if abs(r["pct100"]) < FLAT_PCT]
        dying = [r for r in flat if r["lost"] >= DIE_BAR]
        if not flat:
            verdict = "no flat-loss run"
        elif 3 * len(dying) >= 2 * len(flat):
            verdict = "units keep dying while the loss is flat"
        elif 3 * (len(flat) - len(dying)) >= 2 * len(flat):
            verdict = "units stop dying when the loss is flat"
        else:
            verdict = "mixed"
        print(f"  {size}: {len(sub)} runs, {len(flat)} flat, {len(dying)} of the flat ones lose >= {DIE_BAR}"
              f" per 100k -> {verdict}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
