#!/usr/bin/env python3
"""
Per-unit TEMPORAL participation ratio, frm vs frm+rws, one panel per seed.

    temporal PR_i = (sum_s r_is)^2 / sum_s r_is^2        s = pooled (time, trial) samples

The dual of the PR used everywhere else in this project. `pr_matrix` reports PR over UNITS - the
effective number of units carrying activity. This is PR over TIME - the effective number of samples
each unit is active for. Divided by the sample count it is exactly the Treves-Rolls lifetime
sparseness, so it is bounded in (0, 1] and needs no threshold.

WHY THIS QUANTITY. frm pins each unit's soft-max activity and succeeds: the CV of the frm quantity
across live units is 0.125 under frm and 0.112 under frm+rws, i.e. identical. What frm does NOT
constrain is how much of the time a unit sits near that peak, and that is what temporal PR measures.
It is the sharpest discriminator found among eleven peakiness measures (CV ratio 2.21; duty cycle
2.11-2.17; peak/mean, peak/median, gini and kurtosis all fail to separate the conditions at all).

⚠️ THE MEDIAN RATE IS EXACTLY ZERO FOR MOST UNITS (75% under frm, 59% under both), so median-based
peakiness measures are degenerate here and must not be substituted for this one.

⚠️ SEEDS ARE NOT PAIRED ACROSS CONDITIONS. frm and frm+rws were trained from different random seeds;
panel i shows the i-th seed of each, not a matched pair.

Dead units are excluded (participation < SILENT_FLIPFLOP): temporal PR is 0/0 for an all-zero unit.
The excluded fraction is annotated per panel, because it differs sharply between conditions and is
the OTHER thing rws does.

Output: img/internal_figures/temporal_pr_N{N}_k{k}.png

Usage:  python flipflop_temporal_pr.py [N] [k]        (defaults 2000 3)
"""

import os
import sys
import glob
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from pr_matrix import ROOTS, R2_MIN
from flipflop_fixedpoints import load_net
from flipflop_bouquet import run_trials

N_TRIALS = 64
COLS = {"frm": "#d62728", "both": "#1f77b4"}


def temporal_pr(rates):
    """Per-unit temporal participation ratio and the live mask.

    Args:
        rates: (N, T, B) noise-free firing rates.
    Returns:
        (tpr, live, n_samples): tpr is (N,) with nan for dead units, live is the (N,) boolean
        mask (participation >= SILENT_FLIPFLOP), n_samples = T*B.
    """
    x = rates.reshape(rates.shape[0], -1).astype(np.float64)
    live = (x.std(1) + np.quantile(x, 0.9, axis=1)) >= SILENT_FLIPFLOP
    denom = (x ** 2).sum(1)
    tpr = np.where(denom > 0, x.sum(1) ** 2 / np.maximum(denom, 1e-300), np.nan)
    return tpr, live, x.shape[1]


def seeds_for(pen, N, k):
    """Run folders for one cell, ordered by seed, with their seed ids.

    Args:
        pen: "frm" or "both"; N: hidden size; k: bits.
    Returns:
        list of (folder, seed_id) for runs passing the r2 gate.
    """
    out = []
    for folder in sorted(glob.glob(
            f"{ROOTS['penlong']}/EqType=h_k={k}_N={N}_pen={pen}_iters=*/*/")):
        npz = glob.glob(folder + "*LastParams*.npz")
        if not npz:
            continue
        try:
            if not (float(os.path.basename(npz[0]).split("_")[0]) >= R2_MIN):
                continue
        except ValueError:
            continue
        from omegaconf import OmegaConf
        cfgs = glob.glob(folder + "*_config.yaml")
        out.append((folder, int(OmegaConf.load(cfgs[0]).seed) if cfgs else -1))
    return out


def main():
    """Histogram temporal PR per unit for frm and frm+rws, one panel per seed."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ps.setup()
    data = {}
    for pen in ("frm", "both"):
        for i, (folder, sd) in enumerate(seeds_for(pen, N, k)):
            tpr, live, n = temporal_pr(run_trials(load_net(folder)[0], folder, N_TRIALS))
            data[(pen, i)] = dict(tpr=tpr[live] / n, dead=float((~live).mean()), seed=sd, n=n)
            print(f"  {pen:<5} seed {sd:<11} live {live.mean():.3f}  "
                  f"tPR/n median {np.median(tpr[live])/n:.4f}  CV {np.std(tpr[live])/np.mean(tpr[live]):.3f}")

    n_pan = max(i for (_, i) in data) + 1
    fig, ax = plt.subplots(1, n_pan, figsize=(5.0 * n_pan, 4.4), squeeze=False, sharex=True,
                           sharey=True)
    allv = np.concatenate([d["tpr"] for d in data.values()])
    bins = np.linspace(0, np.percentile(allv, 99.5), 55)
    for i in range(n_pan):
        a = ax[0][i]
        txt = []
        for pen in ("frm", "both"):
            d = data.get((pen, i))
            if d is None:
                continue
            a.hist(d["tpr"], bins=bins, color=COLS[pen], alpha=.5, label=f"{pen} (seed {d['seed']})")
            a.axvline(np.median(d["tpr"]), color=COLS[pen], ls="--", lw=1.6)
            txt.append(f"{pen}: median {np.median(d['tpr']):.3f}, "
                       f"CV {np.std(d['tpr'])/np.mean(d['tpr']):.3f}, dead {100*d['dead']:.1f}%")
        a.set(xlabel="temporal PR / n_samples   (= lifetime sparseness)",
              ylabel="units" if i == 0 else "", title=f"seed pair {i+1}")
        a.text(.97, .97, "\n".join(txt), transform=a.transAxes, fontsize=7.2, va="top", ha="right",
               bbox=dict(fc="white", ec="0.7", alpha=.9, boxstyle="round,pad=0.35"))
        a.legend(fontsize=8, loc="center right")
        a.grid(alpha=.25)
    ns = next(iter(data.values()))["n"]
    fig.suptitle(f"Per-unit temporal participation ratio — N={N}, k={k}   "
                 f"(n_samples = {ns}, {N_TRIALS} trials x 300 steps)\n"
                 "dashed = median  ·  frm pins each unit's PEAK activity but not how long it stays "
                 "there; rws is what compresses this distribution", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return ps.save(fig, f"temporal_pr_N{N}_k{k}", tight=False)


if __name__ == "__main__":
    main()
