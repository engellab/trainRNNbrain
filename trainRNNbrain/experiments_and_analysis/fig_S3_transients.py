#!/usr/bin/env python3
"""
Figure for elaboration claim S3: the participation penalty is gamed by transients.

The penalty scores each unit by a soft maximum of its rate over (time, trials), so a unit can satisfy
it with brief bursts and sit near zero the rest of the time. The statistic that exposes this is the
per-unit TEMPORAL participation ratio divided by the sample count (= lifetime sparseness, the
fraction of samples a unit is effectively active for; Treves-Rolls):

    tPR_i / n = (sum_s r_is)^2 / (n * sum_s r_is^2)         s = pooled (time, trial) samples

A unit that follows one bit state has tPR/n at that state's duty cycle (0.36 for this task, at every
k); a burst unit has tPR/n near 0. Flip-flop, k = 3, N = 2000, four conditions, three seeds each,
64 noise-free trials per network (as in flipflop_temporal_pr.py).

  left   distribution of tPR/n over live units, one histogram per condition (seeds pooled), with the
         task duty cycle marked. The participation penalty alone puts the MODE at ~0; adding the
         sparsity penalty moves it to the duty cycle.
  right  per-condition summary (mean ± sd over seeds): median tPR/n and the fraction of live units with tPR/n < 0.05 (burst units).

Usage:  python fig_S3_transients.py [N] [k]       (defaults 2000 3)
Output: img/internal_figures/fig_S3_transients.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from flipflop_temporal_pr import temporal_pr
from flipflop_dimensionality import run_folders
from flipflop_diversity import rates_and_targets

PENS = ["none", "rws", "frm", "both"]
COL = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
LABEL = {"none": "no penalty", "rws": "sparsity only", "frm": "participation only", "both": "participation + sparsity"}
N_TRIALS = 64
BURST = 0.05
BINS = np.linspace(0, 0.75, 46)


def duty_cycle(targets):
    """Fraction of samples a bit spends in one of its two states (the task's own duty cycle).

    Args:
        targets: (k, T, B) target bits in {-1, 0, +1}.
    Returns:
        float, mean over bits of P(bit = +1).
    """
    return float((targets > 0).mean())


def per_network(folder):
    """tPR/n over live units, dead fraction, and the duty cycle for one network.

    Args:
        folder: run folder.
    Returns:
        dict with 'x' (live units' tPR/n), 'live' (count of live units), 'N', 'duty'.
    """
    rates, targets = rates_and_targets(folder, N_TRIALS)
    tpr, live, n = temporal_pr(rates)
    return dict(x=tpr[live] / n, live=int(live.sum()), N=int(live.size), duty=duty_cycle(targets))


def mode(x):
    """Histogram mode of tPR/n on the shared bins."""
    h, e = np.histogram(x, bins=BINS)
    return float(0.5 * (e[np.argmax(h)] + e[np.argmax(h) + 1]))


def main():
    """Draw the two panels and write fig_S3_transients.png."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ps.setup()
    res = {p: [] for p in PENS}
    for folder, pen, kk, NN in run_folders():
        if kk == k and NN == N and pen in res:
            res[pen].append(per_network(folder))
            print(f"{pen:5s} {os.path.basename(folder)[:30]}  live tPR/n median {np.median(res[pen][-1]['x']):.3f}  "
                  f"mode {mode(res[pen][-1]['x']):.3f}  live {res[pen][-1]['live']}/{res[pen][-1]['N']}", flush=True)
    duty = np.mean([r["duty"] for p in PENS for r in res[p]])

    fig, ax = plt.subplots(1, 3, figsize=(14, 4.4), gridspec_kw=dict(width_ratios=[2, 1, 1]))
    for p in PENS:
        x = np.concatenate([r["x"] for r in res[p]])
        live = int(np.mean([r["live"] for r in res[p]]))
        ax[0].hist(x, bins=BINS, density=True, histtype="step", lw=1.8, color=COL[p], label=f"{LABEL[p]}: {live} live units of {res[p][0]['N']}")
    ax[0].axvline(duty, color="0.3", ls=":", lw=1)
    ax[0].text(duty + 0.01, ax[0].get_ylim()[1] * 0.55, f"task duty cycle {duty:.2f}", fontsize=8, color="0.3", va="top")
    ax[0].set(xlabel="temporal PR / n_samples  (fraction of time a unit is effectively active)", ylabel="density over live units",
              title="(a) how long each live unit is active")
    ax[0].legend(loc="upper right")
    for p in PENS:
        print(f"{p:5s} mode of tPR/n per seed: " + ", ".join(f"{mode(r['x']):.3f}" for r in res[p]))
    xs = np.arange(len(PENS))
    med = [[float(np.median(r["x"])) for r in res[p]] for p in PENS]
    ax[1].bar(xs, [np.mean(m) for m in med], 0.65, yerr=[np.std(m) for m in med], color=[COL[p] for p in PENS], capsize=2)
    ax[1].axhline(duty, color="0.3", ls=":", lw=1)
    ax[1].text(len(PENS) - 0.6, duty + 0.01, "task duty cycle", ha="right", fontsize=7.5, color="0.3")
    ax[1].set(xticks=xs, xticklabels=[LABEL[p].replace(" ", "\n", 1) for p in PENS], ylim=(0, 0.5),
              ylabel="median tPR / n over live units", title="(b) typical unit: fraction of time active")
    burst = [[int((r["x"] < BURST).sum()) for r in res[p]] for p in PENS]
    ax[2].bar(xs, [np.mean(b) for b in burst], 0.65, yerr=[np.std(b) for b in burst], color=[COL[p] for p in PENS], capsize=2)
    for i, p in enumerate(PENS):
        ax[2].text(i, np.mean(burst[i]) + np.std(burst[i]) + 8, f"{np.mean(burst[i]):.0f}", ha="center", fontsize=8)
    ax[2].set(xticks=xs, xticklabels=[LABEL[p].replace(" ", "\n", 1) for p in PENS],
              ylabel=f"burst units (live, active < {BURST:.0%} of the time)", title=f"(c) number of burst units of {res['none'][0]['N']}")
    for a_ in ax[1:]:
        a_.tick_params(axis="x", labelsize=7.5)
    for p, b, m in zip(PENS, burst, med):
        print(f"{p:5s} median tPR/n={np.mean(m):.3f}±{np.std(m):.3f}  burst units={np.mean(b):.0f}±{np.std(b):.0f}")
    fig.suptitle("S3 — the participation penalty alone is satisfied by bursts; adding the sparsity penalty moves units to the task's duty cycle",
                 fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, "fig_S3_transients", tight=False)


if __name__ == "__main__":
    main()
