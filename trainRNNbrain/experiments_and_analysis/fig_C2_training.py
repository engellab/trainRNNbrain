#!/usr/bin/env python3
"""
Figure for elaboration claim C2: silencing worsens with training time (the size half is fig_P1).

Two panels, one per task: active units (scale-free criterion) vs training iteration, unpenalised ReLU,
N = 1000, every seed, read from ParticipationTrace.pkl. No network is simulated.

  CDDM       CDDM_std_g0_drift, N = 1000 (the longest CDDM budget on disk)
  flip-flop  NBitFlipFlop_std_ksweep, k = 3, N = 1000, 500k iterations

Usage:  python fig_C2_training.py
Output: img/internal_figures/fig_C2_training.png
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_REL
import plotstyle as ps

N = 1000
ROOTS = {"CDDM": f"data/trained_RNNs/CDDM_std_g0_drift/EqType=h_N={N}_iters=*",
         "flip-flop": f"data/trained_RNNs/NBitFlipFlop_std_ksweep/EqType=h_k=3_N={N}_iters=*"}
COL = {"CDDM": "#1f77b4", "flip-flop": "#d62728"}


def traces(pattern):
    """Active-unit count (scale-free criterion) along training for every seed under a cell pattern.

    Args:
        pattern: glob for the cell folder(s); traces are read from every run inside.
    Returns:
        list of (iters (n,), active_count (n,)) per seed.
    """
    out = []
    for f in sorted(glob.glob(os.path.join(pattern, "*", "*ParticipationTrace.pkl"))):
        d = pickle.load(open(f, "rb"))
        P = np.asarray(d["participation"], dtype=float)
        q = np.quantile(P, 0.95, axis=1, keepdims=True)
        out.append((np.asarray(d["participation_iters"]), (P >= SILENT_REL * q).sum(1)))
    return out


def main():
    """Draw the two panels and write fig_C2_training.png."""
    ps.setup()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for a, (task, pat) in zip(ax, ROOTS.items()):
        for i, (it, sf) in enumerate(traces(pat)):
            a.plot(it[1:], sf[1:], color=COL[task], alpha=.75, lw=1.3)
            print(f"{task} seed{i}: active {sf[np.searchsorted(it, 10_000) - 1]:.0f} at 10k, {sf[-1]:.0f} at {it[-1]}")
        a.axhline(N, color="0.5", lw=0.8, ls=":")
        a.set(xscale="log", xlabel="training iteration", ylim=(0, N * 1.1), title=f"{task}, ReLU, N={N}, no penalty")
    ax[0].set_ylabel("active units (scale-free)")
    fig.suptitle("C2 — the number of active units keeps falling with training time", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "fig_C2_training", tight=False)


if __name__ == "__main__":
    main()
