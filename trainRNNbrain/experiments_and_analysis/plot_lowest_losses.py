#!/usr/bin/env python3
"""
Model-free floor estimate: the median of the K lowest loss values each run ever reached, versus N.

No functional form, no extrapolation, no asymptote — just what the optimiser actually achieved. This
sidesteps the misspecification that makes the fitted L_inf unreliable (its estimate drifts
systematically with the fit window).

TWO PROPERTIES OF THIS STATISTIC THAT DECIDE HOW IT MAY BE READ:

  It is a LOWER-TAIL statistic, not a mean. The values land near 0.0155 while the smoothed loss is
  about 0.0221, because the per-iteration loss is a single noisy batch evaluation and its minimum
  samples the bottom of that fluctuation. It therefore tracks the floor plus the noise width, not the
  achievable mean loss.

  It is an EXTREME-VALUE statistic, so it depends on how many draws a run had. A 300k-iteration run
  gets 300k chances at a low value, a 26k run gets 26k. Runs of different length are therefore NOT
  directly comparable, which is why the right-hand panel restricts to the sizes that all ran to at
  least MIN_ITERS.

Output: img/internal_figures/lowest_losses.png

Usage:  python plot_lowest_losses.py [SWEEP_FOLDER] [--logs=DIR]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_fit import load_losses, IMG_DIR
from test_floor_vs_N import losses_from_logs

K = 101
MIN_ITERS = 200000


def mad(x):
    """Median absolute deviation, unscaled (multiply by 1.4826 for a sigma equivalent)."""
    x = np.asarray(x, dtype=float)
    return float(np.median(np.abs(x - np.median(x))))


def lowest_stats(L, k=K):
    """Median and MAD of the k lowest values of a loss trace.

    Args:
        L: loss array; k: how many of the lowest values to summarise.
    Returns:
        (median, mad) of those k values.
    """
    s = np.sort(np.asarray(L, dtype=float))[:k]
    return float(np.median(s)), mad(s)


def main():
    """Plot the model-free floor estimate against network size."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sweep = args[0] if args else "data/trained_RNNs/CDDM_std_g0_drift"
    logdir = ([a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--logs=")] or [None])[0]
    by = load_losses(sweep)
    if logdir:
        for N, v in losses_from_logs(logdir).items():
            by.setdefault(N, []).extend(v)

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    for panel, (only_long, ttl) in enumerate([
            (False, "(a) all runs\nopen = shorter run, so fewer draws AND less trained"),
            (True, f"(b) only runs reaching {MIN_ITERS//1000}k iterations\nthe like-for-like comparison")]):
        med_by_N = {}
        for N in sorted(by):
            for tag, L in by[N]:
                long = len(L) >= MIN_ITERS
                if only_long and not long:
                    continue
                m, d = lowest_stats(L)
                med_by_N.setdefault(N, []).append(m)
                ax[panel].errorbar([N], [m], yerr=[d], fmt="o" if long else "o",
                                   mfc="C0" if long else "none", mec="C0", ecolor="C0",
                                   ms=7, capsize=3, lw=1.2, alpha=.9)
        Ns = sorted(med_by_N)
        ax[panel].plot(Ns, [np.median(med_by_N[N]) for N in Ns], "-", color="C3", lw=1.8,
                       label="median across seeds")
        if only_long and Ns:
            allv = [v for N in Ns for v in med_by_N[N]]
            ax[panel].axhline(np.median(allv), color="grey", ls="--", lw=1,
                              label=f"overall median = {np.median(allv):.5f}")
            ax[panel].set_ylim(min(allv) - 0.0004, max(allv) + 0.0004)
        ax[panel].set(xscale="log", xlabel="$N$",
                      ylabel=f"median of the {K} lowest losses")
        ax[panel].set_title(ttl, fontsize=10)
        ax[panel].legend(fontsize=8)
        ax[panel].grid(alpha=.3)

    fig.suptitle("Model-free floor estimate: lowest loss actually achieved, vs network size\n"
                 "(error bars = MAD within each run; no fit, no extrapolation)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(IMG_DIR, "lowest_losses.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
