#!/usr/bin/env python3
"""
Plot the LOCAL power-law exponent of the training loss as a function of training time.

A single fitted gamma answers "what exponent best describes the whole run". It cannot say whether
one exponent describes the run at all. Fitting three different ranges and getting 0.47, 0.56, 0.61
hints that it does not, but that comparison is coarse and could be produced by an edge effect at the
end of each range. The direct measurement is the local slope:

    D(t) = L(t/2) - L(t)  proportional to  t^(-gamma)          (no L_inf needed, see plot_powerlaw_coords)
    gamma_eff(t) = -d log D / d log t                          measured in a sliding window

If the loss is a true power law, gamma_eff(t) is FLAT. If it rises, the decay is steepening, which
would explain both why the fitted L_inf keeps climbing with fit range and why the three-parameter
fit's gamma sits systematically below the L_inf-free estimate.

The distinction this figure is built to make: a genuine steepening shows gamma_eff rising smoothly
through the MIDDLE of the range, whereas an edge artefact shows it flat until the last few points.

Panels:
  (a) D(t) with the single global fit overlaid — curvature away from the line is the effect
  (b) gamma_eff(t) per seed
  (c) gamma_eff(t) averaged over seeds, one band per size, to check the sizes agree

Output: img/internal_figures/local_exponent.png

Usage:  python plot_local_exponent.py [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_fit import load_losses, IMG_DIR
from plot_powerlaw_coords import doubling_difference

T_MIN = 4000          # first t at which L(t/2) is past the initial transient
WIN = 2.0             # sliding window is [t/WIN, t*WIN], i.e. a factor WIN^2 in t


def doubling_curve(L, npts=70):
    """D(t) = L(t/2) - L(t) on a dense log grid.

    Args:
        L: loss array indexed from iteration 1; npts: number of log-spaced evaluation points.
    Returns:
        (t, D) arrays with D > 0.
    """
    ts = np.unique(np.round(np.logspace(np.log10(T_MIN), np.log10(len(L)), npts)).astype(int))
    return doubling_difference(L, ts)


def local_exponent(t, D, win=WIN):
    """Sliding-window log-log slope of D, returned as gamma_eff = -slope.

    Args:
        t, D: arrays from doubling_curve; win: window half-factor, so the window spans [t/win, t*win].
    Returns:
        (t_c, gamma_eff) at every centre with at least 5 points in its window.
    """
    tc, ge = [], []
    for c in t:
        m = (t >= c / win) & (t <= c * win)
        if m.sum() >= 5:
            tc.append(c)
            ge.append(-np.polyfit(np.log(t[m]), np.log(D[m]), 1)[0])
    return np.array(tc), np.array(ge)


def main():
    """Draw the local-exponent figure and report gamma_eff at a few training ages."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    by = load_losses(sweep)
    Ns = sorted(by)
    cols = plt.cm.plasma(np.linspace(0.1, 0.72, len(Ns)))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    per_N = {}

    for k, N in enumerate(Ns):
        curves = []
        for j, (tag, L) in enumerate(by[N]):
            t, D = doubling_curve(L)
            lbl = f"N={N}" if j == 0 else None
            ax[0].plot(t, D, "o", color=cols[k], ms=3, alpha=.55, label=lbl)
            s, c = np.polyfit(np.log(t), np.log(D), 1)
            ax[0].plot(t, np.exp(c) * t ** s, "-", color=cols[k], lw=1.5, alpha=.9)
            tc, ge = local_exponent(t, D)
            ax[1].plot(tc, ge, "-", color=cols[k], lw=1.4, alpha=.8, label=lbl)
            curves.append((tc, ge))
        grid = np.unique(np.concatenate([c[0] for c in curves]))
        stack = np.array([np.interp(grid, c[0], c[1], left=np.nan, right=np.nan) for c in curves])
        mu, sd = np.nanmean(stack, axis=0), np.nanstd(stack, axis=0)
        ax[2].plot(grid, mu, "-", color=cols[k], lw=2, label=f"N={N}")
        ax[2].fill_between(grid, mu - sd, mu + sd, color=cols[k], alpha=.2)
        per_N[N] = (grid, mu, sd)

    ax[0].set(xscale="log", yscale="log", xlabel="iteration $t$",
              ylabel=r"$D(t)=L(t/2)-L(t)$")
    ax[0].set_title("(a) doubling difference with a single global fit\n"
                    "systematic curvature away from the line")
    for a in (ax[1], ax[2]):
        a.set(xscale="log", xlabel="iteration $t$",
              ylabel=r"$\gamma_{\mathrm{eff}}(t)=-\,\mathrm{d}\log D/\mathrm{d}\log t$")
        a.grid(alpha=.25)
    ax[1].set_title("(b) local exponent, per seed\nflat = true power law")
    ax[2].set_title("(c) local exponent, mean $\\pm$ sd per size\ndo the sizes agree?")
    for a in ax:
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=9)
        a.grid(alpha=.25)
    fig.suptitle(r"Is one exponent enough? Local slope of the $L_\infty$-free doubling difference, "
                 "measured in a sliding factor-4 window", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    out = os.path.join(IMG_DIR, "local_exponent.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    print("gamma_eff(t), mean +- sd over seeds")
    ages = [8000, 15000, 30000, 60000, 120000, 180000]
    print("%6s" % "N" + "".join("%16s" % f"t={a}" for a in ages))
    for N in Ns:
        grid, mu, sd = per_N[N]
        row = "%6d" % N
        for a in ages:
            if grid[0] <= a <= grid[-1]:
                i = int(np.argmin(abs(grid - a)))
                row += "%16s" % ("%.3f+-%.3f" % (mu[i], sd[i]))
            else:
                row += "%16s" % "-"
        print(row)


if __name__ == "__main__":
    main()
