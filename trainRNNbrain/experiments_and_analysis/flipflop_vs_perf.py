#!/usr/bin/env python3
"""
Active units and silent fraction as CONTINUOUS functions of performance, not at chosen levels.

WHY THIS REPLACES PICKING AN L*. Comparing cells at one matched loss forces two awkward choices:
which level, and what to do when the floor differs with N, since a common absolute level then sits at
different depths relative to each network's own asymptote. The repair - match at a common fraction of
the way to the floor - needs L_inf, which is not always identifiable within the budget.

Plotting against performance itself dissolves the problem. Each cell is read at EVERY level it
attains, giving a curve; cells are then compared over the range where their curves overlap, which is
exactly the range where a comparison is licensed. No level is privileged and no floor is estimated.

AXIS. R^2 = 1 - MSE/0.735 and is comparable across k because the flip-flop's target variance is
k-independent (0.727-0.738 measured). Everything interesting happens between R^2 = 0.9 and 0.995, so
a linear R^2 axis wastes nine tenths of its width. The plot uses 1 - R^2 on a LOG axis, reversed so
that better performance runs left to right - the standard trick for a quantity approaching a ceiling.

Each cell is read using the same stable-crossing rule as everywhere else in this project, applied on
a dense grid of levels rather than three or four, so the curve is the protocol run continuously.

Output: img/internal_figures/flipflop_vs_perf.png

Usage:  python flipflop_vs_perf.py
NOTE ON PROVENANCE. Every number that this file previously quoted from the flip-flop came from the
first sweep, which ran `same_batch=True` and therefore trained on 256 frozen trials - memorisation,
not the task. Those numbers are RETRACTED and have been stripped rather than updated; the data is
quarantined in `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/`. Nothing here has yet
been run against the corrected fresh-batch sweep, so this file currently states METHOD only, with no
results. Do not reintroduce a remembered figure into these docstrings.

"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, active_count
from flipflop_ksweep import load, PROBE_EVERY, WINDOW, TGT_VAR

NLEV = 45          # levels per decade-and-a-half of loss; the curve is this finely sampled
MIN_SEEDS = 2      # a cell is plotted only where at least this many seeds reach the level


def trajectory(trace, N, levels):
    """Silent fractions and active counts at every level this run attains.

    Uses the project's stable-crossing rule (last iteration at which the smoothed loss is still
    above the level), with the smoothing done once and reused across all levels.

    Args:
        trace: a loaded trace; N: network size; levels: array of loss thresholds.
    Returns:
        (hard %, scalefree %, M_hard, M_scalefree), each len(levels), NaN where unreachable.
    """
    L = trace["clean"]
    h = WINDOW // 2
    s = np.convolve(L, np.ones(WINDOW) / WINDOW, mode="valid")
    idx = np.arange(h, len(L) - h)
    P = np.array(trace["participation"])
    I = np.array(trace["participation_iters"])

    out = np.full((4, len(levels)), np.nan)
    for j, lev in enumerate(levels):
        above = idx[s > lev]
        # Reject a level the run never gets below, and one it is still above at the end - the
        # crossing must lie INSIDE the run, or it is an extrapolation.
        if not len(above) or above[-1] >= idx[-1]:
            continue
        p = P[np.argmin(np.abs(I - above[-1] * PROBE_EVERY))]
        a_hard = active_count(p, "hard")
        a_sf = active_count(p, "scalefree")
        out[:, j] = (100 * (N - a_hard) / N, 100 * (N - a_sf) / N, a_hard, a_sf)
    return out


def main():
    """Plot active count and silent fraction against R^2, one panel column per N."""
    by, _ = load()
    ks = sorted({k for k, _ in by})
    Ns = sorted({N for _, N in by})
    lo = min(t["clean"][-1] for v in by.values() for t in v) * 0.95
    levels = np.logspace(np.log10(0.12), np.log10(lo), NLEV)
    x = levels / TGT_VAR                      # 1 - R^2

    fig, ax = plt.subplots(2, len(Ns), figsize=(6.3 * len(Ns), 10), squeeze=False)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))

    for j, N in enumerate(Ns):
        for k in ks:
            runs = by.get((k, N), [])
            if not runs:
                continue
            stack = np.array([trajectory(t, N, levels) for t in runs])   # (seeds, 4, levels)
            n_ok = np.sum(~np.isnan(stack[:, 0, :]), axis=0)
            mu = np.full(stack.shape[1:], np.nan)
            good = n_ok >= MIN_SEEDS          # avoids an all-NaN nanmean warning
            if good.any():
                mu[:, good] = np.nanmean(stack[:, :, good], axis=0)
            c = cols[ks.index(k)]
            ax[0][j].plot(x, mu[2], "-", color=c, lw=2.2, label=f"k={k}")
            ax[0][j].plot(x, mu[3], "--", color=c, lw=1.8, alpha=.85)
            ax[1][j].plot(x, mu[0], "-", color=c, lw=2.2, label=f"k={k}")
            ax[1][j].plot(x, mu[1], "--", color=c, lw=1.8, alpha=.85)

        ax[0][j].axhline(N, color="k", ls=":", lw=1.2, alpha=.7, label=f"$M=N$ ({N})")
        ax[0][j].set(xscale="log", yscale="log", xlabel="$1-R^2$  (better $\\rightarrow$)",
                     ylabel="active units $M$", title=f"N={N}  —  active units vs performance",
                     ylim=(N * 0.03, N * 1.5))
        ax[1][j].set(xscale="log", xlabel="$1-R^2$  (better $\\rightarrow$)",
                     ylabel="silent units (% of N)", ylim=(-3, 100),
                     title=f"N={N}  —  silent fraction vs performance")
        for a in (ax[0][j], ax[1][j]):
            a.invert_xaxis()
            a.grid(alpha=.3, which="both")
            sec = a.secondary_xaxis("top", functions=(lambda v: 1 - v, lambda v: 1 - v))
            # Explicit ticks: the auto-locator inherits the log scale and produces an unreadable
            # crowd of labels near R^2 = 1.
            sec.set_xticks([0.90, 0.95, 0.98, 0.99, 0.995])
            sec.set_xticklabels(["0.90", "0.95", "0.98", "0.99", "0.995"], fontsize=8)
            sec.set_xlabel("$R^2$")
        ax[0][j].plot([], [], "-", color="0.3", lw=2, label="solid: hard $p_i<10^{-6}$")
        ax[0][j].plot([], [], "--", color="0.3", lw=2, label="dashed: scale-free")
        ax[0][j].legend(fontsize=8, loc="lower left")
        ax[1][j].legend(fontsize=8, loc="upper left")

    fig.suptitle("Flip-flop: units recruited as a function of performance reached\n"
                 "every cell read at every level it attains, so no $L^*$ is privileged and no floor "
                 "is estimated; curves are comparable only where they overlap", fontsize=12)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "flipflop_vs_perf.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print("\nactive units (hard / scale-free) at selected performance levels")
    for target in (0.95, 0.98, 0.99, 0.993):
        j = int(np.argmin(np.abs((1 - x) - target)))
        print(f"\n  R^2 = {1 - x[j]:.3f}   (loss {levels[j]:.5f})")
        print("      " + "".join(f"{k:>13}" for k in ks) + "   (k)")
        for N in Ns:
            cells = ""
            for k in ks:
                runs = by.get((k, N), [])
                if not runs:
                    cells += "           --"
                    continue
                st = np.array([trajectory(t, N, levels[[j]]) for t in runs])
                if np.sum(~np.isnan(st[:, 0, 0])) < MIN_SEEDS:
                    cells += "           --"
                else:
                    cells += f"{np.nanmean(st[:, 2, 0]):7.0f}/{np.nanmean(st[:, 3, 0]):<5.0f}"
            print(f"  N={N:<4d}{cells}")


if __name__ == "__main__":
    main()
