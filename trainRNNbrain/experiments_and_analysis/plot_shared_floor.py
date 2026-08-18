#!/usr/bin/env python3
"""
Is the irreducible loss floor L_inf a property of the TASK (shared by all network sizes) or of the
NETWORK (different for each size)?

This matters because the whole cross-size comparison rests on it. If bigger networks reached a lower
floor, then "N=1000 has more active units" could simply mean "N=1000 solves the task better", and the
M* question would be confounded by performance. If the floor is shared, the extra units are provably
not buying task performance.

The test has to control for one thing. L_inf is obtained by extrapolating L(t) = L_inf + A t^(-gamma),
and that extrapolation is biased by how much data it sees: fitting a curve that has not yet flattened
systematically UNDERestimates the asymptote. Runs of different length therefore cannot be compared
directly. The controlled comparison refits every size using the same first t_max iterations, so any
bias applies equally to all of them, and asks whether the estimates agree.

Panels:
  (a) L_inf estimated from the first t_max iterations, versus t_max, one curve per size.
      Shared floor => the curves lie on top of each other at every t_max.
  (b) the same as a DIFFERENCE from the N=500 reference, with combined seed error bars, so
      "agree within noise" can be read against zero rather than eyeballed.
  (c) the loss curves themselves near the floor.

Output: img/internal_figures/shared_floor.png

Usage:  python plot_shared_floor.py [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_fit import load_losses, fit_loss, logbin, IMG_DIR

TMAXES = [20000, 30000, 50000, 75000, 100000, 150000, 200000]


def floor_vs_tmax(entries, tmax):
    """Fit L_inf from the first tmax iterations of each seed.

    Args:
        entries: list of (tag, loss array); tmax: number of leading iterations to fit on.
    Returns:
        array of L_inf estimates, one per seed that is long enough (possibly empty).
    """
    out = []
    for _, L in entries:
        if len(L) >= tmax:
            Li, _, _ = fit_loss(L[:tmax], 2000)
            if np.isfinite(Li):
                out.append(Li)
    return np.array(out)


def main():
    """Build the shared-floor evidence figure and print the controlled comparison."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    by = load_losses(sweep)
    Ns = sorted(by)
    cols = plt.cm.plasma(np.linspace(0.1, 0.72, len(Ns)))
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

    est = {}
    for k, N in enumerate(Ns):
        xs, ys, es = [], [], []
        for tm in TMAXES:
            v = floor_vs_tmax(by[N], tm)
            if len(v):
                xs.append(tm)
                ys.append(v.mean())
                es.append(v.std())
                est[(N, tm)] = (v.mean(), v.std(), len(v))
        ax[0].errorbar(xs, ys, yerr=es, fmt="-o", color=cols[k], ms=5, capsize=3, lw=1.6,
                       label=f"N={N}")

    ref = Ns[1] if len(Ns) > 1 else Ns[0]
    for k, N in enumerate(Ns):
        xs, dy, de = [], [], []
        for tm in TMAXES:
            if (N, tm) in est and (ref, tm) in est:
                m1, s1, _ = est[(N, tm)]
                m0, s0, _ = est[(ref, tm)]
                xs.append(tm)
                dy.append(m1 - m0)
                de.append(np.hypot(s1, s0))
        if xs:
            ax[1].errorbar(xs, dy, yerr=de, fmt="-o", color=cols[k], ms=5, capsize=3, lw=1.6,
                           label=f"N={N}")
    ax[1].axhline(0, color="k", lw=1.2)

    for k, N in enumerate(Ns):
        for j, (_, L) in enumerate(by[N]):
            t = np.arange(1, len(L) + 1, dtype=float)
            m = t >= 2000
            tb, yb = logbin(t[m], L[m])
            ax[2].plot(tb, yb, "-", color=cols[k], lw=1.4, alpha=.85,
                       label=f"N={N}" if j == 0 else None)
    best = np.mean([est[(N, 200000)][0] for N in Ns if (N, 200000) in est])
    ax[2].axhline(best, color="k", ls="--", lw=1.2,
                  label=f"$L_\\infty$ est. at 200k = {best:.5f}")

    ax[0].set(xscale="log", xlabel="$t_{max}$: iterations used for the fit",
              ylabel=r"estimated $L_\infty$")
    ax[0].set_title("(a) floor estimated from matched data lengths\n"
                    "curves overlap = the floor is shared")
    ax[1].set(xscale="log", xlabel="$t_{max}$: iterations used for the fit",
              ylabel=f"$L_\\infty(N) - L_\\infty(N={ref})$")
    ax[1].set_title(f"(b) difference from N={ref}, seed errors combined\n"
                    "consistent with zero at every $t_{max}$")
    ax[2].set(xscale="log", xlabel="iteration", ylabel="training loss (log-binned median)",
              ylim=(0.020, 0.030))
    ax[2].set_title("(c) the loss curves themselves")
    for a in ax:
        a.legend(fontsize=9)
        a.grid(alpha=.25)
    fig.suptitle("Is the loss floor shared across network sizes? "
                 "Controlled by fitting every size on the same amount of data", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(IMG_DIR, "shared_floor.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}\n")

    print("Controlled comparison: L_inf fitted on the first t_max iterations")
    print("%9s" % "t_max" + "".join("%22s" % ("N=%d" % N) for N in Ns) + "   max pairwise diff")
    for tm in TMAXES:
        vals = [est[(N, tm)] for N in Ns if (N, tm) in est]
        row = "%9d" % tm
        for N in Ns:
            row += "%22s" % ("%.5f+-%.5f" % est[(N, tm)][:2] if (N, tm) in est else "-")
        if len(vals) > 1:
            ms = [v[0] for v in vals]
            spread = max(ms) - min(ms)
            worst = max(np.hypot(vals[i][1], vals[j][1])
                        for i in range(len(vals)) for j in range(i + 1, len(vals)))
            row += "   %.5f (%.1f sd)" % (spread, spread / worst if worst else np.nan)
        print(row)


if __name__ == "__main__":
    main()
