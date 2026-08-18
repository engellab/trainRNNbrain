#!/usr/bin/env python3
"""
Fit L(t) = L_inf + A * t^(-gamma) to the training loss and show what it implies for the budget.

The raw training loss is a bad guide to "has this trained enough", because most of it is an
irreducible floor: the task is stochastic, so no amount of optimisation drives the loss to zero.
A 2%-per-doubling improvement sounds small only because it is divided by that floor. Splitting the
loss into L_inf (unreachable) and A*t^(-gamma) (still on the table) makes the honest statement
possible: what FRACTION OF THE REDUCIBLE LOSS is left at a given budget.

The fit is the load-bearing step, so its stability is checked rather than assumed: L_inf is
re-estimated from three different start times, and the spread between them is reported. An
extrapolated asymptote from a finite run is only as good as that spread.

Panels:
  (a) loss vs iteration with the fitted curve and the L_inf asymptote
  (b) L(t) - L_inf on log-log; a power law is a straight line here, so this is the goodness check
  (c) fraction of the reducible loss still remaining, and what it would be at longer budgets

Output: img/internal_figures/loss_fit_N<N>.png

Usage:  python plot_loss_fit.py [SWEEP_FOLDER]
        SWEEP_FOLDER defaults to data/trained_RNNs/CDDM_std_g0_drift
"""

import os
import re
import sys
import glob
import json
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
FIT_STARTS = [2000, 5000, 10000]     # fit-window start times, for the stability check
BUDGETS = [50000, 100000, 200000, 300000, 1000000]


def load_losses(sweep):
    """Load every training-loss curve under a sweep folder, grouped by network size.

    Args:
        sweep: path to the sweep folder.
    Returns:
        dict {N (int): list of (tag, loss array)} with loss indexed from iteration 1.
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(sweep, "*", "*", "*TrainLosses.json"))):
        m = re.search(r"_N=(\d+)_iters=(\d+)", f)
        if not m:
            continue
        L = np.array(json.load(open(f))["train_losses"], dtype=float)
        out.setdefault(int(m.group(1)), []).append((os.path.basename(f)[:9], L))
    return out


def logbin(t, y, nbins=60):
    """Median-reduce (t, y) into log-spaced bins, so the fit is not dominated by late iterations.

    Args:
        t, y: equal-length arrays; nbins: number of log-spaced bins.
    Returns:
        (tb, yb) bin-centre and bin-median arrays, empty bins dropped.
    """
    edges = np.logspace(np.log10(t[0]), np.log10(t[-1]), nbins + 1)
    idx = np.digitize(t, edges) - 1
    tb, yb = [], []
    for b in range(nbins):
        m = idx == b
        if m.sum() > 2:
            tb.append(np.median(t[m]))
            yb.append(np.median(y[m]))
    return np.array(tb), np.array(yb)


def fit_loss(L, t_start):
    """Fit L(t) = L_inf + A t^(-gamma) on iterations >= t_start, using log-binned medians.

    Args:
        L: loss array indexed from iteration 1; t_start: first iteration to include.
    Returns:
        (L_inf, A, gamma), or (nan, nan, nan) if the fit does not converge.
    """
    t = np.arange(1, len(L) + 1, dtype=float)
    m = t >= t_start
    tb, yb = logbin(t[m], L[m])
    try:
        (Li, A, g), _ = curve_fit(lambda x, Li, A, g: Li + A * x ** (-g), tb, yb,
                                  p0=[yb[-1] * 0.9, 1.0, 0.5], maxfev=40000)
        return float(Li), float(A), float(g)
    except Exception:
        return float("nan"), float("nan"), float("nan")


def plot_for_N(entries, N, out):
    """Three-panel loss-fit summary for one network size, one colour per seed.

    Args:
        entries: list of (tag, loss array); N: network size; out: output png path.
    """
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    cols = plt.cm.viridis(np.linspace(0.12, 0.8, len(entries)))
    rows = []
    for k, (tag, L) in enumerate(entries):
        t = np.arange(1, len(L) + 1, dtype=float)
        Li, A, g = fit_loss(L, FIT_STARTS[0])
        stab = [fit_loss(L, s)[0] for s in FIT_STARTS]
        tb, yb = logbin(t[t >= FIT_STARTS[0]], L[t >= FIT_STARTS[0]])

        ax[0].plot(t[::50], L[::50], "-", color=cols[k], lw=.7, alpha=.35)
        ax[0].plot(tb, yb, "o", color=cols[k], ms=3.5)
        ax[0].plot(tb, Li + A * tb ** (-g), "-", color=cols[k], lw=2,
                   label=f"seed {k+1}: $L_\\infty$={Li:.5f}, $\\gamma$={g:.2f}")
        ax[0].axhline(Li, color=cols[k], ls=":", lw=1)

        ax[1].plot(tb, np.maximum(yb - Li, 1e-9), "o", color=cols[k], ms=3.5)
        ax[1].plot(tb, A * tb ** (-g), "-", color=cols[k], lw=2)

        LT = L[-1000:].mean()
        red = A * t[-1] ** (-g)                      # reducible loss still left at end of run
        ax[2].plot(BUDGETS, [100 * A * b ** (-g) / (Li + A * b ** (-g)) for b in BUDGETS],
                   "-o", color=cols[k], ms=5,
                   label=f"seed {k+1}")
        rows.append(dict(tag=tag, Li=Li, A=A, g=g, LT=LT, red=red,
                         frac=100 * red / LT, stab=stab,
                         spread=100 * (max(stab) - min(stab)) / np.mean(stab)))

    ax[0].set(xscale="log", xlabel="iteration", ylabel="training loss",
              ylim=(0, float(np.median([r["LT"] for r in rows])) * 3))
    ax[0].set_title(f"(a) loss and fit $L_\\infty + A\\,t^{{-\\gamma}}$ (N={N})\n"
                    "dotted = fitted asymptote")
    ax[1].set(xscale="log", yscale="log", xlabel="iteration",
              ylabel=r"$L(t)-L_\infty$")
    ax[1].set_title("(b) excess over the asymptote\nstraight line here = the power law holds")
    ax[2].axvline(50000, color="k", ls="--", lw=1)
    ax[2].set(xscale="log", xlabel="training budget (iterations)",
              ylabel="reducible loss remaining (% of total loss)")
    ax[2].set_title("(c) what a longer budget would buy\ndashed = the run we have")
    for a in ax:
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=8)
        a.grid(alpha=.25)
    fig.suptitle(f"Training-loss decomposition, N={N}: most of the loss is an irreducible floor",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")

    print(f"\n  N={N}   L_inf = asymptote, gamma = decay exponent, "
          f"'reducible' = A*T^-gamma at end of run")
    print(f"  {'seed':9s} {'L_inf':>9s} {'gamma':>6s} {'L(T)':>9s} {'irreducible':>12s} "
          f"{'reducible':>10s} {'L_inf spread':>13s}")
    for r in rows:
        print(f"  {r['tag']:9s} {r['Li']:9.5f} {r['g']:6.2f} {r['LT']:9.5f} "
              f"{100*r['Li']/r['LT']:11.1f}% {r['frac']:9.1f}% {r['spread']:12.1f}%")
    return rows


def matched_performance_budget(by_n, targets):
    """Iteration at which each network size first reaches a given training loss.

    This is the answer to "how do I train different sizes comparably". It sidesteps convergence
    entirely: rather than asking whether any size has finished, it asks when each size is EQUALLY
    GOOD AT THE TASK, and compares them there. It needs no fit, no L_inf, no asymptote, and no
    threshold that gets inverted through a power law. It also absorbs the learning-rate difference
    across sizes automatically — a network trained at a lower lr simply needs more iterations to
    reach the same loss, which is exactly what a fair matching should charge it.

    Args:
        by_n: {N: [(tag, loss array)]}; targets: iterable of loss levels to match at.
    Returns:
        {(N, target): (mean_iter, sd_iter)}, missing where a size never reaches that loss.
    """
    out = {}
    for N, entries in by_n.items():
        for tg in targets:
            its = []
            for _, L in entries:
                ts = np.arange(2000, len(L), 500)
                sm = np.array([L[max(t - max(int(0.02 * t), 50), 0):t].mean() for t in ts])
                hit = ts[sm <= tg]
                its.append(hit[0] if len(hit) else np.nan)
            if not np.isnan(its).any():
                out[(N, tg)] = (float(np.mean(its)), float(np.std(its)))
    return out


def compare_sizes(by_n, out):
    """Overlay the loss curves and fits of every network size, colour-coded by N.

    The comparison that matters is whether L_inf — the loss the task allows at all — improves with
    N. If it does not, the extra units are not buying task performance, which bears directly on the
    M* question.

    Args:
        by_n: {N: [(tag, loss array)]}; out: output png path.
    """
    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))
    Ns = sorted(by_n)
    cols = plt.cm.plasma(np.linspace(0.1, 0.7, len(Ns)))
    for k, N in enumerate(Ns):
        Lis, gs = [], []
        for j, (tag, L) in enumerate(by_n[N]):
            t = np.arange(1, len(L) + 1, dtype=float)
            Li, A, g = fit_loss(L, FIT_STARTS[0])
            Lis.append(Li)
            gs.append(g)
            tb, yb = logbin(t[t >= FIT_STARTS[0]], L[t >= FIT_STARTS[0]])
            ax[0].plot(tb, yb, "-", color=cols[k], lw=1.4, alpha=.85,
                       label=f"N={N}" if j == 0 else None)
            ax[1].plot(tb, np.maximum(yb - Li, 1e-9), "-", color=cols[k], lw=1.4, alpha=.85,
                       label=f"N={N}" if j == 0 else None)
            ax[2].plot(BUDGETS, [100 * A * b ** (-g) / (Li + A * b ** (-g)) for b in BUDGETS],
                       "-o", color=cols[k], ms=4, alpha=.85,
                       label=f"N={N}" if j == 0 else None)
        ax[0].axhline(np.mean(Lis), color=cols[k], ls=":", lw=1.2)
        print(f"  N={N:5d}  L_inf = {np.mean(Lis):.5f} +- {np.std(Lis):.5f}   "
              f"gamma = {np.mean(gs):.2f} +- {np.std(gs):.2f}   "
              f"(seed spread in L_inf: {100*(max(Lis)-min(Lis))/np.mean(Lis):.1f}%)")
    ax[0].set(xscale="log", xlabel="iteration", ylabel="training loss (log-binned median)",
              ylim=(0.019, 0.032))
    ax[0].set_title("(a) loss curves, all sizes\ndotted = fitted asymptote $L_\\infty$")
    ax[1].set(xscale="log", yscale="log", xlabel="iteration", ylabel=r"$L(t)-L_\infty$")
    ax[1].set_title("(b) excess over each size's own asymptote")
    ax[2].set(xscale="log", xlabel="training budget (iterations)",
              ylabel="reducible loss remaining (% of total)")
    ax[2].set_title("(c) what a longer budget would buy")
    for a in ax:
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=9)
        a.grid(alpha=.25)
    fig.suptitle("Training loss across network sizes: does a bigger network reach a lower floor?",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    """Fit and plot the loss decomposition for every network size in the sweep."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sweep = args[0] if args else "data/trained_RNNs/CDDM_std_g0_drift"
    by_n = load_losses(sweep)
    if not by_n:
        sys.exit(f"no TrainLosses under {sweep}")
    for N in sorted(by_n):
        plot_for_N(by_n[N], N, os.path.join(IMG_DIR, f"loss_fit_N{N}.png"))
    if len(by_n) > 1:
        print("\n  Across sizes:")
        compare_sizes(by_n, os.path.join(IMG_DIR, "loss_fit_compare.png"))


if __name__ == "__main__":
    main()
