#!/usr/bin/env python3
"""
Does silencing continue, or does it stop? Silent-unit count against training iteration.

The count is recorded every `track_every` iterations during training from a NOISE-FREE probe, so this
is a direct measurement over the whole run - no fit, no extrapolation, no endpoint-only inference.

Two criteria are shown because they answer different questions and can disagree by an order of
magnitude:
    hard        p_i < 1e-6              a unit that has genuinely stopped firing
    scale-free  p_i < 0.05 * q95(p)     a unit that is negligible RELATIVE to its network's own
                                        scale, which also moves when the rate distribution stretches

The right-hand panel is the one that answers "does it continue": the change over the last DOUBLING of
the budget, which is flat if silencing has stopped and non-zero if it has not. Plotting the raw count
against a log axis makes late changes look small even when they are large, which is why the rate is
shown separately.

Output: img/internal_figures/silencing_trajectory_N<N>.png

Usage:  python plot_silencing_trajectory.py [N] [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, load_traces, series


def silent_series(trace, criterion, N):
    """Silent-unit COUNT over training.

    Args:
        trace: trace dict; criterion: "hard" or "scalefree"; N: network size.
    Returns:
        (iters, counts). The hard criterion uses the every-probe online counter; the scale-free one
        is recomputed from the stored per-unit vectors, which are saved on a coarser cadence.
    """
    if criterion == "hard":
        it = np.asarray(trace["iters"], dtype=float)
        return it, np.asarray(trace["metrics"]["silent_1em6"], dtype=float)
    P = np.array(trace["participation"])
    I = np.array(trace["participation_iters"], dtype=float)
    return I, np.array([(p < 0.05 * np.quantile(p, 0.95)).sum() for p in P], dtype=float)


def main():
    """Plot the silencing trajectory and its per-doubling rate for one network size."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    N = int(args[0]) if args else 2000
    sweep = args[1] if len(args) > 1 else "data/trained_RNNs/CDDM_std_g0_drift"
    by = load_traces(sweep)
    if N not in by:
        sys.exit(f"no traces for N={N}; have {sorted(by)}")
    traces = by[N]
    print(f"N={N}: {len(traces)} seed(s), {len(traces[0]['iters'])} probes, "
          f"up to iteration {traces[0]['iters'][-1]}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5.2))
    cols = {"hard": "C0", "scalefree": "C3"}
    lab = {"hard": r"truly silent  $p_i<10^{-6}$",
           "scalefree": r"scale-free  $p_i<0.05\,q_{95}(p)$"}

    for crit in ("hard", "scalefree"):
        curves = []
        for j, t in enumerate(traces):
            it, c = silent_series(t, crit, N)
            ax[0].plot(it, 100 * c / N, "-", color=cols[crit], lw=1.3, alpha=.75,
                       label=lab[crit] if j == 0 else None)
            curves.append((it, c))
        # change over the last doubling, evaluated on a log grid
        end = min(x[0][-1] for x in curves)
        ts = np.unique(np.round(np.logspace(np.log10(2000), np.log10(end), 40)).astype(int))
        rates = []
        for it, c in curves:
            r = []
            for t_ in ts:
                a = np.interp(t_ / 2, it, c)
                b = np.interp(t_, it, c)
                r.append(100 * (b - a) / N)
            rates.append(r)
        mu, sd = np.mean(rates, axis=0), np.std(rates, axis=0)
        ax[1].plot(ts, mu, "-o", color=cols[crit], ms=3.5, lw=1.6, label=lab[crit])
        ax[1].fill_between(ts, mu - sd, mu + sd, color=cols[crit], alpha=.2)
        print(f"  {crit:10s} final {100*np.mean([c[-1] for _, c in curves])/N:.1f}% silent; "
              f"change over last doubling {mu[-1]:+.2f} pp")

    ax[1].axhline(0, color="k", lw=1.2)
    ax[1].axhline(1, color="grey", ls="--", lw=1, label="1 pp precision target")
    ax[0].set(xscale="log", xlabel="iteration", ylabel="silent units (% of N)",
              title=f"(a) silencing trajectory, N={N}")
    ax[1].set(xscale="log", xlabel="iteration $t$",
              ylabel="change over $t/2 \\to t$ (pp of N)",
              title="(b) is it still going?\nflat at 0 = silencing has stopped")
    for a in ax:
        a.legend(fontsize=8)
        a.grid(alpha=.3)
    fig.suptitle(f"Does silencing continue? N={N}, "
                 f"{len(traces)} seed(s), measured every 10 iterations noise-free", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    out = os.path.join(IMG_DIR, f"silencing_trajectory_N{N}.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
