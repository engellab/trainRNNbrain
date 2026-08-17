#!/usr/bin/env python3
"""
Is the network still marching somewhere, or just jittering in place?

Training never "converges" in the strict sense here: the participation vector keeps changing as a
power law in iteration, so a 1% criterion extrapolates to millions of steps. The answerable question
is whether the motion is still SYSTEMATIC (the weights are being carried in a consistent direction
by the task gradient) or has become DIFFUSIVE (noise-driven random walk about a solution).

Two independent readouts distinguish these:

  scaling of drift with lag   d(L) = ||W(t) - W(t-L)|| / ||W(t)|| measured at several lags L.
                              A random walk gives d ~ L^0.5; systematic motion in a fixed direction
                              gives d ~ L^1. The local exponent alpha = dlog d / dlog L is therefore
                              a direct diffusive-vs-ballistic test, and it needs no threshold.

  directional persistence     cos between consecutive weight displacements. Exactly 0 for an
                              uncorrelated walk, positive while the trajectory keeps its heading.
                              Note the displacements are separated by `track_every` iterations, and
                              Adam's momentum correlates successive steps, so the FLOOR of this
                              quantity is positive rather than zero — read its decay, not its value.

Output: img/internal_figures/drift_N<N>.png, one per network size in the sweep.

Usage:  python plot_drift_curves.py [SWEEP_FOLDER]
        SWEEP_FOLDER defaults to data/trained_RNNs/CDDM_std_g0_drift
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

LAGS = [100, 1000, 10000]
MATS = ["W_inp", "W_rec", "W_out"]
HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")


def load_traces(sweep):
    """Load every ParticipationTrace under a sweep folder, grouped by network size.

    Args:
        sweep: path to the sweep folder holding `EqType=..._N=<N>_iters=<I>/<net>/` subfolders.
    Returns:
        dict {N (int): list of trace dicts}, each trace as written by Trainer (keys `iters`,
        `participation`, `participation_iters`, `metrics`).
    """
    out = {}
    for f in sorted(glob.glob(os.path.join(sweep, "*", "*", "*ParticipationTrace.pkl"))):
        m = re.search(r"_N=(\d+)_iters=(\d+)", f)
        if not m:
            continue
        with open(f, "rb") as fh:
            out.setdefault(int(m.group(1)), []).append(pickle.load(fh))
    return out


def series(trace, key):
    """Extract one metric as (iterations, values) with the NaN padding removed.

    Metrics that are only defined once per lag (the drift distances) are stored NaN-padded on the
    full probe grid; the cosines are defined at every probe. Both are handled by dropping NaNs.

    Args:
        trace: a trace dict from load_traces.
        key: metric name, e.g. "drift_W_rec_lag1000".
    Returns:
        (iters, vals) float arrays of equal length, possibly empty.
    """
    it = np.asarray(trace["iters"], dtype=float)
    v = np.asarray(trace["metrics"][key], dtype=float)
    ok = np.isfinite(v)
    return it[ok], v[ok]


def running_median(y, w):
    """Median filter of window w (odd), edges shrunk rather than padded. Returns array like y."""
    n = len(y)
    h = w // 2
    return np.array([np.median(y[max(0, i - h):min(n, i + h + 1)]) for i in range(n)])


def lag_exponent(trace, mat, l1, l2):
    """Local scaling exponent of drift with lag: alpha = log(d(l2)/d(l1)) / log(l2/l1).

    The two lags are measured at different iterations, so the shorter-lag series is interpolated in
    log-iteration onto the longer-lag grid before the ratio is taken.

    Args:
        trace: trace dict; mat: one of "W_inp"/"W_rec"/"W_out"; l1, l2: lags with l1 < l2.
    Returns:
        (iters, alpha) arrays. alpha ~ 0.5 diffusive, ~ 1 systematic, < 0.5 mean-reverting.
    """
    i1, d1 = series(trace, f"drift_{mat}_lag{l1}")
    i2, d2 = series(trace, f"drift_{mat}_lag{l2}")
    if len(i1) < 2 or len(i2) < 1:
        return np.array([]), np.array([])
    d1i = np.exp(np.interp(np.log(i2), np.log(i1), np.log(d1)))
    return i2, np.log(d2 / d1i) / np.log(l2 / l1)


def overlay(axis, traces, keys, labels, palette, getter, **kw):
    """Draw one coloured curve per compared quantity, repeated once per seed.

    Colour is the ONLY semantic channel: each entry of `keys` gets its own colour and one legend
    entry. Seeds appear as several same-coloured lines, so spread between them reads as seed
    variability rather than as another variable.

    Args:
        axis: matplotlib axis; traces: list of trace dicts (one per seed);
        keys: values passed to getter, one per coloured curve;
        labels: legend label per key; palette: list of colours, same length as keys;
        getter: callable (trace, key) -> (x, y) arrays;
        **kw: forwarded to plot (lw, marker, ...).
    """
    for j, key in enumerate(keys):
        for k, t in enumerate(traces):
            x, y = getter(t, key)
            if len(x):
                axis.plot(x, y, color=palette[j], label=labels[j] if k == 0 else None, **kw)
    axis.legend(fontsize=9)


def plot_size(traces, N, out):
    """Six-panel drift summary for one network size.

    Colour encodes the quantity being compared within each panel (lag, weight matrix, or lag pair);
    the several same-coloured lines are the independent seeds.

    Args:
        traces: list of trace dicts for this N; N: network size; out: output png path.
    """
    fig, ax = plt.subplots(2, 3, figsize=(16, 9))
    lag_cols = ["#1f77b4", "#d62728", "#2ca02c"]                     # lag 100 / 1000 / 10000
    mat_cols = {"W_inp": "#8c564b", "W_rec": "#1f77b4", "W_out": "#e377c2"}
    lw = dict(lw=1.3, alpha=.85)

    # (a) recurrent drift at the three lags
    overlay(ax[0, 0], traces, LAGS, [f"lag $L$={l}" for l in LAGS], lag_cols,
            lambda t, l: series(t, f"drift_W_rec_lag{l}"), **lw)
    ax[0, 0].set(xscale="log", yscale="log", xlabel="iteration",
                 ylabel=r"$\|W_{rec}(t)-W_{rec}(t-L)\|_F\,/\,\|W_{rec}(t)\|_F$")
    ax[0, 0].set_title("(a) recurrent weight drift")

    # (b) the three matrices at the middle lag
    overlay(ax[0, 1], traces, MATS, MATS, [mat_cols[m] for m in MATS],
            lambda t, m: series(t, f"drift_{m}_lag1000"), **lw)
    ax[0, 1].set(xscale="log", yscale="log", xlabel="iteration",
                 ylabel=r"$\|\Delta W\|_F/\|W\|_F$  at $L=1000$")
    ax[0, 1].set_title("(b) which matrix keeps moving")

    # (c) diffusive-vs-systematic exponent
    pairs = [(100, 1000), (1000, 10000)]
    overlay(ax[0, 2], traces, pairs, [f"$L$: {a}$\\to${b}" for a, b in pairs],
            ["#1f77b4", "#ff7f0e"],
            lambda t, p: lag_exponent(t, "W_rec", *p), marker="o", ms=3, **lw)
    ax[0, 2].axhline(0.5, color="k", ls="-", lw=1, alpha=.6)
    ax[0, 2].axhline(1.0, color="r", ls="-", lw=1, alpha=.6)
    ax[0, 2].text(0.02, 0.52, "diffusion", transform=ax[0, 2].get_yaxis_transform(), fontsize=8)
    ax[0, 2].text(0.02, 1.02, "systematic", color="r",
                  transform=ax[0, 2].get_yaxis_transform(), fontsize=8)
    ax[0, 2].set(xscale="log", xlabel="iteration",
                 ylabel=r"$\alpha=\Delta\log d\,/\,\Delta\log L$")
    ax[0, 2].set_title("(c) is the motion a random walk?")

    # (d) directional persistence
    overlay(ax[1, 0], traces, MATS, MATS, [mat_cols[m] for m in MATS],
            lambda t, m: (lambda i, c: (i, running_median(c, 101)))(*series(t, f"cos_{m}")), **lw)
    ax[1, 0].axhline(0, color="k", lw=1)
    ax[1, 0].set(xscale="log", xlabel="iteration",
                 ylabel=r"$\cos\left(\Delta W_t,\ \Delta W_{t-1}\right)$")
    ax[1, 0].set_title("(d) directional persistence (median filt.)")

    # (e) participation drift
    overlay(ax[1, 1], traces, LAGS, [f"lag $L$={l}" for l in LAGS], lag_cols,
            lambda t, l: series(t, f"dp_lag{l}"), **lw)
    ax[1, 1].set(xscale="log", yscale="log", xlabel="iteration",
                 ylabel=r"$\|p(t)-p(t-L)\|\,/\,\|p(t)\|$")
    ax[1, 1].set_title("(e) participation drift")

    # (f) silent-unit count. Nothing is compared here, so this is the one panel where colour is
    # free to carry seed identity — and it is labelled as such.
    seed_cols = plt.cm.viridis(np.linspace(0.15, 0.8, len(traces)))
    for k, t in enumerate(traces):
        ax[1, 2].plot(np.asarray(t["iters"], dtype=float),
                      np.asarray(t["metrics"]["silent_1em6"], dtype=float),
                      color=seed_cols[k], lw=1.2, label=f"seed {k + 1}")
    ax[1, 2].set(xscale="log", xlabel="iteration",
                 ylabel=r"# units with $p_i<10^{-6}$")
    ax[1, 2].set_title(f"(f) silent units (of N={N})")
    ax[1, 2].legend(fontsize=9)

    for a in ax.ravel():
        a.grid(alpha=.25)
    fig.suptitle(f"Drift during training, standard ReLU RNN, no penalties, N={N}\n"
                 f"COLOUR = the quantity compared in that panel (see its legend);  "
                 f"the {len(traces)} same-coloured lines are the {len(traces)} seeds  "
                 f"(panel (f) excepted: there colour = seed)\n"
                 r"$p_i=\mathrm{std}_{t,c}\,r_i+q_{0.9}|r_i|$;  "
                 r"$d(L)=\|W(t)-W(t-L)\|_F/\|W(t)\|_F$", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def report(traces, N):
    """Print the numbers behind the figure: end-state drift, exponents, silent counts."""
    print(f"\n=== N={N}, {len(traces)} seeds ===")
    for t in traces:
        it = np.asarray(t["iters"], dtype=float)
        s = np.asarray(t["metrics"]["silent_1em6"], dtype=float)
        line = [f"iters={int(it[-1]) + 10}"]
        for lag in LAGS:
            i, d = series(t, f"drift_W_rec_lag{lag}")
            if len(d):
                line.append(f"d_rec({lag})={d[-1]:.4f}")
        for (l1, l2) in [(100, 1000), (1000, 10000)]:
            i, a = lag_exponent(t, "W_rec", l1, l2)
            if len(a):
                line.append(f"alpha({l1}-{l2})={a[-1]:.2f}")
        i, c = series(t, "cos_W_rec")
        c = running_median(c, 101)
        first = c[np.searchsorted(i, 1000)] if i[-1] > 1000 else c[0]
        line.append(f"cos_rec {first:.2f}->{c[-1]:.2f}")
        line.append(f"silent {int(s[0])}->{int(s.max())}max->{int(s[-1])}")
        print("  " + "  ".join(line))


def main():
    """Load a drift sweep, write one figure per network size, print the summary table."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    sweep = args[0] if args else "data/trained_RNNs/CDDM_std_g0_drift"
    by_n = load_traces(sweep)
    if not by_n:
        sys.exit(f"no traces under {sweep}")
    for N in sorted(by_n):
        report(by_n[N], N)
        plot_size(by_n[N], N, os.path.join(IMG_DIR, f"drift_N{N}.png"))


if __name__ == "__main__":
    main()
