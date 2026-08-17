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
                              In practice it saturates at that floor ~10x earlier than the exponent
                              transitions, so it is a diagnostic here, NOT a stopping criterion.

alpha is not a single number: it depends on the lag it is measured at. Estimating it from two lags
therefore averages a rising curve and is biased low, not merely noisy. For the participation vector
this is fixable with no re-run — the full per-unit p is stored every `store_participation_every`
iterations, so the displacement can be time-averaged at EVERY multiple of that cadence and the local
slope read off directly (see msd_participation / plot_msd). The weight matrices are not stored, only
their scalar drifts at the configured lags, so resolving alpha for W needs either a longer
`drift_lags` list or (cheaper) storing fixed random projections of each matrix per probe.

Outputs, per network size in the sweep:
  img/internal_figures/drift_N<N>.png      the six online diagnostics
  img/internal_figures/drift_msd_N<N>.png  alpha_p resolved at every lag, and the caging timescale

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


def lag_exponent(trace, what, l1, l2):
    """Local scaling exponent of drift with lag: alpha = log(d(l2)/d(l1)) / log(l2/l1).

    The two lags are measured at different iterations, so the shorter-lag series is interpolated in
    log-iteration onto the longer-lag grid before the ratio is taken.

    Args:
        trace: trace dict;
        what: "W_inp"/"W_rec"/"W_out" for a weight matrix, or "p" for the participation vector;
        l1, l2: lags with l1 < l2.
    Returns:
        (iters, alpha) arrays. alpha ~ 0.5 diffusive, ~ 1 systematic, < 0.5 mean-reverting.
    """
    stem = "dp" if what == "p" else f"drift_{what}"
    i1, d1 = series(trace, f"{stem}_lag{l1}")
    i2, d2 = series(trace, f"{stem}_lag{l2}")
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

    # (c) diffusive-vs-systematic exponent, for the weights and for the participation vector.
    # The participation exponent at the LONGEST lag is the criterion: weights are free to wander
    # along functionally degenerate directions (W_out is doing exactly that, see panel b), so a
    # weight-based stopping rule fires long before the network stops changing what its units do.
    curves = [("W_rec", 100, 1000), ("W_rec", 1000, 10000),
              ("p", 100, 1000), ("p", 1000, 10000)]
    overlay(ax[0, 2], traces, curves,
            [f"{'$W_{rec}$' if w == 'W_rec' else '$p$'}  $L$: {a}$\\to${b}" for w, a, b in curves],
            ["#9ecae1", "#1f77b4", "#ff9896", "#d62728"],   # saturated = the longest, criterion lag
            lambda t, c: lag_exponent(t, c[0], c[1], c[2]), marker="o", ms=3, **lw)
    ax[0, 2].axhline(0.5, color="k", ls="-", lw=1, alpha=.6)
    ax[0, 2].axhline(1.0, color="r", ls="-", lw=1, alpha=.6)
    ax[0, 2].text(0.02, 0.52, "diffusion", transform=ax[0, 2].get_yaxis_transform(), fontsize=8)
    ax[0, 2].text(0.02, 1.02, "systematic", color="r",
                  transform=ax[0, 2].get_yaxis_transform(), fontsize=8)
    ax[0, 2].set(xscale="log", xlabel="iteration",
                 ylabel=r"$\alpha=\Delta\log d\,/\,\Delta\log L$")
    ax[0, 2].set_title("(c) random walk?  ($p$ at longest $L$ = criterion)")

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


def msd_participation(p_store, iters, t0, t1, direction_only=False):
    """Time-averaged mean displacement of the participation vector versus lag.

    The online `dp_lag<L>` metric gives one value per configured lag, so the exponent has to be
    estimated from a two-point slope. The stored participation vectors give EVERY lag that is a
    multiple of store_participation_every, averaged over all time-pairs in the window — a far
    better-conditioned estimate, and it needs no re-run because the vectors are already on disk.
    (The weight matrices are not stored, only their scalar drifts, so the same trick is not
    available for W and more lags there would require re-running with a longer `drift_lags`.)

    Args:
        p_store: (n_stored, N) participation vectors.
        iters: (n_stored,) iteration of each stored vector.
        t0, t1: iteration bounds of the window to average within.
        direction_only: if True, normalise each vector to unit length first, so the result measures
            reorganisation of the participation PATTERN and ignores overall rate inflation. This
            matters here because ||p|| itself grows ~2x over training, and a steadily growing norm
            is a systematic displacement that would masquerade as directed motion.
    Returns:
        (lags, d) arrays, d dimensionless. Empty if the window holds too few vectors.
    """
    sel = (iters >= t0) & (iters <= t1)
    P, I = p_store[sel], iters[sel]
    if len(P) < 8:
        return np.array([]), np.array([])
    if direction_only:
        P = P / np.maximum(np.linalg.norm(P, axis=1, keepdims=True), 1e-30)
    # Normalise by the window-mean norm rather than the instantaneous one: within a window ||p||
    # drifts, and dividing by a moving denominator folds that drift into the numerator.
    den = np.linalg.norm(P, axis=1).mean()
    lags, ds = [], []
    for k in range(1, len(P) // 2 + 1):     # //2 keeps at least half the window as pair statistics
        ds.append(float(np.linalg.norm(P[k:] - P[:-k], axis=1).mean() / den))
        lags.append(float(I[k] - I[0]))
    return np.array(lags), np.array(ds)


def local_slope(x, y, smooth=5):
    """Local log-log slope dlog(y)/dlog(x), median-smoothed. Returns array like x."""
    return running_median(np.gradient(np.log(y), np.log(x)), smooth)


def caging_timescale(p_store, iters, t0, t1, direction_only=False):
    """L*, the lag at which alpha_p(L) crosses 0.5 within a training window.

    Below L* the participation vector is effectively caged (successive changes cancel); above it the
    motion is directed. L* is therefore "how long you have to wait before training takes the network
    somewhere", measured at training age (t0+t1)/2.

    Args:
        p_store, iters, t0, t1, direction_only: as for msd_participation.
    Returns:
        L* in iterations, or nan if the window never crosses 0.5 (or crosses only at its last lag,
        where the crossing is not bracketed and would be an extrapolation).
    """
    L, d = msd_participation(p_store, iters, t0, t1, direction_only=direction_only)
    if not len(L):
        return float("nan")
    a = local_slope(L, d)
    below = np.where(a < 0.5)[0]
    if not len(below) or below[-1] == len(a) - 1:
        return float("nan")
    return float(L[below[-1]])


def plot_msd(traces, N, out):
    """Participation MSD and its exponent across all resolvable lags, in windows across training.

    Args:
        traces: list of trace dicts for this N; N: network size; out: output png path.
    """
    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    end = max(t["participation_iters"][-1] for t in traces)
    wins = [(0, end // 2), (end // 4, 3 * end // 4), (end // 2, end)]
    wcols = ["#c6dbef", "#6baed6", "#08519c"]
    wlab = [f"iter {a // 1000}k-{b // 1000}k" for a, b in wins]
    half = end // 5           # sliding-window half-width for the L*(t) panels

    for col, (dir_only, ttl) in enumerate([(False, r"full $p$"),
                                           (True, r"direction of $p$ only")]):  # cols 0,1
        for w, c, lb in zip(wins, wcols, wlab):
            for t in traces:
                P = np.array(t["participation"])
                I = np.array(t["participation_iters"])
                L, d = msd_participation(P, I, w[0], w[1], direction_only=dir_only)
                if not len(L):
                    continue
                first = t is traces[0]
                ax[0, col].plot(L, d, color=c, lw=1.3, alpha=.85, label=lb if first else None)
                ax[1, col].plot(L, local_slope(L, d), color=c, lw=1.3, alpha=.85,
                                label=lb if first else None)
        ref = np.array([1e2, 1e4])
        base = 0.02 if not dir_only else 0.01
        ax[0, col].plot(ref, base * (ref / 1e2) ** 0.5, "-", color="k", lw=1, alpha=.5)
        ax[0, col].plot(ref, base * (ref / 1e2) ** 1.0, "--", color="r", lw=1, alpha=.5)
        ax[0, col].set(xscale="log", yscale="log", xlabel="lag $L$ (iterations)",
                       ylabel=r"$\langle\|p(t)-p(t-L)\|\rangle_t / \langle\|p\|\rangle$")
        ax[0, col].set_title(f"({'ab'[col]}) displacement vs lag, {ttl}")
        ax[1, col].axhline(0.5, color="k", lw=1, alpha=.6)
        ax[1, col].axhline(1.0, color="r", lw=1, alpha=.6)
        ax[1, col].set(xscale="log", ylim=(-0.15, 1.35), xlabel="lag $L$ (iterations)",
                       ylabel=r"$\alpha_p(L) = \mathrm{d}\log d / \mathrm{d}\log L$")
        ax[1, col].set_title(f"({'cd'[col]}) exponent, {ttl}")
        for r in (0, 1):
            ax[r, col].legend(fontsize=8)
            ax[r, col].grid(alpha=.25)

    # (e,f) the caging timescale versus training age. If L* grows in proportion to t, the network is
    # always still directed on timescales comparable to its own age and never settles in the
    # relative sense, however long it trains.
    centres = np.arange(half + half // 2, end - half + 1, max(half // 4, 1))
    for dir_only, c, nm in [(False, "#08519c", r"full $p$"), (True, "#d62728", r"$\hat p$ only")]:
        T, S = [], []
        for t in traces:
            P = np.array(t["participation"])
            I = np.array(t["participation_iters"])
            for ctr in centres:
                v = caging_timescale(P, I, ctr - half, ctr + half, direction_only=dir_only)
                if np.isfinite(v):
                    T.append(float(ctr))
                    S.append(v)
        if len(T) < 3:
            continue
        T, S = np.array(T), np.array(S)
        b = np.polyfit(np.log(T), np.log(S), 1)[0]
        ax[0, 2].plot(T, S, "o", color=c, ms=5, alpha=.7,
                      label=f"{nm}:  $L^*\\propto t^{{{b:.2f}}}$")
        ax[1, 2].plot(T, S / T, "o", color=c, ms=5, alpha=.7,
                      label=f"{nm}:  {np.mean(S / T):.3f} $\\pm$ {np.std(S / T):.3f}")
    ax[0, 2].plot([centres[0], centres[-1]], [0.083 * centres[0], 0.083 * centres[-1]],
                  "k--", lw=1, alpha=.6, label=r"$L^*=0.083\,t$")
    ax[0, 2].set(xscale="log", yscale="log", xlabel="training age $t$ (iterations)",
                 ylabel=r"$L^*$ = lag where $\alpha_p$ crosses 0.5")
    ax[0, 2].set_title("(e) caging timescale vs training age")
    ax[1, 2].axhline(0, color="k", lw=1)
    ax[1, 2].set(xscale="log", ylim=(0, None), xlabel="training age $t$ (iterations)",
                 ylabel=r"$L^*/t$")
    ax[1, 2].set_title(r"(f) $L^*$ as a fraction of age")
    for r in (0, 1):
        ax[r, 2].legend(fontsize=8)
        ax[r, 2].grid(alpha=.25)

    fig.suptitle(f"Participation drift resolved at every lag, N={N} "
                 f"($p$ stored every 100 iters, averaged over all time-pairs in the window)\n"
                 f"COLOUR = training window;  the {len(traces)} same-coloured lines are the seeds.  "
                 "Solid black = $L^{0.5}$ (diffusion), dashed red = $L^{1}$ (directed).\n"
                 r"Right column uses $\hat p = p/\|p\|$, removing overall rate inflation so only "
                 "reorganisation of the pattern counts.", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


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
        plot_msd(by_n[N], N, os.path.join(IMG_DIR, f"drift_msd_N{N}.png"))


if __name__ == "__main__":
    main()
