#!/usr/bin/env python3
"""
Training-loss curves per seed, on a log iteration axis with the noise smoothed away.

The raw trace is one value per iteration from a NOISY forward pass (Trainer.train_step runs with
w_noise=True), so plotted directly it is a dense band ~0.01 wide that hides the underlying curve
completely once training settles. Two things fix that:

  log-spaced binning   the x axis is logarithmic, so a linear-in-t running mean over-smooths early
                       training and under-smooths late training. Binning in LOG t instead puts a
                       comparable number of points in every visual interval.
  median, not mean     the loss has a heavy lower tail (favourable noise draws), which drags a mean
                       down; the median tracks the typical value.

The IQR band is the noise amplitude, not uncertainty about the curve - it shows how wide the raw
forest is at each stage, which is the reason smoothing was needed in the first place.

Four decay families are fitted to the smoothed curve and overlaid, each carrying a different claim
about how training ends:

  pure power law      L = A t^-g               no floor at all: the loss reaches zero eventually
  power law + floor   L = Linf + A t^-g        scale-free approach to a nonzero floor
  exponential + floor L = Linf + A exp(-t/tau) ONE timescale; effectively done after a few tau
  stretched + floor   L = Linf + A exp(-(t/tau)^b)  a broad DISTRIBUTION of timescales, b<1

Fitting is done on log L (the noise is multiplicative, and the loss spans a decade), on t >= FIT_FROM
to skip the initial transient, and the fits are DRAWN over the full range so the early misfit is
visible rather than hidden. The residual panel is the point of the figure: a family that fits will
leave residuals scattered about zero, whereas systematic arcs mean the shape is wrong no matter how
good the AICc looks.

Output: img/internal_figures/loss_curves_N<N>.png

Usage:  python plot_loss_curves.py [N ...]        (default: 2000 1000)
"""

import os
import sys
import numpy as np
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_fit import load_losses, IMG_DIR

T_MIN = 20
N_BINS = 220
FIT_FROM = 2000        # skip the initial transient; the families describe the approach, not the drop


def f_pow_floor(t, Li, A, g):
    """Power-law approach to a floor: Linf + A t^-gamma."""
    return Li + A * t ** (-g)


def f_pow_pure(t, A, g):
    """Pure power law with no floor: A t^-gamma."""
    return A * t ** (-g)


def f_exp(t, Li, A, tau):
    """Single-timescale exponential approach: Linf + A exp(-t/tau)."""
    return Li + A * np.exp(-t / tau)


def f_stretch(t, Li, A, tau, b):
    """Stretched exponential approach: Linf + A exp(-(t/tau)^beta). Free tau; see f_stretch1."""
    return Li + A * np.exp(-np.power(np.clip(t / tau, 1e-12, None), b))


def f_stretch1(t, Li, A, b):
    """Stretched exponential with tau FIXED at 1: Linf + A exp(-t^beta).

    tau is not identifiable from these curves: profiling it over FOUR orders of magnitude
    (0.1 to 1000) changes the residual sum of squares by under 10%, while A slides by 14x to
    compensate. Freeing it therefore buys no fit quality and destroys the interpretability of both
    A and beta (free-tau fits gave beta = 0.14-0.27 with tau spanning 0.24 to 258 across seeds).
    Fixing tau=1 leaves three parameters - the same count as the power law, so the two can be
    compared fairly - and pins beta to +-0.001.
    """
    return Li + A * np.exp(-np.power(np.clip(t, 1e-12, None), b))


# Both retained families are shown. The stretched exponential is the better SHAPE - it wins not only
# on AICc (which already charges for its fourth parameter) but out-of-sample, with roughly half the
# error when fitted on t<=50k and scored on 50k-200k (see fit_params_vs_N.py). Its tau and beta are
# nevertheless badly identified across seeds, so the power law remains the one to QUOTE parameters
# from. Both are drawn so the residual panel can show which shape is actually right.
FAMILIES = [
    ("power law + floor", f_pow_floor, lambda y: [y.min() * .95, 1.0, .5],
     ([0, 1e-9, .01], [1, 1e6, 3])),
    ("stretched, tau=1", f_stretch1, lambda y: [y.min() * .95, .3, .2],
     ([0, 1e-9, .02], [1, 1e3, 1.5])),
]


def fit_families(t, y):
    """Fit every family to log(y) and rank by AICc.

    Args:
        t, y: log-binned iteration centres and median losses, restricted to the fit range.
    Returns:
        list of (name, params, aicc, predict_callable), sorted best first.
    """
    out = []
    for name, fn, p0, bounds in FAMILIES:
        try:
            p, _ = curve_fit(lambda x, *q: np.log(np.clip(fn(x, *q), 1e-300, None)),
                             t, np.log(y), p0=p0(y), bounds=bounds, maxfev=200000)
            r = np.log(y) - np.log(np.clip(fn(t, *p), 1e-300, None))
            n, k = len(t), len(p)
            a = n * np.log(float(r @ r) / n) + 2 * k + (2 * k * (k + 1)) / max(n - k - 1, 1e-9)
            out.append((name, p, a, (lambda x, f=fn, q=p: f(x, *q))))
        except Exception:
            out.append((name, None, float("inf"), None))
    return sorted(out, key=lambda z: z[2])


def logbin_stats(L, nbins=N_BINS, t_min=T_MIN):
    """Median and interquartile range of the loss in log-spaced iteration bins.

    Args:
        L: loss array, one value per iteration; nbins: number of log-spaced bins;
        t_min: first iteration to include.
    Returns:
        (centres, median, q25, q75), empty bins dropped.
    """
    t = np.arange(1, len(L) + 1, dtype=float)
    edges = np.logspace(np.log10(t_min), np.log10(len(L)), nbins + 1)
    idx = np.digitize(t, edges) - 1
    c, m, lo, hi = [], [], [], []
    for b in range(nbins):
        s = idx == b
        if s.sum() >= 3:
            v = L[s]
            c.append(np.sqrt(edges[b] * edges[b + 1]))
            m.append(np.median(v))
            lo.append(np.percentile(v, 25))
            hi.append(np.percentile(v, 75))
    return tuple(np.array(x) for x in (c, m, lo, hi))


def plot_one(entries, N, out):
    """Loss curves for every seed of one network size, log x, log-binned median with IQR band.

    Args:
        entries: list of (tag, loss array); N: network size; out: output png path.
    """
    fig, ax = plt.subplots(1, 2, figsize=(14.5, 5.6))
    cols = plt.cm.viridis(np.linspace(0.12, 0.78, len(entries)))
    fcol = {"power law + floor": "C3", "stretched, tau=1": "C4"}
    for k, (tag, L) in enumerate(entries):
        c, m, lo, hi = logbin_stats(L)
        ax[0].plot(c, m, "-", color=cols[k], lw=1.6, alpha=.9,
                   label=f"seed {k+1} ({tag[:9]})" if k < 3 else None)
        ax[0].fill_between(c, lo, hi, color=cols[k], alpha=.15, lw=0)
        print(f"  N={N} {tag[:9]}: {len(L)} iters, loss {m[0]:.4f} -> {m[-1]:.5f}")

    # fit families to the FIRST seed only, so the overlay stays readable
    c, m, _, _ = logbin_stats(entries[0][1])
    sel = c >= FIT_FROM
    ranked = fit_families(c[sel], m[sel])
    print(f"  --- decay-family fits (seed 1, t >= {FIT_FROM}), ranked by AICc")
    best = ranked[0][2]
    for name, p, a, pred in ranked:
        if pred is None:
            print(f"      {name:22s} FAILED")
            continue
        if name.startswith("power"):
            lbl = (r"$L_\infty+A\,t^{-\gamma}$: " +
                   f"$L_\\infty$={p[0]:.5f}, $A$={p[1]:.3g}, $\\gamma$={p[2]:.3f}")
        else:
            lbl = (r"$L_\infty+A\,e^{-t^{\beta}}$: " +
                   f"$L_\\infty$={p[0]:.5f}, $A$={p[1]:.3g}, $\\beta$={p[2]:.4f}")
        ax[0].plot(c, pred(c), "--", color=fcol[name], lw=2.2,
                   label=f"{lbl}  (dAICc {a-best:+.1f})")
        ax[1].plot(c[sel], np.log(m[sel]) - np.log(pred(c[sel])), "-", color=fcol[name], lw=1.6,
                   label=name)
        ps = ", ".join(f"{v:.4g}" for v in p)
        print(f"      {name:22s} dAICc {a-best:+7.1f}   params [{ps}]")

    ax[1].axhline(0, color="k", lw=1.2)
    ax[0].set(xscale="log", yscale="log", xlabel="iteration", ylabel="training loss",
              title="(a) loss with power-law and stretched-exponential fits\n"
                    "(both drawn beyond their fit range on purpose)")
    ax[1].set(xscale="log", xlabel="iteration", ylabel=r"$\log L_{data}-\log L_{fit}$",
              title="(b) residuals — flat about zero = right shape;\na systematic bow = wrong shape")
    for a in ax:
        a.grid(alpha=.3, which="both")
        a.legend(fontsize=7.5)
    fig.suptitle(f"Training loss with decay fits, N={N} — log-binned median per seed\n"
                 r"both families have THREE parameters ($\tau$ fixed at 1: it is unidentifiable, "
                 r"profiling it over $10^4$ changes RSS by <10%)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.89])
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


def main():
    """Plot loss curves for each requested network size."""
    args = [int(a) for a in sys.argv[1:] if a.isdigit()] or [2000, 1000]
    by = load_losses("data/trained_RNNs/CDDM_std_g0_drift")
    for N in args:
        if N not in by:
            print(f"no data for N={N}; have {sorted(by)}")
            continue
        plot_one(by[N], N, os.path.join(IMG_DIR, f"loss_curves_N{N}.png"))


if __name__ == "__main__":
    main()
