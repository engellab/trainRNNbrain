#!/usr/bin/env python3
"""
Is silencing governed by ONE variable - capacity per unit of task demand - rather than by N and k?

THE CLAIM UNDER TEST. The (N, k) contour map showed iso-M contours running diagonally: the active
count can be held fixed by trading network size against task complexity. If that is exactly true then
the silent fraction does not depend on N and k separately, but on a single combination

    x = N / D(k)          D(k) = task demand

and every cell, whatever its N and k, falls on ONE master curve when plotted against x. That is a
much stronger statement than "silencing rises with N and falls with k" - it is a law with a formula,
and it predicts cells that were never run.

TWO CANDIDATE DEMANDS, from the structure of the task itself:

    power        D = k^alpha     demand grows with the readout dimension (k independent bits)
    exponential  D = 2^(beta k)  demand grows with the number of attractor states (2^k corners)

Both are fitted; whichever collapses better wins, and the fitted exponent is the result.

HOW IT IS TESTED, rather than eyeballed. Four nested models on the same seed-level points:

    N-only      quadratic in log N, k ignored                        3 params
    k-only      quadratic in log k, N ignored                        3 params
    COLLAPSE    quadratic in log x, with the exponent fitted         4 params
    saturated   a free mean for every one of the 15 cells           15 params

The collapse is real if it fits nearly as well as the SATURATED model - i.e. one variable captures
what fifteen free cell means capture. That is the strong test, and an F-test against the saturated
model is the honest way to fail it. Beating the N-only and k-only models is necessary but weak.

Read at several performance levels, because a collapse that only holds at one level is a coincidence.
The exponent should also be stable across levels; if it drifts, say so.

Output: img/internal_figures/flipflop_collapse.png

Usage:  python flipflop_collapse.py
"""

import os
import sys
import numpy as np
from scipy.stats import f as fdist
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import IMG_DIR
from flipflop_ksweep import load, stable_crossing, silence_at, TGT_VAR

N_BOOT = 400


def points(by, ks, Ns, thr, crit):
    """Seed-level (k, N, silent%) at one performance level.

    Args:
        by: loaded traces; ks, Ns: axes; thr: loss level, or None for the endpoint;
        crit: 0 for the hard criterion, 1 for scale-free.
    Returns:
        (k, N, y) float arrays, one entry per seed that reaches the level.
    """
    K, NN, Y = [], [], []
    for k in ks:
        for N in Ns:
            for t in by.get((k, N), []):
                x = None if thr is None else stable_crossing(t["clean"], thr)
                if thr is not None and x is None:
                    continue
                K.append(k)
                NN.append(N)
                Y.append(silence_at(t, N, x)[crit])
    return np.array(K, float), np.array(NN, float), np.array(Y, float)


def rss_poly(x, y, deg=2):
    """Residual sum of squares of a degree-`deg` polynomial fit of y on x."""
    if len(np.unique(x)) <= deg:
        return float(np.sum((y - y.mean()) ** 2))
    r = y - np.polyval(np.polyfit(x, y, deg), x)
    return float(r @ r)


def best_collapse(K, NN, Y, family, grid):
    """Exponent giving the tightest collapse, and its RSS.

    Args:
        K, NN, Y: seed-level arrays; family: "power" (x = logN - a*logK) or "exp" (x = logN - a*K);
        grid: exponents to search.
    Returns:
        (best exponent, RSS at it, RSS over the whole grid).
    """
    curve = []
    for a in grid:
        x = np.log(NN) - (a * np.log(K) if family == "power" else a * K)
        curve.append(rss_poly(x, Y))
    curve = np.array(curve)
    j = int(np.argmin(curve))
    return grid[j], curve[j], curve


def main():
    """Fit and test the single-variable collapse at several performance levels."""
    by, _ = load()
    ks = sorted({k for k, _ in by})
    Ns = sorted({N for _, N in by})
    grid_p = np.linspace(0.0, 20.0, 801)
    grid_e = np.linspace(0.0, 8.0, 801)

    readings = [(None, "endpoint (300k)"), (0.0375, "R^2 = 0.949"),
                (0.0220, "R^2 = 0.970"), (0.0150, "R^2 = 0.980")]
    crit_name = {0: "hard  p<1e-6", 1: "scale-free"}
    store = {}

    for crit in (1, 0):
        print(f"\n{'=' * 78}\nCRITERION: {crit_name[crit]}\n{'=' * 78}")
        for thr, lab in readings:
            K, NN, Y = points(by, ks, Ns, thr, crit)
            ncell = len({(a, b) for a, b in zip(K, NN)})
            if len(Y) < 12 or ncell < 8:
                print(f"\n{lab}: only {ncell} cells reachable — skipped")
                continue

            a_p, rss_p, _ = best_collapse(K, NN, Y, "power", grid_p)
            a_e, rss_e, _ = best_collapse(K, NN, Y, "exp", grid_e)
            rss_Nonly = rss_poly(np.log(NN), Y)
            rss_konly = rss_poly(np.log(K), Y)
            # Saturated: a free mean per cell. Its RSS is pure within-cell (seed) scatter, so it is
            # the floor any model could reach without fitting noise.
            rss_sat = float(sum(((Y[m] - Y[m].mean()) ** 2).sum()
                                for m in [(K == a) & (NN == b) for a in ks for b in Ns]
                                if m.sum()))
            n = len(Y)
            best_fam, a_best, rss_best = (("power", a_p, rss_p) if rss_p <= rss_e
                                          else ("exp", a_e, rss_e))

            # F-test of COLLAPSE against SATURATED: can one variable stand in for 15 cell means?
            d1, d2 = ncell - 4, n - ncell
            F = ((rss_best - rss_sat) / d1) / (rss_sat / d2) if d2 > 0 and d1 > 0 else np.nan
            p = 1 - fdist.cdf(F, d1, d2) if np.isfinite(F) else np.nan

            boot = []
            for _ in range(N_BOOT):
                i = np.random.randint(0, n, n)
                if len(np.unique(K[i])) < 3 or len(np.unique(NN[i])) < 2:
                    continue
                boot.append(best_collapse(K[i], NN[i], Y[i], best_fam,
                                          grid_p if best_fam == "power" else grid_e)[0])
            lo, hi = np.percentile(boot, [2.5, 97.5]) if boot else (np.nan, np.nan)

            tot = float(((Y - Y.mean()) ** 2).sum())
            print(f"\n{lab}   ({ncell} cells, {n} seed points)")
            print(f"  variance explained:  N only {1 - rss_Nonly / tot:6.3f}   "
                  f"k only {1 - rss_konly / tot:6.3f}   COLLAPSE {1 - rss_best / tot:6.3f}   "
                  f"saturated {1 - rss_sat / tot:6.3f}")
            print(f"  best family: {best_fam}   "
                  + (f"D = k^{a_best:.2f}  [95% CI {lo:.2f}, {hi:.2f}]" if best_fam == "power"
                     else f"D = 2^({a_best:.2f}k)  [95% CI {lo:.2f}, {hi:.2f}]"))
            print(f"    (power  D=k^{a_p:.2f} RSS {rss_p:8.1f}   |   "
                  f"exp D=2^({a_e:.2f}k) RSS {rss_e:8.1f})")
            # DEGENERACY GUARD. A "collapse" that drives the exponent to the search boundary is not
            # a collapse: x = N/D(k) then varies almost entirely with k and the model silently
            # becomes the k-only model. Passing the F-test in that state means "k alone matches 15
            # cell means", which is a different and much weaker claim. Detect it before reporting.
            gmax = (grid_p if best_fam == "power" else grid_e)[-1]
            # Unidentified if the point estimate OR the bootstrap upper bound sits at the search
            # boundary: widening the grid then just moves the answer, which is the signature of an
            # exponent running to infinity rather than a real optimum.
            edge = (a_best >= 0.98 * gmax) or (a_best <= 1e-9) or (np.isfinite(hi) and hi >= 0.98 * gmax)
            beats_k = (1 - rss_best / tot) > (1 - rss_konly / tot) + 0.01
            print(f"  COLLAPSE vs SATURATED:  F({d1},{d2}) = {F:.2f}, p = {p:.3g}")
            if edge or not beats_k:
                print("  -> DEGENERATE: exponent at the search boundary and/or no better than "
                      "k-only.\n     The 'single variable' has collapsed to k alone - N carries "
                      "no independent weight,\n     so there is no capacity-vs-demand trade-off "
                      "to collapse here.")
            elif p > 0.05:
                print("  -> collapse HOLDS (one variable does the work of 15 cell means)")
            else:
                print("  -> collapse REJECTED (cells differ beyond a single variable)")
            store[(crit, thr)] = (K, NN, Y, best_fam, a_best, lab, p)

    # ---- figure: before and after, at the endpoint, for the scale-free criterion --------------
    keys = [k for k in store if k[0] == 1]
    if not keys:
        print("\nnothing to plot")
        return
    fig, ax = plt.subplots(2, len(keys), figsize=(5.6 * len(keys), 9), squeeze=False)
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(ks)))
    for c, key in enumerate(keys):
        K, NN, Y, fam, a, lab, p = store[key]
        for k in ks:
            m = K == k
            if m.sum():
                ax[0][c].plot(NN[m], Y[m], "o", color=cols[ks.index(k)], ms=7, label=f"k={k}")
        ax[0][c].set(xscale="log", xlabel="$N$", ylabel="silent units (% of N)",
                     title=f"BEFORE — {lab}\nfive separate curves", ylim=(-3, 100))
        xx = np.log(NN) - (a * np.log(K) if fam == "power" else a * K)
        for k in ks:
            m = K == k
            if m.sum():
                ax[1][c].plot(np.exp(xx[m]), Y[m], "o", color=cols[ks.index(k)], ms=7, label=f"k={k}")
        if len(np.unique(xx)) > 2:
            xs = np.linspace(xx.min(), xx.max(), 100)
            ax[1][c].plot(np.exp(xs), np.polyval(np.polyfit(xx, Y, 2), xs), "k-", lw=1.6,
                          alpha=.7, label="master curve")
        dlab = f"N/k^{{{a:.2f}}}" if fam == "power" else f"N/2^{{{a:.2f}k}}"
        ax[1][c].set(xscale="log", xlabel=f"${dlab}$   (capacity per unit demand)",
                     ylabel="silent units (% of N)", ylim=(-3, 100),
                     title=f"AFTER — collapse onto ${dlab}$\n"
                           f"vs saturated model: p = {p:.3g}")
        for a_ in (ax[0][c], ax[1][c]):
            a_.legend(fontsize=8)
            a_.grid(alpha=.3)
    fig.suptitle("Does one variable — units per unit of task demand — govern silencing?\n"
                 "scale-free criterion; collapse holds only if it matches a free mean per cell",
                 fontsize=12)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "flipflop_collapse.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
