#!/usr/bin/env python3
"""
Statistical test: does the active-unit count M saturate with network size N, or follow a power law?

THE TWO HYPOTHESES, stated as they differ in the data:

  H0  power law      M = A * N^k          k constant in N; M unbounded as N grows
  H1  saturation     M -> M* as N -> inf  the local exponent dlog M/dlog N falls toward 0

On log-log axes H0 is a straight line and H1 is a line that BENDS DOWNWARD. So the sharp version of
the question is whether the data require curvature, and the sharp test is whether a quadratic term
in log N is significantly negative.

WHY N=100 IS EXCLUDED FROM EVERY FIT. At N=100 the networks sit at 94-99% active, i.e. hard against
the M <= N ceiling: a 100-unit network cannot show that it "wants" 120 active units. That censoring
flattens M(100) and therefore INFLATES the apparent exponent from 100 to 500, manufacturing exactly
the downward curvature H1 predicts. Including it would bias the test toward saturation. Only the
ceiling-free sizes (N >= 500, all below 78% active) are fitted.

TEST 1 - curvature, pooled across levels.
    log M = a_level + k * log N                    (H0, one intercept per matching level)
    log M = a_level + k * log N + c * (log N)^2    (H1)
  Nested, so a likelihood-ratio / F test on c is exact under Gaussian errors. Curvature is
  identifiable from only three distinct N because seeds provide replicate variance. Saturation
  predicts c < 0. Pooling levels shares one curvature across four otherwise-independent datasets,
  which is what buys the degrees of freedom that a single level cannot provide.

TEST 2 - explicit saturating fits, per level.
    hyperbolic    M = M* N / (N + N0)
    exponential   M = M* (1 - exp(-N/N0))
  Reported with the profile-likelihood confidence interval on M*. The informative outcome is usually
  not the point estimate but the UPPER limit: if it runs away, the data cannot bound a ceiling, and
  "saturation" is unfalsifiable rather than merely disfavoured.

TEST 3 - identification of k, per level, with bootstrap CI over seeds.

CRITERION FIXED BEFORE RUNNING: curvature is called significant at p < 0.05. Model comparison uses
AICc (n is small); dAICc > 4 substantial, > 10 decisive.

Output: img/internal_figures/saturation_test.png

Usage:  python test_saturation.py [SWEEP_FOLDER]
"""

import os
import sys
import numpy as np
from scipy import stats
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import load_traces
from plot_loss_fit import IMG_DIR
from plot_M_vs_N import T_at_loss, active_count, ladder, CRITERIA

N_MIN = 500          # ceiling-free sizes only; see module docstring
N_BOOT = 4000


def collect(by, levels, criterion):
    """Per-seed active counts at every matching level.

    Args:
        by: {N: [traces]}; levels: list of L* values; criterion: "hard" or "scalefree".
    Returns:
        dict {L*: {N: [M per seed]}}.
    """
    out = {}
    for Lstar in levels:
        d = {}
        for N in sorted(by):
            vals = []
            for t in by[N]:
                T = T_at_loss(t["loss"], Lstar)
                P = np.array(t["participation"])
                I = np.array(t["participation_iters"])
                if T is not None and T <= I[-1]:
                    vals.append(active_count(P[np.argmin(abs(I - T))], criterion))
            if vals:
                d[N] = vals
        out[Lstar] = d
    return out


def design(data, quad):
    """Build the pooled regression matrices: one intercept per level, shared slope (and curvature).

    Args:
        data: {L*: {N: [M]}}; quad: include the (log N)^2 column.
    Returns:
        (X, y, level_index) with y = log M.
    """
    rows, y, li = [], [], []
    levels = sorted(data)
    for i, Lstar in enumerate(levels):
        for N, vals in data[Lstar].items():
            if N < N_MIN:
                continue
            for v in vals:
                r = [0.0] * len(levels)
                r[i] = 1.0
                r.append(np.log(N))
                if quad:
                    r.append(np.log(N) ** 2)
                rows.append(r)
                y.append(np.log(v))
                li.append(i)
    return np.array(rows), np.array(y), np.array(li)


def ols(X, y):
    """Ordinary least squares. Returns (beta, residual sum of squares, dof)."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    r = y - X @ beta
    return beta, float(r @ r), len(y) - X.shape[1]


def aicc(rss, n, k):
    """Small-sample corrected AIC for a Gaussian least-squares fit."""
    a = n * np.log(rss / n) + 2 * k
    return a + (2 * k * (k + 1)) / max(n - k - 1, 1e-9)


def sat_hyper(N, Mstar, N0):
    """Hyperbolic saturation M = M* N / (N + N0); tends to M* as N grows."""
    return Mstar * N / (N + N0)


def sat_exp(N, Mstar, N0):
    """Exponential saturation M = M* (1 - exp(-N/N0))."""
    return Mstar * (1.0 - np.exp(-N / N0))


def powerlaw(N, A, k):
    """Unbounded power law M = A N^k."""
    return A * N ** k


def profile_upper(Ns, Ms, fn, p0, bounds):
    """Profile-likelihood 95% upper limit on the first parameter (M*).

    Scans M* upward, re-optimising the other parameter, until the residual sum of squares exceeds the
    chi-square(1) 95% threshold relative to the best fit. Returns inf if it never does, which is the
    substantive answer: the data do not bound the ceiling.

    Args:
        Ns, Ms: data; fn: model; p0, bounds: as for curve_fit.
    Returns:
        (best_Mstar, upper_limit).
    """
    try:
        p, _ = curve_fit(fn, Ns, Ms, p0=p0, bounds=bounds, maxfev=60000)
    except Exception:
        return float("nan"), float("inf")
    rss0 = float(np.sum((Ms - fn(Ns, *p)) ** 2))
    n = len(Ns)
    thr = rss0 * np.exp(stats.chi2.ppf(0.95, 1) / n)     # Gaussian profile threshold
    best = p[0]
    for mult in np.logspace(0, 3, 200):
        M = best * mult
        try:
            q, _ = curve_fit(lambda x, N0: fn(x, M, N0), Ns, Ms, p0=[p[1]],
                             bounds=([bounds[0][1]], [bounds[1][1]]), maxfev=60000)
            rss = float(np.sum((Ms - fn(Ns, M, q[0])) ** 2))
        except Exception:
            continue
        if rss > thr:
            return best, M
    return best, float("inf")


def main():
    """Run the three tests, print them, and draw the supporting figure."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    by = load_traces(sweep)
    levels = ladder(by)
    fig, ax = plt.subplots(2, 3, figsize=(17, 9.5))
    cols = plt.cm.viridis(np.linspace(0.08, 0.78, len(levels)))

    for r, (crit, cname) in enumerate(CRITERIA):
        data = collect(by, levels, crit)
        print(f"\n{'='*78}\nCRITERION: {crit}  ({cname})\n{'='*78}")

        # ---- TEST 1: pooled curvature
        X0, y0, _ = design(data, quad=False)
        X1, y1, _ = design(data, quad=True)
        b0, rss0, df0 = ols(X0, y0)
        b1, rss1, df1 = ols(X1, y1)
        F = ((rss0 - rss1) / (df0 - df1)) / (rss1 / df1)
        p_curv = 1 - stats.f.cdf(F, df0 - df1, df1)
        se = np.sqrt(rss1 / df1 * np.linalg.pinv(X1.T @ X1)[-1, -1])
        print(f"\nTEST 1  pooled curvature (N >= {N_MIN}, n={len(y0)} seed-level points)")
        print(f"  H0 straight : k = {b0[-1]:.3f}          RSS={rss0:.5f}  dof={df0}  "
              f"AICc={aicc(rss0, len(y0), X0.shape[1]):.1f}")
        print(f"  H1 curved   : c = {b1[-1]:+.4f} +- {se:.4f}  RSS={rss1:.5f}  dof={df1}  "
              f"AICc={aicc(rss1, len(y1), X1.shape[1]):.1f}")
        print(f"  F({df0-df1},{df1}) = {F:.2f},  p = {p_curv:.3f}   "
              f"-> curvature {'SIGNIFICANT' if p_curv < 0.05 else 'not significant'} at 0.05")
        print(f"  (saturation predicts c < 0; observed c is "
              f"{'negative' if b1[-1] < 0 else 'POSITIVE'})")

        # ---- TESTS 2 and 3, per level
        print(f"\nTEST 2/3  per level (N >= {N_MIN})")
        print(f"  {'L*':>9} {'k (95% CI)':>22} {'dAICc sat-hyp':>14} {'dAICc sat-exp':>14} "
              f"{'M* hyper (95% upper)':>26}")
        for i, Lstar in enumerate(levels):
            Ns, Ms = [], []
            for N, vals in data[Lstar].items():
                if N >= N_MIN:
                    Ns += [N] * len(vals)
                    Ms += vals
            Ns, Ms = np.array(Ns, float), np.array(Ms, float)
            lk = np.polyfit(np.log(Ns), np.log(Ms), 1)[0]
            boot = []
            for _ in range(N_BOOT):
                idx = np.random.randint(0, len(Ns), len(Ns))
                if len(np.unique(Ns[idx])) < 2:
                    continue
                boot.append(np.polyfit(np.log(Ns[idx]), np.log(Ms[idx]), 1)[0])
            lo, hi = np.percentile(boot, [2.5, 97.5])

            fits = {}
            for nm, fn, p0, bd in (
                    ("pow", powerlaw, [10.0, 0.5], ([1e-6, -2], [1e6, 3])),
                    ("hyp", sat_hyper, [max(Ms) * 2, 1000.0], ([1, 1], [1e7, 1e8])),
                    ("exp", sat_exp, [max(Ms) * 1.5, 1000.0], ([1, 1], [1e7, 1e8]))):
                try:
                    p, _ = curve_fit(fn, Ns, Ms, p0=p0, bounds=bd, maxfev=60000)
                    rss = float(np.sum((Ms - fn(Ns, *p)) ** 2))
                    fits[nm] = (aicc(rss, len(Ns), 2), p)
                except Exception:
                    fits[nm] = (float("inf"), None)
            best_ms, up = profile_upper(Ns, Ms, sat_hyper, [max(Ms) * 2, 1000.0],
                                        ([1, 1], [1e7, 1e8]))
            upstr = "inf (unbounded)" if not np.isfinite(up) else f"{up:.0f}"
            print(f"  {Lstar:9.5f} {f'{lk:.3f} [{lo:.3f},{hi:.3f}]':>22} "
                  f"{fits['hyp'][0]-fits['pow'][0]:>14.1f} {fits['exp'][0]-fits['pow'][0]:>14.1f} "
                  f"{f'{best_ms:.0f}  (<{upstr})':>26}")

            # ---- figure
            xs = np.logspace(np.log10(400), np.log10(6000), 100)
            allN = sorted(data[Lstar])
            mu = [np.mean(data[Lstar][N]) for N in allN]
            sd = [np.std(data[Lstar][N]) for N in allN]
            ax[r, 0].errorbar(allN, mu, yerr=sd, fmt="o", color=cols[i], ms=6, capsize=3,
                              label=f"$L^*$={Lstar:.5f}")
            ax[r, 0].plot(xs, powerlaw(xs, *fits["pow"][1]), "-", color=cols[i], lw=1.6)
            if fits["hyp"][1] is not None:
                ax[r, 0].plot(xs, sat_hyper(xs, *fits["hyp"][1]), ":", color=cols[i], lw=1.6)
            ax[r, 1].errorbar([Lstar], [lk], yerr=[[lk - lo], [hi - lk]], fmt="o", color=cols[i],
                              ms=8, capsize=4)

        ax[r, 0].axvline(N_MIN, color="grey", ls="-", lw=1, alpha=.5)
        ax[r, 0].text(N_MIN * 1.05, 105, "fits use $N\\geq$500\n(ceiling-free)", fontsize=7)
        ax[r, 0].set(xscale="log", yscale="log", xlabel="$N$", ylabel="active units $M$",
                     title=f"(a) solid = power law, dotted = saturating\n{cname}")
        ax[r, 0].legend(fontsize=7)
        ax[r, 1].set(xlabel="matching level $L^*$", ylabel="$k$ (bootstrap 95% CI)",
                     title=f"(b) identification of $k$\n{cname}")
        ax[r, 1].invert_xaxis()

        # curvature panel: local exponent per consecutive ceiling-free pair
        for i, Lstar in enumerate(levels):
            Ns = sorted(n for n in data[Lstar] if n >= N_MIN)
            mu = [np.mean(data[Lstar][n]) for n in Ns]
            ks = [np.log(mu[j + 1] / mu[j]) / np.log(Ns[j + 1] / Ns[j]) for j in range(len(Ns) - 1)]
            mids = [np.sqrt(Ns[j] * Ns[j + 1]) for j in range(len(Ns) - 1)]
            ax[r, 2].plot(mids, ks, "-o", color=cols[i], ms=6, label=f"$L^*$={Lstar:.5f}")
        ax[r, 2].axhline(0, color="r", ls="--", lw=1, alpha=.6)
        ax[r, 2].set(xscale="log", ylim=(-0.05, 0.8), xlabel="$N$", ylabel="local $k$",
                     title=f"(c) does $k$ fall toward 0?\n{cname}")
        ax[r, 2].legend(fontsize=7)
        for c in range(3):
            ax[r, c].grid(alpha=.25)

    fig.suptitle("Saturation vs power law for M(N), ceiling-free sizes only", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    out = os.path.join(IMG_DIR, "saturation_test.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
