#!/usr/bin/env python3
"""
Heavy-tailedness of the unit distribution, measured five independent ways at a FIXED dimension.

⚠️ THIS REPLACES THE MARDIA COLUMN IN `flipflop_structure_matrix.py`, WHICH IS CONFOUNDED. There,
Mardia kurtosis was computed on each net's own 99%-variance embedding, whose dimension m ranges from
14 to 88 across conditions. Mardia's Gaussian expectation is m(m+2), so the statistic scales with m,
and m correlates with the raw kurtosis at r = +0.755 over the 336 nets. Dividing by m(m+2) REVERSES
the ordering (none 0.554 -> both 0.942), and with n = 132 against m = 88 the covariance is nearly
singular, so the Mahalanobis distances underneath are unreliable exactly where m is largest.
Everything here is therefore computed at M_FIX dimensions for every condition, giving n/m ~ 13.

Five measures, deliberately not variants of one idea - a claim that survives all five is real,
one that survives only the 4th-moment ones is an outlier artefact:

  kurt_mardia  Mardia multivariate kurtosis excess. 4th moment: maximal outlier sensitivity.
  ks_chi2      KS distance between the squared Mahalanobis distances and chi2_M, their exact
               Gaussian distribution. Whole-shape, no moment assumption.
  tail99       fraction of points beyond the chi2_M 99th percentile; 0.01 under a Gaussian.
               Counts outliers rather than weighting them.
  hill         Hill tail index on the upper decile of Mahalanobis distance. SMALLER = heavier tail.
               A genuine extreme-value estimator, not a moment.
  uni_kurt     mean univariate excess kurtosis over the M_FIX PCs. Marginal rather than joint.

Output: img/internal_figures/heavytail_matrix.png
        data/heavytail_cache.npz

Usage:  python flipflop_heavytail.py [--recompute]
"""

import os
import sys
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from pr_matrix import PENS
from flipflop_dimensionality import run_folders, grid
from flipflop_fixedpoints import load_net
from flipflop_bouquet import run_trials

N_TRIALS = 32
N_KEEP = 132           # global minimum live count over the grid
M_FIX = 10             # fixed embedding dimension: n/m ~ 13, well-conditioned everywhere
CACHE = "data/heavytail_cache.npz"
KEYS = ("kurt_mardia", "ks_chi2", "tail99", "hill", "uni_kurt")
FIELDS = ("pen", "k", "N") + KEYS


def mahal2(Y):
    """Squared Mahalanobis distances of points from their own mean.

    Args:
        Y: (n, m) points.
    Returns:
        (n,) squared distances; distributed as chi2_m if Y is Gaussian.
    """
    Z = Y - Y.mean(0)
    Si = np.linalg.pinv(np.atleast_2d(np.cov(Z, rowvar=False)))
    return np.einsum("ij,jk,ik->i", Z, Si, Z)


def measure(rates):
    """Five heavy-tail statistics for one network, all at M_FIX dimensions.

    Args:
        rates: (N, T, B) noise-free firing rates.
    Returns:
        dict of scalars, all nan if the net has fewer than N_KEEP live units.
    """
    X = rates.reshape(rates.shape[0], -1).astype(np.float64)
    live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
    if live.sum() < N_KEEP:
        return {kk: np.nan for kk in KEYS}
    rng = np.random.default_rng(0)
    Xl = X[live][rng.choice(int(live.sum()), N_KEEP, replace=False)]
    Xl = Xl / np.maximum(np.linalg.norm(Xl, axis=1, keepdims=True), 1e-300)   # shape only
    Z = Xl - Xl.mean(0, keepdims=True)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    Y = U[:, :M_FIX] * s[:M_FIX]

    d2 = mahal2(Y)
    n, m = Y.shape
    Zc = Y - Y.mean(0)
    Si = np.linalg.pinv(np.atleast_2d(np.cov(Zc, rowvar=False)))
    Mx = Zc @ Si @ Zc.T
    kurt = float((np.diag(Mx) ** 2).mean() - m * (m + 2))
    ks = float(stats.kstest(d2, "chi2", args=(m,)).statistic)
    tail = float((d2 > stats.chi2.ppf(0.99, m)).mean())
    r = np.sort(np.sqrt(np.maximum(d2, 0)))[::-1]
    kk = max(5, int(0.1 * n))
    hill = float(kk / np.log(r[:kk] / r[kk]).sum()) if r[kk] > 0 else np.nan
    uni = float(np.mean([stats.kurtosis(Y[:, j], fisher=True) for j in range(m)]))
    return dict(kurt_mardia=kurt, ks_chi2=ks, tail99=tail, hill=hill, uni_kurt=uni)


def compute():
    """Run the grid and cache the heavy-tail statistics.

    Returns:
        dict of arrays keyed by FIELDS.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if all(f in z.files for f in FIELDS):
            return {kk: z[kk] for kk in z.files}
    fold = run_folders()
    print(f"simulating {len(fold)} nets (fixed m={M_FIX}, n={N_KEEP})")
    rec = {kk: [] for kk in FIELDS}
    for i, (folder, pen, k, N) in enumerate(fold, 1):
        m = measure(run_trials(load_net(folder)[0], folder, N_TRIALS))
        for kk, v in zip(("pen", "k", "N"), (pen, k, N)):
            rec[kk].append(v)
        for kk in KEYS:
            rec[kk].append(m[kk])
        if i % 50 == 0 or i == len(fold):
            print(f"  {i}/{len(fold)}")
    rec = {kk: np.array(v) for kk, v in rec.items()}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **rec)
    return rec


def main():
    """Tabulate and plot the five heavy-tail measures over the (N, k) grid."""
    if "--recompute" in sys.argv and os.path.exists(CACHE):
        os.remove(CACHE)
    ps.setup()
    rec = compute()
    ks_ = sorted(set(rec["k"].tolist()))
    Ns = sorted(set(rec["N"].tolist()))

    LAB = {"kurt_mardia": ("Mardia kurtosis excess", "0 = Gaussian; higher = heavier", 1),
           "ks_chi2": ("KS vs chi2", "0 = Gaussian; higher = heavier", 1),
           "tail99": ("frac beyond chi2 99%", "0.01 = Gaussian; higher = heavier", 1),
           "hill": ("Hill tail index", "SMALLER = heavier tail", -1),
           "uni_kurt": ("mean univariate kurtosis", "0 = Gaussian; higher = heavier", 1)}
    print(f"\ngrid means at fixed m={M_FIX}, n={N_KEEP} "
          f"(arrow = direction of HEAVIER tails)\n")
    print(f"{'pen':<6}" + "".join(f"{LAB[k][0][:15]:>17}" for k in KEYS))
    means = {}
    for p in PENS:
        row = f"{p:<6}"
        for kk in KEYS:
            m = (rec["pen"] == p) & np.isfinite(rec[kk])
            v = float(np.mean(rec[kk][m])) if m.any() else np.nan
            means[(p, kk)] = v
            row += f"{v:>17.3f}"
        print(row)
    print(f"\n{'measure':<24}{'heaviest -> lightest':<44}consistent with `both` heaviest?")
    for kk in KEYS:
        sgn = LAB[kk][2]
        order = sorted(PENS, key=lambda p: -sgn * means[(p, kk)])
        print(f"{LAB[kk][0]:<24}{' > '.join(order):<44}"
              f"{'YES' if order[0] == 'both' else 'no (' + order[0] + ' heaviest)'}")

    fig, ax = plt.subplots(2, 3, figsize=(16, 8.4))
    for a, kk in zip(ax.ravel(), KEYS):
        for p in PENS:
            Z, S, _ = grid(rec, p, kk, ks_, Ns)
            mu = np.nanmean(Z, axis=0)
            sd = np.nanstd(Z, axis=0)
            ps.band(a, ks_, mu, sd, {"none": "#7f7f7f", "rws": "#2ca02c",
                                     "frm": "#d62728", "both": "#1f77b4"}[p], label=p)
        a.set(xlabel="k (bits)", ylabel=LAB[kk][0], xticks=ks_,
              title=f"{LAB[kk][0]}\n{LAB[kk][1]}")
        a.grid(alpha=.25); a.legend(fontsize=7)
    ax.ravel()[-1].axis("off")
    ax.ravel()[-1].text(.05, .5, "Five independent tail measures.\n\nA claim that survives all "
                        "five is real;\none that survives only the 4th-moment\nones is an outlier "
                        f"artefact.\n\nFixed m = {M_FIX} for every condition,\nso nothing here "
                        "scales with embedding\ndimension — the defect that invalidated\nthe "
                        "Mardia column in structure_matrix.", fontsize=9, va="center")
    fig.suptitle(f"Heavy-tailedness of the unit distribution, five ways — fixed m={M_FIX}, "
                 f"n={N_KEEP} units per net\naveraged over N; shaded = spread across sizes",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    return ps.save(fig, "heavytail_matrix", tight=False)


if __name__ == "__main__":
    main()
