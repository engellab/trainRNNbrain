#!/usr/bin/env python3
"""
Is there geometric STRUCTURE in unit space beyond the covariance? Manifolds, not just clusters.

⚠️ WHY NEITHER EARLIER TEST ANSWERS THIS. `flipflop_clustering.py` uses a permutation null that
destroys all between-unit covariance, so its silhouette gap only shows the data is anisotropic - a
plain Gaussian blob scores the same. `flipflop_epairs.py` fixes the null but its statistic saturates
(every ratio < 0.25, so ePAIRS is pinned near +1 in all conditions) and it only sees ANGLES, so it
is blind to curved or filamentary structure that keeps the angular distribution unchanged.

The test here is INTRINSIC DIMENSION against a covariance-matched Gaussian. A Gaussian cloud in m
dimensions has intrinsic dimension ~ m; points on a d-dimensional manifold have ID ~ d << m however
the cloud is stretched. The null therefore controls anisotropy, sample size AND embedding dimension
at once, and any ID gap is geometric structure that the covariance does not explain.

  ID_2NN   Facco et al. (2017) two-nearest-neighbour maximum-likelihood estimator. For each point
           mu = r2/r1; the ML estimate is d = n / sum(log mu), fitted on the lower 90% of mu to
           discard the tail where the locally-uniform assumption fails.
  ID gap   ID_null - ID_real, in units of the null's own ID. > 0 means the real cloud is
           CONCENTRATED ON SOMETHING LOWER-DIMENSIONAL than its own covariance implies.
  KS       Kolmogorov-Smirnov distance between the real and null nearest-neighbour distance
           distributions: an omnibus check that does not assume the structure is a manifold.

⚠️ FINITE SAMPLES BIAS ID DOWNWARD, WHICH IS EXACTLY WHY THE NULL IS NOT OPTIONAL. With ~300 points
in 30-80 dimensions the estimator returns far less than m even for a perfect Gaussian, so a raw ID
is uninterpretable. Only the gap means anything.

Both variants are run because they ask different questions:
  raw    structure in responses including amplitude
  norm   each unit scaled to unit length first: structure in response SHAPE alone

Output: img/internal_figures/manifold_N{N}_k{k}.png

Usage:  python flipflop_manifold.py [N] [k]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from flipflop_heterogeneity import folders_for
from flipflop_diversity import rates_and_targets

PENS = ["none", "rws", "frm", "both"]
COLS = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
N_TRIALS = 48
VAR_KEEP = 0.99
N_NULL = 60
SEED = 0


def id_2nn(X, discard=0.0):
    """Facco two-NN maximum-likelihood intrinsic dimension.

    PRINCIPLE. For points sampled from a locally uniform density on a d-dimensional manifold, the
    volume enclosed by the first neighbour and the volume of the shell out to the second are
    INDEPENDENT Exp(1) variables (Poisson points arrive at unit rate in "volume time"). With
    mu = r2/r1 that gives mu^d = 1 + v2/v1, and since v2/v1 has survival 1/(1+x),

        P(mu > m) = m^(-d)          i.e.  log mu ~ Exp(d)

    The density cancels, which is why no assumption about how the points are scattered is needed.
    The maximum-likelihood estimate is d = n / sum(log mu), equivalently 1 / mean(log mu).

    ⚠️ `discard` DEFAULTS TO 0 AND SHOULD NORMALLY STAY THERE. That MLE is derived for the FULL
    sample. Trimming the upper tail removes the largest log mu terms, shrinks the denominator and
    inflates d by ~30%: calibrated against known dimensions at n=275, trimming 10% gave 1.29 / 2.60
    / 4.04 / 6.12 for true d = 1 / 2 / 3 / 5, against 0.95 / 1.96 / 3.01 / 4.60 untrimmed. Trimming
    belongs with the CDF-slope estimator, not with this one; mixing them was a real error in an
    earlier version of this file and it inflated every reported ID.

    Args:
        X: (n, d) points; discard: upper fraction of the mu distribution to drop; leave at 0 unless
            deliberately pairing it with a CDF fit.
    Returns:
        float ID estimate, or nan if degenerate.
    """
    n = X.shape[0]
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    np.fill_diagonal(D, np.inf)
    r = np.sort(D, axis=1)[:, :2]
    ok = (r[:, 0] > 0) & np.isfinite(r[:, 1])
    mu = r[ok, 1] / r[ok, 0]
    mu = np.sort(mu[mu > 1])
    if mu.size < 10:
        return float("nan")
    if discard > 0:
        mu = mu[:max(10, int(mu.size * (1 - discard)))]
    return float(mu.size / np.log(mu).sum())


def nn_dists(X):
    """Nearest-neighbour distance for each point.

    Args:
        X: (n, d) points.
    Returns:
        (n,) distances to the closest other point.
    """
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0))
    np.fill_diagonal(D, np.inf)
    return D.min(1)


def ks(a, b):
    """Two-sample Kolmogorov-Smirnov distance.

    Args:
        a, b: 1-D samples.
    Returns:
        float sup |F_a - F_b|.
    """
    v = np.sort(np.concatenate([a, b]))
    fa = np.searchsorted(np.sort(a), v, side="right") / a.size
    fb = np.searchsorted(np.sort(b), v, side="right") / b.size
    return float(np.abs(fa - fb).max())


def embed(X, n_keep, rng, normalise):
    """Subsample live units and project onto the PCs carrying VAR_KEEP of across-unit variance.

    Args:
        X: (n_live, S) live-unit rates; n_keep: common subsample size; rng: Generator;
        normalise: scale each unit to unit length first.
    Returns:
        (Y, m): (n_keep, m) embedding and its dimension.
    """
    idx = rng.choice(X.shape[0], size=min(n_keep, X.shape[0]), replace=False)
    Z = X[idx].astype(np.float64)
    if normalise:
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)
    Z = Z - Z.mean(0, keepdims=True)
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    m = int(np.searchsorted(np.cumsum(s ** 2) / np.sum(s ** 2), VAR_KEEP) + 1)
    return U[:, :m] * s[:m], m


def analyse(Y, seed=SEED, n_null=N_NULL):
    """Intrinsic-dimension gap and NN-distance KS against a covariance-matched Gaussian.

    Args:
        Y: (n, m) embedding; seed: RNG seed; n_null: Gaussian draws.
    Returns:
        dict with id_real, id_null, id_gap (relative), ks_stat and ks_p.
    """
    rng = np.random.default_rng(seed)
    cov = np.atleast_2d(np.cov(Y, rowvar=False))
    mu = Y.mean(0)
    idr, dr = id_2nn(Y), nn_dists(Y)
    ids, kss = [], []
    for _ in range(n_null):
        G = rng.multivariate_normal(mu, cov, size=Y.shape[0])
        ids.append(id_2nn(G))
        kss.append(ks(dr, nn_dists(G)))
    ids = np.array([v for v in ids if np.isfinite(v)])
    # KS of real-vs-null, calibrated against null-vs-null of the same size
    null_ks = []
    for _ in range(n_null):
        a = rng.multivariate_normal(mu, cov, size=Y.shape[0])
        b = rng.multivariate_normal(mu, cov, size=Y.shape[0])
        null_ks.append(ks(nn_dists(a), nn_dists(b)))
    k_real, null_ks = float(np.mean(kss)), np.array(null_ks)
    return dict(id_real=idr, id_null=float(ids.mean()),
                id_gap=float((ids.mean() - idr) / max(ids.mean(), 1e-12)),
                ks_stat=k_real, ks_p=float((null_ks >= k_real).mean()))


def main():
    """Compare geometric structure across penalties, raw and unit-normalised."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ps.setup()
    live_sets = {}
    for pen in PENS:
        for folder in folders_for(pen, N, k):
            r, _ = rates_and_targets(folder, N_TRIALS)
            X = r.reshape(r.shape[0], -1)
            live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
            live_sets.setdefault(pen, []).append(X[live])
    n_keep = min(x.shape[0] for v in live_sets.values() for x in v)

    res = {}
    for normalise in (False, True):
        tag = "norm" if normalise else "raw"
        print(f"\n{'='*94}\nGEOMETRIC STRUCTURE, {tag} — N={N}, k={k}, "
              f"{n_keep} units per net, {VAR_KEEP:.0%} variance, "
              f"covariance-matched Gaussian null\n{'='*94}")
        print(f"{'pen':<6}{'m':>5}{'ID real':>9}{'ID null':>9}{'ID gap':>9}"
              f"{'KS':>8}{'KS p':>8}   reading")
        for pen in PENS:
            rows = []
            for X in live_sets.get(pen, []):
                Y, m = embed(X, n_keep, np.random.default_rng(SEED), normalise)
                a = analyse(Y)
                a["m"] = m
                rows.append(a)
            if not rows:
                continue
            res[(pen, tag)] = rows
            f = lambda kk: float(np.mean([r[kk] for r in rows]))
            rd = ("STRUCTURE beyond the covariance" if f("id_gap") > 0.15 and f("ks_p") < 0.05
                  else "weak" if f("id_gap") > 0.05 else "Gaussian-like")
            print(f"{pen:<6}{f('m'):>5.0f}{f('id_real'):>9.2f}{f('id_null'):>9.2f}"
                  f"{f('id_gap'):>+9.3f}{f('ks_stat'):>8.3f}{f('ks_p'):>8.3f}   {rd}")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for a, key, lab, note in [
            (ax[0], "id_gap", "intrinsic-dimension gap",
             "(ID_null - ID_real)/ID_null\n>0 = concentrated below its own covariance"),
            (ax[1], "id_real", "intrinsic dimension ID",
             "directly comparable: same n in every condition"),
            (ax[2], "ks_stat", "KS, NN-distance distribution",
             "real vs covariance-matched Gaussian")]:
        w = 0.38
        for j, tag in enumerate(("raw", "norm")):
            xs = [p for p in PENS if (p, tag) in res]
            mu = [np.mean([r[key] for r in res[(p, tag)]]) for p in xs]
            sd = [np.std([r[key] for r in res[(p, tag)]]) for p in xs]
            a.bar(np.arange(len(xs)) + (j - .5) * w, mu, w, yerr=sd, capsize=3,
                  color=[COLS[p] for p in xs], alpha=.9 if tag == "raw" else .45,
                  edgecolor="k" if tag == "norm" else "none", linewidth=.7,
                  label="raw" if j == 0 else "unit-normalised")
            a.set_xticks(range(len(xs))); a.set_xticklabels(xs)
        a.axhline(0, ls="--", c="0.4", lw=1)
        a.set(ylabel=lab, title=f"{lab}\n{note}")
        a.grid(alpha=.25, axis="y"); a.legend(fontsize=8)
    fig.suptitle(f"Geometric structure in unit space — N={N}, k={k}\n"
                 "intrinsic dimension vs a covariance-matched Gaussian: detects manifolds, not "
                 "just clusters", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return ps.save(fig, f"manifold_N{N}_k{k}", tight=False)


if __name__ == "__main__":
    main()
