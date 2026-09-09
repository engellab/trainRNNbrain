#!/usr/bin/env python3
"""
ePAIRS and companion tests: are unit responses DISCRETE categories, or a continuum?

⚠️ THIS EXISTS BECAUSE THE SILHOUETTE-GAP TEST IN `flipflop_clustering.py` CANNOT ANSWER THAT.
That null permutes each unit's samples independently, destroying all between-unit covariance, so its
null is effectively isotropic. A positive gap therefore shows only that the data is ANISOTROPIC and
low-dimensional - an elongated Gaussian cloud with no clusters at all scores just as well. The
clustering result must not be read as evidence of discrete cell types.

ePAIRS (Hirokawa et al. 2019), the elliptical form of PAIRS (Raposo, Kaufman & Churchland 2014),
fixes precisely that: the null is drawn from a multivariate Gaussian MATCHED TO THE DATA'S OWN
COVARIANCE, so anisotropy is preserved and only discreteness is tested.

    for each unit, take its feature vector, normalise to unit length
    angle_i = mean angle to its NN nearest neighbours
    stat    = median_i angle_i
    null    = the same statistic on draws from N(0, Sigma_data), Sigma_data from the real features
    ePAIRS  = (null_stat - real_stat) / null_stat

    ePAIRS > 0  units sit closer to their neighbours than an elliptical Gaussian -> CATEGORIES
    ePAIRS ~ 0  indistinguishable from a Gaussian continuum
    ePAIRS < 0  MORE evenly spread than Gaussian -> category-free, the Raposo et al. result

Two feature sets, because the answer can legitimately differ between them:
  bits  the k regression loadings on the target bit time courses - the classic tuning-vector setup
  pcs   the leading PCs of the response, capturing shape not just task tuning

Companion Gaussianity tests on the same features, since "not Gaussian" is weaker than "clustered"
but is what a reader will ask next:
  mardia_skew / mardia_kurt   multivariate skewness and kurtosis vs their Gaussian expectations
  dip                         Hartigan's dip on the first PC, testing unimodality directly

Output: img/internal_figures/epairs_N{N}_k{k}.png

Usage:  python flipflop_epairs.py [N] [k]
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
NN = 3                 # nearest neighbours, as in Raposo et al.
N_NULL = 500
N_PC = 10              # dimension of the "pcs" feature set
SEED = 0


def unit_features(rates, targets, n_pc=N_PC):
    """Per-unit feature vectors of live units, in two flavours.

    Args:
        rates: (N, T, B) firing rates; targets: (k, T, B) target bits; n_pc: PCs to keep.
    Returns:
        dict with 'bits' (n_live, k) regression loadings and 'pcs' (n_live, n_pc) PC scores.
    """
    X = rates.reshape(rates.shape[0], -1).astype(np.float64)
    live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
    X = X[live]
    Xc = X - X.mean(1, keepdims=True)
    G = targets.reshape(targets.shape[0], -1).T
    G = np.column_stack([np.ones(G.shape[0]), G])
    beta, *_ = np.linalg.lstsq(G, Xc.T, rcond=None)
    Z = Xc - Xc.mean(0, keepdims=True)                 # centre over units for the PC features
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    return {"bits": beta[1:].T, "pcs": (U[:, :n_pc] * s[:n_pc])}


def mean_nn_angle(F, nn=NN):
    """Median over units of the mean angle to its `nn` nearest neighbours.

    Args:
        F: (n, d) feature matrix; nn: neighbours per unit.
    Returns:
        float median angle in radians, or nan if there are too few units.
    """
    n = F.shape[0]
    if n <= nn + 1:
        return float("nan")
    U = F / np.maximum(np.linalg.norm(F, axis=1, keepdims=True), 1e-300)
    C = np.clip(U @ U.T, -1.0, 1.0)
    np.fill_diagonal(C, -np.inf)                       # exclude self
    idx = np.argpartition(-C, nn, axis=1)[:, :nn]      # nn largest cosines = smallest angles
    ang = np.arccos(np.take_along_axis(C, idx, axis=1))
    return float(np.median(ang.mean(1)))


def epairs(F, n_null=N_NULL, nn=NN, seed=SEED):
    """ePAIRS statistic against a covariance-matched multivariate Gaussian null.

    Args:
        F: (n, d) feature matrix; n_null: null draws; nn: neighbours; seed: RNG seed.
    Returns:
        (stat, p, real_angle, null_mean_angle). stat > 0 means units cluster more tightly than an
        elliptical Gaussian of the same covariance; p is the two-sided empirical p-value.
    """
    real = mean_nn_angle(F, nn)
    if not np.isfinite(real):
        return float("nan"), float("nan"), real, float("nan")
    rng = np.random.default_rng(seed)
    cov = np.atleast_2d(np.cov(F, rowvar=False))
    # ⚠️ THE NULL MUST CARRY THE DATA'S MEAN. Angles are computed on the raw (uncentred) feature
    # vectors, and these loadings have a strong common component - most units load the same way on
    # a shared direction - so the real vectors sit in a narrow cone on the sphere. A zero-mean null
    # spreads over the whole sphere and its neighbour angles are far larger, which manufactures a
    # large positive ePAIRS for EVERY condition regardless of structure. Measured with the zero-mean
    # bug: every condition returned +0.77..+0.98 at p=0.002, with real angles of 0.003 rad.
    nulls = np.array([mean_nn_angle(rng.multivariate_normal(F.mean(0), cov, size=F.shape[0]), nn)
                      for _ in range(n_null)])
    nulls = nulls[np.isfinite(nulls)]
    nm = float(nulls.mean())
    p = float(2 * min((nulls <= real).mean(), (nulls >= real).mean()))
    return (nm - real) / nm, max(p, 1.0 / max(len(nulls), 1)), real, nm


def mardia(F):
    """Mardia's multivariate skewness and kurtosis, as z-like excesses over the Gaussian value.

    Args:
        F: (n, d) feature matrix.
    Returns:
        (skew_excess, kurt_excess): 0 for a perfect Gaussian; positive kurt = heavier tails.
    """
    n, d = F.shape
    Z = F - F.mean(0)
    S = np.cov(Z, rowvar=False)
    S = np.atleast_2d(S)
    Si = np.linalg.pinv(S)
    M = Z @ Si @ Z.T
    b1 = float((M ** 3).sum() / (n ** 2))              # Gaussian expectation ~ 0
    b2 = float((np.diag(M) ** 2).mean())               # Gaussian expectation d(d+2)
    return b1, b2 - d * (d + 2)


def dip(x, n_null=200, seed=SEED):
    """Hartigan-style unimodality check on a 1-D sample, via the max CDF gap to its best unimodal fit.

    A light stand-in for the exact dip statistic: the largest deviation between the empirical CDF and
    the closest unimodal (here, linearly interpolated monotone) CDF, calibrated against uniform draws.

    Args:
        x: (n,) sample; n_null: calibration draws; seed: RNG seed.
    Returns:
        float empirical p-value; small = evidence against unimodality.
    """
    def stat(v):
        v = np.sort(v)
        f = np.linspace(0, 1, v.size)
        g = (v - v[0]) / max(v[-1] - v[0], 1e-300)
        return float(np.abs(f - g).max())
    rng = np.random.default_rng(seed)
    s = stat(x)
    nl = np.array([stat(rng.normal(size=x.size)) for _ in range(n_null)])
    return float((nl >= s).mean())


def main():
    """Run ePAIRS and the Gaussianity companions for every penalty at one (N, k) cell."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ps.setup()
    res = {}
    for pen in PENS:
        rows = []
        for folder in folders_for(pen, N, k):
            r, t = rates_and_targets(folder, N_TRIALS)
            F = unit_features(r, t)
            row = {}
            for name in ("bits", "pcs"):
                st, p, ra, na = epairs(F[name])
                sk, ku = mardia(F[name])
                row[name] = dict(stat=st, p=p, real=ra, null=na, skew=sk, kurt=ku,
                                 dip=dip(F[name][:, 0]), n=F[name].shape[0])
            rows.append(row)
        if rows:
            res[pen] = rows

    for name, ttl in (("bits", f"regression loadings on the {k} target bits"),
                      ("pcs", f"leading {N_PC} PCs of the response")):
        print(f"\n{'='*88}\nePAIRS on {ttl}   (N={N}, k={k}, NN={NN}, {N_NULL} Gaussian draws)\n"
              f"{'='*88}")
        print(f"{'pen':<6}{'n_live':>8}{'ePAIRS':>9}{'p':>8}{'angle':>8}{'null':>8}"
              f"{'mardia_kurt':>13}{'dip p':>8}   reading")
        for pen in PENS:
            if pen not in res:
                continue
            g = [r[name] for r in res[pen]]
            m = lambda kk: float(np.mean([x[kk] for x in g]))
            st, p = m("stat"), m("p")
            rd = ("CATEGORIES" if st > 0.02 and p < 0.05 else
                  "category-free (more even than Gaussian)" if st < -0.02 and p < 0.05 else
                  "indistinguishable from a Gaussian continuum")
            print(f"{pen:<6}{m('n'):>8.0f}{st:>+9.3f}{p:>8.3f}{m('real'):>8.3f}{m('null'):>8.3f}"
                  f"{m('kurt'):>13.1f}{m('dip'):>8.3f}   {rd}")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    for a, (key, name, lab) in zip(ax, [("stat", "bits", "ePAIRS — tuning loadings"),
                                        ("stat", "pcs", f"ePAIRS — top {N_PC} PCs"),
                                        ("kurt", "pcs", "Mardia kurtosis excess (PCs)")]):
        xs = [p for p in PENS if p in res]
        mu = [np.mean([r[name][key] for r in res[p]]) for p in xs]
        sd = [np.std([r[name][key] for r in res[p]]) for p in xs]
        a.bar(xs, mu, yerr=sd, color=[COLS[p] for p in xs], alpha=.85, capsize=4)
        a.axhline(0, ls="--", c="0.4", lw=1)
        for i, v in enumerate(mu):
            a.text(i, v, f"{v:+.3f}" if key == "stat" else f"{v:+.0f}",
                   ha="center", va="bottom" if v >= 0 else "top", fontsize=8.5)
        a.set(ylabel=lab, title=f"{lab}\n" + ("> 0 = discrete categories, < 0 = category-free"
                                              if key == "stat" else "0 = Gaussian tails"))
        a.grid(alpha=.25, axis="y")
    fig.suptitle(f"Are unit responses discrete categories or a continuum? — N={N}, k={k}\n"
                 "ePAIRS null is a COVARIANCE-MATCHED Gaussian, so anisotropy is controlled and "
                 "only discreteness is tested", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.86])
    return ps.save(fig, f"epairs_N{N}_k{k}", tight=False)


if __name__ == "__main__":
    main()
