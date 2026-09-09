#!/usr/bin/env python3
"""
Do the penalties produce more tightly CLUSTERED unit responses? N=2000, k=3.

Each unit is a point in sample space: the (N, T, B) rate tensor is flattened to (N, T*B), centred
over units, and projected onto the leading PCs that carry VAR_KEEP of the across-unit variance,
giving (n, m). Units are then k-means clustered for 2..30 clusters and scored.

⚠️ TWO CONFOUNDS MAKE THE NAIVE COMPARISON MEANINGLESS, AND BOTH ARE CONTROLLED HERE.

1. UNIT COUNT AND EMBEDDING DIMENSION DIFFER ENORMOUSLY BETWEEN CONDITIONS. Measured at N=2000,
   k=3: none 285 live units and m=13, rws 298/16, frm 1802/54, both 2000/34. Silhouette rises with
   dimension and varies with sample size, so a raw comparison would mostly report those two numbers.
   Every condition is therefore subsampled to a COMMON number of live units before the PCA.

2. SILHOUETTE HAS NO MEANINGFUL ZERO. On structureless data it is not 0 but a positive value set by
   n and m. Every curve is therefore scored against a NULL built from the same net: each unit's
   sample vector is permuted independently, destroying between-unit structure while preserving each
   unit's own marginal distribution, then pushed through the identical pipeline. The reported effect
   is the GAP, real minus null. Only the gap supports "more clustered".

⚠️ AMPLITUDE IS A THIRD CONFOUND, so both variants are provided. Unnormalised, k-means largely
separates loud units from quiet ones, and the conditions differ hugely in amplitude spread (that is
exactly what frm equalises). `--norm` scales every unit to unit length first, clustering on response
SHAPE alone. A claim that holds in only one variant is a claim about amplitude, not about clustering.

Output: img/internal_figures/clustering_N{N}_k{k}[_norm].png

Usage:  python flipflop_clustering.py [N] [k] [--norm]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from flipflop_heterogeneity import folders_for
from flipflop_diversity import rates_and_targets

PENS = ["none", "rws", "frm", "both"]
COLS = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
N_TRIALS = 48
VAR_KEEP = 0.99
KRANGE = list(range(2, 31))
SEED = 0


def embed(X, n_keep, rng, normalise):
    """Subsample live units, centre over units, and project onto the leading PCs.

    Args:
        X: (n_live, S) rate matrix of live units only.
        n_keep: number of units to subsample to, for cross-condition comparability.
        rng: numpy Generator for the subsample.
        normalise: scale each unit to unit length first, clustering on shape rather than amplitude.
    Returns:
        (Y, m): (n_keep, m) embedding and the number of components retained.
    """
    idx = rng.choice(X.shape[0], size=min(n_keep, X.shape[0]), replace=False)
    Z = X[idx].astype(np.float64)
    if normalise:
        Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)
    Z = Z - Z.mean(0, keepdims=True)                    # centre over UNITS: PCA of the unit ensemble
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    var = np.cumsum(s ** 2) / np.sum(s ** 2)
    m = int(np.searchsorted(var, VAR_KEEP) + 1)
    return U[:, :m] * s[:m], m


def score_curve(Y):
    """Cluster the embedding for every k in KRANGE and score it.

    Args:
        Y: (n, m) embedding.
    Returns:
        dict of arrays keyed sil / ch / db / inertia, aligned to KRANGE.
    """
    out = {kk: [] for kk in ("sil", "ch", "db", "inertia")}
    for kc in KRANGE:
        km = KMeans(n_clusters=kc, n_init=10, random_state=SEED).fit(Y)
        lab = km.labels_
        out["sil"].append(float(silhouette_score(Y, lab)))
        out["ch"].append(float(calinski_harabasz_score(Y, lab)))
        out["db"].append(float(davies_bouldin_score(Y, lab)))
        out["inertia"].append(float(km.inertia_))
    return {kk: np.array(v) for kk, v in out.items()}


def main():
    """Compare clustering quality across penalties, against a per-net permutation null."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    N = int(args[0]) if args else 2000
    k = int(args[1]) if len(args) > 1 else 3
    normalise = "--norm" in sys.argv
    ps.setup()

    # first pass: how many live units does the smallest condition have?
    live_sets = {}
    for pen in PENS:
        for folder in folders_for(pen, N, k):
            r, _ = rates_and_targets(folder, N_TRIALS)
            X = r.reshape(r.shape[0], -1)
            live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
            live_sets.setdefault(pen, []).append(X[live])
    n_keep = min(x.shape[0] for v in live_sets.values() for x in v)
    print(f"N={N}, k={k}, {N_TRIALS} trials, {'unit-normalised' if normalise else 'raw amplitude'}")
    print(f"common subsample: {n_keep} live units per net "
          f"(smallest condition sets it)\n")

    res = {}
    for pen in PENS:
        rows = []
        for X in live_sets.get(pen, []):
            rng = np.random.default_rng(SEED)
            Y, m = embed(X, n_keep, rng, normalise)
            real = score_curve(Y)
            # null: permute each unit's samples independently -> no between-unit structure left
            rng2 = np.random.default_rng(SEED + 1)
            Xp = np.array([rng2.permutation(row) for row in X])
            Yn, mn = embed(Xp, n_keep, np.random.default_rng(SEED), normalise)
            null = score_curve(Yn)
            rows.append(dict(m=m, m_null=mn, real=real, null=null, n_live=X.shape[0]))
        if rows:
            res[pen] = rows

    print(f"{'pen':<6}{'live':>7}{'m(99%)':>9}{'peak sil':>10}{'at k':>6}"
          f"{'null sil':>10}{'GAP':>8}{'peak gap':>10}{'at k':>6}")
    for pen in PENS:
        if pen not in res:
            continue
        f = lambda g: np.mean([r[g[0]][g[1]] if len(g) > 1 else r[g[0]] for r in res[pen]], axis=0)
        sil, nul = f(("real", "sil")), f(("null", "sil"))
        gap = sil - nul
        i, j = int(np.argmax(sil)), int(np.argmax(gap))
        print(f"{pen:<6}{np.mean([r['n_live'] for r in res[pen]]):>7.0f}"
              f"{np.mean([r['m'] for r in res[pen]]):>9.1f}{sil[i]:>10.3f}{KRANGE[i]:>6}"
              f"{nul[i]:>10.3f}{sil[i]-nul[i]:>8.3f}{gap[j]:>10.3f}{KRANGE[j]:>6}")

    PANELS = [("sil", "silhouette (raw)", "higher = tighter clusters"),
              ("gap", "silhouette GAP vs null", "real - permuted; the only fair comparison"),
              ("inertia", "within-cluster inertia (normalised)", "elbow = natural cluster count"),
              ("db", "Davies-Bouldin", "LOWER = tighter clusters")]
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.4))
    for a, (key, lab, note) in zip(ax, PANELS):
        for pen in PENS:
            if pen not in res:
                continue
            if key == "gap":
                cur = np.array([r["real"]["sil"] - r["null"]["sil"] for r in res[pen]])
            elif key == "inertia":
                cur = np.array([r["real"]["inertia"] / r["real"]["inertia"][0] for r in res[pen]])
            else:
                cur = np.array([r["real"][key] for r in res[pen]])
            ps.band(a, KRANGE, cur.mean(0), cur.std(0), COLS[pen], label=pen, marker="")
        a.set(xlabel="number of clusters", ylabel=lab, title=f"{lab}\n{note}")
        if key == "gap":
            a.axhline(0, ls="--", c="0.5", lw=1)
        a.grid(alpha=.25); a.legend(fontsize=8)
    fig.suptitle(f"Unit clustering — N={N}, k={k}, "
                 f"{'unit-normalised (shape only)' if normalise else 'raw amplitude'}  ·  "
                 f"every condition subsampled to {n_keep} live units, {VAR_KEEP:.0%} variance kept",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.87])
    return ps.save(fig, f"clustering_N{N}_k{k}" + ("_norm" if normalise else ""), tight=False)


if __name__ == "__main__":
    main()
