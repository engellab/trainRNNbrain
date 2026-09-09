#!/usr/bin/env python3
"""
Arm structure of the selectivity star: purity, occupancy, and 2k-cluster separability.

MOTIVATION, FROM LOOKING AT THE CLOUD. `flipflop_unitcloud.py` shows the unit population in
selectivity space is a 6-ARMED STAR at k=3 - dense arms along +-bit1, +-bit2, +-bit3 with a core at
the origin - not a blob, not clusters, not a manifold. Every statistic tried before that (silhouette
against a permutation null, intrinsic dimension, Mardia kurtosis) was fitting the wrong model to
that shape, which is why their answers kept reversing. The right descriptors are how PURE the arms
are and how EVENLY they are occupied.

  hoyer      per unit, (sqrt(d) - ||b||_1/||b||_2) / (sqrt(d) - 1) over the d = 2k loadings.
             0 = tuning spread evenly over every channel, 1 = concentrated on exactly one. No
             threshold, no clustering, no null. It is the per-unit dual of the participation ratios
             used elsewhere here, since ||b||_1/||b||_2 = sqrt(PR of the loading vector).
  purity     per unit, max_j |beta_j| / ||beta||: the share of its selectivity on its dominant
             channel. Reported against the value for RANDOM directions in the same dimension, which
             is the "no arms" baseline and is NOT 1/sqrt(d).
  arm_PR     participation ratio of the counts in the 2k arms, so 2k = perfectly even occupancy and
             1 = every unit on one arm. Says whether the star is symmetric or lopsided.
  sil_2k     silhouette of k-means with EXACTLY 2k clusters on the unit-normalised loadings.
             Fixing the cluster count is what makes this trustworthy where the 2..30 sweep was not:
             the number is set by the task, not chosen to flatter the data.
  sil_null   the same on n uniform random directions on the sphere. Silhouette has no meaningful
             zero, so only sil_2k - sil_null is interpretable.

⚠️ THE REGRESSORS ARE HALF-WAVE RECTIFIED, NOT THE RAW BITS. An earlier version regressed on the k
signed bit traces. That is misspecified for ReLU units: a unit that falls to EXACTLY zero for one
sign of a bit cannot be fitted by a line through three levels. The rectified basis relu(+b_j),
relu(-b_j) spans the linear one (relu(b) - relu(-b) = b) plus the absolute values, so it is strictly
richer, and it raised median R2 by ~0.14 in every condition. Cross-validated, so the extra
parameters cannot flatter it.
⚠️ NO REGULARISATION, AND THAT IS CHECKED RATHER THAN ASSUMED. Design matrix is (9600, 2k+1);
condition number 6.7 at k=3 (collinearity would show above ~30), and the in-sample vs 5-fold-CV R2
gap is 0.0005. The pairs relu(+b)+relu(-b) = 1[b != 0] do not collapse onto the intercept because
each bit sits at zero for a different 23-33% of samples.

⚠️ UNITS WITH NO TUNING ARE EXCLUDED, and the excluded fraction is reported. A unit whose bit
regression explains almost none of its variance has a loading DIRECTION that is pure noise, and
including those fills the core with random directions that depress purity for every condition
equally. Gate: live (participation >= SILENT_FLIPFLOP) and R2 >= R2_GATE.

⚠️ EVERY CELL IS SUBSAMPLED TO A COMMON N_KEEP, since silhouette and arm counts both move with n.

⚠️ k=1 IS DEGENERATE: with one bit there are two arms and purity is 1.0 by construction. It is
computed but must not be read as evidence of anything.

Output: img/internal_figures/arms_matrix.png
        data/arms_cache.npz

Usage:  python flipflop_arms.py [--recompute]
"""

import os
import sys
import numpy as np
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from pr_matrix import PENS
from flipflop_dimensionality import run_folders, grid
from flipflop_fixedpoints import load_net
from flipflop_diversity import rates_and_targets, pr_of

N_TRIALS = 32
N_KEEP = 100          # common subsample; below the smallest tuned-unit count in the grid
R2_GATE = 0.15        # below this a unit's loading direction is noise
CACHE = "data/arms_cache.npz"
KEYS = ("tuned_frac", "r2_med", "r2_q25", "hoyer", "hoyer_q25", "purity",
        "purity_null", "arm_pr", "sil_2k", "sil_null")
FIELDS = ("pen", "k", "N") + KEYS
SEED = 0


def random_purity(k, n=20000, seed=SEED):
    """Median dominant-axis share for uniform random directions in R^k.

    This is the "no arm structure" baseline. It is NOT 1/sqrt(k): a random unit vector is not
    uniformly spread over its coordinates, so the naive baseline would understate chance.

    Args:
        k: dimension; n: samples; seed: RNG seed.
    Returns:
        float median of max_j |u_j| over random unit vectors u.
    """
    rng = np.random.default_rng(seed)
    U = rng.normal(size=(n, k))
    U /= np.linalg.norm(U, axis=1, keepdims=True)
    return float(np.median(np.abs(U).max(1)))


def hoyer(B):
    """Hoyer sparsity of each row.

    Args:
        B: (n, d) loading vectors.
    Returns:
        (n,) sparsity in [0, 1]; 0 = perfectly even over the d channels, 1 = one channel only.
    """
    d = B.shape[1]
    l1 = np.abs(B).sum(1)
    l2 = np.linalg.norm(B, axis=1)
    return (np.sqrt(d) - l1 / np.maximum(l2, 1e-300)) / (np.sqrt(d) - 1)


def measure(rates, targets, k):
    """Arm statistics for one network.

    Args:
        rates: (N, T, B) rates; targets: (k, T, B) target bits; k: number of bits.
    Returns:
        dict of scalars, nan where there are too few tuned units.
    """
    X = rates.reshape(rates.shape[0], -1).astype(np.float64)
    live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
    Xc = X[live] - X[live].mean(1, keepdims=True)
    Braw = targets.reshape(k, -1).T
    G = np.column_stack([np.ones(Braw.shape[0]), np.maximum(Braw, 0), np.maximum(-Braw, 0)])
    beta, *_ = np.linalg.lstsq(G, Xc.T, rcond=None)
    resid = Xc.T - G @ beta
    r2 = 1.0 - (resid ** 2).sum(0) / np.maximum((Xc.T ** 2).sum(0), 1e-300)
    B = beta[1:].T                                   # (n_live, 2k) rectified loadings
    tuned = (r2 >= R2_GATE) & (np.linalg.norm(B, axis=1) > 0)
    out = {kk: np.nan for kk in KEYS}
    out["tuned_frac"] = float(tuned.mean()) if live.sum() else np.nan
    if tuned.sum() < N_KEEP:
        return out
    rng = np.random.default_rng(SEED)
    Bt = B[tuned][rng.choice(int(tuned.sum()), N_KEEP, replace=False)]
    U = Bt / np.linalg.norm(Bt, axis=1, keepdims=True)

    out["r2_med"] = float(np.median(r2[tuned]))
    out["r2_q25"] = float(np.quantile(r2[tuned], 0.25))
    hy = hoyer(Bt)
    out["hoyer"] = float(np.median(hy))
    out["hoyer_q25"] = float(np.quantile(hy, 0.25))
    out["purity"] = float(np.median(np.abs(U).max(1)))
    out["purity_null"] = random_purity(2 * k)
    arm = np.abs(U).argmax(1)                    # the 2k rectified channels ARE the arms
    counts = np.bincount(arm, minlength=2 * k).astype(float)
    out["arm_pr"] = pr_of(counts) if counts.sum() > 0 else np.nan
    if k >= 2:
        lab = KMeans(n_clusters=2 * k, n_init=10, random_state=SEED).fit_predict(U)
        out["sil_2k"] = float(silhouette_score(U, lab))
        R = rng.normal(size=U.shape)
        R /= np.linalg.norm(R, axis=1, keepdims=True)
        ln = KMeans(n_clusters=2 * k, n_init=10, random_state=SEED).fit_predict(R)
        out["sil_null"] = float(silhouette_score(R, ln))
    return out


def compute():
    """Sweep the grid and cache the arm statistics.

    Returns:
        dict of arrays keyed by FIELDS.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if all(f in z.files for f in FIELDS):
            return {kk: z[kk] for kk in z.files}
    fold = run_folders()
    print(f"simulating {len(fold)} nets (R2 gate {R2_GATE}, subsample {N_KEEP} tuned units)")
    rec = {kk: [] for kk in FIELDS}
    for i, (folder, pen, k, N) in enumerate(fold, 1):
        r, t = rates_and_targets(folder, N_TRIALS)
        m = measure(r, t, k)
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
    """Tabulate and plot arm purity, occupancy and 2k-cluster separability over the grid."""
    if "--recompute" in sys.argv and os.path.exists(CACHE):
        os.remove(CACHE)
    ps.setup()
    rec = compute()
    ks = sorted(set(rec["k"].tolist()))
    Ns = sorted(set(rec["N"].tolist()))
    rec["purity_excess"] = rec["purity"] - rec["purity_null"]
    rec["sil_excess"] = rec["sil_2k"] - rec["sil_null"]
    rec["arm_pr_frac"] = rec["arm_pr"] / (2.0 * rec["k"])

    SHOW = [("tuned_frac", "tuned fraction", "live units with R2 >= %.2f" % R2_GATE),
            ("hoyer", "Hoyer sparsity of loadings", "1 = one channel only, 0 = spread evenly"),
            ("purity_excess", "arm purity excess", "median dominant-axis share minus random"),
            ("sil_excess", "silhouette excess (2k clusters)", "vs uniform random directions"),
            ("arm_pr_frac", "arm evenness", "PR of arm counts / 2k; 1 = perfectly even")]
    for key, lab, note in SHOW:
        print(f"\n{'='*72}\n{lab} — {note}\n{'='*72}")
        print(f"{'pen':<6}" + "".join(f"{f'k={k}':>8}" for k in ks) + f"{'  k>=2 mean':>12}")
        for p in PENS:
            row = f"{p:<6}"
            for k in ks:
                m = (rec["pen"] == p) & (rec["k"] == k) & np.isfinite(rec[key])
                row += f"{np.mean(rec[key][m]):>8.3f}" if m.any() else f"{'-':>8}"
            mm = (rec["pen"] == p) & (rec["k"] >= 2) & np.isfinite(rec[key])
            row += f"{np.mean(rec[key][mm]):>12.3f}" if mm.any() else f"{'-':>12}"
            print(row)

    fig, ax = plt.subplots(2, 3, figsize=(18, 9))
    for a, (key, lab, note) in zip(ax.ravel(), SHOW):
        for p in PENS:
            Z, S, _ = grid(rec, p, key, ks, Ns)
            ps.band(a, ks, np.nanmean(Z, axis=0), np.nanstd(Z, axis=0),
                    {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728",
                     "both": "#1f77b4"}[p], label=p)
        a.set(xlabel="k (bits)", ylabel=lab, xticks=ks, title=f"{lab}\n{note}")
        a.axvspan(0.6, 1.4, color="0.9", zorder=0)
        a.text(1, a.get_ylim()[0], " k=1\n degenerate", fontsize=6.5, va="bottom", color="0.4")
        a.grid(alpha=.25); a.legend(fontsize=8)
    ax.ravel()[-1].axis("off")
    fig.suptitle("Arm structure of the selectivity star, over the (N, k) grid\n"
                 f"cluster count FIXED at 2k (one per signed bit axis)  ·  "
                 f"{N_KEEP} tuned units per net  ·  shaded = spread across N", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.91])
    return ps.save(fig, "arms_matrix", tight=False)


if __name__ == "__main__":
    main()
