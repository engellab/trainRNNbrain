#!/usr/bin/env python3
"""
Geometric STRUCTURE of unit space over the whole (N, k) grid, per penalty.

Companion to `flipflop_dimensionality.py`. That one asks how many dimensions the population
TRAJECTORY occupies. This one asks how much structure there is in the space of UNITS: treating each
unit as a point whose coordinates are its response profile, how many parameters describe the family
of profiles, and how far is that family from a Gaussian cloud?

  ID_raw    intrinsic dimension of the unit cloud, two-NN maximum likelihood (Facco et al. 2017)
  ID_norm   the same after each unit is scaled to unit length: structure in response SHAPE alone
  kurt      Mardia multivariate kurtosis excess of the embedding; 0 = Gaussian, large = heavy tails
  redund    mean |correlation| between unit pairs; -> 1 clones, -> 0 unrelated

⚠️ EVERY CELL IS SUBSAMPLED TO A COMMON N_KEEP LIVE UNITS. Live counts run from 132 (rws, N=500) to
2000 (both), and every one of these statistics moves with sample size, so without this the matrices
would largely be plotting the active-unit count that `pr_matrix` already reports. The estimator was
calibrated at this n against known dimensions: true d = 1/2/3/4/5 recovered as
1.00/2.06/2.79/3.66/4.52 with sd 0.08-0.45, so it is near-unbiased over the range these networks
occupy and the sd sets the smallest difference worth reading.

⚠️ ID IS REPORTED WITHOUT A NULL, DELIBERATELY. A covariance-matched Gaussian null was tried and it
SATURATES: every condition sits at gap 0.75-0.94 with KS p = 0.000, so it ranks nothing. Since n and
the estimator are identical in every cell, ID itself is directly comparable and is the statistic to
read. See flipflop_manifold.py for the null version and why it is not used here.

⚠️ THE MLE IS d = n / sum(log mu) ON THE FULL SAMPLE. An earlier version trimmed the top 10% of mu
before applying it, which inflated every ID by ~30% (true d=3 read as 4.04). Do not reintroduce the
trim; it belongs with the CDF-slope estimator, not this one.

Output: img/internal_figures/structure_matrix.png
        data/structure_cache.npz  (delete or pass --recompute to rebuild)

Usage:  python flipflop_structure_matrix.py [--recompute]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from pr_matrix import PENS
from flipflop_dimensionality import run_folders, grid
from flipflop_fixedpoints import load_net
from flipflop_bouquet import run_trials
from flipflop_manifold import id_2nn, embed
from flipflop_epairs import mardia

N_TRIALS = 32
N_KEEP = 132           # global minimum live count over the grid (rws, N=500)
CACHE = "data/structure_cache.npz"
FIELDS = ("pen", "k", "N", "live", "id_raw", "id_norm", "m_raw", "m_norm", "kurt", "redund")


def measure(rates):
    """Structure statistics for one network's rate tensor.

    Args:
        rates: (N, T, B) noise-free firing rates.
    Returns:
        dict of scalars; all nan if the net has fewer than N_KEEP live units.
    """
    X = rates.reshape(rates.shape[0], -1).astype(np.float64)
    live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
    nan = dict(live=float(live.sum()), id_raw=np.nan, id_norm=np.nan, m_raw=np.nan,
               m_norm=np.nan, kurt=np.nan, redund=np.nan)
    if live.sum() < N_KEEP:
        return nan
    Xl = X[live]
    Yr, mr = embed(Xl, N_KEEP, np.random.default_rng(0), False)
    Yn, mn = embed(Xl, N_KEEP, np.random.default_rng(0), True)
    Z = Yn - Yn.mean(0, keepdims=True)
    Z = Z / np.maximum(np.linalg.norm(Z, axis=1, keepdims=True), 1e-300)
    C = Z @ Z.T
    iu = np.triu_indices(C.shape[0], k=1)
    return dict(live=float(live.sum()), id_raw=id_2nn(Yr), id_norm=id_2nn(Yn),
                m_raw=float(mr), m_norm=float(mn), kurt=mardia(Yn)[1],
                redund=float(np.abs(C[iu]).mean()))


def compute():
    """Run every net in the grid and cache the structure statistics.

    Returns:
        dict of arrays keyed by FIELDS, one entry per usable run.
    """
    if os.path.exists(CACHE):
        z = np.load(CACHE, allow_pickle=True)
        if all(f in z.files for f in FIELDS):
            return {kk: z[kk] for kk in z.files}
        print("cache is stale; recomputing")
    fold = run_folders()
    print(f"simulating {len(fold)} nets ({N_TRIALS} trials, subsample {N_KEEP} live units)")
    rec = {kk: [] for kk in FIELDS}
    for i, (folder, pen, k, N) in enumerate(fold, 1):
        m = measure(run_trials(load_net(folder)[0], folder, N_TRIALS))
        for kk, v in zip(("pen", "k", "N"), (pen, k, N)):
            rec[kk].append(v)
        for kk in FIELDS[3:]:
            rec[kk].append(m[kk])
        if i % 25 == 0 or i == len(fold):
            print(f"  {i}/{len(fold)}")
    rec = {kk: np.array(v) for kk, v in rec.items()}
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **rec)
    return rec


def main():
    """Plot the structure statistics over the (N, k) grid, one column per penalty."""
    if "--recompute" in sys.argv and os.path.exists(CACHE):
        os.remove(CACHE)
    ps.setup()
    rec = compute()
    ks = sorted(set(rec["k"].tolist()))
    Ns = sorted(set(rec["N"].tolist()))

    ROWS = [("id_raw", "$ID$ (raw)", "intrinsic dim of the unit cloud", "viridis", None),
            ("id_norm", "$ID$ (shape)", "unit-normalised: shape only", "viridis", None),
            ("kurt", "Mardia kurtosis", "0 = Gaussian, large = heavy tails", "inferno", None),
            ("redund", "mean |corr|", "1 = clones, 0 = unrelated", "magma", (0, 0.5))]
    for key, lab, note, _, _ in ROWS:
        print(f"\n{'='*70}\n{lab} — {note}\n{'='*70}")
        print(f"{'pen':<6}" + "".join(f"{f'k={k}':>8}" for k in ks) + f"{'  mean':>9}")
        for p in PENS:
            row = f"{p:<6}"
            for k in ks:
                m = (rec["pen"] == p) & (rec["k"] == k) & np.isfinite(rec[key])
                row += f"{np.mean(rec[key][m]):>8.2f}" if m.any() else f"{'-':>8}"
            mm = (rec["pen"] == p) & np.isfinite(rec[key])
            row += f"{np.mean(rec[key][mm]):>9.2f}" if mm.any() else f"{'-':>9}"
            print(row)

    fig, ax = plt.subplots(len(ROWS) + 1, len(PENS),
                           figsize=(4.3 * len(PENS), 4.0 * (len(ROWS) + 1)), squeeze=False)
    for c_i, pen in enumerate(PENS):
        for r_i, (key, lab, note, cmap, lim) in enumerate(ROWS):
            a = ax[r_i][c_i]
            Z, S, _ = grid(rec, pen, key, ks, Ns)
            if not np.isfinite(Z).any():
                a.text(.5, .5, f"no {pen} data", ha="center", va="center", transform=a.transAxes,
                       color="0.5"); a.set_xticks([]); a.set_yticks([]); continue
            vals = rec[key][np.isfinite(rec[key])]
            vmin, vmax = lim if lim else (np.percentile(vals, 2), np.percentile(vals, 98))
            im = a.imshow(Z, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(Ns)):
                for j in range(len(ks)):
                    if np.isfinite(Z[i, j]):
                        rel = (Z[i, j] - vmin) / max(vmax - vmin, 1e-12)
                        col = "white" if rel < 0.6 else "black"
                        a.text(j, i, f"{Z[i, j]:.2f}" if abs(Z[i, j]) < 100 else f"{Z[i, j]:.0f}",
                               ha="center", va="bottom", fontsize=7.2, color=col)
                        a.text(j, i, f"±{S[i, j]:.2f}" if abs(S[i, j]) < 100 else f"±{S[i,j]:.0f}",
                               ha="center", va="top", fontsize=5.4, color=col, alpha=.85)
                    else:
                        a.text(j, i, "·", ha="center", va="center", color="0.6", fontsize=9)
            a.set(xticks=range(len(ks)), xticklabels=ks, yticks=range(len(Ns)),
                  yticklabels=[str(n) for n in Ns], xlabel="k (bits)")
            if c_i == 0:
                a.set_ylabel(f"{lab}\nN (units)")
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
            a.set_title((f"{pen}\n" if r_i == 0 else "") + f"{lab} — {note}",
                        fontsize=10.2, fontweight="bold" if r_i == 0 else "normal")

        b = ax[len(ROWS)][c_i]
        Z, S, _ = grid(rec, pen, "id_norm", ks, Ns)
        if np.isfinite(Z).any():
            for i, N in enumerate(Ns):
                if np.isfinite(Z[i]).any():
                    ps.band(b, ks, Z[i], S[i], ps.col_n(N), label=f"N={N}")
            b.set(xlabel="k (bits)", ylabel="$ID$ (shape)" if c_i == 0 else "", xticks=ks,
                  ylim=(0, None), title="$ID$ (shape) vs k\nshaded = seed sd")
            b.axhspan(0, 0.45, color="0.85", zorder=0)
            b.text(.02, .02, "grey band: estimator sd at this n", transform=b.transAxes,
                   fontsize=6.5, va="bottom", color="0.35")
            b.legend(fontsize=7, loc="upper right"); b.grid(alpha=.25)
    fig.suptitle("Geometric structure of UNIT space over the (N, k) grid, per penalty\n"
                 f"each unit is a point; every cell subsampled to {N_KEEP} live units so the "
                 "statistics are not reporting the active-unit count", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    return ps.save(fig, "structure_matrix", tight=False)


if __name__ == "__main__":
    main()
