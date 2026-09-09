#!/usr/bin/env python3
"""
Selectivity of single units over the (N, k) grid: how well the bits explain a unit, and how
concentrated its tuning is. Laid out like `pr_matrix`: a matrix per penalty, then curves vs k.

TWO NUMBERS, BOTH PER UNIT, BOTH FROM THE SAME REGRESSION.

  R^2     Each live unit's rate is regressed on the HALF-WAVE RECTIFIED bit traces - relu(+b_j) and
          relu(-b_j) for each of the k bits, so 2k regressors plus an intercept, 2k+1 = 7 parameters
          at k=3. R^2 is the fraction of that unit's activity the bits account for. A unit near 1 is
          doing the task: watch it and you can read off the remembered bits. A unit near 0 is active
          but its activity has nothing to do with what is being remembered.
  Hoyer   (sqrt(d) - ||b||_1/||b||_2) / (sqrt(d) - 1) over the d = 2k loadings: 1 = all tuning on a
          single channel, 0 = spread evenly over every channel. Threshold-free, needs no clustering
          and no null, and it is the per-unit dual of the participation ratios used elsewhere in
          this project since ||b||_1/||b||_2 = sqrt(PR of the loading vector).

⚠️ THE BASIS MUST BE RECTIFIED, NOT THE RAW SIGNED BITS. ReLU units are non-negative and many fall
to EXACTLY zero for one sign of a bit, which a line through three levels cannot fit. The rectified
basis spans the linear one (relu(b) - relu(-b) = b) plus the absolute values, so it is strictly
richer; switching to it raised median R^2 by ~0.14 in every condition. It also settles a control
that a linear fit could not: frm's untuned units are NOT merely rectified-tuned - its untuned
fraction only falls 0.414 -> 0.340 under the richer basis.

⚠️ NO REGULARISATION, VERIFIED RATHER THAN ASSUMED. Design matrix (9600, 7) at k=3, condition number
6.7, in-sample vs 5-fold-CV R^2 gap 0.0005. Nothing to regularise away at n/p ~ 1370.

Reads `data/arms_cache.npz`, written by flipflop_arms.py; run that first.

Output: img/internal_figures/selectivity_matrix.png

Usage:  python flipflop_selectivity_matrix.py
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from pr_matrix import PENS
from flipflop_arms import compute, R2_GATE, N_KEEP
from flipflop_dimensionality import grid

ROWS = [("r2_med", "median $R^2$", "variance of a unit explained by the bits", "viridis", (0, 1)),
        ("hoyer", "median Hoyer sparsity", "1 = tuning on one channel, 0 = spread evenly",
         "magma", (0, 1))]


def main():
    """Plot median R^2 and Hoyer sparsity over the (N, k) grid, one column per penalty."""
    ps.setup()
    rec = compute()
    ks = sorted(set(rec["k"].tolist()))
    Ns = sorted(set(rec["N"].tolist()))

    for key, lab, note, _, _ in ROWS:
        print(f"\n{'='*74}\n{lab} — {note}\n{'='*74}")
        print(f"{'pen':<6}" + "".join(f"{f'k={k}':>8}" for k in ks) + f"{'  mean':>9}")
        for p in PENS:
            row = f"{p:<6}"
            for k in ks:
                m = (rec["pen"] == p) & (rec["k"] == k) & np.isfinite(rec[key])
                row += f"{np.mean(rec[key][m]):>8.3f}" if m.any() else f"{'-':>8}"
            mm = (rec["pen"] == p) & np.isfinite(rec[key])
            row += f"{np.mean(rec[key][mm]):>9.3f}" if mm.any() else f"{'-':>9}"
            print(row)

    fig, ax = plt.subplots(4, len(PENS), figsize=(4.3 * len(PENS), 16.4), squeeze=False)
    for c_i, pen in enumerate(PENS):
        for r_i, (key, lab, note, cmap, lim) in enumerate(ROWS):
            a = ax[r_i][c_i]
            Z, S, _ = grid(rec, pen, key, ks, Ns)
            if not np.isfinite(Z).any():
                a.text(.5, .5, f"no {pen} data", ha="center", va="center", transform=a.transAxes,
                       color="0.5"); a.set_xticks([]); a.set_yticks([]); continue
            im = a.imshow(Z, cmap=cmap, vmin=lim[0], vmax=lim[1], aspect="auto")
            for i in range(len(Ns)):
                for j in range(len(ks)):
                    if np.isfinite(Z[i, j]):
                        col = "white" if Z[i, j] < 0.6 else "black"
                        a.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="bottom", fontsize=7.5,
                               color=col)
                        a.text(j, i, f"±{S[i, j]:.2f}", ha="center", va="top", fontsize=5.6,
                               color=col, alpha=.85)
                    else:
                        a.text(j, i, "·", ha="center", va="center", color="0.6", fontsize=9)
            a.set(xticks=range(len(ks)), xticklabels=ks, yticks=range(len(Ns)),
                  yticklabels=[str(n) for n in Ns], xlabel="k (bits)")
            if c_i == 0:
                a.set_ylabel("N (units)")
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
            a.set_title((f"{pen}\n" if r_i == 0 else "") + f"{lab}\n{note}",
                        fontsize=10.2, fontweight="bold" if r_i == 0 else "normal")

        for r_i, (key, lab, note, _, lim) in enumerate(ROWS):
            b = ax[2 + r_i][c_i]
            Z, S, _ = grid(rec, pen, key, ks, Ns)
            if not np.isfinite(Z).any():
                b.set_xticks([]); b.set_yticks([]); continue
            for i, N in enumerate(Ns):
                if np.isfinite(Z[i]).any():
                    ps.band(b, ks, Z[i], S[i], ps.col_n(N), label=f"N={N}")
            b.set(xlabel="k (bits)", ylabel=lab if c_i == 0 else "", xticks=ks, ylim=lim,
                  title=f"{lab} vs k\nshaded = seed sd")
            b.legend(fontsize=7, loc="lower right"); b.grid(alpha=.25)

    fig.suptitle("Single-unit selectivity over the (N, k) grid, per penalty\n"
                 f"each live unit regressed on the 2k rectified bit traces + intercept  ·  "
                 f"{N_KEEP} tuned units per net (R² ≥ {R2_GATE})  ·  "
                 "$R^2$ = how much of a unit the bits explain, Hoyer = how concentrated its tuning is",
                 fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    return ps.save(fig, "selectivity_matrix", tight=False)


if __name__ == "__main__":
    main()
