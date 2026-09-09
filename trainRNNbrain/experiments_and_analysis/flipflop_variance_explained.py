#!/usr/bin/env python3
"""
Variance carried by the leading PCs, over the (N, k) grid, per penalty.

The assumption-free companion to `flipflop_dimensionality.py`. D_PR and D_95 are single summaries
of the activity covariance spectrum; this reports the spectrum at a FIXED cut instead - what
fraction of the population variance the first 5 (and 1, and 10) principal components carry - so a
reader can see the concentration directly rather than through a moment ratio.

Same nets, same spectra, same cache (`data/dimensionality_cache.npz`) as the dimensionality figure,
so the two are strictly like-for-like: noise-free rates, relu applied, pooled over time and trials,
centred per neuron, silent units contributing zero variance.

⚠️ VE AND D_PR CAN DISAGREE AND BOTH ARE PLOTTED. A spectrum with a heavy head and a heavy tail can
have high VE@5 AND high D_PR; VE@5 is blind to everything past the fifth eigenvalue, and D_PR is a
moment ratio that no fixed cut reproduces. Row 3 carries D_PR next to VE@5 so neither is read alone.

Output: img/internal_figures/variance_explained_matrix.png

Usage:  python flipflop_variance_explained.py [--recompute]
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from pr_matrix import PENS
from flipflop_dimensionality import CACHE, compute, grid


def main():
    """Plot VE@5 over the (N, k) grid plus VE vs k curves and the VE@n profile."""
    if "--recompute" in sys.argv and os.path.exists(CACHE):
        os.remove(CACHE)
    ps.setup()
    rec = compute()
    ks = sorted(set(rec["k"].tolist()))
    Ns = sorted(set(rec["N"].tolist()))

    print(f"\n{'='*78}\nfraction of population variance in the leading PCs "
          f"(mean over the whole grid)\n{'='*78}")
    print(f"{'pen':<6}{'VE@1':>9}{'VE@5':>9}{'VE@10':>9}{'D_PR':>9}{'D_95':>9}{'n':>6}")
    for p in PENS:
        m = rec["pen"] == p
        if not m.any():
            continue
        f = lambda kk: float(np.nanmean(rec[kk][m]))
        print(f"{p:<6}{f('ve1'):>9.3f}{f('ve5'):>9.3f}{f('ve10'):>9.3f}"
              f"{f('d_pr'):>9.2f}{f('d_95'):>9.1f}{int(m.sum()):>6}")

    print(f"\nVE@5 by k (mean over N), showing the task-complexity trend")
    print(f"{'pen':<6}" + "".join(f"{f'k={k}':>8}" for k in ks))
    for p in PENS:
        row = f"{p:<6}"
        for k in ks:
            m = (rec["pen"] == p) & (rec["k"] == k)
            row += f"{np.nanmean(rec['ve5'][m]):>8.3f}" if m.any() else f"{'-':>8}"
        print(row)

    fig, ax = plt.subplots(3, len(PENS), figsize=(4.3 * len(PENS), 11.2), squeeze=False)
    for c_i, pen in enumerate(PENS):
        # ---- row 0: VE@5 over the (N, k) grid -------------------------------------------------
        a = ax[0][c_i]
        Z, S, _ = grid(rec, pen, "ve5", ks, Ns)
        if np.isfinite(Z).any():
            im = a.imshow(Z, cmap="magma", vmin=0, vmax=1, aspect="auto")
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
            fig.colorbar(im, ax=a, fraction=0.046, pad=0.02)
        else:
            a.text(.5, .5, f"no {pen} data", ha="center", va="center", transform=a.transAxes,
                   color="0.5"); a.set_xticks([]); a.set_yticks([])
        if c_i == 0:
            a.set_ylabel("N (units)")
        a.set_title(f"{pen}\nVE@5 — variance in the first 5 PCs", fontsize=10.5,
                    fontweight="bold")

        # ---- row 1: VE@5 vs k, one curve per N ------------------------------------------------
        b = ax[1][c_i]
        if np.isfinite(Z).any():
            for i, N in enumerate(Ns):
                if np.isfinite(Z[i]).any():
                    ps.band(b, ks, Z[i], S[i], ps.col_n(N), label=f"N={N}")
            b.set(xlabel="k (bits)", ylabel="VE@5" if c_i == 0 else "", xticks=ks, ylim=(0, 1.05),
                  title="VE@5 vs k\nhow concentrated the spectrum stays as the task hardens")
            b.legend(fontsize=7, loc="upper right"); b.grid(alpha=.25)

        # ---- row 2: the VE@n profile, and D_PR for contrast ------------------------------------
        c = ax[2][c_i]
        for i, N in enumerate(Ns):
            m = (rec["pen"] == pen) & (rec["N"] == N)
            if not m.any():
                continue
            prof = [np.nanmean(rec[kk][m]) for kk in ("ve1", "ve5", "ve10")]
            c.plot([1, 5, 10], prof, "o-", color=ps.col_n(N), lw=1.4, label=f"N={N}")
        mm = rec["pen"] == pen
        if mm.any():
            c.text(.97, .05, f"grid mean $D_{{PR}}$ = {np.nanmean(rec['d_pr'][mm]):.1f}\n"
                             f"grid mean $D_{{95}}$ = {np.nanmean(rec['d_95'][mm]):.0f}",
                   transform=c.transAxes, fontsize=8, ha="right", va="bottom",
                   bbox=dict(fc="white", ec="0.7", alpha=.9, boxstyle="round,pad=0.3"))
        c.set(xlabel="number of leading PCs", ylabel="cumulative variance" if c_i == 0 else "",
              xticks=[1, 5, 10], ylim=(0, 1.05),
              title="cumulative variance vs PC count\naveraged over k")
        c.legend(fontsize=7, loc="lower right"); c.grid(alpha=.25)

    fig.suptitle("Variance carried by the leading principal components, over the (N, k) grid\n"
                 "same nets and spectra as dimensionality_matrix.png — a fixed cut through the "
                 "spectrum rather than a moment summary", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "variance_explained_matrix", tight=False)


if __name__ == "__main__":
    main()
