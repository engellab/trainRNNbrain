#!/usr/bin/env python3
"""
Active units as a joint function of network size and task complexity, at matched performance.

The per-N and per-k slices are in flipflop_ksweep.py; this puts both axes on one map, which is the
form the "capacity versus demand" reading needs. If silencing is governed by a single ratio of units
to task demand rather than by two independent factors, the iso-M contours should run diagonally -
constant M along lines where N and k rise together - rather than horizontally (N only) or vertically
(k only).

HONESTY CONSTRAINTS BUILT INTO THE FIGURE.

  Three N values only. Every contour interpolates across just two intervals in N, so the sampled
  cells are overlaid as markers and each is annotated with its measured value. The colour field is
  a reading aid; the numbers are the data.

  Unreachable cells are left blank, not filled. A common L* must be reached by every seed in a cell,
  and deeper levels may simply not be attained by the smaller networks within the budget.
  Interpolating over those cells would invent the part of the map the sweep could not measure, so
  they are marked instead.

  Both criteria get a row, because "non-zero" and "doing work" are different claims and the hard
  criterion is the flattering one.

Output: img/internal_figures/flipflop_contour.png

Usage:  python flipflop_contour.py [L* ...]
NOTE ON PROVENANCE. Every number that this file previously quoted from the flip-flop came from the
first sweep, which ran `same_batch=True` and therefore trained on 256 frozen trials - memorisation,
not the task. Those numbers are RETRACTED and have been stripped rather than updated; the data is
quarantined in `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/`. Nothing here has yet
been run against the corrected fresh-batch sweep, so this file currently states METHOD only, with no
results. Do not reintroduce a remembered figure into these docstrings.

"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import IMG_DIR
from flipflop_ksweep import load, stable_crossing, silence_at, PROBE_EVERY, TGT_VAR


def grid(by, ks, Ns, thr):
    """Mean active counts on the (k, N) grid at one performance level.

    Args:
        by: loaded traces; ks, Ns: sorted axes; thr: L* level, or None for the endpoint reading.
    Returns:
        (M_hard, M_scalefree), each a (len(Ns), len(ks)) array with NaN where the level is
        unreachable by at least one seed of that cell.
    """
    H = np.full((len(Ns), len(ks)), np.nan)
    S = np.full((len(Ns), len(ks)), np.nan)
    for i, N in enumerate(Ns):
        for j, k in enumerate(ks):
            runs = by.get((k, N), [])
            hard, sf = [], []
            for t in runs:
                x = None if thr is None else stable_crossing(t["clean"], thr)
                if thr is not None and x is None:
                    continue
                h, s, a = silence_at(t, N, x)
                hard.append(a)
                sf.append(N * (1 - s / 100))
            # Require EVERY seed of the cell to reach the level, so a cell is never represented by
            # its luckiest seed - that would bias exactly the cells nearest the reachability edge.
            if hard and len(hard) == len(runs):
                H[i, j], S[i, j] = np.mean(hard), np.mean(sf)
    return H, S


def panel(ax, ks, Ns, M, title, vmax):
    """Draw one filled-contour panel with the sampled cells overlaid and annotated.

    Args:
        ax: axes; ks, Ns: axis values; M: (len(Ns), len(ks)) active counts, NaN where unreachable;
        title: panel title; vmax: shared colour ceiling so panels are comparable.
    """
    K, NN = np.meshgrid(np.array(ks, float), np.array(Ns, float))
    Mm = np.ma.masked_invalid(M)
    if Mm.count():
        ax.contourf(K, NN, Mm, levels=np.linspace(0, vmax, 21), cmap="viridis", vmin=0, vmax=vmax)
        cs = ax.contour(K, NN, Mm, levels=np.linspace(0, vmax, 9), colors="w", linewidths=.8,
                        alpha=.65)
        ax.clabel(cs, inline=True, fontsize=7, fmt="%.0f")
    for i, N in enumerate(Ns):
        for j, k in enumerate(ks):
            if np.isnan(M[i, j]):
                ax.plot(k, N, "x", color="0.35", ms=9, mew=2)
            else:
                ax.plot(k, N, "o", mfc="none", mec="w", ms=9, mew=1.4)
                ax.annotate(f"{M[i, j]:.0f}", (k, N), textcoords="offset points", xytext=(0, 9),
                            ha="center", fontsize=8, color="w", weight="bold")
    ax.set(xlabel="task complexity $k$ (bits)", ylabel="$N$", yscale="log", xticks=ks,
           yticks=Ns, title=title, xlim=(min(ks) - 0.55, max(ks) + 0.55))
    ax.set_yticklabels([str(n) for n in Ns])
    ax.grid(alpha=.2, color="w", lw=.4)


def main():
    """Map active units over (N, k) at several matched-performance levels, both criteria."""
    levels = [float(a) for a in sys.argv[1:]] or [0.022, 0.015, 0.010]
    by, dropped = load()
    ks = sorted({k for k, _ in by})
    Ns = sorted({N for _, N in by})
    readings = [(None, "endpoint (300k)")] + [(L, f"$L^*$={L:.4f}  ($R^2$={1 - L / TGT_VAR:.3f})")
                                              for L in levels]

    fig, ax = plt.subplots(2, len(readings), figsize=(5.2 * len(readings), 10.2), squeeze=False,
                           constrained_layout=True)
    vmax = float(max(Ns))
    for c, (thr, lab) in enumerate(readings):
        H, S = grid(by, ks, Ns, thr)
        panel(ax[0][c], ks, Ns, H, f"hard $p_i<10^{{-6}}$\n{lab}", vmax)
        panel(ax[1][c], ks, Ns, S, f"scale-free $p_i<0.05\\,q_{{95}}(p)$\n{lab}", vmax)
        print(f"\n=== {lab} ===")
        for name, M in (("hard", H), ("scale-free", S)):
            print(f"  {name}:")
            print("      " + "".join(f"{k:>9}" for k in ks) + "   (k)")
            for i, N in enumerate(Ns):
                cells = "".join("     --  " if np.isnan(v) else f"{v:9.0f}" for v in M[i])
                print(f"  N={N:<4d}{cells}")

    sm = plt.cm.ScalarMappable(cmap="viridis", norm=plt.Normalize(0, vmax))
    cb = fig.colorbar(sm, ax=ax, fraction=.02, pad=.01)
    cb.set_label("active units $M$")
    fig.suptitle("Active units over (network size, task complexity) at matched performance\n"
                 "numbers are measured cell means; x = level unreachable by that cell within 300k; "
                 "colour between cells is interpolation across only 3 values of $N$", fontsize=12)
    out = os.path.join(IMG_DIR, "flipflop_contour.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
