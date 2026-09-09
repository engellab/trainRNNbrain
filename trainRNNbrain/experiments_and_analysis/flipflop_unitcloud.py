#!/usr/bin/env python3
"""
Rotating 3-D point cloud of UNITS, one panel per penalty, for direct visual inspection.

Every earlier structure statistic in this project reduced this cloud to a number, and several of
those numbers reversed once a confound was controlled (silhouette against a covariance-destroying
null, intrinsic dimension at a single k, Mardia kurtosis at unmatched embedding dimension). This
script draws the thing itself so the shape can be judged directly rather than through an estimator.

Each point is one LIVE unit. Two spaces are available:

  pcs   the leading 3 principal components of the unit ensemble (units centred over the population,
        so the axes describe how units DIFFER from one another).
  sel   for k=3 only, the exact selectivity space: each unit's three regression loadings on the
        three target bits. No PCA and no information discarded - at k=3 the tuning space IS
        three-dimensional, so this is the literal selectivity configuration.

Colour is log10 participation, on a scale shared across panels, so amplitude structure is visible
rather than hidden by it.

⚠️ AXES ARE AUTOSCALED PER PANEL. The four conditions differ by orders of magnitude in activity
scale, so a shared axis would render three of the four as a dot. Compare SHAPE across panels, never
extent; the per-panel ranges are printed and written into each subplot title.

⚠️ UNIT COUNTS DIFFER AND ARE NOT EQUALISED HERE. none/rws have ~300 live units at N=2000, k=3
against 2000 for both. That difference is part of what is being looked at, so no subsampling is
applied - unlike the statistical scripts, where it had to be.

Output: img/internal_figures/unitcloud_{space}_N{N}_k{k}.gif

Usage:  python flipflop_unitcloud.py [N] [k] [pcs|sel|both]
"""

import os
import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, SILENT_FLIPFLOP
from flipflop_heterogeneity import folders_for
from flipflop_diversity import rates_and_targets

PENS = ["none", "rws", "frm", "both"]
N_TRIALS = 48
FRAMES = 120


def cloud(folder, space, normalise=False):
    """Live-unit point cloud in 3 dimensions, plus each unit's participation.

    Args:
        folder: run folder; space: "pcs" or "sel"; normalise: scale each unit to unit length first
            (pcs only; makes the cloud describe response shape rather than amplitude).
    Returns:
        (P, part): (n_live, 3) coordinates and (n_live,) participation values.
    """
    rates, targets = rates_and_targets(folder, N_TRIALS)
    X = rates.reshape(rates.shape[0], -1).astype(np.float64)
    part = X.std(1) + np.quantile(X, 0.9, axis=1)
    live = part >= SILENT_FLIPFLOP
    X, part = X[live], part[live]
    Xc = X - X.mean(1, keepdims=True)
    if space == "sel":
        G = targets.reshape(targets.shape[0], -1).T
        G = np.column_stack([np.ones(G.shape[0]), G])
        beta, *_ = np.linalg.lstsq(G, Xc.T, rcond=None)
        return beta[1:4].T, part
    if normalise:
        Xc = Xc / np.maximum(np.linalg.norm(Xc, axis=1, keepdims=True), 1e-300)
    Z = Xc - Xc.mean(0, keepdims=True)              # centre over UNITS
    U, s, _ = np.linalg.svd(Z, full_matrices=False)
    return (U[:, :3] * s[:3]), part


def main():
    """Render a rotating 4-panel GIF of the unit clouds."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    which = sys.argv[3] if len(sys.argv) > 3 else "both"
    spaces = ["pcs", "sel"] if which == "both" else [which]
    if "sel" in spaces and k != 3:
        print("note: 'sel' plots the first 3 of k loadings; exact only at k=3")

    for space in spaces:
        data = {}
        for pen in PENS:
            f = folders_for(pen, N, k)
            if f:
                data[pen] = cloud(f[0], space)
        if not data:
            print(f"no data for N={N}, k={k}"); continue

        allp = np.concatenate([p for _, p in data.values()])
        vmin, vmax = np.log10(max(allp.min(), 1e-6)), np.log10(allp.max())
        axlab = ("PC1", "PC2", "PC3") if space == "pcs" else ("bit 1", "bit 2", "bit 3")

        print(f"\nspace={space}, N={N}, k={k}")
        for pen, (P, _) in data.items():
            print(f"  {pen:<5} {P.shape[0]:>5} live units   ranges "
                  + ", ".join(f"[{P[:, j].min():+.3g}, {P[:, j].max():+.3g}]" for j in range(3)))

        fig = plt.figure(figsize=(19, 5.4))
        axes, lims = [], {}
        for i, pen in enumerate(PENS):
            a = fig.add_subplot(1, 4, i + 1, projection="3d")
            axes.append(a)
            if pen in data:
                P = data[pen][0]
                # robust per-axis limits, padded, so a single extreme unit cannot flatten the cloud
                lo = np.percentile(P, 0.5, axis=0)
                hi = np.percentile(P, 99.5, axis=0)
                pad = 0.15 * np.maximum(hi - lo, 1e-9)
                lims[pen] = list(zip(lo - pad, hi + pad))

        def draw(elev, azim):
            """Render one view of all four clouds."""
            for a, pen in zip(axes, PENS):
                a.clear()
                if pen not in data:
                    a.set_title(f"{pen}\nno data", fontsize=9); a.set_axis_off(); continue
                P, part = data[pen]
                a.scatter(P[:, 0], P[:, 1], P[:, 2], s=5, alpha=.55,
                          c=np.log10(np.maximum(part, 1e-6)), cmap="turbo",
                          vmin=vmin, vmax=vmax, linewidths=0, rasterized=True)
                a.set(xlabel=axlab[0], ylabel=axlab[1], zlabel=axlab[2],
                      xlim=lims[pen][0], ylim=lims[pen][1], zlim=lims[pen][2])
                a.view_init(elev=elev, azim=azim)
                a.grid(alpha=.2)
                a.set_title(f"{pen} — {P.shape[0]} live units\n"
                            f"axes autoscaled per panel", fontsize=9, fontweight="bold")

        draw(18, 45)
        fig.suptitle(f"Unit clouds in {'3 PCs of unit space' if space == 'pcs' else 'selectivity '
                     'space (bit loadings, exact at k=3)'} — N={N}, k={k}\n"
                     "each point is one live unit  ·  colour = log10 participation (shared scale)  "
                     "·  ⚠ compare SHAPE, not extent", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.88])

        out = os.path.join(IMG_DIR, f"unitcloud_{space}_N{N}_k{k}.gif")
        FuncAnimation(fig, lambda i: (draw(18 + 12 * np.sin(2 * np.pi * i / FRAMES), i * 3), ())[1],
                      frames=FRAMES, blit=False).save(out, writer=PillowWriter(fps=20), dpi=70)
        plt.close(fig)
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
