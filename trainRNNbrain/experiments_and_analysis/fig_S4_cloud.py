#!/usr/bin/env python3
"""
Figure for elaboration claim S4 / CS1: the selectivity configuration of the units, drawn directly.

Flip-flop, k = 3, N = 2000, one seed per condition. Each point is one LIVE unit placed at its three
regression loadings on the three remembered bits (flipflop_unitcloud.cloud, space "sel") - the exact
selectivity space at k = 3, no projection. A unit on an axis follows one bit (pure selectivity); a
unit off-axis mixes bits; the origin carries no bit information. The unpenalised and sparsity-only
networks show a 6-armed star of pure units; the participation penalty alone fills the space between
the arms with mixed units; with both penalties the star returns, now with every unit on it. Colour is
log10 participation on a shared scale. Axes are scaled per panel (activity scales differ by orders of
magnitude between conditions): compare SHAPE, not extent.

The quantification of this picture is fig_S4_selectivity.png (Hoyer selectivity vs N, four
conditions, both tasks) and mixedsel_N2000_k3.png (task-free NMF factor counts).

Usage:  python fig_S4_cloud.py [N] [k] [seed_index]      (defaults 2000 3 0)
Output: img/internal_figures/fig_S4_cloud.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from flipflop_unitcloud import cloud
from flipflop_dimensionality import run_folders

PENS = ["none", "rws", "frm", "both"]
LABEL = {"none": "no penalty", "rws": "sparsity only", "frm": "participation only", "both": "participation + sparsity"}
VIEW = dict(elev=22, azim=35)


def main():
    """Draw the four selectivity clouds and write fig_S4_cloud.png."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    si = int(sys.argv[3]) if len(sys.argv) > 3 else 0
    ps.setup()
    folders = {p: sorted(f for f, pen, kk, NN in run_folders() if pen == p and kk == k and NN == N) for p in PENS}
    clouds = {p: cloud(folders[p][si], "sel") for p in PENS}
    lo = min(np.log10(c[1]).min() for c in clouds.values()); hi = max(np.log10(c[1]).max() for c in clouds.values())

    fig = plt.figure(figsize=(9, 8.5))
    for i, p in enumerate(PENS):
        P, part = clouds[p]
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.log10(part), cmap="viridis", vmin=lo, vmax=hi, s=9, alpha=.75, lw=0)
        r = np.abs(P).max()
        for a in range(3):
            v = np.zeros((2, 3)); v[0, a], v[1, a] = -r, r
            ax.plot(v[:, 0], v[:, 1], v[:, 2], color="0.75", lw=0.8)
        ax.set(xlim=(-r, r), ylim=(-r, r), zlim=(-r, r), xlabel="bit 1", ylabel="bit 2", zlabel="bit 3",
               title=f"{LABEL[p]}\n{len(P)} live units")
        ax.view_init(**VIEW)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
        print(f"{p:5s} {len(P)} live units, |loading| range {r:.3g}")
    fig.colorbar(sc, ax=fig.axes, shrink=0.45, pad=0.03, label="log10 participation")
    fig.suptitle(f"S4 — unit selectivity in the 3-bit space (flip-flop k={k}, N={N}): pure units sit on the axes, mixed units between them",
                 fontsize=10.5)
    return ps.save(fig, "fig_S4_cloud", tight=False)


if __name__ == "__main__":
    main()
