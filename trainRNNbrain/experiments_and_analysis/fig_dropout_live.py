"""Live units vs iteration, dropout vs no dropout, 3-bit flip-flop N=1000 (job 6201944).

The 150k table (`flipflop_dropout_readout.py`) says dropout keeps ~40-80% more units alive in the
`none` and `rws` arms. Silence in this project keeps growing with training everywhere, so the
table alone cannot distinguish "dropout holds units open" from "dropout merely slows the same
silencing down". This figure draws the whole trace: live units (scale-free, and the flip-flop
absolute 4e-2) against iteration for both dropout kinds and their 3-seed no-dropout reference,
plus the clean loss, so the cost is visible next to the count.

Usage: python fig_dropout_live.py [<trained_RNNs root>]
Writes img/internal_figures/dropout_live_vs_iter.png
"""
import glob
import os
import pickle
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainRNNbrain.experiments_and_analysis.common import (DATA_DIR, IMG_DIR, SILENT_FLIPFLOP,
                                                           active_count, logbin)
from trainRNNbrain.experiments_and_analysis.flipflop_dropout_readout import DROP_SUB, READ_AT, REFS

ARMS = ["none", "rws", "frm", "both"]
COL = {"-- none": "#7a7a72", "mute": "#2a78d6", "dead": "#eb6834"}
OUT = os.path.join(IMG_DIR, "dropout_live_vs_iter.png")


def trace(net_dir):
    """Live-unit and clean-loss traces of one network, cut at READ_AT.

    Args:
        net_dir: per-network folder holding *_ParticipationTrace.pkl.
    Returns:
        dict with `pit` (participation snapshot iterations), `sf` / `ab` (live counts under the
        scale-free and absolute 4e-2 criteria at those iterations), `it` and `loss` (clean-loss
        probe iterations and values), or None if there is no trace.
    """
    f = glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))
    if not f:
        return None
    with open(f[0], "rb") as fh:
        tr = pickle.load(fh)
    pit = np.asarray(tr["participation_iters"])
    k = pit <= READ_AT
    P = [np.asarray(p, dtype=float) for p, keep in zip(tr["participation"], k) if keep]
    it = np.asarray(tr["iters"])
    loss = np.asarray(tr["metrics"]["loss_clean_train"], dtype=float)
    m = it[:len(loss)] <= READ_AT
    return dict(pit=pit[k],
                sf=np.array([active_count(p, "scalefree") for p in P]),
                ab=np.array([active_count(p, SILENT_FLIPFLOP) for p in P]),
                it=it[:len(loss)][m], loss=loss[m])


def cell(path):
    """Every network's trace in a cell folder, as a list (empty if the folder is absent)."""
    return [t for t in (trace(d) for d in sorted(glob.glob(os.path.join(path, "*", "")))) if t]


def draw(ax, traces, colour, label, xkey, ykey, logy=False):
    """Plot one condition: each seed log-binned, the mean over seeds as the solid line."""
    if not traces:
        return
    curves = []
    for t in traces:
        pos = t[xkey] > 0                    # iteration 0 has no log-bin; logbin needs t[0] > 0
        x, y = logbin(t[xkey][pos], t[ykey][pos], nbins=60)
        if not len(x):
            continue
        curves.append((x, y))
        ax.plot(x, y, color=colour, lw=0.7, alpha=0.35, zorder=2)
    if not curves:
        return
    grid = curves[0][0]
    stack = [np.interp(grid, x, y) for x, y in curves]
    ax.plot(grid, np.mean(stack, axis=0), color=colour, lw=2, label=label, zorder=3)
    if logy:
        ax.set_yscale("log")


def main(root):
    """Draw the 3 x 4 figure (live scale-free / live 4e-2 / clean loss, one column per arm)."""
    fig, axes = plt.subplots(3, 4, figsize=(15, 8.4), sharex=True)
    for j, pen in enumerate(ARMS):
        # the no-dropout side pools the historical reference cell with the same-launcher
        # `do=none` controls, exactly as flipflop_dropout_readout.py does
        conds = [("-- none", cell(os.path.join(root, REFS[pen]))
                             + cell(os.path.join(root, DROP_SUB,
                                    f"EqType=h_k=3_N=1000_pen={pen}_do=none")))]
        for kind in ("mute", "dead"):
            conds.append((kind, cell(os.path.join(
                root, DROP_SUB, f"EqType=h_k=3_N=1000_pen={pen}_do={kind}"))))
        for name, ts in conds:
            lab = (f"no dropout ({len(ts)} seeds)" if name == "-- none"
                   else f"dropout: {name} ({len(ts)} seeds)")
            draw(axes[0, j], ts, COL[name], lab, "pit", "sf")
            draw(axes[1, j], ts, COL[name], lab, "pit", "ab")
            draw(axes[2, j], ts, COL[name], lab, "it", "loss", logy=True)
        axes[0, j].set_title(f"penalty: {pen}", fontsize=11)
        for i in range(3):
            axes[i, j].set_xscale("log")
            axes[i, j].grid(alpha=0.25, lw=0.5)
            axes[i, j].spines[["top", "right"]].set_visible(False)
            if i < 2:
                axes[i, j].set_ylim(0, 1050)
        # One no-dropout `none` seed blows up to L ~ 1e7 at ~3.5k and is rolled back by the
        # trainer's spike guard; unclipped it flattens every other curve. Same clip as
        # fig_dmts_pen_curves.py.
        axes[2, j].set_ylim(1e-2, 1.0)
        axes[2, j].set_xlabel("iteration")
    axes[0, 0].set_ylabel("live units (scale-free)")
    axes[1, 0].set_ylabel("live units (absolute 4e-2)")
    axes[2, 0].set_ylabel("clean training loss")
    axes[0, 0].legend(fontsize=8.5, frameon=False, loc="lower left")
    fig.suptitle("Dropout vs no dropout, 3-bit flip-flop, N=1000, 150k iterations: does dropout hold units open, "
                 "or only slow the silencing? (7 seeds per condition)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print("saved", os.path.normpath(OUT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
