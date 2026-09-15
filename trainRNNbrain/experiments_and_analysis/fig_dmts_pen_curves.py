"""Clean loss and live units against iteration for the DMTS_long penalty grid, one column per N.

Reads the npz written by `dmts_readout.py --dump` (per network: probe iterations, clean training
loss, silent_1em6 count) and draws, per N: top - clean loss (log-log) with the r2 = 0.9 level
marked; bottom - live units (N minus the 1e-6 silent count) on a log iteration axis. Four arms
in a fixed colour order (none, rws, frm, both), every seed as its own thin line: the rolling
median over 21 probes (210 iterations, the read-out's smoothing) drawn solid, the raw probe faint
behind it, so the brief memory-loss episodes (loss back at the plateau level) stay visible without
hiding the trend. The loss axis is clipped at 0.3 - one seed's gradient blow-up to 1e7 (recovered
by the trainer's rollback) would otherwise flatten everything. Output:
img/internal_figures/dmts_pen_curves.png.

Usage: python fig_dmts_pen_curves.py [data/dmts_curves_150k.npz]
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from trainRNNbrain.experiments_and_analysis.common import IMG_DIR
from trainRNNbrain.experiments_and_analysis.dmts_readout import smooth

PENS = ["none", "rws", "frm", "both"]
COLOR = {"none": "#2a78d6", "rws": "#eb6834", "frm": "#1baf7a", "both": "#eda100"}
LABEL = {"none": "none", "rws": "rws 0.05", "frm": "frm 0.1", "both": "frm + rws"}


def main(path):
    """Draw the figure from the curves npz at `path` and save it to IMG_DIR."""
    d = np.load(path)
    var = float(d["target_variance"])
    keys = sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_iters")})
    Ns = sorted({int(k.split("_")[0]) for k in keys})
    fig, axes = plt.subplots(2, len(Ns), figsize=(4.2 * len(Ns), 6.4), sharex=True, squeeze=False)
    for j, N in enumerate(Ns):
        ax_l, ax_u = axes[0, j], axes[1, j]
        for pen in PENS:
            first = True
            for k in keys:
                n, p, _ = k.split("_")
                if int(n) != N or p != pen:
                    continue
                it, L, s = d[k + "_iters"], d[k + "_loss"], d[k + "_silent"]
                it, L, s = it[it > 0], L[it > 0], s[it > 0]
                ax_l.plot(it, L, color=COLOR[pen], lw=0.4, alpha=0.18)
                ax_l.plot(it, smooth(L), color=COLOR[pen], lw=1.0, alpha=0.9,
                          label=LABEL[pen] if first else None)
                ax_u.plot(it, N - s, color=COLOR[pen], lw=0.4, alpha=0.18)
                ax_u.plot(it, smooth(N - s), color=COLOR[pen], lw=1.0, alpha=0.9,
                          label=LABEL[pen] if first else None)
                first = False
        ax_l.axhline(0.1 * var, color="0.6", lw=0.8, ls="--")
        ax_l.text(15, 0.1 * var * 1.15, r"clean $r^2 = 0.9$", color="0.4", fontsize=8)
        ax_l.set_xscale("log"); ax_l.set_yscale("log")
        ax_l.set_ylim(5e-7, 0.3)
        if j == 0:
            ax_l.legend(loc="lower left", fontsize=8, frameon=False, title="penalty", title_fontsize=8)
        ax_l.set_title(f"N = {N}", fontsize=10)
        ax_l.set_ylabel("clean loss (noise-free probe)" if j == 0 else "")
        ax_u.set_ylabel("live units ($p_i \\geq 10^{-6}$)" if j == 0 else "")
        ax_u.set_xlabel("iteration")
        ax_u.set_ylim(0, N * 1.05)
        for ax in (ax_l, ax_u):
            ax.grid(True, which="major", color="0.9", lw=0.6)
            ax.spines[["top", "right"]].set_visible(False)
    fig.suptitle("DMTS, 16-tau delay: penalty arms, 3 seeds each, 150k iterations", fontsize=10)
    fig.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, "dmts_pen_curves.png")
    fig.savefig(out, dpi=150)
    print("saved", os.path.abspath(out))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/dmts_curves_150k.npz")
