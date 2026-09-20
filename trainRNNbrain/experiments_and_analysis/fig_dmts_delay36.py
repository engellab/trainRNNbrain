"""DMTS with a 36-tau delay (T=500): does the unpenalised network find the memory at all?

Pavel's check (job 6201936, 2 nets, N=1000, one seed per arm). At a 16-tau delay both arms escape
the no-memory plateau and the rate penalty only wins on SPEED; the prediction was that at 36 tau
the unpenalised network would not escape at all within 150k while frm would.

Plots clean r2 = 1 - loss_clean_train / Var(target over the scored steps) against iteration, which
is the quantity the prediction is about, rather than the raw loss. Raw probes are drawn as faint
points so the memory-loss EPISODES are visible (the 16-tau grid found every arm has them), with a
rolling-median line over them. The r2 = 0.9 escape criterion and the no-memory plateau level are
marked. The right panel is live units, on the same iteration axis.

Reads the npz written by `dmts_readout.py --sub DMTS_std_delay36 --dump ...`.
Usage: python fig_dmts_delay36.py [data/dmts_curves_delay36.npz]
Writes img/internal_figures/dmts_delay36.png
"""
import os
import sys

import matplotlib
matplotlib.use("Agg")
import numpy as np
from matplotlib import pyplot as plt

from trainRNNbrain.experiments_and_analysis.common import IMG_DIR
from trainRNNbrain.experiments_and_analysis.dmts_readout import smooth

COLOR = {"none": "#2a78d6", "frm": "#1baf7a", "both": "#eda100"}
LABEL = {"none": "no penalty", "frm": "frm 0.1", "both": "frm 0.1 + rws 0.05"}
ESCAPE = 0.9
OUT = os.path.join(IMG_DIR, "dmts_delay36.png")


def main(path):
    """Draw the two-panel 36-tau figure from the curves npz at `path`."""
    d = np.load(path)
    var = float(d["target_variance"])
    keys = sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_iters")})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(13, 4.8))
    summary, drawn = [], set()
    for k in keys:
        N, pen, _ = k.split("_")
        N = int(N)
        it, L, s = d[k + "_iters"], d[k + "_loss"], d[k + "_silent"]
        r2 = 1.0 - L / var
        ax.plot(it, r2, color=COLOR[pen], lw=0.5, alpha=0.12)            # raw probes: the episodes
        seen = pen in drawn
        ax.plot(it, smooth(r2, 201), color=COLOR[pen], lw=1.8, label=None if seen else LABEL[pen])
        ax2.plot(it, N - s, color=COLOR[pen], lw=1.4, label=None if seen else LABEL[pen])
        drawn.add(pen)
        above = r2 >= ESCAPE
        first = int(it[np.flatnonzero(above)[0]]) if above.any() else None
        half = it > it[-1] / 2
        summary.append((pen, first, float(np.mean(above[half])), float(r2[half].max())))
        if pen == "none" and "plateau" not in dir():
            # The plateau level is READ OFF the unpenalised arm, which never leaves it, rather
            # than carried over from the 16-tau grid (whose plateau loss is a different number).
            plateau = float(np.median(r2[half]))

    ax.axhline(ESCAPE, color="0.4", lw=0.9, ls="--")
    ax.text(150, ESCAPE + 0.02, r"escape criterion, clean $r^2 = 0.9$", fontsize=8, color="0.35")
    ax.axhline(plateau, color="0.6", lw=0.9, ls=":")
    ax.text(155, plateau - 0.04, f"no-memory plateau, clean $r^2$ = {plateau:.3f}\n"
            "reads the decision cue, carries\nnothing across the delay",
            fontsize=8, color="0.45", va="top", ha="left")
    ax.set_xscale("log"); ax.set_xlim(120, 1.6e5); ax.set_ylim(-0.15, 1.05)
    ax.set_xlabel("iteration"); ax.set_ylabel("clean $r^2$ (noise-free probe)")
    ax.set_title("Does the memory appear?", fontsize=10)
    ax.legend(loc="lower left", fontsize=9, frameon=False)

    ax2.set_xscale("log"); ax2.set_xlim(120, 1.6e5); ax2.set_ylim(0, 1050)
    ax2.set_xlabel("iteration"); ax2.set_ylabel(r"live units ($p_i \geq 10^{-6}$)")
    ax2.set_title("Live units", fontsize=10)
    ax2.legend(loc="lower left", fontsize=9, frameon=False)
    for a in (ax, ax2):
        a.grid(True, which="major", color="0.9", lw=0.6)
        a.spines[["top", "right"]].set_visible(False)

    fig.suptitle("DMTS, 36-tau delay (T=500), N=1000, 150k iterations, 3 seeds per arm:\n"
                 "frm alone finds the memory in 3/3 seeds; adding rws makes it 0/3, "
                 "like no penalty at all", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved", os.path.abspath(OUT))
    for pen, first, frac, best in summary:
        print(f"  {LABEL[pen]:12} first r2>=0.9 at {str(first) if first else 'NEVER':>8};  "
              f"fraction of the second half above 0.9 = {frac:.3f};  best r2 in second half {best:.4f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/dmts_curves_delay36.npz")
