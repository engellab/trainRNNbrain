#!/usr/bin/env python3
"""Plot B3's training losses with the read-out's fitted floor, bar, crossing and bursts, and print why
each run is or is not read at its floor.

B3 is the N = 1000 flip-flop factorial of rate penalty (0.03) x synaptic noise (1.0) x rescaling
(read by read_b3_b4_4000.py). Per run this prints the production read-out (common.noisy_floor): the
fitted floor, the number of bursts and their share of training, the end-of-run level over the floor
(late_rise), the crossing at each margin, the reason a run gets no read, and the number of iterations
whose loss is not finite (noisy_floor drops them before it counts bursts). The control cell is left
out: its runs are warm-start chains whose parents a local mirror may lack, and read_b3_b4_4000.py
reads it on Della.

Usage:  python trainRNNbrain/experiments_and_analysis/plot_b3_loss_floor.py [TRAINED_RNNS_DIR]
        (default: the repository's data/trained_RNNs; writes img/internal_figures/b3_loss_at_floor_n1000.png)
"""

import glob
import os
import sys

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C

matplotlib.use("Agg")

CELLS = [("rate", "frm0.03"), ("noise", "sw1.0"), ("rescale", "rsc10"), ("rate+noise", "frm0.03_sw1.0"),
         ("rate+rescale", "frm0.03_rsc10"), ("noise+rescale", "sw1.0_rsc10"),
         ("rate+noise+rescale", "frm0.03_sw1.0_rsc10")]
GRID = "NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm="
OUT = os.path.join(C.IMG_DIR, "b3_loss_at_floor_n1000.png")


def diagnose(run_dir):
    """Run the production floor read-out on one run.

    Args:
        run_dir: str, a run folder holding *TrainLosses.json.
    Returns:
        (it, L, nf, line): iterations and per-iteration noisy loss (float arrays, non-finite values
        kept), the common.noisy_floor dict, and a one-line text summary.
    """
    it, L = C.noisy_loss_series(run_dir)
    nf = C.noisy_floor(it, L, task=C.task_of(run_dir))
    fin = np.isfinite(L)
    share = sum(b1 - b0 + 1 for b0, b1, _ in nf["bursts"]) / max(it[-1] - nf["burst_from"] + 1, 1)
    floor = "none" if nf["floor"] is None else f"{nf['floor']:.4f}"
    cross = {k: (int(v) if np.isfinite(v) else None) for k, v in nf["t_stars"].items()}
    line = (f"floor={floor} bursts={len(nf['bursts'])} ({share:.1%}) late_rise={nf['late_rise']:.3f} "
            f"non-finite={int((~fin).sum())} crossings={cross} reason={nf['reason']}")
    return it, L, nf, line


def main():
    """Print one line per B3 remedy run and save the 7 x 3 grid of loss curves."""
    root = sys.argv[1] if len(sys.argv) > 1 else C.DATA_DIR
    fig = plt.figure(figsize=(15, 23))
    gs = fig.add_gridspec(len(CELLS), 3, left=0.05, right=0.99, bottom=0.03, top=0.98, hspace=0.35, wspace=0.18)
    for i, (cell, arm) in enumerate(CELLS):
        runs = sorted(d for d in glob.glob(os.path.join(root, GRID + arm, "*")) if os.path.isdir(d))
        for j, rd in enumerate(runs[:3]):
            seed = rd.split("seed=")[1].split(";")[0]
            it, L, nf, line = diagnose(rd)
            tag = " (diverged, nan_ folder)" if os.path.basename(rd).startswith("nan") else ""
            print(f"{cell:<19s} s{seed}{tag}: {line}")
            fin = np.isfinite(L)
            ax = fig.add_subplot(gs[i, j])
            ax.loglog(it[fin], L[fin], lw=0.2, color="0.75")
            ax.loglog(it[fin], nf["smoothed"], lw=1, color="k")
            if nf["floor"]:
                ax.axhline(nf["floor"], color="tab:blue", lw=0.8)
                ax.axhline(1.05 * nf["floor"], color="tab:red", lw=0.8, ls="--")
            if np.isfinite(nf["t_star"]):
                ax.axvline(nf["t_star"], color="tab:green", lw=0.8)
            for b0, b1, _ in nf["bursts"]:
                ax.axvspan(b0, b1, color="orange", alpha=0.3, lw=0)
            lo = np.nanpercentile(L[fin], 0.5)
            ax.set_ylim(lo * 0.8, max(np.median(L[fin][:50]) * 2, lo * 10))
            ax.set_title(f"{cell} s{seed}{tag}: {nf['reason'] or 'read at %.0f' % nf['t_star']}", fontsize=8)
            ax.set_xlabel("iteration", fontsize=7)
            ax.set_ylabel("noisy training loss", fontsize=7)
            ax.tick_params(labelsize=7)
    fig.savefig(OUT, dpi=90)
    print("wrote", OUT)


if __name__ == "__main__":
    main()
