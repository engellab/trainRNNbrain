"""Hyper flip-flop grid at 150k: live units and r^2 vs k, none vs both, one panel per N.

Reads the per-net table printed by `flipflop_hyper_readout.py --seeds` (saved as
data/hyper_readout_150k_seeds.txt; the traces live on Spock) and draws two rows x three columns:
live units (scale-free criterion) and validation r^2 against k, for N = 500 / 1000 / 2000, with the
plain flip-flop (unpenalised, same k, N and iteration) as the grey reference. Every seed is a small
marker (jittered in k); the line joins the per-cell means.

Usage: python fig_hyper_readout.py [data/hyper_readout_150k_seeds.txt]
"""
import os
import sys
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COL = {"hyper none": "#2a78d6", "hyper both": "#eb6834", "plain none": "#7a7a72"}
LS = {"hyper none": "-", "hyper both": "-", "plain none": "--"}   # the grey reference is also dashed, so identity is never colour alone
LABEL = {"hyper none": "hyper, no penalty", "hyper both": "hyper, frm + rws", "plain none": "plain flip-flop, no penalty"}
NS = (500, 1000, 2000)
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "img", "internal_figures", "hyper_readout_150k.png")


def load(path):
    """Parse the per-net table into {(task, pen, k, N): list of (live_sf, r2)}."""
    rows = {}
    for line in open(path):
        m = re.match(r"(hyper|plain)\s+(none|both)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+([\d.-]+)", line)
        if m:
            task, pen, k, N, sf, ab, r2 = m.groups()
            rows.setdefault((task, pen, int(k), int(N)), []).append((int(sf), float(r2)))
    return rows


def main(path):
    """Draw and save the figure."""
    rows = load(path)
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 6.4), sharex=True)
    for j, N in enumerate(NS):
        for series in ("plain none", "hyper none", "hyper both"):
            task, pen = series.split()
            ks = sorted(k for (t, p, k, n) in rows if t == task and p == pen and n == N and k in (2, 4, 6, 8))
            if not ks:
                continue
            for row, col in ((0, 0), (1, 1)):
                ax = axes[row, j]
                means = [np.mean([v[col] for v in rows[(task, pen, k, N)]]) for k in ks]
                ax.plot(ks, means, color=COL[series], lw=2, ls=LS[series], label=LABEL[series], zorder=2)
                for k in ks:
                    seeds = [v[col] for v in rows[(task, pen, k, N)]]
                    jit = np.linspace(-0.12, 0.12, len(seeds)) if len(seeds) > 1 else np.zeros(1)
                    ax.scatter(k + jit, seeds, s=22, color=COL[series], edgecolor="white", lw=0.6, zorder=3)
        axes[0, j].axhline(N, color="#7a7a72", lw=0.8, ls=":")
        axes[0, j].text(8.1, N, f"N = {N}", va="bottom", ha="right", fontsize=8, color="#555")
        axes[0, j].set_title(f"N = {N}", fontsize=11)
        axes[0, j].set_ylim(0, N * 1.08)
        axes[1, j].set_ylim(0.25, 1.0)
        axes[1, j].set_xlabel("k (bits); hyper read-out has 2^k − 1 channels")
        for ax in axes[:, j]:
            ax.set_xticks((2, 4, 6, 8)); ax.grid(alpha=0.25, lw=0.5); ax.spines[["top", "right"]].set_visible(False)
    axes[0, 0].set_ylabel("live units at 150k (scale-free)")
    axes[1, 0].set_ylabel("validation r²")
    axes[1, 0].legend(loc="lower left", fontsize=9, frameon=False)
    fig.suptitle("Hyper flip-flop grid, 150k iterations, every seed a point, line = mean: recruitment and performance vs task demand", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT, dpi=120)
    print("saved", os.path.normpath(OUT))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "data/hyper_readout_150k_seeds.txt")
