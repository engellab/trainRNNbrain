#!/usr/bin/env python3
"""
Population-level statistics of networks that solve the task equally well.

Four panels, each a statistic that is routinely computed on model RNNs and compared against neural
recordings: effective dimensionality, choice selectivity, total metabolic cost, and how concentrated
that cost is across units. If these differ between networks with indistinguishable task performance,
then the scientific conclusion drawn from them depends on a training choice nobody reports.

Usage:  python plot_population_distortion.py [CSV] [OUT]
"""

import sys
import csv
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from collections import defaultdict

PENS = ["none", "rws", "frm", "both"]
EQ_COL = {"h": "#0072B2", "s": "#E69F00"}     # CVD-validated pair
T95 = 2.776                                    # two-sided 95%, n=5


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "data/trained_RNNs/population_distortion.csv"
    out = sys.argv[2] if len(sys.argv) > 2 else "img/internal_figures/population_distortion.png"
    d = defaultdict(list)
    for r in csv.DictReader(open(src)):
        d[(r["eq"], r["penalty"])].append(r)

    panels = [("pr", "effective dimensionality\n(participation ratio)", 1.0),
              ("sel_choice", "units selective to choice (%)", 100.0),
              ("energy", "total metabolic cost\n$\\sum_i \\langle r_i^2 \\rangle$", 1.0),
              ("energy_hhi", "concentration of that cost\n(HHI; lower = more even)", 1.0)]
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    for ax, (field, label, scale) in zip(axes, panels):
        w = 0.38
        for j, eq in enumerate(("h", "s")):
            m, h, xs = [], [], []
            for x, pen in enumerate(PENS):
                v = np.array([float(r[field]) for r in d[(eq, pen)]]) * scale
                m.append(v.mean()); h.append(T95 * v.std(ddof=1) / np.sqrt(len(v)))
                xs.append(x + (j - 0.5) * w)
            bars = ax.bar(xs, m, w, yerr=h, capsize=2, color=EQ_COL[eq], label=f"{eq} equation",
                          edgecolor="white", linewidth=1.2)
            for b, val, e in zip(bars, m, h):
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e,
                        f"{val:.3g}", ha="center", va="bottom", fontsize=7, color="0.25")
        ax.set_xticks(range(len(PENS))); ax.set_xticklabels(PENS)
        ax.set_xlabel("penalty")
        ax.set_ylabel(label, fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(axis="y", color="0.92", lw=0.8); ax.set_axisbelow(True)
        if field == "energy_hhi":
            ax.set_yscale("log")
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Networks with indistinguishable task performance (R² = 0.84–0.87) have very "
                 "different population statistics — N = 1000, 5 nets per cell, 95% CI", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


if __name__ == "__main__":
    main()
