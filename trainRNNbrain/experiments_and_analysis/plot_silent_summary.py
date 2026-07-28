#!/usr/bin/env python3
"""
Bar-chart summaries of silent-unit counts, from the per-net CSV written by collect_stats.py.

Three figures:
  1. silent_vs_N_std.png                      size sweep in standard RNNs, h and s separately,
                                              both the strict (p < 1e-6) and scale-free metrics
  2. silent_by_penalty_N1000_std.png          the four penalty configurations at N=1000
  3. silent_constrained_vs_unconstrained_h.png  h equation: constrained vs standard architecture

Metrics (participation p = std(fr) + q0.9(|fr|) over time and conditions):
  hard_1em6 : p < 1e-6            -- truly silent; a ReLU unit that never fires has p = 0
  rel_5p95  : p < 0.05 * p95(p)   -- scale-free, comparable across conditions of different scale

Usage:  python plot_silent_summary.py [CSV] [OUTDIR]
"""

import os
import sys
import csv
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt

# Okabe-Ito subset, fixed order = none, rws, frm, both. Validated for CVD separation and lightness;
# the sub-3:1 contrast of the warm hues is relieved by the direct value labels drawn on every bar.
PAL = {"none": "#0072B2", "rws": "#009E73", "frm": "#E69F00", "both": "#CC79A7"}
ARCH = {"constrained": "#0072B2", "standard": "#E69F00"}
METRIC_LABEL = {"hard_1em6": "truly silent  (p < 1e-6)",
                "rel_5p95": "silent, scale-free  (p < 5% of p95)"}


def load(path):
    """Read the per-net CSV into a dict keyed by (sweep, eq, N, penalty) -> list of row dicts."""
    d = defaultdict(list)
    with open(path) as f:
        for r in csv.DictReader(f):
            d[(r["sweep"], r["eq"], int(r["N"]), r["penalty"])].append(
                {k: float(v) for k, v in r.items() if k not in ("sweep", "eq", "N", "penalty")})
    return d


def mean_std(rows, metric):
    """Mean and std (over networks) of one metric, as percentages."""
    v = np.array([r[metric] for r in rows]) * 100
    return v.mean(), v.std()


def style(ax, ylabel):
    """Apply the shared recessive-axis style."""
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="0.9", lw=0.8)
    ax.set_axisbelow(True)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 100)


def label_bars(ax, bars, vals, errs=None):
    """Direct value labels, placed clear of the error-bar cap.

    Also the accessibility relief required for the sub-3:1 hues in the palette.
    """
    errs = errs if errs is not None else [0] * len(vals)
    for b, v, e in zip(bars, vals, errs):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + e + 2.5, f"{v:.0f}",
                ha="center", va="bottom", fontsize=7, color="0.25")


def fig_size_sweep(d, out):
    """Silent fraction vs network size, standard RNNs, none vs frm, both metrics."""
    Ns, pens = [100, 250, 500, 1000], ["none", "frm"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7), sharey=True)
    for i, eq in enumerate(("h", "s")):
        for j, metric in enumerate(("hard_1em6", "rel_5p95")):
            ax = axes[i, j]
            w = 0.38
            for k, pen in enumerate(pens):
                m, s, xs = [], [], []
                for x, N in enumerate(Ns):
                    rows = d.get(("std_Nsweep", eq, N, pen)) or d.get(("std", eq, N, pen))
                    if not rows:
                        continue
                    mu, sd = mean_std(rows, metric)
                    m.append(mu); s.append(sd); xs.append(x + (k - 0.5) * w)
                bars = ax.bar(xs, m, w, yerr=s, capsize=2, color=PAL[pen],
                              label=pen, edgecolor="white", linewidth=1.2)
                label_bars(ax, bars, m, s)
            ax.set_xticks(range(len(Ns))); ax.set_xticklabels([str(n) for n in Ns])
            ax.set_xlabel("network size N")
            style(ax, "% of units silent" if j == 0 else "")
            ax.set_title(f"{eq} equation — {METRIC_LABEL[metric]}", fontsize=9)
    axes[0, 0].legend(frameon=False, fontsize=8, title="penalty", title_fontsize=8)
    fig.suptitle("Standard RNNs: silent units grow with network size; frm removes them at every size",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def fig_penalties(d, out):
    """The four penalty configurations at N=1000, standard RNNs, h and s separately."""
    pens = ["none", "rws", "frm", "both"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for i, eq in enumerate(("h", "s")):
        ax = axes[i]
        w = 0.38
        for j, metric in enumerate(("hard_1em6", "rel_5p95")):
            m, s, xs = [], [], []
            for x, pen in enumerate(pens):
                rows = d.get(("std", eq, 1000, pen))
                mu, sd = mean_std(rows, metric)
                m.append(mu); s.append(sd); xs.append(x + (j - 0.5) * w)
            bars = ax.bar(xs, m, w, yerr=s, capsize=2,
                          color=("#0072B2" if j == 0 else "#009E73"),
                          label=METRIC_LABEL[metric], edgecolor="white", linewidth=1.2)
            label_bars(ax, bars, m, s)
        ax.set_xticks(range(len(pens))); ax.set_xticklabels(pens)
        ax.set_xlabel("penalty configuration")
        style(ax, "% of units silent" if i == 0 else "")
        ax.set_title(f"{eq} equation, N=1000", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8)
    fig.suptitle("Standard RNNs, N=1000: only the firing-rate penalty removes silent units", fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def fig_arch(d, out):
    """h equation: constrained (Dale + I/O positivity) vs standard architecture."""
    pens = ["none", "rws", "frm", "both"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for j, metric in enumerate(("hard_1em6", "rel_5p95")):
        ax = axes[j]
        w = 0.38
        for k, (sweep, lab) in enumerate((("dale", "constrained"), ("std", "standard"))):
            m, s, xs = [], [], []
            for x, pen in enumerate(pens):
                rows = d.get((sweep, "h", 1000, pen))
                mu, sd = mean_std(rows, metric)
                m.append(mu); s.append(sd); xs.append(x + (k - 0.5) * w)
            bars = ax.bar(xs, m, w, yerr=s, capsize=2, color=ARCH[lab], label=lab,
                          edgecolor="white", linewidth=1.2)
            label_bars(ax, bars, m, s)
        ax.set_xticks(range(len(pens))); ax.set_xticklabels(pens)
        ax.set_xlabel("penalty configuration")
        style(ax, "% of units silent" if j == 0 else "")
        ax.set_title(METRIC_LABEL[metric], fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, title="architecture", title_fontsize=8)
    fig.suptitle("h equation, N=1000: the effect is not an artifact of the biological constraints",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)


def main():
    """Generate all three summary figures."""
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/trained_RNNs/silent_stats_all.csv"
    outdir = sys.argv[2] if len(sys.argv) > 2 else "img/internal_figures"
    os.makedirs(outdir, exist_ok=True)
    d = load(csv_path)
    fig_size_sweep(d, os.path.join(outdir, "silent_vs_N_std.png"))
    fig_penalties(d, os.path.join(outdir, "silent_by_penalty_N1000_std.png"))
    fig_arch(d, os.path.join(outdir, "silent_constrained_vs_unconstrained_h.png"))


if __name__ == "__main__":
    main()
