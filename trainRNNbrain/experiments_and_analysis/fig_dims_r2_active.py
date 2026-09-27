"""Dimensionality, task performance and active-unit count, one point per trained network.

Three quantities, so three pairwise panels per figure. Every point is one network, gated on a
recomputed r2 that reproduces the value stored at training time.

Dimensionality is the participation ratio of the firing-rate covariance over the ACTIVE units,
(sum ev)^2 / sum ev^2: the number of directions the population actually uses. Reported noise-free,
which is the signal the network computes with rather than the variance its own noise injects.

Three figures:
  fig_dims_r2_active_rules.png  the revival rules at N=1000, every setting matched but the rule
  fig_dims_r2_active_sizes.png  control / duplication / mute / dead dropout across N=500-4000
  fig_dims_r2_active_copy.png   what a copy inherits, from iid weights through to an exact copy
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

DATA = "/private/tmp/claude-502/-Users-pt1290-Documents-GitHub-trainRNNbrain/1bc3843f-10a1-471f-875d-5f58458e098b/scratchpad/all_dims.json"
OUTDIR = "img/internal_figures"

# Okabe-Ito, which stays separable under the common forms of colour blindness
OI = {"black": "#000000", "orange": "#E69F00", "sky": "#56B4E9", "green": "#009E73",
      "yellow": "#F0E442", "blue": "#0072B2", "vermillion": "#D55E00", "purple": "#CC79A7",
      "grey": "#999999"}

RULES = [("control g0", OI["black"], "o"), ("random", OI["purple"], "v"),
         ("zero_out", OI["yellow"], "v"), ("orth", OI["blue"], "v"),
         ("mix", OI["green"], "v"), ("rescale", OI["sky"], "v"),
         ("bias_kick", OI["vermillion"], "s"), ("copy jitter 0", OI["orange"], "D")]

SIZES = [("control", OI["black"], "o"), ("duplication", OI["orange"], "D"),
         ("mute dropout", OI["blue"], "s"), ("dead dropout", OI["vermillion"], "^")]

COPY = [("control g0", OI["black"], "o"), ("copy iid", OI["sky"], "v"),
        ("copy permuted", OI["green"], "s"), ("copy jitter 0", OI["orange"], "D"),
        ("copy jitter 0.05", OI["yellow"], "D"), ("copy jitter 0.3", OI["purple"], "D"),
        ("copy jitter 1.0", OI["blue"], "D"), ("copy jitter 3.0", OI["vermillion"], "D")]


def panel(ax, rows, series, xk, yk, xlabel, ylabel, size_by_N=False):
    """One pairwise scatter, one marker per network.

    Args:
        ax: matplotlib axes.
        rows: list of per-network dicts.
        series: list of (arm label, colour, marker).
        xk, yk: keys into each row for the two axes.
        xlabel, ylabel: axis labels.
        size_by_N: scale the marker with network size, for the size-series figure.
    """
    for arm, col, mk in series:
        s = [r for r in rows if r["arm"] == arm]
        if not s:
            continue
        x = np.array([r[xk] for r in s], float)
        y = np.array([r[yk] for r in s], float)
        sz = (np.array([r["N"] for r in s], float) / 25.0 + 30) if size_by_N else \
            np.full(len(s), 80.0)
        ax.scatter(x, y, s=sz, color=col, marker=mk, zorder=3, edgecolor="white", linewidth=1.0)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(color="0.93", lw=0.8, zorder=0)


def figure(rows, series, title, fname, size_by_N=False):
    """A three-panel figure covering every pair of the three quantities.

    Args:
        rows: list of per-network dicts.
        series: list of (arm label, colour, marker).
        title: figure suptitle.
        fname: output file name inside OUTDIR.
        size_by_N: scale markers with network size.
    """
    fig, axes = plt.subplots(1, 3, figsize=(15.5, 4.9))
    panel(axes[0], rows, series, "n_active", "dims_clean",
          "active units", "dimensions used", size_by_N)
    panel(axes[1], rows, series, "n_active", "recomputed",
          "active units", "held-out $r^2$", size_by_N)
    panel(axes[2], rows, series, "recomputed", "dims_clean",
          "held-out $r^2$", "dimensions used", size_by_N)
    handles = [plt.Line2D([], [], color=c, marker=m, ls="none", ms=8,
                          markeredgecolor="white", label=a) for a, c, m in series
               if any(r["arm"] == a for r in rows)]
    axes[0].legend(handles=handles, frameon=False, fontsize=8.5, loc="best", ncol=2)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    os.makedirs(OUTDIR, exist_ok=True)
    fig.savefig(os.path.join(OUTDIR, fname), dpi=160)
    print(f"wrote {OUTDIR}/{fname}")


def table(rows, series, label):
    """Per-cell means, printed so the figure can be checked against numbers."""
    print(f"\n{label}")
    print(f"{'arm':>17s} {'N':>5s} {'n':>2s} {'active':>7s} {'dims':>7s} {'r2':>8s}")
    print("-" * 52)
    for arm, _, _ in series:
        for N in sorted({r["N"] for r in rows if r["arm"] == arm}):
            s = [r for r in rows if r["arm"] == arm and r["N"] == N]
            if not s:
                continue
            print(f"{arm:>17s} {N:5d} {len(s):2d} "
                  f"{np.mean([r['n_active'] for r in s]):7.0f} "
                  f"{np.mean([r['dims_clean'] for r in s]):7.2f} "
                  f"{np.mean([r['recomputed'] for r in s]):8.4f}")


if __name__ == "__main__":
    rows = json.load(open(DATA))
    n1000 = [r for r in rows if r["N"] == 1000]
    figure(n1000, RULES,
           "Revival rules at N = 1000: every setting matched except the rule",
           "fig_dims_r2_active_rules.png")
    figure(rows, SIZES,
           "Across network size: marker size is N (500, 1000, 2000, 4000)",
           "fig_dims_r2_active_sizes.png", size_by_N=True)
    figure(n1000, COPY,
           "What a copy inherits: from iid weights, through scrambled and jittered, to exact",
           "fig_dims_r2_active_copy.png")
    table(n1000, RULES, "Revival rules, N=1000")
    table(rows, SIZES, "Size series")
    table(n1000, COPY, "Copy decomposition, N=1000")
