#!/usr/bin/env python3
"""Internal figure: criticality measures of the default Maze networks over training, and lambda1 on all three
tasks (rules docs/project_trajectory/2026-11-20_1733_rules-criticality-maze.md).

Top row, Maze N = 1,000 (three networks): largest Lyapunov exponent lambda1, gradient propagation over 500 and
over 100 RNN steps, active units, against the PPO update (0 = init). The filled marker on each line is the
snapshot the gate reads (at or just after the learned point). Bottom left: lambda1 at init and at the
learned point (floor crossing on 3BFF and CDDM) for every network of the three tasks. Bottom right, Maze:
spectral radius of W_rec, fraction of units above threshold, median spectral radius of the step Jacobian.

Usage: python fig_criticality_maze.py --maze <criticality_maze json>... --sup data/criticality/phase1_3bff.json
           data/criticality/phase1_cddm.json --gate-rows data/criticality/gate_rows.json --out <png>
"""

import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ("#2a78d6", "#eb6834", "#1baf7a")     # categorical slots 1-3 of the dataviz reference palette
COLS = (("lyap_med", "largest Lyapunov exponent λ1\n(per RNN step)"),
        ("gain500_med", "gradient propagation over\n500 steps, log10 G500"),
        ("gain100_med", "gradient propagation over\n100 steps, log10 G100"),
        ("active", "active units (of 1,000)"))
COLS2 = (("rho_W", "spectral radius of W_rec"),
         ("gate_on", "fraction of units above\nthreshold per step"),
         ("oprad_med", "median spectral radius\nof the step Jacobian"))
TASK_COLORS = {"3BFF": "#7a5cc4", "CDDM": "#c4a12a", "Maze": "#2a78d6"}


def maze_panels(axes, rows, cols=COLS, legend=True):
    """Draw Maze measures over training, one axis per measure.

    Args:
        axes: matplotlib axes, one per entry of cols; rows: criticality_maze.py rows;
        cols: tuple of (row key, y label); legend: bool, draw the legend on the last axis.
    Returns:
        None.
    """
    for j, net in enumerate(sorted({r["network"] for r in rows})):
        R = sorted((r for r in rows if r["network"] == net), key=lambda r: r["iteration"])
        r0 = R[0]
        for ax, (key, ylab) in zip(axes, cols):
            ax.plot([r["iteration"] for r in R], [r[key] for r in R], color=COLORS[j], lw=2, marker=".",
                    label=f"s{r0['seed']} (learned at {r0['learned_update']})")
            rt = next(r for r in R if r["kind"] == "snapshot" and r["iteration"] == r0["learned_snapshot"])
            ax.plot([rt["iteration"]], [rt[key]], "o", ms=9, color=COLORS[j], mec="white", mew=2)
    for ax, (key, ylab) in zip(axes, cols):
        ax.set_ylabel(ylab, fontsize=9)
        ax.set_xlabel("PPO update (0 = init)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.set_xlim(-100, 4100)
        if key == "lyap_med":
            ax.axhline(0, color="0.4", lw=1, ls="--")
            ax.axhline(math.log(0.9), color="0.4", lw=1, ls=":")
            ax.set_ylim(-0.11, 0.014)
            ax.text(150, 0.002, "edge of chaos (λ1 = 0)", fontsize=8, color="0.3", va="bottom")
            ax.text(150, math.log(0.9) + 0.003, "every unit inactive (log 0.9)", fontsize=8, color="0.3")
    if legend:
        axes[-1].legend(title="network (seed)", fontsize=8, title_fontsize=8, frameon=False)


def cross_panel(ax, rows, sup, it_read):
    """lambda1 at init and at the learned point (or floor crossing), one line per network, per task.

    Args:
        ax: matplotlib axis; rows: Maze rows; sup: supervised criticality_read rows; it_read: dict network ->
        iteration read at the floor crossing.
    Returns:
        None.
    """
    pts = []
    for net, it in it_read.items():
        R = [r for r in sup if r["network"] == net]
        r0 = next(r for r in R if r["kind"] == "init")
        rt = next(r for r in R if r["iteration"] == it and r["kind"] != "init")
        pts.append(({"NBitFlipFlop": "3BFF"}.get(r0["task"], r0["task"]), r0["lyap_med"], rt["lyap_med"]))
    for net in sorted({r["network"] for r in rows}):
        R = [r for r in rows if r["network"] == net]
        r0 = next(r for r in R if r["kind"] == "init")
        rt = next(r for r in R if r["kind"] == "snapshot" and r["iteration"] == r0["learned_snapshot"])
        pts.append(("Maze", r0["lyap_med"], rt["lyap_med"]))
    order = ("CDDM", "3BFF", "Maze")
    for task, l0, lt in pts:
        x = order.index(task)
        ax.plot([x - 0.15, x + 0.15], [l0, lt], color=TASK_COLORS[task], lw=1.5, marker="o", ms=5)
    ax.axhline(0, color="0.4", lw=1, ls="--")
    ax.set_xticks(range(3))
    ax.set_xticklabels([f"{t}\ninit → learned" for t in order], fontsize=9)
    ax.set_ylabel("λ1 (per RNN step)", fontsize=9)
    ax.set_xlim(-0.5, 2.5)
    ax.grid(alpha=0.25, lw=0.5, axis="y")
    ax.set_title("λ1 at init and at the learned point\n(3BFF, CDDM: floor crossing)", fontsize=9, loc="left")


def main():
    """Draw the figure and save it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--maze", nargs="+", required=True)
    ap.add_argument("--sup", nargs="+", required=True)
    ap.add_argument("--gate-rows", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for f in a.maze for r in json.load(open(f))]
    sup = [r for f in a.sup for r in json.load(open(f))]
    fig = plt.figure(figsize=(15, 7.4), constrained_layout=True)
    gs = fig.add_gridspec(2, 4)
    maze_panels([fig.add_subplot(gs[0, c]) for c in range(4)], rows)
    cross_panel(fig.add_subplot(gs[1, 0]), rows, sup, json.load(open(a.gate_rows)))
    maze_panels([fig.add_subplot(gs[1, c]) for c in range(1, 4)], rows, COLS2, legend=False)
    fig.suptitle("Default Maze networks, N = 1,000: criticality measures over training "
                 "(filled marker: snapshot at or after the learned point)", fontsize=11)
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
