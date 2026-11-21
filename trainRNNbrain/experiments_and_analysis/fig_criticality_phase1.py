#!/usr/bin/env python3
"""Internal figure: criticality measures of the claim-1 controls over training (Phase 1 of the rules
docs/project_trajectory/2026-11-20_1643_rules-criticality-projection.md).

Rows: CDDM, 3BFF (three networks each, N = 1,000). Columns: largest Lyapunov exponent lambda1 (median over
trials), gradient propagation over 100 steps G100, spectral radius of W_rec, active units. The filled marker
on each line is the network's floor crossing (the weights the gate reads).

Usage: python fig_criticality_phase1.py --json <criticality_read json>... --gate <gate_rows.json> --out <png>
"""

import argparse
import json
import math

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

COLORS = ("#2a78d6", "#eb6834", "#1baf7a")     # categorical slots 1-3 of the dataviz reference palette
COLS = (("lyap_med", "largest Lyapunov exponent λ1\n(per step)"),
        ("gain100_med", "gradient propagation over\n100 steps, log10 G100"),
        ("rho_W", "spectral radius of W_rec"),
        ("active", "active units (of 1,000)"))
TASKS = (("CDDM", "CDDM"), ("NBitFlipFlop", "3BFF"))


def main():
    """Draw the 2 x 4 grid and save it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", nargs="+", required=True)
    ap.add_argument("--gate", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    rows = [r for f in a.json for r in json.load(open(f))]
    gate_it = json.load(open(a.gate))
    fig, axes = plt.subplots(2, 4, figsize=(15, 6.4), constrained_layout=True)
    for i, (task, label) in enumerate(TASKS):
        nets = sorted({r["network"] for r in rows if r["task"] == task})
        for j, net in enumerate(nets):
            R = sorted((r for r in rows if r["network"] == net),
                       key=lambda r: (r["iteration"], r["kind"] != "init"))
            R = [r for k, r in enumerate(R) if k == 0 or r["iteration"] != R[k - 1]["iteration"]]
            x = [r["iteration"] for r in R]
            for c, (key, ylab) in enumerate(COLS):
                ax = axes[i, c]
                ax.plot(x, [r[key] for r in R], color=COLORS[j], lw=2, label=f"s{R[0]['seed']}")
                rt = next(r for r in R if r["iteration"] == gate_it[net])
                ax.plot([rt["iteration"]], [rt[key]], "o", ms=8, color=COLORS[j], mec="white", mew=2)
        for c, (key, ylab) in enumerate(COLS):
            ax = axes[i, c]
            ax.set_xscale("symlog", linthresh=500)
            ax.set_xlim(0, 2.1e5 if task == "NBitFlipFlop" else 1.05e5)
            ax.set_xticks([0, 1e3, 1e4, 1e5])
            ax.grid(alpha=0.25, lw=0.5)
            ax.set_ylabel(ylab, fontsize=9)
            if key == "lyap_med":
                ax.axhline(0, color="0.4", lw=1, ls="--")
                ax.axhline(math.log(0.9), color="0.4", lw=1, ls=":")
                ax.set_ylim(-0.11, 0.014)
                ax.text(600, 0.002, "edge of chaos (λ1 = 0)", fontsize=8, color="0.3", va="bottom")
                ax.text(600, math.log(0.9) + 0.003, "every unit inactive (log 0.9)", fontsize=8, color="0.3")
            if i == 1:
                ax.set_xlabel("training iteration (0 = init)")
        axes[i, 0].set_title(label, loc="left", fontweight="bold")
        axes[i, 3].legend(title="network (seed)", fontsize=8, title_fontsize=8, frameon=False)
    fig.suptitle("Claim-1 controls, N = 1,000: criticality measures over training "
                 "(filled marker: floor crossing)", fontsize=11)
    fig.savefig(a.out, dpi=150)


if __name__ == "__main__":
    main()
