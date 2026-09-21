"""Paper figure: the three remedies, ordered by how much of the network they recover.

The paper's spine. Each remedy is better than the last and each has a price, so the figure has to
show recovery and cost side by side rather than in separate display items:

  (a) how many units each remedy leaves alive      3-bit flip-flop, N=1000, 150k, 7 seeds per cell
  (b) what it costs in task performance            same networks, noise-free task loss
  (c) what `rws` buys on top of `frm`              units that satisfy frm only transiently
  (d) what `rws` costs                             DMTS at a 36-tau delay, 3 seeds per arm

Panel (d) is in this figure and not its own because the trade is only legible when the gain and the
loss are one glance apart: `frm + rws` is the only arm that recovers the whole network AND the only
arm that never learns the long-delay memory task.

Conventions. Live counts are the scale-free criterion (a unit is silent below 5% of its own
network's q95 participation); the task-calibrated absolute criterion (4e-2, Otsu on this task's
pooled log participation) gives 291 ± 18 against 263 ± 13 for the baseline and is reported in the
legend so the claim is not resting on one threshold. Task loss is `loss_clean_train`, probed inside
the trainer noise-free and with dropout OFF, so every arm is scored on the same quantity; it is NOT
the total training objective, which is not comparable across penalties.

Usage: python fig_remedies.py [<trained_RNNs root>]
Writes img/internal_figures/fig_remedies.png
"""
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainRNNbrain.experiments_and_analysis.common import DATA_DIR, IMG_DIR, SILENT_FLIPFLOP
from trainRNNbrain.experiments_and_analysis.flipflop_dropout_readout import collect, REFS, DROP_SUB

N_UNITS = 1000
OUT = os.path.join(IMG_DIR, "fig_remedies.png")

# Categorical slots 1-3 of the project's validated palette (CVD-checked: worst adjacent pair
# dE 9.2 deutan, 27.6 normal). The baseline is NOT a series - it is drawn as neutral ink, because
# it is the reference every remedy is measured against.
INK, MUTED, GRID = "#0b0b0b", "#898781", "#e1e0d9"
BASE_COL = "#7a7a72"
COND = [
    ("baseline",   "none", "none", BASE_COL),
    ("dropout",    "none", "dead", "#2a78d6"),
    ("frm",        "frm",  "none", "#eb6834"),
    ("frm + rws",  "both", "none", "#1baf7a"),
]


def cells(root):
    """Read the four conditions of panels (a) and (b).

    Args:
        root: trained_RNNs folder.
    Returns:
        list of (label, colour, array) where array is (n_seeds, 4) of
        [live_scalefree, live_abs4e-2, clean_loss, r2] as `flipflop_dropout_readout.read_net` gives.
    """
    out = []
    for label, pen, kind, col in COND:
        ctrl = os.path.join(root, DROP_SUB, f"EqType=h_k=3_N={N_UNITS}_pen={pen}_do=none")
        if kind == "none":                       # no dropout: pool the historical reference cell
            a = collect(os.path.join(root, REFS[pen]), ctrl)
        else:
            a = collect(os.path.join(root, DROP_SUB, f"EqType=h_k=3_N={N_UNITS}_pen={pen}_do={kind}"))
        out.append((label, col, a))
    return out


def strip(ax, data, col_idx, ylabel, baseline=None, pct_of=None):
    """One panel: every seed as a dot, the condition mean as a wide dash, in fixed condition order.

    A dot strip rather than bars: with 7 seeds the spread is the point, and bars would hide it
    behind a summary the reader cannot check.

    Args:
        ax: axes; data: output of `cells`; col_idx: which column of the array to plot;
        ylabel: y label; baseline: value to draw as a reference line, or None;
        pct_of: if given, annotate each mean as a percentage of this (the network size).
    """
    rng = np.random.default_rng(0)
    for x, (label, col, a) in enumerate(data):
        if not len(a):
            continue
        y = a[:, col_idx]
        ax.scatter(x + rng.uniform(-0.13, 0.13, len(y)), y, s=26, color=col,
                   edgecolor="white", linewidth=0.7, zorder=3)
        ax.plot([x - 0.28, x + 0.28], [y.mean()] * 2, color=col, lw=2.6, zorder=4)
        txt = f"{y.mean():.0f}" if pct_of else f"{y.mean():.4f}"
        if pct_of:
            txt += f"\n{y.mean() / pct_of:.0%}"
        ax.annotate(txt, (x, y.mean()), textcoords="offset points", xytext=(0, 13),
                    ha="center", fontsize=8.5, color=INK, zorder=5)
    if baseline is not None:
        ax.axhline(baseline, color=BASE_COL, lw=0.9, ls=":", zorder=1)
    ax.set_xticks(range(len(data)))
    ax.set_xticklabels([d[0] for d in data], fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def panel_dmts(ax, npz_path):
    """Panel (d): per-seed escape time on the 36-tau memory task, 'never' marked explicitly.

    Args:
        ax: axes; npz_path: the dump written by `dmts_readout.py --sub DMTS_std_delay36 --dump`.
    Returns:
        True if drawn, False if the dump is absent.
    """
    if not os.path.exists(npz_path):
        return False
    d = np.load(npz_path)
    var = float(d["target_variance"])
    keys = sorted({k.rsplit("_", 1)[0] for k in d.files if k.endswith("_iters")})
    arms = {"none": ("no penalty", BASE_COL), "frm": ("frm", "#eb6834"),
            "both": ("frm + rws", "#1baf7a")}
    order = ["none", "frm", "both"]
    for x, pen in enumerate(order):
        label, col = arms[pen]
        ks = [k for k in keys if k.split("_")[1] == pen]
        # Seeds are jittered in x: without it the three "never" markers land on one point and the
        # panel reads as n=1, which is the opposite of what it is meant to show.
        jit = np.linspace(-0.17, 0.17, len(ks)) if len(ks) > 1 else np.zeros(1)
        for j, k in zip(jit, ks):
            it, L = d[k + "_iters"], d[k + "_loss"]
            r2 = 1.0 - L / var
            above = np.flatnonzero(r2 >= 0.9)
            if above.size:
                ax.scatter([x + j], [it[above[0]]], s=44, color=col, edgecolor="white",
                           lw=0.7, zorder=3)
            else:
                ax.scatter([x + j], [1.7e5], s=62, marker="x", color=col, lw=2.1, zorder=3)
        n_esc = sum(1 for k in ks if (1.0 - d[k + "_loss"] / var >= 0.9).any())
        ax.annotate(f"{n_esc}/{len(ks)}", (x, 4.2e5), ha="center", fontsize=9,
                    color=col, annotation_clip=False)
    ax.axhline(1.5e5, color=MUTED, lw=0.8, ls="--")
    ax.text(-0.42, 2.45e5, "never learned", ha="left", va="center", fontsize=8, color=MUTED)
    ax.set_yscale("log")
    ax.set_ylim(5e3, 3e5)
    ax.set_xlim(-0.5, 2.5)
    ax.set_xticks(range(3))
    ax.set_xticklabels([arms[p][0] for p in order], fontsize=9)
    ax.set_ylabel("iteration the memory appears\n(36-tau delay, 3 seeds)", fontsize=9)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    return True


def main(root):
    """Draw the four-panel remedies figure."""
    data = cells(root)
    base_live = data[0][2][:, 0].mean()
    base_loss = data[0][2][:, 2].mean()

    fig, axes = plt.subplots(1, 4, figsize=(15.5, 4.3))
    strip(axes[0], data, 0, f"live units of {N_UNITS}\n(scale-free criterion)",
          baseline=base_live, pct_of=N_UNITS)
    axes[0].set_ylim(0, N_UNITS * 1.16)
    axes[0].set_title("(a) what each remedy recovers", fontsize=10, loc="left")

    strip(axes[1], data, 2, "task loss (noise-free, dropout off)", baseline=base_loss)
    axes[1].set_title("(b) what it costs", fontsize=10, loc="left")

    axes[2].set_title("(c) what rws buys on top of frm", fontsize=10, loc="left")
    axes[2].text(0.5, 0.5, "transience panel\n(pending source)", ha="center", va="center",
                 fontsize=9, color=MUTED, transform=axes[2].transAxes)
    axes[2].set_xticks([]); axes[2].set_yticks([])
    for s in axes[2].spines.values():
        s.set_color(GRID)

    ok = panel_dmts(axes[3], os.path.join(os.path.dirname(DATA_DIR), "dmts_curves_delay36.npz"))
    axes[3].set_title("(d) what rws costs", fontsize=10, loc="left")
    if not ok:
        axes[3].text(0.5, 0.5, "dmts_curves_delay36.npz missing", ha="center", va="center",
                     fontsize=9, color=MUTED, transform=axes[3].transAxes)

    fig.suptitle("Three remedies, ordered by how much of the network they recover — and what each one costs",
                 fontsize=11.5, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved", os.path.normpath(OUT))
    for label, _, a in data:
        if len(a):
            print(f"  {label:11} live {a[:,0].mean():6.0f} ± {a[:,0].std():<4.0f}  "
                  f"abs4e-2 {a[:,1].mean():6.0f}  loss {a[:,2].mean():.5f}  n={len(a)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
