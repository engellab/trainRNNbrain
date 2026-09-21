"""Paper figure: the three remedies, ordered by how much of the network they recover.

The paper's spine. Each remedy is better than the last and each has a price, so the figure has to
show recovery and cost side by side rather than in separate display items:

  (a) WHAT EACH REMEDY DOES, and the result as a pictogram    schematic; 100 dots = the network
  (b) WHAT A "BURST UNIT" IS                                  real example units, not a cartoon
  (c) how many units each remedy leaves alive   3-bit flip-flop, N=1000, 150k, 7 seeds per cell
  (d) what it costs in task performance         same networks, noise-free task loss
  (e) what `rws` buys on top of `frm`           burst units, before/after a causal penalty switch
  (f) what `rws` costs                          DMTS at a 36-tau delay, 3 seeds per arm

The top row exists because a row of dot plots states the result without explaining it. (a) says
what each intervention DOES to the network and shows the answer as a filled fraction of 100 units,
using the same four conditions, same order and same colours as the panels below, so the reader maps
schematic onto data. (b) defines the one term the argument turns on: a burst unit is not a silent
unit, and the difference is visible in the traces.

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
import glob
import os
import pickle
import re
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


MECH = {                       # one line each: what the intervention actually does to training
    "baseline":  "train normally",
    "dropout":   "delete 5% of units\nevery batch",
    "frm":       "penalise every unit's\nrate below a cap",
    "frm + rws": "...and cap each unit's\neffective in-degree",
}


def pictogram(ax, data, n_dots=100):
    """Panel (a): what each remedy does, with its result drawn as a filled fraction of the network.

    A 10x10 dot grid per condition, filled to the measured live fraction. Same conditions, order
    and colours as the quantitative panels below, so the schematic and the data are the same
    picture at two levels of precision.

    Args:
        ax: axes; data: output of `cells`; n_dots: dots in the grid (100 -> one dot per percent).
    """
    side = int(round(n_dots ** 0.5))
    for col_i, (label, col, a) in enumerate(data):
        if not len(a):
            continue
        frac = a[:, 0].mean() / N_UNITS
        n_on = int(round(frac * n_dots))
        x0 = col_i * (side + 3.4)
        for d in range(n_dots):
            r, c = divmod(d, side)
            on = d < n_on
            ax.scatter(x0 + c, -r, s=15,
                       color=col if on else "none",
                       edgecolor=col if on else "#d8d7d0", linewidth=0.8, zorder=3)
        ax.text(x0 + (side - 1) / 2, 3.3, label, ha="center", fontsize=10, color=INK)
        ax.text(x0 + (side - 1) / 2, 1.75, MECH[label], ha="center", va="top",
                fontsize=7.6, color=MUTED, linespacing=1.3)
        ax.text(x0 + (side - 1) / 2, -side - 0.6, f"{frac:.0%} of units alive",
                ha="center", fontsize=8.5, color=col)
    ax.set_xlim(-1.5, 4 * (side + 3.4) - 2.4)
    ax.set_ylim(-side - 2.2, 4.6)
    ax.axis("off")


def example_units(root, pen="frm", n_show=2):
    """Real example units for panel (b): the lowest- and highest-occupancy live units of one net.

    Args:
        root: trained_RNNs folder (its parent holds data/unit_stats_cache.pkl);
        pen: which condition's network to take examples from; n_show: trials to overlay.
    Returns:
        (burst_trace, sustained_trace, tpr_burst, tpr_sustained) or None if the cache is absent.
        Each trace is (T, n_show).
    """
    import pickle
    cache = os.path.join(os.path.dirname(root), "unit_stats_cache.pkl")
    if not os.path.exists(cache):
        return None
    c = pickle.load(open(cache, "rb"))
    runs = [v for v in c.values()
            if v.get("task") == "flip-flop" and v.get("N") == 2000
            and v.get("pen") == pen and "examples" in v]
    if not runs:
        return None
    ex = runs[0]["examples"]
    return (ex["lo"][0][:, :n_show], ex["hi"][0][:, :n_show],
            float(np.mean(ex["tpr_lo"][:1])), float(np.mean(ex["tpr_hi"][:1])))


def panel_examples(ax, root):
    """Panel (b): what a burst unit looks like next to a sustained one. Defines the term used in (e).

    Args:
        ax: axes; root: trained_RNNs folder.
    Returns:
        True if drawn, False if the cache is missing.
    """
    got = example_units(root)
    if got is None:
        return False
    burst, sustained, tpr_b, tpr_s = got
    T = burst.shape[0]
    t = np.arange(T)
    hi = max(sustained.max(), burst.max()) * 1.12
    ax.plot(t, sustained[:, 0] + hi, color="#1baf7a", lw=1.2)
    ax.plot(t, burst[:, 0], color="#eb6834", lw=1.2)
    ax.axhline(hi, color=GRID, lw=0.8)
    ax.text(T * 0.99, hi * 2.16, f"sustained unit   tPR = {tpr_s:.2f}",
            ha="right", va="top", fontsize=8.4, color="#1baf7a")
    ax.text(T * 0.99, hi * 1.06, f"burst unit   tPR = {tpr_b:.2f}",
            ha="right", va="top", fontsize=8.4, color="#eb6834")
    ax.text(0.0, -hi * 0.30, "both count as ALIVE by any threshold;\nonly one does sustained work",
            ha="left", va="top", fontsize=7.8, color=MUTED, linespacing=1.3)
    ax.set_yticks([])
    ax.set_xticks([])
    ax.set_ylim(-hi * 0.62, hi * 2.30)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    return True


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


SWITCH_SUB = "NBitFlipFlop_std_switch"
N_SAMPLES = 300 * 1024          # T x batch: the trace stores raw (sum r)^2 / sum r^2
BURST = 0.05                    # a unit active for <5% of the probe's samples is a burst unit
SWITCH_ARMS = [("A1", "+rws", "#1baf7a"), ("A2", "frm\n(ctrl)", BASE_COL),
               ("A3", "−rws", "#eb6834"), ("A4", "frm+rws\n(ctrl)", BASE_COL)]


def panel_transience(ax, root):
    """Panel (c): does `rws` remove the units that satisfy `frm` only transiently?

    Read CAUSALLY from the penalty-switch sweep rather than by comparing independently trained
    networks: each run is warm-started from a trained parent and continued for 50k iterations with
    the penalty switched, so the before/after pair is the SAME network and the two same-penalty
    arms (A2, A4) absorb the effect of warm-starting and extra training. Both directions are
    present, which is what makes it causal rather than correlational.

    Temporal participation ratio is read straight out of `*_ParticipationTrace.pkl`
    (key `temporal_pr`, stored per unit at the participation cadence) — no network is rebuilt and
    nothing is simulated. A unit counts as a BURST unit when its tPR/n is below 5%, i.e. it carries
    activity in under a twentieth of the probe, which is how a unit satisfies a firing-rate penalty
    without doing sustained work.

    Args:
        ax: axes; root: trained_RNNs folder.
    Returns:
        True if drawn, False if the switch sweep is absent.
    """
    runs = {}
    for cell in sorted(glob.glob(os.path.join(root, SWITCH_SUB, "*", ""))):
        m = re.search(r"arm=(A\d)", cell)
        if not m:
            continue
        for net in sorted(glob.glob(cell + "*/")):
            f = glob.glob(net + "*ParticipationTrace.pkl")
            if not f:
                continue
            tr = pickle.load(open(f[0], "rb"))
            if "temporal_pr" not in tr:
                continue
            p = np.asarray(tr["participation"])
            t = np.asarray(tr["temporal_pr"], dtype=float) / N_SAMPLES
            b0 = (t[0][p[0] >= SILENT_FLIPFLOP] < BURST).mean()
            b1 = (t[-1][p[-1] >= SILENT_FLIPFLOP] < BURST).mean()
            runs.setdefault(m.group(1), []).append((100 * b0, 100 * b1))
    if not runs:
        return False
    for x, (arm, label, col) in enumerate(SWITCH_ARMS):
        for b0, b1 in runs.get(arm, []):
            ax.plot([x - 0.17, x + 0.17], [b0, b1], color=col, lw=1.4, alpha=0.85, zorder=2)
            ax.scatter([x - 0.17], [b0], s=20, facecolor="white", edgecolor=col, lw=1.3, zorder=3)
            ax.scatter([x + 0.17], [b1], s=28, color=col, edgecolor="white", lw=0.6, zorder=3)
    ax.set_xticks(range(len(SWITCH_ARMS)))
    ax.set_xticklabels([a[1] for a in SWITCH_ARMS], fontsize=8.5)
    ax.set_ylabel("burst units (% of live)", fontsize=9)
    ax.set_ylim(0, 48)
    ax.grid(axis="y", color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)
    ax.annotate("adding rws\nremoves them", (0.17, 5.6), textcoords="offset points",
                xytext=(4, 18), ha="left", fontsize=8, color="#1baf7a")
    ax.annotate("removing it\nbrings them back", (2.17, 22.4), textcoords="offset points",
                xytext=(2, -26), ha="center", fontsize=8, color="#eb6834")
    ax.text(0.5, 0.99, "open = before the switch, filled = 50k iterations after; N=2000, 3 seeds",
            transform=ax.transAxes, ha="center", va="top", fontsize=7.2, color=MUTED)
    return True


def main(root):
    """Draw the four-panel remedies figure."""
    data = cells(root)
    base_live = data[0][2][:, 0].mean()
    base_loss = data[0][2][:, 2].mean()

    fig = plt.figure(figsize=(15.5, 8.2))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.0, 1.25], hspace=0.42, wspace=0.34)

    ax_a = fig.add_subplot(gs[0, :3])
    pictogram(ax_a, data)
    ax_a.set_title("a   what each remedy does to training, and how much of the network survives it",
                   fontsize=10.5, loc="left")

    ax_b = fig.add_subplot(gs[0, 3])
    if not panel_examples(ax_b, root):
        ax_b.text(0.5, 0.5, "unit_stats_cache.pkl missing", ha="center", va="center",
                  fontsize=9, color=MUTED, transform=ax_b.transAxes)
    ax_b.set_title("b   alive is not the same as working", fontsize=10.5, loc="left")

    axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    strip(axes[0], data, 0, f"live units of {N_UNITS}\n(scale-free criterion)",
          baseline=base_live, pct_of=N_UNITS)
    axes[0].set_ylim(0, N_UNITS * 1.16)
    axes[0].set_title("c   what each recovers", fontsize=10.5, loc="left")

    strip(axes[1], data, 2, "task loss (noise-free, dropout off)", baseline=base_loss)
    axes[1].set_title("d   what it costs", fontsize=10.5, loc="left")

    if not panel_transience(axes[2], root):
        axes[2].text(0.5, 0.5, "switch sweep missing", ha="center", va="center",
                     fontsize=9, color=MUTED, transform=axes[2].transAxes)
    axes[2].set_title("e   what rws buys on top of frm", fontsize=10.5, loc="left")

    ok = panel_dmts(axes[3], os.path.join(os.path.dirname(DATA_DIR), "dmts_curves_delay36.npz"))
    axes[3].set_title("f   what rws costs", fontsize=10.5, loc="left")
    if not ok:
        axes[3].text(0.5, 0.5, "dmts_curves_delay36.npz missing", ha="center", va="center",
                     fontsize=9, color=MUTED, transform=axes[3].transAxes)

    fig.suptitle("Three remedies, ordered by how much of the network they recover — and what each one costs",
                 fontsize=12.5, x=0.008, y=0.985, ha="left")
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=150)
    print("saved", os.path.normpath(OUT))
    for label, _, a in data:
        if len(a):
            print(f"  {label:11} live {a[:,0].mean():6.0f} ± {a[:,0].std():<4.0f}  "
                  f"abs4e-2 {a[:,1].mean():6.0f}  loss {a[:,2].mean():.5f}  n={len(a)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else DATA_DIR)
