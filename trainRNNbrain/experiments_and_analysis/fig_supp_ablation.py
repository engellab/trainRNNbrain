#!/usr/bin/env python3
"""
Extended Data Fig. 1 of the paper (companion of Figure 1d): removing the inactive units of every
default network of the size series, at every size (round 2, dead-relu-paper#52; #45, #38 items 4-5).

Figure 1(d) draws the removal curves at N = 1,000. This figure shows the same test on every network
the ablation was run on - three tasks, N = 500 to 4,000 - one column per task:

  (a) held-out R^2 of each network intact (ink dash), with every inactive unit removed (filled) and
      with as many units removed in random order (open, mean of five orders), against network size
      N on the x axis. N is an axis, so marker size encodes nothing.
  (b) the quietest-first removal curve of each network, the x axis in units of the network's own
      inactive count, so x = 1 is the point at which every inactive unit has been removed.

Networks and measurement point (agrees with Methods, Training): 3BFF networks are measured at
75,000 iterations, the end of the second of their three consecutive training runs (40,000 + 35,000
of 40,000 + 35,000 + 125,000; the size series continues them to 200,000); CDDM at 100,000 and DMTS
at 150,000 iterations, the end of their training. 36 networks; the one DMTS network with intact
R^2 < 0.8 (it never learned the task) is left out, so 35 are drawn. Scores below -0.5 are drawn at
-0.5.

Data: the JSON files ablate_silent_units.py wrote on 14 Oct 2026 to Della
/scratch/gpfs/TENGEL/pt1290/c1_ablation_seeded (*__end.json), copied unchanged to ABLATION_DIR:
  rsync -a 'della:/scratch/gpfs/TENGEL/pt1290/c1_ablation_seeded/*__end.json' data/c1_ablation_seeded/
Run from the checkout root, on the Mac (Helvetica):
  python trainRNNbrain/experiments_and_analysis/fig_supp_ablation.py
Output: img/internal_figures/fig_supp_ablation.pdf (+ .svg; vector only, paperstyle.save)
"""

import glob
import json
import os
import sys

import numpy as np
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

ABLATION_DIR = "data/c1_ablation_seeded"   # local copy of Della's c1_ablation_seeded, see docstring
READOUT = "end"                            # the file suffix ablate_silent_units gives this read
# (task, sweep-name prefixes of the cells, colour, minimum intact R^2 for a run to count as solved)
TASKS = [("3BFF", ("NBitFlipFlop_ff_revive_c75000", "NBitFlipFlop_paper_grid_c75000"),
          ps.TASK_3BFF, None),
         ("CDDM", ("CDDM_paper_grid",), ps.TASK_CDDM, None),
         ("DMTS", ("DMTS_paper_grid_d5",), ps.TASK_DMTS, 0.8)]
SIZES = [500, 1000, 2000, 4000]
Y_FLOOR = -0.5
JITTER = 0.12                  # half-width, in size slots, over which a size's networks are spread

FIG_H_MM = 90.0                # under the float's 0.40 \\textheight cap (91.4 mm), so it prints at 100%
ROW_A = (12.0, 27.0)           # (top, height) of row a, mm from the canvas top
ROW_B = (55.0, 25.0)           # (top, height) of row b
LEFT_MM, PANEL_W_MM, GAP_MM = 14.0, 45.0, 6.5   # first column's left edge, column width, gap
LETTER_X_MM = 1.0


def load(ablation_dir=ABLATION_DIR, readout=READOUT):
    """Read every ablation result file of one read-out and group it by task.

    Args:
        ablation_dir: str, folder of ablate_silent_units JSON files; readout: their suffix ('end').
    Returns:
        (data, dropped): data maps task -> list of result dicts (keys N, r2_full, r2_no_silent,
        r2_random_same_count, ks, quiet_first, n_silent, label, ...); dropped lists the labels of
        runs left out because their intact R^2 is under the task's minimum.
    """
    data, dropped = {t[0]: [] for t in TASKS}, []
    for f in sorted(glob.glob(os.path.join(ablation_dir, f"*__{readout}.json"))):
        with open(f) as fh:
            r = json.load(fh)
        for task, prefixes, _col, min_r2 in TASKS:
            if r.get("sweep", "").startswith(prefixes):
                if min_r2 is None or r["r2_full"] >= min_r2:
                    data[task].append(r)
                else:
                    dropped.append(r["label"])
    return data, dropped


def ax_mm(fig, left, top, width, height):
    """Add an axes placed in millimetres from the canvas's top-left corner.

    Args:
        fig: the figure; left, top, width, height: in mm.
    Returns:
        the new Axes.
    """
    W, H = fig.get_figwidth() / ps.MM, fig.get_figheight() / ps.MM
    return fig.add_axes([left / W, 1.0 - (top + height) / H, width / W, height / H])


def panel_a(ax, runs, col):
    """Row (a), one task: held-out R^2 intact, with every inactive unit removed and at random, per N.

    Args:
        ax: axes; runs: the task's result dicts; col: the task colour.
    Returns:
        None.
    """
    for i, N in enumerate(SIZES):
        mine = sorted([r for r in runs if r["N"] == N], key=lambda r: r["label"])
        xs = i + (np.linspace(-JITTER, JITTER, len(mine)) if len(mine) > 1 else np.zeros(len(mine)))
        for x, r in zip(xs, mine):
            rnd = float(np.mean(r["r2_random_same_count"]))
            ax.plot(x, r["r2_full"], "_", ms=3.6, mew=0.9, color=ps.INK, zorder=5)
            ax.plot(x, max(r["r2_no_silent"], Y_FLOOR), "o", ms=ps.MARKER_PT, color=col, mec="none",
                    zorder=4, clip_on=False)
            ax.plot(x, max(rnd, Y_FLOOR), "o", ms=ps.MARKER_PT, mfc="white", mec=col, mew=0.6,
                    zorder=3, clip_on=False)
    ax.set(xlim=(-0.6, len(SIZES) - 0.4), ylim=(Y_FLOOR - 0.04, 1.05))
    ax.set_xticks(range(len(SIZES)))
    ax.set_xticklabels([ps.fmt_count(n) for n in SIZES])
    ax.tick_params(axis="x", length=0)
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ps.ygrid(ax)


def panel_b(ax, runs, col):
    """Row (b), one task: quietest-first removal curves against removed units / own inactive count.

    Args:
        ax: axes; runs: the task's result dicts; col: the task colour.
    Returns:
        None.
    """
    for r in runs:
        x = np.asarray(r["ks"], float) / max(r["n_silent"], 1)
        y = np.maximum(np.asarray(r["quiet_first"], float), Y_FLOOR)
        ax.plot(x, y, "-", lw=ps.THIN_LW, color=col, alpha=0.7)
    ax.axvline(1.0, color=ps.INK, lw=0.6, ls=(0, (5, 2.5)))
    ax.set(xlim=(0, 1.6), ylim=(Y_FLOOR - 0.04, 1.05))
    ax.set_xticks([0, 0.5, 1.0, 1.5])
    ax.set_yticks([-0.5, 0, 0.5, 1.0])
    ps.ygrid(ax)


def main():
    """Draw Extended Data Fig. 1 and print the numbers its caption and Results 1 quote. Returns the PDF path."""
    ps.setup()
    data, dropped = load()
    fig = ps.figure(FIG_H_MM)
    axes_a, axes_b = [], []
    for k, (task, _pre, col, _min) in enumerate(TASKS):
        left = LEFT_MM + k * (PANEL_W_MM + GAP_MM)
        ax_a = ax_mm(fig, left, ROW_A[0], PANEL_W_MM, ROW_A[1])
        ax_b = ax_mm(fig, left, ROW_B[0], PANEL_W_MM, ROW_B[1])
        panel_a(ax_a, data[task], col)
        panel_b(ax_b, data[task], col)
        ax_a.set_title(task, fontsize=ps.LABEL_PT, color=ps.INK, pad=3)
        if k:
            ax_a.tick_params(labelleft=False)
            ax_b.tick_params(labelleft=False)
        axes_a.append(ax_a)
        axes_b.append(ax_b)
    axes_a[0].set_ylabel(ps.LABELS["r2"])
    axes_b[0].set_ylabel(ps.LABELS["r2"])
    axes_a[1].set_xlabel(ps.LABELS["size"])
    axes_b[1].set_xlabel("units removed quietest first, in units of the network's own inactive count")
    axes_b[0].text(0.96, Y_FLOOR + 0.04, "every inactive\nunit removed", ha="right", va="bottom",
                   fontsize=ps.TICK_PT, color=ps.INK, linespacing=1.15)
    # the legend sits in the strip above row a, off the data
    fig.legend(handles=[
        Line2D([], [], ls="", marker="_", ms=3.6, mew=0.9, color=ps.INK, label="intact network"),
        Line2D([], [], ls="", marker="o", ms=ps.MARKER_PT, color=ps.MUTED, mec="none",
               label="every inactive unit removed"),
        Line2D([], [], ls="", marker="o", ms=ps.MARKER_PT, mfc="white", mec=ps.MUTED, mew=0.6,
               label="as many units removed in random order")],
        loc="upper left", bbox_to_anchor=(LEFT_MM / ps.W_FULL_MM, 1.0 - 2.5 / FIG_H_MM), ncol=3,
        fontsize=ps.TICK_PT, handlelength=1.2, columnspacing=1.6, borderaxespad=0.0, frameon=False)
    for ax, letter in ((axes_a[0], "a"), (axes_b[0], "b")):
        bb = ax.get_position()
        ps.panel_letter(ax, letter, x_mm=bb.x0 * ps.W_FULL_MM - LETTER_X_MM,
                        y_mm=(ROW_A[0] - 5.5) if letter == "a" else 6.5)
    out = ps.save(fig, "fig_supp_ablation")

    print("\n--- numbers quoted in the caption and Results 1 ---")
    print(f"  left out (intact R2 under the task minimum): {dropped}")
    n_all = sum(len(v) for v in data.values())
    print(f"  networks drawn: {n_all} (+ {len(dropped)} left out = {n_all + len(dropped)})")
    for task, _pre, _col, _min in TASKS:
        rs = data[task]
        by_n = {N: sum(r["N"] == N for r in rs) for N in SIZES}
        d = np.array([r["r2_no_silent"] - r["r2_full"] for r in rs])
        rnd = np.array([np.mean(r["r2_random_same_count"]) for r in rs])
        print(f"  {task:5} {len(rs):2d} nets {by_n}; intact R2 {min(r['r2_full'] for r in rs):.3f} to "
              f"{max(r['r2_full'] for r in rs):.3f}; removing every inactive unit changes R2 by "
              f"{d.min():+.3f} to {d.max():+.3f}; as many at random: {rnd.min():.3f} to {rnd.max():.3f}; "
              f"random minus inactive-removed: {(rnd - (d + [r['r2_full'] for r in rs])).max():+.3f} max")
    return out


if __name__ == "__main__":
    main()
