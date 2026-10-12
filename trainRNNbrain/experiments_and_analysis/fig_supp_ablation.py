#!/usr/bin/env python3
"""
Supplementary figure to Figure 1(d): the silent units do no work, in EVERY network at EVERY size.

Figure 1(d) draws the removal curves at N = 1000. This figure shows the same test on every standard
control the ablation was run on - three tasks, N = 500 to 4000 - in two panels:

  (a) EVERY NETWORK, ONE POINT EACH   held-out r2 after removal against the intact network's.
                                      Filled: every silent unit removed. Open: the same number of
                                      units removed in random order (mean of five orders). The
                                      filled points sit on the diagonal; the open ones fall to
                                      chance and below.
  (b) THE CLIFF IS WHERE THE SILENT   quietest-first removal curves, the x axis in units of each
      UNITS END                       network's OWN silent count, so 1 is the moment the first
                                      working unit is removed. Every network holds its score up to
                                      1 and loses it after, whatever its size.

Scores below -0.5 are drawn at the axis floor.

Data: the JSONs ablate_silent_units.py writes, read from ABLATION_DIR (Della). Build on Della:
  python trainRNNbrain/experiments_and_analysis/fig_supp_ablation.py [--readout end]
Output: img/internal_figures/fig_supp_ablation.pdf (+ .svg)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps

ABLATION_DIR = "/scratch/gpfs/TENGEL/pt1290/c1_ablation"
# (label, sweep-name prefixes of the cells, colour, minimum intact r2 for a run to count as solved)
TASKS = [("3-bit flip-flop", ("NBitFlipFlop_ff_revive_c75000", "NBitFlipFlop_paper_grid_c75000"),
          ps.SLOTS[0], None),
         ("CDDM", ("CDDM_paper_grid",), ps.SLOTS[1], None),
         ("DMTS, 5$\\tau$ delay", ("DMTS_paper_grid_d5",), ps.SLOTS[2], 0.8)]
SIZES = {500: 2.0, 1000: 3.0, 2000: 4.2, 4000: 5.6}      # marker size by N
Y_FLOOR = -0.5


def load(readout):
    """Every ablation result of one read-out, grouped by task label."""
    out = {t[0]: [] for t in TASKS}
    for f in sorted(glob.glob(os.path.join(ABLATION_DIR, f"*__{readout}.json"))):
        with open(f) as fh:
            r = json.load(fh)
        for label, prefixes, _, min_r2 in TASKS:
            if r.get("sweep", "").startswith(prefixes):
                if min_r2 is None or r["r2_full"] >= min_r2:
                    out[label].append(r)
    return out


def panel_a(ax, data):
    """Removal against intact score, one point per network."""
    lo = Y_FLOOR
    for label, _, col, _ in TASKS:
        for r in data[label]:
            ms = SIZES.get(r["N"], 3.0)
            rnd = float(np.mean(r["r2_random_same_count"]))
            ax.plot(r["r2_full"], max(rnd, lo), "o", ms=ms, mfc="none", mec=col, mew=0.6, zorder=3)
            ax.plot(r["r2_full"], max(r["r2_no_silent"], lo), "o", ms=ms, color=col, mec="none",
                    zorder=4)
    ax.plot([lo, 1.02], [lo, 1.02], "-", lw=0.6, color=ps.MUTED, zorder=1)
    ax.axhline(0, color=ps.FAINT, lw=0.5, ls=(0, (3, 2)), zorder=1)
    ax.set(xlim=(0.78, 1.01), ylim=(lo - 0.03, 1.05), xlabel="held-out $r^2$, intact network",
           ylabel="held-out $r^2$ after removal")
    ps.ygrid(ax)
    hs = [Line2D([], [], ls="", marker="o", color=col, mec="none", ms=3.4, label=lab)
          for lab, _, col, _ in TASKS]
    hs += [Line2D([], [], ls="", marker="o", color=ps.INK, mec="none", ms=3.4,
                  label="every silent unit removed"),
           Line2D([], [], ls="", marker="o", mfc="none", mec=ps.INK, ms=3.4,
                  label="as many removed at random")]
    hs += [Line2D([], [], ls="", marker="o", color=ps.MUTED, mec="none", ms=s, label=f"N = {N}")
           for N, s in SIZES.items()]
    ax.legend(handles=hs, loc="lower right", fontsize=5.3, ncol=2, columnspacing=0.8)


def panel_b(ax, data):
    """Quietest-first curves against removed units / own silent count."""
    for label, _, col, _ in TASKS:
        for r in data[label]:
            x = np.asarray(r["ks"], float) / max(r["n_silent"], 1)
            y = np.maximum(np.asarray(r["quiet_first"], float), Y_FLOOR)
            ax.plot(x, y, "-", lw=0.4 + 0.15 * np.log2(r["N"] / 500), color=col, alpha=0.6)
    ax.axvline(1.0, color=ps.INK, lw=0.6, ls=(0, (3, 2)))
    ax.text(1.03, 0.08, "first working\nunit removed", fontsize=5.4, color=ps.INK, va="bottom")
    ax.set(xlim=(0, 1.6), ylim=(Y_FLOOR - 0.03, 1.05),
           xlabel="units removed quietest first,\nin units of the network's own silent count",
           ylabel="held-out $r^2$")
    ps.ygrid(ax)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--readout", default="end")
    a = ap.parse_args()
    ps.setup()
    data = load(a.readout)
    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(ps.W2, 70 * ps.MM),
                                     gridspec_kw=dict(wspace=0.32))
    panel_a(ax_a, data)
    panel_b(ax_b, data)
    ps.panel_letter(ax_a, "a")
    ps.panel_letter(ax_b, "b")
    out = ps.save(fig, f"fig_supp_ablation{'' if a.readout == 'end' else '_' + a.readout}")
    print("\n--- numbers ---")
    for label, _, _, _ in TASKS:
        rs = data[label]
        if not rs:
            print(f"  {label}: no results")
            continue
        d = np.array([r["r2_no_silent"] - r["r2_full"] for r in rs])
        print(f"  {label:22} {len(rs):2d} nets, N = {sorted({r['N'] for r in rs})}: removing every "
              f"silent unit changes r2 by {d.min():+.4f} to {d.max():+.4f}; as many at random "
              f"gives {min(np.mean(r['r2_random_same_count']) for r in rs):.3f} to "
              f"{max(np.mean(r['r2_random_same_count']) for r in rs):.3f}")
    return out


if __name__ == "__main__":
    main()
