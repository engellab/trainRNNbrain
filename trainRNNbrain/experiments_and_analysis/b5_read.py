#!/usr/bin/env python3
"""B5, the optimizer control: every finished run of every arm, read at its floor and at 200k.

Each run is read twice:
  - at its own floor: where its noisy training loss (joined along its warm-start chain) first
    reaches 1.05x its fitted floor, or 1.05x the minimum of its running median when the fit's
    stretch exponent sits on its bound (`common.noisy_floor`, `floor_kind`). The B5 runs kept only
    a rolling restart checkpoint, so R^2 there is ESTIMATED from the loss ratio
    (`fig_knobs_at_floor.floor_r2`; 36 seeded noise draws);
  - at 200k: the active count of the last participation probe and the held-out R^2 of the final
    weights (36 draws).
The active count at the floor is the participation probe nearest the crossing.

SGD lr 10 is left out (Pavel judged it pointless; it diverged). The SGD lr 3, wd 1e-6 arm fails the
task (R^2 0.30-0.44) and is listed but drawn off the bottom of the axes.

Usage (from the repository root, with `data` pointing at the trained networks):
    python trainRNNbrain/experiments_and_analysis/b5_read.py
Writes img/internal_figures/diag_b5_optimizer.{png,pdf} and prints the per-run table.
"""

import glob
import os
import re
import sys

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import fig_paper_F1 as F1
from common import DATA_DIR, noisy_floor
from fig_knobs_at_floor import R2_DRAWS, floor_r2
from heldout_r2 import heldout_r2, load_cache, save_cache

SWEEP = os.path.join(DATA_DIR, "NBitFlipFlop_b5_optimizer")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "img", "internal_figures",
                   "diag_b5_optimizer")
SKIP_LR = {"10"}          # SGD lr 10: dropped by Pavel
# SGD learning rate -> colour (light to dark orange); filled marker = wd 1e-6, open = wd 0
SGD_COLOR = {"0.03": "#f7b37a", "0.1": "#f28a3c", "0.3": "#d9620f", "1": "#a8470a", "3": "#6b2c05"}
ADAM_STYLE = {("adam", "0"): dict(color="#2a76d2", marker="o", label="Adam, wd 0"),
              ("adam", "1e-6"): dict(color="#808080", marker="o", mec="black", label="Adam, wd 1e-6 (standard)"),
              ("adamw", "1e-6"): dict(color="#1a9e77", marker="s", label="AdamW, wd 1e-6")}


def arm_of(cell):
    """Parse a B5 cell folder name.

    Args:
        cell: folder name such as 'EqType=h_N=1000_opt=sgd_wd=0_lr=0.3'.
    Returns:
        (optimizer, weight decay, learning rate) as strings.
    """
    m = re.search(r"opt=(\w+?)_wd=([^_]+)_lr=([^_/]+)", cell)
    return m.group(1), m.group(2), m.group(3)


def style_of(opt, wd, lr):
    """Marker style and legend label for one arm.

    Args:
        opt, wd, lr: the arm, as returned by arm_of.
    Returns:
        dict of matplotlib marker keywords plus 'label'.
    """
    if opt != "sgd":
        return dict(ADAM_STYLE[(opt, wd)])
    c = SGD_COLOR[lr]
    if wd == "0":
        return dict(color="none", mec=c, marker="^", mew=1.5, label=f"SGD lr {lr}, wd 0")
    return dict(color=c, mec=c, marker="^", label=f"SGD lr {lr}, wd 1e-6")


def read_arm(cell, cache):
    """Read every finished run of one arm at its floor and at the end of training.

    Args:
        cell: the arm's folder (one sub-folder per finished run).
        cache: an open heldout_r2 cache.
    Returns:
        list of dicts, one per run: name, t_star (nan if no floor read), kind (floor_kind or the
        reason there is no read), act_floor, r2_floor (nan when no read), act_end, it_end, r2_end.
    """
    rows = []
    for r in F1.runs_with_loss(cell):
        run_dir = os.path.join(cell, r["name"])
        it, L = np.asarray(r["it_loss"], float), np.asarray(r["loss"], float)
        ita, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
        f = noisy_floor(it, L)
        row = dict(name=r["name"], act_end=act[-1], it_end=ita[-1],
                   r2_end=heldout_r2(run_dir, draws=R2_DRAWS, cache=cache))
        if f["reason"]:
            row.update(t_star=np.nan, kind=f["reason"], act_floor=np.nan, r2_floor=np.nan)
        else:
            r2f, _ = floor_r2(run_dir, it, L, f, cache)
            row.update(t_star=f["t_star"], kind=f["floor_kind"],
                       act_floor=act[np.argmin(np.abs(ita - f["t_star"]))], r2_floor=r2f)
        rows.append(row)
    return rows


def main():
    """Read every B5 arm, print the per-run table, and draw the two-panel figure."""
    cache = load_cache()
    arms = []
    for cell in sorted(glob.glob(os.path.join(SWEEP, "*/"))):
        opt, wd, lr = arm_of(os.path.basename(cell.rstrip("/")))
        if opt == "sgd" and lr in SKIP_LR:
            continue
        rows = read_arm(cell, cache)
        save_cache(cache)
        if rows:
            arms.append(((opt, wd, lr), rows))
    # order: Adam arms, then SGD by learning rate, wd 1e-6 before wd 0
    arms.sort(key=lambda a: (a[0][0] == "sgd", float(a[0][2]), a[0][1] == "0"))

    print(f"{'arm':<22}{'run':<12}{'cross (k)':>10} {'kind':<14}{'act@floor':>10}{'R2@floor':>10}"
          f"{'act@end':>9}{'end (k)':>8}{'R2@end':>8}")
    for (opt, wd, lr), rows in arms:
        for r in rows:
            print(f"{opt + ' wd ' + wd + ' lr ' + lr:<22}{r['name'][:10]:<12}{r['t_star'] / 1e3:>10.0f} "
                  f"{str(r['kind'])[:13]:<14}{r['act_floor']:>10.0f}{r['r2_floor']:>10.4f}"
                  f"{r['act_end']:>9.0f}{r['it_end'] / 1e3:>8.0f}{r['r2_end']:>8.4f}")

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.4), sharey=True)
    handles = []
    for (opt, wd, lr), rows in arms:
        st = style_of(opt, wd, lr)
        n_floor = sum(np.isfinite(r["r2_floor"]) for r in rows)
        label = st.pop("label") + f" (n = {len(rows)}" + (f", {n_floor} at floor)" if n_floor < len(rows) else ")")
        kw = dict(ms=8, ls="none", mfc=st.pop("color"), **st)
        kw.setdefault("mec", kw["mfc"])
        for ax, a_key, r_key in ((axes[0], "act_floor", "r2_floor"), (axes[1], "act_end", "r2_end")):
            x = [r[a_key] for r in rows]
            y = [r[r_key] for r in rows]
            ax.plot(x, y, **kw)
        # legend entry built independently of the data, so an arm whose first run has no floor
        # read still appears
        handles.append(Line2D([], [], label=label, **kw))
    for ax in axes:
        ax.set_xlim(0, 1000)
        ax.set_ylim(0.87, 0.97)
        ax.set_xlabel("active units (of 1000)")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("held-out R²")
    axes[0].set_title("both read where each run's noisy loss first reaches 1.05x its floor\n"
                      "(R² estimated from the loss ratio: no weights were kept at the floor)", fontsize=10)
    axes[1].set_title("both read at 200k (final weights)", fontsize=10)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(0.775, 0.5), fontsize=8.5, frameon=False)
    fig.suptitle("3-bit flip-flop, N = 1000: optimizer control (B5). Filled triangles: SGD with wd 1e-6; "
                 "open: SGD with wd 0.\nSGD lr 3, wd 1e-6 fails the task (R² 0.30-0.44) and is off the "
                 "bottom; SGD lr 10 left out. Only finished runs are drawn (SGD wd 0: lr 1 has 1 of 3, lr 3 none yet).",
                 fontsize=10)
    fig.subplots_adjust(left=0.06, right=0.765, bottom=0.11, top=0.82, wspace=0.08)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    fig.savefig(OUT + ".png", dpi=150)
    fig.savefig(OUT + ".pdf")
    print("wrote", os.path.abspath(OUT) + ".{png,pdf}")


if __name__ == "__main__":
    main()
