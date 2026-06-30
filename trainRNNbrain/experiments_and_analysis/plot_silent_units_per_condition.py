#!/usr/bin/env python3
"""
Plot the number of silent units per condition for the Silent-ReLU CDDM sweep (CDDM_4a031e).

Reads the per-condition silent-unit table produced by count_silent_units.py
(<repo>/data/trained_RNNs/CDDM_4a031e/silent_units_per_condition.csv) and draws a grouped
bar chart: one bar per condition (equation type x N x lambda_rws x lambda_frm), grouped by
(equation type, N) and coloured by the (lambda_rws, lambda_frm) penalty combination, with
std error bars over the 5 networks in each condition.

A unit is "silent" if its peak firing rate over the noise-free CDDM validation batch is
< 0.01 (the 'dead_abs' criterion in count_silent_units.py). The integer above each bar is
the mean silent-unit count; the percent is the fraction of that condition's N units.

Output: <repo>/img/internal_figures/silent_units_per_condition.png

Run from this directory:  python plot_silent_units_per_condition.py
"""
import os
import sys
import csv
import re
import numpy as np
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = sys.argv[1] if len(sys.argv) > 1 else "CDDM_4a031e"   # sweep folder under data/trained_RNNs
TAG = SWEEP.replace("CDDM_", "")
SUF = "" if SWEEP == "CDDM_4a031e" else f"_{TAG}"             # suffix non-default sweeps' figures
CSV = os.path.join(HERE, "../../data/trained_RNNs", SWEEP, "silent_units_per_condition.csv")
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
OUT = os.path.join(IMG_DIR, f"silent_units_per_condition{SUF}.png")

COND_RE = re.compile(r"EqType=(?P<eq>[hs])_N=(?P<N>\d+)_LmbdRWS=(?P<rws>[\d.]+)_LmbdFR=(?P<frm>[\d.]+)")

# (lambda_rws, lambda_frm) -> (legend label, colour). Order = bar order within each group.
PENALTY = [
    (("0", "0"),      ("none  (Lrws=0, Lfr=0)",      "#c44e52")),   # unmodified baseline
    (("0.05", "0"),   ("rws only  (Lrws=0.05)",      "#dd8452")),
    (("0", "0.2"),    ("fr only  (Lfr=0.2)",         "#55a868")),
    (("0.05", "0.2"), ("both  (Lrws=0.05, Lfr=0.2)", "#4c72b0")),
]
EQ_NAME = {"h": "h (hidden)", "s": "s (rate)"}


def load(csv_path):
    """Read the silent-unit CSV into a dict keyed by (eq, N, rws, frm).

    Args:
        csv_path: path to silent_units_per_condition.csv written by count_silent_units.py.
    Returns:
        dict {(eq:str, N:int, rws:str, frm:str): {mean, std, pct, n_nets}} for the
        'dead_abs' (peak rate < 0.01) silent-unit criterion.
    """
    rows = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            m = COND_RE.search(r["condition"])
            key = (m["eq"], int(m["N"]), m["rws"], m["frm"])
            rows[key] = dict(mean=float(r["dead_abs_mean"]), std=float(r["dead_abs_std"]),
                             pct=float(r["dead_abs_pct"]), n_nets=int(r["n_nets"]))
    return rows


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"Missing {CSV}\nRun count_silent_units.py first to generate it.")
    rows = load(CSV)

    groups = [(eq, N) for eq in ("h", "s") for N in (100, 500, 1000)]   # 6 x-groups
    width = 0.2
    offsets = np.array([-1.5, -0.5, 0.5, 1.5]) * width

    fig, ax = plt.subplots(figsize=(13, 6))
    x = np.arange(len(groups))

    for pi, ((rws, frm), (label, color)) in enumerate(PENALTY):
        means = np.array([rows.get((eq, N, rws, frm), {}).get("mean", np.nan) for eq, N in groups])
        stds = np.array([rows.get((eq, N, rws, frm), {}).get("std", 0.0) for eq, N in groups])
        pcts = [rows.get((eq, N, rws, frm), {}).get("pct", np.nan) for eq, N in groups]
        pos = x + offsets[pi]
        ax.bar(pos, means, width=width, yerr=stds, capsize=2, color=color,
               edgecolor="black", linewidth=0.4, label=label, error_kw=dict(lw=0.8))
        # annotate count (and percent) above each bar
        for xp, m, p in zip(pos, means, pcts):
            if np.isnan(m):
                continue
            txt = f"{m:.0f}" if m < 1 else f"{m:.0f}\n{p:.0f}%"
            ax.text(xp, m + 0.01 * ax.get_ylim()[1], txt, ha="center", va="bottom",
                    fontsize=6.5, rotation=0)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{EQ_NAME[eq]}\nN={N}" for eq, N in groups])
    ax.set_ylabel("Number of silent units  (peak firing rate < 0.01)")
    ax.set_title(f"Silent units per condition — CDDM ReLU-Dale sweep "
                 f"({TAG}, mean ± std over 5 nets/condition)")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(title="penalty", frameon=False, ncol=2, loc="upper left")
    ax.margins(y=0.12)
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
