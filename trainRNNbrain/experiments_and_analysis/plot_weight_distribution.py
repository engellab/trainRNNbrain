#!/usr/bin/env python3
"""Pooled recurrent-weight |W_rec| distribution (log-log) — verify the reflective boundary was active.

Sanity check for weight_boundary: the "sticky" clamp sets Dale-violating W_rec entries to exactly
+/-eps (1e-12), so |W_rec| must show a spike at 1e-12; "reflective" uses |param|*sign and should
have NO spike there.

Two panels (top: sticky, bottom: reflective). Each overlays N=1000, h-equation nets for two
conditions — no penalties (Lrws=0, Lfr=0) and both penalties (Lrws=0.05, Lfr=0.2) — pooling the
off-diagonal |W_rec| entries across the 5 nets of each condition (same colour scheme as the other
figures). A dashed line marks eps=1e-12.

Output: <repo>/img/internal_figures/weight_distribution_sticky_vs_reflective_h_N1000.png
Run from this directory:  python plot_weight_distribution.py
"""
import os
import glob
import json
import numpy as np
from matplotlib import pyplot as plt

from trainRNNbrain.utils import unjsonify

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../../data/trained_RNNs")
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
OUT = os.path.join(IMG_DIR, "weight_distribution_sticky_vs_reflective_h_N1000.png")
EPS = 1e-12

SWEEPS = [("sticky", "CDDM_4a031e_g0"), ("reflective", "CDDM_2bc3c1_g0_reflective")]
CONDS = [
    ("none  (Lrws=0, Lfr=0)",       "#c44e52", "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"),
    ("both  (Lrws=0.05, Lfr=0.2)",  "#4c72b0", "EqType=h_N=1000_LmbdRWS=0.05_LmbdFR=0.2"),
]


def pooled_offdiag_absW(sweep_folder, cond_folder):
    """Pool |W_rec| off-diagonal entries across all nets of one condition.

    Args:
        sweep_folder: sweep dir name under data/trained_RNNs (e.g. CDDM_4a031e_g0).
        cond_folder:  exact condition dir name (e.g. EqType=h_N=1000_LmbdRWS=0_LmbdFR=0).
    Returns:
        1D ndarray of |W_rec[i,j]| for i!=j, concatenated over the condition's nets.
    """
    vals = []
    for d in sorted(glob.glob(os.path.join(DATA, sweep_folder, cond_folder, "*"))):
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))
        if not pj:
            continue
        W = np.asarray(unjsonify(json.load(open(pj[0])))["W_rec"], dtype=float)
        off = ~np.eye(W.shape[0], dtype=bool)
        vals.append(np.abs(W[off]))
    return np.concatenate(vals) if vals else np.array([])


def main():
    data = {(sf, cf): pooled_offdiag_absW(sf, cf) for _, sf in SWEEPS for _, _, cf in CONDS}
    allpos = np.concatenate([v[v > 0] for v in data.values()])
    bins = np.logspace(np.log10(allpos.min() * 0.5), np.log10(allpos.max() * 1.5), 70)

    print(f"{'boundary':11s} {'condition':45s} {'frac |W|~1e-12':>14s}  {'min |W|':>10s}")
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)
    for ax, (bname, sf) in zip(axes, SWEEPS):
        for label, color, cf in CONDS:
            v = data[(sf, cf)]
            frac = float(np.mean((v >= 0.5 * EPS) & (v <= 2 * EPS)))  # mass at the clamp value
            print(f"{bname:11s} {cf:45s} {frac:14.3f}  {v[v>0].min():10.2e}")
            vp = v[v > 0]
            ax.hist(vp, bins=bins, histtype="stepfilled", color=color, alpha=0.12, linewidth=0)
            ax.hist(vp, bins=bins, histtype="step", color=color, linewidth=1.7, label=label)
        ax.axvline(EPS, ls="--", color="0.4", lw=1)
        ax.set_xscale("log")
        ax.set_yscale("log")
        ax.set_ylabel(f"{bname}\ncount (off-diag |W_rec|)")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].text(EPS, axes[0].get_ylim()[1], " eps=1e-12 (sticky clamp)", color="0.4",
                 fontsize=8, va="top", ha="left")
    axes[0].legend(title="penalty", frameon=False, fontsize=9, loc="upper center")
    axes[0].set_title("Pooled |W_rec| distribution — h, N=1000, gamma=0 — "
                      "sticky (top) vs reflective (bottom)")
    axes[-1].set_xlabel("|W_rec|  (off-diagonal)")
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
