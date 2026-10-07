#!/usr/bin/env python3
"""Two panels auditing prune-and-duplicate: what it needs its noise for, and what its units do.

26f  Each arm scored twice on the SAME held-out batch - at the sigma it trained with, and at
     sigma = 0. The gap between the two is how much the trained solution needs its own noise.
     Slide 26 makes the same measurement for synaptic noise, so the two arms are comparable.

26g  Units removed lowest-participation-first, outgoing weights zeroed, nothing retrained, against
     the same number removed at random. Answers whether duplication's recruited units carry task
     information - the question `ablate_rescued_units.py` was written for and has only ever been
     asked of the rate penalty.

Both panels compute their numbers and cache them in data/duplication_audit.json; nothing is typed.

Output: img/internal_figures/slide_26f_duplication_noise.{pdf,svg}
        img/internal_figures/slide_26g_duplication_ablation.{pdf,svg}
Usage:  python fig_duplication_audit.py [--refresh]
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
from ablate_rescued_units import ablate_curve

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from common import DATA_DIR
sys.path.insert(0, os.path.join(REPO, "trainRNNbrain", "experiments_and_analysis"))

CACHE = os.path.join(REPO, "data", "duplication_audit.json")

# (block, arm label, cell). Control and duplication share a budget inside every block.
NOISE_CELLS = [
    ("3-bit flip-flop, 40k", "no pruning", "NBitFlipFlop_ff_revive/EqType=h_k=3_N=1000_pen=none_arm=none"),
    ("3-bit flip-flop, 40k", "rate 0.001", "NBitFlipFlop_replace_variants_fix/EqType=h_k=3_N=1000_mode=copy_rate=0.001_mat=1000"),
    ("3-bit flip-flop, 40k", "rate 0.005", "NBitFlipFlop_replace_variants_fix/EqType=h_k=3_N=1000_mode=copy_rate=0.005_mat=1000"),
    ("3-bit flip-flop, 40k", "rate 0.025", "NBitFlipFlop_copy_perturb/EqType=h_k=3_N=1000_cn=0"),
    ("CDDM, 100k", "no pruning", "CDDM_paper_grid/EqType=h_N=1000_arm=control"),
    ("CDDM, 100k", "duplication", "CDDM_paper_grid/EqType=h_N=1000_arm=duplication"),
]
ABLATE_CELLS = [("no pruning", "CDDM_paper_grid/EqType=h_N=1000_arm=control"),
                ("duplication", "CDDM_paper_grid/EqType=h_N=1000_arm=duplication")]


def build(refresh=False):
    """Compute (or load) both panels' numbers.

    Args:
        refresh: recompute even if the cache exists.
    Returns:
        dict with "noise" (block, arm -> per-seed noisy and sigma=0 scores) and "ablation".
    """
    if os.path.exists(CACHE) and not refresh:
        with open(CACHE) as fh:
            return json.load(fh)
    sys.path.insert(0, "/private/tmp/claude-502/-Users-pt1290-Documents-GitHub-trainRNNbrain/"
                       "d955d9e4-4332-417e-ae30-7fac0bd00cff/scratchpad")
    from dup_audit import both_scores
    out = {"noise": [], "ablation": []}
    for block, arm, pat in NOISE_CELLS:
        dirs = [d for d in sorted(glob.glob(os.path.join(DATA_DIR, pat, "*"))) if os.path.isdir(d)]
        pairs = [both_scores(d) for d in dirs]
        pairs = [(a, b) for a, b in pairs if np.isfinite(a) and np.isfinite(b)]
        if pairs:
            out["noise"].append(dict(block=block, arm=arm,
                                     noisy=[p[0] for p in pairs], clean=[p[1] for p in pairs]))
            print(f"  {block:22s} {arm:12s} n={len(pairs)}")
    for arm, pat in ABLATE_CELLS:
        dirs = [d for d in sorted(glob.glob(os.path.join(DATA_DIR, pat, "*"))) if os.path.isdir(d)][:3]
        cs = []
        for d in dirs:
            try:
                cs.append(ablate_curve(d))
            except Exception as e:
                print(f"  ablation {arm}: skipped - {type(e).__name__}")
        if cs:
            out["ablation"].append(dict(
                arm=arm, fracs=list(np.asarray(cs[0]["fracs"], float)),
                low=list(np.mean([c["r2_low"] for c in cs], axis=0)),
                rand=list(np.mean([c["r2_rand"] for c in cs], axis=0)),
                n_active=float(np.mean([c["n_active"] for c in cs])),
                r2_full=float(np.mean([c["r2_full"] for c in cs])), n=len(cs)))
            print(f"  ablation {arm:12s} n={len(cs)}")
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(out, fh, indent=1)
    return out


def noise_slide(d, name="slide_26f_duplication_noise"):
    """Held-out score against the same networks at sigma = 0, one line per arm."""
    ps.setup()
    blocks = list(dict.fromkeys(r["block"] for r in d["noise"]))
    fig, axes = plt.subplots(1, len(blocks), figsize=(150 * ps.MM, 62 * ps.MM), sharey=False)
    axes = np.atleast_1d(axes)
    for ax, block in zip(axes, blocks):
        rows = [r for r in d["noise"] if r["block"] == block]
        for i, r in enumerate(rows):
            col = ps.BASE if "no pruning" in r["arm"] else ps.COND_COL["duplicate"]
            n, c = np.mean(r["noisy"]), np.mean(r["clean"])
            ax.plot([0, 1], [n, c], "-o", color=col, ms=3.6, lw=1.3, mec="none",
                    alpha=1.0 if "no pruning" in r["arm"] else 0.45 + 0.18 * i, zorder=4)
            ax.annotate(f"{r['arm']}  {n - c:+.3f}", (1, c), textcoords="offset points",
                        xytext=(5, 0), fontsize=6.0, color=col, va="center")
        ax.set(xlim=(-0.12, 1.75), xticks=[0, 1],
               xticklabels=["its own\nnoise", "$\\sigma = 0$"], title=block)
        ps.ygrid(ax)
    axes[0].set_ylabel("held-out $R^2$")
    # The sign convention is the one thing the panel cannot show: a positive number means the arm
    # scores BETTER in its own noise than without it.
    fig.suptitle("Each network scored twice on one held-out batch; "
                 "the number is own-noise minus $\\sigma = 0$",
                 fontsize=8.0, color=ps.INK, y=1.04)
    return ps.save(fig, name, w_mm=150)


def ablation_slide(d, name="slide_26g_duplication_ablation"):
    """Task R^2 against surviving units, removing the least active first and at random."""
    ps.setup()
    fig, ax = plt.subplots(figsize=(132 * ps.MM, 68 * ps.MM))
    for r in d["ablation"]:
        col = ps.BASE if "no pruning" in r["arm"] else ps.COND_COL["duplicate"]
        surv = 1000 * (1.0 - np.asarray(r["fracs"], float))
        ax.plot(surv, r["low"], "-o", color=col, ms=3.0, lw=1.3, mec="none", zorder=5,
                label=f"{r['arm']}  ({r['n_active']:.0f} active)")
        ax.plot(surv, r["rand"], ls=(0, (2.4, 1.6)), color=col, lw=1.0, alpha=0.55, zorder=4)
        ax.axvline(r["n_active"], color=col, lw=0.7, ls=":", alpha=0.7, zorder=2)
    ax.set(xlabel="units surviving (of 1000)", ylabel="task $R^2$")
    ps.ygrid(ax)
    ax.legend(loc="lower right", fontsize=6.8, frameon=False)
    ax.text(0.02, 0.04, "solid: least active removed first        dashed: removed at random\n"
                        "dotted vertical: that arm's active-unit count",
            transform=ax.transAxes, fontsize=6.2, color=ps.MUTED, va="bottom", linespacing=1.4)
    fig.suptitle("CDDM, $N$ = 1000, 100,000 iterations, 3 networks per arm",
                 fontsize=8.0, color=ps.INK, y=1.03)
    return ps.save(fig, name, w_mm=132)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="recompute instead of reading the cache")
    a = ap.parse_args()
    data = build(refresh=a.refresh)
    noise_slide(data)
    ablation_slide(data)
