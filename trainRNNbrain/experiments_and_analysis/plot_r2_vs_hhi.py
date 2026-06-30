#!/usr/bin/env python3
"""
R^2 vs HHI-of-participation scatter for the N=1000 nets of the Silent-ReLU CDDM sweep (CDDM_4a031e).

For each N=1000 network we reconstruct the RNN from LastParams, run the noise-free CDDM batch,
and compute per-unit participation (= std(rate) + 0.9-quantile(|rate|) over time & trials; same
metric as participation.png). We then summarise that net by:

  - HHI of participation: H = sum_i s_i^2 with s_i = p_i / sum_j p_j  (Herfindahl-Hirschman
    Index, the concentration of participation across units). 1/H is the effective number of
    participating units; H = 1/N (vertical dashed line) is perfectly even participation, large
    H means a few units dominate (the rest are silent/near-silent).
  - validation R^2: the score in the net's folder name (computed by run_experiment.py).

One point per net (5 nets x 4 penalty conditions = 20 per equation type), coloured by the
(lambda_rws, lambda_frm) penalty combination (same scheme as the other sweep figures). The two
equation types (h, s) are drawn as separate panels.

This is the diagnostic for "does lambda_rws add anything alongside lambda_frm?": compare the
'fr only' (green) and 'both' (blue) clouds in HHI (participation evenness) and R^2 (performance).

Output: <repo>/img/internal_figures/r2_vs_hhi_N1000.png

Run from this directory:  python plot_r2_vs_hhi.py
"""
import os
import re
import sys
import csv
import glob
import numpy as np
from matplotlib import pyplot as plt

from plot_participation_histograms import build_cddm_batch, net_participation, PENALTY, EQ_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = sys.argv[1] if len(sys.argv) > 1 else "CDDM_4a031e"   # sweep folder under data/trained_RNNs
TAG = SWEEP.replace("CDDM_", "")
SUF = "" if SWEEP == "CDDM_4a031e" else f"_{TAG}"
ROOT = os.path.join(HERE, "../../data/trained_RNNs", SWEEP)
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")

COND_RE = re.compile(r"EqType=(?P<eq>[hs])_N=(?P<N>\d+)_LmbdRWS=(?P<rws>[\d.]+)_LmbdFR=(?P<frm>[\d.]+)")
N_TARGET = 1000
OUT = os.path.join(IMG_DIR, f"r2_vs_hhi_N1000{SUF}.png")
CACHE_CSV = os.path.join(ROOT, "r2_hhi_N1000.csv")   # per-net (eq, rws, frm, hhi, r2); skips forward passes


def hhi(participation):
    """Herfindahl-Hirschman Index of a participation vector.

    Args:
        participation: ndarray (N,) of per-unit participation (>= 0; silent units ~ 0).
    Returns:
        float H = sum_i (p_i / sum_j p_j)^2 over finite units, or nan if total is non-positive.
        H in [1/N, 1]; 1/N = perfectly even, larger = concentrated in fewer units.
    """
    p = participation[np.isfinite(participation)]
    tot = p.sum()
    if tot <= 0 or p.size == 0:
        return np.nan
    s = p / tot
    return float(np.sum(s * s))


def r2_from_dir(leaf_dir):
    """Validation R^2 of a net, parsed from the leading score in its folder name."""
    return float(os.path.basename(leaf_dir).split("_")[0])


def compute_rows():
    """Run the N=1000 nets and return rows (eq, rws, frm, hhi, r2). Slow (forward passes)."""
    leaf_dirs = sorted(d for d in glob.glob(os.path.join(ROOT, f"EqType=*_N={N_TARGET}_*", "*"))
                       if os.path.isdir(d))
    if not leaf_dirs:
        raise SystemExit(f"No N={N_TARGET} net folders under {ROOT}")
    sample_cfg = glob.glob(os.path.join(leaf_dirs[0], "*_config.yaml"))[0]
    input_batch = build_cddm_batch(sample_cfg)
    print(f"computing HHI + R^2 for {len(leaf_dirs)} N={N_TARGET} nets...")
    rows = []
    for d in leaf_dirs:
        m = COND_RE.search(os.path.basename(os.path.dirname(d)))
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))
        if not pj:
            print(f"  [skip] no LastParams in {d}");  continue
        H = hhi(net_participation(pj[0], input_batch))
        rows.append((m["eq"], m["rws"], m["frm"], H, r2_from_dir(d)))
    return rows


def main():
    recompute = "--recompute" in sys.argv
    if os.path.exists(CACHE_CSV) and not recompute:
        print(f"loading cached HHI+R^2 from {os.path.normpath(CACHE_CSV)} (use --recompute to rebuild)")
        with open(CACHE_CSV) as f:
            rows = [(r["eq"], r["rws"], r["frm"], float(r["hhi"]), float(r["r2"]))
                    for r in csv.DictReader(f)]
    else:
        rows = compute_rows()
        with open(CACHE_CSV, "w", newline="") as f:
            w = csv.writer(f); w.writerow(["eq", "rws", "frm", "hhi", "r2"]); w.writerows(rows)
        print(f"cached -> {os.path.normpath(CACHE_CSV)}")

    points = {}
    for eq, rws, frm, H, R in rows:
        points.setdefault((eq, rws, frm), []).append((H, R))

    all_inv = [1.0 / H for *_, H, _ in rows if np.isfinite(H) and H > 0]
    xlo, xhi = 0.7 * min(all_inv), 1.4 * N_TARGET

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.6), sharex=True, sharey=True)
    for ax, eq in zip(axes, ("h", "s")):
        for (rws, frm), (label, color) in PENALTY:
            pts = points.get((eq, rws, frm), [])
            if not pts:
                continue
            invH = np.array([1.0 / p[0] for p in pts]); R = np.array([p[1] for p in pts])
            ax.scatter(invH, R, s=70, color=color, edgecolor="black", linewidth=0.5,
                       alpha=0.9, label=label, zorder=3)
        ax.axvline(N_TARGET, ls="--", color="0.5", lw=1, zorder=1)
        ax.text(N_TARGET, ax.get_ylim()[0], "even (N=%d) " % N_TARGET, color="0.5",
                fontsize=8, va="bottom", ha="right")
        ax.set_xscale("log")
        ax.set_xlim(xlo, xhi)
        ax.set_xlabel("1/HHI of participation   ≈   # participating units")
        ax.set_title(f"{EQ_NAME[eq]} equation")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("validation $R^2$")
    axes[0].legend(title="penalty", frameon=False, fontsize=9, loc="center")
    fig.suptitle(f"Task performance vs participation spread — N={N_TARGET} "
                 f"CDDM ReLU-Dale nets ({TAG})", y=1.0)
    plt.tight_layout()

    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
