#!/usr/bin/env python3
"""
Silent-unit fraction vs recurrent training-noise level for the CDDM_fb2792_g0_noise sweep.

Reads silent_units_per_condition.csv (written by count_silent_units_noise.py) and draws
silent fraction (mean +/- std over the 5 nets/condition) as a function of the recurrent
noise sigma_rec the network was TRAINED with, one line per equation type (h, s). Both the
absolute (peak<0.01) and scale-free (peak<5% of net p95) silent criteria are shown.

All nets are scored on the same noise-free CDDM batch, so this isolates how much silence
gets baked into the trained weights as a function of training noise — it is not a readout
of instantaneous noise-driven silencing.

Output: <repo>/img/internal_figures/silent_vs_noise_fb2792_g0_noise.png

Run from this directory:  python3 plot_silent_vs_noise.py
"""
import os
import csv
import numpy as np
from matplotlib import pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "../../data/trained_RNNs/CDDM_fb2792_g0_noise/silent_units_per_condition.csv")
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
OUT = os.path.join(IMG_DIR, "silent_vs_noise_fb2792_g0_noise.png")

EQ_STYLE = {"h": ("h (hidden)", "#c44e52"), "s": ("s (rate)", "#4c72b0")}


def load(csv_path):
    """Read the noise-sweep silent CSV into {eq: (sigmas, dead_pct, dead_std, rel_pct, rel_std)}.

    Args:
        csv_path: path to silent_units_per_condition.csv (must have eq, sigma_rec,
            dead_abs_pct/std, silent_rel_pct/std, N columns).
    Returns:
        dict keyed by eq ('h'/'s') -> tuple of 1-D arrays sorted by sigma_rec:
        (sigma_rec, dead_abs_pct, dead_abs_std_pct, silent_rel_pct, silent_rel_std_pct).
    """
    by_eq = {}
    with open(csv_path) as f:
        for r in csv.DictReader(f):
            eq = r["eq"]
            N = float(r["N"])
            by_eq.setdefault(eq, []).append((
                float(r["sigma_rec"]),
                float(r["dead_abs_pct"]), 100 * float(r["dead_abs_std"]) / N,
                float(r["silent_rel_pct"]), 100 * float(r["silent_rel_std"]) / N,
            ))
    out = {}
    for eq, recs in by_eq.items():
        recs.sort()
        out[eq] = tuple(np.array(c) for c in zip(*recs))
    return out


def main():
    if not os.path.exists(CSV):
        raise SystemExit(f"Missing {CSV}\nRun count_silent_units_noise.py first.")
    data = load(CSV)

    fig, ax = plt.subplots(figsize=(8, 5.5))
    for eq, (sig, dpct, dstd, rpct, rstd) in data.items():
        label, color = EQ_STYLE[eq]
        # scale-free (relative) criterion — solid, filled markers
        ax.errorbar(sig, rpct, yerr=rstd, marker="o", color=color, capsize=3,
                    lw=2, label=f"{label} — silent<5% p95")
        # absolute (peak<0.01) — dashed, open markers
        ax.errorbar(sig, dpct, yerr=dstd, marker="s", color=color, capsize=3,
                    lw=1.5, ls="--", mfc="white", label=f"{label} — dead<0.01")

    ax.set_xlabel("recurrent training noise  σ_rec")
    ax.set_ylabel("silent-unit fraction  (%)")
    ax.set_title("Silent fraction vs recurrent noise — CDDM ReLU-Dale, N=1000, γ=0\n"
                 "(ReLU 'none' penalty; scored noise-free; mean ± std, 5 nets/σ)")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=200)
    print(f"wrote {os.path.normpath(OUT)}")


if __name__ == "__main__":
    main()
