#!/usr/bin/env python3
"""
Drift diagnostics over training for a SINGLE network: all four variables on one axes.

Every cross-condition claim in this project rests on the lag exponent alpha, which is a summary. This
plots the underlying quantities for one network so the summary can be checked against what it came
from: the raw displacement at each lag, the exponent fitted from them, and whether one power law
describes the whole lag range.

⚠️ ALPHA IS AN EXPONENT, NOT A DIFFUSION COEFFICIENT. Displacement over a lag L is modelled as
|W(t+L) - W(t)| / |W(t)| ~ L^alpha, and it is alpha that is plotted:

    alpha ~ 1.0   ballistic - updates are BIASED, the network is still travelling somewhere
    alpha ~ 0.5   diffusive - updates have decorrelated, it is jittering in place
    alpha < 0.5   confined  - mean-reverting, held inside a basin

A diffusion coefficient in the physical sense (D with displacement^2 = 2DL) is only defined once the
motion IS diffusive, i.e. once alpha ~ 0.5. Panel (b) shows the raw displacement that alpha is fitted
from, so the amplitude is visible alongside the exponent.

VARIABLES: W_inp, W_rec, W_out and the participation vector `p` (logged as dp_lag*). They settle at
different times - measured unpenalised, W_out freezes by ~5k, the participation vector by ~25k, W_rec
at ~19-41k and W_inp last at ~35-122k - so which one a criterion uses matters.

Output: img/internal_figures/drift_single_<pen>_k<k>_N<N>.png

Usage:  python plot_drift_single.py [pen] [k] [N]        e.g.  python plot_drift_single.py both 2 500
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, LAGS, drift_alpha, drift_alpha_pairwise, series
import plotstyle as ps

ROOTS = ["data/trained_RNNs/NBitFlipFlop_std_penlong",
         "data/trained_RNNs/NBitFlipFlop_std_pen",
         "data/trained_RNNs/NBitFlipFlop_std_ksweep",
         "data/trained_RNNs/NBitFlipFlop_std_bigN"]
VARS = ["W_inp", "W_rec", "W_out", "p"]
VLAB = {"W_inp": "$W_{inp}$", "W_rec": "$W_{rec}$", "W_out": "$W_{out}$",
        "p": "participation vector"}
VCOL = {"W_inp": "C1", "W_rec": "C0", "W_out": "C2", "p": "C3"}
THRESH = 0.6


def find(pen, k, N):
    """Locate one trace matching a condition, preferring the longest budget.

    Args:
        pen: "none"/"rws"/"frm"/"both"; k: bits; N: units.
    Returns:
        (path, trace dict) for the median-length match.
    Raises:
        SystemExit naming what was searched if nothing matches.
    """
    hits = []
    for root in ROOTS:
        for f in sorted(glob.glob(os.path.join(root, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(r"_k=(\d+)_N=(\d+)(?:_pen=([a-z]+))?", f)
            if not m:
                continue
            if (m.group(3) or "none") == pen and int(m.group(1)) == k and int(m.group(2)) == N:
                hits.append(f)
    if not hits:
        raise SystemExit(f"no trace for pen={pen} k={k} N={N} under:\n  " + "\n  ".join(ROOTS))
    hits.sort(key=lambda f: os.path.getsize(f))
    f = hits[len(hits) // 2]                       # median by size = median budget
    with open(f, "rb") as fh:
        return f, pickle.load(fh)


def main():
    """Plot alpha, raw displacement and per-decade alpha for one network."""
    ps.setup()
    pen = sys.argv[1] if len(sys.argv) > 1 else "both"
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 2
    N = int(sys.argv[3]) if len(sys.argv) > 3 else 500
    path, tr = find(pen, k, N)
    budget = len(tr["metrics"].get("loss_clean_train", [])) * 10
    print(f"{pen} k={k} N={N}  ({budget:,} iterations)\n  {os.path.basename(os.path.dirname(path))}")

    fig, ax = plt.subplots(1, 3, figsize=(17, 5.2))

    # (a) the exponent, which every criterion in the project thresholds
    # ⚠️ Per-probe alpha MIXES TWO ESTIMATORS. drift_alpha fits across whichever lags are finite at
    # that probe, and the Trainer logs lag100 every 100 iterations, lag1000 every 1000, lag10000
    # every 10000 - so 9 probes in 10 use two lags and every tenth uses three. Measured on this
    # network: W_inp 2-lag median 0.849 (sd 0.224) vs 3-lag 0.795 (sd 0.128); participation 0.096
    # (sd 0.339) vs 0.239 (sd 0.195). The three-lag estimate spans the full decade range and is
    # markedly less noisy, so it is drawn bold and the two-lag points are shown faint behind it.
    # Never quote alpha[-1]: the final probe is often a two-lag outlier (0.05 on this network,
    # against a tail median of 0.72).
    print("\n  alpha, median over the final 50k (NOT the last probe, which is a noisy outlier):")
    print("    %-8s %10s %10s %10s" % ("variable", "tail med", "last probe", "3-lag only"))
    for v in VARS:
        it, al = drift_alpha(tr, v)
        if not len(al):
            continue
        w = 15
        sm = np.array([np.median(al[max(0, i - w // 2):i + w // 2 + 1]) for i in range(len(al))])
        ax[0].plot(it, sm, color=VCOL[v], lw=2.0, label=VLAB[v])
        ax[0].plot(it, al, color=VCOL[v], lw=0.5, alpha=.20)
        tail = al[it > it.max() - 50_000]
        three = al[np.isclose(np.mod(it, 10_000), 0)]
        t3 = three[three.size // 2:]
        print("    %-8s %10.3f %10.3f %10.3f"
              % (v, np.median(tail), al[-1], np.median(t3) if len(t3) else np.nan))
    ax[0].axhline(1.0, color="k", ls="--", lw=1, alpha=.6)
    ax[0].axhline(0.5, color="k", ls="-", lw=1, alpha=.6)
    ax[0].axhline(THRESH, color="C3", ls=":", lw=1.4)
    for y, t in [(1.0, "ballistic — still travelling"), (0.5, "diffusive — jitter"),
                 (THRESH, f"threshold {THRESH}")]:
        ax[0].text(2.2e3, y + .015, t, fontsize=7.5,
                   color="C3" if y == THRESH else "k")
    ax[0].set(xscale="log", xlabel="iteration", ylabel=r"lag exponent $\alpha$", ylim=(-0.05, 1.25),
              title=f"(a) $\\alpha$ over training — {pen}, k={k}, N={N}\n"
                    "faint = per probe (noisy: mixes 2- and 3-lag fits), bold = running median")
    ax[0].legend(fontsize=8, loc="lower left")

    # (b) the raw displacement alpha is fitted from, so amplitude is visible next to exponent
    for v in VARS:
        for L, ls in zip(LAGS, ["-", "--", ":"]):
            key = f"dp_lag{L}" if v == "p" else f"drift_{v}_lag{L}"
            if key not in tr["metrics"]:
                continue
            i, d = series(tr, key)
            if len(d):
                ax[1].plot(i, d, ls, color=VCOL[v], lw=1.3, alpha=.85,
                           label=f"{VLAB[v]} lag {L}" if L == LAGS[1] else None)
    ax[1].set(xscale="log", yscale="log", xlabel="iteration",
              ylabel=r"$\|\Delta W\|/\|W\|$ over the lag",
              title="(b) the raw displacement $\\alpha$ is fitted from\n"
                    "line style = lag (100 / 1000 / 10000)")
    ax[1].legend(fontsize=7.5)

    # (c) does ONE power law describe the whole lag range?
    for v in VARS:
        pw = drift_alpha_pairwise(tr, v)
        for (l1, l2), (i, a) in pw.items():
            ls = "-" if l1 == LAGS[0] else "--"
            w = 5
            sm = np.array([np.median(a[max(0, j - w // 2):j + w // 2 + 1]) for j in range(len(a))])
            ax[2].plot(i, sm, ls, color=VCOL[v], lw=1.6,
                       label=f"{VLAB[v]}  {l1}→{l2}")
    ax[2].axhline(0.5, color="k", lw=1, alpha=.6)
    ax[2].axhline(THRESH, color="C3", ls=":", lw=1.4)
    ax[2].set(xscale="log", xlabel="iteration", ylabel=r"$\alpha$ from ONE decade", ylim=(-0.05, 1.35),
              title="(c) is one power law enough?\nsolid and dashed apart = no")
    ax[2].legend(fontsize=6.5, ncol=2)

    fig.suptitle(f"Drift diagnostics for a single {pen} network (k={k}, N={N}, "
                 f"{budget:,} iterations) — $\\alpha$ is an EXPONENT, not a diffusion coefficient",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, f"drift_single_{pen}_k{k}_N{N}", tight=False)


if __name__ == "__main__":
    main()
