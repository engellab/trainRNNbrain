#!/usr/bin/env python3
"""
Figure for elaboration claim C1: silencing afflicts any non-negative activation, not just ReLU.

Two panels, one per task; no network is simulated:

  CDDM       N = 1000, h equation, no penalty: ReLU vs softplus(25) vs leaky-ReLU (the 2026-07-01 sweeps;
             E1 reruns them on the standard network and this panel is then redrawn from CDDM_std_g0_activations).
             Active units under the scale-free criterion (bars) and the hard criterion
             (markers). Softplus has NO hard-silent unit (its floor is soft) and the same
             scale-free count: the criterion must be scale-free to travel across activations.
             Source: silent_units_per_condition.csv written by count_silent_units.py in
             CDDM_2bc3c1_g0_reflective, CDDM_fb2792_g0_softplus25, CDDM_fb2792_g0_leakyrelu.
  flip-flop  k = 3, N = 1000, no penalty: ReLU vs bounded sigmoid(7.5(x - 0.3)), active units
             (scale-free) along training from ParticipationTrace.pkl, every seed. The sigmoid runs stop
             at 150k, where the two activations are compared; ReLU continues to 500k.

Usage:  python fig_C1_activations.py
Output: img/internal_figures/fig_C1_activations.png
"""

import os
import sys
import csv
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_REL
import plotstyle as ps

CDDM_ACT = [("ReLU", "CDDM_2bc3c1_g0_reflective"),
            ("softplus (β=25)", "CDDM_fb2792_g0_softplus25"),
            ("leaky-ReLU (0.01)", "CDDM_fb2792_g0_leakyrelu")]
CDDM_ROW = "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"
FF_N = 1000
FF_TRACES = {"ReLU": f"data/trained_RNNs/NBitFlipFlop_std_ksweep/EqType=h_k=3_N={FF_N}_iters=500000",
             "sigmoid": f"data/trained_RNNs/NBitFlipFlop_std_sigmoid/EqType=h_k=3_N={FF_N}_iters=150000"}
MATCH_ITER = 150_000
COL = {"CDDM": "#1f77b4", "ReLU": "#7f7f7f", "sigmoid": "#9467bd"}


def cddm_activation_rows():
    """Active-unit counts at N = 1000, h equation, no penalty, per activation, under both criteria.

    Returns:
        list of (label, active_hard, sd, active_scalefree, sd, n_nets), counts of units.
    """
    rows = []
    for label, sweep in CDDM_ACT:
        with open(f"data/trained_RNNs/{sweep}/silent_units_per_condition.csv") as f:
            r = next(r for r in csv.DictReader(f) if r["condition"] == CDDM_ROW)
        N = float(r["N"])
        rows.append((label, N - float(r["dead_abs_mean"]), float(r["dead_abs_std"]),
                     N - float(r["silent_rel_mean"]), float(r["silent_rel_std"]), int(r["n_nets"])))
    return rows


def ff_traces(act):
    """Active-unit count (scale-free criterion) along training for every seed of one flip-flop activation.

    Args:
        act: 'ReLU' or 'sigmoid'.
    Returns:
        list of (iters (n,), active_count (n,)) per seed.
    """
    out = []
    for f in sorted(glob.glob(os.path.join(FF_TRACES[act], "*", "*ParticipationTrace.pkl"))):
        d = pickle.load(open(f, "rb"))
        P = np.asarray(d["participation"], dtype=float)
        q = np.quantile(P, 0.95, axis=1, keepdims=True)
        out.append((np.asarray(d["participation_iters"]), (P >= SILENT_REL * q).sum(1)))
    return out


def main():
    """Draw the two panels and write fig_C1_activations.png."""
    ps.setup()
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))

    rows = cddm_activation_rows()
    x = np.arange(len(rows))
    ax[0].bar(x, [r[3] for r in rows], 0.55, yerr=[r[4] for r in rows], color=COL["CDDM"], label="scale-free: p < 5% of q95(p)")
    ax[0].scatter(x, [r[1] for r in rows], marker="D", color="0.2", zorder=4, label="hard: peak rate < 0.01")
    for xi, r in zip(x, rows):
        ax[0].text(xi, r[3] + 25, f"{r[3]:.0f}", ha="center", fontsize=8)
        print(f"CDDM {r[0]}: active scale-free {r[3]:.0f}±{r[4]:.0f}  hard {r[1]:.0f}±{r[2]:.0f}  n={r[5]}")
    ax[0].axhline(1000, color="0.5", lw=0.8, ls=":")
    ax[0].text(-0.4, 1010, "N = 1000", fontsize=7.5, color="0.4")
    ax[0].set(xticks=x, xticklabels=[f"{r[0]}\n(n={r[5]})" for r in rows], ylabel="active units", ylim=(0, 1100),
              title="CDDM, N=1000, no penalty")
    ax[0].legend(loc="upper right")

    for act in ("ReLU", "sigmoid"):
        for i, (it, sf) in enumerate(ff_traces(act)):
            ax[1].plot(it[1:], sf[1:], color=COL[act], alpha=.75, lw=1.3, label=act if i == 0 else None)
            print(f"flip-flop {act} N={FF_N} seed{i}: active {sf[np.searchsorted(it, MATCH_ITER, side='right') - 1]:.0f} at {MATCH_ITER}")
    ax[1].axvline(MATCH_ITER, color="0.5", lw=0.8, ls=":")
    ax[1].text(MATCH_ITER * 0.93, 30, "sigmoid runs end", fontsize=7.5, color="0.4", ha="right")
    ax[1].axhline(FF_N, color="0.5", lw=0.8, ls=":")
    ax[1].set(xscale="log", xlabel="training iteration", ylabel="active units (scale-free)", ylim=(0, FF_N * 1.1),
              title=f"flip-flop k=3, N={FF_N}, no penalty")
    ax[1].legend(loc="upper left")

    fig.suptitle("C1 — ReLU, softplus, leaky-ReLU and a bounded sigmoid keep the same number of units active", fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "fig_C1_activations", tight=False)


if __name__ == "__main__":
    main()
