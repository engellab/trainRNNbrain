#!/usr/bin/env python3
"""
Figure for elaboration claim S2: what the penalties cost in task performance (CDDM: nothing; flip-flop: a
10-15% higher loss floor, about one point of R^2).

Two panels, task loss of the four conditions (none / rws / frm / both) vs N. Only the TASK term,
noise off - never TrainLosses.json, which is task + lambda*penalty with noise on (paper.md §7).

  left   CDDM: noise-free masked MSE of each network's final parameters on the training batch
         (penalty_matched.clean_loss; the same quantity as panel (d) of penalty_matched.png).
         Sweeps CDDM_std_g0_drift (none) and CDDM_std_g0_penalties (rws / frm / both), N = 500..5000.
  right  flip-flop, k = 3: each run's fitted loss floor of the noise-free task loss recorded during
         training (loss_clean_train; pr_matrix.fit_floor over the run's own budget), N = 500..2000.
         The floor rather than the last value, because the penalised runs have different budgets and
         the floor is what each network converges to.

Usage:  python fig_S2_cost.py
Output: img/internal_figures/fig_S2_cost.png
"""

import os
import sys
import glob
import numpy as np
import pickle
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
import pr_matrix
from penalty_matched import build_batches, clean_loss, collect

PENS = ["none", "rws", "frm", "both"]
COL = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
LABEL = {"none": "no penalty", "rws": "sparsity only", "frm": "participation only", "both": "participation + sparsity"}
FF_K = 3
CACHE = "data/fig_S2_cost_cache.pkl"      # ponytail: the CDDM half simulates ~45 nets (~20 min); cache it. Delete to recompute.


def cddm_losses():
    """Noise-free task loss per network, CDDM, four conditions.

    Returns:
        dict pen -> dict N -> list of losses over seeds.
    """
    runs = collect()
    cfg = OmegaConf.load(glob.glob(os.path.join(next(iter(runs.values()))[0]["folder"], "*_config.yaml"))[0])
    mask, train, heldout = build_batches(cfg)
    out = {p: {} for p in PENS}
    for (N, pen), rr in sorted(runs.items()):
        for r in rr:
            res = clean_loss(r["folder"], mask, train, heldout)
            if res:
                out[pen].setdefault(N, []).append(res[1])
                print(f"CDDM {pen:5s} N={N}: {res[1]:.5f}", flush=True)
    return out


def flipflop_floors():
    """Fitted floor of the noise-free task loss per run, flip-flop k = FF_K, four conditions.

    Returns:
        dict pen -> dict N -> list of floors over seeds.
    """
    out = {p: {} for p in PENS}
    for r in pr_matrix.load():
        if r["k"] != FF_K:
            continue
        fl = pr_matrix.fit_floor(r["loss"], r["budget"])
        if fl is not None and np.isfinite(fl) and fl > 0:
            out[r["pen"]].setdefault(r["N"], []).append(float(fl))
    return out


def panel(ax, data, title, ylabel):
    """One loss-vs-N panel, four conditions, mean ± sd over seeds."""
    for p in PENS:
        Ns = sorted(data[p])
        if not Ns:
            continue
        mu = [np.mean(data[p][N]) for N in Ns]; sd = [np.std(data[p][N]) for N in Ns]
        ax.errorbar(Ns, mu, yerr=sd, fmt="o-", color=COL[p], capsize=2, label=LABEL[p])
        print(f"{title[:9]} {p:5s} " + "  ".join(f"N={N}: {m:.5f}±{s:.5f} (n={len(data[p][N])})" for N, m, s in zip(Ns, mu, sd)))
    ax.set(xscale="log", xlabel="network size N", ylabel=ylabel, title=title, ylim=(0, None))
    ax.legend(loc="lower left")


def main():
    """Draw the two panels and write fig_S2_cost.png."""
    ps.setup()
    if os.path.exists(CACHE):
        cd = pickle.load(open(CACHE, "rb"))
    else:
        cd = cddm_losses()
        pickle.dump(cd, open(CACHE, "wb"))
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    panel(ax[0], cd, "CDDM: noise-free task loss, final weights", "masked MSE, noise off")
    panel(ax[1], flipflop_floors(), f"flip-flop k={FF_K}: fitted floor of the noise-free task loss", "loss floor (task term only)")
    fig.suptitle("S2 — task cost of the penalties: none on CDDM (a gain at large N); a 10–15% higher loss floor on the flip-flop", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "fig_S2_cost", tight=False)


if __name__ == "__main__":
    main()
