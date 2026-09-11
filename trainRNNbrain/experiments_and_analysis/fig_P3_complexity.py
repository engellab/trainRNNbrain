#!/usr/bin/env python3
"""
Elaboration subclaim P(c): task complexity barely changes the number of active units - and the
apparent increase at fixed compute is convergence depth. Flip-flop only (CDDM has one complexity).

Two panels, unpenalised ReLU, scale-free criterion, k = 1..8, N = 500..4000, every run: the active
count divided by the fitted size dependence, M / (A N^b), against k, so what is left is the k
dependence alone. Left: every run read at the same iteration (100k, the largest budget every cell
reaches). Right: every run read where its loss first reaches 1.10x its own fitted floor. The joint
law M = A N^b k^c is fitted on each panel's read-out (pr_matrix.fit_power_law, bootstrap CI on c).

Usage:  python fig_P3_complexity.py
Output: img/internal_figures/fig_P3_complexity.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import active_count
import plotstyle as ps
import pr_matrix
from fig_P_active_units import flipflop_runs, FIXED_ITER


def panel(ax, runs, title):
    """M / (A N^b) vs k, per run, with the joint-law fit and its c."""
    fn = lambda p: active_count(p, "scalefree")
    law = pr_matrix.fit_law(runs, "none", fn)
    ks = sorted({r["k"] for r in runs}); Ns = sorted({r["N"] for r in runs})
    for r in runs:
        v = pr_matrix.measure(r, fn)
        if np.isfinite(v):
            ax.scatter(r["k"] + 0.12 * (Ns.index(r["N"]) - 1.5), v / (law["A"] * r["N"] ** law["b"]), s=16, alpha=.7, color=ps.col_n(r["N"]),
                       label=f"N = {r['N']}" if r is next(x for x in runs if x["N"] == r["N"]) else None)
    kk = np.array([min(ks), max(ks)], float)
    ax.plot(kk, kk ** law["c"], "--", color="0.3", lw=1.2, label=f"fit $k^{{c}}$, c = {law['c']:+.2f} [{law['c_ci'][0]:+.2f}, {law['c_ci'][1]:+.2f}]")
    ax.set(xlabel="task complexity k (bits)", ylabel="active units / fitted size dependence  $M / (A N^b)$", title=title, ylim=(0, 2))
    ax.legend(fontsize=8)
    print(f"{title}: b={law['b']:.3f} [{law['b_ci'][0]:.3f}, {law['b_ci'][1]:.3f}]  c={law['c']:+.3f} [{law['c_ci'][0]:+.3f}, {law['c_ci'][1]:+.3f}]  "
          f"M(k=8)/M(k=1) = {8 ** law['c']:.2f}  n={law['n']}")


def main():
    """Draw the two panels and write fig_P3_complexity.png."""
    ps.setup()
    runs = [r for r in pr_matrix.load() if r["pen"] == "none"]
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    panel(ax[0], flipflop_runs([dict(r) for r in runs], FIXED_ITER), f"read at iteration {FIXED_ITER // 1000}k (matched compute)")
    panel(ax[1], flipflop_runs([dict(r) for r in runs]), "read at 1.10× own loss floor (matched state)")
    fig.suptitle("P(c) — eight-fold more bits change the active count by a few percent once convergence depth is matched (flip-flop, unpenalised)", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "fig_P3_complexity", tight=False)


if __name__ == "__main__":
    main()
