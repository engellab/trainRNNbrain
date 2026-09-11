#!/usr/bin/env python3
"""
Elaboration claim C5: the sparsity penalty does not rescue units - it parks them just above the hard
threshold. The proof is that the two silence criteria DISAGREE for the sparsity penalty and agree for
every other condition. Active-unit counts, no penalty vs sparsity penalty, vs N, both tasks, from the
participation traces only.

  CDDM       every network at the iteration where the matching penalty run stopped (penalty_matched):
             hard criterion p >= 1e-6 (solid) and scale-free p >= 0.05 q95(p) (dashed)
  flip-flop  k = 3, every run at 1.10x its own loss floor (pr_matrix): task-calibrated absolute
             criterion p >= 4e-2 (solid) and scale-free (dashed); the CDDM hard threshold 1e-6 is
             also drawn (dotted) to show that it reports nearly every flip-flop unit as active.

Usage:  python fig_C5_criteria.py
Output: img/internal_figures/fig_C5_criteria.png
"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, SILENT_HARD, active_count
import plotstyle as ps
import pr_matrix
from penalty_matched import collect

COL = {"none": "#7f7f7f", "rws": "#2ca02c"}
LABEL = {"none": "no penalty", "rws": "sparsity penalty"}


def cddm_counts():
    """dict (pen, crit) -> dict N -> list of active counts, none read at the rws run's budget."""
    runs = collect()
    out = {}
    for (N, pen), rr in runs.items():
        if pen not in COL or N < 500:
            continue
        budget = min(r["budget"] for r in runs.get((N, "rws"), rr))
        for r in rr:
            P, I = np.array(r["trace"]["participation"]), np.array(r["trace"]["participation_iters"])
            p = P[np.argmin(np.abs(I - budget))]
            for crit in ("hard", "scalefree"):
                out.setdefault((pen, crit), {}).setdefault(N, []).append(active_count(p, crit))
    return out


def flipflop_counts():
    """dict (pen, crit) -> dict N -> list of active counts at the excess read-out, k = 3."""
    runs = [r for r in pr_matrix.load() if r["pen"] in COL and r["k"] == 3]
    for r in runs:
        r["floor"] = pr_matrix.fit_floor(r["loss"], r["budget"])
        r["T"] = pr_matrix.excess_time(r["loss"], r["floor"], pr_matrix.EXCESS_DELTA)
    out = {}
    for r in runs:
        if not np.isfinite(r["T"]):
            continue
        for crit, key in ((SILENT_FLIPFLOP, "abs"), ("scalefree", "scalefree"), (SILENT_HARD, "hard1e-6")):
            v = pr_matrix.measure(r, lambda p, c=crit: active_count(p, c))
            if np.isfinite(v):
                out.setdefault((r["pen"], key), {}).setdefault(r["N"], []).append(v)
    return out


def draw(ax, data, styles, title):
    """Lines per (pen, crit) with the style map crit -> (linestyle, label)."""
    for (pen, crit), byN in sorted(data.items()):
        ls, lab = styles[crit]
        Ns = sorted(byN)
        mu = [np.mean(byN[N]) for N in Ns]; sd = [np.std(byN[N]) for N in Ns]
        ax.errorbar(Ns, mu, yerr=sd, fmt="o" + ls, color=COL[pen], capsize=2, label=f"{LABEL[pen]}, {lab}")
        print(f"{title[:9]} {pen:4s} {crit:9s}: " + "  ".join(f"N={N}: {m:.0f}±{s:.0f}" for N, m, s in zip(Ns, mu, sd)))
    Ns = sorted({N for byN in data.values() for N in byN})
    ax.plot(Ns, Ns, ":", color="0.4", lw=1, label="M = N")
    ax.set(xscale="log", yscale="log", xlabel="network size N", ylabel="active units", title=title)
    ax.legend(fontsize=7.5)


def main():
    """Draw the two panels and write fig_C5_criteria.png."""
    ps.setup()
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.4))
    draw(ax[0], cddm_counts(), {"hard": ("-", "hard p ≥ 10⁻⁶"), "scalefree": ("--", "scale-free")}, "CDDM, read at the sparsity run's budget")
    draw(ax[1], flipflop_counts(), {"abs": ("-", "absolute p ≥ 4·10⁻²"), "scalefree": ("--", "scale-free"), "hard1e-6": (":", "p ≥ 10⁻⁶ (CDDM threshold)")},
         "flip-flop k=3, read at 1.10× own loss floor")
    fig.suptitle("C5 — the sparsity penalty raises the count under a hard threshold and lowers it under the scale-free one: its 'rescued' units sit just above the floor",
                 fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "fig_C5_criteria", tight=False)


if __name__ == "__main__":
    main()
