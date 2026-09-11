#!/usr/bin/env python3
"""
Elaboration claim C4(a): early in training the whole population is suppressed, only the units the
solution needs recover, and under the participation penalty units are caught before they stay down.
Read from ParticipationTrace.pkl only; no network is simulated.

  left column   participation quantiles (5, 25, 50, 75, 95%) vs iteration over the first EARLY
                iterations, one network per condition (none vs participation penalty), both tasks:
                the global collapse and the recovery of the survivors
  right column  the number of units per network that were silent for at least ENDURE iterations and
                are active again at the end of the trace ("recovered"), and the number silent at
                the end ("stayed down"), per condition, mean ± sd over seeds

Silent = below the task's criterion (flip-flop p < 4e-2; CDDM scale-free p < 0.05 q95(p) of that
probe). Traces store participation every 100 iterations in these sweeps, so ENDURE is measured at
that resolution and the ~20-iteration global collapse recorded from the every-10 ptrack sweeps
(2026-07-25) is below this figure's resolution.

Sources: CDDM_std_g0_drift (none) and CDDM_std_g0_penalties (frm), N = 1000; NBitFlipFlop_std_ksweep
(none) and NBitFlipFlop_std_penlong (frm), k = 3, N = 1000.

Usage:  python participation_recovery.py
Output: img/internal_figures/fig_C4_recovery.png
"""

import os
import sys
import glob
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, SILENT_REL
import plotstyle as ps

N = 1000
EARLY = 20_000
ENDURE = 500
SRC = {"CDDM": {"none": f"data/trained_RNNs/CDDM_std_g0_drift/EqType=h_N={N}_iters=*",
                "frm": f"data/trained_RNNs/CDDM_std_g0_penalties/EqType=h_N={N}_pen=frm"},
       "flip-flop": {"none": f"data/trained_RNNs/NBitFlipFlop_std_ksweep/EqType=h_k=3_N={N}_iters=*",
                     "frm": f"data/trained_RNNs/NBitFlipFlop_std_penlong/EqType=h_k=3_N={N}_pen=frm*"}}
COL = {"none": "#7f7f7f", "frm": "#d62728"}
LABEL = {"none": "no penalty", "frm": "participation penalty"}
QS = (5, 25, 50, 75, 95)


def traces(pattern):
    """(iters, participation (n_probes, N)) for every run under a cell pattern."""
    out = []
    for f in sorted(glob.glob(os.path.join(pattern, "*", "*ParticipationTrace.pkl"))):
        d = pickle.load(open(f, "rb"))
        out.append((np.asarray(d["participation_iters"]), np.asarray(d["participation"], float)))
    return out


def silent_mask(task, P):
    """(n_probes, N) bool: silent at each probe under the task's criterion."""
    if task == "flip-flop":
        return P < SILENT_FLIPFLOP
    return P < SILENT_REL * np.quantile(P, 0.95, axis=1, keepdims=True)


def recovery(task, it, P):
    """(recovered, stayed_down): units silent for >= ENDURE consecutive iterations that are active at
    the end, and units silent at the end."""
    sil = silent_mask(task, P)
    step = int(np.median(np.diff(it)))
    need = max(1, int(np.ceil(ENDURE / step)))
    endured = np.zeros(P.shape[1], bool)
    run = np.zeros(P.shape[1], int)
    for row in sil:
        run = np.where(row, run + 1, 0)
        endured |= run >= need
    return int((endured & ~sil[-1]).sum()), int(sil[-1].sum())


def main():
    """Draw the figure and print the recovery counts."""
    ps.setup()
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5), gridspec_kw=dict(width_ratios=[1.6, 1]))
    for i, task in enumerate(("CDDM", "flip-flop")):
        stats = {}
        for pen in ("none", "frm"):
            tr = traces(SRC[task][pen])
            it, P = tr[0]
            m = it <= EARLY
            q = np.percentile(P[m], QS, axis=1)
            ax[i, 0].plot(it[m][1:], q[2][1:], color=COL[pen], lw=1.8, label=f"{LABEL[pen]}: median unit")
            ax[i, 0].fill_between(it[m][1:], q[1][1:], q[3][1:], color=COL[pen], alpha=.2, lw=0)
            ax[i, 0].fill_between(it[m][1:], q[0][1:], q[4][1:], color=COL[pen], alpha=.08, lw=0)
            stats[pen] = np.array([recovery(task, it_, P_) for it_, P_ in tr])
            print(f"{task:9s} {pen:4s} n={len(tr)}: recovered after >= {ENDURE} silent iterations {stats[pen][:, 0].mean():.0f} ± {stats[pen][:, 0].std():.0f}; "
                  f"silent at end {stats[pen][:, 1].mean():.0f} ± {stats[pen][:, 1].std():.0f}  (trace to {it[-1]}, step {int(np.median(np.diff(it)))})")
        ax[i, 0].set(xscale="log", yscale="log", xlabel="training iteration", ylabel="participation (5–95% band, quartiles, median)",
                     title=f"{task}, N={N}: the early collapse and who comes back")
        ax[i, 0].legend(loc="lower right", fontsize=8)
        xs = np.arange(2); w = 0.35
        for j, pen in enumerate(("none", "frm")):
            mu = stats[pen].mean(0); sd = stats[pen].std(0)
            ax[i, 1].bar(xs + (j - 0.5) * w, mu, w, yerr=sd, color=COL[pen], capsize=2, label=LABEL[pen])
            for x_, m_, s_ in zip(xs, mu, sd):
                ax[i, 1].text(x_ + (j - 0.5) * w, m_ + s_ + 8, f"{m_:.0f}", ha="center", fontsize=8)
        ax[i, 1].set(xticks=xs, xticklabels=[f"silent ≥ {ENDURE} iterations,\nactive at the end", "silent at the end"],
                     ylabel="units", title=f"{task}: recovery vs staying down (of {N})")
        ax[i, 1].legend(fontsize=8, loc="upper center")
    fig.suptitle("C4 — without a penalty the median unit sinks and the lower band collapses over training, and the units that go down mostly stay down;\n"
                 "with the participation penalty the band holds (flip-flop: units flicker below the threshold and come back)", fontsize=9.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return ps.save(fig, "fig_C4_recovery", tight=False)


if __name__ == "__main__":
    main()
