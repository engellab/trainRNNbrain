#!/usr/bin/env python3
"""
What rws does to individual units: temporal-PR trajectories through the penalty switch.

A1 = an frm network continued under frm + rws (treatment).
A2 = the SAME frm parent continued under frm alone (control). The arms are paired: A1 rep i and
A2 rep i branch from the identical network, so any difference is rws and not the seed.

⚠️ THE CONTROL IS NOT OPTIONAL AND IT IS NOT INERT. Warm-starting resets Adam and adds 50k
iterations; A2 measures exactly that. Reading A1 against zero rather than against A2 would attribute
the whole change to rws.

THREE VIEWS OF THE SAME DATA, because summary statistics were misleading here:
  row 1  the full distribution over training, as a density. Shows whether the population moves as a
         block or converges.
  row 2  individual units, sampled STRATIFIED BY STARTING VALUE (equal numbers from each quintile of
         the initial temporal PR) and coloured by it. A random sample would be dominated by the bulk
         and would hide what happens to the low-occupancy units, which are the ones at issue.
  row 3  quantiles plus the dead fraction, so compression and revival are separated.

⚠️ WHY BOTH THE SPREAD AND THE RANK CORRELATION ARE NEEDED. rws compresses the distribution ~2x
(final IQR 0.14-0.21 against the control's 0.32-0.34). That compression MECHANICALLY depresses any
rank correlation, so a low Spearman between start and end does NOT by itself demonstrate that units
reshuffled - it is also what convergence to a common value looks like. The distinguishing statistic
is corr(change, starting value): -0.67 under A1 against -0.40 under A2, i.e. the units that started
lowest gain the most. Levelling, not translation and not reshuffling.

⚠️ CHOOSE THE SEED DELIBERATELY AND SAY WHICH. rep 0 is the ONLY one of the six runs that hit the
frm gradient-spike problem: 498 skipped updates and 1.12% of probes above 10x the median loss,
against ZERO skipped updates in every other run and 0.02% in the controls. Those spikes appear in
the figure as abrupt collapses of the whole population, which are transient excursions of the
network, not anything rws does. They are NOT the spike guard (0 rollbacks in every run) and NOT
batch noise (on a fixed net across 12 independent batches the population median moves by sd 0.0011
and no unit moves by more than 0.1, against swings of 0.3-0.5 here). rep 1 is representative.

Reads the reduced trace `switch_traj.npz` (temporal PR every 250 iterations, all units, 6 runs)
extracted from the full traces on the cluster.

Usage:  python flipflop_switch_trajectories.py [npz] [rep]

Output: img/internal_figures/switch_trajectories.png

"""

import os
import sys
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps

ARMS = [("A1", "frm → frm + rws   (treatment)"), ("A2", "frm → frm   (control)")]
N_TRAJ_PER_BIN = 6
N_BINS = 5
YMAX = 0.75


def main():
    """Plot distribution, sampled trajectories and quantiles for the treatment and control arms."""
    path = sys.argv[1] if len(sys.argv) > 1 else "switch_traj.npz"
    rep = sys.argv[2] if len(sys.argv) > 2 else "1"
    z = np.load(path)
    ps.setup()
    fig, ax = plt.subplots(3, 2, figsize=(14, 12), sharex=True)

    for c, (arm, title) in enumerate(ARMS):
        T = z[f"{arm}_{rep}_tpr"].astype(float)          # (n_time, n_units)
        it = z[f"{arm}_{rep}_it"].astype(float)
        t0 = T[0]

        # ---- row 0: the whole distribution over training -------------------------------------
        edges = np.linspace(0, YMAX, 61)
        H = np.stack([np.histogram(T[i], bins=edges)[0] for i in range(T.shape[0])], axis=1)
        a = ax[0][c]
        a.imshow(H, origin="lower", aspect="auto", cmap="magma",
                 extent=[it[0], it[-1], edges[0], edges[-1]],
                 norm=plt.matplotlib.colors.PowerNorm(0.4))
        a.set(ylabel="temporal PR / n" if c == 0 else "", ylim=(0, YMAX),
              title=f"{arm} — {title}\ndistribution of all {T.shape[1]} units over training")

        # ---- row 1: stratified individual trajectories ----------------------------------------
        b = ax[1][c]
        qs = np.quantile(t0, np.linspace(0, 1, N_BINS + 1))
        rng = np.random.default_rng(0)
        cmap = plt.cm.viridis
        for q in range(N_BINS):
            pool = np.flatnonzero((t0 >= qs[q]) & (t0 <= qs[q + 1]))
            if pool.size == 0:
                continue
            for u in rng.choice(pool, min(N_TRAJ_PER_BIN, pool.size), replace=False):
                b.plot(it, T[:, u], lw=0.9, alpha=.8,
                       color=cmap(q / max(N_BINS - 1, 1)))
        b.set(ylabel="temporal PR / n" if c == 0 else "", ylim=(0, YMAX),
              title="individual units, sampled evenly across starting value\n"
                    "colour = quintile of temporal PR at the switch (dark = lowest)")
        b.grid(alpha=.25)

        # ---- row 2: quantiles and the dead fraction -------------------------------------------
        d = ax[2][c]
        for lo, hi, al in ((10, 90, .18), (25, 75, .3)):
            d.fill_between(it, np.percentile(T, lo, axis=1), np.percentile(T, hi, axis=1),
                           color="#1f77b4", alpha=al, lw=0)
        d.plot(it, np.median(T, axis=1), color="#1f77b4", lw=2, label="median")
        d.set(xlabel="iterations since the switch", ylim=(0, YMAX),
              ylabel="temporal PR / n" if c == 0 else "",
              title="quantiles (median, 25-75, 10-90) and dead fraction")
        d.grid(alpha=.25)
        e = d.twinx()
        e.plot(it, (T == 0).mean(1), color="#d62728", lw=1.6, ls="--", label="dead fraction")
        e.set_ylabel("dead fraction", color="#d62728")
        e.set_ylim(0, 0.25); e.tick_params(axis="y", colors="#d62728")
        d.legend(fontsize=8, loc="lower right")

        rho = spearmanr(t0, T[-1]).statistic
        liv = t0 > 0
        cc = np.corrcoef(t0[liv], T[-1][liv] - t0[liv])[0, 1]
        d.text(.03, .97, f"median {np.median(t0):.3f} → {np.median(T[-1]):.3f}\n"
                         f"IQR {np.subtract(*np.percentile(T[-1],[75,25])):.3f} at end\n"
                         f"dead {(t0==0).mean():.3f} → {(T[-1]==0).mean():.3f}\n"
                         f"ρ(start,end) = {rho:.2f}\n"
                         f"corr(Δ, start) = {cc:+.2f}",
               transform=d.transAxes, fontsize=8, va="top",
               bbox=dict(fc="white", ec="0.7", alpha=.9, boxstyle="round,pad=0.35"))

    fig.suptitle("What rws does to individual units — paired penalty switch, N=2000, k=3, seed rep "
                 f"{rep}\nboth columns start from the IDENTICAL frm network; only the penalty after "
                 "the switch differs", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, f"switch_trajectories_rep{rep}", tight=False)


if __name__ == "__main__":
    main()
