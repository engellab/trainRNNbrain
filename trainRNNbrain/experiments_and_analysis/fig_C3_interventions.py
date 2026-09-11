#!/usr/bin/env python3
"""
Figure for elaboration claim C3: many interventions failed to keep more units active.

Two panels of active-unit counts at N = 1000 on CDDM, h equation, no penalty, one bar per
intervention, mean ± sd over networks, N = 1000 marked; every label carries the read-out iteration. Every bar reads an existing per-network
table; no network is simulated. THE TWO PANELS USE DIFFERENT SILENCE CRITERIA AND ARE NOT
COMPARABLE TO EACH OTHER - each has its own reference bar:

  left   participation criterion p >= 0.05 q95(p), the project standard. Standard-network sweeps,
         30k iterations unless stated:
           baseline              silent_stats_v2.csv, sweep std (trainable bias, self-connections on)
           5k / 200k iterations  silent_stats_v2.csv sweep largeN; CDDM_std_g0_drift N=1000 last probe
           self-connections off  silent_stats_all.csv, sweep nodale_bias (trainable bias, self-conn off)
           bias fixed at 0       silent_stats_all.csv, sweep nodale (self-conn off, bias fixed)
           metabolic cost        silent_stats_v2.csv, sweep metabolic, lambda in {0.01, 0.1, 1, 10}
  right  peak-rate criterion (peak < 5% of the 95th-percentile peak; count_silent_units.py), because
         these 2026-07-01 sweeps have no participation traces. Earlier architecture; the reference
         is the family's own default (sigma_rec = 0.05; ReLU). E1 reruns the activations on the
         standard network.
           recurrent noise       CDDM_fb2792_g0_noise, sigma_rec in {0, .01, .05, .1}
           activation            CDDM_2bc3c1_g0_reflective (ReLU), CDDM_fb2792_g0_softplus25,
                                 CDDM_fb2792_g0_leakyrelu

Not on the chart: input weight scale (T5, running), weight decay 0 vs 1e-6 (run 2026-07-28, traces
not synced), larger N (fig_P_active_units).

Usage:  python fig_C3_interventions.py
Output: img/internal_figures/fig_C3_interventions.png
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

N = 1000
D = "data/trained_RNNs"


def v2_rows(sweep, **match):
    """Active counts from silent_stats_v2.csv rows matching sweep, eq=h, N, and the given columns.

    Args:
        sweep: 'std', 'metabolic' or 'largeN'; match: column -> value (strings compared as floats).
    Returns:
        list of active counts, one per network.
    """
    out = []
    for r in csv.DictReader(open(f"{D}/silent_stats_v2.csv")):
        if r["sweep"] == sweep and r["eq"] == "h" and int(r["N"]) == N and \
           all(float(r[k]) == v for k, v in match.items()):
            out.append(N * (1 - float(r["rel_5p95"])))
    return out


def all_rows(sweep):
    """Active counts from silent_stats_all.csv rows of one sweep at eq=h, N, no penalty."""
    return [N * (1 - float(r["rel_5p95"])) for r in csv.DictReader(open(f"{D}/silent_stats_all.csv"))
            if r["sweep"] == sweep and r["eq"] == "h" and int(r["N"]) == N and r["penalty"] == "none"]


def cond_row(sweep, condition):
    """(mean, sd) active counts from a sweep's silent_units_per_condition.csv row."""
    r = next(r for r in csv.DictReader(open(f"{D}/{sweep}/silent_units_per_condition.csv")) if r["condition"] == condition)
    return N - float(r["silent_rel_mean"]), float(r["silent_rel_std"])


def drift_last():
    """Active count at the last probe of every CDDM_std_g0_drift N=1000 trace (200k iterations)."""
    out = []
    for f in glob.glob(f"{D}/CDDM_std_g0_drift/EqType=h_N={N}_iters=*/*/*ParticipationTrace.pkl"):
        p = np.asarray(pickle.load(open(f, "rb"))["participation"], float)[-1]
        out.append(int((p >= SILENT_REL * np.quantile(p, 0.95)).sum()))
    return out


def draw(ax, bars, ref_index, title, groups):
    """One bar panel with a reference line at bars[ref_index] and bracket labels over groups."""
    for i, (lab, (m, sd, n), c) in enumerate(bars):
        ax.bar(i, m, 0.7, yerr=sd, color=c, capsize=2)
        ax.text(i, m + sd + 15, f"{m:.0f}", ha="center", fontsize=7.5)
        print(f"{lab.replace(chr(10), ' '):18s} active {m:.0f} ± {sd:.0f}  (n={n})")
    ref = bars[ref_index][1][0]
    ax.axhline(ref, color="0.3", lw=0.8, ls="--")
    ax.axhline(N, color="0.5", lw=0.8, ls=":")
    ax.text(len(bars) - 0.5, N + 12, "N = 1000", ha="right", fontsize=8, color="0.4")
    ax.text(len(bars) - 0.5, ref + 12, "reference", ha="right", fontsize=8, color="0.3")
    ax.set(xticks=np.arange(len(bars)), xticklabels=[b[0] for b in bars], ylabel="active units", ylim=(0, 1120), title=title)
    ax.tick_params(axis="x", labelsize=7.5)
    for lo, hi, t in groups:
        ax.annotate("", xy=(lo - 0.4, 1045), xytext=(hi + 0.4, 1045), arrowprops=dict(arrowstyle="-", color="0.5"))
        ax.text((lo + hi) / 2, 1058, t, ha="center", fontsize=7.5, color="0.4")


def main():
    """Draw the two panels and write fig_C3_interventions.png."""
    ps.setup()
    ms = lambda v: (float(np.mean(v)), float(np.std(v)), len(v))
    left = [("baseline\n(self-conn. on,\nbias trainable)\n30k", ms(v2_rows("std", rws=0, frm=0, met=0)), "0.3"),
            ("baseline\n5k", ms(v2_rows("largeN", rws=0, frm=0, met=0)), "#8c564b"),
            ("baseline\n200k", ms(drift_last()), "#8c564b"),
            ("self-conn. off,\nbias trainable\n30k", ms(all_rows("nodale_bias")), "#17becf"),
            ("self-conn. off,\nbias fixed at 0\n30k", ms(all_rows("nodale")), "#17becf")]
    left += [(f"metabolic\nλ={lam:g}\n30k", ms(v2_rows("metabolic", rws=0, frm=0, met=lam)), "#e377c2") for lam in (0.01, 0.1, 1.0, 10.0)]
    right = []
    for sig in ("0", "0.01", "0.05", "0.1"):
        r = next(r for r in csv.DictReader(open(f"{D}/CDDM_fb2792_g0_noise/silent_units_per_condition.csv")) if r["condition"] == f"EqType=h_N={N}_sigrec={sig}")
        right.append((f"noise\nσ={sig}\n30k", (N - float(r["silent_rel_mean"]), float(r["silent_rel_std"]), int(r["n_nets"])), "#ff7f0e"))
    for lab, sweep in (("ReLU\n30k", "CDDM_2bc3c1_g0_reflective"), ("softplus\n30k", "CDDM_fb2792_g0_softplus25"), ("leaky ReLU\n30k", "CDDM_fb2792_g0_leakyrelu")):
        right.append((lab, cond_row(sweep, f"EqType=h_N={N}_LmbdRWS=0_LmbdFR=0") + (5,), "#9467bd"))

    fig, ax = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw=dict(width_ratios=[9, 7]))
    draw(ax[0], left, 0, "standard network, participation criterion", [(1, 2, "training length"), (3, 3, "self-connections"), (4, 4, "bias"), (5, 8, "metabolic cost")])
    draw(ax[1], right, 2, "2026-07-01 sweeps (earlier architecture), peak-rate criterion", [(0, 3, "recurrent noise (ref: σ=0.05)"), (4, 6, "activation (ref: ReLU)")])
    fig.suptitle("C3 — CDDM, N=1000, no penalty: no intervention keeps more than ~600 of the 1000 units active (criteria differ between panels; compare within a panel only)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, "fig_C3_interventions", tight=False)


if __name__ == "__main__":
    main()
