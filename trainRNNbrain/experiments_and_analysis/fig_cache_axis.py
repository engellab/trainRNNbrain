#!/usr/bin/env python3
"""
One statistic vs network size, four penalty conditions, one panel per task — from the characterize
cache. Each call draws ONE figure for ONE claim of the abstract elaboration:

  python fig_cache_axis.py active     -> fig_S1_active.png       S1/CS2: the participation penalty keeps
                                                                  (nearly) every unit active; rws alone does not
  python fig_cache_axis.py sel        -> fig_S4_selectivity.png  S4/CS3: frm units lose selectivity as N grows;
                                                                  adding rws restores it and it no longer degrades
  python fig_cache_axis.py temp       -> fig_S5_temporal.png     S5: frm units are transient on the flip-flop
                                                                  (and NOT on CDDM — the caveat is in the figure)
  python fig_cache_axis.py d_pr       -> fig_CS5_dimensionality.png  CS5: identical performance, several-fold
                                                                  different dimensionality

Source: data/characterize_cache.pkl written by characterize.py (end-of-training networks, N = 500..5000
CDDM and 500..4000 flip-flop, k = 3; three seeds per cell). Definitions as in characterize.py:
active = M, the number of units above the task criterion (log-log with the M = N diagonal); sel = median Hoyer sparsity of a unit's rectified-regression
coefficients (1 = one task variable, 0 = evenly mixed); temp = median Hoyer sparsity of a unit's
trace (1 = fires at one instant, 0 = constant); d_pr = participation-ratio dimensionality of the
rate covariance. No network is simulated here.

Usage:  python fig_cache_axis.py {active|sel|temp|d_pr}
"""

import os
import re
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps

CACHE = "data/characterize_cache.pkl"
MIN_N = 500          # N = 100 excluded everywhere (Pavel, 2026-09-11)
PENS = ["none", "rws", "frm", "both"]
COL = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
LABEL = {"none": "no penalty", "rws": "sparsity penalty only", "frm": "participation penalty only",
         "both": "participation + sparsity"}
AXES = {
    "active": dict(key="active_units", ylabel="active units M", ylim=None, log=True, stem="fig_S1_active",
                   title="S1 — the participation penalty keeps nearly every unit active; the sparsity penalty alone does not"),
    "sel": dict(key="sel", ylabel="selectivity (Hoyer of tuning coefficients; 1 = one variable)", ylim=(0, 1), stem="fig_S4_selectivity",
                title="S4 / CS3 — participation-penalty units lose selectivity with N; adding the sparsity penalty restores it"),
    "temp": dict(key="temp", ylabel="temporal sparsity (Hoyer of a unit's trace; 1 = one instant)", ylim=(0, 1), stem="fig_S5_temporal",
                 title="S5 — participation-penalty units are transient on the flip-flop, not on CDDM"),
    "d_pr": dict(key="d_pr", ylabel="dimensionality $D_{PR}$", ylim=None, stem="fig_CS5_dimensionality",
                 title="CS5 — same task, same performance, several-fold different dimensionality"),
}


def load(path):
    """Group the cache by (task, penalty, N).

    Args:
        path: characterize_cache.pkl.
    Returns:
        dict task -> dict pen -> dict N -> list of measure dicts (one per seed).
    """
    out = {"CDDM": {p: {} for p in PENS}, "flip-flop": {p: {} for p in PENS}}
    for run, m in pickle.load(open(path, "rb")).items():
        if "FlipFlop" in run:
            task = "flip-flop"
            frm = float(re.search(r"Lfrm=([^;]+)", run).group(1)) > 0
            rws = float(re.search(r"Lrws=([^;]+)", run).group(1)) > 0
            pen = {(False, False): "none", (False, True): "rws", (True, False): "frm", (True, True): "both"}[(frm, rws)]
        else:
            task = "CDDM"
            mm = re.search(r"_pen=([a-z]+)", run)
            pen = mm.group(1) if mm else "none"
        if int(m["N"]) < MIN_N:
            continue
        m = dict(m, active_units=float(m["active_frac"]) * int(m["N"]))
        out[task][pen].setdefault(int(m["N"]), []).append(m)
    return out


def main():
    """Draw the requested axis vs N for both tasks and write its figure."""
    spec = AXES[sys.argv[1]]
    ps.setup()
    data = load(CACHE)
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2), sharey=True)
    for a, task in zip(ax, ("CDDM", "flip-flop")):
        for pen in PENS:
            byN = data[task][pen]
            Ns = sorted(byN)
            if not Ns:
                continue
            mu = [np.mean([m[spec["key"]] for m in byN[N]]) for N in Ns]
            sd = [np.std([m[spec["key"]] for m in byN[N]]) for N in Ns]
            a.errorbar(Ns, mu, yerr=sd, fmt="o-", color=COL[pen], label=LABEL[pen], capsize=2)
            print(f"{task:9s} {pen:5s} " + "  ".join(f"N={N}: {m:.2f}±{s:.2f}" for N, m, s in zip(Ns, mu, sd)))
        a.set(xscale="log", xlabel="network size N", title=task if task == "CDDM" else "flip-flop, k = 3")
        if spec.get("log"):
            Ns = sorted({N for p in PENS for N in data[task][p]})
            a.plot(Ns, Ns, ":", color="0.4", lw=1, label="M = N")
            a.set_yscale("log")
        if spec["ylim"]:
            a.set_ylim(*spec["ylim"])
    ax[0].set_ylabel(spec["ylabel"])
    ax[0].legend(loc="best")
    fig.suptitle(spec["title"], fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, spec["stem"], tight=False)


if __name__ == "__main__":
    main()
