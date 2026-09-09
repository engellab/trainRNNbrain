#!/usr/bin/env python3
"""Does `rws` close the transient loophole in `frm`? Group-A metrics from the pre-registered plan.

PLAN: docs/experiments/frm_rws_heterogeneity.md, written 2026-08-20 BEFORE any of these numbers
existed. Directions, thresholds and the primary statistic are taken from it unchanged.

HYPOTHESIS (H).  `frm` scores each unit by a soft maximum over time and conditions at tau = 0.1,
which is effectively its PEAK. A unit can therefore satisfy the penalty with one large transient and
do nothing else. H says `rws` suppresses those transients and turns the units into sustained
contributors, so `frm` alone should look transient and `both` should look sustained.

FALSIFIER, stated in the plan before any measurement: if `frm` and `both` are indistinguishable on
group A - median peak-to-mean differing by less than 1.3x with overlapping seed-level means - then
H is WRONG and the paper must not claim this mechanism.

A6 IS THE PRIMARY STATISTIC because it is not a proxy: it evaluates the trained network against the
penalty's OWN objective at a tau that admits transients (0.1, as configured) and at one that does
not (10), and counts units passing only under the former.

⚠️ DEVIATION FROM THE PLAN. The plan names CDDM (12 penalty cells). It was written before the
flip-flop penalty grid existed; that grid is now complete (4 penalties x 3 sizes x 8 k x 3 seeds)
and is the paper's testbed, so it is used here instead. k is fixed to 3. Every threshold, direction
and the falsifier are unchanged. CDDM remains available for the literal pre-registered replication.

Output: img/internal_figures/flipflop_heterogeneity.png

Usage:  python flipflop_heterogeneity.py [k]
"""

import os
import sys
import glob
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, SILENT_FLIPFLOP, participation
from flipflop_bouquet import best_net_pen, _score, PENLONG

PEN = "data/trained_RNNs/NBitFlipFlop_std_pen"      # rws at N<=2000
KSWEEP = "data/trained_RNNs/NBitFlipFlop_std_ksweep"  # none at N<=2000
BIGN = "data/trained_RNNs/NBitFlipFlop_std_bigN"      # none AND rws at N=4000
from flipflop_fixedpoints import load_net
from flipflop_bouquet import run_trials
import plotstyle as ps

PENS = ("none", "rws", "frm", "both")
NS = (500, 1000, 2000, 4000)
KS = tuple(range(1, 9))
TAU_PEN = 0.1        # the tau the penalty actually uses
TAU_MEAN = 10.0      # a tau at which the statistic is effectively the mean
UPV, CAP_FR = 100, 0.3
N_TRIALS = 64


def soft_stat(r, tau):
    """The penalty's own statistic s_tau(r) = tau*(logsumexp(r/tau) - log n), per unit.

    tau -> 0 gives max(r); tau -> infinity gives mean(r). At the configured tau = 0.1 a transient
    scores far higher than a flat unit of the same mean rate, which is the loophole under test.

    Args:
        r: (N, M) firing rates, M = time x trials flattened; tau: softness.
    Returns:
        (N,) array of the statistic.
    """
    a = r / tau
    mx = a.max(axis=1, keepdims=True)
    return tau * (mx[:, 0] + np.log(np.exp(a - mx).sum(axis=1)) - np.log(r.shape[1]))


def cap_for(N):
    """The frm activity cap for a network of size N: cap_fr * log1p(UpV)/log1p(N)."""
    return CAP_FR * np.log1p(UPV) / np.log1p(N)


def metrics(rates, N):
    """Group-A per-unit metrics over ACTIVE units only.

    Args:
        rates: (N, T, B) noise-free firing rates; N: network size (sets the cap).
    Returns:
        dict of per-unit arrays (A1, A2, A3, A4, A5) plus A6 counts and the active mask.
    """
    Nn, T, B = rates.shape
    r = rates.reshape(Nn, T * B)
    act = participation(rates) >= SILENT_FLIPFLOP           # same criterion as the M analysis
    mean = r.mean(axis=1)
    peak = soft_stat(r, TAU_PEN)
    with np.errstate(divide="ignore", invalid="ignore"):
        A1 = peak / mean
        A2 = np.quantile(r, 0.99, axis=1) / mean
        A3 = (r > 0.5 * peak[:, None]).mean(axis=1)
        pk_tc = rates.max(axis=1)                            # (N, B) peak within each trial
        A4 = (pk_tc > 0.5 * peak[:, None]).mean(axis=1)
        A5 = r.sum(axis=1) ** 2 / (r.shape[1] * (r ** 2).sum(axis=1))
    s01, s10 = soft_stat(r, TAU_PEN), soft_stat(r, TAU_MEAN)
    cap = cap_for(N)
    loophole = (s01 >= cap) & (s10 < cap)                    # A6: passes only at the permissive tau
    return dict(A1=A1[act], A2=A2[act], A3=A3[act], A4=A4[act], A5=A5[act],
                A6_frac=float(loophole[act].mean()) if act.any() else np.nan,
                n_active=int(act.sum()), N=Nn)


def folders_for(pen, N, k):
    """Every usable net folder for one grid cell, newest naming conventions included.

    ⚠️ THREE ROOTS AND TWO NAMING CONVENTIONS. `none` lives in ksweep, `rws` in std_pen, frm/both
    in penlong, and BOTH none and rws at N=4000 live in bigN. penlong/ksweep/bigN cells end in
    `_iters=<n>`; std_pen cells do NOT. A pattern assuming one root or one convention silently
    returns zero folders and the condition vanishes from the table with no error - this has already
    happened twice here.

    Args:
        pen: penalty name; N: size; k: bits.
    Returns:
        list of folder paths whose r2 prefix is >= 0.5, best first, at most 3.
    """
    roots = {"none": [KSWEEP, BIGN], "rws": [PEN, BIGN],
             "frm": [PENLONG], "both": [PENLONG]}[pen]
    pats = ([f"EqType=h_k={k}_N={N}_iters=*"] if pen == "none" else []) + \
           [f"EqType=h_k={k}_N={N}_pen={pen}", f"EqType=h_k={k}_N={N}_pen={pen}_iters=*"]
    out = [d for root in roots for pat in pats
           for d in glob.glob(os.path.join(root, pat, "*"))
           if os.path.isdir(d) and _score(d) >= 0.5]
    return sorted(set(out), key=_score, reverse=True)[:3]


def grid_figure(rows, key, label, note):
    """pr_matrix-style figure for one metric: heatmap over (N,k) per penalty, plus curves vs k.

    Args:
        rows: list of per-run dicts carrying pen, N, k and the metric under `key`;
        key: metric field name; label: axis label; note: one-line description for the title.
    Returns:
        path of the written figure.
    """
    cell = {}
    for pen in PENS:
        for N in NS:
            for k in KS:
                v = [r[key] for r in rows
                     if (r["pen"], r["N"], r["k"]) == (pen, N, k) and np.isfinite(r[key])]
                if v:
                    cell[(pen, N, k)] = (float(np.mean(v)), float(np.std(v)), len(v))
    vals = [m for (m, _, _) in cell.values()]
    vmin, vmax = np.percentile(vals, 2), np.percentile(vals, 98)
    fig, ax = plt.subplots(2, len(PENS), figsize=(4.3 * len(PENS), 8.2), squeeze=False)
    im = None
    for c, pen in enumerate(PENS):
        Z = np.full((len(NS), len(KS)), np.nan)
        S = np.full((len(NS), len(KS)), np.nan)
        for i, N in enumerate(NS):
            for j, k in enumerate(KS):
                if (pen, N, k) in cell:
                    Z[i, j], S[i, j], _ = cell[(pen, N, k)]
        a = ax[0][c]
        if np.isfinite(Z).any():
            im = a.imshow(Z, cmap="viridis", vmin=vmin, vmax=vmax, aspect="auto")
            for i in range(len(NS)):
                for j in range(len(KS)):
                    if np.isfinite(Z[i, j]):
                        col = "white" if Z[i, j] < (vmin + vmax) / 2 else "black"
                        a.text(j, i, f"{Z[i, j]:.2f}", ha="center", va="center",
                               fontsize=7, color=col)
                    else:
                        a.text(j, i, "·", ha="center", va="center", color="0.6", fontsize=9)
        a.set(xticks=range(len(KS)), xticklabels=KS, yticks=range(len(NS)),
              yticklabels=[str(n) for n in NS], title=pen)
        if c == 0:
            a.set_ylabel("N (units)")
        b = ax[1][c]
        for i, N in enumerate(NS):
            if np.isfinite(Z[i]).any():
                ps.band(b, list(KS), Z[i], S[i], ps.col_n(N), label=f"N={N}")
        b.set(xlabel="k (bits)", xticks=list(KS), ylim=(vmin, vmax))
        if c == 0:
            b.set_ylabel(label)
        b.legend(fontsize=7)
    if im is not None:
        cb = fig.colorbar(im, ax=ax, fraction=0.018, pad=0.015)
        cb.set_label(label, fontsize=9)
    fig.suptitle(f"{key} — {label}\n{note}   ·   active units only, mean over 3 seeds",
                 fontsize=12)
    return ps.save(fig, f"flipflop_het_{key}", tight=False)


META = {
    "A1": ("peak-to-mean  softmax_0.1(r)/mean(r)", "high = transient, ->1 = sustained"),
    "A2": ("robust peak-to-mean  q99/mean", "A1 without softmax sensitivity to one sample"),
    "A3": ("duty cycle  frac(t,c) with r > 0.5 peak", "low = near its peak almost never"),
    "A4": ("condition breadth  frac(c) with peak_c > 0.5 peak", "low = fires in few conditions"),
    "A5": ("temporal PR  (sum r)^2/(n sum r^2)", "effective fraction of samples active"),
    "A6": ("loophole fraction  s(0.1)>=cap AND s(10)<cap", "PRIMARY: passes only at permissive tau"),
}


def main():
    """Sweep the whole (penalty, N, k) grid and emit one pr_matrix-style figure per metric."""
    rows = []
    for pen in PENS:
        for N in NS:
            for k in KS:
                for f in folders_for(pen, N, k):
                    rnn, _ = load_net(f)
                    m = metrics(run_trials(rnn, f, N_TRIALS), N)
                    rows.append(dict(pen=pen, N=N, k=k,
                                     A1=float(np.median(m["A1"])), A2=float(np.median(m["A2"])),
                                     A3=float(np.median(m["A3"])), A4=float(np.median(m["A4"])),
                                     A5=float(np.median(m["A5"])), A6=m["A6_frac"]))
        print(f"  {pen}: {sum(1 for r in rows if r['pen']==pen)} nets")
    ps.setup()
    for key, (label, note) in META.items():
        print(f"{key}: {grid_figure(rows, key, label, note)}")


if __name__ == "__main__":
    main()
