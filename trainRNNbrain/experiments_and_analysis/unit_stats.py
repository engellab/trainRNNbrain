#!/usr/bin/env python3
"""
Per-unit statistics behind elaboration claims S4 (mixed selectivity) and S5 (transients), as COUNTS
of units, on both tasks, with one simulation pass per network cached in data/unit_stats_cache.pkl.

Per network (flip-flop k = 3, N = 500..4000; CDDM N = 500..5000; four conditions; every seed):
  live      units above the task's silence criterion (flip-flop p >= 4e-2; CDDM p >= 0.05 q95(p))
  burst     live units active less than BURST of the time: temporal PR / n_samples < BURST, where
            tPR_i = (sum_s r_is)^2 / sum_s r_is^2 over pooled (time, trial) samples (lifetime sparseness)
  tuned     live units whose rectified regression on the task variables reaches R^2 >= R2_GATE
  mixed     tuned units whose joint regression beats the best SINGLE-variable regression by more than
            MIX_GAIN in R^2 - the unit needs a second task variable to be explained
  loadings  signed regression loadings on three task variables for the selectivity cloud
            (flip-flop: the three bits; CDDM: motion, colour, choice coherence, with context as a
            fourth regressor that is fitted but not drawn)
  examples  traces of the EX lowest- and EX highest-tPR live units on each unit's EX_TRIALS most active
            trials, for the example figure; stored for N = 2000 only

Regression design follows characterize.py: flip-flop time-resolved on relu(+-b_j) per bit; CDDM on
the decision-epoch mean per condition with [ctx, relu(+-motion), relu(+-colour), relu(+-choice)].
Single-variable models: one bit (flip-flop) or one of {context, motion, colour, choice} (CDDM).

Figures (each a separate call, all from the cache):
  python unit_stats.py mixed     -> fig_S4_mixed.png        mixed-selective units vs N, both tasks
  python unit_stats.py bursts    -> fig_S5_bursts.png       burst units vs N, both tasks
  python unit_stats.py cloud     -> fig_S4_cloud_cddm.png   CDDM units in (motion, colour, choice) loading space, 2 x 2
  python unit_stats.py examples  -> fig_S5_examples.png     example unit traces, low vs high tPR, both tasks
  python unit_stats.py tuning    -> fig_S4_tuning.png       per-unit R^2 of the rectified-factor fit and Hoyer of its coefficients, N = 2000
  python unit_stats.py compute   -> fill the cache only
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, SILENT_REL, participation
import plotstyle as ps
from characterize import cddm_folders, cddm_rates, hoyer, FF_K, FF_TRIALS
from flipflop_dimensionality import run_folders
from flipflop_diversity import rates_and_targets

CACHE = "data/unit_stats_cache.pkl"
MIN_N = 500
BURST = 0.05
R2_GATE = 0.15
MIX_GAIN = 0.10
EX, EX_TRIALS, EX_N = 4, 6, 2000
PENS = ["none", "rws", "frm", "both"]
COL = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
LABEL = {"none": "no penalty", "rws": "sparsity only", "frm": "participation only", "both": "participation + sparsity"}


def r2_of(G, Y):
    """R^2 of a least-squares fit of each row of Y (units x samples) on the design G (samples x p)."""
    beta, *_ = np.linalg.lstsq(G, Y.T, rcond=None)
    resid = Y.T - G @ beta
    ss = ((Y - Y.mean(1, keepdims=True)) ** 2).sum(1)
    return 1.0 - (resid ** 2).sum(0) / np.maximum(ss, 1e-300), beta


def unit_stats(rates, live, Y, groups, signed):
    """Burst, tuned and mixed counts plus loadings for one network.

    Args:
        rates: (N, T, B) rates; live: (N,) bool; Y: (N, S) regression targets (rows = units);
        groups: list of (name, columns) of the rectified design, each a single task variable, with
            columns indexing the full design G that is their concatenation (intercept first);
        signed: (S, 3) signed values of the three drawn task variables per sample.
    Returns:
        dict with counts and per-unit arrays.
    """
    x = rates.reshape(rates.shape[0], -1).astype(np.float64)
    denom = (x ** 2).sum(1)
    tpr = np.where(denom > 0, x.sum(1) ** 2 / np.maximum(denom, 1e-300), np.nan) / x.shape[1]
    G = np.column_stack([np.ones(Y.shape[1])] + [c for _, c in groups])
    Yc = Y - Y.mean(1, keepdims=True)
    r2_joint, beta_rect = r2_of(G, Yc)
    hoyer_rect = hoyer(np.abs(beta_rect[1:].T))                       # over the rectified factors, intercept excluded
    r2_single = np.full(Y.shape[0], -np.inf)
    for _, c in groups:
        r2_single = np.maximum(r2_single, r2_of(np.column_stack([np.ones(Y.shape[1]), c]), Yc)[0])
    tuned = live & (r2_joint >= R2_GATE)
    mixed = tuned & (r2_joint - r2_single > MIX_GAIN)
    _, beta = r2_of(np.column_stack([np.ones(signed.shape[0]), signed]), Yc)
    return dict(live=int(live.sum()), burst=int((live & (tpr < BURST)).sum()), tuned=int(tuned.sum()),
                mixed=int(mixed.sum()), tpr=tpr, live_mask=live, tuned_mask=tuned, loadings=beta[1:4].T,
                r2=r2_joint, hoyer=hoyer_rect, n_factors=G.shape[1] - 1)


def flipflop_one(folder):
    """Statistics for one flip-flop network."""
    rates, targets = rates_and_targets(folder, FF_TRIALS)
    live = participation(rates) >= SILENT_FLIPFLOP
    B = targets.reshape(targets.shape[0], -1).T                      # (S, k) signed bits
    groups = [(f"bit{j}", np.column_stack([np.maximum(B[:, j], 0), np.maximum(-B[:, j], 0)])) for j in range(B.shape[1])]
    Y = rates.reshape(rates.shape[0], -1)
    return unit_stats(rates, live, Y, groups, B[:, :3]), rates


def cddm_one(folder):
    """Statistics for one CDDM network (decision-epoch mean per condition, as characterize.py)."""
    rates, G, dec_on = cddm_rates(folder)
    p = participation(rates)
    live = p >= SILENT_REL * np.quantile(p, 0.95)
    Y = rates[:, dec_on:, :].mean(1)
    groups = [("ctx", G[:, 1:2]), ("motion", G[:, 2:4]), ("colour", G[:, 4:6]), ("choice", G[:, 6:8])]
    signed = np.column_stack([G[:, 2] - G[:, 3], G[:, 4] - G[:, 5], G[:, 6] - G[:, 7]])
    return unit_stats(rates, live, Y, groups, signed), rates


def compute():
    """Simulate every network once; return the cache dict folder -> row."""
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    jobs = [("flip-flop", f, pen, N) for f, pen, k, N in run_folders() if k == FF_K and N >= MIN_N]
    jobs += [("CDDM", f, pen, N) for f, pen, N in cddm_folders() if N >= MIN_N]
    for i, (task, folder, pen, N) in enumerate(jobs, 1):
        if folder in cache:
            continue
        st, rates = (flipflop_one if task == "flip-flop" else cddm_one)(folder)
        row = dict(task=task, pen=pen, N=N, live=st["live"], burst=st["burst"], tuned=st["tuned"], mixed=st["mixed"])
        if N == EX_N:
            order = np.argsort(np.where(st["live_mask"], st["tpr"], np.nan))
            lo = [u for u in order if st["live_mask"][u]][:EX]
            hi = [u for u in order[::-1] if st["live_mask"][u] and np.isfinite(st["tpr"][u])][:EX]
            # each unit's EX_TRIALS most active trials (by peak rate), so a unit that fires in a few
            # conditions is shown where it fires; a burst unit is still a burst within those trials
            top = lambda u: np.argsort(rates[u].max(0))[::-1][:EX_TRIALS]
            row["examples"] = dict(lo=np.stack([rates[u][:, np.sort(top(u))] for u in lo]),
                                   hi=np.stack([rates[u][:, np.sort(top(u))] for u in hi]),
                                   tpr_lo=st["tpr"][lo], tpr_hi=st["tpr"][hi])
            row["loadings"] = st["loadings"][st["live_mask"]]
            row["part"] = participation(rates)[st["live_mask"]]
            row["r2_units"] = st["r2"][st["live_mask"]]                 # joint rectified fit, live units
            row["hoyer_units"] = st["hoyer"][st["tuned_mask"]]          # coefficient sparsity, tuned units
            row["n_factors"] = st["n_factors"]
        cache[folder] = row
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"  {i}/{len(jobs)} {task} {pen} N={N}: live {row['live']} burst {row['burst']} tuned {row['tuned']} mixed {row['mixed']}", flush=True)
    return cache


def counts_vs_N(rows, key, stem, ylabel, title):
    """Two-panel count-vs-N figure, four conditions, both tasks."""
    fig, ax = plt.subplots(1, 2, figsize=(10.5, 4.2))
    for a, task in zip(ax, ("CDDM", "flip-flop")):
        Ns_all = sorted({r["N"] for r in rows if r["task"] == task})
        for pen in PENS:
            byN = {}
            for r in rows:
                if r["task"] == task and r["pen"] == pen:
                    byN.setdefault(r["N"], []).append(r[key])
            Ns = sorted(byN)
            if not Ns:
                continue
            mu = [np.mean(byN[N]) for N in Ns]; sd = [np.std(byN[N]) for N in Ns]
            a.errorbar(Ns, mu, yerr=sd, fmt="o-", color=COL[pen], capsize=2, label=LABEL[pen])
            print(f"{task:9s} {pen:5s} {key}: " + "  ".join(f"N={N}: {m:.0f}±{s:.0f}" for N, m, s in zip(Ns, mu, sd)))
        a.plot(Ns_all, Ns_all, ":", color="0.4", lw=1, label="M = N")
        a.set(xscale="log", yscale="symlog", xlabel="network size N", title=task if task == "CDDM" else "flip-flop, k = 3")
        a.set_ylim(0, None)
    ax[0].set_ylabel(ylabel)
    ax[0].legend(loc="upper left")
    fig.suptitle(title, fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.90])
    return ps.save(fig, stem, tight=False)


def cloud_cddm(rows):
    """CDDM units at their (motion, colour, choice) loadings, 2 x 2 over conditions, N = EX_N, first seed."""
    fig = plt.figure(figsize=(9, 8.5))
    pick = {}
    for r in rows:
        if r["task"] == "CDDM" and r["N"] == EX_N and "loadings" in r and r["pen"] not in pick:
            pick[r["pen"]] = r
    lo = min(np.log10(r["part"]).min() for r in pick.values()); hi = max(np.log10(r["part"]).max() for r in pick.values())
    for i, pen in enumerate(PENS):
        r = pick[pen]; P = r["loadings"]
        ax = fig.add_subplot(2, 2, i + 1, projection="3d")
        sc = ax.scatter(P[:, 0], P[:, 1], P[:, 2], c=np.log10(r["part"]), cmap="viridis", vmin=lo, vmax=hi, s=9, alpha=.75, lw=0)
        rr = np.abs(P).max()
        for a in range(3):
            v = np.zeros((2, 3)); v[0, a], v[1, a] = -rr, rr
            ax.plot(v[:, 0], v[:, 1], v[:, 2], color="0.75", lw=0.8)
        ax.set(xlim=(-rr, rr), ylim=(-rr, rr), zlim=(-rr, rr), xlabel="motion", ylabel="colour", zlabel="choice",
               title=f"{LABEL[pen]}\n{len(P)} live units")
        ax.view_init(elev=22, azim=35)
        ax.set_xticklabels([]); ax.set_yticklabels([]); ax.set_zticklabels([])
    fig.colorbar(sc, ax=fig.axes, shrink=0.45, pad=0.03, label="log10 participation")
    fig.suptitle(f"S4 — CDDM units in (motion, colour, choice) loading space, N={EX_N}: a few high-participation units on the axes, a core of mixed units in every condition",
                 fontsize=10.5)
    return ps.save(fig, "fig_S4_cloud_cddm", tight=False)


def examples(rows):
    """Example traces: lowest- and highest-tPR live units of the participation-penalty network, both tasks."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 6))
    for i, task in enumerate(("flip-flop", "CDDM")):
        r = next(r for r in rows if r["task"] == task and r["pen"] == "frm" and "examples" in r)
        ex = r["examples"]
        for j, (side, lab) in enumerate((("lo", "burst units: lowest tPR / n"), ("hi", "sustained units: highest tPR / n"))):
            a = ax[i, j]
            X = ex[side]                                              # (EX, T, EX_TRIALS)
            T = X.shape[1]
            for u in range(X.shape[0]):
                tr = np.concatenate([X[u, :, b] for b in range(X.shape[2])])
                tr = tr / max(tr.max(), 1e-12)                        # each unit scaled to its own peak
                a.plot(np.arange(tr.size), tr + u * 1.1, lw=0.9, color=plt.cm.tab10(u))
            for b in range(1, X.shape[2]):
                a.axvline(b * T, color="0.85", lw=0.7)
            a.set(title=f"{task}, participation penalty, N={r['N']}: {lab}\n(tPR/n = " + ", ".join(f"{v:.2f}" for v in ex["tpr_" + side]) + ")",
                  xlabel=f"time, each unit's {X.shape[2]} most active trials concatenated", yticks=[])
            a.set_ylabel("rate / own peak (units offset)")
    fig.suptitle("S5 — what a burst unit and a sustained unit look like (four units each, six trials)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    return ps.save(fig, "fig_S5_examples", tight=False)


def tuning(rows):
    """Per-unit R^2 of the joint rectified fit (live units) and Hoyer of its coefficients (tuned units),
    N = EX_N, both tasks, four conditions, seeds pooled; medians in the legend."""
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for i, task in enumerate(("flip-flop", "CDDM")):
        rr = [r for r in rows if r["task"] == task and r["N"] == EX_N and "r2_units" in r]
        nf = rr[0]["n_factors"]
        for pen in PENS:
            r2 = np.concatenate([r["r2_units"] for r in rr if r["pen"] == pen])
            r2 = r2[np.isfinite(r2)]                                  # a unit with zero variance in Y has no R²
            hy = np.concatenate([r["hoyer_units"] for r in rr if r["pen"] == pen])
            hy = hy[np.isfinite(hy)]
            nl = int(np.mean([len(r["r2_units"]) for r in rr if r["pen"] == pen]))
            nt = int(np.mean([len(r["hoyer_units"]) for r in rr if r["pen"] == pen]))
            ax[i, 0].hist(np.clip(r2, 0, 1), bins=np.linspace(0, 1, 41), density=True, histtype="step", lw=1.8, color=COL[pen],
                          label=f"{LABEL[pen]}: median R² {np.median(r2):.2f}, {nl} live units")
            ax[i, 1].hist(hy, bins=np.linspace(0, 1, 41), density=True, histtype="step", lw=1.8, color=COL[pen],
                          label=f"{LABEL[pen]}: median {np.median(hy):.2f}, {nt} tuned units")
            print(f"{task:9s} {pen:5s} R2 median {np.median(r2):.3f} (q25 {np.quantile(r2, .25):.3f}, q75 {np.quantile(r2, .75):.3f}), "
                  f"tuned {nt}/{nl}; Hoyer median {np.median(hy):.3f} (q25 {np.quantile(hy, .25):.3f}, q75 {np.quantile(hy, .75):.3f})")
        ax[i, 0].axvline(R2_GATE, color="0.4", ls=":", lw=1)
        ax[i, 0].set(xlabel=f"R² of each live unit's response fitted on {nf} rectified task factors + constant", ylabel="density",
                     title=f"{task}, N={EX_N}: how well {nf} factors explain each unit")
        ax[i, 1].set(xlabel=f"Hoyer sparsity of the {nf} tuning coefficients (1 = one factor, 0 = all equal)", ylabel="density",
                     title=f"{task}, N={EX_N}: how many factors a tuned unit follows")
        ax[i, 0].legend(fontsize=7.5); ax[i, 1].legend(fontsize=7.5)
    fig.suptitle("S4 — per-unit tuning: R² of the rectified-factor fit and the sparsity of its coefficients (seeds pooled)", fontsize=10.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return ps.save(fig, "fig_S4_tuning", tight=False)


def main():
    """Dispatch on the figure name; 'compute' only fills the cache."""
    ps.setup()
    what = sys.argv[1] if len(sys.argv) > 1 else "compute"
    rows = list(compute().values())
    if what == "mixed":
        counts_vs_N(rows, "mixed", "fig_S4_mixed", "mixed-selective units",
                    "S4 — mixed-selective units (tuned; joint R² gain > 0.1 from a second variable)\nthe sparsity penalty removes them on the flip-flop and doubles them on CDDM")
    elif what == "bursts":
        counts_vs_N(rows, "burst", "fig_S5_bursts", f"burst units (live, active < {BURST:.0%} of the time)",
                    "S5 — burst units (live, active < 5% of the time)\nthe participation penalty makes them on both tasks; the sparsity penalty removes them on the flip-flop, not on CDDM")
    elif what == "cloud":
        cloud_cddm(rows)
    elif what == "examples":
        examples(rows)
    elif what == "tuning":
        tuning(rows)


if __name__ == "__main__":
    main()
