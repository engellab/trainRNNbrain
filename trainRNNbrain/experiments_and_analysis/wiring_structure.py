#!/usr/bin/env python3
"""
Wiring statistics behind elaboration claims S3 (the loophole's wiring side), CS1 and CS3: how many
units a unit listens to, whether it listens to units of its own role, and whether the recurrent
wiring is modular. Both tasks, four conditions, N = 2000, every seed; one simulation pass per network
cached in data/wiring_cache.pkl. Replaces the scratchpad analysis of 2026-09-10 (14:31, 15:36, 16:17).

Per network:
  S_i        effective in-degree of unit i, (sum_j |W_ij|)^2 / sum_j W_ij^2 over its recurrent row
             (20 equal inputs -> 20; 2000 equal inputs -> 2000; one dominant input -> 1)
  role       each tuned unit's preferred rectified task factor: argmax over the joint regression
             coefficients (flip-flop: 2k = 6 bit states; CDDM: 7 factors), on live units with
             joint R^2 >= R2_GATE (unit_stats.py machinery)
  share_i    fraction of unit i's incoming |W| mass (FULL row, never top-k) that comes from units of
             the same role; chance = fraction of tuned units in that role; reported as share / chance
  Q_wiring   modularity of a spectral partition of |W_rec| into n_roles clusters, minus the same on a
             within-row-permuted W (preserves every unit's in-degree); matched n = MATCH_N live units
  Q_activity modularity of a spectral partition of the activity correlation matrix into n_roles
             clusters, minus the same with every unit's time series permuted (destroys correlation)
  ARI        adjusted Rand index between the wiring and activity partitions (matched n)
  Q_task     modularity of |W_rec| under the TASK-ROLE partition itself (labels, no clustering), minus
             the row-shuffle null (matched n, tuned units)
  like2like  Spearman correlation over unit pairs between |W_ij| + |W_ji| and the activity correlation
             |corr(r_i, r_j)| (no labels, no partition; the Ko 2011 statistic), matched n

The five wiring statistics use different ingredients: Q_wiring (clustering, no labels), Q_task
(labels, no clustering), like2like (neither), share (labels, per unit), ARI (two partitions).

Figures:
  python wiring_structure.py            -> fig_CS1_wiring.png  (2 x 2: rows = tasks; left in-degree S per
                                           unit, right same-role share over chance; four conditions)
  python wiring_structure.py --recompute
The modularity / ARI table is printed.
"""

import os
import sys
import pickle
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.cluster import SpectralClustering
from sklearn.metrics import adjusted_rand_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import plotstyle as ps
from characterize import cddm_folders, cddm_rates, FF_K, FF_TRIALS
from common import SILENT_FLIPFLOP, SILENT_REL, participation
from flipflop_dimensionality import run_folders
from flipflop_diversity import rates_and_targets
from flipflop_fixedpoints import load_net
from unit_stats import r2_of, R2_GATE

CACHE = "data/wiring_cache.pkl"
N_TARGET = 2000
MATCH_N = 275
N_SUB = 3
PENS = ["none", "rws", "frm", "both"]
COL = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
LABEL = {"none": "no penalty", "rws": "sparsity only", "frm": "participation only", "both": "participation + sparsity"}


def modularity(A, labels):
    """Newman modularity of a labelled partition on a non-negative weighted adjacency A (symmetric)."""
    k = A.sum(1); m = A.sum() / 2
    if m <= 0:
        return 0.0
    q = 0.0
    for c in np.unique(labels):
        idx = labels == c
        q += A[np.ix_(idx, idx)].sum() / (2 * m) - (k[idx].sum() / (2 * m)) ** 2
    return float(q)


def spectral(A, n):
    """Spectral partition of a non-negative symmetric affinity into n clusters."""
    return SpectralClustering(n_clusters=n, affinity="precomputed", random_state=0, assign_labels="discretize").fit_predict(A + 1e-9)


def wiring_stats(W, W_out, rates, live, role, n_roles, rng):
    """In-degree, same-role share, and matched-n modularity / ARI for one network.

    Args:
        W: (N, N) recurrent weights, W[i, j] = j -> i; W_out: (n_out, N); rates: (N, T, B);
        live: (N,) bool; role: (N,) int with -1 for untuned units; n_roles: number of roles; rng.
    Returns:
        dict of per-unit arrays (S, share_over_chance on tuned units) and matched-n scalars.
    """
    absW = np.abs(W)
    S = absW.sum(1) ** 2 / np.maximum((W ** 2).sum(1), 1e-300)
    tuned = role >= 0
    share = np.full(W.shape[0], np.nan)
    for i in np.flatnonzero(tuned):
        same = (role == role[i]) & tuned
        same[i] = False
        tot = absW[i, tuned].sum() - absW[i, i]
        chance = (same.sum()) / max(tuned.sum() - 1, 1)
        share[i] = (absW[i, same].sum() / tot) / max(chance, 1e-12) if tot > 0 else np.nan
    X = rates.reshape(rates.shape[0], -1)
    qs = dict(Qw=[], Qw0=[], Qa=[], Qa0=[], ari=[], Qt=[], Qt0=[], l2l=[])
    live_idx = np.flatnonzero(live)
    tuned_idx = np.flatnonzero(tuned)
    for _ in range(N_SUB):
        sub = rng.choice(live_idx, min(MATCH_N, live_idx.size), replace=False)
        A = absW[np.ix_(sub, sub)]; A = 0.5 * (A + A.T); np.fill_diagonal(A, 0)
        lw = spectral(A, n_roles)
        Wp = absW[np.ix_(sub, sub)].copy()
        for r in range(Wp.shape[0]):
            Wp[r] = rng.permutation(Wp[r])
        Ap = 0.5 * (Wp + Wp.T); np.fill_diagonal(Ap, 0)
        C = np.corrcoef(X[sub]); C = np.nan_to_num(np.abs(C)); np.fill_diagonal(C, 0)
        la = spectral(C, n_roles)
        Xp = np.stack([rng.permutation(x) for x in X[sub]])
        Cp = np.corrcoef(Xp); Cp = np.nan_to_num(np.abs(Cp)); np.fill_diagonal(Cp, 0)
        qs["Qw"].append(modularity(A, lw)); qs["Qw0"].append(modularity(Ap, spectral(Ap, n_roles)))
        qs["Qa"].append(modularity(C, la)); qs["Qa0"].append(modularity(Cp, spectral(Cp, n_roles)))
        qs["ari"].append(adjusted_rand_score(lw, la))
        # like-to-like: pair weight vs pair activity correlation, no labels, no partition
        iu = np.triu_indices(sub.size, 1)
        Wsym = absW[np.ix_(sub, sub)] + absW[np.ix_(sub, sub)].T
        qs["l2l"].append(float(spearmanr(Wsym[iu], C[iu]).correlation))
        # task-role partition on tuned units: labels, no clustering
        subt = rng.choice(tuned_idx, min(MATCH_N, tuned_idx.size), replace=False)
        At = absW[np.ix_(subt, subt)]; At = 0.5 * (At + At.T); np.fill_diagonal(At, 0)
        Wtp = absW[np.ix_(subt, subt)].copy()
        for r in range(Wtp.shape[0]):
            Wtp[r] = rng.permutation(Wtp[r])
        Atp = 0.5 * (Wtp + Wtp.T); np.fill_diagonal(Atp, 0)
        qs["Qt"].append(modularity(At, role[subt])); qs["Qt0"].append(modularity(Atp, role[subt]))
    return dict(S=S, share=share, tuned=tuned, **{k: float(np.mean(v)) for k, v in qs.items()})


def one(task, folder, rng):
    """All wiring statistics for one network folder."""
    net, _ = load_net(folder)
    if task == "flip-flop":
        rates, targets = rates_and_targets(folder, FF_TRIALS)
        live = participation(rates) >= SILENT_FLIPFLOP
        B = targets.reshape(targets.shape[0], -1).T
        G = np.column_stack([np.ones(B.shape[0])] + [np.maximum(s * B[:, j], 0) for j in range(B.shape[1]) for s in (1, -1)])
        Y = rates.reshape(rates.shape[0], -1)
    else:
        rates, G, dec_on = cddm_rates(folder)
        p = participation(rates)
        live = p >= SILENT_REL * np.quantile(p, 0.95)
        Y = rates[:, dec_on:, :].mean(1)
    Yc = Y - Y.mean(1, keepdims=True)
    r2, beta = r2_of(G, Yc)
    role = np.where(live & (r2 >= R2_GATE), np.argmax(np.abs(beta[1:]), axis=0), -1)
    st = wiring_stats(net.W_rec, net.W_out, rates, live, role, G.shape[1] - 1, rng)
    st.update(live=int(live.sum()), n_tuned=int((role >= 0).sum()))
    return st


def compute(recompute=False):
    """Fill the cache for every N = N_TARGET network on both tasks."""
    cache = {} if recompute else (pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {})
    jobs = [("flip-flop", f, pen) for f, pen, k, N in run_folders() if k == FF_K and N == N_TARGET]
    jobs += [("CDDM", f, pen) for f, pen, N in cddm_folders() if N == N_TARGET]
    rng = np.random.default_rng(0)
    for i, (task, folder, pen) in enumerate(jobs, 1):
        if folder in cache:
            continue
        st = one(task, folder, rng)
        cache[folder] = dict(st, task=task, pen=pen)
        pickle.dump(cache, open(CACHE, "wb"))
        print(f"  {i}/{len(jobs)} {task} {pen}: live {st['live']} tuned {st['n_tuned']}  S median {np.median(st['S'][st['tuned']]):.0f}  "
              f"share/chance median {np.nanmedian(st['share']):.2f}  Qw {st['Qw']:.3f} (null {st['Qw0']:.3f})  "
              f"Qa {st['Qa']:.3f} (null {st['Qa0']:.3f})  ARI {st['ari']:.2f}", flush=True)
    return cache


def main():
    """Draw fig_CS1_wiring.png and print the modularity table."""
    ps.setup()
    rows = list(compute("--recompute" in sys.argv).values())
    fig, ax = plt.subplots(2, 2, figsize=(12, 7.5))
    for i, task in enumerate(("flip-flop", "CDDM")):
        for pen in PENS:
            rr = [r for r in rows if r["task"] == task and r["pen"] == pen]
            if not rr:
                continue
            S = np.concatenate([r["S"][r["tuned"]] for r in rr])
            sh = np.concatenate([r["share"][r["tuned"]] for r in rr]); sh = sh[np.isfinite(sh)]
            ax[i, 0].hist(np.log10(S), bins=np.linspace(0, 3.4, 45), density=True, histtype="step", lw=1.8, color=COL[pen],
                          label=f"{LABEL[pen]}: median S = {np.median(S):.0f}")
            ax[i, 1].hist(sh, bins=np.linspace(0, 8, 45), density=True, histtype="step", lw=1.8, color=COL[pen],
                          label=f"{LABEL[pen]}: median {np.median(sh):.2f}×")
            print(f"{task:9s} {pen:5s} S median {np.median(S):.0f}  share/chance median {np.median(sh):.2f}  "
                  f"Qw excess {np.mean([r['Qw'] - r['Qw0'] for r in rr]):.3f}  Qtask excess {np.mean([r['Qt'] - r['Qt0'] for r in rr]):.3f}  "
                  f"like2like {np.mean([r['l2l'] for r in rr]):.3f} ± {np.std([r['l2l'] for r in rr]):.3f}  "
                  f"Qa excess {np.mean([r['Qa'] - r['Qa0'] for r in rr]):.3f}  ARI {np.mean([r['ari'] for r in rr]):.2f} ± {np.std([r['ari'] for r in rr]):.2f}")
        ax[i, 0].set(xlabel="log10 effective in-degree S of a tuned unit", ylabel="density", title=f"{task}, N={N_TARGET}: how many units a unit listens to")
        ax[i, 0].axvline(np.log10(20), color="0.4", ls=":", lw=1); ax[i, 0].text(np.log10(20) + 0.03, ax[i, 0].get_ylim()[1] * 0.9, "target 20", fontsize=8, color="0.4")
        ax[i, 1].axvline(1, color="0.4", ls=":", lw=1); ax[i, 1].text(1.05, ax[i, 1].get_ylim()[1] * 0.9, "chance", fontsize=8, color="0.4")
        ax[i, 1].set(xlabel="same-role share of a tuned unit's input weight, over chance", ylabel="density",
                     title=f"{task}, N={N_TARGET}: does a unit listen to units of its own role?")
        ax[i, 0].legend(fontsize=7.5); ax[i, 1].legend(fontsize=7.5)
    fig.suptitle("CS1 — wiring: the participation penalty alone makes units listen to everyone at chance; the sparsity penalty caps the in-degree and restores same-role wiring",
                 fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return ps.save(fig, "fig_CS1_wiring", tight=False)


if __name__ == "__main__":
    main()
