#!/usr/bin/env python3
"""
FUNCTIONAL diversity of units: are the recruited units distinct, or redundant copies?

⚠️ THIS IS A DIFFERENT QUESTION FROM EVERY HETEROGENEITY MEASURE ALREADY IN THE PROJECT.
`pr_matrix` (PR over units), `flipflop_temporal_pr` (PR over time) and `flipflop_heterogeneity`
(A1-A6: peak-to-mean, duty cycle, loophole fraction) all measure HOW MUCH and WHEN a unit is active.
None of them measures WHAT a unit does. That gap matters here: `both` has M/N = 1.00 at N=2000, k=8
- two thousand active units - inside an effective dimensionality of ~12.7. Either those units are
~160-fold redundant, or they carry diverse loadings within a low-dimensional space, and no measure
in the project currently distinguishes those.

Three measures, all on the noise-free rates of live units only (silent units would otherwise
dominate every one of them):

  D_PR      (Sum L)^2 / Sum L^2 of the COVARIANCE spectrum. Amplitude and shape mixed; this is what
            flipflop_dimensionality.py reports, repeated here so the comparison is like-for-like.
  D_shape   the same PR of the CORRELATION spectrum, i.e. after each unit's time course is centred
            and scaled to unit norm. AMPLITUDE-FREE: it counts distinct response SHAPES. The ratio
            D_shape / D_PR says how much of the apparent low dimensionality is a few loud units
            rather than genuine redundancy.
  redundancy  mean |correlation| over unit pairs. -> 1 means clones, -> 0 means unrelated profiles.

  selectivity_PR  each unit's rate regressed on the k target bit time courses; the k loading
            weights are normalised to unit length and their second-moment matrix is formed. Its PR
            is the effective number of SELECTIVITY directions the population spans, bounded by k.
            Near k = mixed selectivity spread over the task variables; near 1 = every unit tuned to
            the same combination.
  R2_median  median fraction of a unit's variance explained by those k regressors, so
            selectivity_PR is not read on units the task variables do not describe.

Output: img/internal_figures/diversity_N{N}_k{k}.png  (plus a table on stdout)

Usage:  python flipflop_diversity.py [N] [k]        (defaults 2000 3)
"""

import os
import sys
import glob
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP
import plotstyle as ps
from flipflop_heterogeneity import folders_for
from flipflop_fixedpoints import load_net
from trainRNNbrain.training.training_utils import prepare_task_arguments

PENS = ["none", "rws", "frm", "both"]
N_TRIALS = 48
COLS = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}


def pr_of(lam):
    """Participation ratio of a non-negative spectrum.

    Args:
        lam: (n,) eigenvalues.
    Returns:
        float (Sum lam)^2 / Sum lam^2, or nan if the spectrum is empty.
    """
    lam = np.clip(np.asarray(lam, float), 0, None)
    s = lam.sum()
    return float(s * s / (lam ** 2).sum()) if s > 0 else float("nan")


def rates_and_targets(folder, n_trials=N_TRIALS):
    """Noise-free rates and the target bit time courses for the same trials.

    Args:
        folder: run folder; n_trials: batch size.
    Returns:
        (rates, targets): (N, T, B) and (k, T, B).
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    cfg.task.batch_size = n_trials
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, targets, _ = task.get_batch()
    net, _ = load_net(folder)
    net.clear_history(); net.y = net.y_init
    net.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    return np.maximum(np.array(net.get_history()), 0.0), targets


def measure(rates, targets):
    """All diversity measures for one network.

    Args:
        rates: (N, T, B) firing rates; targets: (k, T, B) target bit values.
    Returns:
        dict of scalars, plus 'offdiag' (the pairwise |correlation| values) for plotting.
    """
    N = rates.shape[0]
    X = rates.reshape(N, -1).astype(np.float64)
    live = (X.std(1) + np.quantile(X, 0.9, axis=1)) >= SILENT_FLIPFLOP
    X = X[live]
    Xc = X - X.mean(1, keepdims=True)

    d_pr = pr_of(np.linalg.eigvalsh(Xc @ Xc.T))                    # covariance spectrum
    nrm = np.linalg.norm(Xc, axis=1, keepdims=True)
    Xn = Xc / np.maximum(nrm, 1e-300)
    C = Xn @ Xn.T                                                  # correlation matrix
    d_shape = pr_of(np.linalg.eigvalsh(C))
    iu = np.triu_indices(C.shape[0], k=1)
    off = np.abs(C[iu])

    # selectivity: regress each unit on the k target bit time courses (plus an intercept)
    G = targets.reshape(targets.shape[0], -1).T                    # (samples, k)
    G = np.column_stack([np.ones(G.shape[0]), G])
    beta, *_ = np.linalg.lstsq(G, Xc.T, rcond=None)                # (k+1, n_live)
    resid = Xc.T - G @ beta
    ss_tot = (Xc.T ** 2).sum(0)
    r2 = 1.0 - (resid ** 2).sum(0) / np.maximum(ss_tot, 1e-300)
    B = beta[1:].T                                                 # (n_live, k) loadings
    Bn = B / np.maximum(np.linalg.norm(B, axis=1, keepdims=True), 1e-300)
    sel_pr = pr_of(np.linalg.eigvalsh(Bn.T @ Bn))                  # bounded by k

    return dict(live=float(live.mean()), d_pr=d_pr, d_shape=d_shape,
                ratio=d_shape / d_pr, redundancy=float(off.mean()),
                sel_pr=sel_pr, r2_med=float(np.median(r2)), offdiag=off)


def main():
    """Compute and plot the diversity measures for every penalty at one (N, k) cell."""
    N = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    k = int(sys.argv[2]) if len(sys.argv) > 2 else 3
    ps.setup()
    res = {}
    for pen in PENS:
        rows = []
        for folder in folders_for(pen, N, k):
            rr, tt = rates_and_targets(folder)
            rows.append(measure(rr, tt))
        if rows:
            res[pen] = rows

    print(f"\nN={N}, k={k}, {N_TRIALS} trials, live units only "
          f"(participation >= {SILENT_FLIPFLOP:g})\n")
    print(f"{'pen':<6}{'live':>7}{'D_PR':>8}{'D_shape':>9}{'ratio':>7}"
          f"{'redundancy':>12}{'sel_PR':>8}{'  (max %d)' % k}{'R2_med':>9}")
    for pen in PENS:
        if pen not in res:
            continue
        m = lambda kk: float(np.mean([r[kk] for r in res[pen]]))
        s = lambda kk: float(np.std([r[kk] for r in res[pen]]))
        print(f"{pen:<6}{m('live'):>7.3f}{m('d_pr'):>8.2f}{m('d_shape'):>9.1f}"
              f"{m('ratio'):>7.1f}{m('redundancy'):>12.3f}{m('sel_pr'):>8.2f}"
              f"{'':>10}{m('r2_med'):>9.3f}   (n={len(res[pen])}, sd D_shape {s('d_shape'):.1f})")

    fig, ax = plt.subplots(1, 3, figsize=(15.5, 4.4))
    xs = [p for p in PENS if p in res]
    for a, key, lab, note in [
            (ax[0], "d_shape", "$D_{shape}$", "effective number of distinct response SHAPES\n"
                                              "(amplitude removed)"),
            (ax[1], "redundancy", "mean |correlation|", "1 = clones, 0 = unrelated profiles"),
            (ax[2], "sel_pr", "selectivity PR", f"effective selectivity directions (max k={k})")]:
        mu = [np.mean([r[key] for r in res[p]]) for p in xs]
        sd = [np.std([r[key] for r in res[p]]) for p in xs]
        a.bar(xs, mu, yerr=sd, color=[COLS[p] for p in xs], alpha=.85, capsize=4)
        for i, v in enumerate(mu):
            a.text(i, v, f"{v:.2f}" if v < 20 else f"{v:.0f}", ha="center", va="bottom", fontsize=9)
        a.set(ylabel=lab, title=f"{lab}\n{note}")
        a.grid(alpha=.25, axis="y")
        if key == "sel_pr":
            a.axhline(k, ls="--", c="0.4", lw=1)
            a.text(.02, k, f" k={k} ceiling", va="bottom", fontsize=7.5, color="0.4",
                   transform=a.get_yaxis_transform())
    fig.suptitle(f"Functional diversity of active units — N={N}, k={k}\n"
                 "are recruited units distinct, or redundant copies?", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    return ps.save(fig, f"diversity_N{N}_k{k}", tight=False)


if __name__ == "__main__":
    main()
