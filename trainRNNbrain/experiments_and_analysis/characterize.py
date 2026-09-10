#!/usr/bin/env python3
"""
Full characterisation of none / rws / frm / both networks on BOTH tasks (CDDM and n-bit flip-flop)
with ONE statistic: Hoyer sparsity, bounded 0..1.

  hoyer(v) = (sqrt(d) - ||v||_1 / ||v||_2) / (sqrt(d) - 1)     for a non-negative vector v of length d
           = 0 when every entry is equal (v is spread over all d slots)
           = 1 when a single entry carries everything
  ||v||_1 / ||v||_2 = sqrt(PR(v)), so this is the participation ratio normalised to [0, 1] and flipped:
  hoyer = (sqrt(d) - sqrt(PR)) / (sqrt(d) - 1).

The same statistic applied to four different vectors gives the four axes of the characterisation:

  dim      v = eigenvalues of the unit covariance, d = N.   1 = one population dimension, 0 = N.
           (effective dimensionality; also reported raw as D_PR)
  part     v = per-unit participation p_i = std + q90, d = N.   1 = one unit does everything,
           0 = every unit participates equally. (how concentrated the activity is over units)
  temp     v = one unit's rate over all (time, trial) samples, d = S; median over live units.
           1 = fires at one instant, 0 = constant. (temporal sparseness; the dual of temporal PR)
  sel      v = |regression coefficients| of one unit on rectified task variables, d = number of
           regressors; median over tuned units. 1 = one variable, 0 = evenly mixed.
           MIXED SELECTIVITY IS DEFINED HERE AS 1 - sel: the degree to which a unit's tuning is
           spread over several task variables in a rectified linear regression.

Regressors.  flip-flop: relu(+b_j), relu(-b_j) for the k bits, time-resolved over every sample,
             plus an intercept (as in flipflop_arms.py).
             CDDM: per-condition decision-epoch mean response regressed on [ctx (+-1),
             relu(+-motion coh), relu(+-colour coh), relu(+-choice)] plus an intercept: 7 coefficients.
             The intercept is never part of the Hoyer vector.
             ⚠️ R2 and tuned fraction are NOT comparable across tasks (time-resolved vs trial-mean
             regression); orderings across penalties WITHIN a task are.

Live units.  flip-flop: p >= SILENT_FLIPFLOP (4e-2, Otsu-calibrated). CDDM: scale-free rule
             p >= 0.05 q95(p), because rws parks units just above the 1e-6 line (paper §2.1).
Tuned units. live and R2 >= R2_GATE.

Also fits M = A N^b to the active count of the unpenalised (and rws) nets on each task and reports
the N at which 1000 and 2000 active units would be reached. ⚠️ End-of-training networks, whose
budgets differ by N on CDDM (large N trained for fewer iterations, hence read EARLIER in their
silencing, hence with MORE active units): the fitted exponent is therefore an over-estimate and
the N-needed an under-estimate. Conservative for the claim "a lot of units are needed".

Data: CDDM none from CDDM_std_g0_drift (N=100..5000) + CDDM_std_g0_N10k; rws/frm/both from
CDDM_std_g0_penalties (N=500..5000). Flip-flop k=3 from run_folders() (N=500..4000).
CDDM is simulated on every third condition (150 of 450) so N=10000 fits in memory.

Usage:  python characterize.py [--recompute]
Output: img/internal_figures/characterize_matrix.png, data/characterize_cache.pkl
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, SILENT_REL, participation
import plotstyle as ps
from pr_matrix import PENS
from flipflop_dimensionality import run_folders
from flipflop_diversity import rates_and_targets
from flipflop_fixedpoints import load_net
from trainRNNbrain.training.training_utils import prepare_task_arguments

CACHE = "data/characterize_cache.pkl"
R2_GATE = 0.15
FF_K = 3
FF_TRIALS = 32
CDDM_ROOTS = ["data/trained_RNNs/CDDM_std_g0_drift", "data/trained_RNNs/CDDM_std_g0_penalties",
              "data/trained_RNNs/CDDM_std_g0_N10k"]
CDDM_STRIDE = 3
COL = {"none": "#7f7f7f", "rws": "#2ca02c", "frm": "#d62728", "both": "#1f77b4"}
MEASURES = [("active_frac", "active fraction M/N", "task criterion; line = power-law fit for none"),
            ("dim", "dimensionality sparsity", "Hoyer of covariance eigenvalues; 1 = one dimension"),
            ("part", "participation sparsity", "Hoyer of unit participation; 1 = one unit"),
            ("temp", "temporal sparsity", "median Hoyer of a unit's trace; 1 = one instant"),
            ("sel", "selectivity sparsity", "median Hoyer of tuning coefficients; 1 = one variable")]


def hoyer(v):
    """Hoyer sparsity of a non-negative vector (or of each row of a 2-D array).

    Args:
        v: (d,) or (n, d) array, entries >= 0.
    Returns:
        float or (n,) values in [0, 1]; 0 = all entries equal, 1 = a single non-zero entry.
    """
    v = np.atleast_2d(np.asarray(v, dtype=np.float64))
    d = v.shape[1]
    l1 = v.sum(1)
    l2 = np.sqrt((v ** 2).sum(1))
    h = (np.sqrt(d) - l1 / np.maximum(l2, 1e-300)) / (np.sqrt(d) - 1)
    return h[0] if h.shape[0] == 1 else h


def cddm_folders():
    """Every CDDM run folder with its (penalty, N).

    Returns:
        list of (folder, pen, N).
    """
    out = []
    for root in CDDM_ROOTS:
        for cond in sorted(glob.glob(os.path.join(root, "EqType=h_N=*"))):
            m = re.search(r"N=(\d+)(?:_pen=([a-z]+))?", os.path.basename(cond))
            for f in sorted(glob.glob(os.path.join(cond, "*", "*LastParams*.npz"))):
                out.append((os.path.dirname(f), m.group(2) or "none", int(m.group(1))))
    return out


def cddm_rates(folder):
    """Noise-free CDDM rates on every CDDM_STRIDE-th condition, plus the regression design.

    Args:
        folder: run folder.
    Returns:
        (rates (N, T, B) float32, G (B, 8) design with intercept first, dec_on int).
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, _, conditions = task.get_batch()
    inputs, conditions = inputs[:, :, ::CDDM_STRIDE], conditions[::CDDM_STRIDE]
    net, _ = load_net(folder)
    net.clear_history(); net.y = net.y_init
    net.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    rates = np.maximum(np.array(net.get_history(), dtype=np.float32), 0.0)
    ctx = np.array([1.0 if c["context"] == "motion" else -1.0 for c in conditions])
    mot = np.array([c["motion_coh"] for c in conditions])
    col = np.array([c["color_coh"] for c in conditions])
    cho = np.array([c["correct_choice"] for c in conditions], dtype=float)
    G = np.column_stack([np.ones_like(ctx), ctx, np.maximum(mot, 0), np.maximum(-mot, 0),
                         np.maximum(col, 0), np.maximum(-col, 0), np.maximum(cho, 0), np.maximum(-cho, 0)])
    return rates, G, int(cfg.task.T_dec_on / cfg.model.dt)


def measure(rates, live, Y, G):
    """The four Hoyer measures (plus their raw companions) for one network.

    Args:
        rates: (N, T, B) non-negative rates; live: (N,) bool; Y: (N, n_samples) responses to regress
               (time-resolved traces or per-condition means); G: (n_samples, p+1) design, intercept
               first.
    Returns:
        dict of scalars.
    """
    N = rates.shape[0]
    X = rates.reshape(N, -1).astype(np.float64)
    p = participation(rates)
    Xl = X[live] - X[live].mean(1, keepdims=True)
    lam = np.clip(np.linalg.eigvalsh(Xl @ Xl.T), 0, None) if live.sum() > 1 else np.zeros(1)
    lam_full = np.zeros(N); lam_full[:lam.size] = lam
    row = dict(N=N, n_live=int(live.sum()), active_frac=float(live.mean()),
               d_pr=float(lam.sum() ** 2 / max((lam ** 2).sum(), 1e-300)),
               dim=float(hoyer(lam_full)), part=float(hoyer(p)))
    th = hoyer(X[live])
    row["temp"] = float(np.median(th))
    row["tpr_med"] = float(np.median(X[live].sum(1) ** 2 / np.maximum((X[live] ** 2).sum(1), 1e-300)
                                     / X.shape[1]))
    Yc = Y[live] - Y[live].mean(1, keepdims=True)
    beta, *_ = np.linalg.lstsq(G, Yc.T, rcond=None)
    resid = Yc.T - G @ beta
    r2 = 1.0 - (resid ** 2).sum(0) / np.maximum((Yc.T ** 2).sum(0), 1e-300)
    B = np.abs(beta[1:].T)
    tuned = (r2 >= R2_GATE) & (B.sum(1) > 0)
    row["tuned_frac"] = float(tuned.mean()) if live.sum() else np.nan
    row["r2_med"] = float(np.median(r2[tuned])) if tuned.any() else np.nan
    row["sel"] = float(np.median(hoyer(B[tuned]))) if tuned.any() else np.nan
    row["tuned_per_N"] = float(tuned.sum() / N)
    return row


def compute():
    """Simulate and measure every network on both tasks, with a per-folder cache.

    Returns:
        list of dict rows with 'task', 'pen', 'N', 'folder' and the measures.
    """
    cache = pickle.load(open(CACHE, "rb")) if os.path.exists(CACHE) else {}
    jobs = [("flipflop", f, pen, N) for f, pen, k, N in run_folders() if k == FF_K]
    jobs += [("cddm", f, pen, N) for f, pen, N in cddm_folders()]
    rows = []
    for i, (task, folder, pen, N) in enumerate(jobs, 1):
        if folder not in cache:
            if task == "flipflop":
                rates, targets = rates_and_targets(folder, FF_TRIALS)
                live = participation(rates) >= SILENT_FLIPFLOP
                Braw = targets.reshape(targets.shape[0], -1).T
                G = np.column_stack([np.ones(Braw.shape[0]), np.maximum(Braw, 0), np.maximum(-Braw, 0)])
                Y = rates.reshape(rates.shape[0], -1)
            else:
                rates, G, dec_on = cddm_rates(folder)
                p = participation(rates)
                live = p >= SILENT_REL * np.quantile(p, 0.95)
                Y = rates[:, dec_on:, :].mean(1)
            cache[folder] = measure(rates, live, Y, G)
            pickle.dump(cache, open(CACHE, "wb"))
            print(f"  {i}/{len(jobs)} {task} {pen} N={N}: live {cache[folder]['n_live']}, "
                  f"dim {cache[folder]['dim']:.3f} part {cache[folder]['part']:.3f} "
                  f"temp {cache[folder]['temp']:.3f} sel {cache[folder]['sel']:.3f}", flush=True)
        rows.append(dict(cache[folder], task=task, pen=pen, folder=folder))
    return rows


def fit_law(rows, task, pen):
    """Power law M = A N^b through the active counts of one condition.

    Args:
        rows: output of compute(); task, pen: condition.
    Returns:
        (A, b, N_for_1000, N_for_2000) or None if fewer than 3 sizes.
    """
    pts = [(r["N"], r["n_live"]) for r in rows if r["task"] == task and r["pen"] == pen and r["n_live"] > 0]
    if len({n for n, _ in pts}) < 3:
        return None
    lN, lM = np.log([n for n, _ in pts]), np.log([m for _, m in pts])
    b, a = np.polyfit(lN, lM, 1)
    A = np.exp(a)
    return A, b, (1000 / A) ** (1 / b), (2000 / A) ** (1 / b)


def main():
    """Print the characterisation tables and the active-count laws; draw the matrix figure."""
    if "--recompute" in sys.argv and os.path.exists(CACHE):
        os.remove(CACHE)
    ps.setup()
    rows = compute()
    for task in ("cddm", "flipflop"):
        print(f"\n{'='*100}\n{task.upper()}{'' if task == 'cddm' else f' (k={FF_K})'} — mean ± sd over seeds\n{'='*100}")
        print(f"{'pen':<6}{'N':>6}{'live':>12}{'dim':>14}{'D_PR':>10}{'part':>14}{'temp':>14}{'sel':>14}{'tuned':>8}{'R2':>7}")
        for pen in PENS:
            for N in sorted({r["N"] for r in rows if r["task"] == task and r["pen"] == pen}):
                rr = [r for r in rows if r["task"] == task and r["pen"] == pen and r["N"] == N]
                f = lambda key: f"{np.nanmean([r[key] for r in rr]):.3f}±{np.nanstd([r[key] for r in rr]):.3f}"
                print(f"{pen:<6}{N:>6}{np.mean([r['n_live'] for r in rr]):>12.0f}{f('dim'):>14}"
                      f"{np.mean([r['d_pr'] for r in rr]):>10.1f}{f('part'):>14}{f('temp'):>14}{f('sel'):>14}"
                      f"{np.nanmean([r['tuned_frac'] for r in rr]):>8.2f}{np.nanmean([r['r2_med'] for r in rr]):>7.2f}")
        print("\nactive-count law M = A N^b (end-of-training nets):")
        for pen in PENS:
            law = fit_law(rows, task, pen)
            if law:
                print(f"  {pen:<5} b = {law[1]:.3f}   N for 1000 active: {law[2]:.3g}   for 2000: {law[3]:.3g}")

    fig, ax = plt.subplots(len(MEASURES), 2, figsize=(13, 3.2 * len(MEASURES)))
    for j, task in enumerate(("cddm", "flipflop")):
        for i, (key, lab, note) in enumerate(MEASURES):
            a = ax[i, j]
            for pen in PENS:
                Ns = sorted({r["N"] for r in rows if r["task"] == task and r["pen"] == pen})
                if not Ns:
                    continue
                mu = [np.nanmean([r[key] for r in rows if r["task"] == task and r["pen"] == pen and r["N"] == N]) for N in Ns]
                sd = [np.nanstd([r[key] for r in rows if r["task"] == task and r["pen"] == pen and r["N"] == N]) for N in Ns]
                ps.band(a, Ns, np.array(mu), np.array(sd), COL[pen], label=pen)
                if key == "active_frac" and pen == "none" and fit_law(rows, task, pen):
                    A, b, *_ = fit_law(rows, task, pen)
                    nn = np.geomspace(min(Ns), max(Ns), 50)
                    a.plot(nn, A * nn ** b / nn, "--", color=COL[pen], lw=1, label=f"none fit: M ∝ N^{b:.2f}")
            a.set(xscale="log", xlabel="N", ylabel=lab, title=f"{task.upper() if task == 'cddm' else f'flip-flop k={FF_K}'}: {lab}\n{note}")
            if key != "active_frac":
                a.set_ylim(0, 1)
            a.grid(alpha=.25); a.legend(fontsize=7)
    fig.suptitle("Four Hoyer sparsities (0 = spread, 1 = concentrated) over N, per penalty, both tasks\n"
                 "shaded = spread across seeds", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    return ps.save(fig, "characterize_matrix", tight=False)


if __name__ == "__main__":
    main()
