#!/usr/bin/env python3
"""
Do silent units distort the population-level analyses used to compare RNN models with neural data?

The comparison between a model circuit and a recorded population is made through summary statistics
of the population: how many dimensions it uses, what fraction of cells are tuned to each task
variable, how activity is distributed across cells. If half the model's units never fire, those
statistics describe a population that has no counterpart in cortex.

For every network this computes, over the noise-free CDDM batch:

  pr            participation ratio of the unit covariance, (sum L)^2 / sum L^2 — the standard
                "effective dimensionality" of population activity
  pr_over_N     the same as a fraction of the network, which is how "the circuit uses k dimensions
                out of N" is usually phrased
  sel_<var>     fraction of units selective to context / motion / colour / choice, by variance
                explained (eta^2 > SEL_THR) of the trial-averaged decision-epoch response
  sel_<var>_act the same fraction computed over ACTIVE units only — what you would measure if you
                recorded from the units that actually fire
  n_active      number of units that are not silent
  rate_cv_act   coefficient of variation (std/mean) of the per-unit mean firing rate, across ACTIVE
                units — how heterogeneous the active population is. Cortical rate distributions are
                strongly heterogeneous (roughly lognormal), so a network whose active units all fire
                at the same rate is LESS data-like in this respect, not more.
  rate_p90_p50  ratio of the 90th percentile to the median per-unit rate, across active units — the
                same heterogeneity read as a tail measure, robust to outliers
  sigma_log     std of log10(mean rate) across ACTIVE units. Cortical rate distributions are close to
                lognormal, and sigma_log is the shape parameter the literature reports — roughly 1 in
                log10 units, i.e. about a decade of spread between typical slow and fast cells. It is
                scale-free (rescaling all rates shifts the mean of the log, not its spread), unlike
                CV, and it is stable on heavy tails. Requires positive rates, hence active units only.
  within_cv     WITHIN-trial temporal variability: for each unit the temporal std of its rate is
                averaged over conditions and divided by its overall mean rate; reported as the median
                over active units. This asks whether a unit is actually MODULATED by the task or
                merely sits at a constant rate — the direct test of whether an activity penalty is
                satisfied by tonic firing rather than by useful dynamics.
  energy        total metabolic cost, sum over units of mean(fr^2)
  energy_hhi    concentration of that cost across units (1/N = even, 1 = one unit carries it all)
  mean_rate     mean firing rate over units, time and conditions

Reporting each statistic both over all units and over active units only is the point: the gap
between them is the distortion.

Usage:  python population_distortion.py [SWEEP_FOLDER ...] [--out out.csv]
"""

import os
import sys
import csv
import glob
import json
import re
import numpy as np
import hydra
from omegaconf import OmegaConf

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import unjsonify, filter_kwargs
from common import hhi, participation

SILENT_THR = 1e-6      # participation below this = silent
SEL_THR = 0.10         # eta^2 above this = selective to that variable


def participation_ratio(X):
    """Effective dimensionality (sum L)^2 / sum L^2 of the covariance of X (units x samples).

    Note this is invariant to appending all-zero units: they contribute zero eigenvalues, which
    change neither sum. That is itself a finding — see the module docstring.
    """
    Xc = X - X.mean(axis=1, keepdims=True)
    lam = np.linalg.eigvalsh(np.cov(Xc))
    lam = np.clip(lam, 0, None)
    return float(lam.sum() ** 2 / np.maximum((lam ** 2).sum(), 1e-30))


def eta2(resp, groups):
    """Variance of each unit's response explained by a categorical factor.

    Args:
        resp: (units, conditions) trial-averaged responses.
        groups: (conditions,) integer group label per condition.
    Returns:
        (units,) eta^2 in [0, 1].
    """
    total = resp.var(axis=1)
    between = np.zeros(resp.shape[0])
    gm = resp.mean(axis=1)
    for g in np.unique(groups):
        m = groups == g
        between += m.mean() * (resp[:, m].mean(axis=1) - gm) ** 2
    return between / np.maximum(total, 1e-30)


def analyse_net(folder, task, inputs, conditions, dec_on):
    """Compute the distortion statistics for one trained network folder."""
    cfgs = glob.glob(folder + "*_config.yaml")
    pjs = glob.glob(folder + "*_LastParams_*.json")
    if not cfgs or not pjs:
        return None
    cfg = OmegaConf.load(cfgs[0])
    params = unjsonify(json.load(open(pjs[0])))
    rnn = RNN_numpy(**OmegaConf.to_container(filter_kwargs(RNN_numpy, params), resolve=True), seed=0)
    fr, _ = PerformanceAnalyzer(rnn).get_firing_rate_trajectories(inputs)

    p = participation(fr)
    active = p >= SILENT_THR
    N = fr.shape[0]

    # trial-averaged response over the decision epoch, one value per unit per condition
    resp = fr[:, dec_on:, :].mean(axis=1)

    ctx = np.array([1 if c["context"] == "motion" else 0 for c in conditions])
    mot = np.array([np.sign(c["motion_coh"]) for c in conditions])
    col = np.array([np.sign(c["color_coh"]) for c in conditions])
    cho = np.array([c["correct_choice"] for c in conditions])

    rate = fr.mean(axis=(1, 2))          # per-unit mean firing rate
    ra = rate[active]
    ra_pos = ra[ra > 0]
    # within-trial temporal variability, per unit: mean over conditions of the temporal std,
    # normalised by the unit's own mean rate so it measures modulation rather than scale
    tstd = fr.std(axis=1).mean(axis=1)                      # (units,)
    within = tstd[active] / np.maximum(rate[active], 1e-30)
    row = {"N": N, "silent_frac": float((~active).mean()), "n_active": int(active.sum()),
           "rate_cv_act": float(ra.std() / np.maximum(ra.mean(), 1e-30)),
           "rate_p90_p50": float(np.percentile(ra, 90) / np.maximum(np.median(ra), 1e-30)),
           "sigma_log": float(np.std(np.log10(ra_pos))) if ra_pos.size > 1 else float("nan"),
           "within_cv": float(np.median(within)),
           "pr": participation_ratio(fr.reshape(N, -1)),
           "pr_active": participation_ratio(fr[active].reshape(active.sum(), -1)),
           "energy": float((fr ** 2).mean(axis=(1, 2)).sum()),
           "energy_hhi": hhi((fr ** 2).mean(axis=(1, 2))),
           "mean_rate": float(fr.mean())}
    row["pr_over_N"] = row["pr"] / N
    row["pr_over_active"] = row["pr"] / max(active.sum(), 1)
    for name, g in (("ctx", ctx), ("motion", mot), ("color", col), ("choice", cho)):
        e = eta2(resp, g)
        row[f"sel_{name}"] = float((e > SEL_THR).mean())
        row[f"sel_{name}_act"] = float((e[active] > SEL_THR).mean())
    return row


def main():
    """Run over every network of every sweep folder given, and write one CSV row per network."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    out = "population_distortion.csv"
    if "--out" in sys.argv:
        out = sys.argv[sys.argv.index("--out") + 1]
    folders = args or ["CDDM_std_g0"]

    cfg0 = OmegaConf.load(glob.glob(folders[0] + "/*/*/*_config.yaml")[0])
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg0.task, dt=cfg0.model.dt))
    inputs, _, conditions = task.get_batch()
    dec_on = int(cfg0.task.T_dec_on / cfg0.model.dt)
    print(f"batch: {inputs.shape}, decision epoch from step {dec_on}")

    rows = []
    for sweep in folders:
        for d in sorted(glob.glob(sweep + "/*/*/")):
            m = re.search(r"EqType=(\w+)_N=(\d+)_LmbdRWS=([\d.]+)_LmbdFR=([\d.]+)", d)
            if not m:
                continue
            r = analyse_net(d, task, inputs, conditions, dec_on)
            if r is None:
                continue
            r.update(sweep=os.path.basename(sweep.rstrip("/")), eq=m.group(1),
                     penalty=("none" if (m.group(3) == "0" and m.group(4) == "0") else
                              "rws" if m.group(4) == "0" else
                              "frm" if m.group(3) == "0" else "both"))
            rows.append(r)
            print(f"  {r['sweep']} {r['eq']}/{r['penalty']} N={r['N']}: "
                  f"silent {r['silent_frac']:.1%}, PR {r['pr']:.1f}, sel_ctx {r['sel_ctx']:.1%}")
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {out}: {len(rows)} networks")


if __name__ == "__main__":
    main()
