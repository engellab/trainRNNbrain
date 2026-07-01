#!/usr/bin/env python3
"""
Silent-unit counts for the recurrent-noise sweep (CDDM_fb2792_g0_noise).

Same scoring as count_silent_units.py, but the condition directories are named
`EqType=<eq>_N=<N>_sigrec=<sig>` (training recurrent-noise level), so we parse
sigma_rec instead of the (lambda_rws, lambda_frm) penalty pair and group nets by
(equation_type, sigma_rec).

Every net is evaluated on the SAME noise-free CDDM batch (sigma_rec=0 in the forward
pass) regardless of the noise it was trained with — we are measuring the silence that
got baked into the trained weights, exactly as the baseline/penalty sweeps do.

A unit is "silent" if its peak firing rate over (time, conditions) < 0.01 (dead_abs);
a scale-free companion criterion (peak < 5% of the net's 95th-pct peak) is also reported.

Output: per-(eq, sigma_rec) mean +/- std silent count/fraction, printed and written to
        <ROOT>/silent_units_per_condition.csv.

Usage:  python3 count_silent_units_noise.py [SWEEP_FOLDER_FULL_PATH]
"""

import os
import re
import sys
import csv
import glob
import json
import numpy as np
from collections import defaultdict
from omegaconf import OmegaConf, DictConfig

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.utils import unjsonify, filter_kwargs
from trainRNNbrain.training.training_utils import prepare_task_arguments
import hydra

ROOT = sys.argv[1] if len(sys.argv) > 1 else \
    "/Users/pt1290/Documents/GitHub/trainRNNbrain/data/trained_RNNs/CDDM_fb2792_g0_noise"
DEAD_ABS = 0.01       # absolute peak-rate floor for a "dead" unit
REL_FRAC = 0.05       # relative peak-rate fraction (vs net's 95th pct) for "silent"
COND_RE = re.compile(r"EqType=(?P<eq>[hs])_N=(?P<N>\d+)_sigrec=(?P<sig>[\d.]+)")


def build_task(any_config_path):
    """Instantiate the CDDM task and return one noise-free input batch.

    Args:
        any_config_path: path to any saved <score>_config.yaml (task config is
            constant across the sweep, so one batch serves every net).
    Returns:
        input_batch: ndarray (n_inputs, T, n_conditions) for RNN_numpy.run.
    """
    cfg = OmegaConf.load(any_config_path)
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_cfg)
    input_batch, _, _ = task.get_batch()
    return input_batch


def net_peak_rates(params_json, input_batch):
    """Reconstruct one RNN from LastParams and return its per-unit peak firing rate.

    Args:
        params_json: path to a <score>_LastParams_CDDM.json file.
        input_batch: CDDM input batch (n_inputs, T, n_conditions).
    Returns:
        peak: ndarray (N,) of each unit's max firing rate over time and conditions.
    """
    with open(params_json) as f:
        params = unjsonify(json.load(f))
    net_cfg = filter_kwargs(RNN_numpy, params)
    if isinstance(net_cfg, DictConfig):
        net_cfg = OmegaConf.to_container(net_cfg, resolve=True)
    rnn = RNN_numpy(**net_cfg, seed=0)

    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=input_batch, sigma_rec=0, sigma_inp=0)
    fr = rnn.get_firing_rate_history()            # (N, T, B)
    return np.abs(fr).max(axis=(1, 2))            # (N,) peak rate per unit


def main():
    leaf_dirs = sorted(glob.glob(os.path.join(ROOT, "EqType=*", "*")))
    leaf_dirs = [d for d in leaf_dirs if os.path.isdir(d)]
    if not leaf_dirs:
        raise SystemExit(f"No net folders found under {ROOT}")

    sample_cfg = glob.glob(os.path.join(leaf_dirs[0], "*_config.yaml"))[0]
    input_batch = build_task(sample_cfg)
    print(f"CDDM batch: inputs shape {input_batch.shape} (n_inputs, T, n_conditions)\n")

    by_cond = defaultdict(list)
    for d in leaf_dirs:
        cond = os.path.basename(os.path.dirname(d))
        m = COND_RE.search(cond)
        if m is None:
            print(f"  [skip] cond name did not match sigrec regex: {cond}")
            continue
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))
        if not pj:
            print(f"  [skip] no LastParams in {d}")
            continue
        peak = net_peak_rates(pj[0], input_batch)
        N = peak.size
        hi = np.quantile(peak, 0.95)
        dead_abs = int(np.sum(peak < DEAD_ABS))
        silent_rel = int(np.sum(peak < REL_FRAC * hi)) if hi > 0 else N
        by_cond[cond].append(dict(N=N, dead_abs=dead_abs, silent_rel=silent_rel,
                                  med_peak=float(np.median(peak)), p95_peak=float(hi)))

    # sort conditions by (eq, sigma_rec) numerically
    def sortkey(c):
        m = COND_RE.search(c)
        return (m["eq"], float(m["sig"]))

    rows = []
    header = f"{'condition':44s} {'nets':>4s} {'N':>5s}  {'dead<0.01':>16s}  {'silent<5%p95':>16s}  {'med_peak':>8s}"
    print(header)
    print("-" * len(header))
    for cond in sorted(by_cond, key=sortkey):
        recs = by_cond[cond]
        N = recs[0]["N"]
        da = np.array([r["dead_abs"] for r in recs], float)
        sr = np.array([r["silent_rel"] for r in recs], float)
        mp = np.mean([r["med_peak"] for r in recs])
        da_pct = 100 * da / N
        sr_pct = 100 * sr / N
        print(f"{cond:44s} {len(recs):>4d} {N:>5d}  "
              f"{da.mean():5.1f}+-{da.std():4.1f}({da_pct.mean():4.1f}%)  "
              f"{sr.mean():5.1f}+-{sr.std():4.1f}({sr_pct.mean():4.1f}%)  {mp:8.3f}")
        m = COND_RE.search(cond)
        rows.append([cond, m["eq"], float(m["sig"]), len(recs), N,
                     round(da.mean(), 2), round(da.std(), 2), round(da_pct.mean(), 1),
                     round(sr.mean(), 2), round(sr.std(), 2), round(sr_pct.mean(), 1),
                     round(mp, 4)])

    out_csv = os.path.join(ROOT, "silent_units_per_condition.csv")
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["condition", "eq", "sigma_rec", "n_nets", "N",
                    "dead_abs_mean", "dead_abs_std", "dead_abs_pct",
                    "silent_rel_mean", "silent_rel_std", "silent_rel_pct", "median_peak_rate"])
        w.writerows(rows)
    print(f"\nwrote {out_csv}")


if __name__ == "__main__":
    main()
