#!/usr/bin/env python3
"""
Re-score the recurrent-noise sweep onto the PARTICIPATION rule every other panel uses.

WHY THIS EXISTS. CDDM_fb2792_g0_noise saved no participation traces, so its silence number came from
a peak-rate rule - a unit silent below 5% of the 95th-percentile peak rate - and read 524 active for
its sigma = 0.05 reference. The metabolic reference, the same task at the same size and the same
30,000-iteration budget, reads 414 under the participation rule. The gap was the measuring stick, not
the networks, and it made the deck's slide 14 look inconsistent with slides 12 and 13.

The trained weights ARE on disk, so each net is rebuilt and run noise-free on the CDDM batch - the
condition Trainer.track_participation_ logs under, and the condition every other active-unit count in
this project is measured in - and scored with the same scale-free rule. Re-scored, the sigma = 0.05
reference reads 443, which sits beside the metabolic sweep's 414 as two independent sweeps of the
same thing rather than as two different measurements.

⚠️ THIS SWEEP STORES PARAMETERS AS JSON, not the npz every later sweep uses - it predates the npz
format. The npz loader the other offline re-scores use finds nothing here and returns no nets at all,
silently, which is how this looked like "the sweep has no data" rather than "the loader is wrong".

Usage:  python cddm_noise_participation.py [--refresh]
Output: data/trained_RNNs/CDDM_fb2792_g0_noise/silent_units_per_condition_participation.csv
        (eq, sigma_rec, active_mean, active_std, n_nets, active_counts) - the columns
        fig_paper_F1.noise_active reads, plus the per-seed counts the original sweep never kept.
"""

import argparse
import csv
import glob
import json
import os
import sys

import numpy as np
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..")))
from common import DATA_DIR, active_count
from fig_slides import cddm_batch_and_mask
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.utils import filter_kwargs, unjsonify

SWEEP = os.path.join(DATA_DIR, "CDDM_fb2792_g0_noise")
OUT = os.path.join(SWEEP, "silent_units_per_condition_participation.csv")
SIGMAS = ["0", "0.01", "0.05", "0.1"]
EQ = "h"
# A STRIDED subsample of the trial axis, not the first N. TaskCDDM.get_batch enumerates the
# coherence x context grid in a fixed order, so the leading trials are one context out of two.
TRIALS = 128


def offline_participation(folder, inputs):
    """Per-unit participation of one net's final parameters on a noise-free run.

    Args:
        folder: a net folder holding `*LastParams*.json` and `*_config.yaml`;
        inputs: (n_inputs, T, B) input batch to drive the network with.
    Returns:
        1-D participation array std(fr) + q_0.9(|fr|) of length N, or None if the net cannot be
        rebuilt from what the folder holds.
    """
    pf = glob.glob(os.path.join(folder, "*LastParams*.json"))
    cfgs = glob.glob(os.path.join(folder, "*_config.yaml"))
    if not pf or not cfgs:
        return None
    p = unjsonify(json.load(open(pf[0])))
    cfg = OmegaConf.load(cfgs[0])
    p["activation_args"] = OmegaConf.to_container(cfg.model.activation_args, resolve=True)
    # the config is the authority on the equation form, and this JSON carries it too - passing both
    # is a duplicate keyword
    p.pop("equation_type", None)
    rnn = RNN_numpy(**filter_kwargs(RNN_numpy, p),
                    equation_type=str(cfg.model.equation_type), seed=0)
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    fr = rnn.get_firing_rate_history()                       # (N, T, B)
    r = fr.reshape(fr.shape[0], -1)
    return r.std(axis=1) + np.quantile(np.abs(r), 0.9, axis=1)


def cell_counts(sigma, eq=EQ):
    """Active-unit counts, per seed, for one recurrent-noise level.

    Args:
        sigma: sigma_rec as it appears in the cell directory name; eq: equation type to read.
    Returns:
        list of ints, one per net, empty if the cell has no usable nets.
    """
    cell = os.path.join(SWEEP, f"EqType={eq}_N=1000_sigrec={sigma}")
    folders = sorted(d for d in glob.glob(os.path.join(cell, "*/"))
                     if os.path.basename(d.rstrip("/")).split("_")[0] != "nan")
    if not folders:
        return []
    inputs, _target, _mask, _var = cddm_batch_and_mask(folders[0])
    sub = np.arange(0, inputs.shape[2], max(1, inputs.shape[2] // TRIALS))[:TRIALS]
    inputs = inputs[:, :, sub]
    out = []
    for f in folders:
        p = offline_participation(f, inputs)
        if p is not None:
            out.append(int(active_count(p, "scalefree")))
    return out


def main(refresh=False):
    """Write the per-condition CSV. Returns its path, or None if nothing could be scored."""
    if os.path.exists(OUT) and not refresh:
        print(f"{OUT} exists; pass --refresh to recompute")
        return OUT
    rows = []
    for sigma in SIGMAS:
        counts = cell_counts(sigma)
        if not counts:
            print(f"  sigma {sigma}: no nets")
            continue
        # the per-seed counts go in too. The original sweep kept only a per-condition summary, which
        # is why slide 14 was the one panel that could not draw its individual networks.
        rows.append(dict(eq=EQ, sigma_rec=sigma, active_mean=f"{np.mean(counts):.4f}",
                         active_std=f"{np.std(counts, ddof=1) if len(counts) > 1 else 0.0:.4f}",
                         n_nets=len(counts),
                         active_counts=";".join(str(c) for c in counts)))
        print(f"  sigma {sigma:>5s}: {counts}  mean {np.mean(counts):.0f}")
    if not rows:
        return None
    with open(OUT, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["eq", "sigma_rec", "active_mean", "active_std", "n_nets",
                                           "active_counts"])
        w.writeheader()
        w.writerows(rows)
    print(f"\nwrote {OUT}")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute even if the CSV exists")
    main(**vars(ap.parse_args()))
