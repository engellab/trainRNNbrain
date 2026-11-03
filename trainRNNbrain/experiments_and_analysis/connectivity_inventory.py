#!/usr/bin/env python3
"""Inventory of candidate unseen default networks of N = 1,000 for the r12-weights confirmatory test.

For every run of every candidate cell: the config fields that define the standard network (final run
and first run of the warm-start chain), the floor crossing (common.read_run), the kept snapshot at the
crossing, and every link of the warm-start chain with the iteration its final weights stand for.
Reads only; computes no connectivity statistic. Runs on Della with PYTHONPATH=<analysis checkout>.

Usage (Della, CPU job): python connectivity_inventory.py <out.json>
"""

import glob
import json
import os
import sys

from omegaconf import OmegaConf

sys.path.insert(0, os.path.join(os.environ["PYTHONPATH"].split(":")[0], "trainRNNbrain", "experiments_and_analysis"))
from common import read_run, warm_start_chain    # noqa: E402

D = "/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs"
CELLS = [
    "NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none",
    "NBitFlipFlop_std_dropfix_c200000/EqType=h_k=3_N=1000_pen=none_do=none",
    "CDDM_paper_grid/EqType=h_N=1000_arm=control",
    "CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000",
    "CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05_iters=200000",
    "CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05",
    "CDDM_std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000",
    "CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
    "CDDM_std_g0_bias0/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
    "CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=none",
    "CDDM_screen3/EqType=h_N=1000_pen=none_arm=none",
    "CDDM_c1_snapshots/EqType=h_N=1000_arm=control",
    "CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=0_iters=200000",
    "CDDM_std_g0_wd0_long/EqType=h_N=1000_wd=0_iters=300000",
    "DMTS_paper_grid_d5/EqType=h_N=1000_arm=control",
    "DMTS_paper_grid/EqType=h_N=1000_arm=control",
    "DMTS_std_settings/EqType=h_N=1000_arm=default_iters=150000",
    "DMTS_std_settings/EqType=h_N=1000_arm=wd0_iters=150000",
    "DMTS_c1_snapshots/EqType=h_N=1000_arm=control",
]
KEYS = ["model.activation_args.name", "model.equation_type", "model.N", "model.bias_range", "model.dale",
        "model.gamma", "model.self_connections", "model.spectral_rad", "model.sigma_rec", "model.sigma_inp",
        "model.init_draws", "model.input_row_norm", "model.inp_weight_cap", "model.connectivity_density_rec",
        "trainer.lr", "trainer.weight_decay", "trainer.same_batch", "trainer.lambda_orth", "trainer.lambda_met",
        "trainer.lambda_rws", "trainer.lambda_frm", "trainer.dropout", "trainer.prune_reinit",
        "trainer.synaptic_scaling", "trainer.max_iter", "trainer.iter_offset", "trainer.checkpoint_keep",
        "task.taskname", "task.T", "seed", "paths.init_from", "paths.checkpoint_to"]


def summary(cfg_path):
    """The KEYS of one config as a dict (missing keys as None)."""
    c = OmegaConf.load(cfg_path)
    return {k: (OmegaConf.to_container(v) if OmegaConf.is_config(v) else v)
            for k in KEYS for v in [OmegaConf.select(c, k, default=None)]}


def main():
    """Write one record per run of every cell to argv[1]."""
    out = []
    for cell in CELLS:
        for d in sorted(glob.glob(os.path.join(D, cell, "*relu*"))):
            if os.path.basename(d).startswith("nan"):
                continue
            chain = warm_start_chain(d)
            links = []
            for c in chain:
                cf = sorted(glob.glob(os.path.join(c, "*_config.yaml")))[0]
                s = summary(cf)
                links.append(dict(run=c, config=cf, end_iter=(s["trainer.iter_offset"] or 0) + s["trainer.max_iter"],
                                  last_params=sorted(glob.glob(os.path.join(c, "*LastParams*.npz"))), cfg=s))
            try:
                r = read_run(d)
                rr = dict(t_star=r["t_star"], reason=r["reason"], snapshot=r["snapshot"], active=r["active"],
                          active_end=r["active_end"], it_end=r["it_end"], floor_kind=r["floor_kind"])
            except Exception as e:                       # record and go on; inventory only
                rr = dict(error=repr(e))
            out.append(dict(cell=cell, run=d, links=links, read=rr))
            print(cell, os.path.basename(d)[:40], rr.get("t_star"), rr.get("reason"), rr.get("snapshot"), flush=True)
    with open(sys.argv[1], "w") as fh:
        json.dump(out, fh, indent=1, default=str)


if __name__ == "__main__":
    main()
