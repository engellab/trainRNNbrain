#!/usr/bin/env python3
"""Read the first Della batch of the uniform-interventions programme at N = 1,000 and 75,000 iterations (issue #6).

QUESTION. Eleven Della arrays (entries 2026-11-05_0903, _2256 and _2259) fill the supervised cells
that the interventions matrix (docs/interventions_matrix.md) found missing: 3BFF weight decay 1e-5
and 1e-4, CDDM rescaling, the CDDM optimiser comparison, the CDDM rate penalty 0.1 alone and with the
sparsity penalty 0.05, the 3BFF settings ladders (activity noise, spectral radius, recurrent-weight
penalty, metabolic penalty, sparsity penalty alone, activations, input-weight cap 0.3) and the third
seeds of the CDDM spectral-radius and recurrent-weight-penalty cells. This script reads every run
that has finished, applies the pre-registered criteria of those three entries, and lists the array
tasks still running or pending. It is read_uniform_cells.py (yesterday's read-out of the existing
cells) with two more verdict kinds; its loaders and its rule criterion are imported from there.

WHICH RUNS. data/della_batch1_status.tsv holds one row per array task (job_task, Slurm state, the
seed and experiment tag from the job log, the trainer's skipped-update line), written from `sacct`
and ~/trainRNNbrain_logs on Della on 6 Nov. A run folder only exists once training has finished
(its name starts with the final score), so every folder under a cell is a completed run; the table
says which completed tasks must have a folder on the Mac (a missing one is printed) and which tasks
are still pending (listed per arm). Completed runs were copied to data/trained_RNNs/ with rsync
(config, TrainLosses.json, ParticipationTrace.pkl, final weights; no PNGs, optimiser states or
checkpoints).

READ-OUT (the same for every cell; the pre-registered one). At FIXED_IT = 75,000 iterations, the end
of these runs: the active count (common.active_mask on the stored participation trace, measured on
a noise-free pass, at the probe nearest 75,000) and the held-out R^2 under the training noise
(heldout_r2: sigma_rec and sigma_inp from the run's config, sigma_w = 0, R2_DRAWS seeded draws) on
the final weights, which are the 75,000-iteration weights (r2_how "final"). Controls and comparison
cells that end later are read as the caches hold them: the 3BFF control
NBitFlipFlop_b5_optimizer/..._opt=adam_wd=1e-6_lr=1e-3 and the B5 arms from data/b5_read_cache.pkl
(R^2 at 75k estimated from the 200k weights, r2_how "estimate"), the CDDM ladders of Figure 3 and the
flip-flop's second control from data/fig_paper_F2_fails_cache.pkl; active counts of cached cells
whose traces are on the Mac are re-read from disk and checked against the cache. A run is broken and
excluded, and counted, if its noisy loss has a non-finite value or if the trainer skipped more than
MAX_SKIP of its gradient updates (the job log's "gradient spikes" line).

CRITERIA (docs/interventions_matrix.md and the three entries):
  rule (CDDM rescaling, rate penalty 0.1, rate 0.1 + sparsity 0.05): "works" if every seed has more
    active units than the largest control count and the arm's mean R^2 is at least the controls'
    mean minus R2_TOL (read_uniform_cells.verdict); otherwise "fails", with the condition that failed.
  setting (3BFF ladders, CDDM third seeds): "increase" or "decrease" if every seed lies outside the
    range of the control's seeds, on the same side; otherwise "no change". A cell whose mean R^2 is
    below R2_INTERPRET has its count reported but not interpreted. The uniform question is whether
    the direction matches the other task's ladder at 75,000, read with the same rule against that
    ladder's own control; the CDDM third seeds are read against the sr 1.2 cell.
  optimiser (CDDM AdamW, SGD 0.03-1, with the ladder's Adam wd 0 cell): a run with R^2 below
    R2_LEARNED did not learn and is excluded, counted. Class A "more active units at no cost in R^2"
    (count and R^2 conditions of the rule criterion both pass), class B "more active units only at
    lower R^2" (count passes, R^2 fails), class C "does not increase the number of active units".
    CDDM reproduces the 3BFF pattern if AdamW and Adam wd 0 are in class A and every SGD arm whose
    count passes is in class B. The 3BFF B5 arms are classed the same way beside them.
  An arm with fewer finished runs than N_SEEDS gets a provisional verdict and its pending tasks.
  The tanh cell is also read for saturation (entry 2259): on a noise-free pass over the held-out
    batch a unit is responsive when mean(1 - r^2) > RESPONSIVE_SLOPE, else saturated (pinned near
    +1 or -1), the test of fig_tanh_saturation.py, with its trace proxy (p < P_RESPONSIVE) beside it.

OUTPUT. One table per group on stdout, and data/read_uniform_cells_2.json: per run cell, name,
seed, job task, t_read, probe, active, r2, r2_how, r2_end, it_end, skipped share; per arm the runs,
the verdict dict, the config keys on which it differs from its control, and its pending tasks; the
other task's ladder where a group compares directions; the broken runs; every pending task.

Usage (repository root, CPU; about 40 minutes the first time, then from the held-out cache):
  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/read_uniform_cells_2.py
  --draws 36 (noise draws per held-out R^2), --out <json>, --status <tsv>
"""

import argparse
import csv
import glob
import json
import os
import pickle
import re
import sys
import time

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
from common import DATA_DIR, runs_with_loss
from fig_paper_F2_fails import FIXED_IT
from fig_tanh_saturation import P_RESPONSIVE
from read_remedies import SHARED_FIELDS, cfg_of, sel
from read_uniform_cells import (F2_CACHE, R2_TOL, CDDM_CTRL, CDDM_CTRL_F3A, read_cell, read_cell_from_f2_cache,
                                seed_of, verdict)
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import import_any, filter_kwargs

B5_CACHE = os.path.normpath(os.path.join(DATA_DIR, "..", "b5_read_cache.pkl"))
STATUS = os.path.normpath(os.path.join(DATA_DIR, "..", "della_batch1_status.tsv"))
OUT = os.path.normpath(os.path.join(DATA_DIR, "..", "read_uniform_cells_2.json"))
N_SEEDS = 3             # runs per cell the pre-registrations expect
R2_LEARNED = 0.8        # an optimiser run below this held-out R^2 did not learn the task (entry 2256)
R2_INTERPRET = 0.7      # a setting cell with mean R^2 below this has its count reported, not interpreted (entry 2259)
MAX_SKIP = 0.05         # a run that skipped more than this share of its gradient updates is broken
RESPONSIVE_SLOPE = 0.5  # a tanh unit is responsive when mean(1 - r^2) over the trial set exceeds this (fig_tanh_saturation.py)
# the keys an intervention may change, printed beside the shared settings when an arm differs on them
INTERVENTION_KEYS = ("model.spectral_rad", "model.inp_weight_cap", "model._target_", "trainer.optimizer",
                     "trainer.momentum", "trainer.lambda_rwm", "trainer.lambda_met", "trainer.lambda_rws",
                     "trainer.lambda_frm", "trainer.prune_reinit", "trainer.prune_args", "trainer.light_outputs",
                     "trainer.checkpoint_every", "trainer.max_iter")
CHECK_FIELDS = SHARED_FIELDS + INTERVENTION_KEYS

FF_CTRL = "NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=adam_wd=1e-6_lr=1e-3"
FF_CTRL2 = "NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000"
CDDM_DRIFT = "CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000"       # the CDDM ladders' default rung
CDDM_WD0 = "CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=0_iters=200000"
CDDM_SR12 = "CDDM_std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000"
CDDM_NOISE = "CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec={v}_iters=200000"
CDDM_PEN = "CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS={rws}_LmbdFR={frm}"


def b5(opt, wd, lr):
    """The B5 cell path of one 3BFF optimiser arm (opt, weight decay, base learning rate as strings)."""
    return f"NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt={opt}_wd={wd}_lr={lr}"


def ff(sweep, tag):
    """A 3BFF package-2 cell path: NBitFlipFlop_<sweep>/EqType=h_N=1000_<tag>_iters=75000."""
    return f"NBitFlipFlop_{sweep}/EqType=h_N=1000_{tag}_iters=75000"


# A source names where a cell's read comes from: "local" (read_cell on the folders on the Mac),
# "f2" (fig_paper_F2_fails cache) or "b5" (b5_read_cache). Arms are (label, cell, source).
GROUPS = [
    dict(name="3BFF weight decay", task="3-bit flip-flop", kind="setting",
         control=(FF_CTRL, "b5"), second=(FF_CTRL2, "f2"),
         arms=[("wd 0", b5("adam", "0", "1e-3"), "b5"),
               ("wd 1e-5", ff("std_g0_weightdecay_bias0", "wd=1e-5"), "local"),
               ("wd 1e-4", ff("std_g0_weightdecay_bias0", "wd=1e-4"), "local")],
         other=dict(task="CDDM (Figure 3b ladder, 200k runs read at 75k)", control=(CDDM_DRIFT, "f2"),
                    arms=[("wd 0", CDDM_WD0, "f2"),
                          ("wd 1e-5", "CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=1e-5_iters=200000", "f2"),
                          ("wd 1e-4", "CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=1e-4_iters=200000", "f2")])),
    dict(name="CDDM rescaling", task="CDDM", kind="rule", control=(CDDM_CTRL, "local"), second=(CDDM_CTRL_F3A, "f2"),
         arms=[("rescaling (target 10x the rate cap, the paper's arm)", "CDDM_target_ladder/EqType=h_N=1000_tgt=10.0", "local")]),
    dict(name="CDDM optimiser", task="CDDM", kind="optimiser", control=(CDDM_CTRL, "local"), second=(CDDM_CTRL_F3A, "f2"),
         arms=[("Adam wd 0", CDDM_WD0, "f2"),
               ("AdamW", "CDDM_b5_optimizer/EqType=h_N=1000_opt=adamw_wd=1e-6_lr=1e-3", "local")] +
              [(f"SGD lr {lr}", f"CDDM_b5_optimizer/EqType=h_N=1000_opt=sgd_wd=1e-6_lr={lr}", "local")
               for lr in ("0.03", "0.1", "0.3", "1")],
         other=dict(task="3-bit flip-flop (B5, 200k runs read at 75k)", control=(FF_CTRL, "b5"),
                    arms=[("Adam wd 0", b5("adam", "0", "1e-3"), "b5"), ("AdamW", b5("adamw", "1e-6", "1e-3"), "b5")] +
                         [(f"SGD lr {lr}", b5("sgd", "1e-6", lr), "b5") for lr in ("0.03", "0.1", "0.3", "1")])),
    dict(name="CDDM rate penalty 0.1", task="CDDM", kind="rule", control=(CDDM_CTRL, "local"), second=(CDDM_CTRL_F3A, "f2"),
         arms=[("rate 0.1", "CDDM_std_g0_penalties_bias0/EqType=h_N=1000_pen=frm0.1_iters=75000", "local"),
               ("rate 0.1 + sparsity 0.05", "CDDM_std_g0_penalties_bias0/EqType=h_N=1000_pen=frm0.1_rws0.05_iters=75000", "local")],
         references=[("rate 0.2 (5 runs)", CDDM_PEN.format(rws="0", frm="0.2"), "f2"),
                     ("rate 0.2 + sparsity 0.05 (5 runs)", CDDM_PEN.format(rws="0.05", frm="0.2"), "f2")]),
    dict(name="3BFF activity noise", task="3-bit flip-flop", kind="setting",
         control=(FF_CTRL, "b5"), second=(FF_CTRL2, "f2"),
         arms=[(f"sigma_rec {v}", ff("std_g0_noise_bias0", f"sigrec={v}"), "local") for v in ("0", "0.01", "0.1")],
         other=dict(task="CDDM (Figure 3 noise ladder, 5 runs per rung)", control=(CDDM_NOISE.format(v="0.05"), "f2"),
                    arms=[(f"sigma_rec {v}", CDDM_NOISE.format(v=v), "f2") for v in ("0", "0.01", "0.1")])),
    dict(name="3BFF spectral radius and recurrent-weight penalty", task="3-bit flip-flop", kind="setting",
         control=(FF_CTRL, "b5"), second=(FF_CTRL2, "f2"),
         arms=[("spectral radius 0.5", ff("std_g0_recinit", "sr=0.5"), "local"),
               ("spectral radius 2.0", ff("std_g0_recinit", "sr=2.0"), "local"),
               ("lambda_rwm 0.05", ff("std_g0_recinit", "rwm=0.05"), "local"),
               ("lambda_rwm 0.2", ff("std_g0_recinit", "rwm=0.2"), "local")],
         other=dict(task="CDDM (CDDM_std_g0_recinit with the third seeds)", control=(CDDM_SR12, "local"),
                    arms=[("spectral radius 0.5", "CDDM_std_g0_recinit/EqType=h_N=1000_sr=0.5_iters=75000", "local"),
                          ("spectral radius 2.0", "CDDM_std_g0_recinit/EqType=h_N=1000_sr=2.0_iters=75000", "local"),
                          ("lambda_rwm 0.05", "CDDM_std_g0_recinit/EqType=h_N=1000_rwm=0.05_iters=75000", "local"),
                          ("lambda_rwm 0.2", "CDDM_std_g0_recinit/EqType=h_N=1000_rwm=0.2_iters=75000", "local")])),
    dict(name="CDDM spectral radius and recurrent-weight penalty, third seeds", task="CDDM", kind="setting",
         control=(CDDM_SR12, "local"), second=None,
         arms=[("spectral radius 0.5", "CDDM_std_g0_recinit/EqType=h_N=1000_sr=0.5_iters=75000", "local"),
               ("spectral radius 2.0", "CDDM_std_g0_recinit/EqType=h_N=1000_sr=2.0_iters=75000", "local"),
               ("lambda_rwm 0.05", "CDDM_std_g0_recinit/EqType=h_N=1000_rwm=0.05_iters=75000", "local"),
               ("lambda_rwm 0.2", "CDDM_std_g0_recinit/EqType=h_N=1000_rwm=0.2_iters=75000", "local")]),
    dict(name="3BFF metabolic penalty", task="3-bit flip-flop", kind="setting",
         control=(FF_CTRL, "b5"), second=(FF_CTRL2, "f2"),
         arms=[(f"lambda_met {v}", ff("std_g0_metabolic_bias0", f"LmbdMet={w}"), "local")
               for v, w in (("0.01", "0.01"), ("0.1", "0.1"), ("1", "1.0"), ("10", "10.0"))],
         other=dict(task="CDDM (Figure 3 metabolic ladder, 75k top-ups)", control=(CDDM_DRIFT, "f2"),
                    arms=[(f"lambda_met {v}", f"CDDM_std_g0_metabolic_bias0_c75000/EqType=h_N=1000_LmbdMet={w}", "f2")
                          for v, w in (("0.01", "0.01"), ("0.1", "0.1"), ("1", "1.0"), ("10", "10.0"))])),
    dict(name="3BFF sparsity penalty alone", task="3-bit flip-flop", kind="setting",
         control=(FF_CTRL, "b5"), second=(FF_CTRL2, "f2"),
         arms=[("lambda_rws 0.05", "NBitFlipFlop_std_penlong/EqType=h_k=3_N=1000_pen=rws_iters=75000", "local")],
         other=dict(task="CDDM (Figure 3a, 5 runs per cell)", control=(CDDM_CTRL_F3A, "f2"),
                    arms=[("lambda_rws 0.05", CDDM_PEN.format(rws="0.05", frm="0"), "f2")])),
    dict(name="3BFF activations", task="3-bit flip-flop", kind="setting",
         control=(FF_CTRL, "b5"), second=(FF_CTRL2, "f2"),
         arms=[(a, ff("std_g0_activations_bias0", f"act={t}"), "local")
               for a, t in (("softplus beta 25", "softplus25"), ("leaky ReLU 0.01", "leakyrelu"),
                            ("shifted sigmoid", "sigmoid"), ("tanh", "tanh"))],
         other=dict(task="CDDM (Figure 3 activations, 200k runs read at 75k)", control=(CDDM_DRIFT, "f2"),
                    arms=[(a, f"CDDM_std_g0_activations_bias0/EqType=h_N=1000_act={t}_iters=200000", "f2")
                          for a, t in (("softplus beta 25", "softplus25"), ("leaky ReLU 0.01", "leakyrelu"),
                                       ("shifted sigmoid", "sigmoid"))])),
    dict(name="3BFF input-weight cap 0.3", task="3-bit flip-flop", kind="setting",
         control=(FF_CTRL, "b5"), second=(FF_CTRL2, "f2"),
         arms=[("cap 0.3", "NBitFlipFlop_screen3/EqType=h_N=1000_pen=none_arm=cap_cap=0.3_iters=75000", "local")],
         other=dict(task="CDDM (Figure 3e, screen3 top-ups)", control=("CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=none", "f2"),
                    arms=[("cap 0.3", "CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=cap", "f2")])),
]


def load_status(path):
    """The per-task status table.

    Args:
        path: a TSV with columns job_task, state, seed, tag, skipped_updates (see the module docstring).
    Returns:
        list of row dicts, with `skipped` (int updates skipped, parsed from the trainer's line, 0 when
        the line is absent) added to each.
    """
    with open(path) as fh:
        rows = list(csv.DictReader(fh, delimiter="\t"))
    for r in rows:
        m = re.match(r"(\d+) updates skipped", r["skipped_updates"] or "")
        r["skipped"] = int(m.group(1)) if m else 0
    return rows


def tasks_of(cell, status):
    """The status rows whose tag is this cell (the cell path relative to DATA_DIR)."""
    return [r for r in status if r["tag"] == cell]


def read_cell_from_b5_cache(cell, b5c):
    """A B5 cell's FIXED_IT read as b5_read_cache.pkl holds it, with the active count re-read from disk.

    Args:
        cell: a NBitFlipFlop_b5_optimizer cell path; b5c: the loaded cache, a list of
            ((opt, wd, lr), runs) pairs, each run holding `fixed` (t_read, probe, active, r2, r2_how).
    Returns:
        list of run dicts as read_cell's (without r2_end): cell, name, seed, t_read, probe, active,
        active_cache, r2, r2_how, r2_source = "b5_read_cache.pkl"; empty when the cell is not cached.
    """
    m = re.search(r"opt=([a-z]+)_wd=([^_]+)_lr=([^/]+)$", cell)
    runs = dict((tuple(k), v) for k, v in b5c).get((m.group(1), m.group(2), m.group(3)), []) if m else []
    local = {r["name"]: r for r in runs_with_loss(os.path.join(DATA_DIR, cell))}
    out = []
    for r in runs:
        f = r["fixed"]
        rec = dict(cell=cell, name=r["name"], seed=seed_of(r["name"]), t_read=f["t_read"], probe=f["probe"],
                   active=int(f["active"]), active_cache=int(f["active"]), r2=float(f["r2"]), r2_how=f["r2_how"],
                   r2_source="b5_read_cache.pkl")
        if r["name"] in local:
            lr = local[r["name"]]
            it_a = np.asarray(lr["it_act"], float)
            k = int(np.argmin(np.abs(it_a - f["t_read"])))
            rec["active"], rec["probe"] = int(lr["active"][k]), float(it_a[k])
            if rec["active"] != rec["active_cache"]:
                print(f"    MISMATCH {r['name'][:40]}: trace says {rec['active']}, cache {rec['active_cache']}")
        out.append(rec)
        print(f"    {cell[-52:]:52s} seed {rec['seed']:>10s}  active {rec['active']:4d}  "
              f"R2 {rec['r2']:.4f} ({rec['r2_how']}, B5 cache)", flush=True)
    return out


def tanh_saturation(net_dir):
    """Saturation of a tanh network's units on one noise-free pass over the held-out batch.

    The test of fig_tanh_saturation.py: a unit is responsive when the mean slope of its activation,
    mean(1 - r^2) over every time step and trial (r = tanh(x), x the state), exceeds RESPONSIVE_SLOPE;
    otherwise it is saturated (pinned near +1 or -1, so its activation passes no gradient), which
    participation scores as active. The network is rebuilt from its config as heldout_r2.score_net
    does, with the final weights, and run on heldout_r2.heldout_batch with w_noise=False. Beside it,
    the trace proxy of the same script: units with participation below P_RESPONSIVE at the last probe.

    Args:
        net_dir: a run folder holding *LastParams*, *_config.yaml and *ParticipationTrace.pkl.
    Returns:
        dict(responsive, saturated: unit counts; N; slope_median: the median over units of
        mean(1 - r^2); frac_pinned: the share of (unit, step, trial) samples with |r| > 0.9;
        responsive_trace: units with p < P_RESPONSIVE at the last probe), or None without weights.
    """
    cfgs = sorted(glob.glob(os.path.join(net_dir, "*_config.yaml")))
    params, _ = H._params(net_dir)
    if not cfgs or params is None:
        return None
    cfg = OmegaConf.load(cfgs[0])
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in tcfg:
        del tcfg._target_
    rnn = hydra.utils.instantiate(filter_kwargs(import_any(cfg.model._target_),
                                                OmegaConf.merge(OmegaConf.create(cfg.model), tcfg)))
    with torch.no_grad():
        for nm, prm in rnn.named_parameters():
            if nm in params:
                prm.copy_(torch.from_numpy(np.asarray(params[nm], dtype=np.float32)))
        if "y_init" in params and getattr(rnn, "y_init", None) is not None:
            rnn.y_init.copy_(torch.from_numpy(np.asarray(params["y_init"], dtype=np.float32)))
        bi, _, _ = H.heldout_batch(cfg)
        r = rnn.activation(rnn(bi, w_noise=False)[0]).numpy()          # (N, T, B)
    slope = (1.0 - r ** 2).reshape(r.shape[0], -1).mean(axis=1)
    with open(glob.glob(os.path.join(net_dir, "*ParticipationTrace.pkl"))[0], "rb") as fh:
        p_last = np.asarray(pickle.load(fh)["participation"])[-1]
    return dict(responsive=int((slope > RESPONSIVE_SLOPE).sum()), saturated=int((slope <= RESPONSIVE_SLOPE).sum()),
                N=int(r.shape[0]), slope_median=float(np.median(slope)), frac_pinned=float((np.abs(r) > 0.9).mean()),
                responsive_trace=int((p_last < P_RESPONSIVE).sum()))


def read_source(cell, source, cache, f2, b5c, draws, status):
    """Every run of a cell from the named source, with broken runs removed and provenance checked.

    Args:
        cell: cell path relative to DATA_DIR; source: "local", "f2" or "b5"; cache: open heldout_r2
        cache; f2: the Figure 2 cache; b5c: the B5 cache; draws: noise draws; status: load_status rows.
    Returns:
        (runs, broken): runs are the read dicts, each local run also carrying it_end, job_task (the
        completed task that made it, "" when the run predates the status table) and skipped_share;
        broken lists dicts(cell, name, reason) for runs excluded (non-finite loss, skipped share over
        MAX_SKIP). A completed task of the status table without a run folder on the Mac is printed.
    """
    if source == "f2":
        return read_cell_from_f2_cache(cell, f2), []
    if source == "b5":
        return read_cell_from_b5_cache(cell, b5c), []
    runs = read_cell(cell, cache, draws)
    by_seed = {r["seed"]: r for r in tasks_of(cell, status) if r["state"] == "COMPLETED"}
    losses = {x["name"]: np.asarray(x["loss"], float) for x in runs_with_loss(os.path.join(DATA_DIR, cell))}
    found = set()
    kept, broken = [], []
    for r in runs:
        row = by_seed.get(r["seed"])
        r["job_task"] = row["job_task"] if row else ""
        r["skipped_share"] = (row["skipped"] / r["it_end"]) if row else 0.0
        if row:
            found.add(r["seed"])
        if not np.isfinite(losses.get(r["name"], np.zeros(1))).all():
            broken.append(dict(cell=cell, name=r["name"], reason="non-finite loss"))
        elif r["skipped_share"] > MAX_SKIP:
            broken.append(dict(cell=cell, name=r["name"], reason=f"skipped {100 * r['skipped_share']:.1f}% of gradient updates"))
        else:
            kept.append(r)
            c = cfg_of(os.path.join(DATA_DIR, cell, r["name"]))
            if c is not None and sel(c, "model.activation_args.name") == "tanh":
                r["saturation"] = tanh_saturation(os.path.join(DATA_DIR, cell, r["name"]))
                print(f"      tanh saturation: {r['saturation']}")
    for seed, row in by_seed.items():
        if seed not in found:
            print(f"    MISSING on the Mac: completed task {row['job_task']} (seed {seed}) of {cell}")
    return kept, broken


def config_differences(arm_cell, ctrl_cell):
    """The settings on which an arm's saved config differs from its control's.

    Args:
        arm_cell, ctrl_cell: cell paths relative to DATA_DIR; the first run folder of each is read.
    Returns:
        dict key -> [arm value, control value] over CHECK_FIELDS, empty when the configs agree on all
        of them or when either cell has no run folder on the Mac.
    """
    def first_cfg(cell):
        """The saved config of a cell's first run folder, or None."""
        dirs = sorted(d for d in glob.glob(os.path.join(DATA_DIR, cell, "*")) if os.path.isdir(d))
        return cfg_of(dirs[0]) if dirs else None
    a, c = first_cfg(arm_cell), first_cfg(ctrl_cell)
    if a is None or c is None:
        return {}
    return {k: [sel(a, k), sel(c, k)] for k in CHECK_FIELDS if sel(a, k) != sel(c, k)}


def direction(arm, ctrl):
    """"increase", "decrease" or "no change": every arm count outside the controls' range, same side.

    Args:
        arm, ctrl: lists of run dicts with `active`.
    Returns:
        str, or "not read" when the arm has no runs.
    """
    if not arm:
        return "not read"
    aa, ca = [r["active"] for r in arm], [r["active"] for r in ctrl]
    if min(aa) > max(ca):
        return "increase"
    if max(aa) < min(ca):
        return "decrease"
    return "no change"


def setting_verdict(arm, ctrl, n_expected=N_SEEDS):
    """The settings criterion (entries 0903 and 2259) applied to one cell.

    Args:
        arm, ctrl: run dicts of the cell and of its control; n_expected: runs the cell should have.
    Returns:
        dict: direction, n_read, n_expected, complete (bool), interpreted (False when the mean R^2 is
        below R2_INTERPRET), active (the counts), mean_active_arm, mean_active_ctrl, ctrl_range,
        mean_r2_arm, mean_r2_ctrl, r2_gap and text (one line).
    """
    v = dict(direction=direction(arm, ctrl), n_read=len(arm), n_expected=n_expected,
             complete=len(arm) >= n_expected, active=[r["active"] for r in arm],
             mean_active_ctrl=float(np.mean([r["active"] for r in ctrl])),
             ctrl_range=[min(r["active"] for r in ctrl), max(r["active"] for r in ctrl)],
             mean_r2_ctrl=float(np.mean([r["r2"] for r in ctrl])))
    if not arm:
        v.update(interpreted=False, mean_active_arm=None, mean_r2_arm=None, r2_gap=None,
                 text=f"not read (0 of {n_expected} runs finished)")
        return v
    v["mean_active_arm"] = float(np.mean(v["active"]))
    v["mean_r2_arm"] = float(np.mean([r["r2"] for r in arm]))
    v["r2_gap"] = v["mean_r2_arm"] - v["mean_r2_ctrl"]
    v["interpreted"] = bool(v["mean_r2_arm"] >= R2_INTERPRET)
    text = v["direction"]
    if not v["interpreted"]:
        text = f"count reported, not interpreted (mean R2 {v['mean_r2_arm']:.3f} below {R2_INTERPRET})"
    if not v["complete"]:
        text = f"provisional ({len(arm)} of {n_expected} runs): {text}"
    v["text"] = text
    return v


def optimiser_class(arm, ctrl, n_expected=N_SEEDS):
    """The optimiser classes of entry 2256 applied to one arm.

    Args:
        arm, ctrl: run dicts of the arm and of the control; n_expected: runs the arm should have.
    Returns:
        dict: the rule verdict's fields over the runs that learned (R^2 >= R2_LEARNED), plus
        n_unlearned, n_read, n_expected, complete, cls ("A", "B", "C" or "none learned") and text.
    """
    learned = [r for r in arm if r["r2"] >= R2_LEARNED]
    v = verdict(learned, ctrl) if learned else dict(count_ok=False, r2_ok=False, works=False)
    v.update(n_unlearned=len(arm) - len(learned), n_read=len(arm), n_expected=n_expected,
             complete=len(arm) >= n_expected)
    if not learned:
        v["cls"], v["text"] = "none learned", "no run learned the task" if arm else f"not read (0 of {n_expected} runs finished)"
    elif v["count_ok"] and v["r2_ok"]:
        v["cls"], v["text"] = "A", "A: more active units at no cost in R2"
    elif v["count_ok"]:
        v["cls"], v["text"] = "B", f"B: more active units only at lower R2 (gap {v['r2_gap']:+.3f})"
    else:
        v["cls"], v["text"] = "C", "C: does not increase the number of active units"
    if v["n_unlearned"]:
        v["text"] += f"; {v['n_unlearned']} run(s) below R2 {R2_LEARNED} excluded"
    if not v["complete"]:
        v["text"] = f"provisional ({len(arm)} of {n_expected} runs): " + v["text"]
    return v


def rule_verdict(arm, ctrl, n_expected=N_SEEDS):
    """read_uniform_cells.verdict with the run count beside it (provisional below n_expected)."""
    if not arm:
        return dict(works=False, n_read=0, n_expected=n_expected, complete=False,
                    text=f"not read (0 of {n_expected} runs finished)")
    v = verdict(arm, ctrl)
    v.update(n_read=len(arm), n_expected=n_expected, complete=len(arm) >= n_expected)
    if not v["complete"]:
        v["text"] = f"provisional ({len(arm)} of {n_expected} runs): " + v["text"]
    return v


def pattern_text(arms, other):
    """Whether CDDM reproduces the 3BFF optimiser pattern (entry 2256).

    Args:
        arms: dict label -> optimiser_class dict for CDDM; other: the same for the 3BFF B5 arms.
    Returns:
        str stating the test and its outcome, with every arm's class on both tasks.
    """
    adam = all(arms[k]["cls"] == "A" for k in ("Adam wd 0", "AdamW"))
    sgd = [k for k in arms if k.startswith("SGD") and arms[k].get("count_ok")]
    sgd_ok = all(arms[k]["cls"] == "B" for k in sgd)
    out = ("CDDM reproduces the 3BFF pattern" if adam and sgd_ok else "CDDM does not reproduce the 3BFF pattern")
    why = []
    if not adam:
        why.append("Adam wd 0 / AdamW not both in class A (" + ", ".join(f"{k} {arms[k]['cls']}" for k in ("Adam wd 0", "AdamW")) + ")")
    if not sgd_ok:
        why.append("an SGD arm whose count passes is not in class B")
    if not sgd:
        why.append("no SGD arm's count passes")
    classes = "; ".join(f"{k}: CDDM {arms[k]['cls']}, 3BFF {other[k]['cls'] if k in other else '-'}" for k in arms)
    return out + (" (" + "; ".join(why) + ")" if why else "") + ". Classes: " + classes


def run_line(label, r):
    """One table row for a run (label, seed, iteration read, active, R2, R2 of the final weights, task)."""
    src = (", B5 cache" if r.get("r2_source", "").startswith("b5") else ", F2 cache") if "r2_source" in r else ""
    r2_end = f"{r['r2_end']:.3f}" if "r2_end" in r else "-"
    return (f"| {label} | {r['seed']} | {r['t_read']:,.0f} ({r['probe']:,.0f}) | {r['active']} | "
            f"{r['r2']:.3f} ({r['r2_how']}{src}) | {r2_end} | {r.get('job_task', '-') or '-'} |")


def print_group(g, arms, ctrl, second, verdicts, diffs, other, refs, pending):
    """Print one group's tables: every run, the control runs, each arm's verdict, the other task's ladder.

    Args:
        g: the GROUPS entry; arms: dict label -> run dicts; ctrl, second: the control runs; verdicts:
        dict label -> verdict dict; diffs: dict label -> config differences; other: dict with
        "control" runs and "arms" (label -> runs) of the other task's ladder, or None; refs: dict
        label -> runs of reference cells, or {}; pending: dict label -> list of pending status rows.
    """
    print(f"\n=== {g['name']} ({g['task']}, N = 1,000, read at {FIXED_IT:,} iterations) ===")
    print("| cell | seed | iteration read (probe) | active units | held-out R2 at 75k (how) | R2 of final weights | job task |")
    print("|---|---|---|---|---|---|---|")
    rows = list(arms.items()) + [("control: " + os.path.basename(g["control"][0]), ctrl)]
    if second:
        rows.append(("second control: " + os.path.basename(g["second"][0]), second))
    rows += [("reference: " + k, v) for k, v in refs.items()]
    for label, runs in rows:
        for r in runs:
            print(run_line(label, r))
    kind = g["kind"]
    print("\n| arm | runs read | active (seeds) | mean | control range | mean R2 | R2 gap | verdict |"
          + (" direction on the other task |" if other else ""))
    print("|---|---|---|---|---|---|---|---|" + ("---|" if other else ""))
    for label, runs in arms.items():
        v = verdicts[label]
        act = ", ".join(str(r["active"]) for r in runs) or "-"
        mean = f"{np.mean([r['active'] for r in runs]):.0f}" if runs else "-"
        r2 = f"{np.mean([r['r2'] for r in runs]):.3f}" if runs else "-"
        gap = f"{np.mean([r['r2'] for r in runs]) - np.mean([r['r2'] for r in ctrl]):+.3f}" if runs else "-"
        line = (f"| {label} | {v['n_read']}/{v['n_expected']} | {act} | {mean} | "
                f"{min(r['active'] for r in ctrl)}-{max(r['active'] for r in ctrl)} | {r2} | {gap} | {v['text']} |")
        if other:
            o = other["arms"].get(label)
            od = direction(o, other["control"]) if o is not None else "no cell"
            same = ("same" if kind == "setting" and v.get("direction") == od and od != "not read" and v.get("complete")
                    else "differs" if kind == "setting" and v.get("complete") and od not in ("not read", "no cell") and v.get("interpreted")
                    else "-")
            line += f" {od} ({same}) |" if kind == "setting" else f" {other['verdicts'][label]['cls'] if label in other['verdicts'] else '-'} |"
        print(line)
        if any("saturation" in r for r in runs):
            print("    tanh saturation (responsive = mean(1 - r^2) > 0.5 on a noise-free pass; trace proxy p < 1.6): "
                  + "; ".join(f"seed {r['seed']}: responsive {r['saturation']['responsive']}, saturated "
                              f"{r['saturation']['saturated']} of {r['saturation']['N']}, median slope "
                              f"{r['saturation']['slope_median']:.2f}, |r| > 0.9 on {100 * r['saturation']['frac_pinned']:.0f}% "
                              f"of samples, trace proxy {r['saturation']['responsive_trace']}"
                              for r in runs if "saturation" in r))
        if diffs.get(label):
            print(f"    config differs from the control on: {diffs[label]}")
        if pending.get(label):
            print(f"    pending: " + ", ".join(f"{p['job_task']} ({p['state'].lower()}" + (f", seed {p['seed']}" if p["seed"] else "") + ")"
                                            for p in pending[label]))
    print(f"    control: active {', '.join(str(r['active']) for r in ctrl)} (mean {np.mean([r['active'] for r in ctrl]):.0f}), "
          f"mean R2 {np.mean([r['r2'] for r in ctrl]):.3f}")
    if second:
        print(f"    second control: active {', '.join(str(r['active']) for r in second)} "
              f"(mean {np.mean([r['active'] for r in second]):.0f}), mean R2 {np.mean([r['r2'] for r in second]):.3f}")
    if other:
        oc = other["control"]
        print(f"\n    other task, {g['other']['task']}: control {os.path.basename(g['other']['control'][0])} active "
              f"{', '.join(str(r['active']) for r in oc)}, mean R2 {np.mean([r['r2'] for r in oc]):.3f}")
        for label, runs in other["arms"].items():
            if runs:
                print(f"    {label:22s} active {', '.join(str(r['active']) for r in runs):22s} mean R2 "
                      f"{np.mean([r['r2'] for r in runs]):.3f}  R2 gap {np.mean([r['r2'] for r in runs]) - np.mean([r['r2'] for r in oc]):+.3f}  "
                      + (f"{direction(runs, oc)}" if kind == "setting" else other["verdicts"][label]["text"]))


def main():
    """Read every group, print its tables and write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--draws", type=int, default=H.R2_DRAWS, help="noise draws per held-out R^2")
    ap.add_argument("--out", default=OUT, help="JSON output path")
    ap.add_argument("--status", default=STATUS, help="the per-task status table (TSV)")
    ap.add_argument("--f2-cache", default=F2_CACHE, help="fig_paper_F2_fails cache")
    ap.add_argument("--b5-cache", default=B5_CACHE, help="b5_read cache (the 3BFF control and B5 arms)")
    a = ap.parse_args()
    t0 = time.time()
    cache = H.load_cache()
    with open(a.f2_cache, "rb") as fh:
        f2 = pickle.load(fh)
    with open(a.b5_cache, "rb") as fh:
        b5c = pickle.load(fh)
    status = load_status(a.status)

    def stamp(p):
        """A file's modification time as 'YYYY-MM-DD HH:MM'."""
        return time.strftime("%Y-%m-%d %H:%M", time.localtime(os.path.getmtime(p)))

    def read(cell, src):
        """read_source with this run's caches, draws and status table."""
        return read_source(cell, src, cache, f2, b5c, a.draws, status)
    result = dict(fixed_it=FIXED_IT, draws=a.draws, r2_tol=R2_TOL, r2_learned=R2_LEARNED, r2_interpret=R2_INTERPRET,
                  max_skip=MAX_SKIP, n_seeds=N_SEEDS,
                  criteria=dict(rule=f"works = every run has more active units at {FIXED_IT:,} than the largest control count "
                                     f"and the arm's mean held-out R^2 is at least the controls' mean minus {R2_TOL}",
                                setting="increase/decrease = every run's count outside the controls' range on the same side; "
                                        f"else no change; a cell with mean R^2 below {R2_INTERPRET} is reported, not interpreted",
                                optimiser=f"runs with R^2 below {R2_LEARNED} excluded; A = count and R^2 conditions pass, "
                                          "B = count passes and R^2 fails, C = count fails; CDDM reproduces 3BFF if Adam wd 0 "
                                          "and AdamW are A and every SGD arm whose count passes is B"),
                  status_table=dict(path=os.path.relpath(a.status), mtime=stamp(a.status)),
                  f2_cache=dict(path=os.path.relpath(a.f2_cache), mtime=stamp(a.f2_cache)),
                  b5_cache=dict(path=os.path.relpath(a.b5_cache), mtime=stamp(a.b5_cache)),
                  groups=[], broken=[], pending=[r for r in status if r["state"] != "COMPLETED"])
    for g in GROUPS:
        print(f"\n## {g['name']}", flush=True)
        verdict_of = dict(rule=rule_verdict, setting=setting_verdict, optimiser=optimiser_class)[g["kind"]]
        ctrl, _ = read(*g["control"])
        second = read(*g["second"])[0] if g.get("second") else []
        arms, diffs, verdicts, pending = {}, {}, {}, {}
        for label, cell, src in g["arms"]:
            arms[label], broken = read(cell, src)
            result["broken"] += broken
            H.save_cache(cache)
            diffs[label] = config_differences(cell, g["control"][0])
            verdicts[label] = verdict_of(arms[label], ctrl)
            pending[label] = [r for r in tasks_of(cell, status) if r["state"] != "COMPLETED"]
        refs = {label: read(cell, src)[0] for label, cell, src in g.get("references", [])}
        other = None
        if g.get("other"):
            oc, _ = read(*g["other"]["control"])
            oarms = {label: read(cell, src)[0] for label, cell, src in g["other"]["arms"]}
            overd = {label: verdict_of(runs, oc) for label, runs in oarms.items()}
            other = dict(control=oc, arms=oarms, verdicts=overd)
        print_group(g, arms, ctrl, second, verdicts, diffs, other, refs, pending)
        rec = dict(name=g["name"], task=g["task"], kind=g["kind"],
                   control=dict(cell=g["control"][0], source=g["control"][1], runs=ctrl),
                   second_control=(dict(cell=g["second"][0], source=g["second"][1], runs=second) if g.get("second") else None),
                   arms=[dict(label=label, cell=cell, source=src, runs=arms[label], verdict=verdicts[label],
                              config_differs_from_control=diffs[label], pending=pending[label])
                         for label, cell, src in g["arms"]],
                   references=[dict(label=label, cell=cell, runs=refs[label]) for label, cell, _ in g.get("references", [])])
        if other:
            rec["other"] = dict(task=g["other"]["task"], control=dict(cell=g["other"]["control"][0], runs=other["control"]),
                                arms=[dict(label=label, cell=cell, runs=other["arms"][label], verdict=other["verdicts"][label])
                                      for label, cell, _ in g["other"]["arms"]])
        if g["kind"] == "optimiser":
            rec["pattern"] = pattern_text(verdicts, other["verdicts"])
            print(f"\n    {rec['pattern']}")
        result["groups"].append(rec)
    H.save_cache(cache)
    if result["broken"]:
        print("\nbroken runs excluded:", result["broken"])
    print(f"\npending tasks: {len(result['pending'])} of {len(status)}")
    with open(a.out, "w") as fh:
        json.dump(result, fh, indent=1, default=lambda x: x.item() if hasattr(x, "item") else str(x))
    print(f"\nwrote {os.path.relpath(a.out)} ({time.time() - t0:.0f} s)")


if __name__ == "__main__":
    main()
