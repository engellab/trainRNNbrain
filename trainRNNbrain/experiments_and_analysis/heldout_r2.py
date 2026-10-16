#!/usr/bin/env python3
"""THE performance measure for this project: held-out R^2.

ONE NOISE CONDITION, DIFFERENT INPUTS. Every network is scored on data it never trained on, running at the
sigma_rec and sigma_inp it trained with and sigma_w = 0, averaged over DRAWS noise draws. So the arms
differ in their inputs and in nothing else: no arm is measured with its wiring fluctuating while
another is not, and no arm is handed the noise-free dynamics it never trained for.

WHY THE NOISE-FREE SCORE IS GONE. It was reported for years as "clean R^2" and it is misleading in
both directions. It flatters nothing uniformly: a network trained under noise can come to use it, so
removing the noise costs an unconverged network 6 points and a duplication network 13, which reads as
a difference between arms when it is a difference in how far each has converged. The ranking of arms
inverts between the two scores - frm+rws is last noise-free and first under noise - so the choice of
evaluation, not the remedy, decided the result. This module exists so that choice is made once.

HELD-OUT MEANS WHAT THE TASK CAN WITHHOLD.
  CDDM          coherences INTERLEAVED with the training grid - the midpoint of each adjacent pair,
                so every condition lies inside the trained range and none was seen. This is the batch
                `run_experiment` already builds for its validation probe.
  NBitFlipFlop  a fresh draw: the input stream is sampled per batch, so new draws are unseen by
                construction.
  DMTS          a fresh draw, for the same reason.

THE SCORE IS REPEATABLE (since 2026-10-14, Pavel: "it's ok to apply the fix project-wide"). Before,
two scores of the same weights differed: a sampled task (flip-flop, DMTS) has `task.seed: None`, so its
generator was unseeded and every call drew a different held-out batch (np.random.seed does not reach
a np.random.default_rng instance), and the network's noise generator was unseeded too. One flip-flop
ablation score spread by 0.03 (SD) over four runs on identical weights. Now the task's generator is
seeded with HELDOUT_SEED, so every scorer (this module, ablate_silent_units, the figure builders) sees
the same held-out batch for a given task, and noise draw d is seeded with NOISE_SEED + d, as the
ablation code always did. CDDM's held-out batch is a fixed grid and was already deterministic.

Results are cached in data/heldout_r2_cache.json, keyed by the parameter file's path, size and mtime,
so rebuilding a figure rescores nothing and retraining a cell invalidates only that cell. The key
ends with SCHEME, so a value scored before the seeding is never reused.

AT THE FLOOR. `floor_r2` gives held-out R^2 at a run's floor crossing (on a kept snapshot, else the
common.r2_at_floor_estimate from the final weights' held-out R^2), and `read_rung` reads a whole cell
at the floor; both take the read-out from common.read_run.

Usage:  python heldout_r2.py "data/trained_RNNs/NBitFlipFlop_std_dropfix/*/*"   # populate the cache
"""
import glob
import json
import os
import sys

import numpy as np
import torch
import hydra
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.utils import import_any, filter_kwargs, unjsonify
from common import r2_at_floor_estimate, read_run, runs_with_loss

DRAWS = 9
# noise draws per held-out R^2 in the figures. One CDDM draw has an SD of ~0.011, so the project's 9
# draws leave ~0.004 of noise and two scorers differed by 0.01-0.013 on identical weights; 36 halves it.
# Same measure, more draws. (Moved here from fig_knobs_at_floor on 2026-10-15.)
R2_DRAWS = 36
BATCH = 256                      # trials per draw; the estimate's noise is set by trials x draws
HELDOUT_SEED = 20261012          # seeds the task's generator, so the held-out batch is fixed per task
NOISE_SEED = 1000                # noise draw d is seeded with NOISE_SEED + d (as ablate_silent_units)
SCHEME = "seeded-v1"             # cache-key tag; values from the unseeded scorer carry no tag
CACHE = os.path.join(REPO, "data", "heldout_r2_cache.json")


def _params(net_dir):
    """The saved parameters of one run, from either save format.

    Args:
        net_dir: a trained-network folder.
    Returns:
        (dict of name -> array, key string identifying the file's current contents), or (None, None).
    """
    hits = (sorted(glob.glob(os.path.join(net_dir, "*LastParams*.np[zy]")))
            or sorted(glob.glob(os.path.join(net_dir, "*LastParams*.json"))))
    if not hits:
        return None, None
    f = hits[0]
    st = os.stat(f)
    key = f"{os.path.relpath(f, REPO)}|{st.st_size}|{int(st.st_mtime)}"
    if f.endswith(".json"):
        with open(f) as fh:
            return {k: np.asarray(v) for k, v in unjsonify(json.load(fh)).items()}, key
    z = np.load(f, allow_pickle=True)
    return {k: z[k] for k in z.files}, key


def heldout_batch(cfg):
    """Inputs, targets and mask the network never trained on, the same on every call.

    The task's generator is seeded with HELDOUT_SEED, so a sampled task (flip-flop, DMTS) returns
    the same trials every time; training drew from an unseeded generator, so they are still unseen.

    Args:
        cfg: the run's saved config.
    Returns:
        (inputs tensor, targets tensor, mask) ready for the model's forward pass.
    """
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = HELDOUT_SEED                     # a sampled task draws the same held-out trials every call
    if "coherences" in cfg.task:                 # CDDM: the midpoints of the trained grid
        cohs = sorted(float(c) for c in cfg.task.coherences)
        tcfg.coherences = [0.5 * (cohs[i] + cohs[i + 1]) for i in range(len(cohs) - 1)]
    task = hydra.utils.instantiate(tcfg)
    if hasattr(task, "batch_size") and "coherences" not in cfg.task:
        task.batch_size = BATCH                  # a grid task sizes its own batch; a sampled one does not
    bi, bt, _ = task.get_batch()
    return (torch.tensor(np.asarray(bi), dtype=torch.float32),
            torch.tensor(np.asarray(bt), dtype=torch.float32),
            get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt))


def score_net(net_dir, draws=DRAWS):
    """Held-out R^2 in the common noise condition for one saved network.

    Args:
        net_dir: a trained-network folder holding *LastParams* and *_config.yaml;
        draws: noise realisations to average over.
    Returns:
        float R^2, or nan if the run has no saved parameters.
    """
    cfgs = sorted(glob.glob(os.path.join(net_dir, "*_config.yaml")))
    params, _ = _params(net_dir)
    if not cfgs or params is None:
        return float("nan")
    cfg = OmegaConf.load(cfgs[0])
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in tcfg:
        del tcfg._target_
    rnn = hydra.utils.instantiate(
        filter_kwargs(import_any(cfg.model._target_),
                      OmegaConf.merge(OmegaConf.create(cfg.model), tcfg)))
    with torch.no_grad():
        for nm, prm in rnn.named_parameters():
            if nm in params:
                prm.copy_(torch.from_numpy(np.asarray(params[nm], dtype=np.float32)))
        if "y_init" in params and getattr(rnn, "y_init", None) is not None:
            rnn.y_init.copy_(torch.from_numpy(np.asarray(params["y_init"], dtype=np.float32)))
        bi, bt, mask = heldout_batch(cfg)
        # sigma_w = 0 so no arm is scored with its wiring fluctuating; sigma_rec and sigma_inp stay
        # at the values in this run's own config, which every arm of a comparison shares.
        if hasattr(rnn, "sigma_w"):
            rnn.sigma_w = 0.0
        vals = []
        for d in range(draws):
            rnn.random_generator.manual_seed(NOISE_SEED + d)
            vals.append(float(Trainer.r2_score(rnn(bi, w_noise=True)[1], bt, mask)))
    return float(np.mean(vals))


def heldout_r2(net_dir, draws=DRAWS, cache=None):
    """score_net with the on-disk cache in front of it.

    Args:
        net_dir: a trained-network folder; draws: noise realisations; cache: an open cache dict, or
        None to load and save it around this one call.
    Returns:
        float R^2.
    """
    own = cache is None
    c = load_cache() if own else cache
    _, key = _params(net_dir)
    if key is None:
        return float("nan")
    key = f"{key}|{draws}|{SCHEME}"
    if key not in c:
        c[key] = score_net(net_dir, draws)
        if own:
            save_cache(c)
    return c[key]


def load_cache():
    """The cache dict, empty if none exists yet."""
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            return json.load(fh)
    return {}


def save_cache(c):
    """Write the cache dict."""
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(c, fh, indent=0, sort_keys=True)


def floor_r2(rd, cache, draws=R2_DRAWS):
    """Held-out R^2 at a run's floor crossing: measured on a kept snapshot, else estimated.

    Where the run kept weight snapshots (`trainer.checkpoint_keep`), R^2 is MEASURED on the first
    snapshot at or after the crossing (common.snapshot_at). Otherwise it is estimated from the end
    of the run, common.r2_at_floor_estimate, with R2_end the held-out R^2 of the final weights.

    Args:
        rd: common.read_run's dict for the run, with a read-out (rd["reason"] is None);
        cache: an open heldout_r2 cache; draws: noise draws per score.
    Returns:
        (R^2, measured): held-out R^2 of the snapshot and True, or the estimate and False.
    """
    if rd["snapshot"]:
        return heldout_r2(rd["snapshot"], draws=draws, cache=cache), True
    return r2_at_floor_estimate(rd["L_star"], rd["L_end"],
                                heldout_r2(rd["run_dir"], draws=draws, cache=cache)), False


def read_rung(path, end_at=None, cache=None):
    """Active-unit counts for one cell, at each run's own noisy-loss floor and at the end of its budget.

    Moved here from fig_knobs_at_floor on 2026-10-15, unchanged, so that every script reading a cell
    at its floor (the knob ladders, fig_paper_F2_fails, a4_cddm_std_penalties) gets it from one place.

    Args:
        path: the cell directory holding one folder per seed.
        end_at: iteration to take the "end" count at, or None for the last probe. Set when a ladder's
            control was trained longer than its arms, so both ends are read at the same budget.
        cache: an open heldout_r2 cache to also take R^2 at the floor (floor_r2, R2_DRAWS draws),
            or None to skip it.
    Returns:
        dict with keys `floor` (counts where the noisy loss first reaches 1.05x its fitted floor),
        `end` (counts at the last participation probe), `cross` (crossing iterations), `late`
        (end-of-run loss over the floor, for the runs read), `r2` (held-out R^2 at the floor, one
        per `floor` entry; empty without a cache), `measured` (True where that R^2 was scored on a
        snapshot rather than estimated), `n` (runs found) and `dropped` (one reason string per run
        with no usable crossing).
    """
    at_floor, at_end, cross, late, r2, measured, dropped = [], [], [], [], [], [], []
    runs = runs_with_loss(path)
    for r in runs:
        it_a = np.asarray(r["it_act"], float)
        at_end.append(r["active"][-1] if end_at is None
                      else r["active"][int(np.argmin(np.abs(it_a - end_at)))])
        rd = read_run(r["run_dir"], run=r)
        if rd["reason"]:
            dropped.append(rd["reason"])
            continue
        # a top-up whose trace was not joined to its parent's starts late; reading its first probe
        # as "the floor" would be silently wrong
        if not rd["covered"]:
            dropped.append("participation trace does not cover the crossing")
            continue
        at_floor.append(rd["active"])
        cross.append(rd["t_star"])
        late.append(rd["late_rise"])
        if cache is not None:
            v, m = floor_r2(rd, cache)
            r2.append(v)
            measured.append(m)
    return dict(floor=np.array(at_floor, float), end=np.array(at_end, float),
                cross=np.array(cross, float), late=np.array(late, float),
                r2=np.array(r2, float), measured=np.array(measured, bool), n=len(runs),
                dropped=dropped)


def main(patterns):
    """Score every run folder matched by these globs and report per-cell means."""
    c = load_cache()
    for pat in patterns:
        dirs = [d for d in sorted(glob.glob(pat)) if os.path.isdir(d)]
        vals = {}
        for d in dirs:
            v = heldout_r2(d, cache=c)
            if np.isfinite(v):
                vals.setdefault(os.path.dirname(d), []).append(v)
            save_cache(c)
        for cell, vs in vals.items():
            print(f"  {os.path.relpath(cell, os.path.join(REPO, 'data', 'trained_RNNs')):70s} "
                  f"n={len(vs)} r2 {np.mean(vs):.4f} ±{np.std(vs, ddof=1) if len(vs) > 1 else 0:.4f}")
    save_cache(c)


if __name__ == "__main__":
    main(sys.argv[1:] or ["data/trained_RNNs/NBitFlipFlop_std_dropfix/*/*"])
