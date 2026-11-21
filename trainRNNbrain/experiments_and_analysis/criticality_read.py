#!/usr/bin/env python3
"""Criticality of trained networks along training: Lyapunov exponent, gradient propagation, Jacobian radius.

QUESTION (entry docs/project_trajectory/*_rules-criticality-projection.md). Does training move a network
away from the dynamical regime it started in, the way the signal-propagation literature says
initialisation should be chosen? Measured on the network's actual dynamics, not only on W_rec.

WHAT IS MEASURED, per network and weight state (init rebuilt from the saved config and seed; every kept
snapshot found; each run's final weights). Everything on one noise-free pass over the seeded held-out
batch (heldout_r2.heldout_batch), with trainRNNbrain/trainer/criticality.py:
  lyap_*        largest finite-time Lyapunov exponent per trial (1 / step), median and quartiles
  gain<k>_*     log10 of the read-out sensitivity pulled back k steps, median and quartiles
  oprad_med     median spectral radius of the step Jacobian at 12 operating points
  gate_on       mean fraction of units above threshold after the burn-in
  rho_W, smax_W, rownorm_W   spectral radius, largest singular value, mean incoming row norm of W_rec
  active        active units of the same pass (common.active_mask on std + q90 participation)
The init state is checked against the first snapshot: the cosine between W_rec at init and at the first
snapshot must exceed 20x its value for a wrong seed (seed + 1), or the init row is flagged.

Usage (repository root; CPU):
  PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/criticality_read.py \
      --runs <run_dir> [<run_dir> ...] --snaps <c1_snapshots root> --out <json> [--its 500,1000,...]
A run_dir may be a continuation: its warm-start chain (common.warm_start_chain) is followed and the
snapshots of every run in the chain are read, the chain's first config rebuilding the init.
"""

import argparse
import glob
import json
import os
import re
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
from common import active_mask, participation, warm_start_chain
from trainRNNbrain.trainer import criticality as C
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import import_any, filter_kwargs

LAGS = (10, 50, 100, 200)
BURN_IN = 50
SEED_OFFSET = (65537 ** 3) % 7309      # run_experiment: rnn seed = run seed + (i*14653 + (i+65537)^3) % 7309, i = 0
TANGENT_SEED = 20261120


def build_rnn(cfg, seed=None):
    """The network a saved config describes, freshly initialised.

    Args:
        cfg: a run's saved config (OmegaConf); seed: int seed for the network's generator, or None.
    Returns:
        RNN_torch on the CPU.
    """
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in tcfg:
        del tcfg._target_
    args = filter_kwargs(import_any(cfg.model._target_), OmegaConf.merge(OmegaConf.create(cfg.model), tcfg))
    args.seed = seed
    return hydra.utils.instantiate(args)


def load_params(rnn, params):
    """Copy saved arrays (W_rec possibly float16) into the network's parameters, as float32.

    Args:
        rnn: RNN_torch; params: dict name -> array (heldout_r2._params format).
    Returns:
        None.
    """
    with torch.no_grad():
        for nm, prm in rnn.named_parameters():
            if nm in params:
                prm.copy_(torch.from_numpy(np.asarray(params[nm], dtype=np.float32)))
        if "y_init" in params and getattr(rnn, "y_init", None) is not None:
            rnn.y_init.copy_(torch.from_numpy(np.asarray(params["y_init"], dtype=np.float32)))


def measure(rnn, batch):
    """Every measure of the module docstring on one weight state.

    Args:
        rnn: RNN_torch with the weights to measure; batch: (inputs, targets, mask) from heldout_batch.
    Returns:
        dict of floats.
    """
    bi = batch[0]
    alpha = float(rnn.alpha)
    with torch.no_grad():
        states, _ = rnn(bi, w_noise=False)
        W, Wout = rnn.W_rec.detach(), rnn.W_out.detach()
        gates = C.relu_gates(states)
        g = torch.Generator().manual_seed(TANGENT_SEED)
        lam = C.lyapunov_max(W, gates, alpha, burn_in=BURN_IN, generator=g).numpy()
        gains = C.backward_gain(W, gates, alpha, Wout, lags=LAGS)
        T, B = gates.shape[1], gates.shape[2]
        pts = [(t, b) for t in (75, 150, 225, T - 1) for b in (0, B // 3, (2 * B) // 3)]
        orad = C.operating_radius(W, gates, alpha, pts)
        fr = rnn.activation(states).numpy()
    q = lambda x: [float(np.nanquantile(x, a)) for a in (0.25, 0.5, 0.75)]
    out = dict(lyap_q25=q(lam)[0], lyap_med=q(lam)[1], lyap_q75=q(lam)[2], lyap_mean=float(lam.mean()),
               oprad_med=float(np.median(orad)), gate_on=float(gates[:, BURN_IN:, :].mean()),
               rho_W=C.spectral_radius(W), smax_W=float(torch.linalg.svdvals(W.double())[0]),
               rownorm_W=float(W.norm(dim=1).mean()), active=int(active_mask(participation(fr)).sum()))
    for k, v in gains.items():
        v = v.numpy()
        v = v[np.isfinite(v)]
        out[f"gain{k}_q25"], out[f"gain{k}_med"], out[f"gain{k}_q75"] = q(v) if len(v) else [float("nan")] * 3
    return out


def snapshot_dirs(run_dir, snap_root):
    """Kept snapshots of one run: <snap_root>/<sweep>/<cell>/s<seed>/it<iteration>, sorted by iteration.

    Args:
        run_dir: a run folder whose name carries seed=<n> (a local `parent_*` link to it is resolved);
        snap_root: the c1_snapshots root.
    Returns:
        list of (iteration, folder).
    """
    run_dir = os.path.realpath(run_dir.rstrip("/"))
    cell = os.path.dirname(run_dir)
    seed = re.search(r"seed=(\d+)", os.path.basename(run_dir.rstrip("/"))).group(1)
    pat = os.path.join(snap_root, os.path.basename(os.path.dirname(cell)), os.path.basename(cell),
                       f"s{seed}", "it*")
    return sorted((int(os.path.basename(d)[2:]), d) for d in glob.glob(pat))


def first_cfg(d):
    """The saved config of a run or snapshot folder (OmegaConf)."""
    return OmegaConf.load(sorted(glob.glob(os.path.join(d, "*_config.yaml")))[0])


def read_network(run_dir, snap_root, its=None):
    """All weight states of one network, measured.

    Args:
        run_dir: the last run of the network's warm-start chain; snap_root: c1_snapshots root;
        its: set of iterations to keep among the snapshots, or None for every snapshot found.
    Returns:
        list of row dicts (network, seed, task, iteration, kind, and measure()'s keys).
    """
    chain = warm_start_chain(run_dir)
    cfg0 = first_cfg(chain[0])
    seed0 = int(cfg0.seed)
    rnn = build_rnn(cfg0, seed=seed0 + SEED_OFFSET)
    batch = H.heldout_batch(cfg0)
    W_init = rnn.W_rec.detach().clone()
    rows = []
    base = dict(network=os.path.basename(run_dir.rstrip("/")), seed=seed0, task=str(cfg0.task.taskname))
    rows.append(dict(base, iteration=0, kind="init", **measure(rnn, batch)))
    done = 0
    first_snap = None
    for d in chain:
        cfg = first_cfg(d)
        for it, sd in snapshot_dirs(d, snap_root):
            if its is not None and it not in its:
                continue
            p, _ = H._params(sd)
            load_params(rnn, p)
            first_snap = first_snap if first_snap is not None else rnn.W_rec.detach().clone()
            rows.append(dict(base, iteration=it, kind="snapshot", **measure(rnn, batch)))
        done += int(cfg.trainer.max_iter)
        p, _ = H._params(d)
        load_params(rnn, p)
        rows.append(dict(base, iteration=done, kind="final", **measure(rnn, batch)))
    if first_snap is not None:          # the init-rebuild check of the module docstring
        wrong = build_rnn(cfg0, seed=seed0 + SEED_OFFSET + 1).W_rec.detach()
        cos = lambda a, b: float((a * b).sum() / (a.norm() * b.norm()))
        c_ok, c_bad = cos(W_init, first_snap), abs(cos(wrong, first_snap))
        rows[0].update(init_cos_first_snap=c_ok, init_cos_wrong_seed=c_bad, init_ok=bool(c_ok > 20 * c_bad))
    return rows


def main():
    """Parse arguments, read every network, write the JSON."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", nargs="+", required=True)
    ap.add_argument("--snaps", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--its", default=None, help="comma-separated snapshot iterations to keep")
    a = ap.parse_args()
    its = None if a.its is None else {int(x) for x in a.its.split(",")}
    torch.set_grad_enabled(False)
    rows = []
    for r in a.runs:
        rows += read_network(r, a.snaps, its)
        print(f"{os.path.basename(r)[:50]}: {len(rows)} rows so far", flush=True)
        with open(a.out, "w") as fh:
            json.dump(rows, fh, indent=1)


if __name__ == "__main__":
    main()
