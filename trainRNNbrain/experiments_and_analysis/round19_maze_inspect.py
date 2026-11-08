#!/usr/bin/env python3
"""Why do tanh and shifted-sigmoid networks fail to learn the Maze? Inspection behind the round 19 pilot.

QUESTION (round 19, ptolmachev/dead-relu-paper#201; rules in
docs/project_trajectory/*_round19-train-all-rules.md). At the standard settings no tanh and no shifted-sigmoid
network of the Maze suite learned in 2,000 updates. Before choosing the pilot's conditions, this script
reads two things from the existing N = 1,000 runs:
  1. Update size, from each run's Progress.json: the approximate KL divergence between the policy before
     and after each PPO update (approx_kl): its value at update 1, the number of updates above 0.05 and
     0.1, and the largest value with its update. Every activation of the suite is listed, ReLU included,
     so the numbers have a reference.
  2. Saturation, for tanh and the shifted sigmoid: the network is rebuilt (TrainerPPO.load_run) and the
     clean probe is played (the training probe's fixed drops, no noise, the mean action) on 64 episodes.
     A unit-step is FLAT when the activation's slope there is under 10% of its maximum slope: |r| > 0.949
     for tanh (1 - r^2 < 0.1); r > 0.9743 or r < 0.0257 for sigmoid(7.5 (x - 0.3)) (r (1 - r) < 0.025).
     A unit is PINNED when at least 90% of its steps are flat. Reported at the end weights and for the
     untrained network of the same seed (rebuilt from the saved config without loading the trained
     parameters, the network at update 0), and for that untrained network at the pilot's smaller initial
     spectral radius (tanh 0.85, shifted sigmoid 0.45). Also: the spectral radius of W_rec, the median
     row norm of W_inp, the mean |rate| and the participation-rule active count on the same probe.

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/round19_maze_inspect.py [--root DIR] [--out FILE]
Output: data/round19/maze_inspect.json, {"kl": [...], "saturation": [...]}, and a printed table.
"""
import argparse
import glob
import json
import os
import sys

import hydra
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from trainRNNbrain.experiments_and_analysis.common import participation, active_mask   # noqa: E402
from trainRNNbrain.rnns.RNN_torch import RNN_torch                                     # noqa: E402
from trainRNNbrain.trainer.TrainerPPO import load_run                                  # noqa: E402
from trainRNNbrain.utils import filter_kwargs                                          # noqa: E402

ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite")
OUT = os.path.join(REPO, "data", "round19", "maze_inspect.json")
PILOT_RADIUS = {"tanh": 0.85, "sigmoid": 0.45}    # radius x max slope = 0.85 (rl_maze_r19_pilot_della.slurm)
PROBE_EPISODES = 64


def kl_summary(folder):
    """Update-size statistics of one run from its Progress.json.

    Args:
        folder: str, a stage-1 Maze run folder (one Progress file).
    Returns:
        dict: kl_first (float, approx_kl of update 1), n_kl_above_0.05 and n_kl_above_0.1 (int, updates),
        kl_max (float) and kl_max_update (int).
    """
    progress = json.load(open(glob.glob(os.path.join(folder, "*Progress.json"))[0]))
    kl = np.array([rec["approx_kl"] for rec in progress], dtype=float)
    upd = np.array([rec["update"] for rec in progress], dtype=int)
    return {"kl_first": float(kl[0]), "n_kl_above_0.05": int((kl > 0.05).sum()),
            "n_kl_above_0.1": int((kl > 0.1).sum()), "kl_max": float(kl.max()),
            "kl_max_update": int(upd[kl.argmax()])}


def flat_masks(rates, act):
    """Where the activation's slope is under 10% of its maximum.

    Args:
        rates: (N, M) float array, firing rates of N units over M steps; act: str, "tanh" or "sigmoid".
    Returns:
        (hi, lo): two (N, M) bool arrays, the upper and the lower flat part.
    """
    if act == "tanh":
        return rates > 0.949, rates < -0.949
    return rates > 0.9743, rates < 0.0257


def probe_rates(trainer):
    """Play the clean probe (fixed drops, no noise, mean action) and return the rates over alive steps.

    Args:
        trainer: TrainerPPO holding the network to play.
    Returns:
        (rates, success): (N, M) float array over every RNN step before the cheese; float, success rate.
    """
    g = torch.Generator(device=trainer.Task.device)
    g.manual_seed(int(trainer.probe_seed))
    T, k = trainer.T, trainer.substeps
    out = trainer.rollout(PROBE_EPISODES, T, clean=True, generator=g)
    with torch.no_grad():
        states = trainer.sequence(out["u"], out["noise"], alive=out["alive"])[0][:, :T * k]
    rates = trainer.RNN.get_firing_rates(states).cpu().numpy()
    alive = out["alive"][:-1].bool().repeat_interleave(k, dim=0).cpu().numpy()
    return rates[:, alive], float(out["success"].float().mean())


def saturation(trainer, act):
    """Saturation statistics of the network the trainer holds, on the clean probe.

    Args:
        trainer: TrainerPPO; act: str, "tanh" or "sigmoid".
    Returns:
        dict: success, frac_flat_hi, frac_flat_lo (fractions of unit-steps), pinned_hi, pinned_lo (units
        flat on >= 90% of steps), mean_abs_rate, active (participation rule), rho_wrec (spectral radius of
        W_rec), winp_row_norm_median.
    """
    rates, success = probe_rates(trainer)
    hi, lo = flat_masks(rates, act)
    W = trainer.RNN.W_rec.detach().cpu().numpy()
    win = np.linalg.norm(trainer.RNN.W_inp.detach().cpu().numpy(), axis=1)
    return {"success": success, "frac_flat_hi": float(hi.mean()), "frac_flat_lo": float(lo.mean()),
            "pinned_hi": int((hi.mean(1) >= 0.9).sum()), "pinned_lo": int((lo.mean(1) >= 0.9).sum()),
            "mean_abs_rate": float(np.abs(rates).mean()), "active": int(active_mask(participation(rates)).sum()),
            "rho_wrec": float(np.abs(np.linalg.eigvals(W)).max()), "winp_row_norm_median": float(np.median(win))}


def untrained(cfg, radius=None):
    """The untrained network of a run's seed, rebuilt from its saved config.

    Args:
        cfg: the run's saved DictConfig; radius: float or None, model.spectral_rad to use instead of the
            saved one.
    Returns:
        RNN_torch with the initial weights of that seed.
    """
    args = filter_kwargs(RNN_torch, cfg.model)
    args.seed = int(cfg.seed)
    if radius is not None:
        args.spectral_rad = radius
    return hydra.utils.instantiate(args)


def main():
    """Compute the KL table for every N = 1,000 suite run and the saturation table for tanh and sigmoid."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--root", default=ROOT, help="the Maze_suite folder")
    ap.add_argument("--out", default=OUT, help="output JSON")
    a = ap.parse_args()
    torch.set_num_threads(8)
    kl_rows, sat_rows = [], []
    for folder in sorted(glob.glob(os.path.join(a.root, "N=1000_act=*", "s*"))):
        cell = os.path.basename(os.path.dirname(folder))
        if "_cont" in cell:
            continue
        row = {"cell": cell, "seed": os.path.basename(folder), **kl_summary(folder)}
        kl_rows.append(row)
        print(f"{cell:28s} {row['seed']}  KL at update 1 {row['kl_first']:.3f}  updates with KL > 0.05: "
              f"{row['n_kl_above_0.05']:3d}, > 0.1: {row['n_kl_above_0.1']:3d}  max {row['kl_max']:.3g} at "
              f"update {row['kl_max_update']}", flush=True)
        if "tanh" not in cell and "sigmoid" not in cell:
            continue
        cfg, task, trainer = load_run(folder)
        act = cfg.model.activation_args["name"]
        for label, radius in (("end", None), ("init", None), ("init_pilot_radius", PILOT_RADIUS[act])):
            if label != "end":
                trainer.RNN = untrained(cfg, radius)
            s = {"cell": cell, "seed": os.path.basename(folder), "weights": label,
                 "radius_init": radius, **saturation(trainer, act)}
            sat_rows.append(s)
            print(f"   {label:18s} success {s['success']:.2f}  pinned {s['pinned_hi']}+{s['pinned_lo']} "
                  f"(upper+lower)  flat steps {s['frac_flat_hi']:.2f}+{s['frac_flat_lo']:.2f}  mean|r| "
                  f"{s['mean_abs_rate']:.2f}  active {s['active']}  rho(W_rec) {s['rho_wrec']:.2f}  "
                  f"|W_inp row| {s['winp_row_norm_median']:.3f}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"probe_episodes": PROBE_EPISODES, "kl": kl_rows, "saturation": sat_rows}, fh, indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
