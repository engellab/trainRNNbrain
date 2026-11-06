#!/usr/bin/env python3
"""Prune-and-retrain on the maze: remove the inactive units, train the agent further, score it.

The maze counterpart of the retrain of ablate_silent_units.py (claim C1; the strip of Figure 1d).
It separates "the inactive units compute something" from "removing them perturbs a tuned system":
if an agent trained further without them reaches the success rate of the intact agent trained
further the same way, the task never needed them. One run folder and one arm per call, so the arms
run as separate cluster jobs (or --arm all for the three in one process):
  intact          nothing removed, trained further: the control
  silent_removed  every inactive unit removed; active = participation p >= 0.05 q95(p) over the
                  clean probe (common.active_mask), the rule of rl_maze_ablate.py
  random_removed  as many units removed at random (the draw seeded from the run's label, as the
                  supervised retrain seeds its draw)
A unit is removed by zeroing its outgoing weights, its column of W_rec and of W_out, and the
columns are held at zero through training by a gradient hook that zeroes their gradient: a zero
weight with a zero gradient stays at zero under Adam, whose weight-decay term is proportional to
the weight. Incoming weights are left alone, as in the supervised retrain.
Training further is TrainerPPO with the run's own saved trainer settings (the size-rule learning
rate, the run's weight decay, clip, epochs, minibatches, substeps; a fresh Adam with the warm-up
ramp, as a continuation without the parent's Adam moments) for RETRAIN_UPDATES updates, the same
in every arm. The score is the success rate under the training noise with the mean action over
fixed drops and seeded draws (rl_maze_ablate.score, the score of the ablation curves), taken at
update 0, straight after the removal, and every RETRAIN_EVERY updates; the clean probe's success
and active count (the training's probe, every probe_every updates) are recorded beside it.
Output: one JSON per (run, arm) in --out, <cell>__s<seed>__<arm>.json, with label, run_dir, arm,
N, n_active, n_silent, removed (unit indices), retrain_updates, retrain_every, curve [(update,
success score)], probe [(update, clean success, active units)], clean_full, score_drops, draws.
Run: python rl_maze_retrain.py <run_dir> --arm {intact,silent_removed,random_removed,all}
     [--out DIR] [--updates 200] [--every 25] [--score-drops 256] [--draws 4] [--probe-drops 128]
"""
import argparse
import json
import os
import sys
import zlib

import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from common import active_mask                                               # noqa: E402
from rl_maze_ablate import probe_participation, score                        # noqa: E402
from trainRNNbrain.trainer.TrainerPPO import TrainerPPO, load_run            # noqa: E402
from trainRNNbrain.utils import filter_kwargs                                # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
OUT_DIR = os.path.join(REPO, "data", "rl_maze_retrain")
ARMS = ("intact", "silent_removed", "random_removed")
RETRAIN_UPDATES = 200   # a tenth of the 2,000-update run; the supervised retrain's 3,000
                        # iterations are 5-15% of a run's training to its floor crossing
RETRAIN_EVERY = 25      # score every this many updates


def hold_out(rnn, removed):
    """Zero the outgoing weights of `removed` and keep them at zero while the RNN trains.

    Args:
        rnn: RNN_torch with W_rec (N, N) and W_out (n_out, N) parameters; removed: int array of
            unit indices (empty for the intact control).
    Returns:
        list of the gradient-hook handles (empty when nothing is removed); remove() them to stop.
    """
    idx = torch.as_tensor(np.ascontiguousarray(removed, dtype=np.int64), device=rnn.W_rec.device)
    if idx.numel() == 0:
        return []
    handles = []
    with torch.no_grad():
        for prm in (rnn.W_rec, rnn.W_out):
            prm[:, idx] = 0.0
            keep = torch.ones(prm.shape[1], device=prm.device, dtype=prm.dtype)
            keep[idx] = 0.0
            handles.append(prm.register_hook(lambda g, m=keep: g * m[None, :]))
    return handles


def build_trainer(cfg, rnn, task, updates, every):
    """A TrainerPPO with the run's saved settings, a fresh Adam, and the retrain's budget.

    Args:
        cfg: the run's saved config (either layout: `trainer` or `ppo` block); rnn: RNN_torch;
        task: TaskMaze; updates: int, PPO updates to train; every: int, checkpoint_every, at
        which run_training calls the scoring callback.
    Returns:
        TrainerPPO, with the learning rate of the size rule lr (100 / N)^lr_scale_exp and the
        run's weight decay, iter_offset 0 and adam_restored False (the warm-up ramp runs).
    """
    hp = cfg.trainer if "trainer" in cfg else cfg.ppo
    lr = float(hp.lr) * (100.0 / float(cfg.model.N)) ** float(hp.lr_scale_exp)
    opt = torch.optim.Adam(rnn.parameters(), lr=lr, weight_decay=float(hp.weight_decay))
    args = OmegaConf.to_container(filter_kwargs(TrainerPPO, hp), resolve=True)
    args.pop("_target_", None)
    args.pop("optimizer", None)    # trainer.optimizer names the run's optimiser; the retrain's is the fresh Adam above
    args["critic"] = bool(hp.get("critic", rnn.n_outputs == 3))
    args.update(n_updates=int(updates), checkpoint_every=int(every), iter_offset=0, adam_restored=False)
    return TrainerPPO(rnn, task, opt, **args)


def retrain(run_dir, arm, out_dir, updates, every, score_drops, draws, probe_drops):
    """Remove the arm's units from one run's agent, train it further and write the arm's JSON.

    Args:
        run_dir: str, the run folder; arm: one of ARMS; out_dir: str; updates, every,
        score_drops, draws, probe_drops: ints (the module docstring).
    Returns:
        the result dict that was written.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, task, played = load_run(run_dir, device)
    rnn, T = played.RNN, played.T
    p, clean_full = probe_participation(task, played, T, probe_drops, played.probe_seed)
    active = active_mask(p)
    N, silent = len(p), np.flatnonzero(~active_mask(p))
    label = f"N={N}_act={cfg.model.activation_args['name']}_s{cfg.seed}"
    rnd = np.random.default_rng(zlib.crc32(label.encode())).permutation(N)[:len(silent)]
    removed = {"intact": np.zeros(0, dtype=np.int64), "silent_removed": silent, "random_removed": rnd}[arm]
    trainer = build_trainer(cfg, rnn, task, updates, every)
    hold_out(rnn, removed)
    curve = [(0, score(task, trainer, T, score_drops, draws))]
    print(f"{label} {arm}: {len(removed)} of {N} units removed ({int(active.sum())} active); "
          f"score {curve[0][1]:.3f} after removal (intact {clean_full:.3f} clean)", flush=True)
    trainer.checkpoint_fn = lambda it, params: curve.append((it, score(task, trainer, T, score_drops, draws)))
    trainer.run_training()
    curve.append((updates, score(task, trainer, T, score_drops, draws)))
    if len(removed):      # the hook's check: the removed units' outgoing weights are still exactly zero
        idx = torch.as_tensor(removed, device=rnn.W_rec.device)
        with torch.no_grad():
            left = max(float(rnn.W_rec[:, idx].abs().max()), float(rnn.W_out[:, idx].abs().max()))
        if left != 0.0:
            raise RuntimeError(f"{label} {arm}: a removed unit's outgoing weight is {left} after training")
    res = dict(label=label, run_dir=os.path.relpath(run_dir, REPO), arm=arm, N=N,
               n_active=int(active.sum()), n_silent=int(len(silent)), removed=removed.tolist(),
               retrain_updates=updates, retrain_every=every, curve=curve,
               probe=list(zip(trainer.probes["update"], trainer.probes["probe_success"], trainer.probes["active"])),
               clean_full=clean_full, score_drops=score_drops, draws=draws, T=T)
    os.makedirs(out_dir, exist_ok=True)
    cell = os.path.basename(os.path.dirname(run_dir.rstrip("/")))
    path = os.path.join(out_dir, f"{cell}__s{cfg.seed}__{arm}.json")
    with open(path, "w") as fh:
        json.dump(res, fh)
    print(f"{label} {arm}: score {curve[0][1]:.3f} after removal -> {curve[-1][1]:.3f} after {updates} "
          f"updates; clean probe {res['probe'][-1][1]:.3f}, {res['probe'][-1][2]} active\n-> {path}", flush=True)
    return res


def main():
    """Parse the arguments and run one arm, or all three in turn."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("run_dir")
    ap.add_argument("--arm", choices=ARMS + ("all",), default="all")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--updates", type=int, default=RETRAIN_UPDATES)
    ap.add_argument("--every", type=int, default=RETRAIN_EVERY)
    ap.add_argument("--score-drops", type=int, default=256)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--probe-drops", type=int, default=128)
    a = ap.parse_args()
    for arm in (ARMS if a.arm == "all" else (a.arm,)):
        retrain(a.run_dir, arm, a.out, a.updates, a.every, a.score_drops, a.draws, a.probe_drops)


if __name__ == "__main__":
    main()
