#!/usr/bin/env python3
"""How fast does Adam's coupled weight decay shrink a weight, and how far do weights move? (mechanism tests, part C)

Question. A reviewer estimated, from saved Adam states, that the default decay of 1e-6 alone shrinks a weight onto an
inactive unit e-fold in 350 to 500 iterations. This script recomputes the estimate from the Adam states (independent code)
and then MEASURES the time constant from weights saved during training, where the loss gradient and the changing gradient
size act too.

Mode `adam` (one saved network with an AdamState file). With v, m Adam's bias-corrected moments of one weight, lr and eps the
optimizer's values and lam = 1e-6:
  gradient size  sqrt(v);   step  |m| / (sqrt(v) + eps), in units of lr;   predicted decay time  (sqrt(v) + eps) / (lr lam).
Medians over W_rec by class of the postsynaptic and the presynaptic unit: A active, I inactive (common.active_mask on the last
stored participation vector). Also the median incoming row norm of W_rec, inactive / active.

Mode `weights` (one network with kept snapshots). For base iterations t and lags D:
  slope b(D) of w(t+D) on w(t) through the origin, over the weights of one class, classes fixed at t and t+D;
  measured decay time  -D / ln b  (a pure exponential decay gives b = exp(-D/tau); added uncorrelated change leaves b unchanged).
  norm of the change of a unit's incoming weights, ||(W_rec, W_inp) row (t+D) - row (t)||, median over the units of a class
  (active; low = inactive but participation >= 1e-6; silent = participation < 1e-6, at t), and, at the first base iteration,
  the norm of the task-loss gradient on the same rows (rms over GRAD_BATCHES training batches under the training noise).
The forward pass and the noise are mech_removal's, not RNN_torch's.

Usage:  python mech_optimizer.py adam OUT_JSON RUN_DIR [RUN_DIR ...]
        python mech_optimizer.py weights OUT_JSON RUN_DIR SNAP_DIR --base 25000 50000 --lags 500 1000 2000 5000 5500
"""
import argparse
import glob
import json
import os
import pickle
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import active_mask  # noqa: E402
from mech_removal import forward, load, noise_draw  # noqa: E402
from trainRNNbrain.training.training_utils import get_training_mask, prepare_task_arguments  # noqa: E402

LAM = 1e-6              # the default weight decay, also used for the what-if of networks trained without it
SILENT = 1e-6           # participation below which a unit is silent (the trace's silent_1em6)
GRAD_BATCHES = 4
GRAD_SEED = 47000       # noise seed of gradient batch k is GRAD_SEED + k


def only(folder, pattern):
    """The single file of `folder` matching `pattern`; raises if there is not exactly one."""
    f = glob.glob(os.path.join(glob.escape(folder), pattern))
    if len(f) != 1:
        raise FileNotFoundError(f"{len(f)} files match {pattern} in {folder}")
    return f[0]


def trace(run_dir):
    """Stored participation vectors of a run.

    Args:
        run_dir: a run folder.
    Returns:
        dict iteration (int) -> (N,) float32 participation vector.
    """
    with open(only(run_dir, "*ParticipationTrace.pkl"), "rb") as fh:
        t = pickle.load(fh)
    return {int(i): np.asarray(p) for i, p in zip(t["participation_iters"], t["participation"])}


def block_medians(x, act):
    """Median of a (post, pre) matrix in the four blocks AA, IA, AI, II (post first).

    Args:
        x: (N, N) array; act: (N,) bool, active units.
    Returns:
        dict block -> float.
    """
    rows = {"A": act, "I": ~act}
    return {a + b: float(np.median(x[np.ix_(rows[a], rows[b])])) for a in "AI" for b in "AI"}


def do_adam(runs):
    """Mode `adam`, every run.

    Args:
        runs: list of run folders.
    Returns:
        list of dicts, one per run.
    """
    out = []
    for d in runs:
        cfg = OmegaConf.load(only(d, "*_config.yaml"))
        st = torch.load(only(d, "*AdamState*.pt"), map_location="cpu", weights_only=True)
        g = st["param_groups"][0]
        N = int(cfg.model.N)
        s = [x for x in st["state"].values() if tuple(x["exp_avg"].shape) == (N, N)][0]
        n = float(s["step"])
        m = s["exp_avg"].double().numpy() / (1 - g["betas"][0] ** n)
        root_v = np.sqrt(s["exp_avg_sq"].double().numpy() / (1 - g["betas"][1] ** n))
        tr = trace(d)
        p = tr[max(tr)]
        act = active_mask(p)
        (Wr, _, _, _), _ = load(d)
        row = np.linalg.norm(Wr.numpy().astype(np.float64), axis=1)
        grad, step = block_medians(root_v, act), block_medians(np.abs(m) / (root_v + g["eps"]), act)
        tau = block_medians((root_v + g["eps"]) / (g["lr"] * LAM), act)
        rec = dict(run_dir=d, task=str(cfg.task.taskname), N=N, optimizer=str(cfg.trainer.get("optimizer", "adam")),
                   weight_decay=float(g["weight_decay"]), sigma_rec=float(cfg.model.sigma_rec), lr=float(g["lr"]),
                   step=int(n), trace_it=max(tr), active=int(act.sum()), silent=int((p < SILENT).sum()),
                   grad=grad, adam_step=step, tau=tau, grad_ratio=grad["AA"] / max(grad["IA"], 1e-300),
                   step_ratio=step["AA"] / max(step["IA"], 1e-300),
                   row_ratio=float(np.median(row[~act]) / np.median(row[act])), row_active=float(np.median(row[act])),
                   row_inactive=float(np.median(row[~act])))
        out.append(rec)
        print(f"{rec['task']} N={N} {rec['optimizer']} wd={rec['weight_decay']:g} sigma={rec['sigma_rec']:g} "
              f"active {rec['active']} | gradient AA/IA {rec['grad_ratio']:.1f}, step AA/IA {rec['step_ratio']:.2f} "
              f"(IA/AA {1 / rec['step_ratio']:.2f}) | decay time AA {tau['AA']:.0f} IA {tau['IA']:.0f} AI {tau['AI']:.0f} "
              f"II {tau['II']:.0f} | row ratio {rec['row_ratio']:.3f}", flush=True)
    return out


def row_gradients(snap_dir, run_dir):
    """Norm of the task-loss gradient on each unit's incoming weights, rms over GRAD_BATCHES noisy training batches.

    Args:
        snap_dir: folder with the weights; run_dir: the run, for its saved config.
    Returns:
        (N,) float64 array.
    """
    W, _ = load(snap_dir)
    cfg = OmegaConf.load(only(run_dir, "*_config.yaml"))
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tcfg.seed = 12345 if tcfg.get("seed", None) is None else int(tcfg.seed)
    task = hydra.utils.instantiate(tcfg)
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    alpha = float(cfg.model.dt) / float(cfg.model.tau)
    sig = [float(cfg.model[k]) for k in ("sigma_rec", "sigma_inp", "sigma_out")]
    acc = 0.0
    for k in range(GRAD_BATCHES):
        u, y, _ = task.get_batch()
        u, y = (torch.from_numpy(np.asarray(a, dtype=np.float32)) for a in (u, y))
        Wr, Wi = W[0].clone().requires_grad_(True), W[1].clone().requires_grad_(True)
        z = forward((Wr, Wi, W[2], W[3]), u, alpha, noise_draw(GRAD_SEED + k, Wr.shape[0], u.shape[0], u.shape[1], u.shape[2], alpha, sig))
        (z[:, mask] - y[:, mask]).pow(2).mean().backward()
        acc = acc + (Wr.grad.double().pow(2).sum(1) + Wi.grad.double().pow(2).sum(1)).numpy()
    return np.sqrt(acc / GRAD_BATCHES)


def do_weights(run_dir, snap_dir, bases, lags):
    """Mode `weights`, one network.

    Args:
        run_dir: the run folder (trace, config); snap_dir: its it<iteration>/ snapshots; bases, lags: ints.
    Returns:
        dict with one record per (base, lag) and the gradient norms at the first base.
    """
    def weights_at(it):
        """(W_rec, W_inp) at iteration `it` as float64 arrays."""
        (Wr, Wi, _, _), _ = load(os.path.join(snap_dir, f"it{it:07d}"))
        return Wr.double().numpy(), Wi.double().numpy()

    tr = trace(run_dir)
    out = dict(run_dir=run_dir, snap_dir=snap_dir, records=[])
    for k, t in enumerate(bases):
        Wr0, Wi0 = weights_at(t)
        p0 = tr[t]
        a0 = active_mask(p0)
        cls = dict(active=a0, low=~a0 & (p0 >= SILENT), silent=p0 < SILENT)
        if k == 0:
            g = row_gradients(os.path.join(snap_dir, f"it{t:07d}"), run_dir)
            out["gradient"] = dict(base=t, **{c: float(np.median(g[ix])) if ix.any() else None for c, ix in cls.items()},
                                   inactive=float(np.median(g[~a0])))
        for D in lags:
            if not os.path.isdir(os.path.join(snap_dir, f"it{t + D:07d}")) or (t + D) not in tr:
                continue
            Wr1, Wi1 = weights_at(t + D)
            a1 = active_mask(tr[t + D])
            A, I = a0 & a1, ~a0 & ~a1
            slope = {}
            for name, post, pre in (("AA", A, A), ("IA", I, A), ("AI", A, I), ("II", I, I)):
                w0, w1 = Wr0[np.ix_(post, pre)], Wr1[np.ix_(post, pre)]
                slope[name] = float((w0 * w1).sum() / (w0 * w0).sum())
            change = np.sqrt(((Wr1 - Wr0) ** 2).sum(1) + ((Wi1 - Wi0) ** 2).sum(1))
            norm0 = np.sqrt((Wr0 ** 2).sum(1) + (Wi0 ** 2).sum(1))
            rec = dict(base=t, lag=D, n=dict(active=int(a0.sum()), low=int(cls["low"].sum()), silent=int(cls["silent"].sum())),
                       slope=slope, tau={q: (float(-D / np.log(b)) if 0 < b < 1 else None) for q, b in slope.items()},
                       change={c: float(np.median(change[ix])) if ix.any() else None for c, ix in cls.items()},
                       change_inactive=float(np.median(change[~a0])),
                       row_norm={c: float(np.median(norm0[ix])) if ix.any() else None for c, ix in cls.items()})
            out["records"].append(rec)
            print(f"{os.path.basename(snap_dir)} t={t} D={D} n={rec['n']} slope "
                  + " ".join(f"{q} {b:.3f}" for q, b in slope.items())
                  + f" | change active {rec['change']['active']:.4f} inactive {rec['change_inactive']:.4f}", flush=True)
    return out


def main():
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=("adam", "weights"))
    ap.add_argument("out_json")
    ap.add_argument("dirs", nargs="+")
    ap.add_argument("--base", type=int, nargs="+", default=[25000, 50000, 69000])
    ap.add_argument("--lags", type=int, nargs="+", default=[500, 1000, 2000, 5000, 5500])
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    res = do_adam(a.dirs) if a.mode == "adam" else do_weights(a.dirs[0], a.dirs[1], a.base, a.lags)
    with open(a.out_json, "w") as fh:
        json.dump(res, fh, indent=1)


if __name__ == "__main__":
    main()
