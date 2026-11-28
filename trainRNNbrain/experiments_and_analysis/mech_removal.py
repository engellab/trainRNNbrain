#!/usr/bin/env python3
"""How much does held-out R^2 fall when ONE unit is removed? (mechanism tests, part B)

Question. A unit that was inactive when a penalty was switched on, and is active 25,000 iterations later, may or may not
matter to the output. This script removes units one at a time, and in groups, and measures the fall in the project's
performance measure (held-out R^2 under the training noise, heldout_r2.py).

Method.
- Removal: the unit's outgoing weights (its column of W_rec and its column of W_out) are set to zero. Nothing is retrained.
- Paired noise: intact and cut networks see the same noise draws, so the difference has no noise-draw variance to first order.
- Every batch is heldout_r2.heldout_batch (392 CDDM conditions at the midpoints of the trained coherences).
- The forward pass is written here (h equation, Euler, float32), not taken from RNN_torch; the noise comes from a numpy
  generator seeded per draw, so its draws differ from heldout_r2's and only the noise levels are shared.
- Activity: participation std(r) + q0.9(|r|) on a noise-free pass over the task's training grid, active if at least 5% of the
  network's 95th percentile (common.active_mask).
Sets, when --switch gives the weights at the switch: `resurrected` (inactive at the switch, active now), `kept` (active at
both), `off` (inactive now). Without --switch: `kept` is every active unit and `off` every inactive one.

Output: one JSON with the per-unit, per-draw fall in R^2 for every unit of `resurrected` and `kept` and for up to N_OFF units
of `off`, and the fall when each whole group is removed.

Usage (CPU node):  python mech_removal.py RUN_DIR OUT_JSON [--switch PARENT_DIR] [--draws 4] [--n-off 100]
"""
import argparse
import glob
import json
import os
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H  # noqa: E402
from common import active_mask  # noqa: E402
from trainRNNbrain.training.training_utils import prepare_task_arguments  # noqa: E402

REMOVAL_SEED = 20261125      # seeds the sample of `off` units and the random groups
NOISE_BASE = 31000           # noise draw d uses numpy seed NOISE_BASE + d


def load(run_dir):
    """Weights and config of one saved network.

    Args:
        run_dir: folder holding *LastParams* and *_config.yaml.
    Returns:
        (W_rec (N,N), W_inp (N,n_in), W_out (n_out,N), y0 (N,)) float32 tensors, and the OmegaConf config.
    """
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(run_dir, "*_config.yaml")))[0])
    P, _ = H._params(run_dir)
    W = [torch.from_numpy(np.asarray(P[k], dtype=np.float32)) for k in ("W_rec", "W_inp", "W_out", "y_init")]
    return W, cfg


def noise_draw(seed, N, n_in, T, B, alpha, sig):
    """One draw of the three noise terms of the h equation, scaled as in training (sqrt(2/alpha) * sigma).

    Args:
        seed: int; N, n_in, T, B: sizes; alpha: dt/tau; sig: (sigma_rec, sigma_inp, sigma_out).
    Returns:
        tuple of float32 tensors (N,T,B), (n_in,T,B), (N,1,B).
    """
    g = np.random.default_rng(seed)
    q = float(np.sqrt(2.0 / alpha))
    return tuple(torch.from_numpy((q * s * g.standard_normal(shp, dtype=np.float32)).astype(np.float32))
                 for s, shp in zip(sig, ((N, T, B), (n_in, T, B), (N, 1, B))))


def forward(W, u, alpha, noise=None, cut=None, keep=False):
    """Euler pass of the h equation with some units' outgoing weights set to zero.

    Args:
        W: (W_rec, W_inp, W_out, y0); u: (n_in,T,B) inputs; alpha: dt/tau; noise: noise_draw's tuple or None;
        cut: list of unit indices to remove, or None; keep: also return the rates (N,T,B).
    Returns:
        outputs (n_out,T,B), and the rates if keep.
    """
    Wr, Wi, Wo, y0 = W
    if cut is not None and len(cut):
        Wr, Wo = Wr.clone(), Wo.clone()
        Wr[:, cut] = 0.0
        Wo[:, cut] = 0.0
    _, T, B = u.shape
    x = y0.reshape(-1, 1).repeat(1, B)
    out, R = [], []
    for t in range(T):
        r = x.clamp(min=0.0)
        out.append(Wo @ (r if noise is None else r + noise[2][:, 0]))
        if keep:
            R.append(r)
        i = u[:, t] if noise is None else u[:, t] + noise[1][:, t]
        dx = Wr @ r + Wi @ i - x
        if noise is not None:
            dx = dx + noise[0][:, t]
        x = x + alpha * dx
    z = torch.stack(out, 1)
    return (z, torch.stack(R, 1)) if keep else z


def r2(z, y, mask):
    """1 - MSE/variance over the scored steps, all outputs and trials pooled (Trainer.r2_score)."""
    z, y = z[:, mask].double(), y[:, mask].double()
    return float(1.0 - (z - y).pow(2).mean() / (y - y.mean()).pow(2).mean())


def participation(W, cfg):
    """Participation of every unit on a noise-free pass over the task's training grid.

    Args:
        W: weights; cfg: the run's saved config.
    Returns:
        (N,) float64 array.
    """
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    u = torch.from_numpy(np.asarray(task.get_batch()[0], dtype=np.float32))
    _, R = forward(W, u, float(cfg.model.dt) / float(cfg.model.tau), keep=True)
    R = R.reshape(R.shape[0], -1).numpy()
    return np.array([r.std(dtype=np.float64) + np.quantile(np.abs(r), 0.9) for r in R])


def main():
    """Command line: see the module docstring."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("out_json")
    ap.add_argument("--switch", default=None, help="folder with the weights at the switch")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--n-off", type=int, default=100)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("SLURM_CPUS_PER_TASK", "8")))
    W, cfg = load(a.run_dir)
    alpha, N = float(cfg.model.dt) / float(cfg.model.tau), int(cfg.model.N)
    p_now = participation(W, cfg)
    now = active_mask(p_now)
    if a.switch:
        p0 = participation(load(a.switch)[0], cfg)
        then = active_mask(p0)
    else:
        p0, then = None, now
    rng = np.random.default_rng(REMOVAL_SEED)
    sets = dict(resurrected=np.where(now & ~then)[0], kept=np.where(now & then)[0],
                lost=np.where(~now & then)[0], off=np.where(~now)[0])
    off_s = np.sort(rng.choice(sets["off"], min(a.n_off, len(sets["off"])), replace=False)) if len(sets["off"]) else []
    sig = [float(cfg.model[k]) for k in ("sigma_rec", "sigma_inp", "sigma_out")]
    u, y, mask = H.heldout_batch(cfg)
    noises = [noise_draw(NOISE_BASE + d, N, u.shape[0], u.shape[1], u.shape[2], alpha, sig) for d in range(a.draws)]

    def score(cut):
        """R^2 per noise draw with the units `cut` removed."""
        return [r2(forward(W, u, alpha, nz, cut), y, mask) for nz in noises]

    t0 = time.time()
    base = score(None)
    print(f"{a.run_dir}\nactive {int(now.sum())} then {int(then.sum())}; "
          + ", ".join(f"{k} {len(v)}" for k, v in sets.items())
          + f"; intact R2 {np.round(base, 4).tolist()} ({time.time() - t0:.1f} s)", flush=True)
    out = dict(run_dir=a.run_dir, switch=a.switch, draws=a.draws, intact=base, active_now=int(now.sum()),
               active_then=int(then.sum()), p_now=p_now.tolist(), p_then=None if p0 is None else p0.tolist(),
               sets={k: v.tolist() for k, v in sets.items()}, fall={}, group={})
    for name, idx in (("resurrected", sets["resurrected"]), ("off", off_s), ("kept", sets["kept"])):
        out["fall"][name] = {int(i): [b - c for b, c in zip(base, score([int(i)]))] for i in idx}
        f = np.array([np.mean(v) for v in out["fall"][name].values()])
        if len(f):
            print(f"{name}: {len(f)} units, mean fall in R2 median {np.median(f):.2e} q95 {np.quantile(f, .95):.2e} "
                  f"max {f.max():.2e} ({time.time() - t0:.0f} s)", flush=True)
        json.dump(out, open(a.out_json, "w"))
    # groups: all resurrected; as many kept units at random (3 draws); as many of the off units; every off unit
    k = len(sets["resurrected"])
    grp = {"resurrected": [sets["resurrected"]], "all_off": [sets["off"]]}
    if k:
        grp["kept_random"] = [rng.choice(sets["kept"], min(k, len(sets["kept"])), replace=False) for _ in range(3)]
        grp["off_random"] = [rng.choice(sets["off"], min(k, len(sets["off"])), replace=False) for _ in range(3)]
    for name, idxs in grp.items():
        out["group"][name] = [dict(n=len(ix), fall=[b - c for b, c in zip(base, score([int(i) for i in ix]))])
                              for ix in idxs if len(ix)]
        print(name, [(g["n"], round(float(np.mean(g["fall"])), 4)) for g in out["group"][name]], flush=True)
    json.dump(out, open(a.out_json, "w"))


if __name__ == "__main__":
    main()
