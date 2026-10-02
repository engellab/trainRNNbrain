"""Noise-free task r2 for every network of the metabolic ladder.

WHY THIS EXISTS. The run folder's score prefix is `get_validation_score(...)` evaluated at
sigma_rec = sigma_inp = 0.05 - it is r2_noisy, the quantity this project everywhere distinguishes
from r2_clean. That is the wrong probe for a metabolic ladder specifically: the penalty shrinks the
rate scale ~6x while the injected noise stays at a fixed 0.05, so signal-to-noise falls with lambda
for a reason that has nothing to do with whether the network solved the task.

RUN FROM THE WORKTREE PINNED AT 223c550f, the commit that trained this sweep. RNN_torch's
constructor has changed since (parameters removed and added, and `self_connections` flipped its
default), so rebuilding these nets with current code is not safe.

VALIDATION, criterion set before running: the re-simulated NOISY r2 must reproduce each folder's
stored score prefix to within 0.02. The two differ only in the noise draw, so a larger gap means the
rebuild is wrong and the clean numbers must not be trusted.
"""
import glob
import json
import os
import re
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "metwt"))
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask

DATA = "/Users/pt1290/Documents/GitHub/trainRNNbrain/data/trained_RNNs"
CELLS = [("0", f"{DATA}/CDDM_std_g0/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0")] + \
        [(l, f"{DATA}/CDDM_std_g0_metabolic/EqType=h_N=1000_LmbdMet={l}")
         for l in ("0.01", "0.1", "1.0", "10.0")]
TOL = 0.02


def rebuild(net_dir):
    """Rebuild one trained network from the config and parameter JSON saved beside it.

    Args:
        net_dir: path to one run folder.
    Returns:
        (rnn, task, mask, stored r2 from the folder name).
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(net_dir, "*_config.yaml"))[0])
    d = json.load(open(glob.glob(os.path.join(net_dir, "*LastParams*.json"))[0]))
    m = cfg.model
    rnn = RNN_torch(N=int(d["N"]), activation_args=d["activation_args"],
                    equation_type=str(d["equation_type"]), dale=bool(d["dale"]),
                    io_nonnegativity=bool(d["io_nonnegativity"]),
                    self_connections=bool(d["self_connections"]),
                    bias_range=list(d["bias_range"]), gamma=float(d["gamma"]),
                    dt=float(d["dt"]), tau=float(d["tau"]),
                    sigma_rec=float(m.sigma_rec), sigma_inp=float(m.sigma_inp),
                    sigma_out=float(m.sigma_out),
                    n_inputs=int(cfg.task.n_inputs), n_outputs=int(cfg.task.n_outputs), seed=0)
    with torch.no_grad():
        for k in ("W_rec", "W_inp", "W_out"):
            getattr(rnn, k).copy_(torch.tensor(np.asarray(d[k], dtype=np.float32)))
        yi = torch.tensor(np.asarray(d["y_init"], dtype=np.float32))
        if isinstance(rnn.y_init, torch.nn.Parameter):
            rnn.y_init.copy_(yi)
        else:
            rnn.y_init = yi
        if d.get("bias") is not None and np.asarray(d["bias"]).size == rnn.N:
            rnn.bias.copy_(torch.tensor(np.asarray(d["bias"], dtype=np.float32)))
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    stored = float(re.match(r"(-?[0-9.]+|nan)_", os.path.basename(net_dir)).group(1))
    return rnn, task, mask, stored


rows = []
for lam, cell in CELLS:
    for net_dir in sorted(glob.glob(os.path.join(cell, "*"))):
        if not os.path.isdir(net_dir) or os.path.basename(net_dir).startswith("nan"):
            continue
        rnn, task, mask, stored = rebuild(net_dir)
        bi_np, bt_np, _ = task.get_batch()
        bi = torch.tensor(bi_np, dtype=torch.float32)
        bt = torch.tensor(bt_np, dtype=torch.float32)
        with torch.no_grad():
            r2_noisy = float(Trainer.r2_score(rnn(bi, w_noise=True)[1], bt, mask))
            srec, sinp = float(rnn.sigma_rec), float(rnn.sigma_inp)
            rnn.sigma_rec = rnn.sigma_inp = 0.0
            r2_clean = float(Trainer.r2_score(rnn(bi, w_noise=False)[1], bt, mask))
            rnn.sigma_rec, rnn.sigma_inp = srec, sinp
        rows.append((lam, stored, r2_noisy, r2_clean))
        print(f"lam={lam:>5}  stored {stored:.4f}  resim-noisy {r2_noisy:.4f}  "
              f"clean {r2_clean:.4f}  |resim-stored| {abs(r2_noisy-stored):.4f}", flush=True)

print()
gap = max(abs(n - s) for _, s, n, _ in rows)
print(f"VALIDATION: worst |re-simulated noisy - stored| = {gap:.4f}  (tolerance {TOL})")
print("REBUILD VALID" if gap <= TOL else "REBUILD FAILED - do not trust the clean numbers")
print()
print(f"{'lambda':>7} {'n':>2} {'stored (noisy)':>16} {'clean':>18}")
for lam, _ in CELLS:
    g = [r for r in rows if r[0] == lam]
    s = np.array([r[1] for r in g]); c = np.array([r[3] for r in g])
    print(f"{lam:>7} {len(g):2d}   {s.mean():.4f} +/- {s.std(ddof=1):.4f}"
          f"     {c.mean():.4f} +/- {c.std(ddof=1):.4f}")
np.save(os.path.join(HERE, "clean_r2_rows.npy"), np.array(rows, dtype=object), allow_pickle=True)
