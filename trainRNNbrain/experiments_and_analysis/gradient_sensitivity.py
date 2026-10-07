#!/usr/bin/env python3
"""Three per-unit measures of "does learning still reach this unit", untrained twin vs trained.

Follows gradient_participation.py, whose state gradient (A below) is kept for comparison.

  A  g_state  RMS_{t,k} dL/dh_i(t), divided by sqrt(L). The batch-loss gradient on the unit's state.
  B  g_in     per-TRIAL incoming-weight gradient: G_ik = || dL_k / d[W_rec, W_inp][i,:] ||, RMS over
              trials k, divided by sqrt(L). Per trial, because the BATCH weight gradient is ~0 for
              every weight at a converged network (the trials' pulls cancel), so it measures how
              unconverged the network is, not whether a unit can be reached.
  C  s_in     output sensitivity of the incoming weights, no loss at all:
              S_ik = || d y_k / d[W_rec, W_inp][i,:] ||_F over the scored outputs and timepoints,
              RMS over trials. "If this unit's inputs changed, would the output move?" A silent
              ReLU unit scores exactly 0. No residual error enters, so no sqrt(L) normalisation.
     s_state  the same for the unit's state: RMS_{t,k} || d y / d h_i(t) ||.

HOW THE PER-TRIAL WEIGHT NORMS ARE COMPUTED WITHOUT ONE BACKWARD PASS PER TRIAL. The update is
x(t) = x(t-1) + a(-x(t-1) + W_rec f(x(t-1)) + W_inp u(t-1)), so for one trial
dL/dW[i,:] = a sum_t d_i(t) z(t-1), with d = dL/dx and z = [f(x); u]. Its squared norm is
d_i^T K d_i with K = Z^T Z, the (T x T) Gram matrix of z over time for that trial. One backward
pass gives d for every trial at once (trials do not interact), and K is per trial.

THE JACOBIAN NORM IN C IS ESTIMATED, NOT EXACT. ||J||_F^2 = E_v ||v^T J||^2 for v ~ N(0, I) over the
scored outputs (Hutchinson). N_PROBE random cotangents, each one backward pass. The estimate is
unbiased per unit; its relative error per unit is about sqrt(2 / N_PROBE) in the squared norm.

A CHECK RUNS FIRST on the first network: B and C from the Gram trick are compared with plain
autograd on single trials, and the script stops if they disagree by more than 1e-3 relative.

Usage (on Della):
  PYTHONPATH=~/trainRNNbrain_recinit python gradient_sensitivity.py CELL [CELL ...] --out DIR
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from gradient_participation import act_fn, initial_params

N_PROBE = 16


def forward(W, U, dev):
    """Noise-free run; returns states list (each requires grad), rates, outputs and the weights."""
    t = lambda a: torch.as_tensor(np.asarray(a, dtype=np.float32), device=dev)
    Wr, Wi, Wo = (t(W[k]).requires_grad_(True) for k in ("W_rec", "W_inp", "W_out"))
    b, y0 = t(W["bias"]).reshape(-1, 1), t(W["y_init"]).reshape(-1, 1)
    alpha = float(W["dt"]) / float(W["tau"])
    a = W["activation_args"]
    f, _ = act_fn(a["name"], float(a.get("slope", 1.0)))
    B, T = U.shape[2], U.shape[1]
    xs = [y0.repeat(1, B).requires_grad_(True)]
    for k in range(1, T):
        x = xs[-1]
        xs.append(x + alpha * (-x + Wr @ f(x) + Wi @ U[:, k - 1, :] + b))
    H = torch.stack(xs, 1)
    R = f(H)
    out = torch.einsum("oj,jtk->otk", Wo, R)
    return xs, R, out, (Wr, Wi, Wo), alpha


def row_norms_sq(D, Z, alpha):
    """Per (unit, trial) squared norm of alpha * sum_t D[i,t,k] Z[:,t-1,k], via the Gram matrix.

    Args:
        D: (N, T, B) gradient on the states; Z: (M, T, B) presynaptic signal [f(x); u].
    Returns:
        (N, B) tensor.
    """
    d, z = D[:, 1:, :], Z[:, :-1, :]
    out = []
    for k in range(D.shape[2]):            # per trial: K is (T-1)^2, small
        K = z[:, :, k].T @ z[:, :, k]
        dk = d[:, :, k]
        out.append(((dk @ K) * dk).sum(1))
    return alpha ** 2 * torch.stack(out, 1)


def measures(W, u, y, mask, dev, check=False):
    """A, B, C and s_state per unit for one parameter set. Returns dict of numpy arrays + loss."""
    U, Y = (torch.as_tensor(np.asarray(v, dtype=np.float32), device=dev) for v in (u, y))
    m = torch.as_tensor(np.asarray(mask), device=dev, dtype=torch.long)
    xs, R, out, (Wr, Wi, Wo), alpha = forward(W, U, dev)
    B = U.shape[2]
    loss = ((out[:, m, :] - Y[:, m, :]) ** 2).mean()
    Z = torch.cat([R, U], 0)

    D = torch.stack(torch.autograd.grad(loss, xs, retain_graph=True), 1)     # (N, T, B)
    g_state = D.pow(2).mean(dim=(1, 2)).sqrt()
    # per-trial loss L_k = B * (its share of the batch mean), so its gradient is B * D[..., k]
    g_in = (row_norms_sq(D * B, Z, alpha).mean(1)).sqrt()

    gen = torch.Generator(device=dev).manual_seed(0)
    s_in2 = torch.zeros(D.shape[0], B, device=dev)
    s_st2 = torch.zeros(D.shape[0], device=dev)
    sel = torch.zeros_like(out)
    sel[:, m, :] = 1.0
    for _ in range(N_PROBE):
        v = torch.randn(out.shape, generator=gen, device=dev) * sel
        Dv = torch.stack(torch.autograd.grad(out, xs, grad_outputs=v, retain_graph=True), 1)
        s_in2 += row_norms_sq(Dv, Z, alpha) / N_PROBE
        s_st2 += Dv.pow(2).sum(1).mean(1) / N_PROBE        # sum over t of the Jacobian, per trial
    s_in = s_in2.mean(1).sqrt()
    s_state = s_st2.sqrt()

    if check:
        _check(W, u, y, mask, dev, D * B, Z, alpha)
    a = W["activation_args"]
    _, fp = act_fn(a["name"], float(a.get("slope", 1.0)))
    with torch.no_grad():
        Rn = R.detach().cpu().numpy()
    res = {k: v.detach().cpu().numpy().astype(float) for k, v in
           dict(g_state=g_state, g_in=g_in, s_in=s_in, s_state=s_state).items()}
    res["p"] = Rn.std(axis=(1, 2)) + np.quantile(np.abs(Rn), 0.9, axis=(1, 2))
    res["loss"] = float(loss)
    return res


def _check(W, u, y, mask, dev, Dper, Z, alpha):
    """B's Gram trick against plain autograd, on two single trials."""
    for k in (0, u.shape[2] // 2):
        U1 = torch.as_tensor(np.asarray(u[:, :, k:k + 1], dtype=np.float32), device=dev)
        Y1 = torch.as_tensor(np.asarray(y[:, :, k:k + 1], dtype=np.float32), device=dev)
        m = torch.as_tensor(np.asarray(mask), device=dev, dtype=torch.long)
        xs, R, out, (Wr, Wi, Wo), _ = forward(W, U1, dev)
        Lk = ((out[:, m, :] - Y1[:, m, :]) ** 2).mean()
        gr, gi = torch.autograd.grad(Lk, [Wr, Wi])
        direct = (gr.pow(2).sum(1) + gi.pow(2).sum(1))
        trick = row_norms_sq(Dper[:, :, k:k + 1], Z[:, :, k:k + 1], alpha)[:, 0]
        rel = float((direct - trick).abs().max() / direct.max())
        print(f"  CHECK trial {k}: max |direct - gram| / max = {rel:.2e}", flush=True)
        if rel > 1e-3:
            raise SystemExit("Gram-trick check failed")


def main(cells, out_dir, dev):
    os.makedirs(out_dir, exist_ok=True)
    first = True
    for cell in cells:
        for d in sorted(glob.glob(os.path.join(cell, "*/"))):
            name = os.path.basename(d.rstrip("/"))
            if name.startswith("nan"):
                continue
            cfg = OmegaConf.load(glob.glob(os.path.join(d, "*_config.yaml"))[0])
            task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
            u, y, _ = task.get_batch()
            mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
            z = np.load(glob.glob(os.path.join(d, "*LastParams*.npz"))[0], allow_pickle=True)
            Wt = {k: z[k] for k in z.files}
            Wt["activation_args"] = Wt["activation_args"].item()
            Wt["bias"] = Wt["bias"] if Wt["bias"].shape else np.zeros(int(Wt["N"]))
            W0 = initial_params(cfg)
            W0["activation_args"] = dict(W0["activation_args"])
            if W0["bias"] is None:
                W0["bias"] = np.zeros(int(W0["N"]))
            rec = {"cell": os.path.basename(cell), "run": name, "batch": int(u.shape[2])}
            for tag, W in (("init", W0), ("trained", Wt)):
                r = measures(W, u, y, mask, dev, check=first)
                first = False
                rec[tag] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in r.items()}
            json.dump(rec, open(os.path.join(out_dir, f"{rec['cell']}__{name[:40]}.json"), "w"))
            line = [f"{rec['cell'][-30:]} {name[:9]}"]
            for key in ("g_state", "g_in", "s_state", "s_in"):
                g0, g = np.asarray(rec["init"][key]), np.asarray(rec["trained"][key])
                if key.startswith("g"):
                    g0, g = g0 / np.sqrt(rec["init"]["loss"]), g / np.sqrt(rec["trained"]["loss"])
                bar = 0.05 * np.quantile(g0, 0.95)
                line.append(f"{key}: above={int((g >= bar).sum())} "
                            f"med/med0={np.median(g) / np.median(g0):.3g}")
            print(" | ".join(line), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cells", nargs="+")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    main(a.cells, a.out, torch.device("cuda" if torch.cuda.is_available() else "cpu"))
