#!/usr/bin/env python3
"""Why does the incoming-weight gradient reach more flip-flop units than are active? (claim C4)

QUESTION. `gradient_participation.py` finds that g_in (the gradient on a unit's incoming weights)
reaches 1.5-4.5x as many flip-flop units as the participation rule calls active, while on CDDM the
two counts agree. The candidate explanations and their pre-registered predictions are in
docs/notes/agent-c4.md. This script computes the per-unit quantities those tests read.

METHOD. An independent numpy float64 forward pass and a hand-written backward pass through time
(no torch autograd, no `unit_gradients`), derived from the network's update equation
    h[t] = (1 - a) h[t-1] + a (W_rec f(h[t-1]) + W_inp (u[t-1] + n_inp[t-1]) + b + n_rec[t-1])
    out[t] = W_out (f(h[t]) + n_out),          L = mean over (output, scored t, trial) of (out - y)^2
with a = dt / tau, f = ReLU. Backward, with g_r[t] = dL/dr[t] and d[t] = dL/dh[t]:
    g_r[t] = W_out^T dL/dout[t] + a W_rec^T d[t+1]
    d[t]   = f'(h[t]) g_r[t] + (1 - a) d[t+1]
    dL/dW_rec = a sum_t d[t] r[t-1]^T,   dL/dW_inp = a sum_t d[t] u~[t-1]^T,   dL/dW_out = sum_t dL/dout[t] (r[t] + n_out)^T
The ReLU's derivative at exactly h = 0 is run both ways: 0 ("strict") and 1 (torch.clamp's choice).

Per run (read at its noisy-loss floor snapshot, as gradient_participation.py --at floor reads it):
  1. reference batch (task seeded, noise-free): rates, participation, on-counts, both gates' g_in /
     g_out, per-unit dL/dh statistics, and the production `unit_gradients` on the same batch (the
     oracle check);
  2. K training-like batches (fresh seeded batch, the run's own sigma_rec / sigma_inp / sigma_out):
     g_in per batch, and the Adam-step estimate M / (sqrt(V) + eps), M, V = mean of g, g^2 over K;
  3. the effect of one estimated Adam step at the run's learning rate on each unit's own drive,
     directly (dW_rec[i,:] r + dW_inp[i,:] u on the reference batch) and through a full re-run.

Output: one npz per run in --out.
Usage (Della CPU job):
  PYTHONPATH=<checkout> python c4_gradient_mechanism.py RUN_DIR [RUN_DIR ...] --out DIR [--K 10]
"""

import argparse
import glob
import os
import sys
import time

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gradient_participation import read_point, unit_gradients

REL = 0.05            # the scale-free bar used everywhere in the project: g >= REL * q95(g)
NEAR_INPUT = 20       # steps (2 tau) after any nonzero input that count as "input-driven"
REF_SEED = 12345      # task seed of the noise-free reference batch
ADAM_EPS = 1e-8


def load_params(path):
    """Saved network parameters as float64 arrays.

    Args:
        path: a *LastParams*.npz file.
    Returns:
        dict with W_rec (N,N), W_inp (N,n_in), W_out (n_out,N), y0 (N,), b (N,), alpha (float).
    """
    z = np.load(path, allow_pickle=True)
    assert z["activation_args"].item()["name"] == "relu"   # equation type is checked from the config
    N = z["W_rec"].shape[0]
    b = z["bias"] if z["bias"].shape else np.zeros(N)
    return dict(W_rec=z["W_rec"].astype(float), W_inp=z["W_inp"].astype(float),
                W_out=z["W_out"].astype(float), y0=z["y_init"].astype(float).reshape(-1),
                b=np.asarray(b, float).reshape(-1), alpha=float(z["dt"]) / float(z["tau"]),
                slope=float(z["activation_args"].item()["slope"]), raw=z)


def batch(cfg, seed):
    """One task batch with the task's RNG seeded.

    Args:
        cfg: the run's config; seed: int.
    Returns:
        u (n_in,T,B), y (n_out,T,B) float64, and the scored time indices.
    """
    targs = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    targs.seed = int(seed)
    u, y, _ = hydra.utils.instantiate(targs).get_batch()
    return u.astype(float), y.astype(float), get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)


def forward(P, u, rng=None, sig=(0.0, 0.0, 0.0)):
    """Euler forward pass of the h-equation, optionally with the training noise.

    The noise is drawn as RNN_torch.forward draws it: sqrt(2/alpha)*sigma Gaussian on the state
    and on the inputs, and an output noise on the rates that is constant over time per (unit, trial).

    Args:
        P: load_params dict; u: (n_in,T,B); rng: numpy Generator, or None for noise-free;
        sig: (sigma_rec, sigma_inp, sigma_out).
    Returns:
        H (T,N,B) pre-activations, U (T,n_in,B) inputs actually seen (noise included),
        n_out (N,B) output noise, out (n_out,T,B).
    """
    a, slope = P["alpha"], P["slope"]
    n_in, T, B = u.shape
    N = P["W_rec"].shape[0]
    U = np.ascontiguousarray(u.transpose(1, 0, 2))
    no = np.zeros((N, B))
    if rng is not None:
        k = np.sqrt(2 / a)
        U = U + k * sig[1] * rng.standard_normal(U.shape)
        no = k * sig[2] * rng.standard_normal((N, B))
    H = np.empty((T, N, B))
    H[0] = P["y0"][:, None]
    for t in range(1, T):
        nr = (np.sqrt(2 / a) * sig[0] * rng.standard_normal((N, B))) if rng is not None else 0.0
        r = np.maximum(slope * H[t - 1], 0.0)
        H[t] = (1 - a) * H[t - 1] + a * (P["W_rec"] @ r + P["W_inp"] @ U[t - 1] + P["b"][:, None] + nr)
    out = np.einsum("oj,tjb->otb", P["W_out"], np.maximum(slope * H, 0.0) + no[None])
    return H, U, no, out


def backward(P, H, U, no, out, y, mask, gate0):
    """Hand-written backpropagation through time of the masked MSE.

    Args:
        P, H, U, no, out: as from forward(); y: (n_out,T,B) target; mask: scored time indices;
        gate0: the ReLU derivative used at exactly h = 0 (0 = strict, 1 = torch.clamp).
    Returns:
        dict: loss, dW_rec, dW_inp, dW_out, db, and per-unit (N,) statistics of d = dL/dh:
        d_on_ss (sum of d^2 over samples with h > 0), d_ss (sum of d^2 over all samples),
        gr_mean (mean of dL/dr over all samples: < 0 means the loss wants the rate higher),
        gr_ss (sum of (dL/dr)^2).
    """
    a, slope = P["alpha"], P["slope"]
    T, N, B = H.shape
    Wr, Wo = P["W_rec"], P["W_out"]
    dout = np.zeros_like(out)
    m = np.asarray(mask, int)
    dout[:, m, :] = 2 * (out[:, m, :] - y[:, m, :]) / (out.shape[0] * len(m) * B)
    loss = float(((out[:, m, :] - y[:, m, :]) ** 2).mean())
    dWr, dWi = np.zeros_like(Wr), np.zeros_like(P["W_inp"])
    db = np.zeros(N)
    d_on_ss, d_ss, gr_sum, gr_ss = np.zeros(N), np.zeros(N), np.zeros(N), np.zeros(N)
    d_next = np.zeros((N, B))
    for t in range(T - 1, 0, -1):
        h = H[t]
        g_r = Wo.T @ dout[:, t, :] + (a * (Wr.T @ d_next) if t < T - 1 else 0.0)
        fp = slope * ((h > 0) + gate0 * (h == 0))
        d = fp * g_r + (1 - a) * d_next
        r_prev = np.maximum(slope * H[t - 1], 0.0)
        dWr += a * d @ r_prev.T
        dWi += a * d @ U[t - 1].T
        db += a * d.sum(1)
        d2 = d * d
        d_on_ss += (d2 * (h > 0)).sum(1)
        d_ss += d2.sum(1)
        gr_sum += g_r.sum(1)
        gr_ss += (g_r * g_r).sum(1)
        d_next = d
    R = np.maximum(slope * H, 0.0)
    dWo = np.einsum("otb,tjb->oj", dout, R + no[None])
    return dict(loss=loss, dW_rec=dWr, dW_inp=dWi, dW_out=dWo, db=db, d_on_ss=d_on_ss, d_ss=d_ss,
                gr_mean=gr_sum / ((T - 1) * B), gr_ss=gr_ss)


def g_in_out(G):
    """Per-unit incoming and outgoing gradient norms, the production definitions.

    Args:
        G: backward() dict.
    Returns:
        (g_in, g_out), each (N,).
    """
    g_in = np.sqrt((G["dW_rec"] ** 2).sum(1) + (G["dW_inp"] ** 2).sum(1))
    g_out = np.sqrt((G["dW_rec"] ** 2).sum(0) + (G["dW_out"] ** 2).sum(0))
    return g_in, g_out


def own_drive(S_rec, S_inp, H, U, slope):
    """The change of each unit's own drive caused by a weight change, sample by sample.

    Args:
        S_rec (N,N), S_inp (N,n_in): weight changes; H (T,N,B); U (T,n_in,B).
    Returns:
        DD (T-1,N,B): S_rec r[t-1] + S_inp u[t-1] for t = 1..T-1 (aligned with H[1:]).
    """
    T = H.shape[0]
    DD = np.empty((T - 1,) + H.shape[1:])
    for t in range(1, T):
        DD[t - 1] = S_rec @ np.maximum(slope * H[t - 1], 0.0) + S_inp @ U[t - 1]
    return DD


def unit_stats(H, U, slope):
    """Per-unit activity statistics on a batch.

    Args:
        H (T,N,B); U (T,n_in,B); slope: ReLU slope.
    Returns:
        dict of (N,) arrays: p (participation), n_on (samples with h > 0), n_zero (samples with
        h == 0 exactly), n_on_trials (trials with any h > 0), r_max, r_mean_on (mean rate over on
        samples), h_med (median h), h_q99, n_on_near_input (on samples within NEAR_INPUT steps of a
        nonzero input), plus first_input (B,) the first step any input is nonzero, near_share (the
        share of all samples that are near an input), T and B.
    """
    R = np.maximum(slope * H, 0.0)
    Rn = R.transpose(1, 0, 2).reshape(H.shape[1], -1)            # (N, T*B)
    p = Rn.std(1) + np.quantile(np.abs(Rn), 0.9, axis=1)
    on = H > 0
    n_on = on.sum((0, 2))
    act_in = (np.abs(U) > 0).any(1)                               # (T, B): any input nonzero at t
    T = H.shape[0]
    near = np.zeros_like(act_in)
    for t in range(1, T):                                         # input seen at t-1 drives h[t]
        lo = max(0, t - NEAR_INPUT)
        near[t] = act_in[lo:t].any(0)
    first_input = np.argmax(act_in, axis=0)
    hT = H.transpose(1, 0, 2).reshape(H.shape[1], -1)
    return dict(p=p, n_on=n_on, n_zero=(H == 0).sum((0, 2)), n_on_trials=on.any(0).sum(1),
                r_max=Rn.max(1), r_mean_on=Rn.sum(1) / np.maximum(n_on, 1),
                h_med=np.median(hT, 1), h_q99=np.quantile(hT, 0.99, axis=1),
                n_on_near_input=(on & near[:, None, :]).sum((0, 2)), first_input=first_input,
                near_share=near.mean(), T=H.shape[0], B=H.shape[2])


def top_mean(X, H, q=0.99):
    """Per unit, the mean of X over the samples where that unit's h is in its top 1 - q.

    Args:
        X (T-1,N,B) aligned with H[1:]; H (T,N,B).
    Returns:
        (N,) array.
    """
    Hs = H[1:].transpose(1, 0, 2).reshape(H.shape[1], -1)
    Xs = X.transpose(1, 0, 2).reshape(H.shape[1], -1)
    thr = np.quantile(Hs, q, axis=1)[:, None]
    sel = Hs >= thr
    return (Xs * sel).sum(1) / sel.sum(1)


def run_one(run_dir, out_dir, K, at):
    """All per-unit quantities for one run; writes <out_dir>/<cell>__<seed>.npz.

    Args:
        run_dir: a run folder; out_dir: output folder; K: number of noisy training-like batches;
        at: "floor" or "end" (which saved weights to read, see gradient_participation.read_point).
    """
    t0 = time.time()
    cfg = OmegaConf.load(glob.glob(os.path.join(run_dir, "*_config.yaml"))[0])
    path, read, read_iter = read_point(run_dir, at)
    assert cfg.model.equation_type == "h"
    P = load_params(path)
    sig = (float(cfg.model.sigma_rec), float(cfg.model.sigma_inp), float(cfg.model.sigma_out))
    lr = float(cfg.trainer.lr) * (100.0 / float(cfg.model.N)) ** float(cfg.trainer.lr_scale_exp)
    wd = float(cfg.trainer.weight_decay)
    res = dict(read=read, read_iter=-1 if read_iter is None else read_iter, lr=lr, wd=wd, sig=sig)

    # 1. reference batch, noise-free
    u, y, mask = batch(cfg, REF_SEED)
    H, U, no, out = forward(P, u)
    res.update({f"ref_{k}": v for k, v in unit_stats(H, U, P["slope"]).items()})
    for name, gate0 in (("strict", 0.0), ("clamp", 1.0)):
        G = backward(P, H, U, no, out, y, mask, gate0)
        gi, go = g_in_out(G)
        res[f"{name}_g_in"], res[f"{name}_g_out"], res[f"{name}_loss"] = gi, go, G["loss"]
        res[f"{name}_g_rec_row"] = np.sqrt((G["dW_rec"] ** 2).sum(1))
        res[f"{name}_g_inp_row"] = np.sqrt((G["dW_inp"] ** 2).sum(1))
        for k in ("d_on_ss", "d_ss", "gr_mean", "gr_ss"):
            res[f"{name}_{k}"] = G[k]
    print(f"  ref done {time.time() - t0:.0f}s loss={res['strict_loss']:.5f}", flush=True)

    # production gradient on the same batch (torch float32 autograd) -- the oracle check
    z = P["raw"]
    Wt = {k: z[k] for k in z.files}
    Wt["activation_args"] = Wt["activation_args"].item()
    Wt["bias"] = P["b"]
    prod = unit_gradients(Wt, u, y, mask, torch.device("cpu"))
    for k in ("g_in", "g_out", "p", "loss"):
        res[f"prod_{k}"] = prod[k]
    print(f"  prod done {time.time() - t0:.0f}s loss={prod['loss']:.5f}", flush=True)

    # 2. K training-like batches: fresh batch, training noise, strict gate (h == 0 has measure 0)
    gW = {"W_rec": [], "W_inp": []}
    M = {k: np.zeros_like(P[k]) for k in gW}
    V = {k: np.zeros_like(P[k]) for k in gW}
    g_in_noisy, n_on_noisy = [], []
    for k in range(K):
        rng = np.random.default_rng(1000 + k)
        uk, yk, mk = batch(cfg, REF_SEED + 1 + k)
        Hk, Uk, nok, outk = forward(P, uk, rng=rng, sig=sig)
        Gk = backward(P, Hk, Uk, nok, outk, yk, mk, 0.0)
        g_in_noisy.append(g_in_out(Gk)[0])
        n_on_noisy.append((Hk > 0).sum((0, 2)))
        for name, key in (("W_rec", "dW_rec"), ("W_inp", "dW_inp")):
            g = Gk[key] + wd * P[name]          # Adam's coupled L2 decay arrives through the gradient
            M[name] += g / K
            V[name] += g * g / K
        del Hk, Uk, Gk
        print(f"  noisy batch {k} done {time.time() - t0:.0f}s", flush=True)
    res["noisy_g_in"] = np.array(g_in_noisy)
    res["noisy_n_on"] = np.array(n_on_noisy)
    S = {k: -lr * M[k] / (np.sqrt(V[k]) + ADAM_EPS) for k in M}
    # per-row gradient signal-to-noise: |mean| / rms over the K batches, averaged over the row
    snr = lambda k: np.abs(M[k]) / (np.sqrt(V[k]) + 1e-300)
    res["snr_rec_row"], res["snr_inp_row"] = snr("W_rec").mean(1), snr("W_inp").mean(1)
    res["mean_g_in"] = np.sqrt((M["W_rec"] ** 2).sum(1) + (M["W_inp"] ** 2).sum(1))
    res["step_rec_row"] = np.sqrt((S["W_rec"] ** 2).sum(1))

    # 3. effect of one estimated Adam step on each unit's own drive, on the reference batch
    DD = own_drive(S["W_rec"], S["W_inp"], H, U, P["slope"])
    res["dd_top"] = top_mean(DD, H)
    res["dd_mean"] = DD.mean((0, 2))
    on = H[1:] > 0
    res["dd_on"] = (DD * on).sum((0, 2)) / np.maximum(on.sum((0, 2)), 1)
    del DD
    # full re-run with the stepped weights (network effects included)
    P2 = dict(P)
    P2["W_rec"], P2["W_inp"] = P["W_rec"] + S["W_rec"], P["W_inp"] + S["W_inp"]
    H2, U2, _, _ = forward(P2, u)
    st2 = unit_stats(H2, U2, P["slope"])
    res["step_n_on"], res["step_p"], res["step_h_q99"] = st2["n_on"], st2["p"], st2["h_q99"]
    res["step_dh_top"] = top_mean(H2[1:] - H[1:], H)
    del H2

    os.makedirs(out_dir, exist_ok=True)
    cell = os.path.basename(os.path.dirname(run_dir.rstrip("/")))
    sweep = os.path.basename(os.path.dirname(os.path.dirname(run_dir.rstrip("/"))))
    seed = os.path.basename(run_dir.rstrip("/")).split("seed=")[1].split(";")[0]
    fn = os.path.join(out_dir, f"{sweep}__{cell}__s{seed}__{at}.npz")
    np.savez(fn, **{k: np.asarray(v) for k, v in res.items()})
    print(f"wrote {fn} ({time.time() - t0:.0f}s)", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("runs", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--K", type=int, default=10)
    ap.add_argument("--at", choices=["end", "floor"], default="floor")
    a = ap.parse_args()
    for d in a.runs:
        print(d, flush=True)
        run_one(d, a.out, a.K, a.at)
