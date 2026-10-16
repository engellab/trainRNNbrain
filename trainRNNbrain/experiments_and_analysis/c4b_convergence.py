#!/usr/bin/env python3
"""Is the 3-bit flip-flop still learning? Tests of Pavel's hypothesis on claim C4 (agent-c4b).

C4 found that on the flip-flop the gradient reaches the incoming weights of silent units and training
keeps them silent, while on CDDM a third of the units never cross threshold and get exactly zero
gradient. Hypothesis H: the flip-flop has not converged, and the gradient on its silent units is the
signature of learning still going on. The pre-registered predictions and pass/fail bars are in
git show db10e7c:docs/notes/agent-c4b.md; this script computes what they read.

Subcommands (all CPU; run on Della under sbatch, the data live there):
  check     torch gradient (this file) against the C4 numpy float64 BPTT oracle on one noisy batch,
            plus the time of one forward/backward
  loss      T1, slope of the noisy training loss over 50k-iteration windows
  drift     T3, weight-motion directedness: trace drift exponent and per-unit consecutive-displacement
            cosines from the kept snapshots
  grad      T2, at one read point: per-unit gradient signal share over K seeded training-like batches
  finetune  T4, from one run end: 2,000 iterations of continued / annealed / plain-SGD training on
            seeded batches, a fixed seeded evaluation set, then the C4 reach re-read
  report    tables and the figure, from the npz/json outputs (local)

Usage: PYTHONPATH=<checkout> python c4b_convergence.py <subcommand> [args] --out DIR
"""

import argparse
import glob
import json
import os
import pickle
import sys
import time

import hydra
import matplotlib
import numpy as np
import torch
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.training.run_experiment import load_warm_start
from trainRNNbrain.utils import filter_kwargs, import_any

matplotlib.use("Agg")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import common
import c4_gradient_mechanism as C4

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../data")
CELL = "EqType=h_N=1000_arm=control"
REF_SEED = 12345        # reference batch (noise-free), as in C4
REL = 0.05              # the project's scale-free bar: x >= REL * q95(x)
TOP_Q = 0.99            # a unit's "closest to firing" samples: its top 1% of h (C4's definition)

# warm-start chains, oldest run first: (sweep, seed)
CHAINS = {
    "FF-A": [("NBitFlipFlop_c1_snapshots", 2670297180), ("NBitFlipFlop_c1_snapshots_c200000", 1258717025)],
    "FF-B": [("NBitFlipFlop_c1_snapshots", 1542248578), ("NBitFlipFlop_c1_snapshots_c200000", 2461724787),
             ("NBitFlipFlop_c1_snapshots_c300000", 3194339211)],
    "FF-C": [("NBitFlipFlop_c1_snapshots", 2851654641), ("NBitFlipFlop_c1_snapshots_c200000", 4113307577),
             ("NBitFlipFlop_c1_snapshots_c300000", 3204992658)],
    "CDDM-1": [("CDDM_c1_snapshots", 1423687489)],
    "CDDM-2": [("CDDM_c1_snapshots", 2657899762)],
    "CDDM-3": [("CDDM_c1_snapshots", 2670295995)],
    "DMTS-1": [("DMTS_c1_snapshots", 198893703)],
    "DMTS-2": [("DMTS_c1_snapshots", 3322309204)],
    "DMTS-3": [("DMTS_c1_snapshots", 3753404946)],
}
# T2 read points (absolute iterations) per task
GRAD_READS = {"FF": (50000, 100000, 200000, 300000), "CDDM": (25000, 50000, 100000), "DMTS": (50000, 150000)}
WINDOW = 50000          # T1 / T3 window length


# --------------------------------------------------------------------------------------------
# where the weights are
# --------------------------------------------------------------------------------------------

def run_dir(sweep, seed):
    """The run folder of one (sweep, seed) of the claim-1 control cell."""
    hits = glob.glob(os.path.join(DATA, "trained_RNNs", sweep, CELL, f"*seed={seed};*"))
    assert len(hits) == 1, (sweep, seed, hits)
    return hits[0]


def chain_runs(name):
    """The runs of a chain with their absolute iteration ranges.

    Args:
        name: a CHAINS key.
    Returns:
        list of dicts (sweep, seed, dir, start, end), oldest first; `end` is the run's last iteration.
    """
    out = []
    for sweep, seed in CHAINS[name]:
        d = run_dir(sweep, seed)
        cfg = OmegaConf.load(glob.glob(os.path.join(d, "*_config.yaml"))[0])
        start = int(cfg.trainer.get("iter_offset", 0) or 0)
        out.append(dict(sweep=sweep, seed=seed, dir=d, start=start, end=start + int(cfg.trainer.max_iter)))
    return out


def weights_dir(name, it):
    """The folder holding the weights of chain `name` at absolute iteration `it`.

    A run's last iteration is its own folder (LastParams, with AdamState); any other multiple of 500
    is a kept snapshot of the run that covered it.

    Returns:
        (folder, run dict of the run that covers `it`).
    """
    for r in chain_runs(name):
        if it == r["end"]:
            return r["dir"], r
        if r["start"] < it < r["end"]:
            d = os.path.join(DATA, "c1_snapshots", r["sweep"], CELL, f"s{r['seed']}", f"it{it:07d}")
            assert os.path.isdir(d), d
            return d, r
    raise ValueError(f"{name} has no weights at {it}")


def chain_end(name):
    """Last iteration of a chain."""
    return chain_runs(name)[-1]["end"]


def task_of(name):
    """'FF', 'CDDM' or 'DMTS' from a chain name."""
    return name.split("-")[0]


def load_cfg(name):
    """The config of the chain's last run (the task is the same along the chain)."""
    return OmegaConf.load(glob.glob(os.path.join(chain_runs(name)[-1]["dir"], "*_config.yaml"))[0])


def load_W(folder, dtype=torch.float32):
    """Network parameters from a run or snapshot folder, as torch tensors on the CPU.

    Returns:
        dict W_rec (N,N), W_inp (N,n_in), W_out (n_out,N), y0 (N,), b (N,), alpha (float).
    """
    P = C4.load_params(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0])
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)
    return dict(W_rec=t(P["W_rec"]), W_inp=t(P["W_inp"]), W_out=t(P["W_out"]), y0=t(P["y0"]),
                b=t(P["b"]), alpha=P["alpha"])


def batch(cfg, seed, dtype=torch.float32):
    """A seeded task batch as torch tensors: u (n_in,T,B), y (n_out,T,B), mask (LongTensor)."""
    u, y, mask = C4.batch(cfg, seed)
    t = lambda a: torch.as_tensor(np.asarray(a), dtype=dtype)
    return t(u), t(y), torch.as_tensor(np.asarray(mask), dtype=torch.long)


# --------------------------------------------------------------------------------------------
# forward / gradient (torch autograd; checked against C4's numpy BPTT by `check`)
# --------------------------------------------------------------------------------------------

def draw_noise(cfg, shape, seed, dtype=torch.float32):
    """Training noise as the production forward draws it, from a seeded torch generator.

    Args:
        cfg: run config (sigma_rec, sigma_inp, sigma_out, dt, tau); shape: (N, n_in, T, B); seed: int.
    Returns:
        (n_inp (n_in,T,B), n_rec (N,T,B), n_out (N,B)), each already scaled by sqrt(2/alpha) sigma.
    """
    N, n_in, T, B = shape
    g = torch.Generator().manual_seed(int(seed))
    k = np.sqrt(2.0 / (float(cfg.model.dt) / float(cfg.model.tau)))
    s = lambda sig, shp: k * float(sig) * torch.randn(*shp, generator=g, dtype=dtype)
    return s(cfg.model.sigma_inp, (n_in, T, B)), s(cfg.model.sigma_rec, (N, T, B)), s(cfg.model.sigma_out, (N, B))


def forward(W, u, noise=None):
    """Euler forward pass of the h equation (ReLU), differentiable in W.

    h[t] = h[t-1] + a(-h[t-1] + W_rec relu(h[t-1]) + W_inp (u[t-1] + n_inp[t-1]) + b + n_rec[t-1])
    out[t] = W_out (relu(h[t]) + n_out)

    Args:
        W: load_W dict (W_rec, W_inp, W_out may require grad); u: (n_in,T,B);
        noise: draw_noise tuple, or None for noise-free.
    Returns:
        H (N,T,B) pre-activations, out (n_out,T,B).
    """
    a = W["alpha"]
    n_in, T, B = u.shape
    N = W["W_rec"].shape[0]
    x = W["y0"].reshape(-1, 1).expand(N, B)
    b = W["b"].reshape(-1, 1)
    xs = [x]
    for t in range(1, T):
        inp = u[:, t - 1] if noise is None else u[:, t - 1] + noise[0][:, t - 1]
        drive = W["W_rec"] @ torch.relu(x) + W["W_inp"] @ inp + b
        if noise is not None:
            drive = drive + noise[1][:, t - 1]
        x = x + a * (-x + drive)
        xs.append(x)
    H = torch.stack(xs, 1)
    R = torch.relu(H)
    if noise is not None:
        R = R + noise[2][:, None, :]
    return H, torch.einsum("oj,jtb->otb", W["W_out"], R)


def gradient(W, u, y, mask, noise=None):
    """Masked-MSE loss and its gradient on W_rec, W_inp, W_out (the task loss, no weight decay).

    Returns:
        (loss float, dict dW_rec, dW_inp, dW_out).
    """
    Wg = dict(W)
    for k in ("W_rec", "W_inp", "W_out"):
        Wg[k] = W[k].detach().clone().requires_grad_(True)
    _, out = forward(Wg, u, noise)
    loss = ((out[:, mask, :] - y[:, mask, :]) ** 2).mean()
    g = torch.autograd.grad(loss, [Wg["W_rec"], Wg["W_inp"], Wg["W_out"]])
    return float(loss), dict(dW_rec=g[0].detach(), dW_inp=g[1].detach(), dW_out=g[2].detach())


def reference_stats(W, u):
    """Per-unit activity on the noise-free reference batch, and each unit's presynaptic vector at
    its closest-to-firing samples.

    Args:
        W: load_W dict; u: (n_in,T,B) reference inputs.
    Returns:
        dict p (participation), n_on (samples with h > 0), X (N, N + n_in): the mean of
        [relu(h[t-1]); u[t-1]] over the unit's top-1% h samples (t >= 1).
    """
    with torch.no_grad():
        H, _ = forward(W, u)
        N, T, B = H.shape
        R = torch.relu(H)
        Rn = R.reshape(N, -1)
        k90 = int(np.ceil(0.9 * Rn.shape[1]))
        p = Rn.std(1) + Rn.kthvalue(k90, dim=1).values
        n_on = (H > 0).reshape(N, -1).sum(1)
        Hs = H[:, 1:].reshape(N, -1)
        thr = Hs.kthvalue(int(np.ceil(TOP_Q * Hs.shape[1])), dim=1).values.reshape(-1, 1)
        Xr = torch.zeros(N, N)
        Xi = torch.zeros(N, u.shape[0])
        cnt = torch.zeros(N)
        for t in range(1, T):
            sel = (H[:, t] >= thr).to(H.dtype)                       # (N,B)
            Xr += sel @ R[:, t - 1].T
            Xi += sel @ u[:, t - 1].T
            cnt += sel.sum(1)
    X = torch.cat([Xr, Xi], 1) / cnt.clamp_min(1).reshape(-1, 1)
    return dict(p=p.numpy().astype(float), n_on=n_on.numpy(), X=X)


def over(x, rel=REL):
    """Boolean mask x >= rel * q95(x)."""
    return x >= rel * np.quantile(x, 0.95)


def rows(g):
    """Incoming-row gradient matrix [dW_rec | dW_inp], (N, N + n_in)."""
    return torch.cat([g["dW_rec"], g["dW_inp"]], 1)


def classes(ref, g_in_ref):
    """Unit classes on the reference batch (see the notes).

    Returns:
        dict of boolean (N,) masks: active, silent_on, never_on, silent_reached.
    """
    act = over(ref["p"])
    return dict(active=act, silent_on=~act & (ref["n_on"] > 0), never_on=ref["n_on"] == 0,
                silent_reached=~act & over(g_in_ref))


def signal_share(cfg, W, X, K, seed0, out_prefix=None):
    """Gradient signal versus batch-to-batch noise over K seeded training-like batches (T2).

    Args:
        cfg: run config; W: load_W dict; X: (N, N + n_in) presynaptic vectors from reference_stats;
        K: number of batches (even); seed0: task seed of batch 0 (noise seed seed0 + 10**6 + k).
    Returns:
        dict of per-unit arrays: F (signal share), S (debiased squared mean), s2 (per-batch variance),
        c (split-half cosine), c_null (2, N) the same under random batch signs, drive (K, N) the
        drive projection -g . X per batch, t_drive; and network-wide F_net, losses (K,).
    """
    N = W["W_rec"].shape[0]
    sgn = np.random.default_rng(seed0).choice([-1.0, 1.0], size=(2, K))
    acc = {k: 0.0 for k in ("A", "B", "nA0", "nB0", "nA1", "nB1")}
    sq = torch.zeros(N)
    net_sum, net_sq = None, 0.0
    drive, losses = [], []
    for k in range(K):
        u, y, mask = batch(cfg, seed0 + k)
        noise = draw_noise(cfg, (N, u.shape[0], u.shape[1], u.shape[2]), seed0 + 10 ** 6 + k)
        L, g = gradient(W, u, y, mask, noise)
        G = rows(g)
        half = "A" if k < K // 2 else "B"
        acc[half] = acc[half] + G
        for j in range(2):
            acc[f"n{half}{j}"] = acc[f"n{half}{j}"] + float(sgn[j, k]) * G
        sq += (G * G).sum(1)
        flat = torch.cat([g["dW_rec"].reshape(-1), g["dW_inp"].reshape(-1), g["dW_out"].reshape(-1)])
        net_sum = flat if net_sum is None else net_sum + flat
        net_sq += float((flat * flat).sum())
        drive.append((-(G * X).sum(1)).numpy())
        losses.append(L)
        del u, y, noise, g
    mean = (acc["A"] + acc["B"]) / K
    m2 = (mean * mean).sum(1)
    s2 = (sq - K * m2) / (K - 1)
    S = m2 - s2 / K
    F = S / (S + s2)
    cos = lambda a, b: ((a * b).sum(1) / (a.norm(dim=1) * b.norm(dim=1) + 1e-30)).numpy()
    c = cos(acc["A"], acc["B"])
    c_null = np.stack([cos(acc[f"nA{j}"], acc[f"nB{j}"]) for j in range(2)])
    nm = net_sum / K
    nm2 = float((nm * nm).sum())
    ns2 = (net_sq - K * nm2) / (K - 1)
    nS = nm2 - ns2 / K
    drive = np.array(drive)
    t_drive = drive.mean(0) / (drive.std(0, ddof=1) / np.sqrt(K) + 1e-300)
    return dict(F=F.numpy(), S=S.numpy(), s2=s2.numpy(), c=c, c_null=c_null, drive=drive,
                t_drive=t_drive, F_net=nS / (nS + ns2), losses=np.array(losses),
                mean_norm=mean.norm(dim=1).numpy())


# --------------------------------------------------------------------------------------------
# subcommands
# --------------------------------------------------------------------------------------------

def cmd_check(out):
    """Oracle check: this file's torch gradient against C4's numpy float64 BPTT, same noise.

    The noise is drawn with C4.forward's own generator order (numpy) and handed to both. Pass bar
    (pre-registered): per-unit relative difference of the incoming-row gradient norm < 1e-3 at the
    median and < 1e-2 at the 99th percentile over units with g_in >= 1e-3 q95; also prints the time
    of one torch forward + backward at the training batch size.
    """
    res = {}
    for name in ("FF-B", "CDDM-1"):
        cfg = load_cfg(name)
        folder, _ = weights_dir(name, 100000)
        P = C4.load_params(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0])
        u, y, mask = C4.batch(cfg, 777)
        u, y = u[:, :, :128], y[:, :, :128]                # the oracle is slow; 128 trials suffice
        sig = (float(cfg.model.sigma_rec), float(cfg.model.sigma_inp), float(cfg.model.sigma_out))
        rng = np.random.default_rng(5)
        H, U, no, outn = C4.forward(P, u, rng=rng, sig=sig)
        G = C4.backward(P, H, U, no, outn, y, mask, 0.0)
        g_np = np.sqrt((G["dW_rec"] ** 2).sum(1) + (G["dW_inp"] ** 2).sum(1))
        # the same noise for torch: input noise = U - u, recurrent noise recovered from C4's draws
        rng = np.random.default_rng(5)
        a = P["alpha"]
        kk = np.sqrt(2 / a)
        n_in, T, B = u.shape
        N = P["W_rec"].shape[0]
        ni = kk * sig[1] * rng.standard_normal((T, n_in, B))
        nout = kk * sig[2] * rng.standard_normal((N, B))
        nr = np.zeros((N, T, B))
        for t in range(1, T):
            nr[:, t - 1] = kk * sig[0] * rng.standard_normal((N, B))
        noise = tuple(torch.as_tensor(z, dtype=torch.float64) for z in (ni.transpose(1, 0, 2), nr, nout))
        W = load_W(folder, torch.float64)
        to = lambda z: torch.as_tensor(z, dtype=torch.float64)
        L, g = gradient(W, to(u), to(y), torch.as_tensor(np.asarray(mask)), noise)
        g_t = rows(g).norm(dim=1).numpy()
        ok = g_np >= 1e-3 * np.quantile(g_np, 0.95)
        rel = np.abs(g_t - g_np)[ok] / g_np[ok]
        # float32, the precision used for every measurement below
        W32 = load_W(folder)
        L32, g32 = gradient(W32, to(u).float(), to(y).float(), torch.as_tensor(np.asarray(mask)),
                            tuple(z.float() for z in noise))
        rel32 = np.abs(rows(g32).norm(dim=1).numpy() - g_np)[ok] / g_np[ok]
        # timing at the training batch size
        ub, yb, mb = batch(cfg, 778)
        nz = draw_noise(cfg, (N, ub.shape[0], ub.shape[1], ub.shape[2]), 1)
        t0 = time.time()
        gradient(W32, ub, yb, mb, nz)
        dt = time.time() - t0
        res[name] = dict(loss_np=G["loss"], loss_t=L, loss_t32=L32, med64=float(np.median(rel)),
                         q99_64=float(np.quantile(rel, 0.99)), med32=float(np.median(rel32)),
                         q99_32=float(np.quantile(rel32, 0.99)), n=int(ok.sum()), sec_per_grad=dt,
                         B=int(ub.shape[2]))
        print(name, res[name], flush=True)
    json.dump(res, open(os.path.join(out, "check.json"), "w"), indent=1)


def block_slope(it, L, lo, hi, nblk=10):
    """Slope of log(loss) on iteration over block medians of a window (T1).

    Block MEDIANS, not the pre-registered block means: single-iteration loss spikes (up to 10^5 times
    the floor on some runs) dominate a 5,000-iteration mean. The mean-based slope is kept beside it.

    Args:
        it, L: per-iteration series; lo, hi: window [lo, hi); nblk: blocks.
    Returns:
        dict pct100k and t (from block medians), pct100k_mean and t_mean (from block means), cv
        (per-iteration coefficient of variation, iterations within 3x the window median), mean,
        max_over_median (largest single iteration over the window median), blocks (medians).
    """
    m = (it > lo) & (it <= hi)
    x, v = it[m], L[m]
    edges = np.linspace(lo, hi, nblk + 1)
    bx = np.array([x[(x > a) & (x <= b)].mean() for a, b in zip(edges[:-1], edges[1:])])
    out = {}
    for tag, fn in (("", np.median), ("_mean", np.mean)):
        ly = np.log(np.array([fn(v[(x > a) & (x <= b)]) for a, b in zip(edges[:-1], edges[1:])]))
        A = np.vstack([bx, np.ones_like(bx)]).T
        coef = np.linalg.lstsq(A, ly, rcond=None)[0]
        resid = ly - A @ coef
        se = np.sqrt((resid ** 2).sum() / (nblk - 2) / ((bx - bx.mean()) ** 2).sum())
        out["pct100k" + tag] = float(100 * (np.exp(coef[0] * 1e5) - 1))
        out["t" + tag] = float(coef[0] / se)
        if not tag:
            out["blocks"] = np.exp(ly).tolist()
    med = np.median(v)
    ok = v < 3 * med
    out.update(cv=float(v[ok].std() / v[ok].mean()), mean=float(v.mean()), max_over_median=float(v.max() / med))
    return out


def load_trace(name):
    """The participation trace of the chain's last run (it spans the whole chain)."""
    d = chain_runs(name)[-1]["dir"]
    return pickle.load(open(glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))[0], "rb"))


def windows_of(name):
    """T1/T3 windows of a chain: its last 50k, the 150-200k window of a 300k chain, and 50-100k."""
    end = chain_end(name)
    w = {"last": (end - WINDOW, end), "matched": (50000, 100000)}
    if end == 300000:
        w["ff200"] = (150000, 200000)
    return w


def cmd_loss(out):
    """T1: noisy training loss slope per chain and window; also the trace's noise-free training loss."""
    res = {}
    for name in CHAINS:
        it, L = common.noisy_loss_series(chain_runs(name)[-1]["dir"])
        tr = load_trace(name)
        ci, cl = common.series(tr, "loss_clean_train")
        r = {"end": chain_end(name)}
        for w, (lo, hi) in windows_of(name).items():
            r[w] = dict(noisy=block_slope(it, L, lo, hi), clean=block_slope(ci + 1, cl, lo, hi),
                        lo=lo, hi=hi)
        # smoothed curve for the figure: 1000-iteration block medians
        nb = len(L) // 1000
        r["curve_it"] = (np.arange(nb) * 1000 + 500).tolist()
        r["curve"] = np.median(L[:nb * 1000].reshape(nb, 1000), 1).tolist()
        res[name] = r
        print(name, {w: (round(r[w]["noisy"]["pct100k"], 2), round(r[w]["noisy"]["t"], 1),
                         round(r[w]["clean"]["pct100k"], 2), round(r[w]["clean"]["t"], 1))
                     for w in windows_of(name)}, flush=True)
    json.dump(res, open(os.path.join(out, "loss.json"), "w"))


def row_dir(folder):
    """Incoming rows [W_rec | W_inp] of a weights folder, each scaled to unit norm, (N, N + n_in)."""
    P = C4.load_params(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0])
    Wr = np.hstack([P["W_rec"], P["W_inp"]])
    return Wr / np.maximum(np.linalg.norm(Wr, axis=1, keepdims=True), 1e-30), Wr


def cmd_drift(name, out):
    """T3 for one chain: trace drift exponent per window, and per-unit consecutive-displacement
    cosines rho_i(L) of the row directions from the snapshots, with unit classes at window start."""
    res = {}
    tr = load_trace(name)
    al_it, al = {}, {}
    for Wn in ("W_rec", "W_inp"):
        al_it[Wn], al[Wn] = common.drift_alpha(tr, Wn)
    cfg = load_cfg(name)
    u, _, _ = batch(cfg, REF_SEED)
    for w, (lo, hi) in windows_of(name).items():
        r = {}
        for Wn in ("W_rec", "W_inp"):
            m = (al_it[Wn] > lo) & (al_it[Wn] <= hi)
            r[f"alpha_{Wn}"] = float(np.median(al[Wn][m])) if m.any() else float("nan")
        its = list(range(lo, hi + 1, 500))
        D, Raw = {}, {}
        for t in its:
            D[t], Raw[t] = row_dir(weights_dir(name, t)[0])
        f0, _ = weights_dir(name, lo)
        ref = reference_stats(load_W(f0), u)
        act = over(ref["p"])
        cls = dict(active=act, silent_on=~act & (ref["n_on"] > 0), never_on=ref["n_on"] == 0)
        for L in (500, 2500, 10000):
            pts = its[::L // 500]
            d = [D[b] - D[a] for a, b in zip(pts[:-1], pts[1:])]
            cs = []
            for d1, d2 in zip(d[:-1], d[1:]):
                cs.append((d1 * d2).sum(1) / (np.linalg.norm(d1, axis=1) * np.linalg.norm(d2, axis=1) + 1e-30))
            r[f"rho_{L}"] = np.mean(cs, 0)
            # whole matrix, raw weights (the trace's drift is on raw weights)
            dr = [(Raw[b] - Raw[a]).ravel() for a, b in zip(pts[:-1], pts[1:])]
            r[f"rho_net_{L}"] = float(np.mean([np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
                                               for a, b in zip(dr[:-1], dr[1:])]))
        r.update({f"cls_{k}": v for k, v in cls.items()})
        r["p"], r["n_on"] = ref["p"], ref["n_on"]
        r["rel_change"] = np.linalg.norm(Raw[hi] - Raw[lo], axis=1) / np.linalg.norm(Raw[lo], axis=1)
        res[w] = r
        msg = " ".join(f"{k}:n={v.sum()} rho10k={np.median(r['rho_10000'][v]):+.2f} "
                       f"rho2.5k={np.median(r['rho_2500'][v]):+.2f} rho500={np.median(r['rho_500'][v]):+.2f}"
                       for k, v in cls.items() if v.any())
        print(f"{name} {w} {lo}-{hi} alpha rec {r['alpha_W_rec']:.2f} inp {r['alpha_W_inp']:.2f} "
              f"rho_net(500/2.5k/10k) {r['rho_net_500']:+.2f}/{r['rho_net_2500']:+.2f}/{r['rho_net_10000']:+.2f} | {msg}",
              flush=True)
    pickle.dump(res, open(os.path.join(out, f"drift_{name}.pkl"), "wb"))


def cmd_grad(name, it, K, out):
    """T2 at one read point: classes, reference-batch reach, and the K-batch signal share."""
    t0 = time.time()
    cfg = load_cfg(name)
    folder, _ = weights_dir(name, it)
    W = load_W(folder)
    u, y, mask = batch(cfg, REF_SEED)
    ref = reference_stats(W, u)
    L_ref, g = gradient(W, u, y, mask)
    g_in_ref = rows(g).norm(dim=1).numpy()
    cls = classes(ref, g_in_ref)
    del g
    print(f"{name}@{it}: ref loss {L_ref:.5f}; " + ", ".join(f"{k} {v.sum()}" for k, v in cls.items())
          + f" ({time.time() - t0:.0f}s)", flush=True)
    s = signal_share(cfg, W, ref["X"], K, seed0=200000)
    res = dict(it=it, L_ref=L_ref, g_in_ref=g_in_ref, p=ref["p"], n_on=ref["n_on"],
               w_norm=torch.cat([W["W_rec"], W["W_inp"]], 1).norm(dim=1).numpy(), **s,
               **{f"cls_{k}": v for k, v in cls.items()})
    np.savez(os.path.join(out, f"grad_{name}_{it}.npz"), **res)
    summ(name, it, res)
    print(f"  done {time.time() - t0:.0f}s", flush=True)


def summ(name, it, r):
    """Print per-class medians of a T2 result dict."""
    print(f"{name}@{it} F_net {float(r['F_net']):.3f} loss {np.mean(r['losses']):.5f}", flush=True)
    for k in ("active", "silent_reached", "silent_on", "never_on"):
        m = r[f"cls_{k}"]
        if not m.any():
            continue
        nul = np.quantile(r["c_null"][:, m], 0.95)
        print(f"  {k:15s} n={m.sum():4d} F med {np.median(r['F'][m]):+.3f} c med {np.median(r['c'][m]):+.3f} "
              f"(null q95 {nul:+.3f}, share c>null {np.mean(r['c'][m] > nul):.2f}) |t_drive|>2 "
              f"{np.mean(np.abs(r['t_drive'][m]) > 2):.2f} (neg {np.mean(r['t_drive'][m] < -2):.2f})", flush=True)


def build_net(cfg, folder):
    """The production RNN_torch of a run's config with the weights of `folder` loaded."""
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    del task_cfg._target_
    rnn_cfg = OmegaConf.create(cfg.model)
    args = filter_kwargs(import_any(rnn_cfg._target_), OmegaConf.merge(rnn_cfg, task_cfg))
    args.seed = 0
    net = hydra.utils.instantiate(args)
    load_warm_start(net, folder)
    return net


def eval_set(cfg, n=8, seed0=900000):
    """The fixed evaluation set of T4: n seeded batches, each with its noise-generator seed."""
    return [(batch(cfg, seed0 + j), seed0 + j) for j in range(n)]


def evaluate(net, ev):
    """Mean noisy loss (seeded production noise) and mean noise-free loss over the evaluation set."""
    noisy, clean = [], []
    with torch.no_grad():
        for (u, y, m), s in ev:
            net.random_generator.manual_seed(s)
            _, o = net(u, w_noise=True)
            noisy.append(float(((o[:, m, :] - y[:, m, :]) ** 2).mean()))
            _, o = net(u, w_noise=False)
            clean.append(float(((o[:, m, :] - y[:, m, :]) ** 2).mean()))
    return float(np.mean(noisy)), float(np.mean(clean))


def cmd_finetune(name, it, cond, iters, out, K):
    """T4: continue training a run end under one condition, on seeded batches; evaluate on a fixed
    seeded set; then re-read the C4 reach (and the T2 signal share) on the fine-tuned weights.

    Args:
        name: chain; it: a run-end iteration (holds AdamState); cond: "control" (Adam, run lr),
        "anneal" (Adam, lr/10), "sgd" (no momentum, lr = (lr/10)/median sqrt(v_hat) over W_rec);
        iters: fine-tune length; out: output folder; K: batches for the T2 re-read (0 = skip).
    """
    t0 = time.time()
    cfg = load_cfg(name)
    folder, run = weights_dir(name, it)
    assert folder == run["dir"], "fine-tunes start from a run end (with its Adam state)"
    net = build_net(cfg, folder)
    params = {"W_rec": net.W_rec, "W_inp": net.W_inp, "W_out": net.W_out}
    assert {n for n, _ in net.named_parameters()} == set(params)
    lr = float(cfg.trainer.lr) * (100.0 / float(cfg.model.N)) ** float(cfg.trainer.lr_scale_exp)
    wd = float(cfg.trainer.weight_decay)
    st = torch.load(glob.glob(os.path.join(folder, "*AdamState*.pt"))[0], map_location="cpu", weights_only=False)
    # map the saved moments to parameters by shape (W_rec (N,N), W_inp (N,n_in), W_out (n_out,N))
    order = [next(k for k, p in params.items() if tuple(p.shape) == tuple(s["exp_avg"].shape))
             for _, s in sorted(st["state"].items())]
    plist = [params[k] for k in order]
    if cond in ("control", "anneal"):
        opt = torch.optim.Adam(plist, lr=lr, weight_decay=wd)
        opt.load_state_dict(st)
        for gp in opt.param_groups:
            gp["lr"] = lr if cond == "control" else lr / 10
        lr_used = opt.param_groups[0]["lr"]
    else:
        s_rec = st["state"][order.index("W_rec")]
        b2 = st["param_groups"][0]["betas"][1]
        vhat = s_rec["exp_avg_sq"] / (1 - b2 ** float(s_rec["step"]))
        lr_used = (lr / 10) / float(vhat.sqrt().median())
        opt = torch.optim.SGD(plist, lr=lr_used, momentum=0.0, weight_decay=wd)
    ev = eval_set(cfg)
    log = {"lr": lr_used, "cond": cond, "it": [], "noisy": [], "clean": [], "train": []}
    marks = {0, 250, 500, 1000, 1500, 2000, iters // 2, iters}
    max_norm = float(cfg.trainer.max_grad_norm)
    mask = torch.as_tensor(np.asarray(get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)), dtype=torch.long)
    for k in range(iters + 1):
        if k in marks:
            n_, c_ = evaluate(net, ev)
            log["it"].append(k), log["noisy"].append(n_), log["clean"].append(c_)
            print(f"{name}@{it} {cond} k={k} eval noisy {n_:.6f} clean {c_:.6f} "
                  f"train(last100) {np.mean(log['train'][-100:]) if log['train'] else float('nan'):.6f} "
                  f"({time.time() - t0:.0f}s)", flush=True)
        if k == iters:
            break
        u, y, _ = batch(cfg, 700000 + k)
        net.random_generator.manual_seed(700000 + k)
        _, o = net(u, w_noise=True)
        loss = ((o[:, mask, :] - y[:, mask, :]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(plist, max_norm=max_norm)
        opt.step()
        log["train"].append(float(loss))
    tag = f"ft_{name}_{it}_{cond}"
    json.dump(log, open(os.path.join(out, tag + ".json"), "w"))
    # the C4 reach re-read on the fine-tuned weights, same reference batch as T2
    W = {k: v.detach().clone() for k, v in params.items()}
    W.update(y0=net.y_init.detach().reshape(-1).clone(), b=torch.zeros(net.N), alpha=float(net.alpha))
    u, y, m = batch(cfg, REF_SEED)
    ref = reference_stats(W, u)
    L_ref, g = gradient(W, u, y, m)
    g_in_ref = rows(g).norm(dim=1).numpy()
    cls = classes(ref, g_in_ref)
    res = dict(it=it, L_ref=L_ref, g_in_ref=g_in_ref, p=ref["p"], n_on=ref["n_on"],
               **{f"cls_{k}": v for k, v in cls.items()})
    if K:
        res.update(signal_share(cfg, W, ref["X"], K, seed0=200000))
        summ(name + "+" + cond, it, res)
    np.savez(os.path.join(out, tag + ".npz"), **res)
    print(f"{tag}: after: " + ", ".join(f"{k} {v.sum()}" for k, v in cls.items()) + f" ({time.time() - t0:.0f}s)",
          flush=True)


TASK_COLOR = {"FF": "#c0392b", "CDDM": "#2471a3", "DMTS": "#7d8a2e"}


def cmd_report(out):
    """Print every pre-registered quantity and draw the figure, from the outputs in `out`.

    Panels: A noisy training loss (1000-iteration medians) over the chain's own final value;
    B per-unit consecutive-displacement cosine rho(L) of incoming-row directions, medians by class,
    in each chain's last 50k; C gradient signal share F, medians by class, at each read point;
    D fine-tunes: evaluation noisy loss over its start value, per condition.
    """
    print("== check", json.load(open(os.path.join(out, "check.json"))))
    # ---- T1
    loss = json.load(open(os.path.join(out, "loss.json")))
    print("\n== T1 noisy loss slope from block medians, % per 100k (t) [block means, largest spike] | noise-free training-batch loss, % per 100k (t) | per-iter CV")
    for name, r in loss.items():
        print(f"{name:7s} " + "  ".join(
            f"{w} {r[w]['lo'] // 1000}-{r[w]['hi'] // 1000}k: {r[w]['noisy']['pct100k']:+.2f} ({r[w]['noisy']['t']:+.1f}) "
            f"[means {r[w]['noisy']['pct100k_mean']:+.1f}, max spike {r[w]['noisy']['max_over_median']:.0f}x] | "
            f"{r[w]['clean']['pct100k']:+.1f} ({r[w]['clean']['t']:+.1f}) | cv {r[w]['noisy']['cv']:.2f}"
            for w in ("last", "ff200", "matched") if w in r))
    # ---- T3
    print("\n== T3 drift: alpha (trace) and per-unit rho(L) medians by class [n]")
    drift = {n: pickle.load(open(f, "rb")) for n in CHAINS
             for f in glob.glob(os.path.join(out, f"drift_{n}.pkl"))}
    for name, D in drift.items():
        for w, r in D.items():
            cl = " ".join(f"{k}[{r['cls_' + k].sum()}] " + "/".join(f"{np.median(r[f'rho_{L}'][r['cls_' + k]]):+.2f}"
                                                                   for L in (500, 2500, 10000))
                          for k in ("active", "silent_on", "never_on") if r["cls_" + k].any())
            print(f"{name:7s} {w:8s} alpha rec {r['alpha_W_rec']:.2f} inp {r['alpha_W_inp']:.2f} | net "
                  + "/".join(f"{r[f'rho_net_{L}']:+.2f}" for L in (500, 2500, 10000)) + " | " + cl)
    # ---- T2
    print("\n== T2 gradient signal share")
    grads = {}
    for f in sorted(glob.glob(os.path.join(out, "grad_*.npz"))):
        _, name, it = os.path.basename(f)[:-4].split("_")
        grads[(name, int(it))] = r = dict(np.load(f))
        summ(name, int(it), r)
    # ---- T4
    print("\n== T4 fine-tunes: eval noisy loss L(k)/L(0); G; late gain (second half); reach after")
    fts = {}
    for f in sorted(glob.glob(os.path.join(out, "ft_*.json"))):
        _, name, it, cond = os.path.basename(f)[:-5].split("_")
        lg = json.load(open(f))
        fts[(name, int(it), cond)] = lg
        Lx = dict(zip(lg["it"], lg["noisy"]))
        Cx = dict(zip(lg["it"], lg["clean"]))
        n = max(lg["it"])
        G = (Lx[0] - Lx[n]) / Lx[0]
        late = (Lx[n // 2] - Lx[n]) / Lx[0] if n // 2 in Lx else float("nan")
        g0 = grads.get((name, int(it)))
        before = f"{g0['cls_silent_reached'].sum()}" if g0 is not None else "?"
        # a fine-tune stopped early (log-parsed json, no npz) has no reach re-read
        after = "not re-read"
        if os.path.exists(f[:-5] + ".npz"):
            z = np.load(f[:-5] + ".npz")
            after = f"active {z['cls_active'].sum()} silent-reached {z['cls_silent_reached'].sum()}"
        print(f"{name:7s}@{int(it) // 1000}k {cond:8s} lr {lg['lr']:.2e}: " +
              " ".join(f"{k}:{Lx[k] / Lx[0]:.4f}" for k in sorted(Lx)) +
              f" G {100 * G:+.2f}% late {100 * late:+.2f}% | clean {Cx[n] / Cx[0]:.3f} | {after} (before {before})")
    # ---- figure
    fig, ax = plt.subplots(2, 2, figsize=(11, 8.5))
    a = ax[0, 0]
    for name, r in loss.items():
        if task_of(name) == "DMTS":          # dropped from the reduced plan; its loss spikes 2000-fold
            continue
        c = np.array(r["curve"])
        nb = len(c) // 5                     # 5000-iteration medians of the 1000-iteration medians
        c5 = np.median(c[:nb * 5].reshape(nb, 5), 1)
        x5 = np.array(r["curve_it"])[:nb * 5].reshape(nb, 5).mean(1)
        a.plot(x5 / 1000, c5 / np.median(c5[-2:]), color=TASK_COLOR[task_of(name)], lw=1,
               label=task_of(name) if name.endswith("1") or name == "FF-A" else None)
    a.set_yscale("log")
    a.set_ylim(0.95, 2)
    a.set_xlabel("iteration (thousands)")
    a.set_ylabel("noisy training loss / its final value")
    a.set_title("A  noisy training loss (5000-iteration medians)", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=8)
    a = ax[0, 1]
    Ls = np.array([500, 2500, 10000])
    for name, D in drift.items():
        if task_of(name) == "DMTS":
            continue
        r = D["last"]
        for k, ls in (("active", "-"), ("silent_on", "--"), ("never_on", ":")):
            m = r["cls_" + k]
            if m.sum() < 10:
                continue
            a.plot(Ls, [np.median(r[f"rho_{L}"][m]) for L in Ls], ls, color=TASK_COLOR[task_of(name)], marker="o",
                   ms=3, lw=1, label=f"{task_of(name)} {k}" if name.endswith("1") or name == "FF-A" else None)
    a.axhline(0, color="k", lw=0.5)
    a.set_xscale("log")
    a.set_xlabel("lag L (iterations)")
    a.set_ylabel("median cos(consecutive row displacements)")
    a.set_title("B  weight motion in each chain's last 50k", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=7, ncol=2)
    a = ax[1, 0]
    for (name, it), r in grads.items():
        t = task_of(name)
        for k, mk in (("active", "o"), ("silent_reached", "^")):
            m = r["cls_" + k]
            if m.sum() < 5:
                continue
            a.scatter(it / 1000, np.median(r["F"][m]), color=TASK_COLOR[t], marker=mk, s=18,
                      facecolor="none" if k == "silent_reached" else TASK_COLOR[t],
                      label=f"{t} {k}" if (name.endswith("1") or name == "FF-A") and it == GRAD_READS[t][0] else None)
    a.axhline(0, color="k", lw=0.5)
    a.set_xlabel("read point (thousand iterations)")
    a.set_ylabel("median gradient signal share F")
    a.set_title("C  consistent share of the per-batch gradient on incoming rows", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=7)
    a = ax[1, 1]
    for (name, it, cond), lg in fts.items():
        ls = {"control": "-", "anneal": "--", "sgd": ":"}[cond]
        a.plot(lg["it"], np.array(lg["noisy"]) / lg["noisy"][0], ls, color=TASK_COLOR[task_of(name)], lw=1,
               label=f"{task_of(name)} {cond}" if (name.endswith("1") or name == "FF-A") and it < 300000 else None)
    a.set_xlabel("fine-tune iteration")
    a.set_ylabel("evaluation noisy loss / start")
    a.set_title("D  fine-tunes from each run's end", loc="left", fontsize=10)
    a.legend(frameon=False, fontsize=7, ncol=3)
    fig.tight_layout()
    fn = os.path.join(common.IMG_DIR, "c4b_convergence")
    fig.savefig(fn + ".pdf")
    fig.savefig(fn + ".svg")
    print("wrote", fn + ".pdf")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cmd", choices=["check", "loss", "drift", "grad", "finetune", "report"])
    ap.add_argument("--chain")
    ap.add_argument("--it", type=int)
    ap.add_argument("--cond", choices=["control", "anneal", "sgd"])
    ap.add_argument("--iters", type=int, default=2000)
    ap.add_argument("--K", type=int, default=32)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.threads:
        torch.set_num_threads(a.threads)
    os.makedirs(a.out, exist_ok=True)
    if a.cmd == "check":
        cmd_check(a.out)
    elif a.cmd == "loss":
        cmd_loss(a.out)
    elif a.cmd == "drift":
        cmd_drift(a.chain, a.out)
    elif a.cmd == "grad":
        cmd_grad(a.chain, a.it, a.K, a.out)
    elif a.cmd == "finetune":
        cmd_finetune(a.chain, a.it, a.cond, a.iters, a.out, a.K)
    elif a.cmd == "report":
        cmd_report(a.out)
