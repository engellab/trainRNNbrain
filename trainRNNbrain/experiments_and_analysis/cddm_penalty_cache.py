#!/usr/bin/env python3
"""
Score the four CDDM penalty arms at N = 1000 on every measure the closing slides need.

ONE PASS PER NETWORK gives all of them, so the figures cannot disagree about which networks they
describe: noise-free r2, active units, participation-ratio dimensionality, per-unit TEMPORAL
participation ratio, and the recurrent-weight log-magnitude histogram.

WHY TEMPORAL PR. frm drives every unit above the silence bar - on CDDM it puts all 1000 units over
it at every size - but a unit that fires in a brief transient and is quiet the rest of the trial is
not doing the same job as one that is active throughout. tPR/n measures that: 1 for a unit at a
constant rate, near 0 for a burst unit (flipflop_temporal_pr.temporal_pr, the same formula the
flip-flop supplementary uses). The question these slides ask is whether frm buys units that are
genuinely engaged or merely nonzero, and whether rws changes it.

LIVENESS IS THE SCALE-FREE RULE, not the flip-flop's absolute bar, so the counts here match every
other active-unit number in this project. temporal_pr's own live mask is therefore discarded.

Usage:  python cddm_penalty_cache.py [--refresh]
Output: data/cddm_penalty_cache.npz
"""

import argparse
import glob
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..")))
from common import DATA_DIR, SILENT_REL
# ⚠️ FIGURE 2's participation_ratio, not common's. They are different measures sharing a name:
# common's takes a (N,) participation VECTOR and reports how evenly activity is spread, while
# this one takes the (N, samples) rate matrix and reports the covariance's effective rank -
# the dimensionality panel (d) plots. Importing common's and handing it a 2-D array gave a PR
# of 15,931,419 over 450,000 pooled numbers.
from f2_remedies_cache import participation_ratio
from flipflop_temporal_pr import temporal_pr
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask

OUT = "data/cddm_penalty_cache.npz"
N_UNITS = 1000
# the arm -> cell glob map. The control is the unpenalised drift sweep at the same size; the three
# penalty arms are the local-only CDDM_std_g0_penalties sweep (it does not exist on Della).
ARMS = {
    "control": f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N={N_UNITS}_iters=*",
    "frm": f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N={N_UNITS}_pen=frm",
    "rws": f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N={N_UNITS}_pen=rws",
    "both": f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N={N_UNITS}_pen=both",
}
LOG_BINS = np.linspace(-12.0, 4.0, 321)      # same edges as the Figure 2 cache
ZERO_TOL = 1e-12


def build(net_dir):
    """Rebuild one network from its saved parameters and config.

    Args:
        net_dir: a run folder holding `*LastParams*.npz` and `*_config.yaml`.
    Returns:
        (rnn, cfg) or (None, None) if the folder is incomplete.
    """
    npz = glob.glob(os.path.join(net_dir, "*LastParams*.npz"))
    cfgs = glob.glob(os.path.join(net_dir, "*_config.yaml"))
    if not npz or not cfgs:
        return None, None
    raw = dict(np.load(npz[0], allow_pickle=True))
    d = {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == () else v)
         for k, v in raw.items()}
    cfg = OmegaConf.load(cfgs[0])
    m = cfg.model
    # ⚠️ ACTIVATION ARGS COME FROM THE CONFIG, not the npz. The drift sweep predates storable_ and
    # saved activation_args as the dict's KEYS, so dict(d["activation_args"]) raises; the config
    # carries the real mapping on every sweep alike.
    act = OmegaConf.to_container(m.activation_args, resolve=True)
    rnn = RNN_torch(N=int(m.N), activation_args=act,
                    equation_type=str(m.equation_type), dale=bool(m.get("dale", False)),
                    io_nonnegativity=bool(m.get("io_nonnegativity", False)),
                    self_connections=bool(m.get("self_connections", False)),
                    bias_range=list(m.bias_range), gamma=float(m.gamma), dt=float(m.dt),
                    tau=float(m.tau), sigma_rec=float(m.sigma_rec), sigma_inp=float(m.sigma_inp),
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
    return rnn, cfg


def measure(net_dir):
    """Every measure for one network, from a single noise-free pass.

    Args:
        net_dir: a run folder.
    Returns:
        dict with n_active, dims, r2, tpr (per live unit, already divided by the sample count) and
        w_hist, or None if the network cannot be rebuilt.
    """
    rnn, cfg = build(net_dir)
    if rnn is None:
        return None
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    bi, bt, _ = task.get_batch()
    bi_t = torch.tensor(np.asarray(bi), dtype=torch.float32)
    bt_t = torch.tensor(np.asarray(bt), dtype=torch.float32)
    keep = (float(rnn.sigma_rec), float(rnn.sigma_inp))
    rnn.sigma_rec = rnn.sigma_inp = 0.0
    with torch.no_grad():
        states, out = rnn(bi_t, w_noise=False)
        r2 = float(Trainer.r2_score(out, bt_t, mask))
    rnn.sigma_rec, rnn.sigma_inp = keep
    rates = torch.relu(states).numpy()                       # (N, T, B), noise-free
    flat = rates.reshape(rates.shape[0], -1)

    p = flat.std(axis=1) + np.quantile(np.abs(flat), 0.9, axis=1)
    live = p >= SILENT_REL * np.quantile(p, 0.95)
    # temporal_pr's own mask is the flip-flop's absolute bar; the scale-free mask is used instead so
    # these counts line up with every other active-unit number in the project
    tpr, _flipflop_live, n_samples = temporal_pr(rates)
    W = np.asarray(rnn.W_rec.detach().numpy(), dtype=np.float64)
    a = np.abs(W[np.abs(W) > ZERO_TOL])
    hist, _ = np.histogram(np.log10(a), bins=LOG_BINS)
    return dict(n_active=int(live.sum()), r2=r2,
                dims=float(participation_ratio(flat[live])) if live.sum() > 1 else float("nan"),
                tpr=np.asarray(tpr[live] / n_samples, dtype=np.float32),
                w_hist=hist.astype(np.float64), q50_q95=float(np.quantile(p, 0.5) /
                                                              max(np.quantile(p, 0.95), 1e-12)))


def main(refresh=False):
    """Build the cache. Returns its path, or None if no network could be scored."""
    if os.path.exists(OUT) and not refresh:
        print(f"{OUT} exists; pass --refresh to recompute")
        return OUT
    store, rows = {}, []
    for arm, pat in ARMS.items():
        folders = [f for cell in sorted(glob.glob(pat))
                   for f in sorted(glob.glob(os.path.join(cell, "*/")))
                   if os.path.basename(f.rstrip("/")).split("_")[0] != "nan"]
        for i, f in enumerate(folders):
            r = measure(f)
            if r is None:
                continue
            key = f"{arm}|{i}"
            store[f"{key}|tpr"] = r["tpr"]
            store[f"{key}|w_hist"] = r["w_hist"]
            rows.append((arm, r["n_active"], r["r2"], r["dims"], r["q50_q95"], i))
            print(f"  {arm:8s} seed {i}: active {r['n_active']:4d}  r2 {r['r2']:.4f}  "
                  f"dims {r['dims']:6.2f}  median tPR/n {np.median(r['tpr']):.3f}", flush=True)
    if not rows:
        print("no networks scored")
        return None
    store["arm"] = np.array([r[0] for r in rows])
    store["n_active"] = np.array([r[1] for r in rows], float)
    store["r2"] = np.array([r[2] for r in rows], float)
    store["dims"] = np.array([r[3] for r in rows], float)
    store["q50_q95"] = np.array([r[4] for r in rows], float)
    store["seed"] = np.array([r[5] for r in rows], int)
    store["log_bins"] = LOG_BINS
    np.savez_compressed(OUT, **store)
    print(f"\nwrote {OUT}  ({len(rows)} networks)")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true", help="recompute even if the cache exists")
    main(**vars(ap.parse_args()))
