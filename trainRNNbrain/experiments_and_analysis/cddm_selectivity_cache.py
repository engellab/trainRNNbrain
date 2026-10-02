#!/usr/bin/env python3
"""
Selectivity-configuration coordinates for the CDDM networks that HAVE the canonical structure.

WHY NOT THE SWEEP THE OTHER CLOSING SLIDES USE. Slides 28-31 read CDDM_std_g0_penalties, the only
CDDM penalty sweep with four network sizes - but it trains 200,000 iterations, and by then the
selectivity configuration has collapsed to three arms. The 30,000-iteration sweeps (CDDM_std_g0,
CDDM_ptrack_g0) still show the four-armed configuration, and they are also the ones carrying the
animated_selectivity movies. Measured across seeds, `both` at 30k puts its clustering elbow at k = 4
in three of five seeds and k = 5 in the other two; the 200k networks sit at k = 3 with a 19-fold
inertia drop there and nothing beyond.

So this cache reads CDDM_std_g0 at N = 1000, 30,000 iterations: control, frm and frm+rws, every seed,
and stores each unit's coordinates in the top three principal components of its own response.

Usage:  python cddm_selectivity_cache.py [--refresh]
Output: data/cddm_selectivity_cache.npz
"""

import argparse
import glob
import json
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
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import filter_kwargs, unjsonify

OUT = "data/cddm_selectivity_cache.npz"
ROOT = f"{DATA_DIR}/CDDM_std_g0"
ARMS = {"control": "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
        "frm": "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0.2",
        "both": "EqType=h_N=1000_LmbdRWS=0.05_LmbdFR=0.2"}


def build_any(net_dir):
    """Rebuild a network whose parameters are stored as npz OR json.

    This sweep predates the npz format, so the npz-only loaders elsewhere find nothing here and
    return no networks at all, silently.

    Args:
        net_dir: the run folder.
    Returns:
        (rnn, cfg), or (None, None) if neither format is present.
    """
    cfgs = glob.glob(os.path.join(net_dir, "*_config.yaml"))
    if not cfgs:
        return None, None
    cfg = OmegaConf.load(cfgs[0])
    npz = glob.glob(os.path.join(net_dir, "*LastParams*.npz"))
    jsn = glob.glob(os.path.join(net_dir, "*LastParams*.json"))
    if npz:
        raw = dict(np.load(npz[0], allow_pickle=True))
        d = {k: (v.item() if isinstance(v, np.ndarray) and v.dtype == object and v.shape == ()
                 else v) for k, v in raw.items()}
    elif jsn:
        d = unjsonify(json.load(open(jsn[0])))
    else:
        return None, None
    m = cfg.model
    d["activation_args"] = OmegaConf.to_container(m.activation_args, resolve=True)
    d.pop("equation_type", None)
    rnn = RNN_torch(**filter_kwargs(RNN_torch, d), equation_type=str(m.equation_type),
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


def coords(net_dir):
    """Top-three PC coordinates of the live units of one network, from a noise-free pass.

    Args:
        net_dir: the run folder.
    Returns:
        (pcs (n_live, 3) float32, n_live, explained-variance of the top 4 PCs), or None.
    """
    rnn, cfg = build_any(net_dir)
    if rnn is None:
        return None
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    bi, _bt, _c = task.get_batch()
    rnn.sigma_rec = rnn.sigma_inp = 0.0
    with torch.no_grad():
        states, _ = rnn(torch.tensor(np.asarray(bi), dtype=torch.float32), w_noise=False)
    flat = torch.relu(states).numpy().reshape(int(rnn.N), -1).astype(np.float32)
    del states
    p = flat.std(1) + np.quantile(np.abs(flat), 0.9, axis=1)
    live = p >= SILENT_REL * np.quantile(p, 0.95)
    X = flat[live].astype(np.float64)
    Xc = X - X.mean(axis=0, keepdims=True)
    _u, sv, vt = np.linalg.svd(Xc, full_matrices=False)
    var = (sv ** 2 / (sv ** 2).sum())[:4]
    return (Xc @ vt[:3].T).astype(np.float32), int(live.sum()), var.astype(np.float32)


def main(refresh=False):
    """Write the cache. Returns its path, or None if nothing was scored."""
    if os.path.exists(OUT) and not refresh:
        print(f"{OUT} exists; pass --refresh to recompute")
        return OUT
    store = {}
    for arm, cell in ARMS.items():
        folders = [f for f in sorted(glob.glob(os.path.join(ROOT, cell, "*/")))
                   if os.path.basename(f.rstrip("/")).split("_")[0] != "nan"]
        for i, f in enumerate(folders):
            got = coords(f)
            if got is None:
                continue
            pcs, n_live, var = got
            store[f"{arm}|{i}|pcs"] = pcs
            store[f"{arm}|{i}|var"] = var
            print(f"  {arm:8s} seed {i}: {n_live:4d} live units, "
                  f"variance {np.round(var, 2)}", flush=True)
    if not store:
        print("nothing scored")
        return None
    np.savez_compressed(OUT, **store)
    print(f"\nwrote {OUT}")
    return OUT


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--refresh", action="store_true")
    main(**vars(ap.parse_args()))
