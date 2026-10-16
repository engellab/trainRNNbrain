#!/usr/bin/env python3
"""Dimensionality of one trained network's population activity, cached on disk.

THE MEASURE is the participation ratio of the noise-free rate covariance over the ACTIVE units:
(sum of eigenvalues)^2 / (sum of squared eigenvalues) -- the number of directions the population
actually uses, 1 if every unit does the same thing. It is computed over the active units only,
because a silent unit contributes a zero row to the covariance and would otherwise pull the ratio
toward 1 for reasons that have nothing to do with the computation.

⚠️ THE ARCHITECTURE COMES FROM THE CONFIG, NOT THE SAVED PARAMETERS. A `*LastParams*.npz` carries
W_rec, W_inp, W_out, bias, y_init, N, dt, tau, gamma and the masks -- and not `dale`,
`io_nonnegativity`, `self_connections`, `sigma_rec` or `sigma_inp`. Rebuilding from the npz alone
takes RNN_torch's defaults instead (Dale on, I/O non-negativity on, the self-connection diagonal
masked out), which is a different network: measured, that read r2 0.9135 on a flip-flop control
whose real score is 0.9466.

NOISE-FREE, like every other count and dimensionality in this project. Injected noise lifts every
unit onto a floor of about (dt/tau)*sigma_rec, which both inflates the active count and adds
isotropic variance to every direction of the covariance.

Results are cached in data/run_dims_cache.json, keyed by the parameter file's path, size and mtime,
so rebuilding a figure recomputes nothing and retraining a cell invalidates only that cell.

Usage:  python run_dims.py "data/trained_RNNs/CDDM_std_g0_metabolic/*/*"
"""
import glob
import json
import os
import sys

import hydra
import numpy as np
import torch
from omegaconf import OmegaConf

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO)
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import unjsonify
from common import active_mask

CACHE = os.path.join(REPO, "data", "run_dims_cache.json")
TRIALS = 128                 # cap on trials kept for the rate matrix, strided (never the first k:
                             # TaskCDDM enumerates its coherence grid in a fixed order, so the
                             # first 128 of 450 are one context out of two)


def participation_ratio(rates):
    """Covariance participation ratio of a rate matrix.

    Args:
        rates: (n_units, n_samples) float array of firing rates.
    Returns:
        float (sum ev)^2 / sum ev^2 of the unit-by-unit covariance, nan if fewer than two units.
    """
    if rates.shape[0] < 2:
        return float("nan")
    ev = np.linalg.eigvalsh(np.cov(rates))
    ev = ev[ev > 0]
    return float(ev.sum() ** 2 / np.square(ev).sum()) if len(ev) else float("nan")


def _key(net_dir):
    """Cache key for one run folder: its parameter file's path, size and mtime, or None.

    BOTH SAVE FORMATS. The older sweeps - the metabolic ladder among them - write
    `*LastParams*.json`; the later ones write npz. Looking only for npz returned nan for every
    metabolic cell, silently.
    """
    hits = (sorted(glob.glob(os.path.join(net_dir, "*LastParams*.npz")))
            or sorted(glob.glob(os.path.join(net_dir, "*LastParams*.json"))))
    if not hits:
        return None, None
    st = os.stat(hits[0])
    return hits[0], f"{os.path.relpath(hits[0], REPO)}|{st.st_size}|{int(st.st_mtime)}"


def measure(net_dir):
    """Dimensionality and active-unit count of one saved network, from a noise-free pass.

    Args:
        net_dir: a run folder holding `*LastParams*.npz` and `*_config.yaml`.
    Returns:
        (dims, n_active) floats, or (nan, nan) if the folder cannot be rebuilt.
    """
    npz, _ = _key(net_dir)
    cfgs = sorted(glob.glob(os.path.join(net_dir, "*_config.yaml")))
    if npz is None or not cfgs:
        return float("nan"), float("nan")
    if npz.endswith(".json"):
        with open(npz) as fh:
            d = {k: np.asarray(v) for k, v in unjsonify(json.load(fh)).items()}
    else:
        d = dict(np.load(npz, allow_pickle=True))
    cfg = OmegaConf.load(cfgs[0])
    m = cfg.model
    rnn = RNN_torch(N=int(m.N),
                    activation_args=OmegaConf.to_container(m.activation_args, resolve=True),
                    equation_type=str(m.equation_type), dale=bool(m.get("dale", False)),
                    io_nonnegativity=bool(m.get("io_nonnegativity", False)),
                    self_connections=bool(m.get("self_connections", False)),
                    bias_range=list(m.bias_range), gamma=float(m.gamma), dt=float(m.dt),
                    tau=float(m.tau), sigma_rec=0.0, sigma_inp=0.0, sigma_out=0.0,
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
        task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=m.dt))
        bi, _bt, _ = task.get_batch()
        bi = torch.tensor(np.asarray(bi), dtype=torch.float32)
        states, _out = rnn(bi, w_noise=False)
        rates = torch.relu(states).numpy()                    # (N, T, B), noise-free
    flat = rates.reshape(rates.shape[0], -1)
    p = flat.std(axis=1) + np.quantile(np.abs(flat), 0.9, axis=1)
    live = active_mask(p)
    if rates.shape[2] > TRIALS:
        sub = np.arange(0, rates.shape[2], max(1, rates.shape[2] // TRIALS))[:TRIALS]
        flat = rates[:, :, sub].reshape(rates.shape[0], -1)
    return participation_ratio(flat[live]), float(live.sum())


def load_cache():
    """The cache dict, empty if none exists yet."""
    if os.path.exists(CACHE):
        with open(CACHE) as fh:
            return json.load(fh)
    return {}


def save_cache(c):
    """Write the cache dict."""
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    with open(CACHE, "w") as fh:
        json.dump(c, fh, indent=0, sort_keys=True)


def dims(net_dir, cache=None):
    """measure() with the on-disk cache in front of it.

    Args:
        net_dir: a trained-network folder; cache: an open cache dict, or None to load and save it
            around this one call.
    Returns:
        float dimensionality, nan if the run cannot be scored.
    """
    own = cache is None
    c = load_cache() if own else cache
    _f, k = _key(net_dir)
    if k is None:
        return float("nan")
    if k not in c:
        c[k] = measure(net_dir)[0]
        if own:
            save_cache(c)
    return c[k]


def main(patterns):
    """Measure every run folder matched by these globs and report per-cell means."""
    c = load_cache()
    for pat in patterns:
        vals = {}
        for d in [x for x in sorted(glob.glob(pat)) if os.path.isdir(x)]:
            v = dims(d, cache=c)
            if np.isfinite(v):
                vals.setdefault(os.path.dirname(d), []).append(v)
            save_cache(c)
        for cell, vs in vals.items():
            print(f"  {os.path.relpath(cell, os.path.join(REPO, 'data', 'trained_RNNs')):66s} "
                  f"n={len(vs)} dims {np.mean(vs):6.2f} "
                  f"±{np.std(vs, ddof=1) if len(vs) > 1 else 0:.2f}", flush=True)
    save_cache(c)


if __name__ == "__main__":
    main(sys.argv[1:])
