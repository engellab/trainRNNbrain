#!/usr/bin/env python3
"""
Measure the three numbers the talk's opening rests on, once, and cache them.

The opening claims that (1) a trained network's units split into a firing minority and a silent
majority, (2) the split is not there at initialisation, and (3) the deep-reinforcement-learning
literature's own dormant-neuron criterion -- a formula written by other people, for other networks,
on other tasks -- lands on the same units. The third is the one that matters: it is an independently
derived oracle, so agreement is evidence the measure is not an artefact of how this project defines
silence.

THE TWO CRITERIA ARE NOT VARIANTS OF ONE FORMULA.

  this project   p_i = std_t(r_i) + q_0.9(|r_i|),  active when p_i >= 0.05 * q_0.95(p)
                 -- a spread-plus-level statistic, thresholded RELATIVE to the network's own
                 95th percentile, so it has no absolute scale.

  Sokar et al.   s_i = E|h_i| / mean_k E|h_k|,  dormant when s_i <= tau
  (ICML 2023)    -- a mean-rate statistic, normalised by the population MEAN, thresholded at a
                 fixed tau. Written for ReLU layers in deep RL agents.

They share no term: different statistic, different normaliser, different threshold rule. If they
agree on which units are out of service, that agreement is a fact about the networks.

Usage:  python motivation_cache.py [--refresh]
Output: data/motivation_cache.npz
"""

import argparse
import glob
import os
import sys

import numpy as np
import torch
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                                "..", "..")))
from common import DATA_DIR, participation, active_count
from f2_remedies_cache import load_net
from trainRNNbrain.rnns.RNN_torch import RNN_torch

OUT = "data/motivation_cache.npz"
TRIALS = 64                 # trials subsampled from the task's own batch, strided across it
TAUS = (0.0, 0.01, 0.025, 0.1)      # the dormancy thresholds Sokar et al. report
INIT_SEEDS = (7, 11, 23, 41, 97)    # untrained twins, so the init count carries its own spread

# One cell per task, the unpenalised standard network at N = 1000. These are the SAME cells the
# rest of the deck calls its controls, so the opening and the results sections describe one
# population rather than two.
CELLS = {
    "3-bit flip-flop":
        f"{DATA_DIR}/NBitFlipFlop_std_dropout/EqType=h_k=3_N=1000_pen=none_do=none",
    "CDDM":
        f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000",
}


def rates(net, u):
    """Noise-free firing rates of one network on one input batch.

    Noise-free because every active-unit count in this project is read noise-free: injected noise
    puts a floor under every unit's variance, and under that floor the relative criterion counts
    units that the task never drives. See the participation-is-measured-noise-free rule.

    Args:
        net: an RNN_torch; u: (n_inputs, T, B) input tensor.
    Returns:
        (N, T, B) float array of rates.
    """
    net.sigma_rec = net.sigma_inp = 0.0
    with torch.no_grad():
        states, _ = net(u)
    r = net.activation(states) if net.equation_type == "h" else states
    return r.detach().cpu().numpy()


def dormant_counts(r, taus=TAUS):
    """Sokar et al.'s tau-dormant count, applied to an RNN's units.

    Their score normalises each unit's mean absolute activation by the population's mean of the
    same quantity, so a unit scoring 1.0 is exactly average and 0 is never active. Time and trials
    are both expectation axes here, which is what `E_{x in D}` is over in a feedforward layer.

    Args:
        r: (N, T, B) rates; taus: thresholds to count at.
    Returns:
        dict tau -> int count of units with s_i <= tau.
    """
    e = np.abs(r).reshape(r.shape[0], -1).mean(axis=1)
    s = e / e.mean()
    return {float(t): int((s <= t).sum()) for t in taus}


def untrained_twin(cfg, seed):
    """A fresh network with the trained one's architecture and none of its training.

    Every constructor argument is taken from the trained run's own saved config, so the comparison
    is training against no training and not one architecture against another.

    Args:
        cfg: the OmegaConf config saved beside a trained run; seed: the weight draw's seed.
    Returns:
        an untrained RNN_torch.
    """
    m = cfg.model
    return RNN_torch(N=int(m.N), activation_args=OmegaConf.to_container(m.activation_args),
                     equation_type=str(m.equation_type), dale=bool(m.dale),
                     io_nonnegativity=bool(m.io_nonnegativity),
                     self_connections=bool(m.self_connections), bias_range=list(m.bias_range),
                     gamma=float(m.gamma), dt=float(m.dt), tau=float(m.tau),
                     sigma_rec=0.0, sigma_inp=0.0, sigma_out=0.0,
                     n_inputs=int(cfg.task.n_inputs), n_outputs=int(cfg.task.n_outputs),
                     seed=seed)


def measure(cell):
    """Both criteria on every trained net of one cell, and on untrained twins of the first.

    Args:
        cell: a sweep cell directory holding one folder per seed.
    Returns:
        dict of arrays, or None if the cell holds no readable network.
    """
    nets = [d for d in sorted(glob.glob(os.path.join(cell, "*"))) if os.path.isdir(d)]
    if not nets:
        return None
    trained_active, trained_dormant, p_example = [], [], None
    cfg = None
    batch = None
    # ⚠️ THE BUDGET GOES IN THE CACHE. The deck reads 40,000 iterations on the slide after this one
    # and throughout its results sections, so a panel that says only "after training" invites the
    # reader to assume 40,000. These nets run to their own max_iter -- 150,000 on the flip-flop --
    # and the figure now prints it.
    last_iter = float("nan")
    for net_dir in nets:
        rnn, task, _, _, _ = load_net(net_dir)
        if batch is None:
            bi, _, _ = task.get_batch()
            bi = torch.tensor(np.asarray(bi), dtype=torch.float32)
            sub = np.arange(0, bi.shape[-1], max(1, bi.shape[-1] // TRIALS))[:TRIALS]
            batch = bi[:, :, sub]
            cfg = OmegaConf.load(glob.glob(os.path.join(net_dir, "*_config.yaml"))[0])
        r = rates(rnn, batch)
        p = participation(r)
        last_iter = float(OmegaConf.load(
            glob.glob(os.path.join(net_dir, "*_config.yaml"))[0]).trainer.max_iter)
        trained_active.append(active_count(p, "scalefree"))
        trained_dormant.append([dormant_counts(r)[float(t)] for t in TAUS])
        if p_example is None:
            p_example = p

    init_active, init_dormant, p_init = [], [], None
    for seed in INIT_SEEDS:
        r = rates(untrained_twin(cfg, seed), batch)
        p = participation(r)
        init_active.append(active_count(p, "scalefree"))
        init_dormant.append([dormant_counts(r)[float(t)] for t in TAUS])
        if p_init is None:
            p_init = p

    return dict(read_at=np.asarray([last_iter], float),
                trained_active=np.asarray(trained_active, float),
                trained_dormant=np.asarray(trained_dormant, float),
                init_active=np.asarray(init_active, float),
                init_dormant=np.asarray(init_dormant, float),
                p_trained=np.asarray(p_example, float),
                p_init=np.asarray(p_init, float),
                n_units=float(cfg.model.N))


def main(refresh=False):
    """Build the cache. Returns the dict that was written."""
    if os.path.exists(OUT) and not refresh:
        print(f"{OUT} exists; pass --refresh to rebuild")
        return dict(np.load(OUT, allow_pickle=True))
    out = {"taus": np.asarray(TAUS, float), "tasks": np.asarray(list(CELLS), dtype=object)}
    for task, cell in CELLS.items():
        got = measure(cell)
        if got is None:
            print(f"  SKIP {task}: no networks under {cell}")
            continue
        for k, v in got.items():
            out[f"{task}|{k}"] = v
        print(f"{task}: trained active {got['trained_active'].astype(int).tolist()} "
              f"| untrained active {got['init_active'].astype(int).tolist()} "
              f"| dormant at tau={TAUS} {got['trained_dormant'].mean(axis=0).round(0).tolist()}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    np.savez_compressed(OUT, **out)
    print(f"wrote {OUT}")
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--refresh", action="store_true", help="re-measure even if the cache exists")
    main(refresh=ap.parse_args().refresh)
