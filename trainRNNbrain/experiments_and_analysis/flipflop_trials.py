#!/usr/bin/env python3
"""Random single trials of one flip-flop network: network output (solid) vs target (dashed).

A direct check that the network actually tracks the target on individual trials, rather than only
in aggregate r2. Aggregate scores hide per-trial failures - a net that nails 90% of trials and
ignores one channel on the rest can still score well.

Reuses PerformanceAnalyzer.plot_trials (the same routine run_experiment.py uses at save time), so
what is drawn here is what the training pipeline would have drawn. Only the network selection and
the y-limits differ: flip-flop targets are +/-1, while plot_trials defaults to a 0..1 window.

Output: img/internal_figures/flipflop_trials_k<k>_N<N>_<pen>.png

Usage:  python flipflop_trials.py [k] [N] [n_trials] [pen] [sigma]
        sigma: noise level for both sigma_rec and sigma_inp (default 0 = noise-free)
"""

import os
import sys
import glob
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR
from flipflop_bouquet import best_net_pen, _score
from flipflop_fixedpoints import load_net
from trainRNNbrain.analyzers.PerformanceAnalyzer import PerformanceAnalyzer
from trainRNNbrain.training.training_utils import prepare_task_arguments


def main():
    """Draw n_trials random trials of the best net in one (k, N, penalty) cell."""
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 10
    pen = sys.argv[4] if len(sys.argv) > 4 else "none"
    sigma = float(sys.argv[5]) if len(sys.argv) > 5 else 0.0

    folder = best_net_pen(k, N, pen)
    print(f"network: {os.path.basename(folder)[:60]}  (pen={pen}, r2={_score(folder):.4f})")
    rnn, _ = load_net(folder)

    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, targets, conditions = task.get_batch()
    rng = np.random.default_rng(0)                      # fixed: the same trials every run
    inds = rng.choice(inputs.shape[-1], size=min(n_trials, inputs.shape[-1]), replace=False)
    inputs, targets = inputs[..., inds], targets[..., inds]
    print(f"  {len(inds)} trials, target range [{targets.min():+.2f}, {targets.max():+.2f}]")

    mask = np.arange(targets.shape[1])
    lim = 1.15 * float(np.abs(targets).max())
    fig = PerformanceAnalyzer(rnn, task).plot_trials(
        inputs, targets, mask, sigma_rec=sigma, sigma_inp=sigma,
        labels=[f"bit {i+1}" for i in range(k)], ylim=(-lim, lim))
    fig.suptitle(f"{k}-bit flip-flop, N={N}, pen={pen} — solid = network output, dashed = target\n"
                 f"r2={_score(folder):.4f}, noise sigma={sigma}", fontsize=10, y=1.005)

    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"flipflop_trials_k{k}_N{N}_{pen}.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
