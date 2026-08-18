#!/usr/bin/env python3
"""
Deterministic floor estimate: the NOISE-FREE loss of each trained network's final parameters.

WHY THIS REPLACES THE "LOWEST OBSERVED LOSS" STATISTIC. Everything recorded during training comes
from a NOISY forward pass (Trainer.train_step runs the network with w_noise=True), and the batch is
fixed for the whole run (same_batch=True). So the per-iteration loss varies only because of injected
recurrent noise, and taking its minimum samples the favourable tail of that noise — a noise lottery
whose winner depends on how many draws the run had. It is not a property of the trained network.

There is also no validation set anywhere in this project: Trainer.run_training creates val_losses,
never appends to it, and returns it empty. Every loss reported is a TRAINING loss on one fixed batch.
That is a real limitation of the setup, not of this script, and it means "achievable loss" can only
ever mean "on the batch it was trained on".

What is well defined is the network itself: take the final weights, run the task with the noise
switched off, and evaluate the same masked MSE that training optimised. That is deterministic and
gives exactly ONE number per seed - which is enough, because the comparison of interest is across
sizes and the replication is across seeds, not across noise draws.

Output: img/internal_figures/noisefree_loss.png

Usage:  python eval_noisefree_loss.py [SWEEP_FOLDER]
"""

import os
import re
import sys
import glob
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.utils import filter_kwargs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_loss_fit import IMG_DIR


def masked_mse(output, target, mask):
    """Mean squared error over the masked timepoints, matching the training objective.

    Args:
        output: (n_outputs, T, B) network output; target: same shape; mask: timepoint indices.
    Returns:
        scalar MSE.
    """
    return float(((output[:, mask, :] - target[:, mask, :]) ** 2).mean())


def evaluate(folder, task, inputs, target, mask):
    """Noise-free masked MSE of one trained network's final parameters.

    Args:
        folder: net folder holding *_LastParams_*.npz and *_config.yaml;
        task, inputs, target, mask: the shared task batch and training mask.
    Returns:
        (N, mse) or None if the folder lacks the needed files.
    """
    pf = glob.glob(os.path.join(folder, "*LastParams*.npz"))
    if not pf:
        return None
    d = np.load(pf[0], allow_pickle=True)
    params = {k: d[k] for k in d.files}
    params["activation_name"] = "relu"
    params.pop("activation_args", None)
    rnn = RNN_numpy(**filter_kwargs(RNN_numpy, params), seed=0)
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)   # noise OFF - the whole point
    return int(params["N"]), masked_mse(rnn.get_output(), target, mask)


def main():
    """Evaluate every trained network noise-free and plot the result against size."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    cfg = OmegaConf.load(glob.glob(os.path.join(sweep, "*", "*", "*_config.yaml"))[0])
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, target, _ = task.get_batch()
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    print(f"batch {inputs.shape}, target {target.shape}, mask covers {len(mask)} timepoints")

    res = {}
    for folder in sorted(glob.glob(os.path.join(sweep, "*", "*/"))):
        r = evaluate(folder, task, inputs, target, mask)
        if r is None:
            continue
        N, mse = r
        res.setdefault(N, []).append(mse)
        print(f"  N={N:5d}  {os.path.basename(folder.rstrip('/'))[:9]}  noise-free MSE = {mse:.5f}")

    print(f"\n{'N':>7} {'median':>10} {'MAD':>10} {'n':>4}")
    for N in sorted(res):
        v = np.array(res[N])
        print(f"{N:>7} {np.median(v):>10.5f} {np.median(np.abs(v-np.median(v))):>10.5f} {len(v):>4}")

    fig, ax = plt.subplots(figsize=(7, 5))
    for N in sorted(res):
        ax.plot([N] * len(res[N]), res[N], "o", color="C0", ms=8, alpha=.8)
    Ns = sorted(res)
    ax.plot(Ns, [np.median(res[N]) for N in Ns], "-", color="C3", lw=2, label="median across seeds")
    ax.set(xscale="log", xlabel="$N$", ylabel="noise-free masked MSE (final parameters)",
           title="Deterministic floor estimate: noise OFF, final weights\n"
                 "one number per seed; no fit, no noise lottery")
    ax.legend(fontsize=9)
    ax.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "noisefree_loss.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
