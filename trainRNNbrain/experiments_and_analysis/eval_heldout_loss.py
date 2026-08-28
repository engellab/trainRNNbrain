#!/usr/bin/env python3
"""
Held-out validation for the CDDM networks: is the training loss a fair measure of the floor?

THE GAP THIS FILLS. Training uses `same_batch=True` on the FULL 2 x 15 x 15 = 450-condition grid,
reused for every one of 200,000 iterations, and no validation set is evaluated anywhere in the
pipeline (`val_losses` is created empty and never filled). So nothing in the project distinguishes
"learned the task" from "memorised these 450 trials". With 200k passes over a fixed batch that is a
live possibility, and it matters because the whole cross-size protocol reads the floor off the
training loss.

THE HELD-OUT SET. Coherences INTERLEAVED with the training grid — the midpoints between adjacent
training values — so every validation condition sits inside the trained range but was never seen.
This tests interpolation across the stimulus space, which is the relevant generalisation here; an
extrapolation set (coherences beyond +-1) would confound generalisation with range.

FOUR NUMBERS PER NETWORK, because two choices matter independently:
  noise off / on   the clean loss is deterministic (no noise lottery, so far better precision for
                   comparing sizes); the noisy loss is the operational quantity and needs averaging
                   over draws
  train / held-out the gap between them is the memorisation

Output: img/internal_figures/heldout_loss.png

Usage:  python eval_heldout_loss.py [SWEEP_FOLDER]
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

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.utils import filter_kwargs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR

N_NOISE_DRAWS = 8


def interleaved(cohs):
    """Midpoints between adjacent training coherences: inside the trained range, never seen.

    Args:
        cohs: sorted list of training coherence values.
    Returns:
        list of held-out coherence values.
    """
    c = sorted(float(x) for x in cohs)
    return [0.5 * (c[i] + c[i + 1]) for i in range(len(c) - 1)]


def batch_for(cfg, cohs):
    """Instantiate the task with a given coherence grid and return (inputs, targets)."""
    tc = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    tc.coherences = list(cohs)
    task = hydra.utils.instantiate(tc)
    i, t, _ = task.get_batch()
    return i, t


def mse(params, inputs, target, mask, sigma, seed):
    """Masked MSE of one parameter set on one batch at a given noise level."""
    r = RNN_numpy(**filter_kwargs(RNN_numpy, params), seed=seed)
    r.clear_history()
    r.y = r.y_init
    r.run(input_timeseries=inputs, sigma_rec=sigma[0], sigma_inp=sigma[1])
    return float(((r.get_output()[:, mask, :] - target[:, mask, :]) ** 2).mean())


def main():
    """Evaluate every trained network on the training grid and on a held-out grid."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    cfg = OmegaConf.load(glob.glob(os.path.join(sweep, "*", "*", "*_config.yaml"))[0])
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    sr, si = float(cfg.model.sigma_rec), float(cfg.model.sigma_inp)

    tr_c = list(cfg.task.coherences)
    ho_c = interleaved(tr_c)
    itr, ttr = batch_for(cfg, tr_c)
    iho, tho = batch_for(cfg, ho_c)
    print(f"train grid: {len(tr_c)} coherences -> {itr.shape[-1]} conditions")
    print(f"held-out  : {len(ho_c)} coherences -> {iho.shape[-1]} conditions (midpoints, never seen)")

    res = {}
    for folder in sorted(glob.glob(os.path.join(sweep, "*", "*/"))):
        pf = glob.glob(os.path.join(folder, "*LastParams*.npz"))
        if not pf:
            continue
        d = np.load(pf[0], allow_pickle=True)
        p = {k: d[k] for k in d.files}
        p["activation_name"] = "relu"
        p.pop("activation_args", None)
        N = int(p["N"])
        row = {
            "train_clean": mse(p, itr, ttr, mask, (0.0, 0.0), 0),
            "held_clean": mse(p, iho, tho, mask, (0.0, 0.0), 0),
            "train_noisy": np.mean([mse(p, itr, ttr, mask, (sr, si), s) for s in range(N_NOISE_DRAWS)]),
            "held_noisy": np.mean([mse(p, iho, tho, mask, (sr, si), s) for s in range(N_NOISE_DRAWS)]),
        }
        res.setdefault(N, []).append(row)
        print(f"  N={N:5d} {os.path.basename(folder.rstrip('/'))[:9]}  "
              f"clean {row['train_clean']:.5f}/{row['held_clean']:.5f}  "
              f"noisy {row['train_noisy']:.5f}/{row['held_noisy']:.5f}   (train/held-out)")

    keys = ["train_clean", "held_clean", "train_noisy", "held_noisy"]
    print(f"\n{'N':>6} " + " ".join(f"{k:>13}" for k in keys) + f" {'clean gap':>11} {'n':>3}")
    for N in sorted(res):
        m = {k: np.mean([r[k] for r in res[N]]) for k in keys}
        gap = 100 * (m["held_clean"] - m["train_clean"]) / m["train_clean"]
        print(f"{N:>6} " + " ".join(f"{m[k]:>13.5f}" for k in keys) + f" {gap:>10.1f}% {len(res[N]):>3}")

    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    Ns = sorted(res)
    style = {"train_clean": ("C0", "-", "train, noise off"),
             "held_clean": ("C3", "--", "held-out, noise off"),
             "train_noisy": ("C0", ":", "train, noise on"),
             "held_noisy": ("C3", "-.", "held-out, noise on")}
    for panel, ks in enumerate([["train_clean", "held_clean"], ["train_noisy", "held_noisy"]]):
        for k in ks:
            c, ls, lab = style[k]
            mu = [np.mean([r[k] for r in res[N]]) for N in Ns]
            sd = [np.std([r[k] for r in res[N]]) for N in Ns]
            ax[panel].errorbar(Ns, mu, yerr=sd, fmt="o" + ls, color=c, ms=7, capsize=3, lw=1.8,
                               label=lab)
        ax[panel].set(xscale="log", xlabel="$N$", ylabel="masked MSE",
                      title=["(a) noise OFF — deterministic, the precise comparison",
                             "(b) noise ON — operational, averaged over "
                             f"{N_NOISE_DRAWS} draws"][panel])
        ax[panel].legend(fontsize=9)
        ax[panel].grid(alpha=.3)
    fig.suptitle("Training grid vs held-out (interleaved) coherences, final parameters\n"
                 "the train/held-out gap is memorisation; there is no validation set in the pipeline",
                 fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(IMG_DIR, "heldout_loss.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
