#!/usr/bin/env python3
"""
Do the rescued units DO anything? An ablation test.

THE OBJECTION THIS ANSWERS. Every success measure in this project is a count of active units, and
the rate penalty is a penalty on activity, so a referee is entitled to say that the remedy has only
been validated on the measure it optimises: the units are non-zero, but nothing shows they carry
task information. This project's own figure review wrote exactly that sentence.

THE TEST. Take a network trained with the rate penalty, in which essentially all N units are
active. Rank its units by participation and remove the lowest-ranked ones -- these are, by
construction, the units that WOULD have been silent without the penalty. Measure the task r^2 as a
function of how many are removed, and compare against removing the same number of units at random.

  - If the rescued units are decoration, removing them costs much LESS than removing random units,
    because a random set includes genuinely load-bearing units.
  - If the rescued units are doing work, the two curves are close.

A unit is removed by zeroing its outgoing weights -- its column of W_rec and its column of W_out --
which deletes its influence on the rest of the network and on the read-out, leaving the surviving
units' own dynamics untouched. Nothing is retrained: this measures what the trained network was
using, not what it could relearn without them.

The unpenalised network is run through the same procedure as a reference, over its OWN active set.

Usage:  python ablate_rescued_units.py [--task cddm] [--nets 3] [--refresh]
Output: img/internal_figures/fig_ablation.png  and  data/ablation_cache.pkl
"""

import argparse
import glob
import os
import pickle
import sys

import hydra
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import paperstyle as ps
from common import DATA_DIR, SILENT_REL
from flipflop_diversity import load_net
from trainRNNbrain.training.training_utils import prepare_task_arguments

CACHE = "data/ablation_cache.pkl"
FRACTIONS = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
N_RANDOM = 5            # random ablation sets per fraction, to average the control
ARMS = [("none", "no penalty", ps.BASE),
        ("frm", "rate penalty", ps.COND_COL["frm"])]
CELLS = {
    "none": f"{DATA_DIR}/CDDM_std_g0_drift/EqType=h_N=1000_iters=200000",
    "frm":  f"{DATA_DIR}/CDDM_std_g0_penalties/EqType=h_N=1000_pen=frm",
}


def r2_of(output, target, mask_idx):
    """Coefficient of determination of a network output against its target, over scored steps.

    Args:
        output: (n_out, T, B) network output; target: same shape; mask_idx: time indices scored.
    Returns:
        r^2 as a float.
    """
    y = np.asarray(output, float)[:, mask_idx, :]
    t = np.asarray(target, float)[:, mask_idx, :]
    return float(1.0 - ((y - t) ** 2).mean() / max(((t - t.mean()) ** 2).mean(), 1e-300))


def ablate_curve(folder, n_trials=128, rng=None):
    """Task r^2 against the fraction of units removed, lowest-participation-first and at random.

    Args:
        folder: run folder of one trained network; n_trials: batch size for the probe;
        rng: np.random.Generator for the random control sets.
    Returns:
        dict with 'fracs', 'r2_low' (removing the least active first), 'r2_rand'
        (mean over N_RANDOM random sets), 'r2_rand_sd', 'n_active' and 'r2_full'.
    """
    rng = np.random.default_rng(0) if rng is None else rng
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    cfg.task.batch_size = n_trials
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, targets, _ = task.get_batch()
    net, _ = load_net(folder)

    W_rec0 = np.array(net.W_rec, float)
    W_out0 = np.array(net.W_out, float)
    N = W_rec0.shape[0]
    # Scored over the decision epoch: the task has no mask attribute, and the read-out is only
    # meaningful once the decision is due. dec_on comes from the run's own config.
    T = np.asarray(targets).shape[1]
    dec_on = int(getattr(cfg.task, "T_dec_on", 0) / cfg.model.dt)
    mask_idx = np.arange(min(dec_on, T - 1), T)

    def run_with(drop):
        """Simulate the network with the units in `drop` removed; returns its task r^2."""
        net.W_rec = W_rec0.copy()
        net.W_out = W_out0.copy()
        if len(drop):
            net.W_rec[:, drop] = 0.0
            net.W_out[:, drop] = 0.0
        net.clear_history()
        net.y = net.y_init
        net.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
        out = np.asarray(net.get_output()) if hasattr(net, "get_output") else \
            np.einsum("oj,jtk->otk", net.W_out, np.maximum(np.array(net.get_history()), 0.0))
        return r2_of(out, targets, mask_idx), np.array(net.get_history())

    r2_full, hist = run_with(np.array([], int))
    R = np.maximum(hist, 0.0).reshape(N, -1)
    p = R.std(axis=1) + np.quantile(R, 0.9, axis=1)
    n_active = int((p >= SILENT_REL * np.quantile(p, 0.95)).sum())
    order_low = np.argsort(p)                       # least active first

    low, rnd, rnd_sd = [], [], []
    for f in FRACTIONS:
        k = int(round(f * N))
        low.append(run_with(order_low[:k])[0])
        vals = [run_with(rng.choice(N, size=k, replace=False))[0] for _ in range(N_RANDOM)] if k else [r2_full]
        rnd.append(float(np.mean(vals)))
        rnd_sd.append(float(np.std(vals)))
    net.W_rec, net.W_out = W_rec0, W_out0
    return {"fracs": np.array(FRACTIONS), "r2_low": np.array(low), "r2_rand": np.array(rnd),
            "r2_rand_sd": np.array(rnd_sd), "n_active": n_active, "r2_full": r2_full, "N": N}


def measure(refresh=False, n_nets=3):
    """Ablation curves for every arm. Returns dict arm -> list of per-network curve dicts."""
    if os.path.exists(CACHE) and not refresh:
        return pickle.load(open(CACHE, "rb"))
    out = {}
    for pen, _, _ in ARMS:
        rows = []
        for i, folder in enumerate(sorted(glob.glob(os.path.join(CELLS[pen], "*", "")))[:n_nets]):
            rows.append(ablate_curve(folder, rng=np.random.default_rng(100 + i)))
            print(f"  {pen:5} net {i}: r2_full={rows[-1]['r2_full']:.4f} "
                  f"active={rows[-1]['n_active']}")
        out[pen] = rows
    pickle.dump(out, open(CACHE, "wb"))
    return out


def main():
    """Run the ablation, draw the figure, print the numbers. Returns the output path."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--nets", type=int, default=3)
    ap.add_argument("--refresh", action="store_true")
    args = ap.parse_args()

    ps.setup()
    data = measure(refresh=args.refresh, n_nets=args.nets)

    fig, axes = plt.subplots(1, 2, figsize=(ps.W2 * 0.72, 62 * ps.MM))
    for ax, (pen, lab, col) in zip(axes, ARMS):
        rows = data.get(pen, [])
        if not rows:
            continue
        f = rows[0]["fracs"]
        low = np.array([r["r2_low"] for r in rows])
        rnd = np.array([r["r2_rand"] for r in rows])
        ps.band(ax, f, low.mean(0), low.std(0), col) if hasattr(ps, "band") else None
        ax.plot(f, low.mean(0), "-o", color=col, ms=2.8, lw=1.2, label="least active removed first")
        ax.fill_between(f, low.mean(0) - low.std(0), low.mean(0) + low.std(0), color=col,
                        alpha=0.16, lw=0)
        ax.plot(f, rnd.mean(0), "--s", color=ps.MUTED, ms=2.6, lw=1.0, label="random units removed")
        ax.fill_between(f, rnd.mean(0) - rnd.std(0), rnd.mean(0) + rnd.std(0), color=ps.MUTED,
                        alpha=0.14, lw=0)
        frac_active = np.mean([r["n_active"] / r["N"] for r in rows])
        ax.axvline(1 - frac_active, color=ps.BAD, lw=0.8, ls=":", zorder=2)
        ax.text(1 - frac_active, 0.04, f"  {1 - frac_active:.0%} of units\n  are silent here",
                fontsize=5.4, color=ps.BAD, transform=ax.get_xaxis_transform(), va="bottom")
        ax.set(title=f"{lab}  ({np.mean([r['n_active'] for r in rows]):.0f} active)",
               xlabel="fraction of units removed", ylabel="task $r^2$", ylim=(-0.05, 1.0))
        ax.legend(loc="lower left", fontsize=5.8)
        ps.ygrid(ax)
    out = ps.save(fig, "fig_ablation")

    print("\n--- ablation ---")
    for pen, lab, _ in ARMS:
        rows = data.get(pen, [])
        if not rows:
            continue
        f = rows[0]["fracs"]
        low = np.array([r["r2_low"] for r in rows]).mean(0)
        rnd = np.array([r["r2_rand"] for r in rows]).mean(0)
        print(f"  {lab}  (n={len(rows)}, {np.mean([r['n_active'] for r in rows]):.0f} active)")
        for i, fr in enumerate(f):
            print(f"     remove {fr:4.0%}:  least-active-first r2 = {low[i]:7.4f}   "
                  f"random r2 = {rnd[i]:7.4f}   gap = {low[i] - rnd[i]:+.4f}")
    return out


if __name__ == "__main__":
    main()
