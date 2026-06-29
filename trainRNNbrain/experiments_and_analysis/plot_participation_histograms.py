#!/usr/bin/env python3
"""
Pooled participation histograms for the Silent-ReLU CDDM sweep (CDDM_4a031e).

For every trained network we reconstruct the RNN from its saved LastParams, run the
noise-free CDDM validation batch (the exact path run_experiment.py uses for participation.png),
and compute each unit's participation:

    participation = std(rate over time,trials) + 0.9-quantile(|rate| over time,trials)

Neurons are POOLED across the 5 reps of a condition. A "condition" is a
(lambda_rws, lambda_frm) penalty combination at a given (equation_type, N) -> 5 nets,
i.e. 5*N neurons per pooled histogram. The two equation types (h, s) are reported as
SEPARATE figures (they are not pooled together).

Each figure: 3 vertically-stacked subplots (N = 100, 500, 1000); each overlays the 4 penalty
conditions as step histograms in the same colour scheme as the per-condition bar plot.
The near-zero pile is the silent-unit mass; the fr-magnitude penalty removes it.

Output: <repo>/img/internal_figures/participation_histograms_{h,s}.png
Cache : <repo>/data/trained_RNNs/CDDM_4a031e/participation_by_condition.npz  (skips the
        forward passes on re-run; delete it or pass --recompute to rebuild).

Run from this directory:  python plot_participation_histograms.py [--recompute]
"""
import os
import re
import sys
import glob
import json
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig
import hydra

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.utils import unjsonify, filter_kwargs
from trainRNNbrain.training.training_utils import prepare_task_arguments

OmegaConf.register_new_resolver("eval", eval, replace=True)

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "../../data/trained_RNNs/CDDM_4a031e")
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
CACHE = os.path.join(ROOT, "participation_by_condition.npz")

COND_RE = re.compile(r"EqType=(?P<eq>[hs])_N=(?P<N>\d+)_LmbdRWS=(?P<rws>[\d.]+)_LmbdFR=(?P<frm>[\d.]+)")
NS = [100, 500, 1000]
EQ_NAME = {"h": "h (hidden)", "s": "s (rate)"}
# (lambda_rws, lambda_frm) -> (legend label, colour); same scheme as plot_silent_units_per_condition.py
PENALTY = [
    (("0", "0"),      ("none  (Lrws=0, Lfr=0)",      "#c44e52")),
    (("0.05", "0"),   ("rws only  (Lrws=0.05)",      "#dd8452")),
    (("0", "0.2"),    ("fr only  (Lfr=0.2)",         "#55a868")),
    (("0.05", "0.2"), ("both  (Lrws=0.05, Lfr=0.2)", "#4c72b0")),
]


def build_cddm_batch(any_config_path):
    """Instantiate the CDDM task from a saved config and return one noise-free input batch.

    Args:
        any_config_path: path to any <score>_config.yaml (task config is constant across the sweep).
    Returns:
        input_batch: ndarray (n_inputs, T, n_conditions).
    """
    cfg = OmegaConf.load(any_config_path)
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_cfg)
    input_batch, _, _ = task.get_batch()
    return input_batch


def net_participation(params_json, input_batch):
    """Reconstruct one RNN from LastParams and return per-unit participation.

    Args:
        params_json: path to a <score>_LastParams_CDDM.json file.
        input_batch: CDDM input batch (n_inputs, T, n_conditions).
    Returns:
        ndarray (N,) participation = std(rate) + 0.9-quantile(|rate|) over (time, trials).
    """
    with open(params_json) as f:
        params = unjsonify(json.load(f))
    net_cfg = filter_kwargs(RNN_numpy, params)
    if isinstance(net_cfg, DictConfig):
        net_cfg = OmegaConf.to_container(net_cfg, resolve=True)
    rnn = RNN_numpy(**net_cfg, seed=0)
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=input_batch, sigma_rec=0, sigma_inp=0)
    fr = rnn.get_firing_rate_history()                       # (N, T, B)
    return np.std(fr, axis=(1, 2)) + np.quantile(np.abs(fr), q=0.9, axis=(1, 2))


def compute_pooled():
    """Run all nets and pool per-unit participation by (eq, N, lambda_rws, lambda_frm).

    Returns:
        dict {f"{eq}_N{N}_{rws}_{frm}": ndarray of pooled participation values (5*N,)}.
    """
    leaf_dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "EqType=*", "*")) if os.path.isdir(d))
    if not leaf_dirs:
        raise SystemExit(f"No net folders under {ROOT}")
    sample_cfg = glob.glob(os.path.join(leaf_dirs[0], "*_config.yaml"))[0]
    input_batch = build_cddm_batch(sample_cfg)
    print(f"CDDM batch {input_batch.shape}; pooling participation over {len(leaf_dirs)} nets...")

    pooled = {}
    for d in leaf_dirs:
        m = COND_RE.search(os.path.basename(os.path.dirname(d)))
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))
        if not pj:
            print(f"  [skip] no LastParams in {d}");  continue
        key = f"{m['eq']}_N{m['N']}_{m['rws']}_{m['frm']}"      # pools only the 5 reps
        pooled.setdefault(key, []).append(net_participation(pj[0], input_batch))
    return {k: np.concatenate(v) for k, v in pooled.items()}


def plot_eq(eq, pooled, bins, xmax):
    """Draw and save the 3-subplot (N=100/500/1000) participation figure for one equation type.

    Args:
        eq: 'h' or 's'.
        pooled: dict keyed by f"{eq}_N{N}_{rws}_{frm}" -> participation array.
        bins: histogram bin edges (shared across both figures).
        xmax: common x-axis upper limit.
    Returns:
        path to the saved PNG.
    """
    fig, axes = plt.subplots(len(NS), 1, figsize=(9, 10), sharex=True)
    for ax, N in zip(axes, NS):
        for (rws, frm), (label, color) in PENALTY:
            v = pooled.get(f"{eq}_N{N}_{rws}_{frm}")
            if v is None:
                continue
            v = v[np.isfinite(v)]
            # very light fill, then a crisp step outline on top
            ax.hist(v, bins=bins, range=(0, xmax), histtype="stepfilled", color=color,
                    alpha=0.12, linewidth=0)
            ax.hist(v, bins=bins, range=(0, xmax), histtype="step", linewidth=1.7,
                    color=color, label=label)
        ax.set_yscale("log")
        ax.set_ylabel(f"N={N}\nneuron count")
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.set_xlim(0, xmax)
    axes[0].legend(title="penalty", frameon=False, fontsize=9, ncol=2)
    axes[0].set_title(f"Pooled participation histograms — {EQ_NAME[eq]} equation — "
                      f"CDDM ReLU-Dale sweep (4a031e; 5 nets/condition)")
    axes[-1].set_xlabel("participation  =  std(rate) + 0.9-quantile(|rate|)   over time & trials")
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"participation_histograms_{eq}.png")
    fig.savefig(out, dpi=200)
    plt.close(fig)
    return out


def main():
    recompute = "--recompute" in sys.argv
    if os.path.exists(CACHE) and not recompute:
        print(f"loading cached participation from {os.path.normpath(CACHE)} (use --recompute to rebuild)")
        pooled = {k: v for k, v in np.load(CACHE).items()}
    else:
        pooled = compute_pooled()
        np.savez_compressed(CACHE, **pooled)
        print(f"cached -> {os.path.normpath(CACHE)}")

    # common x-range and bins across both equation-type figures (so they are comparable)
    allvals = np.concatenate([v[np.isfinite(v)] for v in pooled.values()])
    xmax = float(np.percentile(allvals, 99.5))
    bins = np.linspace(0, xmax, 61)

    for eq in ("h", "s"):
        print(f"wrote {os.path.normpath(plot_eq(eq, pooled, bins, xmax))}")


if __name__ == "__main__":
    main()
