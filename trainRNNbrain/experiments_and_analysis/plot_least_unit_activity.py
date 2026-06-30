#!/usr/bin/env python3
"""
Least-participating-unit activity maps for the N=1000 nets of the Silent-ReLU CDDM sweep.

For every N=1000 network we reconstruct the RNN from LastParams, run the noise-free CDDM batch,
compute per-unit participation (std(rate) + 0.9-quantile(|rate|) over time & trials), and pick
the SINGLE least-participating unit. We then draw that unit's firing rate as a heatmap:
    x = time, y = task condition (the CDDM trial conditions), colour intensity (Reds) = activity.

Layout: 5 rows (the 5 nets of a condition) x 4 columns (the (lambda_rws, lambda_frm) penalty
conditions). Separate figures per equation type (h, s). The number in each panel is that unit's
participation. The story: in `none`/`rws only` the worst unit is dead (blank); under `fr only`/
`both` even the worst unit carries task-related activity.

Output: <repo>/img/internal_figures/least_unit_activity_{h,s}_N1000.png

Run from this directory:  python plot_least_unit_activity.py
"""
import os
import sys
import re
import glob
import json
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.utils import unjsonify, filter_kwargs
from plot_participation_histograms import build_cddm_batch, EQ_NAME

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = sys.argv[1] if len(sys.argv) > 1 else "CDDM_4a031e"   # sweep folder under data/trained_RNNs
TAG = SWEEP.replace("CDDM_", "")
SUF = "" if SWEEP == "CDDM_4a031e" else f"_{TAG}"
ROOT = os.path.join(HERE, "../../data/trained_RNNs", SWEEP)
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")

COND_RE = re.compile(r"EqType=(?P<eq>[hs])_N=(?P<N>\d+)_LmbdRWS=(?P<rws>[\d.]+)_LmbdFR=(?P<frm>[\d.]+)")
N_TARGET = 1000
# (lambda_rws, lambda_frm) -> (short column label, colour); same scheme/order as the other figures
PENALTY = [
    (("0", "0"),      ("none",     "#c44e52")),
    (("0.05", "0"),   ("rws only", "#dd8452")),
    (("0", "0.2"),    ("fr only",  "#55a868")),
    (("0.05", "0.2"), ("both",     "#4c72b0")),
]


def net_least_unit(params_json, input_batch):
    """Reconstruct one RNN and return its least-participating unit's activity.

    Args:
        params_json: path to a <score>_LastParams_CDDM.json file.
        input_batch: CDDM input batch (n_inputs, T, n_conditions).
    Returns:
        (activity, p_min, idx): activity is the unit's firing rate, shape (T, n_conditions);
        p_min is its participation (std + 0.9-quantile|.|); idx is the unit index.
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
    part = np.std(fr, axis=(1, 2)) + np.quantile(np.abs(fr), q=0.9, axis=(1, 2))
    part = np.where(np.isfinite(part), part, np.inf)         # never pick a diverged unit
    idx = int(np.argmin(part))
    return fr[idx], float(part[idx]), idx


def r2_from_dir(leaf_dir):
    """Validation R^2 of a net, parsed from the leading score in its folder name."""
    return float(os.path.basename(leaf_dir).split("_")[0])


def collect():
    """Run all N=1000 nets and return data[(eq, rws, frm)] = list of (activity, p_min, r2),
    sorted by R^2 descending so row order is deterministic."""
    leaf_dirs = sorted(d for d in glob.glob(os.path.join(ROOT, f"EqType=*_N={N_TARGET}_*", "*"))
                       if os.path.isdir(d))
    if not leaf_dirs:
        raise SystemExit(f"No N={N_TARGET} net folders under {ROOT}")
    input_batch = build_cddm_batch(glob.glob(os.path.join(leaf_dirs[0], "*_config.yaml"))[0])
    print(f"extracting least-participating unit for {len(leaf_dirs)} N={N_TARGET} nets...")
    data = {}
    for d in leaf_dirs:
        m = COND_RE.search(os.path.basename(os.path.dirname(d)))
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))
        if not pj:
            print(f"  [skip] no LastParams in {d}");  continue
        act, pmin, _ = net_least_unit(pj[0], input_batch)
        data.setdefault((m["eq"], m["rws"], m["frm"]), []).append((act, pmin, r2_from_dir(d)))
    for k in data:
        data[k].sort(key=lambda t: -t[2])
    return data


def plot_eq(eq, data, vmax):
    """Draw and save the 5x4 least-unit activity grid for one equation type."""
    nrows = 5
    fig, axes = plt.subplots(nrows, len(PENALTY), figsize=(14, 13), sharex=True, sharey=True)
    im = None
    for col, ((rws, frm), (clabel, color)) in enumerate(PENALTY):
        items = data.get((eq, rws, frm), [])
        for row in range(nrows):
            ax = axes[row, col]
            if row < len(items):
                act, pmin, r2 = items[row]
                T, B = act.shape
                im = ax.imshow(act.T, aspect="auto", origin="lower", cmap="Reds",
                               vmin=0, vmax=vmax, extent=[0, T, 0, B], interpolation="nearest")
                ax.text(0.03, 0.96, f"p={pmin:.3f}", transform=ax.transAxes, va="top", ha="left",
                        fontsize=7, bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6))
            else:
                ax.set_visible(False)
            if row == 0:
                ax.set_title(clabel, color=color, fontsize=12, fontweight="bold")
            if col == 0:
                ax.set_ylabel(f"net {row + 1}\ncondition", fontsize=8)
            if row == nrows - 1:
                ax.set_xlabel("time", fontsize=9)
    fig.suptitle(f"Least-participating unit activity — {EQ_NAME[eq]} equation — "
                 f"N={N_TARGET} CDDM ReLU-Dale ({TAG})", fontsize=13)
    if im is not None:
        cbar = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
        cbar.set_label("firing rate")
    out = os.path.join(IMG_DIR, f"least_unit_activity_{eq}_N{N_TARGET}{SUF}.png")
    os.makedirs(IMG_DIR, exist_ok=True)
    fig.savefig(out, dpi=170, bbox_inches="tight")
    plt.close(fig)
    return out


def main():
    data = collect()
    # common colour scale across both figures (99th pct of all plotted activity)
    allact = np.concatenate([t[0].ravel() for v in data.values() for t in v])
    vmax = float(np.percentile(allact[np.isfinite(allact)], 99))
    print(f"common vmax (99th pct of activity) = {vmax:.3f}")
    for eq in ("h", "s"):
        print(f"wrote {os.path.normpath(plot_eq(eq, data, vmax))}")


if __name__ == "__main__":
    main()
