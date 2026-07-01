#!/usr/bin/env python3
"""
Prevention vs resurrection: per-unit activity at initialisation vs after training.

Each net in these sweeps was trained as a separate job (`n_nets=1`), so its untrained state is a
pure function of its saved config `seed` (the per-net RNN generator is seeded with
`seed + (0*14653 + 65537**3) % 7309 = seed + 3508`; weight init consumes no other RNG, and with
`bias_range=[0,0]` there is no bias draw). We therefore reconstruct each net's **exact** initial
weights, score them on the same noise-free CDDM batch used for the trained weights, and — because
units are never reordered during training — track **the same unit index** before and after.

For every unit we compute, at init and after training:
  - peak rate    = max over (time, conditions) of |firing rate|      (silence thresholds)
  - participation = std(rate) + 0.9-quantile(|rate|) over (time, conditions)  (graded activity)

A unit is silent by two criteria (as elsewhere): dead<0.01 (absolute) and silent<5%p95 (peak below
5% of that net's 95th-percentile peak — scale-free).

Outputs (TAG = sweep folder minus "CDDM_"):
  - data/trained_RNNs/<SWEEP>/init_vs_trained_silent.csv   : per-condition init vs trained silent %
  - data/trained_RNNs/<SWEEP>/init_vs_trained.npz          : cache of pooled per-unit arrays
  - img/internal_figures/init_vs_trained_hist_{h,s}_<TAG>.png   : participation histograms, init vs trained
  - img/internal_figures/init_vs_trained_scatter_{h,s}_<TAG>.png: per-unit init->trained participation

Because the init weights do NOT depend on the penalty, a single pooled init distribution per equation
type is used as the common reference against the four trained (penalty) distributions.

Run from this directory:  python3 plot_init_vs_trained.py <SWEEP_FOLDER> [--recompute]
"""
import os
import re
import sys
import csv
import glob
import json
import numpy as np
from collections import defaultdict
from matplotlib import pyplot as plt
from omegaconf import OmegaConf, DictConfig
import hydra

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.utils import unjsonify, filter_kwargs, import_any
from trainRNNbrain.training.training_utils import prepare_task_arguments
from plot_participation_histograms import build_cddm_batch, PENALTY, EQ_NAME

OmegaConf.register_new_resolver("eval", eval, replace=True)

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = next((a for a in sys.argv[1:] if not a.startswith("-")), "CDDM_fb2792_g0_softplus25")
TAG = SWEEP.replace("CDDM_", "")
ROOT = os.path.join(HERE, "../../data/trained_RNNs", SWEEP)
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
CACHE = os.path.join(ROOT, "init_vs_trained.npz")

COND_RE = re.compile(r"EqType=(?P<eq>[hs])_N=(?P<N>\d+)_LmbdRWS=(?P<rws>[\d.]+)_LmbdFR=(?P<frm>[\d.]+)")
SEED_OFFSET = (0 * 14653 + (0 + 65537) ** 3) % 7309   # per-net seed offset for net index i=0 (n_nets=1)
DEAD_ABS = 0.01
REL_FRAC = 0.05


def peak_and_participation(rnn, input_batch):
    """Run one RNN_numpy on the batch and return (peak, participation) per unit.

    Args:
        rnn: an RNN_numpy ready to run (weights already set).
        input_batch: CDDM input batch (n_inputs, T, n_conditions).
    Returns:
        (peak, part): each ndarray (N,). peak = max|rate| over (t, cond);
        part = std(rate) + 0.9-quantile(|rate|) over (t, cond).
    """
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=input_batch, sigma_rec=0, sigma_inp=0)
    fr = rnn.get_firing_rate_history()                       # (N, T, B)
    peak = np.abs(fr).max(axis=(1, 2))
    part = np.std(fr, axis=(1, 2)) + np.quantile(np.abs(fr), q=0.9, axis=(1, 2))
    return peak, part


def rnn_numpy_from_params(params):
    """Build an RNN_numpy from a get_params()/LastParams-style dict."""
    net_cfg = filter_kwargs(RNN_numpy, params)
    if isinstance(net_cfg, DictConfig):
        net_cfg = OmegaConf.to_container(net_cfg, resolve=True)
    return RNN_numpy(**net_cfg, seed=0)


def init_params_from_config(config_path):
    """Reconstruct a net's untrained (effective) weights from its saved config + seed.

    Mirrors run_experiment.py's per-net RNN construction (net index i=0) so the initial weights
    are byte-for-byte those training started from.

    Args:
        config_path: path to the net's saved <score>_config.yaml.
    Returns:
        dict of initial params (same schema as LastParams), via RNN_torch.get_params().
    """
    cfg = OmegaConf.load(config_path)
    per_net_seed = int(cfg.seed) + SEED_OFFSET
    # replicate run_experiment: merge model cfg with the task cfg (which supplies n_inputs/n_outputs)
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in task_cfg:
        del task_cfg._target_
    rnn_cfg = OmegaConf.create(cfg.model)
    rnn_cls = import_any(getattr(rnn_cfg, "_target_", None)) or RNN_torch
    rnn_args = filter_kwargs(rnn_cls, OmegaConf.merge(rnn_cfg, task_cfg))
    rnn_args.seed = per_net_seed
    rnn_torch = hydra.utils.instantiate(rnn_args)
    return rnn_torch.get_params()


def compute():
    """Reconstruct init + load trained for every net; return pooled per-unit arrays by condition.

    Returns:
        dict keyed by (eq, rws, frm) -> dict with pooled 1-D arrays:
        peak_init, part_init, peak_tr, part_tr, and list of per-net p95(peak_init)/p95(peak_tr).
        Also a special key ('INIT', eq) is NOT used — init is stored per condition for pairing but
        pooled per eq at plot time.
    """
    leaf_dirs = sorted(d for d in glob.glob(os.path.join(ROOT, "EqType=*", "*")) if os.path.isdir(d))
    if not leaf_dirs:
        raise SystemExit(f"No net folders under {ROOT}")
    input_batch = build_cddm_batch(glob.glob(os.path.join(leaf_dirs[0], "*_config.yaml"))[0])
    print(f"CDDM batch {input_batch.shape}; reconstructing init + trained for {len(leaf_dirs)} nets...")

    pooled = defaultdict(lambda: dict(peak_init=[], part_init=[], peak_tr=[], part_tr=[]))
    for k, d in enumerate(leaf_dirs):
        m = COND_RE.search(os.path.basename(os.path.dirname(d)))
        if m is None:
            continue
        cfgp = glob.glob(os.path.join(d, "*_config.yaml"))
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))
        if not cfgp or not pj:
            print(f"  [skip] missing config/params in {d}");  continue

        # trained
        with open(pj[0]) as f:
            tr_params = unjsonify(json.load(f))
        peak_tr, part_tr = peak_and_participation(rnn_numpy_from_params(tr_params), input_batch)
        # init (reconstructed from seed)
        init_params = init_params_from_config(cfgp[0])
        peak_in, part_in = peak_and_participation(rnn_numpy_from_params(init_params), input_batch)

        key = (m["eq"], m["rws"], m["frm"])
        pooled[key]["peak_init"].append(peak_in)
        pooled[key]["part_init"].append(part_in)
        pooled[key]["peak_tr"].append(peak_tr)
        pooled[key]["part_tr"].append(part_tr)
        print(f"  [{k+1}/{len(leaf_dirs)}] {m['eq']} rws={m['rws']} frm={m['frm']}  "
              f"init dead={100*np.mean(peak_in<DEAD_ABS):.1f}%  trained dead={100*np.mean(peak_tr<DEAD_ABS):.1f}%")
    return {k: {kk: np.array(vv) for kk, vv in v.items()} for k, v in pooled.items()}


def silent_pcts(peak_2d):
    """Per-net silent fractions from a (n_nets, N) peak array. Returns (dead_pct, rel_pct) arrays."""
    dead = 100 * np.mean(peak_2d < DEAD_ABS, axis=1)
    p95 = np.quantile(peak_2d, 0.95, axis=1, keepdims=True)
    rel = 100 * np.mean(peak_2d < REL_FRAC * p95, axis=1)
    return dead, rel


def write_csv(pooled):
    """Write per-condition init vs trained silent percentages."""
    out = os.path.join(ROOT, "init_vs_trained_silent.csv")
    with open(out, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["eq", "rws", "frm", "n_nets", "N",
                    "init_dead_pct", "init_rel_pct", "trained_dead_pct", "trained_rel_pct"])
        for (eq, rws, frm), v in sorted(pooled.items()):
            di, ri = silent_pcts(v["peak_init"])
            dt, rt = silent_pcts(v["peak_tr"])
            w.writerow([eq, rws, frm, v["peak_init"].shape[0], v["peak_init"].shape[1],
                        round(di.mean(), 1), round(ri.mean(), 1),
                        round(dt.mean(), 1), round(rt.mean(), 1)])
    print(f"wrote {os.path.normpath(out)}")
    return out


def plot_hist(eq, pooled, xmax):
    """Participation histograms: one pooled init reference vs the 4 trained penalty distributions."""
    init_all = np.concatenate([v["part_init"].ravel() for (e, *_), v in pooled.items() if e == eq])
    fig, ax = plt.subplots(figsize=(9, 5.5))
    bins = np.linspace(0, xmax, 61)
    ax.hist(init_all[np.isfinite(init_all)], bins=bins, histtype="step", linewidth=2.2,
            color="black", linestyle="--", label="init (untrained, pooled)")
    for (rws, frm), (label, color) in PENALTY:
        v = pooled.get((eq, rws, frm))
        if v is None:
            continue
        vals = v["part_tr"].ravel()
        ax.hist(vals[np.isfinite(vals)], bins=bins, histtype="step", linewidth=1.8,
                color=color, label=f"trained: {label}")
    ax.set_yscale("log")
    ax.set_xlim(0, xmax)
    ax.set_xlabel("participation  =  std(rate) + 0.9-quantile(|rate|)")
    ax.set_ylabel("neuron count")
    ax.set_title(f"Participation before vs after training — {EQ_NAME[eq]} — {TAG}")
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"init_vs_trained_hist_{eq}_{TAG}.png")
    fig.savefig(out, dpi=200); plt.close(fig)
    return out


def plot_scatter(eq, pooled, lim):
    """Per-unit init->trained participation (log-log), one panel per penalty.

    Because init participation is a narrow band well above 0 and trained values span >3 decades
    (silent ~1e-3 to active ~1), axes are logarithmic. Guides: dotted y=x (no change) and a
    horizontal 'silent' line; the annotation reports the fraction of units that END silent and
    the correlation between a unit's init and trained participation (near-zero r => init activity
    does not predict which units get silenced -> the fate is decided during training).
    """
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.8), sharex=True, sharey=True)
    sil = 0.05   # participation guide for "silent" (near the inter-mode dip used elsewhere)
    xs = np.concatenate([v["part_init"].ravel() for (e, *_), v in pooled.items() if e == eq])
    xlo, xhi = max(1e-3, 0.5 * np.nanmin(xs)), 1.5 * np.nanmax(xs)
    ylo, yhi = 1e-3, 1.5 * lim
    for ax, ((rws, frm), (label, color)) in zip(axes, PENALTY):
        v = pooled.get((eq, rws, frm))
        ax.set_title(label, color=color, fontsize=11, fontweight="bold")
        ax.plot([ylo, yhi], [ylo, yhi], color="0.6", lw=1, ls=":", zorder=1)
        ax.axhline(sil, color="0.7", lw=0.9, zorder=1)
        if v is not None:
            xi = v["part_init"].ravel(); yt = v["part_tr"].ravel()
            ax.scatter(np.clip(xi, xlo, xhi), np.clip(yt, ylo, yhi),
                       s=4, color=color, alpha=0.15, edgecolor="none", zorder=2)
            end_silent = 100 * np.mean(yt < sil)
            fin = np.isfinite(xi) & np.isfinite(yt) & (xi > 0) & (yt > 0)
            r = np.corrcoef(np.log(xi[fin]), np.log(yt[fin]))[0, 1] if fin.sum() > 2 else np.nan
            ax.text(0.04, 0.96, f"end silent (<{sil}): {end_silent:.0f}%\n"
                                f"corr(log init, log trained) = {r:.2f}",
                    transform=ax.transAxes, va="top", ha="left", fontsize=8,
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.8", alpha=0.8))
        ax.set_xscale("log"); ax.set_yscale("log")
        ax.set_xlim(xlo, xhi); ax.set_ylim(ylo, yhi)
        ax.set_xlabel("participation at INIT")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    axes[0].set_ylabel("participation after TRAINING")
    fig.suptitle(f"Per-unit init → trained participation — {EQ_NAME[eq]} — {TAG} "
                 f"(log-log; dotted = no change; grey line = silent guide {sil})", y=1.02)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, f"init_vs_trained_scatter_{eq}_{TAG}.png")
    fig.savefig(out, dpi=170, bbox_inches="tight"); plt.close(fig)
    return out


def main():
    recompute = "--recompute" in sys.argv
    if os.path.exists(CACHE) and not recompute:
        print(f"loading cache {os.path.normpath(CACHE)} (use --recompute to rebuild)")
        z = np.load(CACHE, allow_pickle=True)
        pooled = z["pooled"].item()
    else:
        pooled = compute()
        np.savez(CACHE, pooled=np.array(pooled, dtype=object))
        print(f"cached -> {os.path.normpath(CACHE)}")

    write_csv(pooled)
    # common axes across both eq figures
    all_part = np.concatenate([v["part_tr"].ravel() for v in pooled.values()] +
                              [v["part_init"].ravel() for v in pooled.values()])
    xmax = float(np.percentile(all_part[np.isfinite(all_part)], 99.5))
    for eq in ("h", "s"):
        if any(e == eq for (e, *_), _ in [(k, v) for k, v in pooled.items()]):
            print(f"wrote {os.path.normpath(plot_hist(eq, pooled, xmax))}")
            print(f"wrote {os.path.normpath(plot_scatter(eq, pooled, xmax))}")


if __name__ == "__main__":
    main()
