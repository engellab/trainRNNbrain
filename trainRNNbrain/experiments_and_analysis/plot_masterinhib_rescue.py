#!/usr/bin/env python3
"""
Can the master-inhibitor-clamped units be rescued? (CDDM_731df4_g0_masterinhib)

Each net has one context-locked master inhibitor holding a fixed random fraction of units (target set
T) silent by deep, gradient-unreachable inhibition (see calibrate_master_inhibitor.py). T is
reconstructable from the net's seed. We reconstruct the exact perturbed init, regenerate T, load the
trained net, and ask whether the T units became active — separately for `none` vs `frm=0.2`, across
the silenced fractions {0.25, 0.5, 0.75, 1.0}.

Prediction: unlike the inhibitory_boost run (100% rescued), the frozen context-locked clamp gives the T
units ~no gradient, so they should stay silent even under `frm`, worsening with fraction; at frac=1.0
the network cannot learn the task (R^2 ~ 0).

"active/rescued" = trained peak firing rate >= 0.01. We also report task R^2 (net folder score) and the
master's own activity (sanity: it should stay active).

Outputs:
  - data/trained_RNNs/CDDM_731df4_g0_masterinhib/masterinhib_rescue.csv
  - img/internal_figures/masterinhib_rescue.png  (T-active% and R^2 vs fraction, none vs frm)

Run from this directory:  python3 plot_masterinhib_rescue.py
"""
import os
import glob
import json
import csv
import re
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf

from trainRNNbrain.utils import unjsonify
from plot_init_vs_trained import (build_cddm_batch, peak_and_participation, rnn_numpy_from_params,
                                  init_params_from_config, SEED_OFFSET)

HERE = os.path.dirname(os.path.abspath(__file__))
SWEEP = "CDDM_731df4_g0_masterinhib"
ROOT = os.path.join(HERE, "../../data/trained_RNNs", SWEEP)
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
DEAD_ABS = 0.01
COND_RE = re.compile(r"EqType=(?P<eq>[hs])_N=(?P<N>\d+)_MIF=(?P<mif>[\d.]+)_LmbdFR=(?P<frm>[\d.]+)")
FRACS = [0.25, 0.5, 0.75, 1.0]


def target_set(cfg_seed, N, exc2inhR, frac):
    """Regenerate the master inhibitor's target set T (matches RNN_torch.__init__)."""
    Ne = int(np.floor(N * exc2inhR / (exc2inhR + 1)))
    mstr = Ne
    others = np.setdiff1d(np.arange(N), [mstr])
    T = np.random.default_rng(int(cfg_seed) + SEED_OFFSET).choice(
        others, int(round(frac * (N - 1))), replace=False)
    return mstr, T


def r2_from_dir(leaf_dir):
    """Validation R^2 from the leading score in the net folder name."""
    return float(os.path.basename(leaf_dir).split("_")[0])


def analyse():
    leaf = sorted(d for d in glob.glob(os.path.join(ROOT, "EqType=*", "*")) if os.path.isdir(d))
    batch = build_cddm_batch(glob.glob(os.path.join(leaf[0], "*_config.yaml"))[0])
    print(f"CDDM batch {batch.shape}; {len(leaf)} nets\n")
    rows = {}
    for d in leaf:
        m = COND_RE.search(os.path.basename(os.path.dirname(d)))
        if not m:
            continue
        eq, frac, frm = m["eq"], float(m["mif"]), m["frm"]
        cfgp = glob.glob(os.path.join(d, "*_config.yaml"))[0]
        pj = glob.glob(os.path.join(d, "*_LastParams_CDDM.json"))
        if not pj:
            continue
        if not np.isfinite(r2_from_dir(d)):          # skip diverged runs (NaN weights, folder score "nan")
            print(f"  [skip diverged] {os.path.basename(d)[:40]}")
            continue
        cfg = OmegaConf.load(cfgp)
        N = int(cfg.model.N)
        mstr, T = target_set(cfg.seed, N, float(cfg.model.exc2inhR), frac)
        Tmask = np.zeros(N, bool); Tmask[T] = True
        nonT = (~Tmask) & (np.arange(N) != mstr)

        ip = init_params_from_config(cfgp)                      # perturbed init (config carries clamp)
        pk_i, _ = peak_and_participation(rnn_numpy_from_params(ip), batch)
        with open(pj[0]) as f:
            tp = unjsonify(json.load(f))
        pk_t, pa_t = peak_and_participation(rnn_numpy_from_params(tp), batch)

        rec = rows.setdefault((eq, frac, frm), dict(initT_sil=[], T_act=[], T_medpart=[],
                                                    master_pk=[], r2=[], nonT_act=[]))
        rec["initT_sil"].append(np.mean(pk_i[T] < DEAD_ABS))
        rec["T_act"].append(np.mean(pk_t[T] >= DEAD_ABS))
        rec["T_medpart"].append(float(np.median(pa_t[T])))
        rec["master_pk"].append(float(pk_t[mstr]))
        rec["r2"].append(r2_from_dir(d))
        rec["nonT_act"].append(np.mean(pk_t[nonT] >= DEAD_ABS) if nonT.any() else np.nan)
        print(f"{eq} frac={frac} frm={frm:>3}  initT_sil={100*rec['initT_sil'][-1]:5.1f}%  "
              f"trainedT_active={100*rec['T_act'][-1]:5.1f}%  master_pk={rec['master_pk'][-1]:.2f}  "
              f"R2={rec['r2'][-1]:.3f}")
    return rows


def summarise(rows):
    out_csv = os.path.join(ROOT, "masterinhib_rescue.csv")
    hdr = ["eq", "frac", "frm", "n_nets", "init_T_silent_pct", "trained_T_active_pct",
           "T_median_participation", "nonT_active_pct", "master_peak", "r2"]
    table = []
    for (eq, frac, frm), r in sorted(rows.items()):
        table.append([eq, frac, frm, len(r["r2"]),
                      round(100 * np.mean(r["initT_sil"]), 1),
                      round(100 * np.mean(r["T_act"]), 1),
                      round(float(np.mean(r["T_medpart"])), 4),
                      round(100 * np.nanmean(r["nonT_act"]), 1),
                      round(float(np.mean(r["master_pk"])), 3),
                      round(float(np.mean(r["r2"])), 4)])
    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f); w.writerow(hdr); w.writerows(table)
    print(f"\nwrote {os.path.normpath(out_csv)}")
    line = f"{'eq':>3} {'frac':>5} {'frm':>4} {'nets':>4} {'initT sil':>10} {'trainedT active':>15} {'R2':>7}"
    print(line); print("-" * len(line))
    for eq, frac, frm, n, iT, tA, _tp, _nA, _mp, r2 in table:
        print(f"{eq:>3} {frac:>5} {frm:>4} {n:>4} {iT:9.1f}% {tA:14.1f}% {r2:7.3f}")
    return table


def plot(rows):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2))
    styles = {("h", "0"): ("h none", "#c44e52", "--"), ("h", "0.2"): ("h frm", "#c44e52", "-"),
              ("s", "0"): ("s none", "#4c72b0", "--"), ("s", "0.2"): ("s frm", "#4c72b0", "-")}
    for (eq, frm), (label, color, ls) in styles.items():
        xs, ta, r2 = [], [], []
        for frac in FRACS:
            r = rows.get((eq, frac, frm))
            if r:
                xs.append(100 * frac); ta.append(100 * np.mean(r["T_act"])); r2.append(np.mean(r["r2"]))
        axes[0].plot(xs, ta, marker="o", color=color, ls=ls, lw=2, label=label)
        axes[1].plot(xs, r2, marker="o", color=color, ls=ls, lw=2, label=label)
    axes[0].set_ylabel("silenced targets active after training  (%)")
    axes[0].set_ylim(-3, 105); axes[0].axhline(0, color="0.8", lw=0.8)
    axes[0].set_title("Rescue of master-clamped units")
    axes[1].set_ylabel("validation $R^2$"); axes[1].set_title("Task performance")
    axes[1].axhline(0, color="0.8", lw=0.8)
    for ax in axes:
        ax.set_xlabel("fraction silenced by master inhibitor  (%)")
        ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
        ax.legend(frameon=False, fontsize=9); ax.grid(alpha=0.25)
    fig.suptitle(f"Master-inhibitor clamp: are the frozen-silent units rescued? — {SWEEP}", y=1.01)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, "masterinhib_rescue.png")
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"wrote {os.path.normpath(out)}")


def main():
    rows = analyse()
    summarise(rows)
    plot(rows)


if __name__ == "__main__":
    main()
