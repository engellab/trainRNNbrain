#!/usr/bin/env python3
"""
Calibrate the TARGETED inhibitory-boost for the deliberate silent-at-init experiment.

Goal: an initialisation where a specific, identity-tracked ~25% of units (set S) are silent
(peak firing rate < 0.01 on the noise-free CDDM batch) while the other ~75% keep normal activity —
so we can train WITH frm and ask whether those specific S units stay silent (prevention) or climb
into the active mode (resurrection).

Why targeted, not global: the untrained net is nearly homogeneous (all units peak within +/-6% of
0.05), so a global inhibitory boost dims the whole net together and never carves out a distinct
silent subpopulation. Instead we over-inhibit ONLY the units in S: after the standard Dale init we
multiply the inhibitory columns (dale_mask == -1) of the S rows of W_rec by a factor c > 1, driving
those units' input net-negative. Training can later reduce those recurrent inhibitory weights -> a
fair prevention/resurrection test.

This calibrates c on the ReLU baseline model config (CDDM_4a031e_g0; relu, gamma=0, N=1000,
spectral_rad=1.2, exc2inhR=4), for h and s, over several seeds. The target is the MINIMUM c that
silences essentially all of S with negligible collateral silencing of the non-S 75%.

Output:
  - printed table (S silent %, non-S silent %, total silent %, non-S median peak) vs c
  - img/internal_figures/inhibitory_boost_calibration.png
  - the recommended c (min c with >= --sfrac_silent of S silent)

Run from this directory:  python3 calibrate_inhibitory_boost.py [--frac 0.25] [--seeds 4]
"""
import os
import sys
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt
from omegaconf import OmegaConf
import hydra

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.utils import filter_kwargs, import_any
from trainRNNbrain.training.training_utils import prepare_task_arguments
from plot_init_vs_trained import (build_cddm_batch, peak_and_participation,
                                  rnn_numpy_from_params)

OmegaConf.register_new_resolver("eval", eval, replace=True)

HERE = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(HERE, "../../data/trained_RNNs")
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
BASELINE = "CDDM_4a031e_g0"
DEAD_ABS = 0.01
COND = {
    "h": "EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
    "s": "EqType=s_N=1000_LmbdRWS=0_LmbdFR=0",
}


def init_params_with_seed(config_path, seed):
    """Build a fresh untrained net from a config with an explicit seed; return get_params() dict."""
    cfg = OmegaConf.load(config_path)
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    if "_target_" in task_cfg:
        del task_cfg._target_
    rnn_cfg = OmegaConf.create(cfg.model)
    rnn_cls = import_any(getattr(rnn_cfg, "_target_", None)) or RNN_torch
    rnn_args = filter_kwargs(rnn_cls, OmegaConf.merge(rnn_cfg, task_cfg))
    rnn_args.seed = int(seed)
    return hydra.utils.instantiate(rnn_args).get_params()


def choose_S(N, frac, seed):
    """Reproducibly choose the silent-target set S: a random `frac` of the N units."""
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(N, size=int(round(frac * N)), replace=False))


def boosted_targeted(params, c, S):
    """Copy params with the inhibitory columns of the S rows of W_rec scaled by c."""
    p = dict(params)
    W = np.array(params["W_rec"], dtype=float).copy()
    inh = np.array(params["dale_mask"]) == -1          # inhibitory sources = columns
    idx = np.ix_(S, np.where(inh)[0])
    W[idx] *= c
    p["W_rec"] = W
    return p


def measure(params, input_batch, S, N):
    """Return (S_silent_frac, nonS_silent_frac, total_silent_frac, nonS_median_peak)."""
    peak, _ = peak_and_participation(rnn_numpy_from_params(params), input_batch)
    mask = np.zeros(N, bool); mask[S] = True
    sil = peak < DEAD_ABS
    return (sil[mask].mean(), sil[~mask].mean(), sil.mean(), float(np.median(peak[~mask])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--frac", type=float, default=0.25)      # |S| as fraction of N
    ap.add_argument("--seeds", type=int, default=4)
    ap.add_argument("--sfrac_silent", type=float, default=0.97)  # want >= this fraction of S silent
    ap.add_argument("--cmax", type=float, default=8.0)
    ap.add_argument("--cstep", type=float, default=0.5)
    args = ap.parse_args()

    cs = np.round(np.arange(1.0, args.cmax + 1e-9, args.cstep), 3)
    seeds = [10007 * (k + 1) + 3 for k in range(args.seeds)]

    results, cstar = {}, {}
    for eq, cond in COND.items():
        cfgp = sorted(glob.glob(os.path.join(DATA, BASELINE, cond, "*", "*_config.yaml")))[0]
        input_batch = build_cddm_batch(cfgp)
        print(f"\n=== {eq} equation ({cond}) — {args.seeds} seeds, |S|={args.frac:.0%} of N ===")
        bases = [init_params_with_seed(cfgp, s) for s in seeds]
        Ns = [np.array(b["W_rec"]).shape[0] for b in bases]
        Ss = [choose_S(N, args.frac, s) for N, s in zip(Ns, seeds)]
        Ssil = np.zeros((len(seeds), len(cs))); nSsil = np.zeros_like(Ssil)
        tot = np.zeros_like(Ssil); nSmed = np.zeros_like(Ssil)
        for si, (base, S, N) in enumerate(zip(bases, Ss, Ns)):
            for ci, c in enumerate(cs):
                a, b_, t, med = measure(boosted_targeted(base, c, S), input_batch, S, N)
                Ssil[si, ci], nSsil[si, ci], tot[si, ci], nSmed[si, ci] = a, b_, t, med
        results[eq] = (cs, Ssil.mean(0), nSsil.mean(0), tot.mean(0), nSmed.mean(0))
        # recommended c: first c where mean S-silent >= sfrac_silent
        ok = np.where(Ssil.mean(0) >= args.sfrac_silent)[0]
        cstar[eq] = float(cs[ok[0]]) if len(ok) else np.nan
        print(f"{'c':>5} {'S silent':>9} {'nonS silent':>12} {'total':>7} {'nonS med peak':>14}")
        for ci, c in enumerate(cs):
            print(f"{c:5.1f} {100*Ssil.mean(0)[ci]:8.1f}% {100*nSsil.mean(0)[ci]:11.1f}% "
                  f"{100*tot.mean(0)[ci]:6.1f}% {nSmed.mean(0)[ci]:13.4f}")
        print(f">>> min c silencing >={100*args.sfrac_silent:.0f}% of S: {cstar[eq]}")

    fig, ax = plt.subplots(figsize=(8.5, 5.5))
    for eq, color in (("h", "#c44e52"), ("s", "#4c72b0")):
        cs, S_, nS_, tot_, _ = results[eq]
        ax.plot(cs, 100 * S_, marker="o", color=color, lw=2, label=f"{eq}: S silent (target 25%)")
        ax.plot(cs, 100 * nS_, marker="x", color=color, lw=1.2, ls="--", label=f"{eq}: non-S silent (want ~0)")
        if np.isfinite(cstar[eq]):
            ax.axvline(cstar[eq], color=color, ls=":", lw=1)
    ax.axhline(100 * 0.97, color="0.6", ls=":", lw=0.8)
    ax.set_xlabel("targeted inhibitory boost c   (W_rec[S, I-cols] × c at init)")
    ax.set_ylabel("silent-at-init fraction within group  (%)")
    ax.set_title("Targeted inhibitory-boost calibration — ReLU baseline, N=1000, γ=0\n"
                 "silence 25% (set S) at init, keep the other 75% active (mean over seeds)")
    ax.set_ylim(0, 100)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, "inhibitory_boost_calibration.png")
    fig.savefig(out, dpi=200)
    print(f"\nwrote {os.path.normpath(out)}")
    print(f"SUMMARY  c*(h)={cstar['h']}  c*(s)={cstar['s']}  "
          f"(min c silencing >={100*args.sfrac_silent:.0f}% of the S=25% set)")


if __name__ == "__main__":
    main()
