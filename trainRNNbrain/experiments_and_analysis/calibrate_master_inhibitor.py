#!/usr/bin/env python3
"""
Characterise the "master-inhibitor" silent-at-init construction (no biases).

Motivation. The inhibitory-boost silencing (calibrate_inhibitory_boost.py) does not make units
truly gradient-dead: their peak is a t=1 initial-condition transient that survives any boost, and
the inhibition onto them comes from the general I-population whose activity DRIFTS during training,
so it can weaken and the units get rescued. To realise Pavel's thought experiment — units held dead
by a SUSTAINED, context-locked inhibition whose weights training cannot reach — we use a single
"master inhibitor":

  - one inhibitory unit m (the first I-unit) is driven ONLY by the two CDDM context cues
    (input channels 0 and 1, on throughout every trial) with weight master_ctx_drive -> it is active
    in every trial and every context, and receives no recurrent input, so the network cannot silence it;
  - m projects deep negative weight (-master_inhib_strength) onto a target set T (a fixed random
    fraction of the other units) and to nobody else -> the units in T are held silent all trial;
  - the non-target units are untouched and keep normal activity.

Because a target unit is dead, the gradient to the master->target weight (and to the context->master
weight) flows through that unit's zero ReLU derivative and is ~0: the clamp is frozen, so this is the
clean "no gradient -> no rescue" test. Sweeping the target fraction over {0.25, 0.5, 0.75, 1.0} spans
"a quarter silenced" to "everything but the master silenced" (Pavel's extreme case).

This script builds fresh ReLU-baseline inits (h and s), applies the construction for each fraction,
and verifies at init: master active, targets silent (peak<0.01), non-targets healthy. Output:
  - printed table (master peak, target silent %, non-target silent %, non-target median peak)
  - img/internal_figures/master_inhibitor_calibration.png

Run from this directory:  python3 calibrate_master_inhibitor.py [--strength 5] [--ctx_drive 1] [--seeds 3]
"""
import os
import glob
import argparse
import numpy as np
from matplotlib import pyplot as plt

import calibrate_inhibitory_boost as C
from plot_init_vs_trained import rnn_numpy_from_params, build_cddm_batch, peak_and_participation

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_DIR = os.path.join(HERE, "../../img/internal_figures")
CTX_CHANNELS = (0, 1)          # CDDM context-cue input channels (motion cue, color cue)
FRACS = (0.25, 0.5, 0.75, 1.0)
DEAD_ABS = 0.01


def master_index(params):
    """Index of the master inhibitor: the first inhibitory unit (dale_mask == -1)."""
    dale = np.asarray(params["dale_mask"])
    return int(np.where(dale == -1)[0][0])


def apply_master_inhibitor(params, frac, strength, ctx_drive, seed):
    """Return params with a context-locked master inhibitor clamping a `frac` target set silent.

    Args:
        params: base init params dict (get_params schema).
        frac: fraction of the other N-1 units to hold silent (target set T).
        strength: magnitude of the deep negative master->target weight (w_inhib > 0).
        ctx_drive: master's weight from each of the two context channels (w_ctx > 0).
        seed: RNG seed for choosing T (reproducible; matches the training-code selection).
    Returns:
        (new_params, m, T): modified params, master index m, target index array T.
    """
    p = dict(params)
    N = np.asarray(params["W_rec"]).shape[0]
    W = np.asarray(params["W_rec"], float).copy()
    Wi = np.asarray(params["W_inp"], float).copy()
    m = master_index(params)
    others = np.setdiff1d(np.arange(N), [m])
    T = np.random.default_rng(seed).choice(others, int(round(frac * (N - 1))), replace=False)
    Wi[m, :] = 0.0
    for ch in CTX_CHANNELS:
        Wi[m, ch] = ctx_drive                    # master driven only by the context cues
    W[m, :] = 0.0                                # master receives no recurrent input (pure clamp)
    W[:, m] = 0.0                                # master projects to nobody...
    W[T, m] = -abs(strength)                     # ...except deep inhibition onto the target set
    p["W_rec"] = W
    p["W_inp"] = Wi
    return p, m, T


def measure(params, batch, m, T):
    """Return (master_peak, target_silent_frac, nonTarget_silent_frac, nonTarget_median_peak)."""
    peak, _ = peak_and_participation(rnn_numpy_from_params(params), batch)
    N = peak.size
    Tmask = np.zeros(N, bool); Tmask[T] = True
    nonT = (~Tmask) & (np.arange(N) != m)
    if not nonT.any():                           # frac=1.0: everything but the master is a target
        return float(peak[m]), np.mean(peak[T] < DEAD_ABS), np.nan, np.nan
    return (float(peak[m]), np.mean(peak[T] < DEAD_ABS),
            np.mean(peak[nonT] < DEAD_ABS), float(np.median(peak[nonT])))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--strength", type=float, default=5.0)   # deep master->target inhibition
    ap.add_argument("--ctx_drive", type=float, default=1.0)  # context->master weight
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()

    seeds = [10007 * (k + 1) + 3 for k in range(args.seeds)]
    results = {}
    for eq, cond in C.COND.items():
        cfgp = sorted(glob.glob(os.path.join(C.DATA, C.BASELINE, cond, "*", "*_config.yaml")))[0]
        batch = build_cddm_batch(cfgp)
        bases = [C.init_params_with_seed(cfgp, s) for s in seeds]
        print(f"\n=== {eq} equation — strength={args.strength}, ctx_drive={args.ctx_drive}, {args.seeds} seeds ===")
        print(f"{'target frac':>11} {'master peak':>11} {'target silent':>13} {'nonT silent':>11} {'nonT med peak':>13}")
        tgt_sil = np.zeros(len(FRACS))
        for fi, frac in enumerate(FRACS):
            mp, ts, ns, nmed = [], [], [], []
            for base, s in zip(bases, seeds):
                p, m, T = apply_master_inhibitor(base, frac, args.strength, args.ctx_drive, s)
                a, b, c, d = measure(p, batch, m, T)
                mp.append(a); ts.append(b); ns.append(c); nmed.append(d)
            tgt_sil[fi] = 100 * np.mean(ts)
            print(f"{frac:11.2f} {np.mean(mp):11.3f} {100*np.mean(ts):12.1f}% {100*np.mean(ns):10.1f}% {np.mean(nmed):13.3f}")
        results[eq] = tgt_sil

    fig, ax = plt.subplots(figsize=(7.5, 5))
    for eq, color in (("h", "#c44e52"), ("s", "#4c72b0")):
        ax.plot([100 * f for f in FRACS], results[eq], marker="o", color=color, lw=2, label=f"{eq} equation")
    ax.set_xlabel("target fraction silenced by the master inhibitor  (%)")
    ax.set_ylabel("targets silent at init  (peak<0.01, %)")
    ax.set_title(f"Master-inhibitor silent-at-init construction — ReLU baseline, N=1000\n"
                 f"(strength={args.strength}, ctx_drive={args.ctx_drive}; targets 100% silent, non-targets untouched)")
    ax.set_ylim(0, 105)
    ax.spines["top"].set_visible(False); ax.spines["right"].set_visible(False)
    ax.legend(frameon=False)
    ax.grid(alpha=0.25)
    plt.tight_layout()
    os.makedirs(IMG_DIR, exist_ok=True)
    out = os.path.join(IMG_DIR, "master_inhibitor_calibration.png")
    fig.savefig(out, dpi=200)
    print(f"\nwrote {os.path.normpath(out)}")


if __name__ == "__main__":
    main()
