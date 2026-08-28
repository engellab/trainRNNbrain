#!/usr/bin/env python3
"""
Penalty comparison done fairly: matched ITERATION budget, and the CLEAN TASK loss.

Two confounds make the naive endpoint table misleading, and both are removed here.

  1. THE LOSS COLUMN IS NOT THE TASK LOSS. `TrainLosses.json` records the quantity the optimiser
     descends, which for a penalised run is task + lambda*penalty, and is evaluated with noise ON.
     At N=2000 that column reads rws 0.02432 (apparently the WORST condition) while its noise-free
     task loss is 0.00745 (better than rws at N=500 or N=1000). The recorded column is ~65% noise
     floor plus a penalty term, so ordering conditions by it compares regularisation strengths, not
     performance. Here the task loss is evaluated with the noise switched OFF on a shared batch.

  2. THE BUDGETS DIFFER. The `none` baseline comes from the drift sweep (200k at N=500/1000, 300k at
     N=2000); the penalty sweep ran 200k/200k/150k. Silencing never settles - it is still moving
     4-11 percentage points per DOUBLING of the budget at the end of every unpenalised run - so a
     baseline given 2x the budget of the condition it is compared against is guaranteed to look
     worse. The silent-unit comparison is therefore read from the `none` PARTICIPATION TRACE at the
     iteration where the matching penalty run stopped, not at the baseline's own endpoint.

Held-out loss uses the interleaved coherence midpoints (the same construction run_experiment.py now
builds during training), so it is a genuinely unseen stimulus set rather than a fresh noise draw on
the training conditions.

Output: img/internal_figures/penalty_matched.png

Usage:  python penalty_matched.py
"""

import os
import re
import sys
import glob
import pickle
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
from common import IMG_DIR, active_count

PEN_SWEEP = "data/trained_RNNs/CDDM_std_g0_penalties"
NONE_SWEEP = "data/trained_RNNs/CDDM_std_g0_drift"
COND = ["none", "rws", "frm", "both"]
COL = {"none": "C7", "rws": "C0", "frm": "C3", "both": "C2"}


def build_batches(cfg):
    """Training batch and a held-out batch of interleaved coherence midpoints.

    Args:
        cfg: a resolved run config (any net's saved `*_config.yaml`).
    Returns:
        (mask, (inputs, target), (v_inputs, v_target)) - mask is the training timepoint index array,
        the two tuples are noise-free task batches with shapes (n_channels, T, B).
    """
    tcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    inputs, target, _ = hydra.utils.instantiate(tcfg).get_batch()

    vcfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    cohs = sorted(float(c) for c in cfg.task.coherences)
    vcfg.coherences = [0.5 * (cohs[i] + cohs[i + 1]) for i in range(len(cohs) - 1)]
    v_inputs, v_target, _ = hydra.utils.instantiate(vcfg).get_batch()

    return get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt), (inputs, target), (v_inputs, v_target)


def clean_loss(folder, mask, train, heldout):
    """Noise-free masked MSE of a net's final parameters on the training and held-out batches.

    Args:
        folder: net folder holding `*LastParams*.npz`; mask: training timepoint indices;
        train, heldout: (inputs, target) batches from build_batches.
    Returns:
        (N, train_mse, heldout_mse), or None if the folder has no saved parameters.
    """
    pf = glob.glob(os.path.join(folder, "*LastParams*.npz"))
    if not pf:
        return None
    d = np.load(pf[0], allow_pickle=True)
    params = {k: d[k] for k in d.files}
    params["activation_name"] = "relu"
    params.pop("activation_args", None)
    rnn = RNN_numpy(**filter_kwargs(RNN_numpy, params), seed=0)

    out = []
    for inputs, target in (train, heldout):
        rnn.clear_history()
        rnn.y = rnn.y_init
        rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
        o = rnn.get_output()
        out.append(float(((o[:, mask, :] - target[:, mask, :]) ** 2).mean()))
    return int(params["N"]), out[0], out[1]


def collect():
    """Gather every net from both sweeps, keyed by (N, condition).

    Returns:
        dict {(N, cond): [{"folder", "trace", "budget"}]}, where `budget` is the run's final
        iteration as recorded in the participation trace.
    """
    out = {}
    pats = [(PEN_SWEEP, r"_N=(\d+)_pen=(\w+)"), (NONE_SWEEP, r"_N=(\d+)_iters=(\d+)")]
    for sweep, pat in pats:
        for f in sorted(glob.glob(os.path.join(sweep, "*", "*", "*ParticipationTrace.pkl"))):
            m = re.search(pat, f)
            if not m:
                continue
            N = int(m.group(1))
            cond = m.group(2) if sweep == PEN_SWEEP else "none"
            with open(f, "rb") as fh:
                tr = pickle.load(fh)
            out.setdefault((N, cond), []).append(
                {"folder": os.path.dirname(f), "trace": tr,
                 "budget": int(np.max(tr["participation_iters"]))})
    # The drift sweep ran N=100 at two budgets; keep only the longest, else they pool as extra seeds.
    for N in {k[0] for k in out}:
        runs = out.get((N, "none"), [])
        if runs:
            best = max(r["budget"] for r in runs)
            out[(N, "none")] = [r for r in runs if r["budget"] == best]
    return out


def silent_at(trace, N, iteration):
    """Silent-unit percentages at the probe nearest a given iteration.

    Args:
        trace: a participation trace; N: network size; iteration: budget to read at.
    Returns:
        (hard %, scale-free %, active count under the hard criterion).
    """
    P, I = np.array(trace["participation"]), np.array(trace["participation_iters"])
    p = P[np.argmin(np.abs(I - iteration))]
    a = active_count(p, "hard")
    return 100 * (N - a) / N, 100 * (N - active_count(p, "scalefree")) / N, a


def main():
    """Report and plot the penalty comparison at matched budget on the clean task loss."""
    runs = collect()
    cfg = OmegaConf.load(glob.glob(os.path.join(PEN_SWEEP, "*", "*", "*_config.yaml"))[0])
    mask, train, heldout = build_batches(cfg)
    print(f"train batch {train[0].shape}, held-out {heldout[0].shape} "
          f"({len(cfg.task.coherences)} -> {heldout[0].shape[-1] // (train[0].shape[-1] // len(cfg.task.coherences))} coherences)")

    Ns = sorted({N for N, c in runs if (N, "rws") in runs})
    res = {}
    for N in Ns:
        # Matched budget: the shortest run present at this size across all four conditions.
        budget = min(r["budget"] for c in COND if (N, c) in runs for r in runs[(N, c)])
        for c in COND:
            if (N, c) not in runs:
                continue
            rows = []
            for r in runs[(N, c)]:
                h, s, a = silent_at(r["trace"], N, budget)
                cl = clean_loss(r["folder"], mask, train, heldout)
                rows.append((h, s, a, cl[1] if cl else np.nan, cl[2] if cl else np.nan))
            res[(N, c)] = (np.array(rows), budget)

    hdr = ("%6s %-5s %8s %14s %14s %12s %12s %12s"
           % ("N", "pen", "budget", "silent hard %", "silent sf %", "active", "clean train", "clean held"))
    print("\n" + hdr + "\n" + "-" * len(hdr))
    for N in Ns:
        for c in COND:
            if (N, c) not in res:
                continue
            v, budget = res[(N, c)]
            print("%6d %-5s %8d %7.1f +- %-4.1f %7.1f +- %-4.1f %7.0f +-%-4.0f %12.5f %12.5f"
                  % (N, c, budget, v[:, 0].mean(), v[:, 0].std(), v[:, 1].mean(), v[:, 1].std(),
                     v[:, 2].mean(), v[:, 2].std(), np.nanmean(v[:, 3]), np.nanmean(v[:, 4])))
        print()

    fig, ax = plt.subplots(1, 4, figsize=(23, 5.2))
    # frm and both both sit exactly on 0% silent / M=N, so one would hide the other; nudge in x.
    off = {"none": 1.0, "rws": 1.0, "frm": 0.97, "both": 1.03}
    for c in COND:
        xs = [N for N in Ns if (N, c) in res]
        if not xs:
            continue
        get = lambda col: ([res[(N, c)][0][:, col].mean() for N in xs],
                           [res[(N, c)][0][:, col].std() for N in xs])
        xo = [N * off[c] for N in xs]
        for col, k in ((0, 0), (1, 1), (2, 2)):
            mu, sd = get(k)
            ax[col].errorbar(xo, mu, yerr=sd, fmt="o-", color=COL[c], ms=8, capsize=4, lw=2, label=c)
        for k, ls, al in ((3, "-", 1.0), (4, "--", .55)):
            mu, sd = get(k)
            ax[3].errorbar(xs, mu, yerr=sd, fmt="o" + ls, color=COL[c], ms=7, capsize=3, lw=2,
                           alpha=al, label=c if ls == "-" else None)

    ax[0].set(xscale="log", xlabel="$N$", ylabel="silent units (% of N)", ylim=(-3, 100),
              title="(a) hard criterion $p_i<10^{-6}$\nat MATCHED budget")
    ax[1].set(xscale="log", xlabel="$N$", ylabel="silent units (% of N)", ylim=(-3, 100),
              title="(b) scale-free $p_i<0.05\\,q_{95}(p)$\nrws REVERSES between (a) and (b)")
    ax[2].set(xscale="log", yscale="log", xlabel="$N$", ylabel="active units $M$",
              title="(c) active units (hard criterion)\ngrey dashed = $M=N$")
    lim = np.array([min(Ns) * .8, max(Ns) * 1.3])
    ax[2].plot(lim, lim, "k--", lw=1, alpha=.5)
    ax[3].set(xscale="log", xlabel="$N$", ylabel="noise-free masked MSE",
              title="(d) CLEAN TASK loss, final weights\nsolid = train batch, faded = held-out")
    for a in ax:
        a.legend(fontsize=9)
        a.grid(alpha=.3)
    fig.tight_layout()
    out = os.path.join(IMG_DIR, "penalty_matched.png")
    fig.savefig(out, dpi=150)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
