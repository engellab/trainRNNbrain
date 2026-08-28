#!/usr/bin/env python3
"""
Performance on the hardest INFORMATIVE trials (0 < |coh| < 0.02) as a function of network size.

WHY THIS BIN. 98% of the clean training loss comes from |coh| < 0.05, and 76% from this bin alone,
so it is where any size effect on task performance must live. It is kept separate from coh = 0
because the two are qualitatively different: at coh = 0 the input carries NO information and the
target (-1) is arbitrary, so that bin can only be memorised, never computed. Here there is a real
but tiny signal, which the network has to resolve against the injected noise - the regime where a
noise-averaging advantage of larger networks should be largest if it exists.

Only these conditions are simulated (the batch is subset before the forward pass), which is ~7x
cheaper than running the full 450-condition grid and discarding most of it.

Output: img/internal_figures/lowcoh_vs_N.png

Usage:  python plot_lowcoh_vs_N.py [SWEEP_FOLDER]
"""

import os
import sys
import glob
import numpy as np
import hydra
from omegaconf import OmegaConf
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments, get_training_mask
from trainRNNbrain.utils import filter_kwargs

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR

DRAWS = 12


def main():
    """Evaluate every trained network on the 0 < |coh| < 0.02 trials, noise off and on."""
    sweep = ([a for a in sys.argv[1:] if not a.startswith("--")] or
             ["data/trained_RNNs/CDDM_std_g0_drift"])[0]
    cfg = OmegaConf.load(glob.glob(os.path.join(sweep, "*", "*", "*_config.yaml"))[0])
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    sr, si = float(cfg.model.sigma_rec), float(cfg.model.sigma_inp)
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inp, tgt, cond = task.get_batch()
    rel = np.array([abs(c["motion_coh"] if c["context"] == "motion" else c["color_coh"])
                    for c in cond])
    sel = (rel > 1e-9) & (rel < 0.02)
    inp, tgt = inp[:, :, sel], tgt[:, :, sel]
    print(f"using {sel.sum()} of {len(rel)} conditions: 0 < |coh| < 0.02 "
          f"(|coh| = {sorted(set(np.round(rel[sel], 6)))})")

    def err(p, sigma, seed):
        r = RNN_numpy(**filter_kwargs(RNN_numpy, p), seed=seed)
        r.clear_history()
        r.y = r.y_init
        r.run(input_timeseries=inp, sigma_rec=sigma[0], sigma_inp=sigma[1])
        return float(((r.get_output()[:, mask, :] - tgt[:, mask, :]) ** 2).mean())

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
        c = err(p, (0.0, 0.0), 0)
        n_ = float(np.mean([err(p, (sr, si), s) for s in range(DRAWS)]))
        res.setdefault(N, []).append((c, n_))
        print(f"  N={N:5d}  clean {c:.5f}  noisy {n_:.5f}  excess {n_-c:.5f}")

    Ns = sorted(res)
    print(f"\n{'N':>7} {'noise OFF':>18} {'noise ON':>18} {'noise excess':>18} {'n':>3}")
    for N in Ns:
        c = np.array([a for a, _ in res[N]])
        n_ = np.array([b for _, b in res[N]])
        print(f"{N:>7} {c.mean():>10.5f}+-{c.std():.5f} {n_.mean():>10.5f}+-{n_.std():.5f} "
              f"{(n_-c).mean():>10.5f}+-{(n_-c).std():.5f} {len(c):>3}")

    xs = np.log10([N for N in Ns for _ in res[N]])
    for nm, i in (("noise OFF", 0), ("noise ON", 1)):
        y = np.array([r[i] for N in Ns for r in res[N]])
        r = stats.linregress(xs, y)
        print(f"  regression {nm:>9}: slope {r.slope:+.5f}/decade, p = {r.pvalue:.3f}")
    y = np.array([r[1] - r[0] for N in Ns for r in res[N]])
    r = stats.linregress(xs, y)
    print(f"  regression   excess: slope {r.slope:+.5f}/decade, p = {r.pvalue:.3f}")

    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5))
    for i, (c, lab) in enumerate([("C0", "noise OFF (deterministic)"), ("C3", "noise ON")]):
        mu = [np.mean([r[i] for r in res[N]]) for N in Ns]
        sd = [np.std([r[i] for r in res[N]]) for N in Ns]
        ax[0].errorbar(Ns, mu, yerr=sd, fmt="o-", color=c, ms=8, capsize=4, lw=2, label=lab)
        for N in Ns:
            ax[0].plot([N] * len(res[N]), [r[i] for r in res[N]], "o", color=c, ms=4, alpha=.35)
    mu = [np.mean([r[1] - r[0] for r in res[N]]) for N in Ns]
    sd = [np.std([r[1] - r[0] for r in res[N]]) for N in Ns]
    ax[1].errorbar(Ns, mu, yerr=sd, fmt="s-", color="C2", ms=8, capsize=4, lw=2)
    ax[0].set(xscale="log", xlabel="$N$", ylabel="masked MSE",
              title="(a) error on the hardest informative trials")
    ax[0].legend(fontsize=9)
    ax[1].set(xscale="log", xlabel="$N$", ylabel="noisy $-$ clean MSE",
              title="(b) noise-induced excess\nlower = better noise tolerance")
    for a in ax:
        a.grid(alpha=.3)
    fig.suptitle(r"Trials with $0<|coh|<0.02$ only — 76% of the total training loss"
                 f"\n({sel.sum()} conditions; mean $\\pm$ sd over seeds, N=2000 has n=1)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = os.path.join(IMG_DIR, "lowcoh_vs_N.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
