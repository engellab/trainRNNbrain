#!/usr/bin/env python3
"""
Is "truly silent" (p_i < 1e-6) a property of the NETWORK, or of the finite set of trials it is
measured on?

THE OBSERVATION THAT PROMPTS THIS. On CDDM, 72-83% of units sit at participation EXACTLY 0.0. On the
n-bit flip-flop only 0.6-0.9% fall below 1e-6, even though the SCALE-FREE silent fraction is nearly
identical in the two tasks (0.84 vs 0.85 at N=2000). So the two tasks agree about how many units are
negligible and disagree completely about how many are exactly zero.

The model configs are identical (same architecture, noise, bias, equation). What differs is the
INPUT SET:

    CDDM        450 enumerable conditions, and `same_batch=True` - the network trains on those exact
                trials and participation is then measured on the SAME ones.
    flip-flop   continuous i.i.d. pulse timings, `same_batch=False` - a fresh batch every iteration,
                and participation measured on yet another fresh batch.

HYPOTHESIS. A ReLU unit reaches p_i = 0 only if its pre-activation stays <= 0 at every timestep of
every trial it is shown. That is achievable against 450 fixed trials and essentially unachievable
against a continuous distribution, because some rare input combination eventually pushes it up. If
so, CDDM's exact zeros are a property of the TASK SET, not of the network, and the task difference is
not a law about task complexity at all.

THE TEST, and the falsifier stated first. Take trained CDDM networks and re-measure participation on
HELD-OUT coherences - the midpoints between trained ones, which the network never saw. Then:

  hard-silent fraction collapses on held-out input   -> the zeros are a task-set artifact. The
                                                        CDDM/flip-flop difference is about input
                                                        enumerability, not about the task.
  hard-silent fraction essentially unchanged         -> HYPOTHESIS FALSIFIED. The units are dead
                                                        regardless of input, the zeros are a real
                                                        property of the network, and the flip-flop
                                                        difference needs another explanation.

Threshold fixed in advance: a drop of more than 20% of the hard-silent fraction (relative) counts as
collapse; less than 5% counts as unchanged. Anything between is indeterminate and gets reported so.

The scale-free fraction is reported alongside throughout, since it is the quantity the two tasks
already agree on and should be roughly stable under this manipulation either way.

Output: img/internal_figures/silence_is_taskset.png

Usage:  python silence_is_taskset.py [SWEEP_FOLDER]
"""

import os
import sys
import glob
import json
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import IMG_DIR, participation, active_count
import plotstyle as ps
from trainRNNbrain.training.training_utils import prepare_task_arguments
from plot_init_vs_trained import rnn_numpy_from_params

DEFAULT_SWEEP = "data/trained_RNNs/CDDM_std_g0_drift"
DROP_COLLAPSE = 0.20      # relative fall in hard-silent fraction that counts as collapse
DROP_UNCHANGED = 0.05     # ... and below which it counts as unchanged


def batches(config_path):
    """Build the TRAINED CDDM input batch and a HELD-OUT one at interpolated coherences.

    The held-out set uses the midpoints between consecutive trained coherences, so it lies inside the
    trained range (no extrapolation) but contains no trial the network has ever seen.

    Args:
        config_path: path to a saved <score>_config.yaml.
    Returns:
        (trained_batch, heldout_batch), each ndarray (n_inputs, T, n_conditions).
    """
    cfg = OmegaConf.load(config_path)
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task = hydra.utils.instantiate(task_cfg)
    trained, _, _ = task.get_batch()

    coh = sorted(float(c) for c in cfg.task.coherences)
    mids = [(a + b) / 2 for a, b in zip(coh[:-1], coh[1:])]
    cfg2 = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg2.task.coherences = mids
    task2 = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg2.task, dt=cfg2.model.dt))
    heldout, _, _ = task2.get_batch()
    return trained, heldout


def load_params(path, config_path=None):
    """Load a saved parameter file, accepting either the .npz or the .json layout.

    ⚠️ THE .npz FILES LOSE `activation_args`. It is a dict, and `np.savez` stores a dict argument as
    an array of its KEYS - the saved value is literally `array(['name', 'slope'])`, with the values
    discarded. Any dict-valued parameter has the same problem. It is therefore restored from the
    saved config, which does keep it; without that, reconstructing the network raises deep inside
    RNN_numpy with an opaque indexing error.

    Args:
        path: path to a <score>_LastParams_*.{npz,json};
        config_path: the net's saved config, used to restore dict-valued parameters.
    Returns:
        dict of parameters, arrays as ndarray and scalars unwrapped from 0-d arrays.
    """
    if path.endswith(".npz"):
        d = np.load(path, allow_pickle=True)
        out = {k: (d[k].item() if d[k].ndim == 0 else d[k]) for k in d.files}
    else:
        with open(path) as fh:
            p = json.load(fh)
        out = {k: (np.array(v) if isinstance(v, list) else v) for k, v in p.items()}
    if config_path:
        m = OmegaConf.load(config_path).model
        if "activation_args" in m:
            out["activation_args"] = OmegaConf.to_container(m.activation_args, resolve=True)
    return out


def rates(params_path, input_batch, config_path=None):
    """Noise-free firing rates of a trained network on one input batch.

    Args:
        params_path: path to a <score>_LastParams_*.{npz,json};
        input_batch: (n_inputs, T, n_conditions); config_path: saved config, for activation_args.
    Returns:
        ndarray (N, T, n_conditions) of rates.
    """
    rnn = rnn_numpy_from_params(load_params(params_path, config_path))
    rnn.clear_history()
    rnn.run(input_timeseries=input_batch, sigma_rec=0, sigma_inp=0)
    return np.array(rnn.get_history())


def main():
    """Re-measure silence on held-out coherences and report against the pre-set thresholds."""
    ps.setup()
    sweep = (sys.argv[1:] or [DEFAULT_SWEEP])[0]
    leaves = sorted(d for d in glob.glob(os.path.join(sweep, "EqType=*", "*")) if os.path.isdir(d))
    if not leaves:
        raise SystemExit(f"no networks under {sweep}")

    rows = []
    print(f"pre-registered: >{100*DROP_COLLAPSE:.0f}% relative drop = task-set artifact; "
          f"<{100*DROP_UNCHANGED:.0f}% = zeros are real\n")
    print("%6s %10s  %-22s %-22s" % ("N", "seed", "hard silent frac", "scale-free silent frac"))
    print("%6s %10s  %10s %10s  %10s %10s" % ("", "", "trained", "held-out", "trained", "held-out"))
    for d in leaves:
        pj = (glob.glob(os.path.join(d, "*_LastParams_*.npz"))
              or glob.glob(os.path.join(d, "*_LastParams_*.json")))
        cfg = glob.glob(os.path.join(d, "*_config.yaml"))
        if not pj or not cfg:
            continue
        N = int(OmegaConf.load(cfg[0]).model.N)
        try:
            tr_b, ho_b = batches(cfg[0])
            p_tr = participation(rates(pj[0], tr_b, cfg[0]))
            p_ho = participation(rates(pj[0], ho_b, cfg[0]))
        except Exception as e:
            print(f"  [skip] {os.path.basename(d)[:9]}: {e}")
            continue
        h_tr = 1 - active_count(p_tr, "hard") / N
        h_ho = 1 - active_count(p_ho, "hard") / N
        s_tr = 1 - active_count(p_tr, "scalefree") / N
        s_ho = 1 - active_count(p_ho, "scalefree") / N
        rows.append((N, h_tr, h_ho, s_tr, s_ho))
        print("%6d %10s  %10.3f %10.3f  %10.3f %10.3f"
              % (N, os.path.basename(d)[:9], h_tr, h_ho, s_tr, s_ho))

    if not rows:
        raise SystemExit("no networks could be evaluated")
    A = np.array([[r[1], r[2], r[3], r[4]] for r in rows])
    drop_hard = 1 - A[:, 1].mean() / max(A[:, 0].mean(), 1e-12)
    drop_sf = 1 - A[:, 3].mean() / max(A[:, 2].mean(), 1e-12)

    print(f"\nmean hard-silent fraction:  trained {A[:,0].mean():.3f} -> held-out {A[:,1].mean():.3f}"
          f"   ({100*drop_hard:+.1f}% relative)")
    print(f"mean scale-free fraction:   trained {A[:,2].mean():.3f} -> held-out {A[:,3].mean():.3f}"
          f"   ({100*drop_sf:+.1f}% relative)")
    if drop_hard > DROP_COLLAPSE:
        verdict = ("COLLAPSE -> the exact zeros are a TASK-SET artifact; CDDM and the flip-flop "
                   "differ because CDDM's input set is finite, not because of the task itself")
    elif drop_hard < DROP_UNCHANGED:
        verdict = ("UNCHANGED -> HYPOTHESIS FALSIFIED. The units are dead for any input, the zeros "
                   "are a real network property, and the flip-flop difference needs another cause")
    else:
        verdict = "INDETERMINATE - between the pre-set thresholds; neither reading is supported"
    print(f"\nVERDICT: {verdict}")

    Ns = sorted({r[0] for r in rows})
    fig, ax = plt.subplots(1, 2, figsize=(12.5, 5.2))
    for j, (lab, i0, i1) in enumerate([("hard  $p_i<10^{-6}$", 0, 1),
                                       ("scale-free  $p_i<0.05q_{95}$", 2, 3)]):
        for N in Ns:
            v = np.array([[r[1 + i0], r[1 + i1]] for r in rows if r[0] == N])
            ax[j].plot([0, 1], v.T, "-o", color=ps.col_n(N), alpha=.75, ms=6)
        ax[j].set(xticks=[0, 1], xticklabels=["trained\ncoherences", "held-out\ncoherences"],
                  ylabel="silent fraction", ylim=(0, 1), title=lab)
        ps.legend_n(ax[j], Ns, loc="best")
    fig.suptitle("Is exact silence a property of the network, or of the trials it is measured on?\n"
                 f"CDDM networks re-measured on unseen coherence midpoints — {verdict[:60]}...",
                 fontsize=11)
    return ps.save(fig, "silence_is_taskset")


if __name__ == "__main__":
    main()
