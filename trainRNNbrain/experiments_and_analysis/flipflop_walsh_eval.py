"""Read-out for NBitFlipFlopWalsh networks: r^2, the Walsh spectrum the network reproduces (by
subset order), and the live-unit count, per net folder.

The spectrum is the point of the task: the target is sum_S c_S prod_S b, and projecting the
network's output onto the same basis says which of the 2^k-1 latent products it actually built,
order by order, with no criterion on individual units. Reported per order as the fraction of the
target's coefficient energy the network recovers (1.0 = that order fully built; the target's own
energy per order is printed for reference) and as the r^2 the recovered spectrum accounts for.

Usage: python flipflop_walsh_eval.py <folder or glob of net folders> [n_trials]
"""
import os
import sys
import glob
import numpy as np
import hydra
from omegaconf import OmegaConf

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, active_count, participation
from flipflop_fixedpoints import load_net
from trainRNNbrain.training.training_utils import prepare_task_arguments

EVAL_SEED = 12345   # fixed trial set for every network


def evaluate(folder, n_trials=512):
    """Simulate one saved network on a fresh fixed trial set.

    Args:
        folder: net folder with *LastParams*.npz and *_config.yaml; n_trials: batch size.
    Returns:
        dict with r2, live counts (scale-free / absolute 4e-2), recovered fraction of coefficient
        energy per order, the target's energy per order, and the r^2 explained per order.
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    cfg.task.batch_size = n_trials
    cfg.task.seed = EVAL_SEED
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, targets, _ = task.get_batch()
    rnn, _ = load_net(folder)
    rnn.clear_history(); rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    rates = np.maximum(np.array(rnn.get_history()), 0.0)          # (N, T, B)
    y_hat = np.array(rnn.get_output())[0]                           # (T, B)
    y = targets[0]
    r2 = 1.0 - ((y_hat - y) ** 2).mean() / y.var()
    bits = task.bit_states(inputs)
    rec, unvisited = task.spectrum(y_hat, bits)
    orders = np.array([len(s) for s in task.subsets])
    c = task.coefs
    frac, energy, r2_order = {}, {}, {}
    for m in range(1, task.n_inputs + 1):
        sel = orders == m
        e = (c[sel] ** 2).sum()
        energy[m] = float(e)
        # projection of the recovered coefficients onto the true ones, relative to the true energy
        frac[m] = float((rec[sel] * c[sel]).sum() / e) if e > 0 else float("nan")
        r2_order[m] = float(1.0 - ((rec[sel] - c[sel]) ** 2).sum() / e) if e > 0 else float("nan")
    p = participation(rates)
    return dict(r2=float(r2), live_sf=active_count(p, "scalefree"), live_abs=active_count(p, SILENT_FLIPFLOP),
                N=rates.shape[0], frac=frac, energy=energy, r2_order=r2_order, unvisited=unvisited,
                spurious=float(((rec - c) ** 2).sum()))


def main(pattern, n_trials=512):
    """Evaluate every net folder matching `pattern` and print one block per folder."""
    folders = sorted(f for f in glob.glob(pattern) if glob.glob(os.path.join(f, "*LastParams*.npz")))
    for f in folders:
        r = evaluate(f, n_trials)
        print(f"{f}\n  r2 {r['r2']:.3f}   live {r['live_sf']} (scale-free) / {r['live_abs']} (abs 4e-2) of {r['N']}"
              f"   unvisited states {r['unvisited']}   spurious coef energy {r['spurious']:.3f}")
        print("  order : " + " ".join(f"{m:>6d}" for m in r['frac']))
        print("  target energy: " + " ".join(f"{v:6.2f}" for v in r['energy'].values()))
        print("  recovered    : " + " ".join(f"{v:6.2f}" for v in r['frac'].values()))
        print("  r2 per order : " + " ".join(f"{v:6.2f}" for v in r['r2_order'].values()))


if __name__ == "__main__":
    main(sys.argv[1], int(sys.argv[2]) if len(sys.argv) > 2 else 512)
