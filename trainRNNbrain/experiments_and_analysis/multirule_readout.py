"""Per-task recruitment of a multi-task (TaskMultiRule) network: how many units does CDDM use
when the same network also serves fourteen other tasks?

Pre-registered read-out of the MultiRule_std_multi grid (slurm/SilentReLU_multirule_spock.slurm).
For every trained network (LastParams .npz + saved config):
  1. Rebuild the network in numpy and the composite task from the saved config.
  2. For every subtask, run its FULL single-task batch (rule on, noise-free) and compute the
     per-unit participation p_i = std(r_i) + q_0.9(|r_i|) over (time, trials), the same statistic
     the trainer logs. Count active units under three criteria, always reported together:
     scale-free (p >= 5% of q95), absolute 1e-6 (CDDM-calibrated) and absolute 4e-2
     (flip-flop-calibrated) - the two absolute ones bracket the task-dependence of that threshold.
  3. The same on a mixed batch (live_all: what the whole network uses).
  4. The CDDM-active set split into units ALSO active on at least one other task (shared) and
     units active on CDDM only (private), scale-free criterion. The split separates "CDDM's own
     computation became more redundant" from "other tasks' units respond to CDDM's inputs".
  5. Per-task r2 on the task's own scoring window, noise-free, so an unlearned subtask is visible.
The single-task CDDM references (CDDM_std_g0_drift / CDDM_std_g0_penalties at the same N and
iteration) are read separately from their participation traces (count_silent_units.py); this
script prints only the multi-task side.

Usage: python multirule_readout.py <trained_RNNs root> [--sub MultiRule_std_multi] [--dump out.npz]
  --dump  save every net's per-task participation vectors and r2 for figures.
"""
import argparse
import glob
import os
import re

import hydra
import numpy as np
from omegaconf import OmegaConf

from trainRNNbrain.experiments_and_analysis.common import SILENT_FLIPFLOP, active_count, participation
from trainRNNbrain.experiments_and_analysis.silence_is_taskset import load_params
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments, r2
from trainRNNbrain.utils import filter_kwargs

if not OmegaConf.has_resolver("eval"):
    OmegaConf.register_new_resolver("eval", eval)


def build(net_dir):
    """Rebuild the numpy RNN and the composite task of one trained network.

    Args:
        net_dir: per-network folder holding *_LastParams_*.npz and *_config.yaml.
    Returns:
        (rnn, task, cfg) - RNN_numpy, TaskMultiRule, the saved OmegaConf config.
    """
    npz = glob.glob(os.path.join(net_dir, "*_LastParams_*.npz"))[0]
    cfg_path = glob.glob(os.path.join(net_dir, "*_config.yaml"))[0]
    cfg = OmegaConf.load(cfg_path)
    params = load_params(npz, cfg_path)
    params["equation_type"] = cfg.model.equation_type
    rnn = RNN_numpy(**filter_kwargs(RNN_numpy, params))
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    task_cfg.seed = 0                                   # fixed trial draw for every net
    task = hydra.utils.instantiate(task_cfg)
    return rnn, task, cfg


def run_noise_free(rnn, X):
    """Noise-free firing rates and outputs of the network on a batch.

    Args:
        rnn: RNN_numpy; X: inputs (n_inputs, T, B).
    Returns:
        (fr (N, T, B), out (n_outputs, T, B)).
    """
    rnn.clear_history()
    rnn.y = rnn.y_init.copy()
    rnn.run(X, sigma_rec=0, sigma_inp=0)
    fr = rnn.get_firing_rate_history()
    out = np.einsum("oj,jtb->otb", rnn.W_out, fr)
    return fr, out


def r2_scored(out, Y, mask):
    """r2 over the scored entries of a batch, pooled over outputs and trials.

    Args:
        out, Y: (n_outputs, T, B); mask: (T, B) bool.
    Returns:
        float.
    """
    sel = out[:, mask].ravel(), Y[:, mask].ravel()
    return float(r2(sel[0], sel[1]))


def readout(net_dir):
    """All read-out quantities of one network.

    Args:
        net_dir: per-network folder.
    Returns:
        dict with per-task participation vectors (`p`, name -> (N,)), per-task r2 (`r2`),
        the mixed-batch participation (`p_all`), and the CDDM shared/private split.
    """
    rnn, task, cfg = build(net_dir)
    p, r2s = {}, {}
    for i, name in enumerate(task.subtask_names):
        X, Y, C = task.task_batch(i)
        fr, out = run_noise_free(rnn, X)
        p[name] = participation(fr)
        r2s[name] = r2_scored(out, Y, task.batch_mask(C))
    X, Y, C = task.get_batch()
    fr, out = run_noise_free(rnn, X)
    p_all = participation(fr)
    act = {n: p[n] >= 0.05 * np.quantile(p[n], 0.95) for n in p}
    cddm = act["CDDM"]
    others = np.any([act[n] for n in p if n != "CDDM"], axis=0)
    return dict(N=int(cfg.model.N), pen=re.search(r"_pen=([a-z]+)", net_dir).group(1),
                seed=int(cfg.seed), p=p, r2=r2s, p_all=p_all, r2_all=r2_scored(out, Y, task.batch_mask(C)),
                cddm_active=int(cddm.sum()), cddm_shared=int((cddm & others).sum()),
                cddm_private=int((cddm & ~others).sum()))


def main():
    """Print one block per network: per-task live counts (three criteria) and r2, then the totals."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sub", default="MultiRule_std_multi")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, a.sub, "*", "*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no networks under {os.path.join(a.root, a.sub)}")
    dump = {}
    for d in dirs:
        r = readout(d)
        print(f"\n=== N={r['N']} pen={r['pen']} seed={r['seed']}   {os.path.basename(d)[:40]}")
        print(f"{'task':18} {'live_sf':>7} {'live_1e-6':>9} {'live_4e-2':>9} {'r2':>7}")
        for name, pv in r["p"].items():
            print(f"{name:18} {active_count(pv, 'scalefree'):>7d} {active_count(pv, 'hard'):>9d} "
                  f"{active_count(pv, SILENT_FLIPFLOP):>9d} {r['r2'][name]:7.3f}")
        pa = r["p_all"]
        print(f"{'ALL (mixed batch)':18} {active_count(pa, 'scalefree'):>7d} {active_count(pa, 'hard'):>9d} "
              f"{active_count(pa, SILENT_FLIPFLOP):>9d} {r['r2_all']:7.3f}")
        print(f"CDDM-active (scale-free) {r['cddm_active']}: shared with >=1 other task {r['cddm_shared']}, "
              f"CDDM-private {r['cddm_private']}")
        if a.dump:
            key = f"{r['N']}_{r['pen']}_{r['seed']}"
            for name, pv in r["p"].items():
                dump[f"{key}_p_{name}"] = pv
                dump[f"{key}_r2_{name}"] = r["r2"][name]
            dump[f"{key}_p_all"] = pa
    if a.dump:
        np.savez_compressed(a.dump, **dump)
        print(f"\nwritten {a.dump}")


if __name__ == "__main__":
    main()
