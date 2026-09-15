"""Per-task recruitment of a multi-task network (TaskYang, or any task exposing the composite API
`subtask_names` / `task_batch(i)` / `batch_mask`): how many units does the FOCUS task (CDDM =
contextdm1 + contextdm2, Mante's task in ring coding) use when the same network also serves the
other tasks? `--focus` takes one rule or a comma-separated set; the focus-active set is the union.

Pre-registered read-out of the Yang_std_multi grid (slurm/SilentReLU_yang_spock.slurm). The
single-task reference (Yang_ctxdm networks, same layout, the two context rules) is read with the SAME script,
so the two sides use one code path.
For every trained network (LastParams .npz + saved config):
  1. Rebuild the network in numpy and the composite task from the saved config.
  2. For every subtask, run its FULL single-task batch (rule on, noise-free) and compute the
     per-unit participation p_i = std(r_i) + q_0.9(|r_i|) over (time, trials), the same statistic
     the trainer logs. Count active units under three criteria, always reported together:
     scale-free (p >= 5% of q95), absolute 1e-6 (CDDM-calibrated) and absolute 4e-2
     (flip-flop-calibrated) - the two absolute ones bracket the task-dependence of that threshold.
  3. The same on a mixed batch (live_all: what the whole network uses).
  4. The focus-task-active set split into units ALSO active on at least one other task (shared)
     and units active on the focus task only (private), scale-free criterion. The split separates
     "the focus task's own computation became more redundant" from "other tasks' units respond to
     its inputs". (Trivially all-shared in a single-task net.)
  5. Per-task r2 on the task's own scoring window, noise-free, so an unlearned subtask is visible.
  6. Per-task ACCURACY (TaskYang only), Yang et al.'s performance criterion: on a responding trial
     the population vector of the response ring, averaged over the second half of the response
     epoch, must point within 36 deg (0.2 pi) of the correct direction AND the fixation output
     must be < 0.5; on a no-response trial the fixation output must stay > 0.5 and the ring max
     < 0.5. Chance for a responding trial is 10%. Pooled r2 is dominated by trials the network
     gets roughly right and cannot tell "solved" from "half the trials wrong"; accuracy can.
Usage: python multitask_readout.py <trained_RNNs root> [--sub Yang_std_multi] [--focus contextdm1,contextdm2]
                                   [--n-trials 256] [--dump out.npz]
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


def accuracy(task, out, Y, C):
    """Fraction of trials answered correctly, Yang-style (see module docstring, item 6).

    Args:
        task: TaskYang (uses task.pref); out: network outputs (n_outputs, T, B); Y: targets;
        C: conditions with "sub" holding t_go, t_end, respond, resp_dir.
    Returns:
        float accuracy in [0, 1], or nan if the task is not a TaskYang.
    """
    if not hasattr(task, "pref"):
        return float("nan")
    ok = []
    z = np.exp(1j * task.pref)
    for b, c in enumerate(C):
        s = c["sub"]
        t0 = (s["t_go"] + s["t_end"]) // 2
        ring = out[1:, t0:s["t_end"], b].mean(axis=1)
        fix = out[0, t0:s["t_end"], b].mean()
        if s["respond"]:
            ang = np.angle((np.clip(ring, 0, None) * z).sum())
            err = np.abs(np.angle(np.exp(1j * (ang - s["resp_dir"]))))
            ok.append(err < 0.2 * np.pi and fix < 0.5)
        else:
            ok.append(fix > 0.5 and ring.max() < 0.5)
    return float(np.mean(ok))


def r2_scored(out, Y, mask):
    """r2 over the scored entries of a batch, pooled over outputs and trials.

    Args:
        out, Y: (n_outputs, T, B); mask: (T, B) bool.
    Returns:
        float.
    """
    sel = out[:, mask].ravel(), Y[:, mask].ravel()
    return float(r2(sel[0], sel[1]))


def readout(net_dir, focus, n_trials):
    """All read-out quantities of one network.

    Args:
        net_dir: per-network folder; focus: the rule whose shared/private split is reported;
        n_trials: trials per rule for the per-task batches.
    Returns:
        dict with per-task participation vectors (`p`, name -> (N,)), per-task r2 (`r2`),
        the mixed-batch participation (`p_all`), and the focus task's shared/private split.
    """
    rnn, task, cfg = build(net_dir)
    p, r2s, acc = {}, {}, {}
    for i, name in enumerate(task.subtask_names):
        X, Y, C = task.task_batch(i, n_trials)
        fr, out = run_noise_free(rnn, X)
        p[name] = participation(fr)
        r2s[name] = r2_scored(out, Y, task.batch_mask(C))
        acc[name] = accuracy(task, out, Y, C)
    X, Y, C = task.get_batch()
    fr, out = run_noise_free(rnn, X)
    p_all = participation(fr)
    act = {n: p[n] >= 0.05 * np.quantile(p[n], 0.95) for n in p}
    foc = focus.split(",")                                  # one rule, or a set (contextdm1,contextdm2)
    cddm = np.any([act[n] for n in foc], axis=0)
    rest = [act[n] for n in p if n not in foc]
    others = np.any(rest, axis=0) if rest else np.zeros_like(cddm)
    return dict(N=int(cfg.model.N), pen=re.search(r"_pen=([a-z]+)", net_dir).group(1),
                seed=int(cfg.seed), p=p, r2=r2s, acc=acc, p_all=p_all, r2_all=r2_scored(out, Y, task.batch_mask(C)),
                acc_all=accuracy(task, out, Y, C),
                cddm_active=int(cddm.sum()), cddm_shared=int((cddm & others).sum()),
                cddm_private=int((cddm & ~others).sum()))


def main():
    """Print one block per network: per-task live counts (three criteria) and r2, then the totals."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--sub", default="Yang_std_multi")
    ap.add_argument("--focus", default="contextdm1,contextdm2")
    ap.add_argument("--n-trials", type=int, default=256)
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()
    dirs = sorted(d for d in glob.glob(os.path.join(a.root, a.sub, "*", "*")) if os.path.isdir(d))
    if not dirs:
        raise SystemExit(f"no networks under {os.path.join(a.root, a.sub)}")
    dump = {}
    for d in dirs:
        r = readout(d, a.focus, a.n_trials)
        print(f"\n=== N={r['N']} pen={r['pen']} seed={r['seed']}   {os.path.basename(d)[:40]}")
        print(f"{'task':18} {'live_sf':>7} {'live_1e-6':>9} {'live_4e-2':>9} {'r2':>7} {'accuracy':>8}")
        for name, pv in r["p"].items():
            print(f"{name:18} {active_count(pv, 'scalefree'):>7d} {active_count(pv, 'hard'):>9d} "
                  f"{active_count(pv, SILENT_FLIPFLOP):>9d} {r['r2'][name]:7.3f} {r['acc'][name]:8.2f}")
        pa = r["p_all"]
        print(f"{'ALL (mixed batch)':18} {active_count(pa, 'scalefree'):>7d} {active_count(pa, 'hard'):>9d} "
              f"{active_count(pa, SILENT_FLIPFLOP):>9d} {r['r2_all']:7.3f} {r['acc_all']:8.2f}")
        print(f"{a.focus}-active (scale-free) {r['cddm_active']}: shared with >=1 other task "
              f"{r['cddm_shared']}, {a.focus}-private {r['cddm_private']}")
        if a.dump:
            key = f"{r['N']}_{r['pen']}_{r['seed']}"
            for name, pv in r["p"].items():
                dump[f"{key}_p_{name}"] = pv
                dump[f"{key}_r2_{name}"] = r["r2"][name]
                dump[f"{key}_acc_{name}"] = r["acc"][name]
            dump[f"{key}_p_all"] = pa
    if a.dump:
        np.savez_compressed(a.dump, **dump)
        print(f"\nwritten {a.dump}")


if __name__ == "__main__":
    main()
