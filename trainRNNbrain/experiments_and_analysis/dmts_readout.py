"""Read-out of the DMTS_long penalty grid (DMTS_std_pen): live units, loss floor, convergence times.

Question: on a task that needs SUSTAINED memory (16-tau silent delay between sample and match), does
the frm / rws penalty change how many units are recruited, the achieved loss, and how FAST the
network trains? Pre-registered in slurm/SilentReLU_dmts_penalties_spock.slurm.

Method, per network, from its ParticipationTrace.pkl (noise-free probe of the training batch every
10 iterations; the participation vector every 100):
  live_sf   active units at the last snapshot at or before READ_AT, scale-free criterion
  live_hard same, absolute 1e-6 (the CDDM threshold; reported alongside, as the project requires)
  live_otsu same, absolute threshold from Otsu's method on the POOLED log participation of every
            network in the sweep (common.otsu_threshold), i.e. calibrated on this task
  L_final   mean clean loss over the last 5% of probes before READ_AT
  t90       first probe at which the smoothed clean loss is <= 1.1 * L_final   (pre-registered)
  t_match   first probe at which the smoothed clean loss is <= the mean L_final of the `none` arm at
            the same N; "never" if it is not reached                             (pre-registered)
  t_r2_0.9  first probe at which the smoothed clean loss is <= 0.1 * Var(target over the scored
            steps), i.e. clean r2 >= 0.9 - an ARM-INDEPENDENT escape time from the no-memory
            plateau. Added after seeing the logs, so NOT pre-registered; labelled as such.
  unstable  fraction of the raw probes in the second half of training (iteration > READ_AT/2) at
            which the clean loss is ABOVE the r2 = 0.9 level: the network has transiently lost the
            memory. Added after seeing the curves (every arm shows such episodes); NOT pre-registered.
"Smoothed" = rolling median over 21 probes (210 iterations); the raw probe is noisy at the floor.
r2 in the table is the folder-prefix score of the LAST network (with noise), as everywhere else.

Usage: python dmts_readout.py <trained_RNNs root> [--read-at 150000] [--seeds] [--dump curves.npz]
  --seeds  one row per network instead of per cell
  --dump   also write every network's clean-loss and silent-count curves to an npz (for plotting)
"""
import argparse
import glob
import os
import pickle
import re
from collections import defaultdict

import numpy as np
from omegaconf import OmegaConf

from trainRNNbrain.experiments_and_analysis.common import active_count, otsu_threshold
from trainRNNbrain.tasks.TaskDMTS import TaskDMTS
from trainRNNbrain.training.training_utils import get_training_mask, prepare_task_arguments

SUB = "DMTS_std_pen"
PENS = ["none", "rws", "frm", "both"]
SMOOTH = 21          # probes in the rolling median (210 iterations)
FLOOR_FRAC = 0.05    # last 5% of probes define L_final


def target_variance(config_yaml):
    """Variance of the DMTS target over the scored steps, pooled over outputs and trials.

    Clean r2 = 1 - clean_loss / this, matching training_utils.r2 with axis=None.

    Args:
        config_yaml: path to a saved run config (holds the task block and model.dt).
    Returns:
        float variance.
    """
    cfg = OmegaConf.load(config_yaml)
    task_cfg = prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt)
    args = {k: v for k, v in dict(task_cfg).items() if k != "_target_"}
    task = TaskDMTS(**args)
    _, targets, _ = task.get_batch()
    mask = get_training_mask(cfg_task=cfg.task, dt=cfg.model.dt)
    return float(np.var(targets[:, mask, :]))


def smooth(x, w=SMOOTH):
    """Rolling median of a 1-d array with an odd window `w`, edges padded by repetition.

    Args:
        x: (T,) array; w: window length in samples (clipped to len(x), forced odd).
    Returns:
        (T,) array.
    """
    x = np.asarray(x, dtype=float)
    w = min(w, len(x) if len(x) % 2 else len(x) - 1)
    if w < 3:
        return x.copy()
    pad = w // 2
    xp = np.concatenate([np.repeat(x[0], pad), x, np.repeat(x[-1], pad)])
    return np.median(np.lib.stride_tricks.sliding_window_view(xp, w), axis=-1)


def first_at_or_below(iters, y, thr):
    """First iteration at which `y` is <= `thr`, or None.

    Args:
        iters: (T,) probe iterations; y: (T,) values; thr: float threshold.
    Returns:
        int iteration or None.
    """
    j = np.flatnonzero(y <= thr)
    return int(iters[j[0]]) if j.size else None


def load_nets(root, read_at):
    """Every network of the sweep with its trace cut at READ_AT.

    Args:
        root: trained_RNNs folder; read_at: iteration at which every cell is read.
    Returns:
        list of dicts with keys N, pen, seed, r2, iters, loss, silent, p (last participation
        vector at or before read_at), config (path).
    """
    nets = []
    for f in sorted(glob.glob(os.path.join(root, SUB, "*", "*", "*ParticipationTrace.pkl"))):
        m = re.search(r"_N=(\d+)_pen=([a-z]+)", f)
        with open(f, "rb") as fh:
            tr = pickle.load(fh)
        it = np.asarray(tr["iters"])
        keep = it <= read_at
        if not keep.any() or it[keep][-1] < read_at - 200:
            continue
        pit = np.asarray(tr["participation_iters"])
        j = np.flatnonzero(pit <= read_at)[-1]
        stem = os.path.basename(f).split("_ParticipationTrace")[0]
        nets.append(dict(
            N=int(m.group(1)), pen=m.group(2), seed=stem.split("_s")[-1],
            r2=float(stem.split("_")[0]) if stem.split("_")[0] != "nan" else float("nan"),
            iters=it[keep],
            loss=np.asarray(tr["metrics"]["loss_clean_train"], dtype=float)[keep],
            silent=np.asarray(tr["metrics"]["silent_1em6"], dtype=float)[keep],
            p=np.asarray(tr["participation"][j], dtype=float),
            config=os.path.join(os.path.dirname(f), stem + "_config.yaml")))
    return nets


def main():
    """Parse arguments, compute every per-network quantity, print the table, optionally dump curves."""
    ap = argparse.ArgumentParser()
    ap.add_argument("root")
    ap.add_argument("--read-at", type=int, default=150000)
    ap.add_argument("--seeds", action="store_true")
    ap.add_argument("--dump", default=None)
    a = ap.parse_args()

    nets = load_nets(a.root, a.read_at)
    if not nets:
        raise SystemExit(f"no traces reaching {a.read_at} under {os.path.join(a.root, SUB)}")
    var = target_variance(nets[0]["config"])
    thr_otsu = otsu_threshold(np.concatenate([n["p"] for n in nets]))
    print(f"target variance over scored steps: {var:.4f}  (clean r2 = 1 - L/{var:.4f}; "
          f"r2=0.9 <-> L={0.1 * var:.4f});  Otsu threshold on pooled participation: {thr_otsu:.3g}")

    for n in nets:
        n["Ls"] = smooth(n["loss"])
        k = max(1, int(FLOOR_FRAC * len(n["loss"])))
        n["L_final"] = float(n["loss"][-k:].mean())
        n["t90"] = first_at_or_below(n["iters"], n["Ls"], 1.1 * n["L_final"])
        n["t_r2"] = first_at_or_below(n["iters"], n["Ls"], 0.1 * var)
        late = n["iters"] > a.read_at / 2
        n["unstable"] = float(np.mean(n["loss"][late] > 0.1 * var)) if late.any() else float("nan")
        n["live_sf"] = active_count(n["p"], "scalefree")
        n["live_hard"] = active_count(n["p"], "hard")
        n["live_otsu"] = active_count(n["p"], float(thr_otsu))
    none_floor = defaultdict(list)
    for n in nets:
        if n["pen"] == "none":
            none_floor[n["N"]].append(n["L_final"])
    for n in nets:
        ref = np.mean(none_floor[n["N"]]) if none_floor[n["N"]] else np.nan
        n["t_match"] = first_at_or_below(n["iters"], n["Ls"], ref) if np.isfinite(ref) else None

    fmt_t = lambda t: f"{t:>7d}" if t is not None else "  never"
    order = lambda n: (n["N"], PENS.index(n["pen"]), n["seed"])
    if a.seeds:
        print(f"{'N':>5} {'pen':5} {'seed':>10} {'r2':>7} {'live_sf':>7} {'live_hd':>7} {'live_ot':>7} "
              f"{'L_final':>9} {'t90':>7} {'t_match':>7} {'t_r2>.9':>7} {'unstable':>8}")
        for n in sorted(nets, key=order):
            print(f"{n['N']:>5} {n['pen']:5} {n['seed']:>10} {n['r2']:7.4f} {n['live_sf']:>7d} "
                  f"{n['live_hard']:>7d} {n['live_otsu']:>7d} {n['L_final']:9.2e} {fmt_t(n['t90'])} "
                  f"{fmt_t(n['t_match'])} {fmt_t(n['t_r2'])} {n['unstable']:8.3f}")
    else:
        cells = defaultdict(list)
        for n in nets:
            cells[(n["N"], n["pen"])].append(n)
        print(f"{'N':>5} {'pen':5} {'n':>2} {'live_sf':>11} {'live_hard':>11} {'live_otsu':>11} "
              f"{'L_final':>9} {'r2':>6} {'t90 (seeds)':>22} {'t_match (seeds)':>22} {'t_r2>.9 (seeds)':>22} {'unstable':>8}")
        for key in sorted(cells, key=lambda k: (k[0], PENS.index(k[1]))):
            c = cells[key]
            ms = lambda f: f"{np.mean([x[f] for x in c]):5.0f} ± {np.std([x[f] for x in c]):<4.0f}"
            ts = lambda f: "/".join(str(x[f]) if x[f] is not None else "never" for x in sorted(c, key=lambda x: (x[f] is None, x[f])))
            print(f"{key[0]:>5} {key[1]:5} {len(c):>2} {ms('live_sf'):>11} {ms('live_hard'):>11} "
                  f"{ms('live_otsu'):>11} {np.mean([x['L_final'] for x in c]):9.2e} "
                  f"{np.nanmean([x['r2'] for x in c]):6.3f} {ts('t90'):>22} {ts('t_match'):>22} {ts('t_r2'):>22} "
                  f"{np.mean([x['unstable'] for x in c]):8.3f}")

    if a.dump:
        out = {"target_variance": var, "otsu": thr_otsu}
        for n in nets:
            k = f"{n['N']}_{n['pen']}_{n['seed']}"
            out[k + "_iters"] = n["iters"]
            out[k + "_loss"] = n["loss"]
            out[k + "_silent"] = n["silent"]
        np.savez_compressed(a.dump, **out)
        print(f"curves written to {a.dump}")


if __name__ == "__main__":
    main()
