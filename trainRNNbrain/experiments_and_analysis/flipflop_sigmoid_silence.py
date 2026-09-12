#!/usr/bin/env python3
"""
Does a bounded, nonlinear positive activation escape the silent-unit phenomenon?

Compares unpenalised sigmoid(7.5 (x - 0.3)) flip-flop networks (NBitFlipFlop_std_sigmoid) with the
unpenalised ReLU networks of the same size and task (ksweep, k=3), on:

  silence, three ways
    scalefree    p < 0.05 q95(p), p = std + q90 of the rate (the project's standard rule)
    unmodulated  std_t(r) < 0.05 q95(std_t(r)) — REQUIRED for a bounded activation: a unit saturated
                 at a constant output has q90 > 0 and passes the standard rule while doing nothing
    abs 4e-2     the ReLU flip-flop threshold, for reference only (meaningless for sigmoid)
  threshold-free  participation Hoyer sparsity and 1/HHI (effective participating units)
  the four axes   characterize.measure (dimensionality, participation, temporal, selectivity),
                  live = modulated units for every network so the axes are comparable
  trajectory      from ParticipationTrace.pkl: scale-free silent fraction and participation Hoyer vs
                  iteration, per seed, sigmoid against ReLU (⚠️ the trace stores std+q90 only, so the
                  modulation criterion is available at the end, not along training)

⚠️ THE ACTIVATION IS READ FROM THE SAVED PARAMETERS. flipflop_fixedpoints.load_net forces relu and
must not be used here. For the h equation the rate is activation(state); the raw history is the
state (rates_and_targets applies np.maximum, which is only right for ReLU).

Decision rule (pre-registered in project_trajectory.md, 2026-09-10 23:08): participation Hoyer
< 0.3 and unmodulated < 10% at both N with task R2 >= 0.8 → concentration is a ReLU-family property;
Hoyer >= 0.5 or unmodulated >= 30% → general to positive activations; in between → graded.

Usage:  python flipflop_sigmoid_silence.py [SIGMOID_ROOT] [--no-baseline]
Output: img/internal_figures/sigmoid_silence.png
"""

import os
import re
import sys
import glob
import pickle
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import SILENT_FLIPFLOP, SILENT_REL, participation
import plotstyle as ps
from characterize import measure, hoyer
from flipflop_dimensionality import run_folders
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.utils import filter_kwargs
from trainRNNbrain.training.training_utils import prepare_task_arguments

SIGMOID_ROOT = "data/trained_RNNs/NBitFlipFlop_std_sigmoid"
N_TRIALS = 32
COL = {"relu": "#7f7f7f", "sigmoid": "#9467bd"}


def load_any(folder):
    """RNN_numpy from a saved run, with the activation and equation type the run actually used.

    Args:
        folder: run folder holding `*LastParams*.npz` and `*_config.yaml`.
    Returns:
        (rnn, cfg).
    """
    d = np.load(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0], allow_pickle=True)
    p = {kk: d[kk] for kk in d.files}
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    # ⚠️ the npz stores a dict's KEYS only for older runs (array(['name', 'slope'])); the run config
    # is the source of truth for the activation and is used for every network.
    p["activation_args"] = OmegaConf.to_container(cfg.model.activation_args, resolve=True)
    rnn = RNN_numpy(**filter_kwargs(RNN_numpy, p), equation_type=str(cfg.model.equation_type), seed=0)
    return rnn, cfg


def rates_targets(folder):
    """Noise-free rates and targets on a fresh batch; rate = activation(state) for the h equation.

    Args:
        folder: run folder.
    Returns:
        (rates (N, T, B), targets (k, T, B), task R2 on the masked output, activation name).
    """
    rnn, cfg = load_any(folder)
    cfg.task.batch_size = N_TRIALS
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, targets, _ = task.get_batch()
    rnn.clear_history(); rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    y = np.array(rnn.get_history())
    r = rnn.activation(y) if rnn.equation_type == "h" else y
    out = np.einsum("on,ntb->otb", rnn.W_out, r)
    r2 = 1.0 - ((out - targets) ** 2).sum() / ((targets - targets.mean()) ** 2).sum()
    return r, targets, float(r2), rnn.activation_args["name"]


def analyse(folder):
    """All end-of-training statistics for one network.

    Args:
        folder: run folder.
    Returns:
        dict with silence fractions, Hoyer axes, R2, and the participation trajectory arrays.
    """
    r, t, r2, act = rates_targets(folder)
    N = r.shape[0]
    X = r.reshape(N, -1)
    p = participation(r)
    tstd = X.std(1)
    unmod = tstd < SILENT_REL * np.quantile(tstd, 0.95)
    B = t.reshape(t.shape[0], -1).T
    G = np.column_stack([np.ones(B.shape[0]), np.maximum(B, 0), np.maximum(-B, 0)])
    m = measure(np.maximum(r, 0), ~unmod, X, G)
    m.update(act=act, r2=r2, N=N,
             scalefree=float((p < SILENT_REL * np.quantile(p, 0.95)).mean()),
             unmodulated=float(unmod.mean()), abs4e2=float((p < SILENT_FLIPFLOP).mean()),
             inv_hhi=float(1.0 / ((p / p.sum()) ** 2).sum()), part_hoyer=float(hoyer(p)))
    tr = glob.glob(os.path.join(folder, "*ParticipationTrace.pkl"))
    if tr:
        d = pickle.load(open(tr[0], "rb"))
        P = np.asarray(d["participation"], dtype=float)                      # (n_iters, N)
        q = np.quantile(P, 0.95, axis=1, keepdims=True)
        m["traj_iters"] = np.asarray(d["participation_iters"])
        m["traj_silent"] = (P < SILENT_REL * q).mean(1)
        m["traj_hoyer"] = hoyer(P)
        # Silence can arrive as an avalanche during a loss spike (2026-09-11: 141 -> 559 units in
        # 500 iterations). Record the largest rise of the silent fraction over any ~1000-iteration
        # window and where it happened, and the silent fraction at 150k for matched-iteration
        # comparison with runs of a different budget.
        step = float(np.median(np.diff(m["traj_iters"]))) if len(m["traj_iters"]) > 1 else 1.0
        lag = max(1, int(round(1000 / step)))
        ts = m["traj_silent"]
        if len(ts) > lag:
            dd = ts[lag:] - ts[:-lag]; j = int(np.argmax(dd))
            m["max_jump_1k"] = float(dd[j]); m["jump_iter"] = int(m["traj_iters"][j + lag])
        else:
            m["max_jump_1k"] = np.nan; m["jump_iter"] = -1
        m["sf_at_150k"] = float(np.interp(150000, m["traj_iters"], ts)) if m["traj_iters"][-1] >= 149000 else np.nan
    return m


def main():
    """Tabulate sigmoid vs ReLU silence and the four axes; plot end-of-training and trajectories."""
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    root = args[0] if args else SIGMOID_ROOT
    ps.setup()
    runs = []
    for f in sorted(glob.glob(os.path.join(root, "*", "*", "*LastParams*.npz"))):
        mm = re.search(r"_N=(\d+)", f)
        runs.append(("sigmoid", int(mm.group(1)), os.path.dirname(f)))
    if "--no-baseline" not in sys.argv:
        Ns = {N for _, N, _ in runs} or {500, 1000}
        runs += [("relu", N, f) for f, pen, k, N in run_folders() if k == 3 and pen == "none" and N in Ns]
    rows = []
    for act, N, f in runs:
        m = analyse(f); m["N"] = N; m["act"] = act; rows.append(m)
        print(f"  {act:<8} N={N:<5} r2={m['r2']:.3f}  scalefree={m['scalefree']:.3f}  unmodulated={m['unmodulated']:.3f}  "
              f"abs4e-2={m['abs4e2']:.3f}  partHoyer={m['part_hoyer']:.3f}  1/HHI={m['inv_hhi']:.0f}  "
              f"sel={m['sel']:.3f} temp={m['temp']:.3f} D_PR={m['d_pr']:.1f}", flush=True)

    print(f"\n{'act':<8}{'N':>6}{'n':>3}{'r2':>7}{'scalefree':>11}{'unmodulated':>13}{'part Hoyer':>12}{'1/HHI':>7}{'sel':>7}{'temp':>7}{'D_PR':>6}")
    for act in ("relu", "sigmoid"):
        for N in sorted({r["N"] for r in rows if r["act"] == act}):
            rr = [r for r in rows if r["act"] == act and r["N"] == N]
            f = lambda k, fmt: fmt.format(np.mean([r[k] for r in rr]), np.std([r[k] for r in rr]))
            print(f"{act:<8}{N:>6}{len(rr):>3}{f('r2', '{:.2f}'):>7}{f('scalefree', '{:.2f}±{:.2f}'):>11}"
                  f"{f('unmodulated', '{:.2f}±{:.2f}'):>13}{f('part_hoyer', '{:.2f}±{:.2f}'):>12}{f('inv_hhi', '{:.0f}'):>7}"
                  f"{f('sel', '{:.2f}'):>7}{f('temp', '{:.2f}'):>7}{f('d_pr', '{:.1f}'):>6}")

    fig, ax = plt.subplots(1, 3, figsize=(16, 4.6))
    for r in rows:
        if "traj_iters" not in r:
            continue
        ls = "-" if r["N"] == max(x["N"] for x in rows) else "--"
        lab = f"{r['act']} N={r['N']}"
        ax[0].plot(r["traj_iters"], r["traj_silent"], ls, color=COL[r["act"]], alpha=.8, label=lab)
        ax[1].plot(r["traj_iters"], r["traj_hoyer"], ls, color=COL[r["act"]], alpha=.8, label=lab)
    ax[0].set(xlabel="iteration", ylabel="silent fraction (p < 0.05 q95)", title="silent fraction along training", xscale="log")
    ax[1].set(xlabel="iteration", ylabel="participation Hoyer sparsity", title="concentration along training", xscale="log", ylim=(0, 1))
    for a in ax[:2]:
        h, l = a.get_legend_handles_labels(); u = dict(zip(l, h)); a.legend(u.values(), u.keys(), fontsize=7); a.grid(alpha=.25)
    for key, mk in (("unmodulated", "o"), ("scalefree", "s"), ("part_hoyer", "^")):
        for act in ("relu", "sigmoid"):
            xs = [r["N"] * (1.03 if act == "sigmoid" else 0.97) for r in rows if r["act"] == act]
            ax[2].scatter(xs, [r[key] for r in rows if r["act"] == act], marker=mk, color=COL[act], label=f"{act}: {key}", alpha=.8)
    ax[2].set(xscale="log", xlabel="N", ylabel="fraction / sparsity", title="end of training", ylim=(0, 1)); ax[2].grid(alpha=.25)
    h, l = ax[2].get_legend_handles_labels(); u = dict(zip(l, h)); ax[2].legend(u.values(), u.keys(), fontsize=7)
    fig.suptitle("Bounded nonlinear activation vs ReLU, unpenalised 3-bit flip-flop", fontsize=12.5)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    return ps.save(fig, "sigmoid_silence", tight=False)


if __name__ == "__main__":
    main()
