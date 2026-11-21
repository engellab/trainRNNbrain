#!/usr/bin/env python3
"""Entropy and correlation set points on 3BFF: the 75,000-iteration read, the verdict and the figure.

RULES. docs/project_trajectory/2026-11-20_2307_rules-entropy-correlation-setpoint.md, committed before any run.

MODE `read` (Della CPU node, where the runs are). For every network of
NBitFlipFlop_entropy_setpoint/EqType=h_N=1000_arm={ent0.01,ent0.1,cor0.01,cor0.1}, of the four default
controls (round18_g1g2.CTRL_CELL) and of the rate penalty at 0.1 (round18_g1g2.FRM_CELL at 0.1):
  - active units at the participation probe nearest 75,000 and held-out R^2 of the weights at 75,000,
    36 draws (round18_g1g2.read_run, the read behind data/round18/g1g2.json);
  - active units at every stored participation vector, joined along the warm-start chain, to 75,000;
  - revival: units inactive at some stored vector after iteration 1,000 and active at the read probe;
  - manipulation check: median over units of |H_i - 0.75| and of |c_i| on the 75,000-iteration weights
    (held-out batch, training noise, noise seed fixed);
  - skipped updates, from the run's Slurm log (the `gradient spikes` line; none printed = 0).
Writes OUT_DIR/entropy_setpoint_reads.json (the R^2 cache is kept in OUT_DIR).

MODE `verdict` (Mac). Prints each arm's networks, the criterion (fig_paper_F5_tasks.rule, tolerance 0.01)
against the controls, the rate penalty beside it, the manipulation check and revival; draws the figure
(active units against iterations, every network, with the controls) to --fig (PDF and PNG).

MODE `maze-read` (Stage 2; rules docs/project_trajectory/*_rules-entropy-correlation-setpoint-maze.md). Every
Maze run of the four arms at N = 1,000 and 2,000 (Maze_interventions/<arm>/s<seed> under --arms1000 and --arms2000)
and the default controls (Maze_suite/N=<N>_act=relu, seeds 1-3, under --suite), joined along the warm-start chain
(common.maze_load_runs) and cut at MAZE_END[N] (update 2,000 at N = 1,000, 4,000 at N = 2,000): active units and
probe success at the probe of that update; the learned update and the count there (common.maze_learned_index); the summed skipped minibatch steps and
restored rollouts; the active count at every probe; the arm's raw set-point penalty at update 1 and averaged over
its last 50 updates (the PPO statistics in Progress.json). The arm is derived from the saved config and must match
the folder. Writes the JSON named by the positional argument.

MODE `maze-verdict` (Mac). Per size and arm, the M09 Maze criterion (fig_paper_F5_tasks.rule, success tolerance
0.05, floor 0.9) against the same-size controls; per term and size the one-value rule; the figure of active units
against updates (--fig).

Usage:
  python entropy_setpoint_read.py read OUT_DIR [--logs DIR]     (Della, repository root on PYTHONPATH)
  python entropy_setpoint_read.py verdict READS_JSON --fig PATH_STEM
  python entropy_setpoint_read.py maze-read OUT_JSON --suite DIR --arms1000 DIR --arms2000 DIR
  python entropy_setpoint_read.py maze-verdict OUT_JSON --fig PATH_STEM
"""

import argparse
import glob
import json
import os
import pickle
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from omegaconf import OmegaConf  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H  # noqa: E402
import fig_paper_F5_tasks as F5T  # noqa: E402
from common import active_mask, maze_learned_index, maze_load_runs, warm_start_chain  # noqa: E402
from criticality_read import build_rnn, load_params  # noqa: E402
from read_remedies import SHARED_FIELDS, cfg_of, sel  # noqa: E402
from round18_g1g2 import CTRL_CELL, FIXED_ITER, FRM_CELL, read_run, run_dirs, state_at  # noqa: E402

ARMS = ("ent0.01", "ent0.1", "cor0.01", "cor0.1")
CELL = "NBitFlipFlop_entropy_setpoint/EqType=h_N=1000_arm={arm}"
TARGET_BITS = 0.75          # configs/experiment/entropy_setpoint.yaml, ent_args.target_bits
TARGET_CORR = 0.0           # cor_args.target_corr
STAT_NOISE_SEED = 20261121  # noise draw of the manipulation-check pass
TOL = 0.01
# the Maze read-out point per size, the end of each size's budget: update 2,000 at N = 1,000 (M09's cell) and
# 4,000 at N = 2,000 (the equal budget of the Maze size series; at 2,000 none of its three controls has learned)
MAZE_END = {1000: 2000, 2000: 4000}
MAZE_SIZES = (1000, 2000)
MAZE_TOL, MAZE_FLOOR = 0.05, 0.9   # M09's Maze success tolerance and the level no network may end below
COLORS = {"control": "0.45", "frm0.1": "#9b59b6", "ent0.01": "#7fb3e6", "ent0.1": "#2a78d6",
          "cor0.01": "#f2a07b", "cor0.1": "#eb6834"}


def unit_stats(state_dir):
    """Per-unit on/off entropy and mean pairwise correlation of h on one weight state.

    Args:
        state_dir: folder holding *LastParams* and *_config.yaml of the state.
    Returns:
        dict med_dev_H (median |H_i - TARGET_BITS|, bits), med_abs_c (median |c_i - TARGET_CORR|),
        med_H, med_c (medians), all floats.
    """
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(state_dir, "*_config.yaml")))[0])
    rnn = build_rnn(cfg, seed=int(cfg.seed))
    params, _ = H._params(state_dir)
    load_params(rnn, params)
    bi = H.heldout_batch(cfg)[0]
    g = rnn.random_generator.get_state()
    rnn.random_generator.manual_seed(STAT_NOISE_SEED)
    with torch.no_grad():
        h, _ = rnn(bi, w_noise=True)
    rnn.random_generator.set_state(g)
    x = h.reshape(h.shape[0], -1).double()
    z = x.mean(1) / x.std(1)
    lp, lq = torch.special.log_ndtr(z), torch.special.log_ndtr(-z)
    Hb = (-(lp.exp() * lp + lq.exp() * lq) / np.log(2.0)).numpy()
    xc = x - x.mean(1, keepdim=True)
    zz = xc / xc.pow(2).mean(1, keepdim=True).sqrt()
    c = (((zz * zz.sum(0)).mean(1) - zz.pow(2).mean(1)) / (zz.shape[0] - 1)).numpy()
    return dict(med_dev_H=float(np.median(np.abs(Hb - TARGET_BITS))), med_abs_c=float(np.median(np.abs(c - TARGET_CORR))),
                med_H=float(np.median(Hb)), med_c=float(np.median(c)))


def chain_trace(run_dir, until=FIXED_ITER):
    """Active units at every stored participation vector, joined along the warm-start chain.

    Args:
        run_dir: a run folder; until: last iteration kept.
    Returns:
        (iterations list, active-count list, list of (N,) bool active masks), sorted by iteration.
    """
    its, acts, masks = [], [], []
    for d in warm_start_chain(run_dir):
        f = glob.glob(os.path.join(d, "*ParticipationTrace.pkl"))
        if not f:
            continue
        tr = pickle.load(open(f[0], "rb"))
        for it, p in zip(tr["participation_iters"], np.asarray(tr["participation"])):
            if it <= until and it not in its:
                m = active_mask(p)
                its.append(float(it)); acts.append(int(m.sum())); masks.append(m)
    o = np.argsort(its)
    return [its[i] for i in o], [acts[i] for i in o], [masks[i] for i in o]


def revived(its, masks, probe):
    """Units inactive at some stored vector after iteration 1,000 and active at the read probe.

    Args:
        its: iterations; masks: (N,) bool active masks at those iterations; probe: the read iteration.
    Returns:
        int count.
    """
    j = int(np.argmin(np.abs(np.asarray(its) - probe)))
    late = [m for i, m in zip(its, masks) if 1000 < i <= its[j]]
    if not late:
        return 0
    ever_off = ~np.logical_and.reduce(late)
    return int((ever_off & masks[j]).sum())


def skipped(run_dir, logs):
    """Skipped updates of one run from its Slurm log, or None when no log names its seed.

    Args:
        run_dir: run folder (its name holds seed=<seed>); logs: folder of *.out files.
    Returns:
        int (0 when the log prints no `gradient spikes` line) or None.
    """
    seed = re.search(r"seed=(\d+)", os.path.basename(run_dir)).group(1)
    for f in glob.glob(os.path.join(logs, "EntSP.*.out")):
        txt = open(f, errors="replace").read()
        if f"'seed': {seed}," in txt:
            m = re.search(r"gradient spikes: (\d+) updates skipped", txt)
            return int(m.group(1)) if m else 0
    return None


def do_read(out_dir, logs):
    """Mode `read`: every network of the four arms, the controls and the rate penalty at 0.1.

    Args:
        out_dir: str, output folder (created); logs: folder of the Slurm logs.
    Returns:
        None.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "r2cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    c0 = cfg_of(run_dirs(CTRL_CELL)[0])
    ctrl_shared = {k: sel(c0, k) for k in SHARED_FIELDS}
    out = {}
    groups = [("control", CTRL_CELL), ("frm0.1", FRM_CELL.format(lam="0.1"))] + [(a, CELL.format(arm=a)) for a in ARMS]
    for name, cell in groups:
        out[name] = []
        for d in run_dirs(cell):
            rec = read_run(d, cache, ctrl_shared)
            its, acts, masks = chain_trace(d)
            rec.update(trace_it=its, trace_active=acts, revived=revived(its, masks, rec["probe_75k"]) if its else None,
                       stats=unit_stats(rec["state"]) if rec["state"] else None,
                       skipped=skipped(d, logs) if name in ARMS else None, cell=cell)
            out[name].append(rec)
            print(f"{name} {os.path.basename(d)[:50]} active {rec['active_75k']} R2 {rec['r2_75k']} "
                  f"revived {rec['revived']} stats {rec['stats']} skipped {rec['skipped']} "
                  f"config differs {rec['config_differs']}", flush=True)
            json.dump(cache, open(cache_path, "w"))
            json.dump(out, open(os.path.join(out_dir, "entropy_setpoint_reads.json"), "w"), indent=1)


def do_verdict(reads_json, fig_stem):
    """Mode `verdict`: print the criterion per arm and term, the descriptions, and draw the figure.

    Args:
        reads_json: str, the file `read` wrote; fig_stem: str, figure path without extension.
    Returns:
        dict arm -> fig_paper_F5_tasks.rule result.
    """
    R = json.load(open(reads_json))
    ca = [r["active_75k"] for r in R["control"]]
    cr = [r["r2_75k"] for r in R["control"] if r["r2_75k"] is not None]
    print(f"controls: active {ca} (largest {max(ca)}), R2 {np.round(cr, 4).tolist()} mean {np.mean(cr):.4f}; "
          f"R2 bar {np.mean(cr) - TOL:.4f}")
    verdicts = {}
    for name in ("frm0.1",) + ARMS:
        recs = R.get(name, [])
        act = [r["active_75k"] for r in recs]
        r2 = [r["r2_75k"] for r in recs if r["r2_75k"] is not None]
        v = F5T.rule(act, r2, ca, cr, TOL) if recs else dict(works=False, text="no finished network", fold=float("nan"),
                                                               dperf=float("nan"))
        verdicts[name] = v
        print(f"\n{name}{' (description only)' if name == 'frm0.1' else ''}: {v['text']}; fold {v['fold']:.2f}, "
              f"dR2 {v['dperf']:+.4f}")
        for r in recs:
            st = r["stats"] or {}
            print(f"   {os.path.basename(r['run_dir'])[:40]}  active {r['active_75k']} (probe {r['probe_75k']:.0f})  "
                  f"R2 {r['r2_75k']}  revived {r['revived']}  skipped {r['skipped']}  "
                  f"|H-0.75| {st.get('med_dev_H', float('nan')):.3f}  |c| {st.get('med_abs_c', float('nan')):.3f}  "
                  f"config differs {r['config_differs']}")
    cs = [r["stats"] for r in R["control"] if r["stats"]]
    print(f"\ncontrols, manipulation stats: |H-0.75| {[round(s['med_dev_H'], 3) for s in cs]}  "
          f"|c| {[round(s['med_abs_c'], 3) for s in cs]}  revived {[r['revived'] for r in R['control']]}")
    for term, arms in (("entropy (A)", ARMS[:2]), ("correlation (B)", ARMS[2:])):
        print(f"{term}: {'PASSES' if any(verdicts[a]['works'] for a in arms) else 'fails'} on 3BFF "
              f"({', '.join(f'{a}: ' + ('passes' if verdicts[a]['works'] else 'fails') for a in arms)})")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True, constrained_layout=True)
    for ax, arms, title in ((axes[0], ARMS[:2], "(A) entropy set point"), (axes[1], ARMS[2:], "(B) correlation set point")):
        for name in ("control", "frm0.1") + arms:
            for k, r in enumerate(R.get(name, [])):
                lab = {"control": "default (controls)", "frm0.1": "rate penalty, λ = 0.1"}.get(name, f"λ = {name[3:]}")
                ax.plot(np.asarray(r["trace_it"]) / 1000, r["trace_active"], color=COLORS[name],
                        lw=1.2 if name in ("control", "frm0.1") else 1.6, alpha=0.9, label=lab if k == 0 else None)
        ax.axhline(max(ca), color="0.45", lw=0.8, ls=":", label=f"largest control count ({max(ca)})")
        ax.set_xlim(0, 75)
        ax.set_ylim(0, 1000)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("iteration (thousands)")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False, loc="lower right", ncol=2)
    axes[0].set_ylabel("active units (of 1,000)")
    fig.savefig(fig_stem + ".pdf")
    fig.savefig(fig_stem + ".png", dpi=150)
    print(f"figure: {fig_stem}.pdf, .png")
    return verdicts


def maze_arm_of(folder):
    """The set-point arm a Maze run's saved config switches on: "ent<lambda>", "cor<lambda>" or "control".

    Args:
        folder: str, a run folder holding *config.yaml.
    Returns:
        str.
    Raises:
        ValueError if both terms are on.
    """
    tr = OmegaConf.load(sorted(glob.glob(os.path.join(folder, "*config.yaml")))[0]).trainer
    ent, cor = float(tr.get("lambda_ent", 0) or 0), float(tr.get("lambda_cor", 0) or 0)
    if ent > 0 and cor > 0:
        raise ValueError(f"{folder}: both set-point terms are on")
    return f"ent{ent:g}" if ent > 0 else f"cor{cor:g}" if cor > 0 else "control"


def maze_point(r, end):
    """The read-out of one joined Maze run, cut at update `end`.

    Args:
        r: dict from common.maze_load_runs; end: int, the read-out update (MAZE_END[N]).
    Returns:
        dict: N, seed, folder, chain, last_update, complete (reached `end` with a probe there), active_end and
        success_end (at the probe of update `end`, None if incomplete), learned_update and active_learned (None
        if the run did not learn), skipped_sum, restored_sum, trace_u and trace_active (every probe), penalty
        (dict term -> [value at the first update, mean over the last 50 updates], for the set-point keys present).
    """
    P = r["probes"]
    u = np.asarray(P["update"])
    m = u <= end
    u, act, succ = u[m], np.asarray(P["active"])[m], np.asarray(P["probe_success"])[m]
    prog = [p for p in r["progress"] if p["update"] <= end]
    last = int(prog[-1]["update"]) if prog else 0
    complete = bool(last >= end and len(u) and int(u[-1]) == end)
    li = maze_learned_index(succ)
    pen = {k: [float(prog[0][k]), float(np.mean([p[k] for p in prog[-50:]]))]
           for k in ("entropy_setpoint", "correlation_setpoint") if prog and k in prog[0]}
    return dict(N=int(r["N"]), seed=int(r["seed"]), folder=r["folder"], chain=list(r["chain"]), last_update=last,
                complete=complete, active_end=int(act[-1]) if complete else None,
                success_end=float(succ[-1]) if complete else None,
                learned_update=None if li is None else int(u[li]), active_learned=None if li is None else int(act[li]),
                skipped_sum=float(sum(p.get("skipped", 0) for p in prog)),
                restored_sum=float(sum(p.get("restored", 0) for p in prog)),
                trace_u=u.tolist(), trace_active=act.tolist(), penalty=pen)


def do_maze_read(out_json, suite, arm_roots):
    """Mode `maze-read`: the controls and the four arms at both sizes, written to out_json.

    Args:
        out_json: str; suite: str, the Maze_suite folder; arm_roots: dict N -> folder holding <arm>/s<seed>.
    Returns:
        dict N -> {"control": [...], arm: [...]} of maze_point dicts.
    """
    out = {}
    for N in MAZE_SIZES:
        cell = f"N={N}_act=relu"
        ctrl = [maze_point(r, MAZE_END[N]) for r in maze_load_runs(suite)
                if os.path.basename(os.path.dirname(r["chain"][0])) == cell and int(r["seed"]) in (1, 2, 3)]
        out[N] = {"control": ctrl}
        for r in maze_load_runs(arm_roots[N]):
            arm = maze_arm_of(r["folder"])
            folder_arm = os.path.basename(os.path.dirname(r["chain"][0]))
            if arm != folder_arm or int(r["N"]) != N:
                raise ValueError(f"{r['folder']}: config arm {arm} N {r['N']}, folder arm {folder_arm} root N {N}")
            out[N].setdefault(arm, []).append(maze_point(r, MAZE_END[N]))
        for name, rows in out[N].items():
            for x in rows:
                print(f"N={N} {name:8s} s{x['seed']} last {x['last_update']} active {x['active_end']} "
                      f"success {x['success_end']} learned {x['learned_update']} skipped {x['skipped_sum']:g} "
                      f"restored {x['restored_sum']:g} penalty {x['penalty']}", flush=True)
    json.dump(out, open(out_json, "w"), indent=1)
    return out


def do_maze_verdict(out_json, fig_stem):
    """Mode `maze-verdict`: the M09 Maze criterion per size and arm, the one-value rule per term, the figure.

    Args:
        out_json: str, the file `maze-read` wrote; fig_stem: str or None, figure path without extension.
    Returns:
        dict (N, arm) -> fig_paper_F5_tasks.rule result, or a dict with works False and text "incomplete".
    """
    R = {int(k): v for k, v in json.load(open(out_json)).items()}
    verdicts = {}
    for N in MAZE_SIZES:
        ctrl = [x for x in R[N]["control"] if x["complete"]]
        ca, cs = [x["active_end"] for x in ctrl], [x["success_end"] for x in ctrl]
        print(f"\nN = {N}, update {MAZE_END[N]:,}: controls active {ca} (largest {max(ca)}), success {np.round(cs, 3).tolist()} "
              f"(mean {np.mean(cs):.3f}; bar {np.mean(cs) - MAZE_TOL:.3f}, floor {MAZE_FLOOR})")
        for arm in ARMS:
            rows = R[N].get(arm, [])
            fin = [x for x in rows if x["complete"]]
            if len(fin) < 3:
                v = dict(works=False, text=f"incomplete ({len(fin)} of 3 finished)", fold=float("nan"),
                         dperf=float("nan"))
            else:
                v = F5T.rule([x["active_end"] for x in fin], [x["success_end"] for x in fin], ca, cs, MAZE_TOL,
                             floor=MAZE_FLOOR)
            verdicts[(N, arm)] = v
            print(f"  {arm}: {v['text']}; fold {v['fold']:.2f}, dsuccess {v['dperf']:+.3f}")
            for x in rows:
                print(f"     s{x['seed']} last {x['last_update']} active {x['active_end']} success {x['success_end']} "
                      f"learned {x['learned_update']} (active {x['active_learned']}) skipped {x['skipped_sum']:g} "
                      f"restored {x['restored_sum']:g} penalty first/last50 {x['penalty']}")
    for term, arms in (("entropy (A)", ARMS[:2]), ("correlation (B)", ARMS[2:])):
        per_n = {N: any(verdicts[(N, a)]["works"] for a in arms) for N in MAZE_SIZES}
        print(f"{term}: {'PASSES' if all(per_n.values()) else 'fails'} on Maze "
              f"({', '.join(f'N = {N}: ' + ('passes' if ok else 'fails') for N, ok in per_n.items())})")
    if fig_stem:
        fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
        for i, N in enumerate(MAZE_SIZES):
            ca = [x["active_end"] for x in R[N]["control"] if x["complete"]]
            for j, (arms, title) in enumerate(((ARMS[:2], "(A) entropy set point"), (ARMS[2:], "(B) correlation set point"))):
                ax = axes[i, j]
                for name in ("control",) + arms:
                    for k, x in enumerate(R[N].get(name, [])):
                        lab = "default (controls)" if name == "control" else f"λ = {name[3:]}"
                        ax.plot(x["trace_u"], x["trace_active"], color=COLORS[name], lw=1.2 if name == "control" else 1.6,
                                alpha=0.9, label=lab if k == 0 else None)
                ax.axhline(max(ca), color="0.45", lw=0.8, ls=":", label=f"largest control count ({max(ca)})")
                ax.set_xlim(0, MAZE_END[N])
                ax.set_ylim(0, N)
                ax.set_title(f"{title}, N = {N:,}", fontsize=10)
                ax.grid(alpha=0.25, lw=0.5)
                ax.legend(fontsize=8, frameon=False, loc="upper right")
                ax.set_xlabel("PPO update")
                if j == 0:
                    ax.set_ylabel(f"active units (of {N:,})")
        fig.savefig(fig_stem + ".pdf")
        fig.savefig(fig_stem + ".png", dpi=150)
        print(f"figure: {fig_stem}.pdf, .png")
    return verdicts


def main():
    """Parse the command line and run the selected mode."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=("read", "verdict", "maze-read", "maze-verdict"))
    ap.add_argument("path")
    ap.add_argument("--logs", default=os.path.expanduser("~/trainRNNbrain_logs"))
    ap.add_argument("--fig", default=None)
    ap.add_argument("--suite", default=None, help="maze-read: the Maze_suite folder (the controls)")
    ap.add_argument("--arms1000", default=None, help="maze-read: Maze_interventions folder of the N = 1,000 arms")
    ap.add_argument("--arms2000", default=None, help="maze-read: Maze_interventions folder of the N = 2,000 arms")
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.path, a.logs)
    elif a.mode == "verdict":
        do_verdict(a.path, a.fig)
    elif a.mode == "maze-read":
        do_maze_read(a.path, a.suite, {1000: a.arms1000, 2000: a.arms2000})
    else:
        do_maze_verdict(a.path, a.fig)


if __name__ == "__main__":
    main()
