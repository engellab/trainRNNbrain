#!/usr/bin/env python3
"""Do the shared initial draws change the number of active units? The read and the verdict.

RULES. docs/project_trajectory/2026-11-25_1303_rules-independent-init.md, committed before any run. The cells, the
reference ranges, the margin and the verdict rule below are copied from that entry.

MODE `read OUT_JSON` (Della CPU node). Every new 3BFF and CDDM network (independent draws) and every shared-draw
reference network: active units at 75,000 iterations and held-out R^2 there (fig_paper_F2_fails.read_fixed; R^2 for the
new networks and the paper's size-series controls only), the floor crossing and the active units there
(common.read_run), the active count at every stored vector of the new networks, their skipped updates (Slurm log),
their draw mode and the config differences from one of the paper's networks.

MODE `maze-read OUT_JSON --new DIR --suite DIR` (Mac). The new Maze networks (<DIR>/<cell>/s<seed>, shared draws) and
the six suite networks (seeds 1-6, N = 1,000, relu): active units and probe success at update 2,000, the learned
point and the active units there (common.maze_learned_index, on the record up to update 2,000 for the new
networks), restored rollouts and skipped minibatches, the active count at every probe.

MODE `verdict READS_JSON MAZE_JSON --fig STEM` (Mac). Prints the rules' tables and verdicts and draws the figure. It
refuses an incomplete set (three networks per task).

Usage: PYTHONPATH=<repo> CUDA_VISIBLE_DEVICES= python independent_init_read.py read OUT.json --logs DIR
"""

import argparse
import glob
import json
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DATA_DIR, maze_learned_index, maze_load_runs, read_run, runs_with_loss  # noqa: E402
from entropy_setpoint_read import chain_trace, flat_cfg, skipped  # noqa: E402
from fig_paper_F2_fails import FIXED_IT, read_fixed  # noqa: E402

N_NEW = 3
NEW = {"3BFF": "NBitFlipFlop_independent_init{suffix}/EqType=h_N=1000_draws=independent",
       "CDDM": "CDDM_independent_init/EqType=h_N=1000_draws=independent"}
# the paper's size-series networks at N = 1,000 (held-out R^2 is read for these and the new ones only)
SERIES = {"3BFF": "NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none",
          "CDDM": "CDDM_paper_grid/EqType=h_N=1000_arm=control"}
# every shared-draw default network of N = 1,000 (rules entry: 13 of 3BFF, 29 of CDDM)
REFERENCE = {"3BFF": [SERIES["3BFF"], "NBitFlipFlop_std_dropfix_c200000/EqType=h_k=3_N=1000_pen=none_do=none",
                      "NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000",
                      "NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control"],
             "CDDM": [SERIES["CDDM"], "CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000",
                      "CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05_iters=200000",
                      "CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05",
                      "CDDM_std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000",
                      "CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
                      "CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=none",
                      "CDDM_c1_snapshots/EqType=h_N=1000_arm=control"]}
# (min, max, mean, margin) of the rules entry; the reader must reproduce the recorded ranges
BOUNDS = {("3BFF", "75k"): (236, 309, 274, 14), ("3BFF", "floor"): (240, 344, 285, 14),
          ("CDDM", "75k"): (311, 354, 333, 17), ("CDDM", "floor"): (390, 468, 423, 21),
          ("Maze", "u2000"): (394, 449, 417, 21), ("Maze", "learned"): (406, 517, 440, 22)}
MAZE_END = 2000
SUITE_LEARNED_END = 4000
R2_TOL, SUCCESS_FLOOR = 0.01, 0.9


def read_network(r, with_r2, cache):
    """One supervised network at both read points.

    Args:
        r: a common.runs_with_loss dict; with_r2: also the held-out R^2 at 75,000; cache: heldout_r2 cache dict.
    Returns:
        dict seed, draws, active_75k, probe_75k, r2_75k, r2_how, t_star, active_floor, probe_floor, floor_reason,
        it_end (last training iteration), run_dir.
    """
    c = flat_cfg(r["run_dir"])
    it = np.asarray(r["it_act"], float)
    k = int(np.argmin(np.abs(it - FIXED_IT)))
    out = dict(run_dir=r["run_dir"], seed=str(c["seed"]), draws=str(c.get("model.init_draws", "absent")),
               active_75k=int(r["active"][k]), probe_75k=float(it[k]), it_end=float(np.asarray(r["it_loss"])[-1]))
    if with_r2:
        f = read_fixed(r, FIXED_IT, cache)
        out.update(r2_75k=float(f["r2"]), r2_how=f["r2_how"])
    rr = read_run(r["run_dir"], run=r)
    out.update(t_star=float(rr["t_star"]), active_floor=rr["active"], probe_floor=rr["probe_iter"],
               floor_reason=rr["reason"])
    return out


def do_read(out_json, logs):
    """Mode `read` (module docstring).

    Args:
        out_json: str, output file; logs: folder of the Slurm logs (IndInit.*.out).
    Returns:
        None.
    """
    cache_path = os.path.splitext(out_json)[0] + "_r2cache.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = {"new": {}, "reference": {}}
    for task in ("3BFF", "CDDM"):
        cells = [NEW[task].format(suffix=s) for s in ("_c200000", "")]
        cell = next((c for c in cells if os.path.isdir(os.path.join(DATA_DIR, c))), cells[-1])
        runs = runs_with_loss(os.path.join(DATA_DIR, cell))
        ref_cfg = flat_cfg(runs_with_loss(os.path.join(DATA_DIR, SERIES[task]))[0]["run_dir"])
        out["new"][task] = []
        for r in runs:
            rec = read_network(r, True, cache)
            its, acts, _ = chain_trace(r["run_dir"], until=1e12)
            c = flat_cfg(r["run_dir"])
            rec.update(trace_it=its, trace_active=acts, skipped=skipped(r["run_dir"], logs, "IndInit"),
                       differs={k: [str(c.get(k, "<absent>")), str(ref_cfg.get(k, "<absent>"))]
                                for k in sorted(set(c) | set(ref_cfg)) if c.get(k, "<absent>") != ref_cfg.get(k, "<absent>")})
            out["new"][task].append(rec)
            print(task, "new", {k: v for k, v in rec.items() if not k.startswith("trace")}, flush=True)
            json.dump(cache, open(cache_path, "w"))
        out["reference"][task] = []
        for cell in REFERENCE[task]:
            for r in runs_with_loss(os.path.join(DATA_DIR, cell)):
                rec = read_network(r, cell == SERIES[task], cache)
                if rec["draws"] == "absent":
                    out["reference"][task].append(dict(rec, cell=cell))
                print(task, "reference", cell, rec["seed"], rec["draws"], rec["active_75k"], rec["active_floor"], flush=True)
            json.dump(cache, open(cache_path, "w"))
    json.dump(out, open(out_json, "w"), indent=1)


def maze_record(r):
    """One Maze network at update 2,000 and at its learned point, on its record up to update 2,000.

    Args:
        r: a common.maze_load_runs dict.
    Returns:
        dict seed, updates (last recorded), active_u2000, success_u2000, learned_update and active_learned (None if
        not learned), restored, skipped (sums over the updates up to 2,000), trace_u, trace_active.
    """
    u = np.asarray(r["probes"]["update"])
    keep = u <= MAZE_END
    u, act, ps = u[keep], np.asarray(r["probes"]["active"])[keep], np.asarray(r["probes"]["probe_success"])[keep]
    k = int(np.argmin(np.abs(u - MAZE_END)))
    i = maze_learned_index(ps)
    prog = [p for p in r["progress"] if p["update"] <= MAZE_END]
    return dict(folder=r["folder"], seed=str(r["seed"]), updates=int(max(p["update"] for p in prog)),
                active_u2000=int(act[k]), success_u2000=float(ps[k]),
                learned_update=None if i is None else int(u[i]), active_learned=None if i is None else int(act[i]),
                restored=float(sum(p.get("restored", 0) for p in prog)),
                skipped=float(sum(p.get("skipped", 0) for p in prog)),
                trace_u=u.tolist(), trace_active=act.tolist())


def do_maze_read(out_json, new_dir, suite_dir):
    """Mode `maze-read` (module docstring).

    Args:
        out_json: str, output file; new_dir: the Maze_independent_init folder; suite_dir: the Maze_suite folder.
    Returns:
        None.
    """
    new = [maze_record(r) for r in maze_load_runs(new_dir) if r["N"] == 1000]
    # the suite holds several continuations per seed; their records up to update 4,000 are the same run's
    suite = {}
    for r in maze_load_runs(suite_dir):
        if r["N"] == 1000 and r["act"] == "relu":
            rec = maze_record(r)
            u = np.asarray(r["probes"]["update"])
            keep = u <= SUITE_LEARNED_END     # the paper's table reads the learned point on the record to 4,000
            i = maze_learned_index(np.asarray(r["probes"]["probe_success"])[keep])
            act = np.asarray(r["probes"]["active"])[keep]
            rec.update(learned_update=None if i is None else int(u[keep][i]),
                       active_learned=None if i is None else int(act[i]))
            if suite.setdefault(r["seed"], rec)["active_u2000"] != rec["active_u2000"]:
                raise ValueError(f"two records of suite seed {r['seed']} disagree at update {MAZE_END}")
    out = {"new": new, "reference": [suite[s] for s in sorted(suite)]}
    json.dump(out, open(out_json, "w"), indent=1)
    for name, recs in out.items():
        for r in recs:
            print(name, {k: v for k, v in r.items() if not k.startswith("trace")})


def classify(values, key):
    """The rules' verdict for one task and read point.

    Args:
        values: list of int active counts of the finished new networks (None where a network has no read there);
        key: a key of BOUNDS.
    Returns:
        (verdict string, list of -1/0/+1 per value: below, inside, above the bounds).
    """
    lo, hi, _, m = BOUNDS[key]
    vals = [v for v in values if v is not None]
    side = [(-1 if v < lo - m else 1 if v > hi + m else 0) for v in vals]
    if len(vals) < 2:
        return "unresolved (fewer than two networks)", side
    if side.count(1) >= 2 or side.count(-1) >= 2:
        return "DEPENDS", side
    if not any(side):
        return "not detected", side
    return "unresolved", side


def summary(values, refs, key):
    """Printed line for one task and read point.

    Args:
        values: active counts of the new networks (None allowed); refs: counts of the reference networks (None
        allowed); key: a key of BOUNDS.
    Returns:
        str.
    """
    lo, hi, mean, m = BOUNDS[key]
    v = np.array([x for x in values if x is not None], float)
    r = np.array([x for x in refs if x is not None], float)
    got = (int(r.min()), int(r.max()), int(round(r.mean())))
    if got != (lo, hi, mean):
        raise ValueError(f"{key}: the reference networks read {got}, the rules record {(lo, hi, mean)}")
    verdict, side = classify(values, key)
    line = f"{key[0]} {key[1]}: new {[int(x) for x in v]} | reference n={len(r)} {lo}-{hi} mean {r.mean():.1f}"
    if len(v) > 1:
        se = np.sqrt(v.var(ddof=1) / len(v) + r.var(ddof=1) / len(r))
        dof = se ** 4 / ((v.var(ddof=1) / len(v)) ** 2 / (len(v) - 1) + (r.var(ddof=1) / len(r)) ** 2 / (len(r) - 1))
        half = stats.t.ppf(0.975, dof) * se
        line += (f" | mean difference {v.mean() - r.mean():+.1f} ({100 * (v.mean() - r.mean()) / r.mean():+.1f}%), "
                 f"95% Welch interval {v.mean() - r.mean() - half:+.1f} to {v.mean() - r.mean() + half:+.1f}")
    return line + f" | bounds {lo - m}-{hi + m} sides {side} -> {verdict}"


def do_verdict(reads_json, maze_json, fig):
    """Mode `verdict` (module docstring).

    Args:
        reads_json, maze_json: outputs of the two reads; fig: path stem of the figure, or None.
    Returns:
        None.
    """
    R = json.load(open(reads_json))
    M = json.load(open(maze_json))
    for task in ("3BFF", "CDDM"):
        new = R["new"][task]
        need = 75000 if task == "3BFF" else 100000
        done = [r for r in new if r["it_end"] >= need - 1]
        if len(done) != N_NEW:
            raise SystemExit(f"{task}: {len(done)} of {N_NEW} networks finished; no verdict on an incomplete set")
        if any(r["draws"] != "independent" for r in new):
            raise SystemExit(f"{task}: a new network is not independent")
        ref = R["reference"][task]
        print("\n" + summary([r["active_75k"] for r in done], [r["active_75k"] for r in ref], (task, "75k")))
        print(summary([r["active_floor"] for r in done], [r["active_floor"] for r in ref], (task, "floor")))
        c = [r["r2_75k"] for r in ref if "r2_75k" in r]
        n = [r["r2_75k"] for r in done]
        ok = abs(np.mean(n) - np.mean(c)) <= R2_TOL
        print(f"{task} held-out R2 at 75k: new {[round(x, 4) for x in n]} mean {np.mean(n):.4f} "
              f"({[r['r2_how'] for r in done]}); the paper's {[round(x, 4) for x in c]} mean {np.mean(c):.4f} "
              f"-> {'within' if ok else 'OUTSIDE'} {R2_TOL}")
        print(f"{task} 500 or more active: {[r['seed'] for r in done if r['active_75k'] >= 500]}; skipped updates {[r['skipped'] for r in done]}")
    new = M["new"]
    if len(new) != N_NEW or any(r["updates"] < MAZE_END for r in new):
        raise SystemExit(f"Maze: incomplete ({len(new)} networks, last updates {[r['updates'] for r in new]})")
    print("\n" + summary([r["active_u2000"] for r in new], [r["active_u2000"] for r in M["reference"]], ("Maze", "u2000")))
    print(summary([r["active_learned"] for r in new], [r["active_learned"] for r in M["reference"]], ("Maze", "learned")))
    print(f"Maze probe success at update 2,000: new {[round(r['success_u2000'], 3) for r in new]}, suite "
          f"{[round(r['success_u2000'], 3) for r in M['reference']]}; learned at {[r['learned_update'] for r in new]}; "
          f"restored {[r['restored'] for r in new]}, skipped {[r['skipped'] for r in new]}")
    if fig:
        draw(R, M, fig)


def draw(R, M, fig):
    """Active units over training: the new networks against the reference range at the primary read point.

    Args:
        R, M: the two reads; fig: path stem (PDF and PNG written).
    Returns:
        None.
    """
    f, axes = plt.subplots(1, 3, figsize=(10.5, 3.2), constrained_layout=True)
    panels = [("3BFF", "iterations", [(r["trace_it"], r["trace_active"]) for r in R["new"]["3BFF"]], "75k", FIXED_IT,
               "independent"),
              ("CDDM", "iterations", [(r["trace_it"], r["trace_active"]) for r in R["new"]["CDDM"]], "75k", FIXED_IT,
               "independent"),
              ("Maze", "updates", [(r["trace_u"], r["trace_active"]) for r in M["new"]], "u2000", MAZE_END, "shared")]
    for ax, (task, xlab, series, key, at, mode) in zip(axes, panels):
        lo, hi, _, m = BOUNDS[(task, key)]
        for x, y in series:
            ax.plot(x, y, color="#2a78d6", lw=1)
        ax.axvspan(at * 0.97, at * 1.03, color="0.9", lw=0)
        ax.errorbar([at], [(lo + hi) / 2], yerr=[[(lo + hi) / 2 - lo], [hi - (lo + hi) / 2]], color="0.3", lw=2, capsize=4)
        ax.errorbar([at], [(lo + hi) / 2], yerr=[[(lo + hi) / 2 - lo + m], [hi + m - (lo + hi) / 2]], color="0.6", lw=1, capsize=2)
        ax.set(title=f"{task}: {mode} draws", xlabel=xlab, ylabel="active units of 1,000")
    f.savefig(fig + ".pdf")
    f.savefig(fig + ".png", dpi=150)


def main():
    """Parse the mode and run it."""
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["read", "maze-read", "verdict"])
    ap.add_argument("path")
    ap.add_argument("maze_json", nargs="?")
    ap.add_argument("--logs", default=os.path.expanduser("~/trainRNNbrain_logs"))
    ap.add_argument("--new")
    ap.add_argument("--suite")
    ap.add_argument("--fig")
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.path, a.logs)
    elif a.mode == "maze-read":
        do_maze_read(a.path, a.new, a.suite)
    else:
        do_verdict(a.path, a.maze_json, a.fig)


if __name__ == "__main__":
    main()
