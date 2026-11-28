#!/usr/bin/env python3
"""The entropy set point (lambda_ent = 0.1) across network sizes: reads, verdicts and figures.

RULES. docs/project_trajectory/2026-11-25_1241_rules-entropy-setpoint-sizes.md, committed before any run. The size test
(S) and the M09 criterion at a size (M) are defined there; names below follow it. The rate-penalty size test (B) is added
by docs/project_trajectory/2026-11-25_1259_rules-entropy-sizes-rate-penalty-test.md.

MODE `read` (Della CPU node). One cell (--cell, a key of CELLS, e.g. ff500) with its controls: per network the floor
crossing (read_remedies.read_one: active units at 1.05x, 1.10x and 1.20x, held-out R^2 at the crossing), the 75,000
read (fig_paper_F2_fails.read_fixed), the end of the budget (last probe, R^2 of the final weights), the joined
participation trace, |H_i - 0.75| on the final weights, skipped updates summed over the segments, and the config
differences of every segment from the first control's. Writes OUT_DIR/<cell>.json (R^2 cache beside it).

MODE `maze` (Mac). Maze N = 500: the arm (--arm, the copied Maze_interventions folder) and the controls (--suite, the
Mac's Maze_suite) by claim_gaps_read.maze_read, cut at update 4,000. Writes OUT_DIR/maze500.json.

MODE `verdict` (Mac). Checks the controls against their recorded values (--recorded, the main checkout's data folder)
and the number of networks; prints S, M and question 2 per cell, and prune-and-duplicate beside them; draws the
figures at --fig (PDF and PNG).

Usage:
  python entropy_sizes_read.py read OUT_DIR --cell ff500 [--logs DIR]      (repository root on PYTHONPATH)
  python entropy_sizes_read.py maze OUT_DIR --arm DIR --suite DIR
  python entropy_sizes_read.py verdict OUT_DIR --recorded DIR --fig PATH_STEM
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

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H  # noqa: E402
import fig_paper_F5_tasks as F5T  # noqa: E402
from claim_gaps_read import maze_cells  # noqa: E402
from common import DATA_DIR, load_run, maze_load_runs, runs_with_loss, warm_start_chain  # noqa: E402
from entropy_setpoint_read import flat_cfg, maze_point, skipped, unit_stats, window_median  # noqa: E402
from fig_paper_F2_fails import read_fixed  # noqa: E402
from read_remedies import read_one  # noqa: E402

FIXED_ITER = 75000
TOL, MAZE_TOL, MAZE_FLOOR = 0.01, 0.05, 0.9
LOSS_SLOPE_FROM, SLOPE_BAR = 150000, 5.0      # G7: units lost per 100,000 iterations on probes from 150,000
FF, CD = "NBitFlipFlop_entropy_setpoint", "CDDM_entropy_setpoint"
REV, GRID = "NBitFlipFlop_ff_revive_c200000", "NBitFlipFlop_paper_grid_c200000"
# cell -> task, N, the arm's last segment, the controls, their number, the end of the budget, Q2 read points, judged
Q2 = {"3BFF": (50000, 75000, 100000, 125000, 150000, 175000, 200000), "CDDM": (50000, 75000, 100000)}
CELLS = {}
for _n, _ctrl, _k in ((500, f"{REV}/EqType=h_k=3_N=500_pen=none_arm=none", 6),
                      (1000, f"{REV}/EqType=h_k=3_N=1000_pen=none_arm=none", 4),
                      (2000, f"{REV}/EqType=h_k=3_N=2000_pen=none_arm=none", 3)):
    CELLS[f"ff{_n}"] = dict(task="3BFF", N=_n, ctrl=_ctrl, n_ctrl=_k, judged=_n != 1000,
                            end=75000 if _n == 1000 else 200000,
                            arm=f"{FF if _n == 1000 else FF + '_c200000'}/EqType=h_N={_n}_arm=ent0.1")
for _n in (500, 1000, 2000, 4000):
    CELLS[f"cddm{_n}"] = dict(task="CDDM", N=_n, ctrl=f"CDDM_paper_grid/EqType=h_N={_n}_arm=control", n_ctrl=3,
                              judged=_n != 1000, end=75000 if _n == 1000 else 100000,
                              arm=f"{CD + '_c100000' if _n == 4000 else CD}/EqType=h_N={_n}_arm=ent0.1")
# Maze N = 500 (entry 2026-11-20_1600, gap A): learned point and update 4,000, success at 4,000
MAZE_CTRL_RECORDED = [(281, 226, 1.000), (259, 226, 0.992), (279, 240, 1.000)]
MAZE_DUP_RECORDED = [(416, 440, 1.000), (401, 406, 1.000), (409, 436, 1.000)]
POINTS = (("active", "r2", "crossing 1.05x"), ("active_d10", None, "crossing 1.10x"),
          ("active_d20", None, "crossing 1.20x"), ("active_75k", "r2_75k", "75,000"))
# (B) bars, and noisy synapses' fold change at the crossing (Maze: learned point), from Supplementary Table 4 and gaps B, C
B_FOLD, B_R2, B_SKIP = 1.5, 0.02, 0.05
SYN_FOLD = {("3BFF", 1000): 2.03, ("3BFF", 2000): 1.54, ("3BFF", 4000): 1.60, ("CDDM", 1000): 1.11, ("Maze", 1000): 0.98}
COL = {"control": "#898781", "ent": "#2a78d6", "dup": "#eb6834"}     # dataviz reference palette, slots 1 and 6


def nonfinite(chain):
    """Number of non-finite values in the saved training loss of a warm-start chain.

    Args:
        chain: list of run folders.
    Returns:
        int.
    """
    n = 0
    for d in chain:
        L = json.load(open(glob.glob(os.path.join(glob.escape(d), "*TrainLosses.json"))[0]))["train_losses"]
        n += int(np.sum(~np.isfinite(np.asarray(L, float))))
    return n


def read_network(run_dir, cache, ref_cfg, logs):
    """One network read at its floor crossing, at 75,000 iterations and at the end of its budget.

    Args:
        run_dir: str, the last segment's folder; cache: dict, an open heldout_r2 cache (filled in place);
        ref_cfg: dict, a control's flattened config, or None for a control; logs: folder of Slurm logs.
    Returns:
        JSON-ready dict: read_remedies.read_one's keys, r2_75k_fixed / how / probe, r2_end, trace_it, trace_active,
        it_end, stats (unit_stats), nonfinite (non-finite saved losses), skipped (sum over segments, None if a log is missing), config_differs (one
        dict per segment, oldest first) and chain (folders, oldest first).
    """
    rec = read_one(run_dir, cache)
    r = load_run(run_dir)
    fx = read_fixed(r, FIXED_ITER, cache)
    chain = list(warm_start_chain(run_dir))
    rec.update(active_75k=int(fx["active"]), probe_75k=fx["probe"], r2_75k_fixed=float(fx["r2"]), r2_75k_how=fx["r2_how"],
               short=fx["short"], r2_end=float(H.heldout_r2(run_dir, draws=H.R2_DRAWS, cache=cache)),
               loss_end=float(np.asarray(r["it_loss"])[-1]), trace_it=[float(x) for x in r["it_act"]],
               trace_active=[int(x) for x in r["active"]], stats=unit_stats(run_dir), chain=chain, nonfinite=nonfinite(chain))
    if ref_cfg is not None:
        sk = [skipped(d, logs, "Ent*") for d in chain]
        rec["skipped"] = None if None in sk else int(sum(sk))
        rec["config_differs"] = []
        for d in chain:
            c = flat_cfg(d)
            rec["config_differs"].append({k: [str(c.get(k, "<absent>")), str(ref_cfg.get(k, "<absent>"))]
                                          for k in sorted(set(c) | set(ref_cfg))
                                          if c.get(k, "<absent>") != ref_cfg.get(k, "<absent>")})
    return rec


def do_read(out_dir, cell, logs):
    """Mode `read`: one cell and its controls.

    Args:
        out_dir: str, output folder (created); cell: key of CELLS; logs: folder of Slurm logs.
    Returns:
        None; writes OUT_DIR/<cell>.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, f"r2cache_{cell}.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    s = CELLS[cell]
    out = dict(cell=cell, **s)
    ref = None
    for group, path in (("control", s["ctrl"]), ("ent", s["arm"])):
        out[group + "_runs"] = []
        for r in runs_with_loss(os.path.join(DATA_DIR, path)):
            rec = read_network(r["run_dir"], cache, ref, logs)
            out[group + "_runs"].append(rec)
            print(f"{cell} {group} seed {rec['seed']} t* {rec.get('t_star')} active {rec.get('active')} r2 {rec.get('r2')} "
                  f"reason {rec['reason']} | 75k {rec['active_75k']} {rec['r2_75k_fixed']:.4f} ({rec['r2_75k_how']}) | end "
                  f"{rec['active_end']} {rec['r2_end']:.4f} | skipped {rec.get('skipped')}", flush=True)
            json.dump(cache, open(cache_path, "w"))
        n_folders = len([d for d in glob.glob(os.path.join(DATA_DIR, path, "*")) if os.path.isdir(d)])
        out[group + "_folders"] = n_folders
        if group == "control":
            ref = flat_cfg(out["control_runs"][0]["run_dir"])
        json.dump(out, open(os.path.join(out_dir, f"{cell}.json"), "w"), indent=1)


def do_maze(out_dir, arm_dir, suite):
    """Mode `maze`: Maze N = 500, the arm and the controls at the learned point and at update 4,000.

    Args:
        out_dir: str; arm_dir: the copied Maze_interventions folder; suite: the Mac's Maze_suite folder.
    Returns:
        None; writes OUT_DIR/maze500.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    ctrl, arm = maze_cells(suite)["N=500_act=relu"], maze_cells(arm_dir)["ent0.1"]
    out = dict(control_runs=[ctrl[k] for k in (1, 2, 3)], ent_runs=[arm[k] for k in sorted(arm)])
    pen = {int(r["seed"]): maze_point(r, 4000)["penalty"].get("entropy_setpoint") for r in maze_load_runs(arm_dir)}
    for x in out["ent_runs"]:
        x["penalty"] = pen[x["seed"]]
    for g in ("control_runs", "ent_runs"):
        for x in out[g]:
            print(g, x["N"], x["seed"], x["last_update"], x["learned_update"], x["r1"], x["r2"], x["success_end"],
                  x["skipped_sum"], x["restored_sum"], x.get("penalty"))
    json.dump(out, open(os.path.join(out_dir, "maze500.json"), "w"), indent=1, default=str)


def recorded(data_dir):
    """The controls' and prune-and-duplicate's recorded reads, from the main checkout's data files.

    Args:
        data_dir: str, holding remedy_reads_31oct/remedies_catalogue.json and round18/g6_reads.json.
    Returns:
        dict (task, N, "control" | "dup") -> dict of lists: active, r2, active_75k, r2_75k (one value per network that
        has one; 3BFF from the catalogue, CDDM from G6).
    """
    out = {}
    for row in json.load(open(os.path.join(data_dir, "remedy_reads_31oct", "remedies_catalogue.json"))):
        who = "control" if row["role"] == "control" else "dup" if row["cell"].startswith(GRID) and "duplication" in row["cell"] else None
        if who:
            out[("3BFF", row["N"], who)] = {k: list(row[k]) for k in ("active", "r2", "active_75k", "r2_75k")}
    g6 = json.load(open(os.path.join(data_dir, "round18", "g6_reads.json")))
    for key, recs in g6.items():
        n, arm = key.split("|")
        if arm in ("control", "duplication"):
            out[("CDDM", int(n), "control" if arm == "control" else "dup")] = {
                k: [r[k] for r in recs if r.get(k) is not None] for k in ("active", "r2", "active_75k", "r2_75k")}
    return out


def r2_75(rec, task):
    """The network's R^2 at 75,000 by M09's read, or None: 3BFF only where weights at 75,000 exist; CDDM also estimated."""
    return None if task == "3BFF" and rec["r2_75k_how"] == "estimate" else rec["r2_75k_fixed"]


def gate(c, rec):
    """Compare a cell's controls with their recorded values; raise SystemExit on any difference.

    Args:
        c: the cell's reads (do_read's dict); rec: recorded()[(task, N, "control")].
    Returns:
        None.
    """
    runs = c["control_runs"]
    got = dict(active=[r["active"] for r in runs], active_75k=[r["active_75k"] for r in runs],
               r2=[r["r2"] for r in runs], r2_75k=[r2_75(r, c["task"]) for r in runs if r2_75(r, c["task"]) is not None])
    for k in ("active", "active_75k"):
        if sorted(got[k]) != sorted(int(x) for x in rec[k]):
            raise SystemExit(f"GATE FAILS {c['cell']}: {k} {sorted(got[k])}, recorded {sorted(rec[k])}")
    for k in ("r2", "r2_75k"):
        a, b = np.sort(got[k]), np.sort(rec[k])
        if len(a) != len(b) or np.max(np.abs(a - b)) > 5e-4:
            raise SystemExit(f"GATE FAILS {c['cell']}: {k} {a}, recorded {b}")
    print(f"  gate passes: controls' counts exact, R^2 within {max(np.max(np.abs(np.sort(got[k]) - np.sort(rec[k]))) for k in ('r2', 'r2_75k')):.1e}")


def count_ok(a, b):
    """The count condition: every value of a is larger than every value of b (False when a or b is empty)."""
    return bool(len(a) and len(b) and min(a) > max(b))


def criterion(act, perf, cact, cperf, tol, floor=None):
    """fig_paper_F5_tasks.rule with the empty case returned as a failure.

    Args:
        act, perf, cact, cperf: lists (arm counts, arm performance, control counts, control performance); tol, floor.
    Returns:
        dict from rule.
    """
    if not act:
        return dict(works=False, text="no network", fold=float("nan"), dperf=float("nan"), count_ok=False)
    return F5T.rule(act, perf, cact, cperf, tol, floor=floor)


def slope_per_100k(rec, frm, to):
    """Units lost per 100,000 iterations: minus the least-squares slope of the active count on probes in [frm, to].

    Args:
        rec: a network dict; frm, to: iterations.
    Returns:
        float, or nan with fewer than 10 probes.
    """
    t, a = np.asarray(rec["trace_it"]), np.asarray(rec["trace_active"], float)
    m = (t >= frm) & (t <= to)
    return float(-np.polyfit(t[m], a[m], 1)[0] * 1e5) if m.sum() >= 10 else float("nan")


def b_outcome(task, n, n_stable, fold, dperf, tol):
    """The rate-penalty size test (B) at one size.

    Args:
        task, n: the cell; n_stable: stable networks of three; fold: fold change at the read point (nan if none);
        dperf: mean performance minus the controls' (nan if none); tol: 0.02 in R^2, 0.05 in success.
    Returns:
        dict: holds (the pass conditions at this size), fails (list of fail conditions met; only N >= 1,000), text.
    """
    holds = bool(n_stable >= 2 and fold >= B_FOLD and dperf >= -tol)
    fails = []
    if n >= 1000:
        if n_stable < 2:
            fails.append(f"{n_stable} of 3 stable")
        if dperf < -tol:
            fails.append(f"performance {dperf:+.4f}")
        if (task, n) in SYN_FOLD and fold < SYN_FOLD[(task, n)]:
            fails.append(f"fold {fold:.2f} below noisy synapses' {SYN_FOLD[(task, n)]}")
    return dict(holds=holds, fails=fails, n_stable=n_stable, fold=fold, dperf=dperf,
                text="fail: " + "; ".join(fails) if fails else "pass conditions hold" if holds else "inconclusive")


def judge_b(c):
    """(B) for a 3BFF or CDDM cell at 1.05x (binds), 1.10x and 1.20x, strict and lenient finite loss; printed.

    Args:
        c: the cell's reads.
    Returns:
        b_outcome dict at 1.05x under the strict reading.
    """
    arm, ctrl, out = c["ent_runs"], c["control_runs"], None
    for strict in (True, False):
        for akey, lab in (("active", "1.05x"), ("active_d10", "1.10x"), ("active_d20", "1.20x")):
            a, k = ([r[akey] for r in rs if r.get(akey) is not None] for rs in (arm, ctrl))
            ns = sum(r.get(akey) is not None and r["skipped"] is not None and r["skipped"] < B_SKIP * r["loss_end"]
                     and (not strict or r["nonfinite"] == 0) for r in arm)
            ra, rk = ([r["r2"] for r in rs if r.get("r2") is not None] for rs in (arm, ctrl))
            b = b_outcome(c["task"], c["N"], ns, float(np.mean(a) / np.mean(k)) if a else float("nan"),
                          float(np.mean(ra) - np.mean(rk)) if ra else float("nan"), B_R2)
            if strict or lab == "1.05x":
                print(f"  B at {lab}{'' if strict else ' (lenient finite loss)'}: {b['n_stable']} of 3 stable, fold "
                      f"{b['fold']:.2f}, dR2 at 1.05x {b['dperf']:+.4f}: {b['text']}")
            if strict and lab == "1.05x":
                out = b
    print(f"    skipped {[r['skipped'] for r in arm]}, non-finite losses {[r['nonfinite'] for r in arm]} "
          f"(controls {[r['nonfinite'] for r in ctrl]}), bursts {[r.get('n_bursts') for r in arm]} (controls {[r.get('n_bursts') for r in ctrl]})")
    return out


def judge(c, rec):
    """Print and return the verdicts of one 3BFF or CDDM cell.

    Args:
        c: the cell's reads; rec: recorded() dict.
    Returns:
        dict: points (label -> arm, control, fold, dr2, count_ok), S, M, dup_M, M_end, q2.
    """
    task, n = c["task"], c["N"]
    ctrl, arm = c["control_runs"], c["ent_runs"]
    print(f"\n=== {task}, N = {n:,} ({len(arm)} networks, {len(ctrl)} controls){'' if c['judged'] else ' (description only)'}")
    if len(ctrl) != c["n_ctrl"] or len(arm) != 3 or c["ent_folders"] != 3 or c["control_folders"] != c["n_ctrl"]:
        raise SystemExit(f"INCOMPLETE {c['cell']}: {len(arm)} networks in {c['ent_folders']} folders, {len(ctrl)} controls")
    if any(r["loss_end"] < c["end"] - 100 for r in arm):
        raise SystemExit(f"INCOMPLETE {c['cell']}: last iterations {[r['loss_end'] for r in arm]}")
    gate(c, rec[(task, n, "control")])
    for r in ctrl + arm:
        r["r2_75k"] = r2_75(r, task)
    out = dict(points={})
    for akey, rkey, lab in POINTS:
        a = [r[akey] for r in arm if r.get(akey) is not None]
        k = [r[akey] for r in ctrl if r.get(akey) is not None]
        d = dict(arm=a, control=k, fold=float(np.mean(a) / np.mean(k)) if a and k else None, count_ok=count_ok(a, k), dr2=None)
        if rkey:
            ra = [r[rkey] for r in arm if r.get(akey) is not None and r.get(rkey) is not None]
            rk = [r[rkey] for r in ctrl if r.get(akey) is not None and r.get(rkey) is not None]
            d["dr2"] = float(np.mean(ra) - np.mean(rk)) if ra and rk else None
        out["points"][lab] = d
        print(f"  {lab:>14}: {a} against {k}; fold {d['fold'] if d['fold'] is None else round(d['fold'], 2)}, "
              f"dR2 {d['dr2'] if d['dr2'] is None else round(d['dr2'], 4)}, count condition {d['count_ok']}")
    print(f"  no crossing: {[(r['seed'], r['reason']) for r in arm + ctrl if r['reason']] or 'none'}")
    for r in arm:
        t = r.get("t_star")
        print(f"    seed {r['seed']}: t* {t if t is None else round(t)}, active {r.get('active')}, R2 {r.get('r2')} "
              f"(measured {r.get('r2_measured')}); 75k {r['active_75k']} R2 {r['r2_75k']:.4f} ({r['r2_75k_how']}); end "
              f"{r['active_end']} R2 {r['r2_end']:.4f}; |H-0.75| {r['stats']['med_dev_H']:.3f}; skipped {r['skipped']}; "
              f"segments {len(r['chain'])}")
    print(f"    controls' |H-0.75| {[round(r['stats']['med_dev_H'], 3) for r in ctrl]}")
    out["B"] = judge_b(c)
    if not c["judged"]:
        return out
    out["S"] = out["points"]["crossing 1.05x"]["count_ok"] and out["points"]["75,000"]["count_ok"]
    ca, cr = [r["active_75k"] for r in ctrl], [r["r2_75k"] for r in ctrl if r["r2_75k"] is not None]
    m = criterion([r["active_75k"] for r in arm], [r["r2_75k"] for r in arm], ca, cr, TOL)
    dup = rec[(task, n, "dup")]
    md = criterion(dup["active_75k"], dup["r2_75k"], ca, cr, TOL)
    me = criterion([r["active_end"] for r in arm], [r["r2_end"] for r in arm], [r["active_end"] for r in ctrl],
                   [r["r2_end"] for r in ctrl], TOL)
    out.update(M=m, dup_M=md, M_end=me)
    print(f"  S: {'PASSES' if out['S'] else 'FAILS'}")
    print(f"  M at 75,000: {m['text']}; fold {m['fold']:.2f}, dR2 {m['dperf']:+.4f} | prune-and-duplicate: {md['text']}; "
          f"fold {md['fold']:.2f}, dR2 {md['dperf']:+.4f}")
    print(f"  M at {c['end']:,} (end): {me['text']}; fold {me['fold']:.2f}, dR2 {me['dperf']:+.4f}")
    pts = [t for t in Q2[task] if t <= c["end"]]
    q = {g: [[window_median(r["trace_it"], r["trace_active"], t) for t in pts] for r in rs]
         for g, rs in (("ent", arm), ("control", ctrl))}
    i75 = pts.index(FIXED_ITER)
    for g, rows in q.items():
        fell = [row[-1] < row[i75] for row in rows]
        print(f"  A(t) {g} at {pts}: {rows}; falls from 75,000 to the end in {sum(fell)} of {len(fell)}: "
              f"{'continues' if all(fell) else 'does not continue' if not any(fell) else 'mixed'}")
        if c["end"] >= 200000:
            sl = [slope_per_100k(r, LOSS_SLOPE_FROM, c["end"]) for r in (arm if g == "ent" else ctrl)]
            print(f"     units lost per 100,000 from 150,000: {np.round(sl, 1).tolist()}; decreasing in "
                  f"{sum(x >= SLOPE_BAR for x in sl)} of {len(sl)}")
    survive = [count_ok([row[i] for row in q["ent"]], [row[i] for row in q["control"]]) for i in range(len(pts))]
    print(f"  count condition of A(t) at {pts}: {survive}")
    out["q2"] = dict(points=pts, rows=q, count_ok=survive)
    for i, r in enumerate(arm):
        for j, diff in enumerate(r["config_differs"]):
            print(f"    config, network {i + 1}, segment {j + 1}: "
                  f"{ {k: v[0] for k, v in diff.items() if not k.startswith(('paths.', 'seed', 'experiment_tag'))} }")
    return out


def judge_maze(m):
    """Print and return the verdicts of Maze N = 500.

    Args:
        m: maze500.json's dict.
    Returns:
        dict: S, M, dup_M, points.
    """
    ctrl, arm = m["control_runs"], m["ent_runs"]
    print("\n=== Maze, N = 500")
    if len(arm) != 3 or any(not x["complete"] for x in arm):
        raise SystemExit("INCOMPLETE maze500")
    got = sorted((x["r1"], x["r2"], round(x["success_end"], 3)) for x in ctrl)
    if got != sorted(MAZE_CTRL_RECORDED):
        raise SystemExit(f"GATE FAILS maze500: {got}, recorded {MAZE_CTRL_RECORDED}")
    print("  gate passes: controls' counts and success exact")
    pts = {}
    for k, lab in (("r1", "learned point"), ("r2", "update 4,000")):
        a, b = [x[k] for x in arm], [x[k] for x in ctrl]
        ok = None not in a and count_ok(a, b)
        pts[lab] = dict(arm=a, control=b, count_ok=ok, fold=float(np.mean([x for x in a if x is not None]) / np.mean(b)))
        print(f"  {lab}: {a} against {b}; fold {pts[lab]['fold']:.2f}; count condition {ok}")
    ma = criterion([x["r2"] for x in arm], [x["success_end"] for x in arm], [x["r2"] for x in ctrl],
                   [x["success_end"] for x in ctrl], MAZE_TOL, MAZE_FLOOR)
    md = criterion([x[1] for x in MAZE_DUP_RECORDED], [x[2] for x in MAZE_DUP_RECORDED], [x["r2"] for x in ctrl],
                   [x["success_end"] for x in ctrl], MAZE_TOL, MAZE_FLOOR)
    S = pts["learned point"]["count_ok"] and pts["update 4,000"]["count_ok"]
    ok = [x for x in arm if x["r1"] is not None and x["flagged_share"] < B_SKIP]
    B = b_outcome("Maze", 500, len(ok), pts["learned point"]["fold"],
                  float(np.mean([x["success_end"] for x in arm]) - np.mean([x["success_end"] for x in ctrl])), MAZE_TOL)
    print(f"  B: {B['n_stable']} of 3 stable (flagged shares {[round(x['flagged_share'], 4) for x in arm]}), fold "
          f"{B['fold']:.2f}, dsuccess {B['dperf']:+.3f}: {B['text']}")
    for x in arm:
        print(f"    s{x['seed']}: learned {x['learned_update']}, {x['r1']} and {x['r2']} active, success {x['success_end']}, "
              f"skipped {x['skipped_sum']:g}, restored {x['restored_sum']:g}, term {x['penalty']}")
    print(f"  S: {'PASSES' if S else 'FAILS'}\n  M at update 4,000: {ma['text']}; fold {ma['fold']:.2f}, dsuccess "
          f"{ma['dperf']:+.3f} | prune-and-duplicate: {md['text']}; fold {md['fold']:.2f}, dsuccess {md['dperf']:+.3f}")
    return dict(S=S, M=ma, dup_M=md, points=pts, B=B)


def draw_traces(cells, stem):
    """Active units against iterations, one panel per 3BFF and CDDM cell, every network and control.

    Args:
        cells: dict cell -> reads; stem: figure path without extension.
    Returns:
        None; writes <stem>_traces.pdf and .png.
    """
    names = [k for k in ("ff500", "ff2000", "cddm500", "cddm2000", "cddm4000") if k in cells]
    rows = [[k for k in names if k.startswith(p)] for p in ("ff", "cddm")]
    fig, axes = plt.subplots(2, max(map(len, rows)), figsize=(4.1 * max(map(len, rows)), 6.4), squeeze=False, constrained_layout=True)
    for i, row in enumerate(rows):
        for j, ax in enumerate(axes[i]):
            if j >= len(row):
                ax.set_axis_off()
                continue
            c = cells[row[j]]
            for g, lab in (("control", "controls"), ("ent", "entropy set point, λ = 0.1")):
                for k, r in enumerate(c[g + "_runs"]):
                    ax.plot(np.asarray(r["trace_it"]) / 1000, r["trace_active"], color=COL[g], lw=1.2, alpha=0.85,
                            label=lab if k == 0 else None)
                    if r.get("t_star") is not None:
                        ax.plot(r["t_star"] / 1000, r["active"], "o", ms=6, color=COL[g], mec="white", mew=1, zorder=5)
            ax.axvline(75, color="0.75", lw=0.8)
            ax.set_xlim(0, c["end"] / 1000)
            ax.set_ylim(0, c["N"])
            ax.set_title(f"{c['task']}, N = {c['N']:,}", fontsize=10)
            ax.set_xlabel("iteration (thousands)")
            ax.set_ylabel(f"active units (of {c['N']:,})")
            ax.grid(alpha=0.25, lw=0.5)
    axes[0][0].legend(fontsize=8, frameon=False, loc="upper right")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}_traces.{ext}", dpi=150)
    print(f"figure: {stem}_traces.pdf, .png")


def draw_folds(cells, maze, rec, stem):
    """Fold change in active units against N, entropy set point and prune-and-duplicate, at both read points.

    Args:
        cells: dict cell -> reads (with judge's r2_75k); maze: maze500 dict or None; rec: recorded(); stem: path stem.
    Returns:
        None; writes <stem>_folds.pdf and .png.
    """
    cols = [("3BFF", "active", "3BFF, floor crossing"), ("3BFF", "active_75k", "3BFF, 75,000 iterations"),
            ("CDDM", "active", "CDDM, floor crossing"), ("CDDM", "active_75k", "CDDM, 75,000 iterations")]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharey=True, constrained_layout=True)
    for ax, (task, key, title) in zip(axes, cols):
        mean = {"ent": [], "dup": []}
        for c in (x for x in cells.values() if x["task"] == task):
            n = c["N"]
            base = np.mean([r[key] for r in c["control_runs"] if r.get(key) is not None])
            vals = {"ent": [r[key] for r in c["ent_runs"] if r.get(key) is not None],
                    "dup": rec[(task, n, "dup")][key] if (task, n, "dup") in rec else []}
            for g, dx in (("ent", 0.96), ("dup", 1.04)):
                if not len(vals[g]):
                    continue
                y = np.asarray(vals[g]) / base
                if g == "ent" and not c["judged"]:   # N = 1,000: seen before the rules, description only: hollow, off the line
                    ax.plot([n * dx] * len(y), y, "o", ms=5, mfc="white", mec=COL[g], mew=1.0)
                    continue
                ax.plot([n * dx] * len(y), y, "o", ms=5, color=COL[g], alpha=0.7, mec="white", mew=0.6)
                mean[g].append((n * dx, float(np.mean(y))))
        for g, lab in (("ent", "entropy set point, λ = 0.1"), ("dup", "prune-and-duplicate")):
            if mean[g]:
                ax.plot(*zip(*sorted(mean[g])), color=COL[g], lw=1.8, label=lab)
        ax.axhline(1, color="0.6", lw=0.8)
        ax.set_xscale("log")
        ax.set_xticks([500, 1000, 2000], ["500", "1,000", "2,000"])
        ax.minorticks_off()
        ax.set_title(title, fontsize=10)
        ax.set_xlabel("network size N")
        ax.grid(alpha=0.25, lw=0.5)
    axes[0].set_ylim(0, None)
    axes[0].set_ylabel("active units / controls' mean")
    axes[0].legend(fontsize=8, frameon=False, loc="upper left")
    for ext in ("pdf", "png"):
        fig.savefig(f"{stem}_folds.{ext}", dpi=150)
    print(f"figure: {stem}_folds.pdf, .png")


def do_verdict(out_dir, data_dir, stem):
    """Mode `verdict`: every cell found in OUT_DIR, then the statements per task and the figures.

    Args:
        out_dir: str, folder of <cell>.json and maze500.json; data_dir: the recorded reads; stem: figure stem or None.
    Returns:
        None.
    """
    rec = recorded(data_dir)
    cells = {k: json.load(open(os.path.join(out_dir, f"{k}.json"))) for k in CELLS if os.path.exists(os.path.join(out_dir, f"{k}.json"))}
    V = {}
    for k, c in cells.items():
        try:
            V[k] = judge(c, rec)
        except SystemExit as e:                  # an incomplete cell or a failed gate gives no verdict, and says so
            print(f"  NO VERDICT: {e}")
    cells = {k: c for k, c in cells.items() if k in V}
    mp = os.path.join(out_dir, "maze500.json")
    maze = json.load(open(mp)) if os.path.exists(mp) else None
    if maze:
        try:
            V["maze500"] = judge_maze(maze)
        except SystemExit as e:
            print(f"  NO VERDICT: {e}")
    print("\n=== statements")
    for task, pre in (("3BFF", "ff"), ("CDDM", "cddm"), ("Maze", "maze")):
        ks = [k for k in V if re.match(pre + r"\d", k) and "S" in V[k]]
        print(f"  {task}: cells {ks}; S fails at {[k for k in ks if not V[k]['S']] or 'none'}; "
              f"M fails at {[k for k in ks if not V[k]['M']['works']] or 'none'}; prune-and-duplicate's M fails at "
              f"{[k for k in ks if not V[k]['dup_M']['works']] or 'none'}; B: { {k: V[k]['B']['text'] for k in ks} }")
    if stem:
        draw_traces(cells, stem)
        draw_folds(cells, maze, rec, stem)


def main():
    """Parse the command line and run a mode."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=("read", "maze", "verdict"))
    ap.add_argument("out_dir")
    ap.add_argument("--cell", choices=sorted(CELLS))
    ap.add_argument("--logs", default=os.path.expanduser("~/trainRNNbrain_logs"))
    ap.add_argument("--arm")
    ap.add_argument("--suite")
    ap.add_argument("--recorded")
    ap.add_argument("--fig")
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.out_dir, a.cell, a.logs)
    elif a.mode == "maze":
        do_maze(a.out_dir, a.arm, a.suite)
    else:
        do_verdict(a.out_dir, a.recorded, a.fig)


if __name__ == "__main__":
    main()
