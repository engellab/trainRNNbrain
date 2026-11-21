#!/usr/bin/env python3
"""Read the claim-gap tests of 19 Nov 2026 by their pre-set rules.

Rules (written and committed before any run):
  gap A  docs/project_trajectory/2026-11-19_1749_rules-maze-duplicate-sizes.md
  gap B  docs/project_trajectory/2026-11-19_1749_rules-cddm-rate-penalty-sizes.md
  gap C  docs/project_trajectory/2026-11-19_1749_rules-maze-rate-penalty-sizes.md

maze (the Mac, on rsynced trees; gaps A and C). Every Maze run is joined along its warm-start chain
(common.maze_load_runs) and cut at update 4,000. R1 = active count at the learned probe
(common.maze_learned_index), R2 = active count at update 4,000. Controls: Maze_suite default ReLU
networks, seeds 1-3 at each size (seeds 4-6 at N = 1,000 reported, not binding).

cddm (Della CPU node; gap B). B2's read (read_b2_b4.py, heldout_r2.read_rung): each run at its own
noisy-loss floor crossing at 1.05x (binding), 1.10x and 1.20x; held-out R^2 at the crossing
(heldout_r2.floor_r2: measured on a kept snapshot, else estimated); updates skipped from the job log's
"gradient spikes" line; a run whose folder name starts with "nan" (non-finite loss) counts as unstable.

Usage:
  python claim_gaps_read.py maze --new-root DIR --suite DIR --sigmaw DIR --out FILE.json
      DIR of --new-root holds cg_N500, cg_N1000, cg_N2000, each a copy of a Della trained_RNNs_rl_maze_cg_N<N>
      tree (cg_N1000 also holds copies of the fz parents under Maze_interventions/<arm>/s<seed>).
  python claim_gaps_read.py cddm --data DIR --logs DIR --sigma CELL --out FILE.json
      DIR of --data holds trained_RNNs and trained_RNNs_cg (Della: ~/trainRNNbrain/data); CELL is the CDDM
      sigma_w 1.0 cell CDDM_synnoise_ext/EqType=h_N=1000_sw=1.0_iters=100000 (on the Mac only; copied to Della).
"""

import argparse
import glob
import json
import os
import re
import sys

import numpy as np
import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import common as C
import heldout_r2 as H

MAX_UPDATE = 4000          # every Maze run is read on updates 1-4,000
SIZES = (500, 1000, 2000)
MAZE_TOL = 0.05            # the paper's Maze success tolerance (B2's 0.02 in R^2)
R2_TOL = 0.02              # B2's R^2 tolerance
MULT_BAR = 1.5             # B2's active-count multiplier
SKIP_BAR = 0.05            # B2's share of skipped updates
CDDM_DELTAS = (0.05, 0.10, 0.20)


# ----------------------------------------------------------------------------------------------- Maze

def maze_cells(root):
    """Every Maze run under root, one per (cell, seed), cut at update MAX_UPDATE.

    Args:
        root: str, a folder of <cell>/s<seed>/ run folders (common.maze_load_runs's layout).
    Returns:
        dict cell name (the folder of the chain's OLDEST run, e.g. "duplicate" or "N=500_act=relu") ->
        dict seed -> maze_read dict. Branches of one chain (long-training continuations) give the same
        record once cut at MAX_UPDATE; the first is kept.
    """
    out = {}
    for r in C.maze_load_runs(root):
        cell = os.path.basename(os.path.dirname(r["chain"][0]))
        out.setdefault(cell, {}).setdefault(int(r["seed"]), maze_read(r))
    return out


def maze_read(r):
    """The read-out of one joined Maze run, cut at update MAX_UPDATE.

    Args:
        r: dict from common.maze_load_runs (probes: update, active, probe_success; progress: one dict
           per PPO update with update, skipped, restored).
    Returns:
        dict: N, seed, chain (folders), last_update (int), complete (reached MAX_UPDATE), learned_update
        (int or None), r1 (active count at the learned probe or None), r2 (active count at update
        MAX_UPDATE or None), success_end (probe success at MAX_UPDATE or None), flagged_share (share of
        updates 1..last with skipped > 0 or restored > 0), skipped_sum, restored_sum.
    """
    P = r["probes"]
    u = np.asarray(P["update"])
    m = u <= MAX_UPDATE
    u, act, succ = u[m], np.asarray(P["active"])[m], np.asarray(P["probe_success"])[m]
    prog = [p for p in r["progress"] if p["update"] <= MAX_UPDATE]
    last = int(prog[-1]["update"]) if prog else 0
    complete = last >= MAX_UPDATE and len(u) > 0 and int(u[-1]) == MAX_UPDATE
    li = C.maze_learned_index(succ)
    flagged = sum(1 for p in prog if p.get("skipped", 0) > 0 or p.get("restored", 0) > 0)
    return dict(N=int(r["N"]), seed=int(r["seed"]), chain=list(r["chain"]), last_update=last,
                complete=bool(complete),
                learned_update=None if li is None else int(u[li]),
                r1=None if li is None else int(act[li]),
                r2=int(act[-1]) if complete else None,
                success_end=float(succ[-1]) if complete else None,
                flagged_share=flagged / max(last, 1),
                skipped_sum=float(sum(p.get("skipped", 0) for p in prog)),
                restored_sum=float(sum(p.get("restored", 0) for p in prog)))


def mean_of(rows, key):
    """Mean of rows[i][key] over the rows where it is not None (nan when there is none)."""
    v = [x[key] for x in rows if x[key] is not None]
    return float(np.mean(v)) if v else float("nan")


def count_condition(arm, ctrl, key):
    """G6's count condition: every arm network's count above every control's (an arm network without
    a count fails it).

    Args:
        arm, ctrl: lists of maze_read dicts; key: "r1" or "r2".
    Returns:
        dict holds (bool), arm (list of counts), ctrl (list of counts), fold (mean arm / mean ctrl).
    """
    a, c = [x[key] for x in arm], [x[key] for x in ctrl if x[key] is not None]
    holds = len(arm) == 3 and all(v is not None for v in a) and len(c) > 0 and min(a) > max(c)
    return dict(holds=bool(holds), arm=a, ctrl=c, fold=mean_of(arm, key) / mean_of(ctrl, key))


def gap_a(arm_by_n, ctrl_by_n, ctrl46):
    """Gap A by its rule: the count condition at R1 and R2 at every size.

    Args:
        arm_by_n: dict N -> list of prune-and-duplicate maze_read dicts; ctrl_by_n: dict N -> list of
        control maze_read dicts (seeds 1-3); ctrl46: list of N = 1,000 controls seeds 4-6 (not binding).
    Returns:
        dict per N (r1, r2 count_condition dicts, success difference, networks below 0.9 at the end),
        the N = 1,000 check against seeds 4-6, and verdict ("increases at every size" or the failing cells).
    """
    out, failing = {}, []
    for N in SIZES:
        arm, ctrl = arm_by_n[N], ctrl_by_n[N]
        cells = {k: count_condition(arm, ctrl, k) for k in ("r1", "r2")}
        failing += [f"N={N} {k}" for k, v in cells.items() if not v["holds"]]
        out[N] = dict(**cells, success_diff=mean_of(arm, "success_end") - mean_of(ctrl, "success_end"),
                      below_0p9=[x["seed"] for x in arm if x["success_end"] is not None and x["success_end"] < 0.9])
    out["n1000_vs_seeds4_6"] = {k: count_condition(arm_by_n[1000], ctrl46, k) for k in ("r1", "r2")}
    out["verdict"] = "increases at every size" if not failing else "fails at " + ", ".join(failing)
    return out


def gap_c(arm_by_n, ctrl_by_n, sigma_rows):
    """Gap C by B2's rule in its Maze form.

    Args:
        arm_by_n: dict N -> list of rate-penalty maze_read dicts; ctrl_by_n: dict N -> controls (seeds
        1-3); sigma_rows: N = 1,000 sigma_w 1.0 maze_read dicts.
    Returns:
        dict per N (stable count, multiplier, success cost, per-run stability), the sigma_w multiplier at
        N = 1,000, and verdict ("supports", "falsifies: ..." or "neither").
    """
    out, supports, falsify = {}, True, []
    sigma_mult = mean_of(sigma_rows, "r1") / mean_of(ctrl_by_n[1000], "r1")
    for N in SIZES:
        arm, ctrl = arm_by_n[N], ctrl_by_n[N]
        stable = [x["complete"] and x["r1"] is not None and x["flagged_share"] < SKIP_BAR for x in arm]
        mult = mean_of(arm, "r1") / mean_of(ctrl, "r1")
        cost = mean_of(arm, "success_end") - mean_of(ctrl, "success_end")
        n_st = int(sum(stable))
        ok = len(arm) == 3 and n_st >= 2 and mult >= MULT_BAR and cost >= -MAZE_TOL
        supports &= bool(ok)
        if N >= 1000:
            if n_st < 2:
                falsify.append(f"N={N}: {n_st} of 3 stable")
            if cost < -MAZE_TOL:
                falsify.append(f"N={N}: success {cost:+.3f}")
            if N == 1000 and mult < sigma_mult:
                falsify.append(f"N={N}: multiplier {mult:.2f} below sigma_w's {sigma_mult:.2f}")
        out[N] = dict(stable=stable, n_stable=n_st, mult=mult, success_cost=cost, supports_here=bool(ok))
    out["sigma_mult_1000"] = sigma_mult
    out["verdict"] = "supports" if supports else ("falsifies: " + "; ".join(falsify) if falsify else "neither")
    return out


def maze_main(a):
    """Read gaps A and C and write the JSON (args: new_root, suite, sigmaw, out)."""
    suite = maze_cells(a.suite)
    ctrl_by_n = {N: [suite[f"N={N}_act=relu"][s] for s in (1, 2, 3)] for N in SIZES}
    ctrl46 = [suite["N=1000_act=relu"][s] for s in (4, 5, 6)]
    new = {N: maze_cells(os.path.join(a.new_root, f"cg_N{N}", "Maze_interventions")) for N in SIZES}
    rows = {arm: {N: [new[N][arm][s] for s in sorted(new[N].get(arm, {}))] for N in SIZES}
            for arm in ("duplicate", "frm0.1")}
    sigma = list(maze_cells(a.sigmaw)["sigmaw1.0"].values())
    res = dict(controls=ctrl_by_n, controls_1000_s4_6=ctrl46, runs=rows, sigmaw1000=sigma,
               gap_a=gap_a(rows["duplicate"], ctrl_by_n, ctrl46),
               gap_c=gap_c(rows["frm0.1"], ctrl_by_n, sigma))
    for arm in rows:
        for N in SIZES:
            for x in rows[arm][N]:
                print(f"{arm:9s} N={N:4d} s{x['seed']} last {x['last_update']:4d} learned {x['learned_update']} "
                      f"R1 {x['r1']} R2 {x['r2']} success {x['success_end']} flagged {x['flagged_share']:.3f}")
    print("gap A:", res["gap_a"]["verdict"])
    print("gap C:", res["gap_c"]["verdict"])
    json.dump(res, open(a.out, "w"), indent=1, default=str)


# ----------------------------------------------------------------------------------------------- CDDM

def log_skips(log_dir, patterns):
    """Updates skipped per run seed, from the "gradient spikes: X updates skipped" line of each job log.

    Args:
        log_dir: str, Della ~/trainRNNbrain_logs; patterns: list of glob patterns of .out files.
    Returns:
        dict seed (int) -> skipped updates (int), for every log that printed a seed and finished training.
        run_experiment.py prints the "gradient spikes" line only when an update was skipped or rolled
        back, so a log that reached "Executed training" without it records 0.
    """
    out = {}
    for pat in patterns:
        for f in glob.glob(os.path.join(log_dir, pat)):
            txt = open(f, errors="replace").read()
            seed = re.search(r"'seed': (\d+)", txt)
            sk = re.search(r"gradient spikes: (\d+) updates skipped", txt)
            if seed and (sk or "Executed training" in txt):
                out[int(seed.group(1))] = int(sk.group(1)) if sk else 0
    return out


def cddm_cell(path, cache, skips=None):
    """B2's read of one CDDM cell at each margin.

    Args:
        path: str, a cell folder (one folder per run); cache: an open heldout_r2 cache;
        skips: dict seed -> skipped updates (from log_skips), or None for a control.
    Returns:
        dict: n_folders, nan_folders (runs with a non-finite loss), and per margin (key "1.05", "1.10",
        "1.20") a list of per-run dicts: seed, t_star, active, r2, measured, reason, skip_share, stable.
    """
    dirs = [d for d in sorted(glob.glob(os.path.join(path, "*"))) if os.path.isdir(d)]
    nan = [d for d in dirs if os.path.basename(d).split("_")[0] == "nan"]
    out = dict(n_folders=len(dirs), nan_folders=len(nan))
    runs = C.runs_with_loss(path)
    for delta in CDDM_DELTAS:
        rows = []
        for r in runs:
            cfg = yaml.safe_load(open(glob.glob(os.path.join(r["run_dir"], "*config.yaml"))[0]))
            seed, max_iter = int(cfg["seed"]), int(cfg["trainer"]["max_iter"])
            rd = C.read_run(r["run_dir"], delta=delta, run=r)
            ok = rd["reason"] is None and rd["covered"]
            v, m = H.floor_r2(rd, cache) if ok else (None, None)
            sk = None if skips is None or seed not in skips else skips[seed] / max_iter
            rows.append(dict(seed=seed, t_star=rd["t_star"] if ok else None, active=rd["active"] if ok else None,
                             r2=v, measured=m, reason=rd["reason"] or (None if ok else "trace does not cover t*"),
                             skip_share=sk, stable=bool(ok and sk is not None and sk < SKIP_BAR)))
        out[f"{1 + delta:.2f}"] = rows
    return out


def cddm_verdict(arm, ctrl, sigma_mult):
    """B2's rule at one margin across sizes.

    Args:
        arm, ctrl: dict N -> list of per-run dicts (cddm_cell at that margin); sigma_mult: the
        sigma_w 1.0 multiplier at N = 1,000 (nan if unread).
    Returns:
        dict per N (n_stable, mult, dr2, supports_here) and verdict.
    """
    out, supports, falsify = {}, True, []
    for N in SIZES:
        a, c = arm[N], [x for x in ctrl[N] if x["active"] is not None]
        read = [x for x in a if x["active"] is not None]
        n_st = int(sum(x["stable"] for x in a))
        mult = (np.mean([x["active"] for x in read]) / np.mean([x["active"] for x in c])) if read and c else np.nan
        dr2 = (np.mean([x["r2"] for x in read]) - np.mean([x["r2"] for x in c])) if read and c else np.nan
        ok = n_st >= 2 and mult >= MULT_BAR and dr2 >= -R2_TOL
        supports &= bool(ok)
        if N >= 1000:
            if n_st < 2:
                falsify.append(f"N={N}: {n_st} of 3 stable")
            if dr2 < -R2_TOL:
                falsify.append(f"N={N}: R2 {dr2:+.4f}")
            if N == 1000 and mult < sigma_mult:
                falsify.append(f"N={N}: multiplier {mult:.2f} below sigma_w's {sigma_mult:.2f}")
        out[N] = dict(n_stable=n_st, mult=float(mult), dr2=float(dr2), supports_here=bool(ok))
    out["verdict"] = "supports" if supports else ("falsifies: " + "; ".join(falsify) if falsify else "neither")
    return out


def cddm_main(a):
    """Read gap B and write the JSON (args: data, logs, sigma, out)."""
    tr, cg = os.path.join(a.data, "trained_RNNs"), os.path.join(a.data, "trained_RNNs_cg")
    pen = "CDDM_std_g0_penalties_bias0/EqType=h_N={N}_pen=frm0.1_iters=75000"
    arm_path = {N: os.path.join(cg if N != 1000 else tr, pen.format(N=N)) for N in SIZES}
    ctrl_path = {N: os.path.join(tr, f"CDDM_paper_grid/EqType=h_N={N}_arm=control") for N in SIZES}
    sigma_path = a.sigma
    skips = log_skips(a.logs, ["CGcddm_N*.out", "StdPen.16957419_*.out"])
    cache_path = os.path.splitext(a.out)[0] + "_r2_cache.json"
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    arm = {N: cddm_cell(arm_path[N], cache, skips) for N in SIZES}
    ctrl = {N: cddm_cell(ctrl_path[N], cache) for N in SIZES}
    sigma = cddm_cell(sigma_path, cache)
    json.dump(cache, open(cache_path, "w"))
    verdicts = {}
    for k in (f"{1 + d:.2f}" for d in CDDM_DELTAS):
        s = [x["active"] for x in sigma[k] if x["active"] is not None]
        c = [x["active"] for x in ctrl[1000][k] if x["active"] is not None]
        sigma_mult = float(np.mean(s) / np.mean(c)) if s and c else float("nan")
        verdicts[k] = {"sigma_mult_1000": sigma_mult,
                       **cddm_verdict({N: arm[N][k] for N in SIZES}, {N: ctrl[N][k] for N in SIZES}, sigma_mult)}
        print(f"margin {k}: {verdicts[k]['verdict']}")
        for N in SIZES:
            v = verdicts[k][N]
            print(f"  N={N}: stable {v['n_stable']}/3, multiplier {v['mult']:.2f}, dR2 {v['dr2']:+.4f}")
    json.dump(dict(arm=arm, ctrl=ctrl, sigma=sigma, paths=dict(arm=arm_path, ctrl=ctrl_path, sigma=sigma_path),
                   skips=skips, verdicts=verdicts), open(a.out, "w"), indent=1, default=str)


def main():
    """Parse the subcommand and run it."""
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    m = sub.add_parser("maze")
    m.add_argument("--new-root", required=True)
    m.add_argument("--suite", required=True)
    m.add_argument("--sigmaw", required=True)
    m.add_argument("--out", required=True)
    c = sub.add_parser("cddm")
    c.add_argument("--data", required=True)
    c.add_argument("--logs", required=True)
    c.add_argument("--sigma", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    maze_main(a) if a.cmd == "maze" else cddm_main(a)


if __name__ == "__main__":
    main()
