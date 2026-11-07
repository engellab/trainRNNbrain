#!/usr/bin/env python3
"""Read the round 18 training package by its rules (docs/project_trajectory/2026-11-07_0110_round18-training-rules.md
and 2026-11-07_0136_round18-parity-rules.md).

Three parts, each a subcommand:

  cddm   (Della CPU node) G18, the CDDM transfer test: every run of CDDMHyper_b4_transfer/EqType=h_N=1000_<cell>
         read as B4 (read_b2_b4.b4_run's quantities) at the floor crossing at 1.05x, and the gate and
         verdict (read_b2_b4.b4_verdict) at 1.05x, 1.10x and 1.20x the floor and at 75k (the last
         probe). Then the parity cells on CDDM at 75k (read_uniform_cells.read_cell, 36 draws) against
         CDDM_paper_grid's control with the paper's criterion (read_uniform_cells.verdict) at
         tolerances 0.005, 0.01 and 0.02, and against the second control at 0.01.
  maze   (Mac, on the rsynced trees) G11, G12, G17 and the Maze parity arms.
         G11: per continued run, success at the last probe, learned or not on the joined record, the
         learned update, active units at the learned probe, at update 2,000 and at update 4,000.
         G12: per new control, the same at 2,000 updates; the weight-decay test (any new learned control
         with 417 or fewer active units at update 2,000); the kept-snapshot check.
         G17: per transfer and fresh run on the second maze, S0 (inactive in the first stored
         participation vector), revived (S0 active in the first stored vector at or after the learned
         update), newly inactive, active at transfer, at the learned probe and at the end; the gate and
         the verdict of the rules entry (B4's bars; "every" and "two thirds" over the old runs read).
         Parity: per arm, active_end, success_end and status, and decision #194's criterion against
         uniform-series controls s1-s3, with the two learned controls as the sensitivity check.

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/round18_training_read.py cddm OUT_DIR
  python trainRNNbrain/experiments_and_analysis/round18_training_read.py maze --root data/trained_RNNs_rl_maze OUT_DIR
Each writes OUT_DIR/round18_<part>.json and prints the tables. The held-out R^2 cache is
OUT_DIR/round18_r2_cache.json, this script's own file.
"""

import argparse
import glob
import json
import math
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
from common import DATA_DIR, active_mask, maze_learned_index, maze_load_runs, read_run, runs_with_loss
from fig_ed_maze import read_points
from fig_ed_maze_interventions import stopped_by_guard
from read_b2_b4 import b4_run, b4_verdict
from read_uniform_cells import read_cell, verdict

CDDM_ROOT = f"{DATA_DIR}/CDDMHyper_b4_transfer"
CDDM_CELLS = ("src=control_age=old", "src=fresh")
MARGINS = (0.10, 0.20)
CDDM_CTRL = "CDDM_paper_grid/EqType=h_N=1000_arm=control"
CDDM_CTRL2 = "CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0"
CDDM_PARITY = {"sigma_w 0.1": "CDDM_synnoise_ext/EqType=h_N=1000_sw=0.1_iters=75000",
               "sigma_w 0.3": "CDDM_synnoise_ext/EqType=h_N=1000_sw=0.3_iters=75000",
               "dropout 0.05, beta 4": "CDDM_parity_r18/EqType=h_N=1000_arm=mute0.05",
               "rescaling, target 4.0": "CDDM_target_ladder/EqType=h_N=1000_tgt=4.0"}
TOLS = (0.005, 0.01, 0.02)
MAZE_PARITY = ("sigmaw0.1", "sigmaw0.3", "dropout0.05_beta4", "rescale_tgt4")
WD_MAX = 417            # the largest active count of weight decay 1e-5 on Maze (rules entry, G12)
SUCCESS_TOL = 0.05      # decision #194: mean success within 0.05 of the controls' mean
LEVEL = 0.9             # decision #194: no network ends below this success


def margin_record(run_dir, delta, base):
    """A B4 record re-read at another floor margin: the crossing and the counts there.

    Args:
        run_dir: str, a run folder; delta: float, the margin (0.10 for 1.10x the floor);
        base: dict, b4_run's record at 1.05x for the same run (S0 and the transfer counts).
    Returns:
        dict with reason, t_star, active0, n_S0, active_star, revived (the keys b4_verdict reads).
    """
    rd = read_run(run_dir, delta=delta)
    rec = dict(name=base["name"], reason=rd["reason"], active0=base["active0"], n_S0=base["n_S0"])
    if rd["reason"] is not None:
        return rec
    with open(glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))[0], "rb") as fh:
        tr = pickle.load(fh)
    P = np.asarray(tr["participation"])
    it_a = np.asarray(tr["participation_iters"], float)
    S0 = ~active_mask(P[0])
    j = np.flatnonzero(it_a >= rd["t_star"])
    Astar = active_mask(P[j[0]] if len(j) else P[-1])
    rec.update(t_star=float(rd["t_star"]), active_star=int(Astar.sum()), revived=int((S0 & Astar).sum()))
    return rec


def end_record(base):
    """A B4 record read at the end of training (75k): the last probe stands in for the crossing.

    Args:
        base: dict, b4_run's record.
    Returns:
        dict with the keys b4_verdict reads, reason None.
    """
    return dict(name=base["name"], reason=None, active0=base["active0"], n_S0=base["n_S0"],
                active_star=base["active_end"], revived=base["S0_active_end"])


def read_cddm(out_dir):
    """G18 and the CDDM parity cells, printed and written to OUT_DIR/round18_cddm.json.

    Args:
        out_dir: str, the output folder (also holds the R^2 cache).
    """
    cache_path = os.path.join(out_dir, "round18_r2_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    cells = {}
    for cell in CDDM_CELLS:
        runs = runs_with_loss(f"{CDDM_ROOT}/EqType=h_N=1000_{cell}")
        cells[cell] = [b4_run(r["run_dir"], cache) for r in runs]
        for r, rec in zip(runs, cells[cell]):
            rec["run_dir"] = r["run_dir"]
    print("G18  cell                  run        t*      act0  |S0|  revived  newly  act*  S0&Aend  act_end  R2")
    for cell, recs in cells.items():
        for r in recs:
            if r["reason"]:
                print(f"     {cell:<21s} {r['name'][:10]} EXCLUDED: {r['reason']}  (act0 {r['active0']}, act_end {r['active_end']})")
                continue
            print(f"     {cell:<21s} {r['name'][:10]} {r['t_star']:>7.0f} {r['active0']:>5d} {r['n_S0']:>5d} "
                  f"{r['revived']:>7d} {r['newly_inactive']:>6d} {r['active_star']:>5d} {r['S0_active_end']:>7d} "
                  f"{r['active_end']:>7d}  {r['r2']:.4f}{'' if r['r2_measured'] else ' (est)'}")
    verdicts = {"1.05": b4_verdict(cells)}
    for d in MARGINS:
        verdicts[f"{1 + d:.2f}"] = b4_verdict({c: [margin_record(r["run_dir"], d, r) for r in recs]
                                               for c, recs in cells.items()})
    verdicts["75k"] = b4_verdict({c: [end_record(r) for r in recs] for c, recs in cells.items()})
    for k, v in verdicts.items():
        print(f"G18 verdict at {k}: {json.dumps(v)}")

    print("\nParity on CDDM at 75k")
    ctrl = read_cell(CDDM_CTRL, cache, H.R2_DRAWS)
    ctrl2 = read_cell(CDDM_CTRL2, cache, H.R2_DRAWS)
    parity = {}
    for label, cell in CDDM_PARITY.items():
        arm = read_cell(cell, cache, H.R2_DRAWS)
        if not arm:
            parity[label] = dict(cell=cell, runs=[], text="no finished network: fails")
            print(f"  {label}: no finished network")
            continue
        out = dict(cell=cell, runs=arm)
        for tol in TOLS:
            v = verdict(arm, ctrl)
            v["r2_ok"] = bool(v["r2_gap"] >= -tol)
            v["works"] = v["count_ok"] and v["r2_ok"]
            out[f"tol={tol}"] = v
        v2 = verdict(arm, ctrl2)
        out["second_control"] = v2
        parity[label] = out
        v = out["tol=0.01"]
        print(f"  {label:24s} active {[r['active'] for r in arm]} (control max {v['max_active_ctrl']}), "
              f"R2 {[round(r['r2'], 4) for r in arm]} mean {v['mean_r2_arm']:.4f} vs {v['mean_r2_ctrl']:.4f} "
              f"(gap {v['r2_gap']:+.4f}): " + ", ".join(f"tol {t}: {'pass' if out[f'tol={t}']['works'] else 'fail'}"
                                                         for t in TOLS)
              + f"; second control: {v2['text']}")
    json.dump(cache, open(cache_path, "w"))
    json.dump(dict(g18=cells, g18_verdicts=verdicts, parity=parity, control=ctrl, second_control=ctrl2),
              open(os.path.join(out_dir, "round18_cddm.json"), "w"), indent=1, default=str)


def at_update(run, update):
    """The active count at the probe of a given update, or None if no probe has it.

    Args:
        run: dict from common.maze_load_runs; update: int.
    Returns:
        int or None.
    """
    hit = np.flatnonzero(np.asarray(run["probes"]["update"]) == update)
    return None if not len(hit) else int(run["probes"]["active"][hit[0]])


def maze_row(run):
    """The descriptive row of one maze run (G11, G12): success, learned point, counts.

    Args:
        run: dict from common.maze_load_runs.
    Returns:
        dict: seed, N, last_update, success_end, learned (update or None), active_learned,
        active_2000, active_4000, active_end.
    """
    pt = read_points(run)
    return dict(seed=run["seed"], N=run["N"], last_update=int(run["probes"]["update"][-1]),
                success_end=pt["success_end"], learned=pt["learned"], active_learned=pt["active_learned"],
                active_2000=at_update(run, 2000), active_4000=at_update(run, 4000), active_end=pt["active_end"])


def transfer_row(run, root):
    """One G17 run read on the second maze: S0, revival at the learned point, the counts.

    Args:
        run: dict from common.maze_load_runs (a Maze_transfer run); root: str, the maze tree, to find
            the source run (the saved config's paths.transfer_from, mapped onto `root`).
    Returns:
        dict: seed, cell, source, source_learned (bool or None for fresh runs), learned, active0
        (probe after update 1), n_S0, revived, newly_inactive, active_star (probe at the learned
        point), active_A (count in the stored vector A*), S0_active_end, active_end, success_end,
        p_update0 and a_update (the updates of the vectors used).
    """
    pr = run["probes"]
    P, pu = np.asarray(pr["p"]), np.asarray(pr["p_update"])
    A0 = active_mask(P[0])
    S0 = ~A0
    Aend = active_mask(P[-1])
    pt = read_points(run)
    cell = os.path.basename(os.path.dirname(run["folder"]))
    cfg_src = None
    for f in glob.glob(os.path.join(run["folder"], "*config.yaml")):
        for line in open(f):
            if line.strip().startswith("transfer_from:"):
                cfg_src = line.split(":", 1)[1].strip().strip("'\"")
    src_learned = None
    if cfg_src and cfg_src != "null":
        tail = cfg_src.split("trained_RNNs_rl_maze/")[-1]
        src_runs = [r for r in maze_load_runs(os.path.join(root, os.path.dirname(os.path.dirname(tail))))
                    if os.path.abspath(r["chain"][0]) == os.path.abspath(os.path.join(root, tail))]
        if src_runs:
            # the source's state at transfer is its update-2,000 record: the chain's first run
            first = src_runs[0]
            n2000 = int(np.searchsorted(np.asarray(first["probes"]["update"]), 2000, side="right"))
            src_learned = maze_learned_index(np.asarray(first["probes"]["probe_success"])[:n2000]) is not None
    rec = dict(seed=run["seed"], cell=cell, source=cfg_src, source_learned=src_learned, learned=pt["learned"],
               active0=int(pr["active"][0]), n_S0=int(S0.sum()), S0_active_end=int((S0 & Aend).sum()),
               active_end=pt["active_end"], success_end=pt["success_end"], p_update0=int(pu[0]))
    if pt["learned"] is None:
        rec["reason"] = "no learned point on the second maze"
        return rec
    j = np.flatnonzero(pu >= pt["learned"])
    Astar = active_mask(P[j[0]]) if len(j) else Aend
    rec.update(reason=None, active_star=pt["active_learned"], active_A=int(Astar.sum()),
               revived=int((S0 & Astar).sum()), newly_inactive=int((A0 & ~Astar).sum()),
               a_update=int(pu[j[0]]) if len(j) else int(pu[-1]))
    return rec


def transfer_verdict(old_all, fresh_all):
    """G17's gate and verdict (rules entry): B4's bars, "every" and "two thirds" over the old runs read.

    Args:
        old_all: list of transfer_row dicts for the old runs whose source learned the first maze;
        fresh_all: list of transfer_row dicts for the fresh runs.
    Returns:
        dict: verdict ("do not come back", "can be recruited", "between the bars", "no verdict"),
        gate, the numbers used, and the per-run revived shares (reported with or without a verdict).
    """
    old = [r for r in old_all if r["reason"] is None]
    fresh = [r for r in fresh_all if r["reason"] is None]
    need_old = max(2, math.ceil(2 * len(old_all) / 3))
    share = [r["revived"] / r["n_S0"] for r in old]
    base = dict(n_old_eligible=len(old_all), n_old_read=len(old), need_old=need_old, n_fresh=len(fresh),
                revived_share=[float(s) for s in share])
    if len(old) < need_old or len(fresh) < 2:
        return dict(base, verdict="no verdict")
    f_mean = float(np.mean([r["active_star"] for r in fresh]))
    f_sd = float(np.std([r["active_star"] for r in fresh], ddof=1))
    old_at_transfer = float(np.mean([r["active0"] for r in old]))
    old_star = float(np.mean([r["active_star"] for r in old]))
    gate = bool(f_mean - old_at_transfer > 3 * f_sd)
    if all(s < 0.10 for s in share) and f_mean - old_star > 3 * f_sd:
        v = "do not come back"
    elif sum(s >= 0.30 for s in share) >= math.ceil(2 * len(old) / 3) or abs(f_mean - old_star) <= f_sd:
        v = "can be recruited"
    else:
        v = "between the bars"
    if not gate:
        v = "gate fails: no B4 verdict (" + v + " by the bars alone)"
    return dict(base, verdict=v, gate=gate, fresh_mean=f_mean, fresh_sd=f_sd, old_at_transfer=old_at_transfer,
                old_star=old_star)


def parity_maze(root):
    """The Maze parity arms against decision #194's criterion.

    Args:
        root: str, the maze tree (holds Maze_interventions/ and Maze_suite/).
    Returns:
        dict arm -> dict(runs, verdict, verdict_learned_controls).
    """
    # the controls' update-2,000 state (their G11 continuation is joined onto the record)
    ctrl_runs = {r["seed"]: r for r in maze_load_runs(os.path.join(root, "Maze_suite"))
                 if r["N"] == 1000 and r["act"] == "relu" and r["seed"] in (1, 2, 3)}
    c_active, c_success = {}, {}
    for s, r in ctrl_runs.items():
        n2000 = int(np.searchsorted(np.asarray(r["probes"]["update"]), 2000, side="right"))
        c_active[s] = int(r["probes"]["active"][n2000 - 1])
        c_success[s] = float(r["probes"]["probe_success"][n2000 - 1])
    learned_ctrl = [s for s in ctrl_runs if c_success[s] >= LEVEL]
    out = {"controls": dict(active=c_active, success=c_success, learned=learned_ctrl)}
    arms = {}
    for r in maze_load_runs(os.path.join(root, "Maze_interventions")):
        arm = os.path.basename(os.path.dirname(r["folder"]))
        if arm in MAZE_PARITY:
            arms.setdefault(arm, []).append(r)
    for arm in MAZE_PARITY:
        runs = arms.get(arm, [])
        rows = []
        for r in runs:
            pt = read_points(r)
            last = int(r["probes"]["update"][-1])
            status = "finished" if last >= 2000 else ("stopped" if stopped_by_guard(r) else "unfinished")
            rows.append(dict(seed=r["seed"], status=status, last_update=last, active_end=pt["active_end"],
                             success_end=pt["success_end"], learned=pt["learned"]))
        fin = [x for x in rows if x["status"] == "finished"]

        def crit(seeds):
            """decision #194's criterion against the controls `seeds`; returns (pass, reasons)."""
            if not fin:
                return False, ["no finished network"]
            why = []
            cmax = max(c_active[s] for s in seeds)
            cmean = float(np.mean([c_success[s] for s in seeds]))
            if not all(x["active_end"] > cmax for x in fin):
                why.append(f"not every network above {cmax} active")
            if np.mean([x["success_end"] for x in fin]) < cmean - SUCCESS_TOL:
                why.append(f"mean success {np.mean([x['success_end'] for x in fin]):.3f} below {cmean - SUCCESS_TOL:.3f}")
            if any(x["success_end"] < LEVEL for x in fin):
                why.append("a network ends below 0.9")
            return not why, why
        ok, why = crit([1, 2, 3])
        ok2, why2 = crit(learned_ctrl)
        out[arm] = dict(runs=rows, passes=ok, why=why, passes_learned_controls=ok2, why_learned_controls=why2)
    return out


def read_maze(root, out_dir):
    """G11, G12, G17 and the Maze parity arms, printed and written to OUT_DIR/round18_maze.json.

    Args:
        root: str, the rsynced maze tree (holds Maze_suite/, Maze_transfer/, Maze_interventions/ and
            the sibling <root>_checkpoints/); out_dir: str, the output folder.
    """
    suite = maze_load_runs(os.path.join(root, "Maze_suite"))
    g11 = [maze_row(r) for r in suite if r["act"] == "relu" and r["N"] in (250, 500, 1000)
           and len(r["chain"]) == 2]
    print("G11  N     seed  last   success  learned  act@learned  act@2000  act@4000")
    for x in g11:
        print(f"     {x['N']:<5d} {x['seed']:<5d} {x['last_update']:<6d} {x['success_end']:.3f}    "
              f"{str(x['learned']):<8s} {str(x['active_learned']):<12s} {str(x['active_2000']):<9s} {x['active_4000']}")
    g12 = [maze_row(r) for r in suite if r["act"] == "relu" and r["N"] == 1000 and r["seed"] in (4, 5, 6)]
    snaps = {}
    for x in g12:
        d = os.path.join(root + "_checkpoints", "Maze_suite", "N=1000_act=relu", f"s{x['seed']}")
        names = sorted(os.listdir(d)) if os.path.isdir(d) else []
        ups = [int(n[1:]) for n in names if n.startswith("u")]
        after = [u for u in ups if x["learned"] is not None and u >= x["learned"]]
        snaps[x["seed"]] = dict(n=len(ups), first=min(ups) if ups else None, last=max(ups) if ups else None,
                                gap_after_learned=(after[0] - x["learned"]) if after else None)
    print("\nG12  seed  last   success  learned  act@learned  act@2000  snapshots")
    for x in g12:
        print(f"     {x['seed']:<5d} {x['last_update']:<6d} {x['success_end']:.3f}    {str(x['learned']):<8s} "
              f"{str(x['active_learned']):<12s} {str(x['active_2000']):<9s} {snaps[x['seed']]}")
    new_learned = [x for x in g12 if x["learned"] is not None and x["success_end"] >= LEVEL]
    wd_lost = [x["seed"] for x in new_learned if x["active_2000"] is not None and x["active_2000"] <= WD_MAX]
    g12_test = dict(new_learned=[x["seed"] for x in new_learned], at_or_below_417=wd_lost,
                    result=("no new seed learned: the control range is unchanged" if not new_learned else
                            f"weight decay 1e-5 loses its Maze direction (seed(s) {wd_lost} at or below 417)"
                            if wd_lost else "every new learned control is above 417: the weight-decay direction stands"))
    print("G12 test:", json.dumps(g12_test))

    tr = maze_load_runs(os.path.join(root, "Maze_transfer"))
    rows = [transfer_row(r, root) for r in tr]
    print("\nG17  cell              seed  src_learned  learned  act0  |S0|  revived  share   newly  act*  S0&Aend  act_end  success")
    for x in rows:
        if x["reason"]:
            print(f"     {x['cell']:<17s} {x['seed']:<5d} {str(x['source_learned']):<12s} EXCLUDED: {x['reason']} "
                  f"(act0 {x['active0']}, act_end {x['active_end']}, success {x['success_end']:.3f})")
            continue
        print(f"     {x['cell']:<17s} {x['seed']:<5d} {str(x['source_learned']):<12s} {x['learned']:<8d} {x['active0']:<5d} "
              f"{x['n_S0']:<5d} {x['revived']:<8d} {x['revived'] / x['n_S0']:.3f}  {x['newly_inactive']:<6d} "
              f"{x['active_star']:<5d} {x['S0_active_end']:<8d} {x['active_end']:<8d} {x['success_end']:.3f}")
    old = [x for x in rows if x["cell"].endswith("src=old") and x["source_learned"]]
    fresh = [x for x in rows if x["cell"].endswith("src=fresh")]
    g17 = transfer_verdict(old, fresh)
    print("G17 verdict:", json.dumps(g17))

    parity = parity_maze(root)
    print("\nParity on Maze (decision #194)")
    print("  controls:", json.dumps(parity["controls"]))
    for arm in MAZE_PARITY:
        p = parity[arm]
        print(f"  {arm:20s} {[(x['seed'], x['status'], x['active_end'], round(x['success_end'], 3)) for x in p['runs']]}: "
              f"{'pass' if p['passes'] else 'fail: ' + '; '.join(p['why'])}; learned controls: "
              f"{'pass' if p['passes_learned_controls'] else 'fail: ' + '; '.join(p['why_learned_controls'])}")
    json.dump(dict(g11=g11, g12=g12, g12_snapshots=snaps, g12_test=g12_test, g17=rows, g17_verdict=g17,
                   parity=parity), open(os.path.join(out_dir, "round18_maze.json"), "w"), indent=1, default=str)


def main():
    """Parse the subcommand and run its read."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="part", required=True)
    c = sub.add_parser("cddm", help="G18 and the CDDM parity cells (Della)")
    c.add_argument("out_dir")
    m = sub.add_parser("maze", help="G11, G12, G17 and the Maze parity arms")
    m.add_argument("--root", default="data/trained_RNNs_rl_maze")
    m.add_argument("out_dir")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    if a.part == "cddm":
        read_cddm(a.out_dir)
    else:
        read_maze(os.path.abspath(a.root), a.out_dir)


if __name__ == "__main__":
    main()
