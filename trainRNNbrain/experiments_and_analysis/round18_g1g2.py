#!/usr/bin/env python3
"""Round 18, G1 and G2: the 3BFF rate penalty at 75,000 iterations, and the per-network technique
criterion on every 3BFF value of Supplementary Table 3a (dead-relu-paper#201).

RULES. docs/project_trajectory/2026-11-07_0100_round18-rules.md (G1, G2), committed before this
script existed.

MODE `read` (Della, where the snapshots are). For each run of
NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm=frm{0.01,0.03,0.1}: the active count at the participation
probe nearest 75,000 (common.load_run, as read_remedies.read_one gives the catalogue's active_75k) and the
held-out R^2 of the kept snapshot at iteration 75,000 (heldout_r2.heldout_r2, R2_DRAWS = 36 draws, the
catalogue's r2_75k measure), plus the run's config fields of read_remedies.SHARED_FIELDS against the
catalogue control's. The N = 1,000 control is re-scored the same way on the three networks with a
75k state, as a check that this path reproduces the catalogue's control R^2. Writes OUT_DIR/g1_reads.json;
the R^2 cache is kept in that folder, not in the shared data/heldout_r2_cache.json.

MODE `verdict` (Mac). Reads g1_reads.json and data/remedy_reads_31oct/remedies_catalogue.json and prints:
  G1  per value: per-network active and R^2, the check against read_b3.json (0.03) and a3_N1000.json
      (0.1), and the criterion at R^2 tolerances 0.005, 0.01, 0.02;
  G2  per N = 1,000 catalogue cell of the six techniques (and the three rate-penalty values of G1): the
      count condition per network, the mean R^2 difference over every network with a 75k score (R^2 <= 0
      included, flagged), and pass/fail at each tolerance; per technique, the one-value rule;
  then the techniques passing on all three tasks at each R^2 tolerance, with the CDDM and Maze verdicts
  from fig_paper_F5_tasks (load_cddm, load_maze; unchanged, Maze at its tolerance 0.05).
Writes data/round18/g1g2.json.

Usage:
  python round18_g1g2.py read OUT_DIR           (Della CPU node, repository root on PYTHONPATH)
  python round18_g1g2.py verdict G1_JSON        (Mac)
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
import ablate_silent_units as A
from common import DATA_DIR, load_run
from read_remedies import SHARED_FIELDS, cfg_of, nearest_probe, sel
from fig_paper_F5_drive import load_a3, load_b3, load_catalogue
import fig_paper_F5_tasks as F5T

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_JSON = os.path.join(REPO, "data", "round18", "g1g2.json")
FIXED_ITER = 75000
LAMBDAS = ("0.01", "0.03", "0.1")
FRM_CELL = "NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm=frm{lam}"
CTRL_CELL = "NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none"
TOLS = (0.005, 0.01, 0.02)
TECHS = ("frm", "frm+rws", "synnoise", "rescale", "duplicate", "dropout")


def run_dirs(cell):
    """Run folders of a cell, skipping diverged ('nan_') folders.

    Args:
        cell: str, path relative to DATA_DIR.
    Returns:
        sorted list of absolute folder paths.
    """
    return sorted(d for d in glob.glob(os.path.join(DATA_DIR, cell, "*"))
                  if os.path.isdir(d) and not os.path.basename(d).startswith("nan"))


def state_at(run_dir, it=FIXED_ITER):
    """The folder of the run's weight state at iteration `it` (a kept snapshot or a segment end), or None.

    Args:
        run_dir: a run folder; it: absolute iteration.
    Returns:
        str folder holding that state's *LastParams* and *_config.yaml, or None.
    """
    its, files = A.chain_snapshots(run_dir)
    hit = [f for i, f in zip(its, files) if int(i) == it]
    return os.path.dirname(hit[0]) if hit else None


def read_run(run_dir, cache, ctrl_shared):
    """Active count and 36-draw held-out R^2 of one run at 75,000 iterations.

    Args:
        run_dir: str; cache: dict, an open held-out R^2 cache (filled in place);
        ctrl_shared: dict field -> value of the control's SHARED_FIELDS.
    Returns:
        JSON-ready dict: run_dir, active_75k, probe_75k, state (folder scored), r2_75k (36 draws),
        config_differs (dict field -> [run value, control value]).
    """
    r = load_run(run_dir)
    it_a, act = np.asarray(r["it_act"], float), np.asarray(r["active"], float)
    a75, p75 = nearest_probe(it_a, act, FIXED_ITER)
    st = state_at(run_dir)
    c = cfg_of(run_dir)
    shared = {k: sel(c, k) for k in SHARED_FIELDS}
    differs = {k: [str(shared[k]), str(ctrl_shared[k])] for k in SHARED_FIELDS if shared[k] != ctrl_shared[k]}
    r2 = H.heldout_r2(st, draws=H.R2_DRAWS, cache=cache) if st else None
    return dict(run_dir=run_dir, active_75k=a75, probe_75k=p75, state=st, r2_75k=r2, config_differs=differs)


def do_read(out_dir):
    """Mode `read`: score the three rate-penalty cells and re-score the control; write g1_reads.json.

    Args:
        out_dir: str, output folder (created).
    Returns:
        None.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "r2cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    ctrl_runs = run_dirs(CTRL_CELL)
    c0 = cfg_of(ctrl_runs[0])
    ctrl_shared = {k: sel(c0, k) for k in SHARED_FIELDS}
    out = {"control": [], "frm": {}}
    for d in ctrl_runs:
        rec = read_run(d, cache, ctrl_shared)
        out["control"].append(rec)
        print(f"control {os.path.basename(d)[:40]} active {rec['active_75k']} R2 {rec['r2_75k']}", flush=True)
    for lam in LAMBDAS:
        out["frm"][lam] = []
        for d in run_dirs(FRM_CELL.format(lam=lam)):
            rec = read_run(d, cache, ctrl_shared)
            out["frm"][lam].append(rec)
            print(f"frm {lam} {os.path.basename(d)[:40]} active {rec['active_75k']} R2 {rec['r2_75k']} "
                  f"config differs {rec['config_differs']}", flush=True)
        json.dump(cache, open(cache_path, "w"))
        json.dump(out, open(os.path.join(out_dir, "g1_reads.json"), "w"), indent=1)


def criterion(active, r2, ctrl_active, ctrl_r2, tol):
    """The paper's technique criterion on one value (fig_paper_F5_tasks.rule, no floor condition).

    Args:
        active: list of int per network; r2: list of float per network with a 75k score;
        ctrl_active, ctrl_r2: the control's lists; tol: float, R^2 tolerance.
    Returns:
        dict from fig_paper_F5_tasks.rule.
    """
    return F5T.rule(active, r2, ctrl_active, ctrl_r2, tol)


def value_row(name, fam, active, r2, ctrl_active, ctrl_r2, n_runs, source):
    """One judged value: per-network numbers, the count condition and the verdict at each tolerance.

    Args:
        name: str, the value's label; fam: technique key; active, r2: per-network lists;
        ctrl_active, ctrl_r2: the control's lists; n_runs: int, networks trained; source: str.
    Returns:
        JSON-ready dict.
    """
    r2 = [float(x) for x in r2]
    res = {f"{t}": criterion(active, r2, ctrl_active, ctrl_r2, t)["works"] for t in TOLS}
    r = criterion(active, r2, ctrl_active, ctrl_r2, 0.01)
    pos = [x for x in r2 if x > 0]
    dr2_pos = float(np.mean(pos) - np.mean(ctrl_r2)) if pos else None
    return dict(name=name, family=fam, source=source, n_runs=n_runs, active=list(map(int, active)), r2=r2,
                min_active=int(min(active)) if active else None, count_ok=r["count_ok"], fold=r["fold"],
                dr2=r["dperf"], dr2_positive_only=dr2_pos, n_r2_le0=sum(x <= 0 for x in r2),
                flag_incomplete=len(active) < n_runs or len(r2) < len(active), passes=res)


def do_verdict(g1_json):
    """Mode `verdict`: print G1 and G2 and write data/round18/g1g2.json.

    Args:
        g1_json: str, the g1_reads.json written by `read`.
    Returns:
        None.
    """
    g1 = json.load(open(g1_json))
    cat = load_catalogue()
    ctrl = next(c for c in cat if c["role"] == "control" and c["N"] == 1000)
    ca, cr = list(ctrl["active_75k"]), list(ctrl["r2_75k"])
    print(f"control (catalogue {ctrl['cell']}): active {ca} (max {max(ca)}), R2 {np.round(cr, 4).tolist()} "
          f"mean {np.mean(cr):.5f}")
    rescored = [r["r2_75k"] for r in g1["control"] if r["r2_75k"] is not None]
    print(f"  re-scored here at 75k: active {[r['active_75k'] for r in g1['control']]}, R2 "
          f"{np.round(rescored, 4).tolist()}  (check: the catalogue's values)")

    # G1
    print("\n=== G1: 3BFF rate penalty at 75,000 iterations, 36 draws ===")
    a3 = [r for r in load_a3()[1000] if r["family"] == "frm"]
    b3 = load_b3()["runs"]["rate"]
    rows = []
    for lam in LAMBDAS:
        recs = g1["frm"][lam]
        act = [r["active_75k"] for r in recs]
        r2 = [r["r2_75k"] for r in recs if r["r2_75k"] is not None]
        row = value_row(f"lambda_frm = {lam}", "frm", act, r2, ca, cr, len(recs), FRM_CELL.format(lam=lam))
        rows.append(row)
        print(f"lambda {lam}: active {act}, R2 {np.round(r2, 4).tolist()}, fold {row['fold']:.2f}, "
              f"dR2 {row['dr2']:+.4f}, count ok {row['count_ok']}, passes {row['passes']}")
        print(f"   config differs from control: {[r['config_differs'] for r in recs]}")
        if lam == "0.1":
            print(f"   check, a3_N1000.json (9 draws): active {[r['n_active'] for r in a3]}, R2 "
                  f"{np.round([r['r2_full'] for r in a3], 4).tolist()}")
        if lam == "0.03":
            print(f"   check, read_b3.json 'rate' fixed: active {[r['fixed']['active'] for r in b3]}, R2 "
                  f"{np.round([r['fixed']['r2'] for r in b3], 4).tolist()}")

    # G2
    print("\n=== G2: every N = 1,000 value of Supplementary Table 3a, per-network criterion ===")
    g2 = list(rows)
    for c in cat:
        if c["N"] != 1000 or c["role"] not in ("topup", "b2") or c["family"] not in TECHS or c["family"] == "frm":
            continue
        g2.append(value_row(c["cell"].split("/", 1)[1], c["family"], c["active_75k"], c["r2_75k"], ca, cr,
                            c["n_runs"], c["cell"]))
    print(f"{'technique':>10} {'pass@0.005/0.01/0.02':>21} {'min act':>7} {'fold':>5} {'dR2':>8} "
          f"{'dR2(>0)':>8} flags  value")
    for fam in TECHS:
        for r in sorted((r for r in g2 if r["family"] == fam), key=lambda r: r["name"]):
            p = " ".join("P" if r["passes"][f"{t}"] else "-" for t in TOLS)
            flags = ("incomplete " if r["flag_incomplete"] else "") + (f"R2<=0:{r['n_r2_le0']}" if r["n_r2_le0"] else "")
            d2p = "--" if r["dr2_positive_only"] is None else f"{r['dr2_positive_only']:+.4f}"
            print(f"{fam:>10} {p:>21} {r['min_active']:>7} {r['fold']:>5.2f} {r['dr2']:>+8.4f} {d2p:>8} "
                  f"{flags:<12} {r['name']}")
    tech_3bff = {}
    for fam in TECHS:
        mine = [r for r in g2 if r["family"] == fam]
        tech_3bff[fam] = {f"{t}": [r["name"] for r in mine if r["passes"][f"{t}"]] for t in TOLS}
    print("\n3BFF one-value rule (values passing):")
    for fam in TECHS:
        print(f"  {fam:>10}: " + "; ".join(f"{t}: {len(tech_3bff[fam][f'{t}'])} of "
                                           f"{sum(r['family'] == fam for r in g2)}" for t in TOLS))

    # all three tasks
    cddm, _ = F5T.load_cddm()
    maze, maze_tol, _ = F5T.load_maze()
    other = {}
    for task, cells, floor, tol_fix in (("CDDM", cddm, None, None), ("maze", maze, F5T.MAZE_LEARNED_SUCCESS, maze_tol)):
        for fam in TECHS:
            mine = [c for c in cells if c["tech"] == fam and c["status"] == "finished"]
            for t in TOLS:
                tt = tol_fix if tol_fix is not None else t
                other[(task, fam, t)] = any(F5T.rule(c["active"], c["perf"], c["ctrl_active"], c["ctrl_perf"], tt,
                                                     floor)["works"] for c in mine)
    print("\nTechniques passing on each task (3BFF from G2; CDDM and Maze from fig_paper_F5_tasks, Maze at "
          f"tolerance {maze_tol}):")
    allthree = {}
    for t in TOLS:
        allthree[f"{t}"] = []
        for fam in TECHS:
            b = bool(tech_3bff[fam][f"{t}"])
            c_, m_ = other[("CDDM", fam, t)], other[("maze", fam, t)]
            if b and c_ and m_:
                allthree[f"{t}"].append(fam)
            print(f"  tol {t}: {fam:>10}  3BFF {'P' if b else '-'}  CDDM {'P' if c_ else '-'}  Maze {'P' if m_ else '-'}")
        print(f"  tol {t}: on all three tasks: {allthree[f'{t}']}")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(dict(control=dict(cell=ctrl["cell"], active=ca, r2=cr), g1=rows, g2=g2, tech_3bff=tech_3bff,
                   other={f"{k[0]}|{k[1]}|{k[2]}": v for k, v in other.items()}, all_three=allthree,
                   rules="docs/project_trajectory/2026-11-07_0100_round18-rules.md"),
              open(OUT_JSON, "w"), indent=1)
    print(f"-> {OUT_JSON}")


def main():
    """Parse the command line and run a mode."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["read", "verdict"])
    ap.add_argument("path")
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.path)
    else:
        do_verdict(a.path)


if __name__ == "__main__":
    main()
