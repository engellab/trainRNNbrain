#!/usr/bin/env python3
"""Round 18, G6: CDDM prune-and-duplicate and read-out dropout at N = 500, 1,000, 2,000 and 4,000,
read as the 3BFF size series of Figure 5c is read (dead-relu-paper#201).

RULE. docs/project_trajectory/2026-11-07_0100_round18-rules.md, G6, committed before this script.

NETWORKS. CDDM_paper_grid/EqType=h_N=<N>_arm={control, duplication, mute}, 3 runs each, 100,000
iterations. Each arm's saved config is compared with its same-size control's on
read_remedies.SHARED_FIELDS; a difference is printed and the cell is not judged.

PER RUN (mode `read`, Della CPU node; the N = 4,000 read-out dropout runs are only there):
  read_remedies.read_one: the floor crossing at 1.05x (active there, active at 1.10x and 1.20x, held-out
      R^2 at the crossing, estimated from the final weights since no snapshots were kept), and the
      reason when the run has no crossing;
  fig_paper_F2_fails.read_fixed at 75,000: active units at the probe nearest 75k and held-out R^2 there
      (estimated from the final weights: 1 - (L(75k)/L_end)(1 - R^2_end)), 36 draws.
Writes OUT_DIR/g6_reads.json (and its own R^2 cache there).

MODE `summary` (Mac): per size and technique, the fold change (mean over the same-size control mean),
the R^2 difference and the criterion's count condition (every arm network above every same-size
control), at 1.05x, 1.10x, 1.20x and at 75k; then the rule's direction ("increases at every size" if the
count condition holds at every size at 1.05x and at 75k). Writes data/round18/g6_cddm_sizes.json.

Usage:
  python round18_cddm_sizes.py read OUT_DIR
  python round18_cddm_sizes.py summary G6_JSON
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from common import DATA_DIR, runs_with_loss
from fig_paper_F2_fails import read_fixed
from read_remedies import SHARED_FIELDS, cfg_of, read_one, sel

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT_JSON = os.path.join(REPO, "data", "round18", "g6_cddm_sizes.json")
SIZES = (500, 1000, 2000, 4000)
ARMS = ("control", "duplication", "mute")
ARM_NAME = {"duplication": "prune-and-duplicate", "mute": "read-out dropout"}
CELL = "CDDM_paper_grid/EqType=h_N={N}_arm={arm}"
FIXED_ITER = 75000
POINTS = (("active", "r2", "1.05x"), ("active_d10", None, "1.10x"), ("active_d20", None, "1.20x"),
          ("active_75k", "r2_75k", "75k"))


def do_read(out_dir):
    """Mode `read`: every run of every size and arm at its floor crossing and at 75k.

    Args:
        out_dir: str, output folder (created).
    Returns:
        None; writes g6_reads.json.
    """
    os.makedirs(out_dir, exist_ok=True)
    cache_path = os.path.join(out_dir, "r2cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}
    out = {}
    for N in SIZES:
        ctrl_dirs = sorted(glob.glob(os.path.join(DATA_DIR, CELL.format(N=N, arm="control"), "*")))
        c0 = cfg_of(ctrl_dirs[0])
        ctrl_shared = {k: sel(c0, k) for k in SHARED_FIELDS}
        for arm in ARMS:
            cell = CELL.format(N=N, arm=arm)
            recs = []
            for r in runs_with_loss(os.path.join(DATA_DIR, cell)):
                d = os.path.join(DATA_DIR, cell, r["name"])
                rec = read_one(d, cache)
                fx = read_fixed(dict(r, run_dir=d), FIXED_ITER, cache)
                rec.update(active_75k=int(fx["active"]), r2_75k=float(fx["r2"]), r2_75k_how=fx["r2_how"],
                           probe_75k=fx["probe"])
                c = cfg_of(d)
                rec["config_differs"] = {k: [str(sel(c, k)), str(ctrl_shared[k])] for k in SHARED_FIELDS
                                         if sel(c, k) != ctrl_shared[k]}
                recs.append(rec)
                print(f"N={N} {arm:>11} {r['name'][:30]} floor {rec.get('t_star')} active {rec.get('active')} "
                      f"r2 {rec.get('r2')} reason {rec['reason']} | 75k active {rec['active_75k']} r2 "
                      f"{rec['r2_75k']:.4f} ({fx['r2_how']}) | differs {rec['config_differs']}", flush=True)
            out[f"{N}|{arm}"] = recs
            json.dump(cache, open(cache_path, "w"))
            json.dump(out, open(os.path.join(out_dir, "g6_reads.json"), "w"), indent=1)


def compare(arm_recs, ctrl_recs, akey, rkey):
    """Fold change, R^2 difference and the count condition at one read-out point.

    Args:
        arm_recs, ctrl_recs: lists of run dicts; akey: the active-count key; rkey: the R^2 key or None.
    Returns:
        dict: n_arm, n_ctrl (runs read at this point), active, ctrl_active, fold, dr2 (None without
        R^2), count_ok (bool or None when a side has no read).
    """
    a = [r[akey] for r in arm_recs if r.get(akey) is not None]
    c = [r[akey] for r in ctrl_recs if r.get(akey) is not None]
    out = dict(n_arm=len(a), n_ctrl=len(c), active=a, ctrl_active=c, fold=None, dr2=None, count_ok=None)
    if a and c:
        out["fold"] = float(np.mean(a) / np.mean(c))
        out["count_ok"] = bool(min(a) > max(c))
    if rkey and a and c:
        ra = [r[rkey] for r in arm_recs if r.get(akey) is not None and r.get(rkey) is not None]
        rc = [r[rkey] for r in ctrl_recs if r.get(akey) is not None and r.get(rkey) is not None]
        if ra and rc:
            out["dr2"] = float(np.mean(ra) - np.mean(rc))
    return out


def do_summary(path):
    """Mode `summary`: print the per-size table and the rule's direction; write data/round18/g6_cddm_sizes.json.

    Args:
        path: str, the g6_reads.json written by `read`.
    Returns:
        None.
    """
    d = json.load(open(path))
    res = {}
    for arm in ("duplication", "mute"):
        print(f"\n=== {ARM_NAME[arm]} ===")
        for N in SIZES:
            arm_recs, ctrl_recs = d[f"{N}|{arm}"], d[f"{N}|control"]
            differs = [r["config_differs"] for r in arm_recs if r["config_differs"]]
            row = {"differs": differs, "no_crossing": [r["reason"] for r in arm_recs + ctrl_recs if r["reason"]]}
            for akey, rkey, lab in POINTS:
                row[lab] = compare(arm_recs, ctrl_recs, akey, rkey)
            res[f"{N}|{arm}"] = row
            print(f"N={N}: config differs {differs or 'no'}; runs without a crossing: {row['no_crossing'] or 'none'}")
            for _, _, lab in POINTS:
                c = row[lab]
                fold = "--" if c["fold"] is None else f"{c['fold']:.2f}"
                dr2 = "--" if c["dr2"] is None else f"{c['dr2']:+.4f}"
                print(f"   {lab:>5}: arm {c['active']} control {c['ctrl_active']} fold {fold} dR2 {dr2} "
                      f"count ok {c['count_ok']}")
        judged = [res[f"{N}|{arm}"] for N in SIZES if not res[f"{N}|{arm}"]["differs"]]
        every = all(r["1.05x"]["count_ok"] and r["75k"]["count_ok"] for r in judged)
        res[f"direction|{arm}"] = dict(sizes_judged=[N for N in SIZES if not res[f"{N}|{arm}"]["differs"]],
                                       increases_at_every_size=bool(every))
        print(f"-> {ARM_NAME[arm]}: count condition at 1.05x and at 75k at every judged size: {every}")
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    json.dump(dict(reads=path, rows=res, rules="docs/project_trajectory/2026-11-07_0100_round18-rules.md"),
              open(OUT_JSON, "w"), indent=1)
    print(f"-> {OUT_JSON}")


def main():
    """Parse the command line and run a mode."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["read", "summary"])
    ap.add_argument("path")
    a = ap.parse_args()
    if a.mode == "read":
        do_read(a.path)
    else:
        do_summary(a.path)


if __name__ == "__main__":
    main()
