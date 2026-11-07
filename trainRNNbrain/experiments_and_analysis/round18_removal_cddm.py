#!/usr/bin/env python3
"""Round 18, G3: removing the added units in CDDM networks, the 3BFF A3 procedure (dead-relu-paper#201).

RULE. docs/project_trajectory/2026-11-07_0100_round18-rules.md, G3, committed before this script.

PROCEDURE. a3_remedy_ablation.ablate_net on every run of each group below, the control first (its mean
offline active count n_ctrl gives k = n_active - n_ctrl for the arms). Arm and control are read at the
same iteration, the one where both kept weights:
  paper grid (100k): CDDM_paper_grid/EqType=h_N=1000_arm={duplication, mute} against arm=control;
  rate penalty (75k): CDDM_std_g0_penalties_bias0/..._pen=frm0.1_iters=75000 and
      CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0.2 against the Figure 3a control
      CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0.
      These runs end at 75k (the c75000 cells continue a 50k parent by 25k; the penalty cell is one 75k
      run), and the parents are not on the Mac, so each run's own LastParams file is read.

VERDICT (the rule). drop = intact R^2 - R^2 with the k added units removed (inactive units left in
place). A technique depends on its added units if drop >= DEPENDS in every network, does not depend on
them if drop < INDEPENDENT in every network, otherwise mixed. Printed beside it: the median of k random
active units removed, the Maze-form removal (inactive + k added, r2_keep_ctrl) and its random counterpart
(r2_random_keep).

Output: data/round18/g3_cddm_removal.json and a table on stdout.
Usage:  CUDA_VISIBLE_DEVICES= PYTHONPATH=. python round18_removal_cddm.py [--group NAME]
"""

import argparse
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import a3_remedy_ablation as A3
from common import DATA_DIR

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OUT = os.path.join(REPO, "data", "round18", "g3_cddm_removal.json")
DEPENDS = 0.10          # R^2 drop at or above this in every network: the output depends on the added units
INDEPENDENT = 0.01      # R^2 drop below this in every network (the criterion's tolerance): it does not
GROUPS = {
    "paper_grid_100k": dict(it=100000, control="CDDM_paper_grid/EqType=h_N=1000_arm=control",
                            arms={"prune-and-duplicate": "CDDM_paper_grid/EqType=h_N=1000_arm=duplication",
                                  "read-out dropout": "CDDM_paper_grid/EqType=h_N=1000_arm=mute"}),
    "rate_75k": dict(it=75000, last_params=True, control="CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0",
                     arms={"rate penalty 0.1": "CDDM_std_g0_penalties_bias0/EqType=h_N=1000_pen=frm0.1_iters=75000",
                           "rate penalty 0.2": "CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0.2"}),
}


def runs(cell):
    """Run folders of a cell (diverged 'nan_' folders skipped), sorted.

    Args:
        cell: str relative to DATA_DIR.
    Returns:
        list of str.
    """
    return sorted(d for d in glob.glob(os.path.join(DATA_DIR, cell, "*"))
                  if os.path.isdir(d) and not os.path.basename(d).startswith("nan"))


def params_file(run_dir, g):
    """The parameter file to ablate: the run's own LastParams for a group marked last_params, else None.

    Args:
        run_dir: str; g: a GROUPS entry.
    Returns:
        str or None (None lets ablate_net look the weights up at g["it"]).
    """
    if not g.get("last_params"):
        return None
    return sorted(glob.glob(os.path.join(run_dir, "*LastParams*.np[zy]")))[0]


def verdict(drops):
    """The G3 reading of one technique from its per-network R^2 drops.

    Args:
        drops: list of float.
    Returns:
        str: "depends", "does not depend" or "mixed".
    """
    if all(d >= DEPENDS for d in drops):
        return "depends"
    if all(d < INDEPENDENT for d in drops):
        return "does not depend"
    return "mixed"


def main():
    """Ablate every group (or one) and print the table and verdicts."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--group", choices=list(GROUPS), default=None)
    a = ap.parse_args()
    out = json.load(open(OUT)) if os.path.exists(OUT) else {}
    for gname, g in GROUPS.items():
        if a.group and gname != a.group:
            continue
        ctrl = [A3.ablate_net(d, None, g["it"], params_file(d, g)) for d in runs(g["control"])]
        ctrl = [r for r in ctrl if r is not None]
        n_ctrl = float(np.mean([r["n_active"] for r in ctrl]))
        print(f"[{gname}] control {g['control']} at {g['it']}: active {[r['n_active'] for r in ctrl]}, "
              f"R2 {[round(r['r2_full'], 4) for r in ctrl]}, n_ctrl {n_ctrl:.1f}", flush=True)
        out[gname] = dict(it=g["it"], control=g["control"], control_runs=ctrl, n_ctrl=n_ctrl, arms={})
        for label, cell in g["arms"].items():
            rs = [A3.ablate_net(d, n_ctrl, g["it"], params_file(d, g)) for d in runs(cell)]
            rs = [r for r in rs if r is not None]
            drops = [r["r2_full"] - r["r2_recruited"] for r in rs if "r2_recruited" in r]
            v = verdict(drops) if drops else "no k > 0"
            out[gname]["arms"][label] = dict(cell=cell, runs=rs, drops=drops, verdict=v)
            for r in rs:
                print(f"  {label:>20} active {r['n_active']:4d} k {r.get('surplus')}  intact {r['r2_full']:.4f}  "
                      f"-inactive {r['r2_no_inactive']:.4f}  -k added {r.get('r2_recruited', np.nan):.4f}  "
                      f"-k random active {np.median(r.get('r2_random_active', [np.nan])):.4f}  "
                      f"-(inactive+k) {r.get('r2_keep_ctrl', np.nan):.4f}  "
                      f"-random N-n_ctrl {np.median(r.get('r2_random_keep', [np.nan])):.4f}", flush=True)
            print(f"  {label}: drops {np.round(drops, 4).tolist()} -> {v}", flush=True)
        os.makedirs(os.path.dirname(OUT), exist_ok=True)
        json.dump(out, open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
