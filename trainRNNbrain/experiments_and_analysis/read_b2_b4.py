#!/usr/bin/env python3
"""Read B2 (rate penalty alone, flip-flop size series) and B4 (plasticity transfer) at the floor.

Both rules were written before any result. B2's is the plan of 13 Oct (agent-bplan §B2, deleted in
eec4a16; quoted in the notebook entry that reports this read). B4's is in
docs/project_trajectory/2026-10-15_1342_open-rules-and-commands-moved-from-agent-notes.md.

B2. Every run of `NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=frm0.1` and of Figure 1's flip-flop
controls at the same N is read at its own noisy-loss floor (heldout_r2.read_rung, held-out R^2 at
R2_DRAWS draws). Per N: active counts, multiplier (mean arm / mean control), R^2 gap, runs read.

B4. Every run of `NBitFlipFlopHyper_b4_transfer/EqType=h_N=<N>_<cell>`, read at its own floor
(common.read_run). S0 = units inactive at the first participation probe; A* = units active at the
first probe at or after t*. Per run: |S0|, revived |S0 & A*|, newly inactive |A0 & ~A*|, active at
t*, |S0 & A_end|, held-out R^2 at t* (heldout_r2.floor_r2). Then the pre-registered gate and bars.

The held-out R^2 cache is this script's own file, not data/heldout_r2_cache.json, which other agents
write concurrently.

Usage:  python trainRNNbrain/experiments_and_analysis/read_b2_b4.py OUT_DIR
        (repository root, CPU node; writes OUT_DIR/read_b2_b4.json and prints the tables)
"""

import argparse
import glob
import json
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import heldout_r2 as H
from common import DATA_DIR, active_mask, read_run, runs_with_loss

SIZES = (500, 1000, 2000, 4000)
B2_ARM = f"{DATA_DIR}/NBitFlipFlop_remedy_grid/EqType=h_N={{N}}_arm=frm0.1"
# Figure 1's flip-flop controls (fig_paper_F1.FF200 and its N = 4000 cell)
B2_CTRL = {N: f"{DATA_DIR}/NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N={N}_pen=none_arm=none"
           for N in (500, 1000, 2000)}
B2_CTRL[4000] = f"{DATA_DIR}/NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control"
B4_ROOT = f"{DATA_DIR}/NBitFlipFlopHyper_b4_transfer"
B4_CELLS = ("src=control_age=old", "src=control_age=young", "src=fresh")


def b2_read(cache):
    """B2 per N: arm and control read at their floors, and the pre-registered checks.

    Args:
        cache: dict, an open held-out R^2 cache (filled in place).
    Returns:
        dict N -> {arm, ctrl: read_rung dicts as lists, mult, dr2, n_read}.
    """
    out = {}
    for N in SIZES:
        arm, ctrl = H.read_rung(B2_ARM.format(N=N), cache=cache), H.read_rung(B2_CTRL[N], cache=cache)
        mult = arm["floor"].mean() / ctrl["floor"].mean() if len(arm["floor"]) and len(ctrl["floor"]) else np.nan
        dr2 = arm["r2"].mean() - ctrl["r2"].mean() if len(arm["r2"]) and len(ctrl["r2"]) else np.nan
        out[N] = dict(arm={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in arm.items()},
                      ctrl={k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in ctrl.items()},
                      mult=float(mult), dr2=float(dr2), n_read=len(arm["floor"]))
    return out


def b4_run(run_dir, cache):
    """One B4 run read at its floor: revival of initially inactive units, counts and R^2.

    Args:
        run_dir: str, a run folder holding *ParticipationTrace.pkl and *TrainLosses.json.
        cache: dict, an open held-out R^2 cache.
    Returns:
        dict with name, reason (None when read), t_star, n_S0, revived, newly_inactive, active0,
        active_star, active_end, S0_active_end, r2, r2_measured; counts are unit numbers out of N.
    """
    rd = read_run(run_dir)
    with open(glob.glob(os.path.join(run_dir, "*ParticipationTrace.pkl"))[0], "rb") as fh:
        tr = pickle.load(fh)
    P = np.asarray(tr["participation"])
    it_a = np.asarray(tr["participation_iters"], float)
    A0, Aend = active_mask(P[0]), active_mask(P[-1])
    S0 = ~A0
    rec = dict(name=rd["name"], reason=rd["reason"], it0=float(it_a[0]), active0=int(A0.sum()),
               n_S0=int(S0.sum()), active_end=int(Aend.sum()), S0_active_end=int((S0 & Aend).sum()))
    if rd["reason"] is not None:
        return rec
    j = np.flatnonzero(it_a >= rd["t_star"])
    Astar = active_mask(P[j[0]]) if len(j) else Aend
    r2, measured = H.floor_r2(rd, cache)
    rec.update(t_star=float(rd["t_star"]), active_star=int(Astar.sum()), revived=int((S0 & Astar).sum()),
               newly_inactive=int((A0 & ~Astar).sum()), r2=float(r2), r2_measured=bool(measured))
    return rec


def b4_verdict(cells):
    """Apply B4's pre-registered gate and bars to the N = 1000 cells.

    Args:
        cells: dict cell name -> list of b4_run dicts.
    Returns:
        dict with gate (bool), the numbers it uses, and verdict: "do not come back", "can be
        recruited", "between the bars", or "no verdict" (fewer than 2 of 3 old or fresh read).
    """
    old = [r for r in cells.get("src=control_age=old", []) if r["reason"] is None]
    fresh = [r for r in cells.get("src=fresh", []) if r["reason"] is None]
    if len(old) < 2 or len(fresh) < 2:
        return dict(verdict="no verdict", n_old=len(old), n_fresh=len(fresh))
    f_mean = np.mean([r["active_star"] for r in fresh])
    f_sd = np.std([r["active_star"] for r in fresh], ddof=1)
    old_at_transfer = np.mean([r["active0"] for r in old])
    old_star = np.mean([r["active_star"] for r in old])
    share = [r["revived"] / r["n_S0"] for r in old]
    gate = bool(f_mean - old_at_transfer > 3 * f_sd)
    if all(s < 0.10 for s in share) and f_mean - old_star > 3 * f_sd:
        v = "do not come back"
    elif sum(s >= 0.30 for s in share) >= 2 or abs(f_mean - old_star) <= f_sd:
        v = "can be recruited"
    else:
        v = "between the bars"
    return dict(verdict=v, gate=gate, fresh_mean=float(f_mean), fresh_sd=float(f_sd),
                old_at_transfer=float(old_at_transfer), old_star=float(old_star),
                revived_share=[float(s) for s in share], n_old=len(old), n_fresh=len(fresh))


def main():
    """Read B2 and B4, print the tables, write OUT_DIR/read_b2_b4.json."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("out_dir", help="folder for read_b2_b4.json and the R^2 cache")
    a = ap.parse_args()
    cache_path = os.path.join(a.out_dir, "read_b2_b4_r2_cache.json")
    cache = json.load(open(cache_path)) if os.path.exists(cache_path) else {}

    b2 = b2_read(cache)
    print("B2  N   arm active at floor      control              mult   dR2     arm dropped")
    for N, v in b2.items():
        print(f"    {N:<5d} {str([int(x) for x in v['arm']['floor']]):<20s} "
              f"{str([int(x) for x in v['ctrl']['floor']]):<20s} {v['mult']:.2f}  {v['dr2']:+.4f} "
              f"{v['arm']['dropped']}")
        print(f"          arm R2 {np.round(v['arm']['r2'], 4).tolist()}  ctrl R2 {np.round(v['ctrl']['r2'], 4).tolist()}")

    b4 = {}
    for d in sorted(glob.glob(f"{B4_ROOT}/EqType=h_N=*_*")):
        cell = os.path.basename(d).split("_", 2)[2]
        N = int(os.path.basename(d).split("_")[1].split("=")[1])
        b4[f"{N}_{cell}"] = [b4_run(r["run_dir"], cache) for r in runs_with_loss(d)]
    print("\nB4  cell                           run         t*      act0  |S0|  revived  newly_inact  act*  S0&Aend  R2")
    for k, rs in b4.items():
        for r in rs:
            if r["reason"]:
                print(f"    {k:<30s} {r['name'][:10]} EXCLUDED: {r['reason']}")
                continue
            print(f"    {k:<30s} {r['name'][:10]} {r['t_star']:>7.0f} {r['active0']:>5d} {r['n_S0']:>5d} "
                  f"{r['revived']:>7d} {r['newly_inactive']:>11d} {r['active_star']:>5d} {r['S0_active_end']:>7d}"
                  f"  {r['r2']:.4f}{'' if r['r2_measured'] else ' (est)'}")
    verdict = b4_verdict({k.split("_", 1)[1]: v for k, v in b4.items() if k.startswith("1000_")})
    print("\nB4 N = 1000 verdict:", json.dumps(verdict))

    json.dump(cache, open(cache_path, "w"))
    json.dump(dict(b2={str(k): v for k, v in b2.items()}, b4=b4, b4_verdict_1000=verdict),
              open(os.path.join(a.out_dir, "read_b2_b4.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
