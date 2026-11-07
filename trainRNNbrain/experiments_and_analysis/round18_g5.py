#!/usr/bin/env python3
"""Round 18, G5: the Maze form of the removal in 3BFF networks (dead-relu-paper#201).

RULE. docs/project_trajectory/2026-11-07_0100_round18-rules.md, G5. Reads the re-run of
a3_remedy_ablation.py at N = 1,000 (Della job 17029916, which adds r2_random_keep: N - n_ctrl units
removed at random from all N, 5 sets) and the stored data/remedy_reads_31oct/a3_N1000.json.
  check  r2_full, r2_recruited and r2_keep_ctrl reproduce the stored values within 0.002 per network;
  rule   per technique, the Maze statement ("removing as many randomly chosen units lowers it further")
         holds if r2_random_keep (median of 5) < r2_keep_ctrl in every network.
Writes data/round18/g5_3bff_maze_form.json.
Usage:  python round18_g5.py NEW_A3_JSON
"""

import json
import os
import sys

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
OLD = os.path.join(REPO, "data", "remedy_reads_31oct", "a3_N1000.json")
OUT = os.path.join(REPO, "data", "round18", "g5_3bff_maze_form.json")
TOL = 0.002


def key(r):
    """A network's identity across the two files: its run folder name."""
    return os.path.basename(r["run_dir"].rstrip("/"))


def main(new_path):
    """Check the re-run against the stored file and print the G5 table and reading.

    Args:
        new_path: str, the re-run's a3_N1000.json.
    Returns:
        None.
    """
    new, old = json.load(open(new_path)), {key(r): r for r in json.load(open(OLD))}
    bad = []
    for r in new:
        o = old.get(key(r))
        if o is None:
            bad.append((key(r), "not in the stored file"))
            continue
        for k in ("r2_full", "r2_recruited", "r2_keep_ctrl"):
            if k in r and abs(r[k] - o[k]) > TOL:
                bad.append((key(r), k, r[k], o[k]))
    print(f"check against the stored a3_N1000.json: {len(new)} networks, {len(bad)} differences above {TOL}")
    for b in bad:
        print("   ", b)
    rows, fams = [], {}
    for r in new:
        if "r2_random_keep" not in r:
            continue
        w = dict(family=r["family"], net=key(r)[:20], n_active=r["n_active"], k=r["surplus"], intact=r["r2_full"],
                 added=r["r2_recruited"], keep=r["r2_keep_ctrl"], random_keep=float(np.median(r["r2_random_keep"])),
                 random_active=float(np.median(r["r2_random_active"])))
        w["random_lower"] = w["random_keep"] < w["keep"]
        rows.append(w)
        fams.setdefault(r["family"], []).append(w)
    print(f"{'family':>10} {'net':>20} {'act':>4} {'k':>4} {'intact':>7} {'-k add':>7} {'-k rnd':>7} {'-inact+k':>8} {'-rnd N-n':>8}")
    for w in rows:
        print(f"{w['family']:>10} {w['net']:>20} {w['n_active']:>4} {w['k']:>4} {w['intact']:7.3f} {w['added']:7.3f} "
              f"{w['random_active']:7.3f} {w['keep']:8.3f} {w['random_keep']:8.3f}")
    summ = {}
    for f, ws in fams.items():
        summ[f] = dict(n=len(ws), median_keep=float(np.median([w["keep"] for w in ws])),
                       median_random_keep=float(np.median([w["random_keep"] for w in ws])),
                       random_lower_every=all(w["random_lower"] for w in ws),
                       random_lower_n=sum(w["random_lower"] for w in ws))
        print(f"  {f}: median inactive+k removed {summ[f]['median_keep']:.3f}, as many random {summ[f]['median_random_keep']:.3f};"
              f" random lower in {summ[f]['random_lower_n']} of {len(ws)} -> holds: {summ[f]['random_lower_every']}")
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(dict(check_differences=bad, rows=rows, families=summ, source=new_path), open(OUT, "w"), indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main(sys.argv[1])
