#!/usr/bin/env python3
"""Round 19: removing the added units of the freeze_after_end=true Maze technique networks (G4's procedure).

WHY. Fig. 5d's Maze sub-panel drew round 18's G4 removal (entry 2026-11-07_0235), made on rate-penalty
networks whose penalty averaged over post-episode steps. Those networks' verdicts are withdrawn (entries
2026-11-08_0940 and _0943); the reruns with trainer.freeze_after_end=true replace them (entries 2026-11-09_0136
and _1221). This script summarises the same removal made on the rerun networks at update 2,000: prune-and-
duplicate s1-s3 and the rate penalty at 0.03, 0.1 and 0.3, s1-s3 (the stage-1 folders of
data/trained_RNNs_rl_maze_fz/Maze_interventions, which hold the update-2,000 weights).

PROCEDURE (unchanged from G4). Each network was scored by
  rl_maze_ablate.py <run_dir> --added 430 --out data/round19/fz_maze_added
k = active units minus 430, the mean of the three uniform-series controls (decision #194); the k added units are
the network's k least participating active units. Scores are success under the training noise, 256 drops x 4
noise draws: intact; the k added units removed with the inactive units left in place; k random active units
(median of 5 sets). The reading per technique is G4's: "depends" if every drop (intact minus k-added score) is at
least 0.10, "does not depend" if every drop is below 0.05, otherwise "mixed".

OUTPUT. data/round19/fz_maze_added_summary.json (rows in G4's format, plus `learned`, whether the network is
learned at update 2,000 by common.maze_learned_index, taken from data/round19/fz_reruns.json) and the table on
stdout.

Usage (repository root):  PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/round19_fz_removal.py
"""
import glob
import json
import os

import numpy as np

REPO = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
IN_DIR = os.path.join(REPO, "data", "round19", "fz_maze_added")
FZ_JSON = os.path.join(REPO, "data", "round19", "fz_reruns.json")
OUT = os.path.join(REPO, "data", "round19", "fz_maze_added_summary.json")
DEPENDS, INDEPENDENT = 0.10, 0.05      # G4's bars on the drop


def learned_at_2000():
    """Whether each fz rerun network is learned at update 2,000.

    Returns:
        dict (arm key, seed) -> bool, from data/round19/fz_reruns.json (common.maze_learned_index on the
        record cut at update 2,000).
    """
    with open(FZ_JSON) as fh:
        d = json.load(fh)
    return {(a["key"], s["seed"]): bool(s["at2000"]["learned"]) for a in d["arms"] for s in a["seeds"]}


def reading(drops):
    """G4's reading of a list of drops: 'depends', 'does not depend' or 'mixed'."""
    if drops and all(x >= DEPENDS for x in drops):
        return "depends"
    if drops and all(x < INDEPENDENT for x in drops):
        return "does not depend"
    return "mixed"


def main():
    """Summarise every removal file, print the table and the per-technique readings, write the summary."""
    learned = learned_at_2000()
    rows = []
    for path in sorted(glob.glob(os.path.join(IN_DIR, "*__s*.json"))):
        r = json.load(open(path))
        cell = os.path.basename(path).split("__")[0]
        seed = int(os.path.basename(path).split("__s")[1].split(".")[0])
        rows.append(dict(cell=cell, file=os.path.basename(path), seed=seed, n_active=r["n_active"], k=r["k"],
                         intact=r["score_full"], no_inactive=r["score_no_inactive"], added=r["score_added"],
                         random_active=float(np.median(r["score_random_active"])),
                         keep=r["score_keep_ctrl"], random_keep=float(np.median(r["score_random_keep"])),
                         n_keep=r["n_remove_keep"], drop=r["score_full"] - r["score_added"],
                         learned=learned[(cell, seed)], run_dir=os.path.relpath(r["run_dir"], REPO)
                         if os.path.isabs(r["run_dir"]) else r["run_dir"]))
    print(f"{'network':>16} {'act':>5} {'k':>4} {'learned':>7} {'intact':>6} {'-inact':>6} {'-k add':>6} "
          f"{'-k rnd':>6} {'drop':>6}")
    for w in rows:
        print(f"{w['file'][:-5]:>16} {w['n_active']:>5} {w['k']:>4} {str(w['learned']):>7} {w['intact']:6.3f} "
              f"{w['no_inactive']:6.3f} {w['added']:6.3f} {w['random_active']:6.3f} {w['drop']:6.3f}")
    tech = {}
    for cell in sorted({w["cell"] for w in rows}):
        mine = [w for w in rows if w["cell"] == cell]
        drops = [w["drop"] for w in mine]
        lower = sum(w["random_active"] < w["added"] for w in mine)
        tech[cell] = dict(drops=drops, verdict=reading(drops), random_lower_in=f"{lower} of {len(mine)}")
        print(f"  {cell}: drops {np.round(drops, 3).tolist()} -> {reading(drops)}; k random active lower than "
              f"k added in {lower} of {len(mine)}")
    rate = [w["drop"] for w in rows if w["cell"].startswith("frm")]
    print(f"  rate penalty, all values pooled: {reading(rate)}")
    print(f"  drops of at least {DEPENDS}: {sum(w['drop'] >= DEPENDS for w in rows)} of {len(rows)}")
    with open(OUT, "w") as fh:
        json.dump(dict(rows=rows, techniques=tech, rate_pooled=reading(rate), ctrl_mean=430), fh, indent=1)
    print(f"-> {OUT}")


if __name__ == "__main__":
    main()
