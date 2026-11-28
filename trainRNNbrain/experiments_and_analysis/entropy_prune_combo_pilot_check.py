#!/usr/bin/env python3
"""The pilot's stop rule of the entropy x prune-and-duplicate test, as a job: exit 0 lets the chain start, 1 stops it.

RULES. docs/project_trajectory/2026-11-25_1245_rules-entropy-prune-combo.md: if an EP pilot runs out of memory, ends
with a non-finite loss or skips more than 5% of its updates, nothing is trained.
The three EP pilots are the 3BFF and CDDM logs (array task 10 of two jobs) and the Maze run folder of task 139.

Usage: python entropy_prune_combo_pilot_check.py FF_LOG CDDM_LOG MAZE_LOG MAZE_PROGRESS_JSON
"""

import json
import math
import re
import sys

MAX_SKIP = 0.05


def supervised(path):
    """Reasons the 3BFF or CDDM pilot of log `path` fails (empty list: passes)."""
    txt = open(path, errors="replace").read()
    done = re.search(r"Executed training in [\d.]+ seconds", txt)
    r2 = re.search(r"torch r2 score: (\S+)", txt)
    skip = re.search(r"gradient spikes: (\d+) updates skipped", txt)
    n = int(skip.group(1)) if skip else 0
    print(f"{path}: done {bool(done)}, r2 {r2 and r2.group(1)}, skipped {n} of 2000")
    out = [] if done else ["did not finish"]
    if not r2 or not math.isfinite(float(r2.group(1))):
        out.append("non-finite score")
    return out + ([f"{n} skipped updates"] if n > MAX_SKIP * 2000 else [])


def maze(log, progress):
    """Reasons the Maze pilot fails (empty list: passes)."""
    txt = open(log, errors="replace").read()
    P = json.load(open(progress))
    bad = sum(1 for p in P if p.get("skipped", 0) or p.get("restored", 0))
    finite = all(math.isfinite(p["discounted_return"]) for p in P)
    print(f"{log}: {len(P)} updates, {bad} with skipped or restored steps, finite {finite}")
    out = [] if "done: 40 updates" in txt and len(P) == 40 else ["did not finish"]
    return out + ([] if finite else ["non-finite return"]) + ([f"{bad} updates skipped or restored"] if bad > MAX_SKIP * 40 else [])


def main():
    """Check the three EP pilots; exit 1 naming every reason one fails."""
    ff, cddm, mlog, mprog = sys.argv[1:5]
    why = [f"3BFF: {x}" for x in supervised(ff)] + [f"CDDM: {x}" for x in supervised(cddm)]
    why += [f"Maze: {x}" for x in maze(mlog, mprog)]
    if why:
        sys.exit("PILOT STOPS THE TEST: " + "; ".join(why))
    print("pilots pass")


if __name__ == "__main__":
    main()
