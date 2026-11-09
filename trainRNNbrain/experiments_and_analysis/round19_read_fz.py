#!/usr/bin/env python3
"""Round 19: the freeze_after_end=true Maze rerun arms read at updates 2,000 and 4,000.

RULES (docs/project_trajectory/): 2026-11-08_0940_round19-penalty-reruns-rules.md (which arms, the criteria at
2,000), 2026-11-08_1758_round19-penalty-reruns-stage2-rules.md ("one stage, report both points") and, for the
4,000 read, 2026-11-07_1111_round19-settings-stage2-rules.md. Tracking: ptolmachev/dead-relu-paper#201.

WHAT IT COMPUTES. The per-network numbers and the 4,000 verdicts come from round19_read.read_settings, unchanged,
run on the fz tree with the ReLU N = 1,000 suite networks (seeds 1-6) as controls. round19_read quotes its 2,000
verdicts from the pre-fix JSON files, so here the 2,000 verdicts are computed from the same per-network numbers:
  techniques (frm arms), issue #5's criterion against the three uniform-series controls (seeds 1-3): every seed's
      active count above every control's, mean success not more than 0.05 below the controls' mean (one-sided,
      as fig_ed_maze_interventions.rule_verdict), no seed below 0.9;
  settings (met arms), the direction rule against the controls learned at 2,000 (five of the six seeds): learned
      seeds only, all above the range "increases", all below "decreases", otherwise "no change", none learned
      "no direction".
An arm with a seed whose record ends before the read point is "incomplete" at that point, as read_settings does
at 4,000; for a setting with no learned seed the reason is printed beside it.

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/round19_read_fz.py [--fz data/trained_RNNs_rl_maze_fz]
      [--suite data/trained_RNNs_rl_maze/Maze_suite] [--out data/round19/fz_reruns.json]
"""
import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, HERE)
from round19_read import read_settings, SUCCESS_TOL   # noqa: E402

TECHNIQUE_CONTROL_SEEDS = (1, 2, 3)   # the uniform-series controls of issue #5


def verdict_2000(arm, controls):
    """The update-2,000 verdict of one arm by the rules of the rerun entry.

    Args:
        arm: dict, one entry of read_settings(...)["arms"] (cls, seeds with at2000).
        controls: list of dicts, read_settings(...)["controls"] (seed, at2000).
    Returns:
        str, the verdict with its reason.
    """
    seeds = arm["seeds"]
    stopped = [s["seed"] for s in seeds if not s["at2000"]["reached"]]
    if arm["cls"] == "setting":
        ref = [c["at2000"]["active"] for c in controls if c["at2000"]["learned"]]
        lo, hi = min(ref), max(ref)
        acts = [s["at2000"]["active"] for s in seeds if s["at2000"]["learned"]]
        if not acts:
            return "no direction (no seed learned)" + (f"; seeds {stopped} stopped" if stopped else "")
        if stopped:
            return f"incomplete (seeds {stopped} stopped)"
        if all(a > hi for a in acts):
            return f"increases the count ({lo}-{hi})"
        if all(a < lo for a in acts):
            return f"decreases the count ({lo}-{hi})"
        return f"no change in the count ({lo}-{hi})"
    if stopped:
        return f"incomplete (seeds {stopped} stopped)"
    tc = [c["at2000"] for c in controls if c["seed"] in TECHNIQUE_CONTROL_SEEDS]
    bar = max(c["active"] for c in tc)
    ref_succ = float(np.mean([c["success"] for c in tc]))
    acts = [s["at2000"]["active"] for s in seeds]
    succ = [s["at2000"]["success"] for s in seeds]
    reasons = []
    if not all(a > bar for a in acts):
        reasons.append(f"not every seed above every control ({bar})")
    if np.mean(succ) - ref_succ < -SUCCESS_TOL:   # one-sided, as fig_ed_maze_interventions.rule_verdict
        reasons.append(f"mean success {np.mean(succ):.3f} against the controls' {ref_succ:.3f}")
    if min(succ) < 0.9:
        reasons.append(f"a seed ends at {min(succ):.2f}")
    return "works" if not reasons else "fails: " + "; ".join(reasons)


def main():
    """Read the fz arms, print one block per arm and write the JSON."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--fz", default=os.path.join(REPO, "data", "trained_RNNs_rl_maze_fz"))
    ap.add_argument("--suite", default=os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite"))
    ap.add_argument("--out", default=os.path.join(REPO, "data", "round19", "fz_reruns.json"))
    a = ap.parse_args()
    st = read_settings(os.path.join(a.fz, "Maze_interventions"), a.suite)
    print("controls (seed | 2,000 learned, success, active | 4,000 learned, success, active):")
    for c in st["controls"]:
        x, y = c["at2000"], c["at4000"]
        print(f"  s{c['seed']} | {x['learned']}, {x['success']:.3f}, {x['active']} | "
              f"{y['learned']}, {y['success']:.3f}, {y['active']}")
    print("reference at 4,000:", st["reference_4000"])
    for arm in st["arms"]:
        arm["verdict_2000"] = verdict_2000(arm, st["controls"])
        print(f"\n{arm['key']} ({arm['cls']})\n  2,000: {arm['verdict_2000']}\n  4,000: "
              f"{arm.get('verdict_4000', 'not continued')}; {arm.get('label', '')}")
        for s in arm["seeds"]:
            cells = []
            for U in (2000, 4000):
                x = s[f"at{U}"]
                cells.append(f"{U}: " + (f"{'L' if x['learned'] else '-'} {x['success']:.2f} {x['active']} "
                                         f"(learned {x['learned_update']}, {x['active_learned']} there)"
                                         if x["reached"] else f"not reached (record ends at {s['end']})"))
            print(f"    s{s['seed']}: " + " | ".join(cells))
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump(st, open(a.out, "w"), indent=1, default=str)
    print("\nwrote", a.out)


if __name__ == "__main__":
    main()
