#!/usr/bin/env python3
"""The entropy term's Maze verdict against default networks matched to it (trainer version, continuation).

RULES. docs/project_trajectory/2026-11-25_*_rules-maze-matched-controls.md, committed before any run.

WHAT IT READS. Every run is joined along its warm-start chain (common.maze_load_runs) and read at
entropy_setpoint_read.MAZE_END (update 2,000 at N = 1,000; 4,000 at N = 2,000) by
entropy_setpoint_read.maze_point, unchanged:
  original controls   <suite>/N=<N>_act=relu (joined with _cont), seeds 1-3
  matched controls    <matched<N>>/control (joined with control_cont), seeds 1-3
  set-point arms      <arms<N>>/<arm>, the arm derived from the saved config
  other techniques    <fz>/<arm>, N = 1,000 only (the corrected-trainer reruns), for the description

WHAT IT PRINTS. Per arm and size, fig_paper_F5_tasks.rule (tolerance 0.05, floor 0.9) against the original
and the matched controls side by side, every finished network counted; the binding verdict of the rules
entry (ent0.1 at both sizes); the matched controls' learned updates, guard counts and update-1 rollout.
Without --matched1000/--matched2000 the original controls alone are used (dry run).

Usage (repository root, Mac):
  python trainRNNbrain/experiments_and_analysis/maze_matched_controls_read.py OUT_JSON --suite DIR
      --arms1000 DIR --arms2000 DIR --fz DIR [--matched1000 DIR --matched2000 DIR] [--fig PATH_STEM]
"""
import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np

import fig_paper_F5_tasks as F5T
from common import maze_load_runs
from entropy_setpoint_read import MAZE_END, MAZE_FLOOR, MAZE_SIZES, MAZE_TOL, maze_arm_of, maze_point

SEEDS = (1, 2, 3)
BINDING = "ent0.1"
TECHNIQUES = ("duplicate", "frm0.03", "frm0.1", "frm0.3", "frm0.1_rws0.05", "dropout0.05_beta4",
              "dropout0.2_beta4", "rescale", "rescale_mild", "rescale_few", "rescale_tgt4")


def points(root, N, cell=None):
    """Read every run of one tree at MAZE_END[N], grouped by the folder of its first stage.

    Args:
        root: str, folder holding <cell>/s<seed>; N: int, the size kept; cell: str or None, keep this
            folder only.
    Returns:
        dict folder -> list of maze_point dicts sorted by seed (seeds 1-3), each with update1 (the first
        update's discounted return, success and steps, measured before any optimizer step).
    """
    out = {}
    for r in maze_load_runs(root):
        name = os.path.basename(os.path.dirname(r["chain"][0]))
        if int(r["N"]) != N or int(r["seed"]) not in SEEDS or (cell is not None and name != cell):
            continue
        x = maze_point(r, MAZE_END[N])
        x["update1"] = [r["progress"][0].get(k) for k in ("discounted_return", "success", "steps")]
        out.setdefault(name, []).append(x)
    return {k: sorted(v, key=lambda x: x["seed"]) for k, v in out.items()}


def verdict(rows, ctrl):
    """The M09 Maze rule on the finished networks of one arm against the finished networks of one control set.

    Args:
        rows, ctrl: lists of maze_point dicts.
    Returns:
        fig_paper_F5_tasks.rule dict plus n_arm and n_ctrl (finished networks on each side); None if either
        side has no finished network.
    """
    a, c = [x for x in rows if x["complete"]], [x for x in ctrl if x["complete"]]
    if not a or not c:
        return None
    v = F5T.rule([x["active_end"] for x in a], [x["success_end"] for x in a], [x["active_end"] for x in c],
                 [x["success_end"] for x in c], MAZE_TOL, floor=MAZE_FLOOR)
    return dict(v, n_arm=len(a), n_ctrl=len(c))


def show(name, rows):
    """Print one line per network of a group.

    Args:
        name: str; rows: list of maze_point dicts.
    Returns:
        None.
    """
    for x in rows:
        print(f"    {name} s{x['seed']} last {x['last_update']} active {x['active_end']} success {x['success_end']} "
              f"learned {x['learned_update']} (active {x['active_learned']}) skipped {x['skipped_sum']:g} restored "
              f"{x['restored_sum']:g} update1 {x['update1']}", flush=True)


def draw(R, stem):
    """Active units against updates: original controls, matched controls and the binding arm, one panel per size.

    Args:
        R: dict N -> dict with original, matched, arms; stem: str, figure path without extension.
    Returns:
        None; writes stem.pdf and stem.png.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, N in zip(axes, MAZE_SIZES):
        for rows, col, ls, lab in ((R[N]["original"], "0.65", "--", "default, first trainer (original controls)"),
                                   (R[N]["matched"], "0.15", "-", "default, corrected trainer (matched controls)"),
                                   (R[N]["arms"][BINDING], "#2a78d6", "-", "entropy set point, λ = 0.1")):
            for k, x in enumerate(rows):
                ax.plot(x["trace_u"], x["trace_active"], color=col, ls=ls, lw=1.4, label=lab if k == 0 else None)
        ax.set_xlim(0, MAZE_END[N])
        ax.set_ylim(0, N)
        ax.set_title(f"Maze, N = {N:,}", fontsize=10)
        ax.set_xlabel("PPO update")
        ax.set_ylabel(f"active units (of {N:,})")
        ax.grid(alpha=0.25, lw=0.5)
        ax.legend(fontsize=8, frameon=False, loc="upper right")
    fig.savefig(stem + ".pdf")
    fig.savefig(stem + ".png", dpi=150)
    print(f"figure: {stem}.pdf, .png")


def main():
    """Read both control sets and every arm, print the verdicts side by side, write the JSON and the figure."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("path")
    ap.add_argument("--suite", required=True)
    for N in MAZE_SIZES:
        ap.add_argument(f"--arms{N}", required=True)
        ap.add_argument(f"--matched{N}", default=None)
    ap.add_argument("--fz", required=True)
    ap.add_argument("--fig", default=None)
    a = ap.parse_args()
    R, binding = {}, {}
    for N in MAZE_SIZES:
        orig = points(a.suite, N, f"N={N}_act=relu")[f"N={N}_act=relu"]
        root = getattr(a, f"matched{N}")
        matched = points(root, N, "control").get("control", []) if root else []
        if any(maze_arm_of(x["folder"]) != "control" for x in matched):
            raise ValueError(f"a matched control at N = {N} has a set-point term on")
        arms = points(getattr(a, f"arms{N}"), N)
        for name, rows in arms.items():
            cfg_arms = [maze_arm_of(x["folder"]) for x in rows]
            if any(c != name for c in cfg_arms):
                raise ValueError(f"N = {N}: folder {name} holds arms {cfg_arms}")
        techs = {k: v for k, v in (points(a.fz, N) if N == 1000 else {}).items() if k in TECHNIQUES}
        missing = set(TECHNIQUES) - set(techs) if N == 1000 else set()
        if missing:
            raise ValueError(f"techniques missing from --fz: {sorted(missing)}")
        R[N] = dict(original=orig, matched=matched, arms=arms, techniques=techs, verdicts={})
        print(f"\nN = {N:,}, update {MAZE_END[N]:,}")
        for lab, ctrl in (("original", orig), ("matched", matched)):
            fin = [x for x in ctrl if x["complete"]]
            if fin:
                suc = float(np.mean([x["success_end"] for x in fin]))
                print(f"  {lab} controls: {len(fin)} of 3 finished; active {[x['active_end'] for x in fin]} (largest "
                      f"{max(x['active_end'] for x in fin)}); success {[round(x['success_end'], 3) for x in fin]} "
                      f"(mean {suc:.3f}, bar {suc - MAZE_TOL:.3f})")
            show(lab, ctrl)
        for name, rows in list(arms.items()) + list(techs.items()):
            vs = {lab: verdict(rows, ctrl) for lab, ctrl in (("original", orig), ("matched", matched))}
            R[N]["verdicts"][name] = vs
            words = {lab: "none" if v is None else f"{v['text']} (fold {v['fold']:.2f}, dsuccess {v['dperf']:+.3f})"
                     for lab, v in vs.items()}
            changed = (vs["original"] and vs["matched"] and vs["original"]["works"] != vs["matched"]["works"])
            n_fin = sum(x["complete"] for x in rows)
            print(f"  {name}: {n_fin} of {len(rows)} finished, active {[x['active_end'] for x in rows]}, success "
                  f"{[x['success_end'] if x['success_end'] is None else round(x['success_end'], 3) for x in rows]}"
                  f"\n      original: {words['original']}\n      matched:  {words['matched']}"
                  f"{'   <-- VERDICT CHANGES' if changed else ''}")
        show(BINDING, arms[BINDING])
        binding[N] = R[N]["verdicts"][BINDING]["matched"]
    if all(v is not None for v in binding.values()):
        ok = {N: bool(v["works"]) for N, v in binding.items()}
        full = all(v["n_ctrl"] == 3 and v["n_arm"] == 3 for v in binding.values())
        print(f"\nbinding: {BINDING} against the matched controls {ok} -> "
              f"{('UPHELD' if full else 'UPHELD on the finished controls only') if all(ok.values()) else 'OVERTURNED'}")
        if a.fig:
            draw(R, a.fig)
    json.dump(R, open(a.path, "w"), indent=1)


if __name__ == "__main__":
    main()
