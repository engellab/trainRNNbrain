#!/usr/bin/env python3
"""Round 19: skipped optimiser steps and post-episode state growth in Maze runs (dead-relu-paper#201).

Entry docs/project_trajectory/2026-11-08_0943_round19-met0.01-frozen-cause.md. Two parts:

  skipped   every *Progress.json under the given roots: the longest run of consecutive skipped optimiser
            steps (the `skipped` field counts, per PPO update, the optimiser steps whose loss was not finite;
            a fully skipped update adds all its n_epochs x n_minibatches steps to the current run, a partly
            skipped update ends it) and the total; prints the records with a run of >= --min-run steps.
  growth    every Maze network (common.maze_load_runs) under the given roots, at its end weights: one training
            rollout of --episodes episodes and its recompute; the largest |state| over running RNN steps and
            over steps after the episode ended, and the number of episodes with |state| > 1e6 in each part.
            Networks with non-finite weights are listed; networks above --max-n units are skipped (CPU).

Usage (repository root):
  python trainRNNbrain/experiments_and_analysis/round19_postepisode.py skipped ROOT [ROOT ...]
  python trainRNNbrain/experiments_and_analysis/round19_postepisode.py growth --out FILE ROOT [ROOT ...]
"""
import argparse
import glob
import json
import os
import sys

import numpy as np
import torch
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(os.path.dirname(HERE))
sys.path.insert(0, REPO)
from trainRNNbrain.experiments_and_analysis.common import maze_load_runs   # noqa: E402
from trainRNNbrain.trainer.TrainerPPO import load_run                      # noqa: E402


def longest_skipped_run(progress, steps_per_update):
    """The longest run of consecutive skipped optimiser steps in one Progress record, and the total.

    Args:
        progress: list of dicts, one per PPO update, each with `update` and optionally `skipped`;
        steps_per_update: int, n_epochs x n_minibatches.
    Returns:
        (longest, start_update, total): ints (start_update None when nothing was skipped).
    """
    best, cur, start, best_start, total = 0, 0, None, None, 0
    for rec in progress:
        k = int(rec.get("skipped", 0) or 0)
        total += k
        if k >= steps_per_update:
            if cur == 0:
                start = rec["update"]
            cur += k
            if cur > best:
                best, best_start = cur, start
        else:
            if cur + k > best:
                best, best_start = cur + k, (start if cur else rec["update"])
            cur = 0
    return best, best_start, total


def scan_skipped(roots, min_run):
    """Print every Progress record under `roots` with a skipped run of at least `min_run` steps.

    Args:
        roots: list of str, folders searched recursively; min_run: int, optimiser steps.
    Returns:
        list of (path, longest, start_update, total, last_update) for every record.
    """
    rows = []
    for root in roots:
        for f in sorted(glob.glob(os.path.join(root, "**", "*Progress.json"), recursive=True)):
            cfgs = sorted(glob.glob(os.path.join(os.path.dirname(f), "*config.yaml")))
            steps = 8
            if cfgs:
                tr = (yaml.safe_load(open(cfgs[0])) or {})
                tr = tr.get("trainer") or tr.get("ppo") or {}
                steps = int(tr.get("n_epochs", 4)) * int(tr.get("n_minibatches", 2))
            p = json.load(open(f))
            best, start, total = longest_skipped_run(p, steps)
            rows.append((f, best, start, total, p[-1]["update"] if p else None))
    print(f"records scanned: {len(rows)}")
    for f, best, start, total, last in rows:
        if best >= min_run:
            print(f"  {f}: longest run {best} steps from update {start}; {total} skipped; record ends at {last}")
    print("largest runs below the bar:", sorted([(r[1], r[0]) for r in rows if r[1] < min_run], reverse=True)[:5])
    return rows


def growth(folder, episodes):
    """Largest |state| in running and in post-episode RNN steps of one training rollout at the end weights.

    Args:
        folder: str, the last run folder of a network's chain; episodes: int, the rollout's batch.
    Returns:
        dict: max_alive, max_after (None if no episode ended), n_alive_1e6, n_after_1e6 (episodes).
    """
    torch.manual_seed(0)
    cfg, task, tr = load_run(folder)
    b = tr.rollout(episodes, tr.T)
    with torch.no_grad():
        st = tr.sequence(b["u"], b["noise"], b["w_seed"], b["outgoing"], b["alive"])[0]
    alive = b["alive"].bool().repeat_interleave(tr.substeps, dim=0)
    a = torch.nan_to_num(st.abs(), nan=float("inf")).amax(dim=0)
    return dict(max_alive=float(a[alive].max()), max_after=float(a[~alive].max()) if (~alive).any() else None,
                n_alive_1e6=int(((a > 1e6) & alive).any(dim=0).sum()),
                n_after_1e6=int(((a > 1e6) & ~alive).any(dim=0).sum()))


def scan_growth(roots, out, episodes, max_n):
    """Run `growth` on every network under `roots`, appending one JSON line per network to `out`.

    Args:
        roots: list of str, Maze trees (folders of <cell>/s<seed>/ runs); out: str, JSON-lines file (networks
        already in it are skipped); episodes: int; max_n: int, larger networks are skipped.
    """
    torch.set_num_threads(8)
    done = {json.loads(line)["folder"] for line in open(out)} if os.path.exists(out) else set()
    for root in roots:
        for r in maze_load_runs(root):
            f = r["folder"]
            if f in done:
                continue
            z = np.load(glob.glob(os.path.join(f, "*Params_*.npz"))[0], allow_pickle=True)
            rec = {"folder": f, "N": int(r["N"]), "last_update": int(r["progress"][-1]["update"])}
            if not np.isfinite(np.asarray(z["W_rec"], float)).all():
                rec["status"] = "non-finite weights"
            elif r["N"] > max_n:
                rec["status"] = f"skipped (N > {max_n})"
            else:
                rec.update(status="ok", episodes=episodes, **growth(f, episodes))
            with open(out, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            print(rec, flush=True)


def main():
    """Parse the part and its arguments and run it."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="part", required=True)
    s = sub.add_parser("skipped")
    s.add_argument("roots", nargs="+")
    s.add_argument("--min-run", type=int, default=100)
    g = sub.add_parser("growth")
    g.add_argument("roots", nargs="+")
    g.add_argument("--out", required=True)
    g.add_argument("--episodes", type=int, default=32)
    g.add_argument("--max-n", type=int, default=2000)
    a = ap.parse_args()
    if a.part == "skipped":
        scan_skipped(a.roots, a.min_run)
    else:
        scan_growth(a.roots, a.out, a.episodes, a.max_n)


if __name__ == "__main__":
    main()
