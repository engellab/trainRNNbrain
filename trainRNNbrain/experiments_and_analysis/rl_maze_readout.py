"""Read-out of the maze-PPO runs (trainRNNbrain/rl): does the agent learn, and how many units stay
active?

Question. For every run under a root folder (default data/trained_RNNs_rl_maze/Maze_first): the
probe success rate and the active-unit count against PPO updates, and the active count at the last
probe against N. Arms (blind: walls only; cue: bearing to the cheese) are line styles, N is the
project's fixed colour map (plotstyle.col_n), seeds are separate thin lines.

Method. Each run folder holds Progress.json (per update) and Participation.npz (per probe), written
by run_maze_ppo.py; nothing is recomputed here. The probe is the clean read-out (noise-free, mean
action, fixed drops), so its success rate is the performance measure and its active count the
participation measure; the training batch's success is shown faint for comparison.

Output. img/internal_figures/rl_maze_<name>.png (curves and endpoints) and
img/internal_figures/rl_maze_<name>_trajectories.png (the runs' own trajectory pictures tiled), and
a table on stdout: one row per run with the final probe success, the final training success, the
mean steps to the cheese in the last 50 updates, and the active count at the last probe.
Run: python trainRNNbrain/experiments_and_analysis/rl_maze_readout.py [--root DIR] [--name first]
"""
import argparse
import glob
import json
import os
import numpy as np
import yaml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from common import IMG_DIR
from plotstyle import col_n

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
ARM_STYLE = {False: ("--", "blind"), True: ("-", "cue")}


def load_runs(root):
    """Every finished run under root.

    Args:
        root: str, folder holding <cell>/s<seed>/ run folders (any depth-2 layout).
    Returns:
        list of dicts: N (int), cue (bool), seed (int), folder (str), progress (list of dicts),
        probes (dict of arrays: update, active, probe_success, p_update, p), sorted by N, cue, seed.
    """
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "*", "s*", "config.yaml"))):
        folder = os.path.dirname(cfg_path)
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        with open(os.path.join(folder, "Progress.json")) as fh:
            progress = json.load(fh)
        probes = dict(np.load(os.path.join(folder, "Participation.npz")))
        runs.append({"N": int(cfg["model"]["N"]), "cue": bool(cfg["env"]["goal_cue"]), "seed": int(cfg["seed"]),
                     "folder": folder, "progress": progress, "probes": probes})
    return sorted(runs, key=lambda r: (r["N"], r["cue"], r["seed"]))


def summarise(runs):
    """One row per run, printed as a table.

    Args:
        runs: list from load_runs.
    Returns:
        list of dicts with N, arm, seed, updates, probe_success (last probe), train_success (mean of
        the last 50 updates), steps (same window), active (last probe), active_min (over probes).
    """
    rows = []
    for r in runs:
        tail = r["progress"][-50:]
        rows.append({"N": r["N"], "arm": ARM_STYLE[r["cue"]][1], "seed": r["seed"],
                     "updates": r["progress"][-1]["update"],
                     "probe_success": float(r["probes"]["probe_success"][-1]),
                     "train_success": float(np.mean([p["success"] for p in tail])),
                     "steps": float(np.mean([p["steps"] for p in tail])),
                     "active": int(r["probes"]["active"][-1]),
                     "active_min": int(r["probes"]["active"].min())})
    print(f"{'N':>5} {'arm':>5} {'seed':>4} {'updates':>7} {'probe succ':>10} {'train succ':>10} {'steps':>6} {'active':>6} {'min':>5}")
    for w in rows:
        print(f"{w['N']:>5} {w['arm']:>5} {w['seed']:>4} {w['updates']:>7} {w['probe_success']:>10.2f} "
              f"{w['train_success']:>10.2f} {w['steps']:>6.0f} {w['active']:>6} {w['active_min']:>5}")
    return rows


def draw_curves(runs, rows, path):
    """Probe success and active count against updates, and the final active count against N.

    Args:
        runs: list from load_runs; rows: list from summarise; path: str, PNG to write.
    """
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    for r in runs:
        ls, _ = ARM_STYLE[r["cue"]]
        c = col_n(r["N"])
        it = [p["update"] for p in r["progress"]]
        axes[0].plot(it, [p["success"] for p in r["progress"]], color=c, ls=ls, lw=0.4, alpha=0.3)
        axes[0].plot(r["probes"]["update"], r["probes"]["probe_success"], color=c, ls=ls, lw=1.0)
        axes[1].plot(r["probes"]["update"], r["probes"]["active"], color=c, ls=ls, lw=1.0)
    for w in rows:
        axes[2].plot(w["N"], w["active"], marker="o", color=col_n(w["N"]),
                     mfc=col_n(w["N"]) if w["arm"] == "cue" else "white", mec=col_n(w["N"]), ls="none")
    axes[0].set_ylabel("success rate (probe; faint: training batch)")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel("active units at the probe")
    axes[1].set_ylim(bottom=0)
    for ax in axes[:2]:
        ax.set_xlabel("PPO update")
    axes[2].set_xlabel("N")
    axes[2].set_ylabel("active units at the last probe")
    axes[2].set_xscale("log")
    axes[2].set_ylim(bottom=0)
    sizes = sorted({r["N"] for r in runs})
    handles = [Line2D([], [], color=col_n(n), lw=1.5, label=f"N = {n}") for n in sizes]
    handles += [Line2D([], [], color="k", ls=ARM_STYLE[a][0], lw=1.0, label=ARM_STYLE[a][1]) for a in (False, True)]
    axes[1].legend(handles=handles, frameon=False, fontsize=8)
    axes[2].legend(handles=[Line2D([], [], marker="o", color="k", mfc="k", ls="none", label="cue"),
                            Line2D([], [], marker="o", color="k", mfc="white", ls="none", label="blind")],
                   frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_trajectories(runs, path):
    """Tile the runs' own trajectories.png (written at their last checkpoint), one panel per run.

    Args:
        runs: list from load_runs; path: str, PNG to write.
    """
    n = len(runs)
    cols = min(4, n)
    rows_ = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows_, cols, figsize=(3.2 * cols, 3.4 * rows_), squeeze=False)
    for ax in axes.flat:
        ax.axis("off")
    for ax, r in zip(axes.flat, runs):
        ax.imshow(plt.imread(os.path.join(r["folder"], "trajectories.png")))
        ax.set_title(f"N = {r['N']}, {ARM_STYLE[r['cue']][1]}, seed {r['seed']}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def main():
    """Parse arguments, load the runs, print the table, write the two figures."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", default=os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_first"))
    ap.add_argument("--name", default="first", help="suffix of the output figures")
    args = ap.parse_args()
    runs = load_runs(args.root)
    if not runs:
        raise SystemExit(f"no runs under {args.root}")
    rows = summarise(runs)
    os.makedirs(IMG_DIR, exist_ok=True)
    draw_curves(runs, rows, os.path.join(IMG_DIR, f"rl_maze_{args.name}.png"))
    draw_trajectories(runs, os.path.join(IMG_DIR, f"rl_maze_{args.name}_trajectories.png"))
    print(f"figures: {IMG_DIR}/rl_maze_{args.name}.png, {IMG_DIR}/rl_maze_{args.name}_trajectories.png")


if __name__ == "__main__":
    main()
