"""Read-out of the maze-PPO runs (trainRNNbrain/rl): does the agent learn, and how many units stay
active?

Question. For every run under a root folder (default data/trained_RNNs_rl_maze/Maze_first): the
probe success rate and the active-unit count against PPO updates, and the active count at the last
probe against N. Arms (walls-only; cue: bearing to the cheese added) are line styles, N is the
project's fixed colour map (plotstyle.col_n), seeds are separate thin lines.

Method. Each run folder holds Progress.json (per update) and Participation.npz (per probe), written
by run_maze_ppo.py; nothing is recomputed here. The probe is the clean read-out (noise-free, mean
action, fixed drops), so its success rate is the performance measure and its active count the
participation measure; the training batch's success is shown faint for comparison.

Two read-out points per run (pre-registered on engellab/trainRNNbrain issue #3): "learned", the
first probe at which the probe success stays at or above LEARNED_SUCCESS for LEARNED_HOLD probes
in a row, and "end", the last probe. Runs with other activations than ReLU are drawn in their own
figure, with the modulation count (std alone, saved as active_mod) beside the p-rule count, since
tanh and sigmoid resting rates clear the p bar on their own.

Output. img/internal_figures/rl_maze_<name>.png (ReLU curves and endpoints),
rl_maze_<name>_activations.png (when several activations are present),
rl_maze_<name>_trajectories.png (the runs' own trajectory pictures tiled), and a table on stdout:
one row per run with the final probe success, the final training success, the mean steps to the
cheese in the last 50 updates, the update at which the task was learned, and the active counts at
"learned" and at the end.
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
ARM_STYLE = {False: ("--", "walls-only"), True: ("-", "cue")}
LEARNED_SUCCESS = 0.9      # the "learned" read-out: probe success at or above this ...
LEARNED_HOLD = 5           # ... for this many probes in a row (50 updates at probe_every 10)
ACT_COLORS = {"relu": "#1a1a1a", "tanh": "#d62728", "sigmoid": "#1f77b4", "softplus": "#2ca02c",
              "leaky_relu": "#9467bd"}


def learned_index(probe_success, hold=LEARNED_HOLD, level=LEARNED_SUCCESS):
    """Index of the first probe from which the success stays at or above `level` for `hold` probes.

    Args:
        probe_success: (n_probes,) array; hold: int; level: float.
    Returns:
        int index into the probe arrays, or None if the task was never learned by that rule.
    """
    ok = np.asarray(probe_success) >= level
    for i in range(len(ok) - hold + 1):
        if ok[i:i + hold].all():
            return i
    return None


def load_runs(root):
    """Every finished run under root.

    Args:
        root: str, folder holding <cell>/s<seed>/ run folders (any depth-2 layout).
    Returns:
        list of dicts: N (int), act (str, the activation name), cue (bool), seed (int), folder
        (str), progress (list of dicts), probes (dict of arrays: update, active, active_mod if
        saved, probe_success, p_update, p), sorted by N, act, cue, seed.
    """
    runs = []
    for cfg_path in sorted(glob.glob(os.path.join(root, "*", "s*", "config.yaml"))):
        folder = os.path.dirname(cfg_path)
        with open(cfg_path) as fh:
            cfg = yaml.safe_load(fh)
        with open(os.path.join(folder, "Progress.json")) as fh:
            progress = json.load(fh)
        probes = dict(np.load(os.path.join(folder, "Participation.npz")))
        runs.append({"N": int(cfg["model"]["N"]), "act": str(cfg["model"]["activation_args"]["name"]),
                     "cue": bool(cfg["env"]["goal_cue"]), "seed": int(cfg["seed"]),
                     "folder": folder, "progress": progress, "probes": probes})
    return sorted(runs, key=lambda r: (r["N"], r["act"], r["cue"], r["seed"]))


def summarise(runs):
    """One row per run, printed as a table.

    Args:
        runs: list from load_runs.
    Returns:
        list of dicts with N, act, arm, seed, updates, probe_success (last probe), train_success
        (mean of the last 50 updates), steps (same window), learned (update of the "learned"
        probe, or None), active_learned (count there, or None), active (last probe), active_min
        (over probes), and active_mod (modulation count at the last probe, or None if not saved).
    """
    rows = []
    for r in runs:
        tail = r["progress"][-50:]
        pr = r["probes"]
        li = learned_index(pr["probe_success"])
        rows.append({"N": r["N"], "act": r["act"], "arm": ARM_STYLE[r["cue"]][1], "seed": r["seed"],
                     "updates": r["progress"][-1]["update"],
                     "probe_success": float(pr["probe_success"][-1]),
                     "train_success": float(np.mean([p["success"] for p in tail])),
                     "steps": float(np.mean([p["steps"] for p in tail])),
                     "learned": None if li is None else int(pr["update"][li]),
                     "active_learned": None if li is None else int(pr["active"][li]),
                     "active": int(pr["active"][-1]),
                     "active_min": int(pr["active"].min()),
                     "active_mod": int(pr["active_mod"][-1]) if "active_mod" in pr else None})
    print(f"{'N':>5} {'act':>11} {'arm':>10} {'seed':>4} {'updates':>7} {'probe succ':>10} {'train succ':>10} "
          f"{'steps':>6} {'learned':>8} {'act@learn':>9} {'act@end':>7} {'min':>5} {'mod@end':>7}")
    fmt = lambda v: "-" if v is None else str(v)
    for w in rows:
        print(f"{w['N']:>5} {w['act']:>11} {w['arm']:>10} {w['seed']:>4} {w['updates']:>7} {w['probe_success']:>10.2f} "
              f"{w['train_success']:>10.2f} {w['steps']:>6.0f} {fmt(w['learned']):>8} {fmt(w['active_learned']):>9} "
              f"{w['active']:>7} {w['active_min']:>5} {fmt(w['active_mod']):>7}")
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
                            Line2D([], [], marker="o", color="k", mfc="white", ls="none", label="walls-only")],
                   frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def draw_activations(runs, path):
    """Active count against updates per activation (p rule solid, modulation count dotted), and
    the counts at the end per activation; one size. PNG at path.

    Args:
        runs: list from load_runs (all sizes; only the most common N is drawn); path: str.
    """
    N = max({r["N"] for r in runs}, key=lambda n: sum(r["N"] == n for r in runs))
    sel = [r for r in runs if r["N"] == N]
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6))
    acts = sorted({r["act"] for r in sel}, key=lambda a: list(ACT_COLORS).index(a) if a in ACT_COLORS else 99)
    for r in sel:
        c = ACT_COLORS.get(r["act"], "#7f7f7f")
        axes[0].plot(r["probes"]["update"], r["probes"]["probe_success"], color=c, lw=1.0)
        axes[1].plot(r["probes"]["update"], r["probes"]["active"], color=c, lw=1.0)
        if "active_mod" in r["probes"]:
            axes[1].plot(r["probes"]["update"], r["probes"]["active_mod"], color=c, lw=1.0, ls=":")
        x = acts.index(r["act"])
        axes[2].plot(x, r["probes"]["active"][-1], "o", color=c, mec="k", mew=0.4)
        if "active_mod" in r["probes"]:
            axes[2].plot(x, r["probes"]["active_mod"][-1], "o", color=c, mfc="white", mec=c)
    axes[0].set_ylabel("probe success")
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylabel(f"active units of {N} (solid: p rule; dotted: modulation)")
    axes[1].set_ylim(0, N)
    for ax in axes[:2]:
        ax.set_xlabel("PPO update")
    axes[2].set_xticks(range(len(acts)))
    axes[2].set_xticklabels(acts, rotation=20)
    axes[2].set_ylabel(f"active units of {N} at the end (filled: p rule; open: modulation)")
    axes[2].set_ylim(0, N)
    axes[0].legend(handles=[Line2D([], [], color=ACT_COLORS.get(a, "#7f7f7f"), lw=1.5, label=a) for a in acts],
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
    relu = [r for r in runs if r["act"] == "relu"]
    draw_curves(relu or runs, [w for w in rows if w["act"] == "relu"] or rows,
                os.path.join(IMG_DIR, f"rl_maze_{args.name}.png"))
    if len({r["act"] for r in runs}) > 1:
        draw_activations(runs, os.path.join(IMG_DIR, f"rl_maze_{args.name}_activations.png"))
    draw_trajectories(runs, os.path.join(IMG_DIR, f"rl_maze_{args.name}_trajectories.png"))
    print(f"figures: {IMG_DIR}/rl_maze_{args.name}.png, {IMG_DIR}/rl_maze_{args.name}_trajectories.png")


if __name__ == "__main__":
    main()
