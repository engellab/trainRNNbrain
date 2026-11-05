#!/usr/bin/env python3
"""Population statistics of the maze agents over all units and over the active units only.

QUESTION. The maze counterpart of population_stats.py (dead-relu-paper #34, #56): does counting the
inactive units change the population statistics an analysis computes over a maze agent? The same
four statistics as the rule entry 2026-10-31_2212 names (S1 the participation-ratio dimensionality
of the rate covariance, S2 the same on standardised rates, S3 the selective and mixed-selective
units at partial R^2 >= 0.10, S4 the k-means functional classes), the same threshold and the same
decision rule, computed over all units and over the active units. Figure 1e draws S3a, the fraction
of units selective for a task variable, for the maze from this file (fig_paper_F1.py, #182).

NETWORKS. Every run under --root (default data/trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu,
the three seeds of the uniform series), at its final weights: no earlier snapshot of a maze run is
saved, and the ablation of Figure 1d uses the same weights. Each entry records whether the run is
learned (common.maze_learned_index on its saved probes) and the probe's success rate; the figure
draws every entry and the caption says which runs are learned.

RESPONSES. The clean probe of training (TrainerPPO.probe: probe_episodes fixed drops, the noise
off, the mean action, T decisions of `substeps` RNN steps each). The rates of every RNN step before
the cheese feed S1 and S2, as the training's participation is taken over those steps. For S3 and S4
one sample is one decision of one episode before the cheese: each unit's rate averaged over that
decision's RNN steps, regressed on five task variables fixed here before any value was computed:
the turn command and the speed command executed at that decision (turn_max tanh(a_0) and
v_max sigmoid(a_1) of the policy means, TaskMaze.step), the geodesic distance from the agent's cell
to the cheese in cells (TaskMaze.dist), and the agent's heading as its cosine and sine. They are the
maze's counterpart of the flip-flop's three bits, CDDM's context, coherences and choice, and DMTS's
sample, test and match: the two outputs of the task and the agent's position in the maze.

Usage:
  python rl_maze_population_stats.py [--root DIR] [--out FILE] [--device cpu|cuda]
Output: data/population_stats_maze.json, {"results": [...]} with one entry per run holding the
dict of population_stats.population_statistics plus label, task "Maze", seed (int), iteration (the
last probe's update), learned (bool), probe_success, n_active_probe, variables, n_samples and
batch_shape; and the per-network table printed. The file is read beside data/population_stats.json
and never merged into it: fig_ed_population.py sorts that file by its three supervised tasks.
"""
import argparse
import glob
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
# population_stats (scikit-learn) before torch: the other order loads two OpenMP runtimes on the
# Mac and its k-means hangs or aborts (population_stats.py says the same at its imports)
from population_stats import population_statistics, _jsonable                # noqa: E402
import torch                                                                 # noqa: E402
from common import participation, active_mask, maze_learned_index           # noqa: E402
from trainRNNbrain.trainer.TrainerPPO import load_run                        # noqa: E402

REPO = os.path.dirname(os.path.dirname(HERE))
ROOT = os.path.join(REPO, "data", "trained_RNNs_rl_maze", "Maze_suite", "N=1000_act=relu")
OUT = os.path.join(REPO, "data", "population_stats_maze.json")
VARIABLES = ["turn", "speed", "distance to cheese", "heading cos", "heading sin"]


def probe_rates(task, trainer):
    """Play the training's clean probe and return the rates with the task variables of every decision.

    Args:
        task: TaskMaze; trainer: TrainerPPO holding the trained RNN (load_run).
    Returns:
        dict: rates (N, M) float32, every RNN step before the cheese, in probe order; Y (N, S)
        float32, each unit's rate averaged over the RNN steps of each decision before the cheese;
        X (S, 5) float64, the task variables of those decisions in the order of VARIABLES;
        success, the probe's success rate; p (N,), participation over the M steps.
    """
    g = torch.Generator(device=task.device)
    g.manual_seed(int(trainer.probe_seed))
    T, k, B = trainer.T, trainer.substeps, trainer.probe_episodes
    out = trainer.rollout(B, T, clean=True, generator=g)
    with torch.no_grad():
        states = trainer.sequence(out["u"], out["noise"])[0][:, :T * k]        # (N, T k, B)
        rates = trainer.RNN.get_firing_rates(states)
        alive = out["alive"][:-1].bool()                                        # (T, B)
        step_alive = alive.repeat_interleave(k, dim=0)                          # (T k, B)
        N = rates.shape[0]
        per_decision = rates.reshape(N, T, k, B).mean(dim=2)                    # (N, T, B)
        a = out["actions"]                                                      # (2, T, B)
        turn = torch.tanh(a[0]) * task.turn_max
        speed = torch.sigmoid(a[1]) * task.v_max
        xy = out["xy"][:-1]                                                     # (T, B, 2), before the decision
        row, col = task.cell(xy.reshape(-1, 2))
        dist = task.dist[row, col].reshape(T, B)
        heading = out["heading"][:-1]                                           # (T, B)
        X = torch.stack([turn, speed, dist, heading.cos(), heading.sin()], dim=-1)   # (T, B, 5)
        flat = rates[:, step_alive].cpu().numpy()
    return dict(rates=flat, Y=per_decision[:, alive].cpu().numpy(),
                X=X[alive].cpu().numpy().astype(np.float64),
                success=float(out["success"].float().mean()), p=participation(flat))


def measure(folder, device):
    """All statistics for one maze run at its final weights.

    Args:
        folder: str, the run folder; device: torch.device.
    Returns:
        dict of population_statistics plus the run's label, task, seed, iteration, learned flag,
        probe success, active count and the regressors' names.
    """
    cfg, task, trainer = load_run(folder, device)
    sim = probe_rates(task, trainer)
    act = active_mask(sim["p"])
    N = sim["rates"].shape[0]
    st = population_statistics(sim["rates"].reshape(N, 1, -1), sim["Y"], sim["X"], act)
    probes = np.load(sorted(glob.glob(os.path.join(folder, "*Participation.npz")))[0])
    learned = maze_learned_index(np.asarray(probes["probe_success"]))
    st.update(label=f"Maze_N={N}_act={cfg.model.activation_args['name']}_s{cfg.seed}", task="Maze",
              seed=int(cfg.seed), iteration=int(probes["update"][-1]), learned=learned is not None,
              learned_update=None if learned is None else int(probes["update"][learned]),
              probe_success=sim["success"], n_active_probe=int(act.sum()), variables=VARIABLES,
              n_samples=int(sim["Y"].shape[1]), batch_shape=[N, int(sim["rates"].shape[1])],
              run_dir=os.path.relpath(folder, REPO))
    return st


def table(results):
    """Print one row per network: the active count, S1 and S2 over all and active units, S3a and S4."""
    print(f"{'seed':>4} {'learned':>7} {'success':>7} {'active':>6} {'S1 all/act':>12} "
          f"{'S2 all/act':>12} {'selective all/act':>18} {'n sel all/act/inact':>20} {'k all/act':>9}")
    for s in results:
        print(f"{s['seed']:>4} {str(s['learned']):>7} {s['probe_success']:>7.2f} {s['n_active']:>6} "
              f"{s['S1']['all']:>5.2f}/{s['S1']['act']:<6.2f} {s['S2']['all']:>5.1f}/{s['S2']['act']:<6.1f} "
              f"{s['S3a']['all']:>8.3f}/{s['S3a']['act']:<9.3f} "
              f"{s['S3a']['n_all']:>6}/{s['S3a']['n_act']}/{s['S3a']['n_inactive']:<6} "
              f"{s['S4']['k_all']:>4}/{s['S4']['k_act']}")


def main():
    """Score every run under --root and write --out."""
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=ROOT, help="folder holding the run folders s<seed>")
    ap.add_argument("--out", default=OUT)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = ap.parse_args()
    folders = sorted(d for d in glob.glob(os.path.join(a.root, "s*")) if os.path.isdir(d))
    if not folders:
        raise SystemExit(f"no run folders under {a.root}")
    results = []
    for folder in folders:
        print(f"== {folder}", flush=True)
        results.append(measure(folder, torch.device(a.device)))
        s = results[-1]
        print(f"   {s['n_active']} of {s['N']} active; {s['n_samples']} decisions; selective "
              f"{s['S3a']['n_all']} of {s['N']} over all units, {s['S3a']['n_act']} of {s['n_active']} "
              f"over the active units", flush=True)
    table(results)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as fh:
        json.dump({"results": _jsonable(results), "variables": VARIABLES}, fh, indent=1)
    print(f"-> {a.out}")


if __name__ == "__main__":
    main()
