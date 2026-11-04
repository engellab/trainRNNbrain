"""Do the inactive units of a trained maze agent do any work? Remove units and score the agent.

The maze counterpart of ablate_silent_units.py (claim 1 on the supervised tasks), with the same
removal and the same three orders. One network at a time:
  1. Rebuild the agent from its run folder (trainRNNbrain.rl.saved.load_run).
  2. Order the units by participation p = std(r) + q0.9(|r|) over the clean probe (noise off, mean
     action, PROBE_DROPS fixed drops, every RNN step before the cheese), the probe the training
     logged; active = p >= 0.05 q95(p) (common.active_mask).
  3. Remove a unit by zeroing its OUTGOING weights, its column of W_rec and of W_out, which is the
     unit clamped to zero rate everywhere downstream; incoming weights are left alone, exactly as
     in ablate_silent_units.py.
  4. Score: the success rate (share of episodes that reach the cheese within T decisions) over
     SCORE_DROPS fixed drops with the RNN's training noise ON and the mean action, averaged over
     DRAWS seeded noise draws; the counterpart of the supervised held-out R^2 under the training
     noise. The clean probe success is recorded beside it.
  5. Three orders, each removing nested sets of k units: quietest first, busiest first, and RANDOM
     random orders; k runs from 0 through the number of inactive units to N.
Output: one JSON per run folder in --out (default data/rl_maze_ablation/<cell>__s<seed>.json) with
N, n_active, ks, score_quiet, score_busy, score_random (RANDOM lists), score_full, clean_full,
label, and the run folder; and a table on stdout.
Run: python trainRNNbrain/experiments_and_analysis/rl_maze_ablate.py <run_dir> [--out DIR]
     [--score-drops 256] [--draws 4] [--probe-drops 128] [--random 5]   (GPU if available)
"""
import argparse
import json
import os
import numpy as np
import torch
from common import participation, active_mask
from trainRNNbrain.rl.ppo import rollout
from trainRNNbrain.rl.saved import load_run

REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
OUT_DIR = os.path.join(REPO, "data", "rl_maze_ablation")
SCORE_SEED = 20261104      # the scoring drops; noise draw d is seeded SCORE_SEED + 1 + d


def probe_participation(env, policy, T, n_drops, probe_seed):
    """Participation per unit over the clean probe (the training's probe, same drops).

    Args:
        env: MazeEnv; policy: MazePolicy; T: int, decisions; n_drops: int; probe_seed: int.
    Returns:
        (p, success): p (N,) float; success float, the clean probe's success rate.
    """
    g = torch.Generator(device=env.device)
    g.manual_seed(int(probe_seed))
    out = rollout(env, policy, n_drops, T, clean=True, generator=g)
    k = policy.substeps
    with torch.no_grad():
        states = policy.sequence(out["u"], out["noise"])[0][:, :T * k]
    rates = policy.rnn.get_firing_rates(states).cpu().numpy()
    alive = out["alive"][:-1].bool().repeat_interleave(k, dim=0).cpu().numpy()
    return participation(rates[:, alive]), float(out["success"].float().mean())


def score(env, policy, T, n_drops, draws):
    """Success rate with the training noise on and the mean action, over fixed drops and draws.

    Args:
        env, policy, T, n_drops: as above; draws: int, seeded noise draws.
    Returns:
        float, the mean success rate over the draws.
    """
    s = []
    for d in range(draws):
        g = torch.Generator(device=env.device)
        g.manual_seed(SCORE_SEED + 1 + d)
        out = rollout(env, policy, n_drops, T, clean=False, generator=g)
        s.append(float(out["success"].float().mean()))
    return float(np.mean(s))


def k_grid(N, n_inactive):
    """Nested removal counts: quarters of the inactive set, then steps through the active set.

    Args:
        N: int; n_inactive: int.
    Returns:
        sorted list of distinct ints from 0 to N - 1.
    """
    ks = {0} | {round(n_inactive * f) for f in (0.25, 0.5, 0.75, 1.0)}
    ks |= {round(n_inactive + (N - n_inactive) * f) for f in (0.1, 0.25, 0.5, 0.75, 0.9)}
    return sorted(k for k in ks if 0 <= k < N)


def removed_score(env, policy, T, n_drops, draws, order, ks):
    """Score the agent with the first k units of `order` removed, for every k in ks.

    Args:
        env, policy, T, n_drops, draws: as in `score`; order: (N,) int array, the removal order;
        ks: list of ints.
    Returns:
        list of floats, one score per k. The weights are restored afterwards.
    """
    rnn = policy.rnn
    order = np.array(order, dtype=np.int64)          # a plain copy: a reversed view has a negative stride
    W_rec, W_out = rnn.W_rec.detach().clone(), rnn.W_out.detach().clone()
    scores = []
    with torch.no_grad():
        for k in ks:
            rnn.W_rec.copy_(W_rec)
            rnn.W_out.copy_(W_out)
            idx = torch.as_tensor(order[:k], device=rnn.device, dtype=torch.long)
            rnn.W_rec[:, idx] = 0.0
            rnn.W_out[:, idx] = 0.0
            scores.append(score(env, policy, T, n_drops, draws))
        rnn.W_rec.copy_(W_rec)
        rnn.W_out.copy_(W_out)
    return scores


def ablate(run_dir, out_dir, score_drops, draws, probe_drops, n_random):
    """Run the whole analysis on one run folder and write its JSON.

    Args:
        run_dir: str; out_dir: str; score_drops, draws, probe_drops, n_random: ints.
    Returns:
        the result dict that was written.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, env, policy = load_run(run_dir, device)
    T = int(cfg.ppo.T)
    p, clean_full = probe_participation(env, policy, T, probe_drops, cfg.ppo.probe_seed)
    active = active_mask(p)
    N, n_active = len(p), int(active.sum())
    ks = k_grid(N, N - n_active)
    quiet = np.argsort(p, kind="stable")
    rng = np.random.default_rng(0)
    res = {"run_dir": run_dir, "N": N, "n_active": n_active, "ks": ks, "T": T,
           "score_drops": score_drops, "draws": draws, "clean_full": clean_full,
           "label": f"N={N}_act={cfg.model.activation_args['name']}_s{cfg.seed}"}
    res["score_full"] = score(env, policy, T, score_drops, draws)
    res["score_quiet"] = removed_score(env, policy, T, score_drops, draws, quiet, ks)
    res["score_busy"] = removed_score(env, policy, T, score_drops, draws, quiet[::-1], ks)
    res["score_random"] = [removed_score(env, policy, T, score_drops, draws, rng.permutation(N), ks)
                           for _ in range(n_random)]
    os.makedirs(out_dir, exist_ok=True)
    cell = os.path.basename(os.path.dirname(run_dir.rstrip("/")))
    path = os.path.join(out_dir, f"{cell}__s{cfg.seed}.json")
    with open(path, "w") as fh:
        json.dump(res, fh)
    print(f"{res['label']}: {n_active} of {N} active; full score {res['score_full']:.2f} (clean {clean_full:.2f})")
    print(f"{'k removed':>9} {'quietest':>9} {'busiest':>8} {'random':>7}")
    for i, k in enumerate(ks):
        print(f"{k:>9} {res['score_quiet'][i]:>9.2f} {res['score_busy'][i]:>8.2f} "
              f"{np.mean([r[i] for r in res['score_random']]):>7.2f}")
    print(f"-> {path}")
    return res


def main():
    """Parse arguments and ablate one run."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=OUT_DIR)
    ap.add_argument("--score-drops", type=int, default=256)
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--probe-drops", type=int, default=128)
    ap.add_argument("--random", type=int, default=5)
    args = ap.parse_args()
    ablate(args.run_dir, args.out, args.score_drops, args.draws, args.probe_drops, args.random)


if __name__ == "__main__":
    main()
