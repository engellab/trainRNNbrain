## 2026-11-03 20:07 — Maze task: the first array converged to standing still; three causes; second array 16794476 at 9e87d5c (issue #2)

Follows `2026-11-03_1914_rl-maze-task-ppo-first-jobs.md`.

**Result of array 16790695 (8 runs, all complete, ~5 GPU-h).** Probe success at the end: blind
0.00, 0.05 (N = 500), 0.01, 0.00 (N = 1,000); cue 0.07, 0.08 (N = 500), 0.08, 0.09 (N = 1,000).
Fails the pre-set rule (cue arm 0.8). Figures: `img/internal_figures/rl_maze_first.png` and
`rl_maze_first_trajectories.png` (script `rl_maze_readout.py`). The active counts at the last
probe (264-394 of 500, 406-614 of 1,000) belong to a stationary policy and say nothing about
dormancy on a solved task.

**Cause 1, measured on the checkpoints (N = 500, cue, seed 1, 128 probe drops).** An untrained
agent that stands still scores a discounted return of 1.70; one that moves at random, 1.20. The
shaping was the discounted potential-based form 0.2 (d − 0.99 d'), which pays a stationary agent
0.002 per cell of distance per step, above the 0.01 step cost beyond 5 cells. Training drove the
speed to 0 in 99 of 100 steps and the log-std from −0.7 to −3.4; the clipped speed had no
gradient at 0. Fix: shaping 0.2 (d − d'); speed = v_max sigmoid(output), turn = turn_max
tanh(output) in the env; a floor on the log-std (ppo.log_std_min = −2.3); wall penalty −0.02.

**Cause 2, found by a local falsification run (N = 100, cue, 150 updates of 64 episodes × 200
decisions; rule set before running: the fix works if probe success exceeds 0.3).** With the reward
fix alone, 0.02. The standard RNN (alpha = dt/tau = 0.1) reacts about 10 steps late, 2 cells at
0.2 cells per step. Variants of the same run: a fast network with alpha 0.5 (diagnostic only),
0.30; learning rate ×3 with gradient clip 5, 0.14; both, 0.20; the standard network with the agent
5× slower (v_max 0.04, 1,500 steps, gamma 0.998), 0.09; the standard network with 5 of its own
steps per decision, observation held, action read from the last state (`ppo.substeps = 5`),
**0.41**, passing the rule. The network is unchanged; the recompute check in
`tests/test_rl_maze.py` covers substeps 1 and 3.

**What runs now.** Commit 9e87d5c; Della worktree `~/trainRNNbrain_rlmaze` moved to it after the
first array finished. `cd ~/trainRNNbrain_rlmaze && sbatch slurm/rl_maze_ppo_della.slurm`:
array 16794476, tasks 1-8, the same grid (N 500/1,000 × blind/cue × seeds 1, 2), 1,000 updates
of 256 episodes × 300 decisions (1,500 RNN steps), 6 h cap each, under 50 GPU-h in all. Outputs
overwrite nothing: the same cell folders `Maze_first/N=<N>_cue=<bool>/s<seed>/` on Della are
rewritten by the new runs (the first array's copies are on the Mac under
`data/trained_RNNs_rl_maze/Maze_first/`). Pass rule unchanged: probe success 0.8 in the cue arm
at both sizes. In the local run the active count rose from 69 to 85 of 100 while the agent
learned; whether that holds at N = 500 and 1,000 is what the array measures.
