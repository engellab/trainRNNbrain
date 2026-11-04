## 2026-11-04 13:37 — Correction: two N = 2,000 maze runs died (NaN in the first update); PPO guards at aed03df; resubmitted as 16859389 (issue #3)

Corrects `2026-11-04_1237_rl-maze-suite-submitted-16856486-16856487.md`: array 16856487 does
not deliver all three N = 2,000 seeds.

**What happened.** Tasks 10 and 11 of 16856487 (N = 2,000, seeds 1 and 2) logged kl NaN at
update 1 and "active 0 of 2,000" from then on; every weight was NaN. The surviving seed 3
(task 12) logged kl 0.24 at update 1, far above the 0.02 to 0.06 of the first update at
N = 1,000, then settled (kl about 0.002). Reading: at N = 2,000 the first update moves the policy
a long way; in a later pass of that update the probability ratio exp(log pi_new - log pi_old)
overflowed, the loss became NaN, and the Adam step wrote NaN into the weights.

**Fix (commit aed03df, trainRNNbrain/rl/ppo.py).** The log-ratio is clamped to [-20, 20] before
it is exponentiated, and a minibatch whose loss is not finite is skipped without an optimiser
step (counted in Progress.json as `skipped`). Both guards do nothing while everything is finite,
so the 18 runs in flight at 6d65d7b are the same algorithm as runs at aed03df; the size series is
not confounded. `tests/test_rl_maze.py::test_ppo_update_guards` checks a clamped ratio and a
skipped minibatch.

**Resubmission.** The two dead tasks were cancelled (about 1.5 GPU-h spent). A second detached
worktree `~/trainRNNbrain_rlmaze2` was pinned at aed03df (the first, `~/trainRNNbrain_rlmaze`,
still has 18 jobs running from 6d65d7b and must not move), and
`REPO=$HOME/trainRNNbrain_rlmaze2 sbatch --array=10,11 --time=24:00:00
slurm/rl_maze_suite_della.slurm` = array 16859389. Same output folders
`Maze_suite/N=2000_act=relu/s1` and `s2` (the dead runs' checkpoints are overwritten).
