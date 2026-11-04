## 2026-11-03 21:49 — Maze task, array 16794613: the task is learnable, and 100-390 units per network stay inactive (issue #2)

Follows `2026-11-03_2007_rl-maze-first-array-fails-substeps-second-array.md` and the correction
`2026-11-03_20xx_rl-maze-second-array-resubmitted-16794613.md` (same day, 16794613 at 9e5d8d1).

**What ran.** Della array 16794613 from `~/trainRNNbrain_rlmaze` at 9e5d8d1: N in {500, 1,000} ×
arms {blind, cue} × seeds {1, 2}; 1,000 PPO updates of 256 episodes × 300 decisions, 5 RNN steps
per decision (configs/rl/maze_ppo.yaml). 1.5 h per run, about 12.5 GPU-h; with the failed first
array about 18 GPU-h in all. Runs: `/home/pt1290/trainRNNbrain/data/trained_RNNs_rl_maze/
Maze_second/N=<N>_cue=<bool>/s<seed>/`, copied to the Mac under `data/trained_RNNs_rl_maze/`.
Read-out: `rl_maze_readout.py --root data/trained_RNNs_rl_maze/Maze_second --name second`;
figures `img/internal_figures/rl_maze_second.png`, `rl_maze_second_trajectories.png`.

**Result.** Probe = noise-free rollout, mean action, 128 fixed drops; active = participation at
least 5% of the 95th percentile over the probe's steps before the cheese (common.active_mask).

| N | arm | seed | probe success | steps to the cheese | active, last probe | active, first probe / minimum |
|---|---|---|---|---|---|---|
| 500 | blind | 1 | 0.22 | 272 | 200 | 271 / 182 |
| 500 | blind | 2 | 0.76 | 209 | 386 | 275 / 267 |
| 500 | cue | 1 | 1.00 | 151 | 388 | 323 / 281 |
| 500 | cue | 2 | 1.00 | 101 | 418 | 326 / 313 |
| 1,000 | blind | 1 | 1.00 | 140 | 620 | 473 / 410 |
| 1,000 | blind | 2 | 1.00 | 135 | 613 | 516 / 398 |
| 1,000 | cue | 1 | 0.99 | 136 | 618 | 605 / 473 |
| 1,000 | cue | 2 | 0.96 | 117 | 694 | 620 / 518 |

The pass rule set before the first array (probe success 0.8 in the cue arm at both sizes) is met:
0.96 to 1.00, with the training batch above 0.8 from update 99 to 170. The blind arm is solved at
N = 1,000 by both seeds (0.8 from update 428 and 496) and partly at N = 500 (0.76, still rising;
0.22 for the seed that loops in the outer corridor).

**Dormant units on a solved task.** In the cue arm 82 to 112 of 500 and 306 to 387 of 1,000 units
are inactive at the last probe; the active share is about 0.8 at N = 500 and about 0.65 at
N = 1,000, the direction the supervised tasks show, on two sizes and two seeds.

**The count's course differs from supervised training.** Every run dips in its first 10 updates
(605 to 473 at N = 1,000, cue, seed 1), rises while the agent learns (670 and 746 at update 200 for
the two N = 1,000 cue seeds), and drifts down slowly once the task is learned (618 and 694 at
update 1,000). The run that did not learn stayed near 200 of 500. Interpretation, not a measured
mechanism: participation follows learning here, and the slow decline over long training that the
supervised tasks show is only hinted at within 1,000 updates. The exploration log-std ended at
-1.4 to -2.3 (the floor is -2.3), so the policies were still stochastic.

**Open for a next package.** A longer budget (3,000 to 5,000 updates, about 5 to 8 h per run) to
see whether the count keeps falling after the task is learned; N = 2,000 and more seeds; a read-out
rule for this task that replaces the loss floor (for example a fixed number of updates after the
probe success first stays above 0.95); more updates for the blind arm at N = 500.
