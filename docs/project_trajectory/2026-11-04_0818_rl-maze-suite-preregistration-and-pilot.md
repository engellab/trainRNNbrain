## 2026-11-04 08:18 — Maze suite: sensors only on a 15 x 15 maze; pre-registration; pilot array 16838504 at 2697878 (issue #3)

Follows `2026-11-03_2149_rl-maze-second-array-result-units-go-dormant.md`. Pavel, 4 Nov: use the
8 egocentric sensors only, make the maze slightly harder, then run a suite like the 3-bit
flip-flop and CDDM ones to see whether the same problem persists. The pre-registration is on
engellab/trainRNNbrain issue #3; this entry records it and what ran.

**Task.** Inputs: the 8 egocentric range sensors, nothing about the cheese (`env.goal_cue: false`).
Maze: 15 x 15 from `trainRNNbrain/rl/make_maze.py --cells 7 --seed 6 --loops 4` (depth-first
search on a 7 x 7 lattice, 4 walls removed for loops), cheese at the centre in a pocket entered
only from above; 36 cells from the far end, 22 on average over free cells, 9 dead ends (the old
maze: 24 and no dead ends). Episodes 500 decisions of 5 RNN steps; everything else as in the
second array, with one change: a move into a wall slides along it (the free component of the move
is kept, x first then y) and the wall cost of 0.02 applies only when the agent is stuck. Measured
reason: with the move cancelled instead, an untrained agent in this maze stood wedged in 77 to 93
of 100 steps (15 to 20 in the old maze) and a local run learned only not to bump; with sliding it
is stuck in 15 of 100 steps.

**Suite (20 runs, about 110 to 130 GPU-h).** N = 250, 500, 1,000, 2,000, ReLU, 3 seeds each;
tanh, shifted sigmoid, softplus (beta 25), leaky ReLU at N = 1,000, 2 seeds each (the supervised
series' model configs); 2,000 updates of 256 episodes; launcher `slurm/rl_maze_suite_della.slurm`,
two groups (N = 2,000 with a 24 h cap). N = 4,000 left out (about 50 h per run).

**Read-out, fixed before the runs.** Every 10 updates a clean probe (noise off, mean action, 128
fixed drops): probe success, the active count under the project's rule (p = std + q0.9 at or
above 5% of q95) and, new, under the modulation criterion (std alone), which is the count read
for tanh and shifted sigmoid, whose resting rate clears the p bar on its own. Two points per run:
"learned" = the first probe at which probe success stays at or above 0.9 for 5 probes, and
"end" = update 2,000. "The problem persists" means: (1) at the end the active count is below N by
more than the seed spread at every size that solves the task; (2) the active share falls from
N = 250 to 2,000 by more than the seed spread, monotonically; (3) the count at the end is below the
count at "learned" in most runs; (4) tanh and sigmoid lose as many units as ReLU or more. Each has
its falsifier stated on the issue. A size that does not solve the task (success below 0.9) is
reported and excluded from 1 to 3.

**Local falsification before submission (N = 100, 64 episodes, 300 updates; rule: probe success
above 0.15): failed for every variant** (move cancelled 0.00 to 0.02; wall cost 0: 0.00 to 0.02;
sliding with and without wall cost: 0.02; discounted return pinned at the step cost). On the old
maze the sensors-only runs needed about 115,000 episodes at N = 1,000, six times the local budget,
so the local scale is probably underpowered rather than the maze unlearnable; a local control on
the old maze at the same scale is running as this is written and its result goes on the issue.

**Pilot.** Review on issue #3 (independent agent): changes requested, both applied in 2697878.
Della worktree `~/trainRNNbrain_rlmaze` at 2697878 (clean). `sbatch --array=4,7 --time=12:00:00
slurm/rl_maze_suite_della.slurm`: array 16838504, tasks 4 (N = 500, seed 1) and 7 (N = 1,000,
seed 1), the suite's own runs, about 11 GPU-h. Rule, set now: if the N = 1,000 run's probe
success exceeds 0.5 by update 1,000, the remaining 18 runs are submitted from the same commit;
otherwise the task is changed (a curriculum of drop distances is the first candidate) and
re-reviewed before anything else runs. Outputs
`/home/pt1290/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/N=<N>_act=<act>/s<seed>/`;
logs `~/trainRNNbrain_logs/MazeSuite.16838504_<i>.out`.
