# Rules, written before submission: the six Maze networks continued from update 12,000 to 20,000

Follows the result entry of 14 Nov (`*_maze-long-training-result.md`): at update 12,000 the return is flat in two
of six networks, so the flat-return test has no verdict. Pavel (14 Nov): continue the training; the return cannot
grow indefinitely.

## What is known before these runs (records to update 12,000)

The return is bounded: an episode's return is highest when the agent takes the shortest path to the goal. The
training success rate has been 0.98 to 1.00 since about update 8,000 in five of the six networks (0.89 in N = 500
seed 1). The return still rises because episodes get shorter: the median episode length falls from 199-260 steps
over updates 2,000-4,000 to 141-227 steps over updates 11,000-12,000. The rise slows: per 2,000 updates, the median
return of N = 1,000 seed 1 rises by 0.37, 0.22, 0.20 and 0.09 over the four continued stages.

## Runs

The six networks whose records reach update 12,000 (N = 500 seeds 1, 3; N = 1,000 seeds 1, 3, 4, 5; launcher
indices 4, 6, 7, 9, 26, 27), four more stages of 2,000 updates (stages 7-10, updates 12,001-20,000), same launcher
(`slurm/rl_maze_long_della.slurm`), same checkout and commit (1a80f5e), same constant learning rate, `INIT_ADAM=false`.
About 6 x 4 x 5.1 = 122 GPU-h. A network that stops with non-finite rollouts is not rerun and not counted; it is
named.

## Readings (thresholds unchanged from entry 2026-11-11_2322 unless stated)

- **A, decrease over updates 12,000 to 20,000.** `maze_long_read.reading_a` with START = 12,000 and END = 20,000:
  a network decreases if it loses at least 5 active units per 1,000 updates and its ten-probe median is lower at
  20,000 than at 12,000. Two-thirds rule (`three_way`) over the networks with a last probe success of at least 0.9.
  Fails if fewer than two thirds decrease: the paper then says that in Maze networks the decrease slows below 5
  units per 1,000 updates after update 12,000.
- **B, flat return (primary for the abstract's sentence).** `maze_flat_return_full.measure`, reading "B": window =
  the last quarter (updates 15,000 to 20,000), flat if the return change per 1,000 updates is below 3% of the rise.
  A verdict needs at least four flat networks. "Decreasing while the return is flat" (two thirds of the flat
  networks lose at least 5 active units per 1,000 updates in the window): the after-convergence sentence of the
  abstract, Introduction, Results 2 and Discussion names all three tasks. "Not decreasing": the paper states that
  in Maze networks the decrease stops when the return is flat. Fewer than four flat networks: no verdict, the
  sentence keeps two tasks, and no further continuation at a constant rate is submitted without a new plan.
- **C, flat success rate (new; defined after seeing the records to 12,000, applied only to updates after 12,000).**
  Window = updates 12,000 to 20,000. The training success rate is flat if the medians of its first and last 100
  updates in the window differ by less than 0.01. Decreasing as in A. Two-thirds rule over the networks with a flat
  success rate, at least four needed. The flat condition is expected to hold; the open part is the decrease. A
  passing C supports only the sentence "the number of active units decreases while the success rate no longer
  changes", never the sentence about the return.

`maze_long_read.py` will be changed only to take these windows (START, END, reading C); the change is committed
before the records are read.
