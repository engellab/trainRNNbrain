# Maze long training: a second network stopped, at update 8,500; seven of nine remain

Follows entry `2026-11-12_*_maze-long-one-network-stopped.md`.

- **What failed.** Task 28 of stage three (job 17379994_28: N = 1,000, seed 6, updates 8,001 to 10,000)
  exited with code 1 after 1 h 20 min, at update 500 of 2,000 (update 8,500 on the joined clock), with the
  same `safe_rollout` stop: the rollout was non-finite at updates 498, 499 and 500 after weights had been
  restored from the last checkpoint. The last log line shows a probe with 0 active units and success 0.00,
  which is the probe of the non-finite network and not a measurement of a trained one.
- **What was cancelled.** Its stage four, 17379995_28. Nothing was deleted.
- **State.** Seven networks are in stage four (updates 10,001 to 12,000), started between 12:50 and 13:15
  Eastern; 23 stage tasks completed, 2 failed.
- **Consequence.** As for the first stopped network: not rerun, not counted by `maze_long_read.py`
  (record short of update 12,000), named in the report. Two of nine default networks stopping with
  non-finite rollouts between updates 4,000 and 12,000 is itself reported with the result.
