## 2026-11-25 13:12 — Matched Maze controls: submitted, time limits set to the cap, read job chained on Della

Follows the rules `2026-11-25_1237_rules-maze-matched-controls.md`, committed and pushed at 1fa9d54 before any job.
Branch `wp/maze-matched-controls`. Written before any network was read.

**Jobs** (Della, pinned clone `~/trainRNNbrain_entsp_maze` at e555fae, clean; commands as in the rules):
- 18374320, tasks 1-3: N = 1,000, 2,000 updates, all on A100 40 GB cards of the i14 nodes.
- 18374321, tasks 1-3: N = 2,000, updates 1-2,000, on 80 GB cards.
- 18374322, tasks 1-3: N = 2,000, updates 2,001-4,000, each after the matching task of 18374321 (`aftercorr`).

**Why the plan changes.** Firstmate (25 Nov, 21:09 UTC) expects the Della connection to drop for long stretches and
asks that the chain, the read included, runs without a watcher and cannot exceed the cap.

**Time limits that respect the 120 GPU-hour cap unwatched.** At `--time=23:00:00` the nine tasks could use 207
GPU-hours. The limits are lowered, with `scontrol update`, so that their sum is 120:

| Job | Tasks | Limit | Sum | Time expected per task | Basis |
|---|---|---|---|---|---|
| 18374320 | 3 | 12 h | 36 | 9.1 h | 15.8-16.2 s per update, measured over the first 110-120 updates |
| 18374321 | 3 | 11 h | 33 | 8.4 h | 14.6-15.2 s per update, measured over the first 30-90 updates |
| 18374322 | 3 | 17 h | 51 | 8.5 h | 8.3-8.7 h in the entropy networks' continuations |

The rules asked for limits that cover a node at twice the usual time. The continuation's limit does (its node is not
known yet). The first six tasks run on known nodes at measured speeds, and their limits are 1.3 times the measured
time. A task that reaches its limit is continued under the wall-clock rule, within the GPU-hours that remain.

**Read job.** `slurm/maze_matched_read_della.slurm` (CPU), run from a clean clone `~/trainRNNbrain_mctrl_read` of this
branch, depends on both final training jobs (`afterany:18374320:18374322`, so a failed task is read too, as not
finished). It runs the committed reader and then the independent recount, on the trees on Della. The original controls
are copied from the Mac to `trained_RNNs_rl_maze_mctrl_read/Maze_suite_mac` (36 files: config, `Progress.json` and
`Participation.npz`), because the Della folder of that name holds other runs. Output goes to
`trained_RNNs_rl_maze_mctrl_read/{reads,verdict,oracle}_matched.*`. If a first-stage task fails, its continuation never
starts and the read job stays pending; it is then run by hand under the rules.

A dry run of the read job, without the matched controls, must reproduce the original verdicts on Della and pass the
recount before the real read job is submitted. Checks 1 and 2 of the rules (saved configs; update-1 rollouts) are made
on the first checkpoints.

The config check compares keys only, so it says nothing about the networks' results. No active count or success after
update 1 is compared before the jobs end.
