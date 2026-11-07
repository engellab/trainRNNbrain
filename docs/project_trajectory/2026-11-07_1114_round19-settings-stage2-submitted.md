## 2026-11-07 11:14 — Round 19 part 2 submitted: the 63 Maze settings networks to 4,000 updates (rules in 2026-11-07_1111_round19-settings-stage2-rules.md)

Code: commit 7889823, clean detached worktrees `~/trainRNNbrain_r19s` on Della and Spock. Every job continues
its arm's update-2,000 run with `PARENT_ROOT=<maze tree>/Maze_interventions EXTRA_UPDATES=2000
INIT_ADAM=false` (stage 2) into `Maze_interventions/<arm>_cont/s<seed>`. The 18 Della parents were copied from
Spock through the Mac into the same layout on Della (`~/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_interventions/`,
sigmaw1.0, met10 and the four SGD arms; the parameter-file checksums match on the three machines for the
one checked, sgd1 s1). Times are US Eastern.

| Job | Cluster, id | Indices | Networks | GPU-h | Expected end |
|---|---|---|---|---|---|
| sigmaw1.0 s1-s3 | Della 17049801 (`rl_maze_interventions_della.slurm`, `--time=24:00:00`) | 16-18 | 3 | 47 | about 16 h after start |
| met10, sgd0.03, sgd0.1, sgd0.3, sgd1 | Della 17049802 (`--time=12:00:00`) | 91-108 | 15 | 80 | about 5.5 h after start |
| the other 45 | Spock 7398207 (`rl_maze_interventions_spock.slurm`, `--time=12:00:00`) | 7-9, 13-15, 19-21, 26-30, 37-39, 43-45, 49, 61-63, 67-72, 79-90, 94-96 | 45 | 262 | about 6 h after each task starts |

All three arrays were pending at submission (Della: priority and resources; Spock: behind the lab's queue).
Total about 389 GPU-h. The controls at 4,000 (ReLU N = 1,000 seeds 4-6) are part 1's Spock 7398076, due about
19:00.
