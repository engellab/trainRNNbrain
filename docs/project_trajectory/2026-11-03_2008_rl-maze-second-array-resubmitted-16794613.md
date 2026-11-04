## 2026-11-03 20:08 — Correction: the second maze array is 16794613, writing to Maze_second/ (issue #2)

Corrects `2026-11-03_2007_rl-maze-first-array-fails-substeps-second-array.md`, which said that
array 16794476 would rewrite the first array's cell folders under `Maze_first/`. It was cancelled
while every task was still pending (no file written), the launcher now tags its runs
`second/N=<N>_cue=<bool>` (commit 9e5d8d1, the only change), the Della worktree
`~/trainRNNbrain_rlmaze` was moved to 9e5d8d1, and the array was resubmitted as **16794613**
(tasks 1-8, the same grid and budget). Outputs:
`/home/pt1290/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_second/N=<N>_cue=<bool>/s<seed>/`;
logs `~/trainRNNbrain_logs/MazePPO.16794613_<i>.out`. The first array's folders under
`Maze_first/` are untouched on Della and copied to the Mac.
