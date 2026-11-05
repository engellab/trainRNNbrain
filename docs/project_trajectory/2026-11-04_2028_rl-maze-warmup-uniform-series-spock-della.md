## 2026-11-04 20:28 — Maze: the first-update jump explained, learning-rate warm-up, one protocol for N = 250 to 4,000; uniform series on Spock (7383494, 7383495) and Della (16883344) at 9dce02e (issue #3)

Follows `2026-11-04_1237_rl-maze-suite-submitted-16856486-16856487.md` and the N = 2,000
corrections of the same day. Pavel, 4 Nov: uniform data at N = 250, 500, 1,000, 2,000, 4,000 for
all three tasks; Spock may be used again.

**Why N = 2,000 died at the paper's learning rate.** From the suite's Progress.json: the first
update's KL (the policy change in one update) grows with size, seed means 0.04 (N = 250), 0.047
(500), 0.146 (1,000), and 0.24 for the one N = 2,000 seed that survived update 1; every later
update is at 0.003 at every size, including N = 2,000 at a quarter of the rate (0.0025 to
0.0027). Adam's first steps move every parameter by about lr before its moment estimates exist,
so the first change of the read-out, a sum over N units, scales with lr x N; the rule
lr = 1e-3 (100/N)^(1/3) does not shrink fast enough, and at N = 2,000 the ReLU dynamics (no
ceiling) blew up over the 2,500-step episode. Gradient clipping (norm 0.5, already on) cannot
help, since Adam sets the step size; a cubic term (gamma 0.1) would bound the state but changes
the standard network and would have to go on every task.

**Fix.** `ppo.warmup_updates: 50` (commit 9dce02e): the rate ramps linearly from 0 over the first
50 updates, the same at every size. Pilots of 60 updates at the paper's rate, rule set before
(no restore, no skipped minibatch, maximum KL over updates 30-60 below 0.05): N = 2,000, Spock
job 7383478, max KL 0.018, 14 s per update; N = 4,000, Della job 16881728 on an 80 GB card, max KL
0.024, 33 s per update, probe success 0.24 at update 60. Both pass. Launchers at 9dce02e:
`slurm/rl_maze_suite_della.slurm` (indices 21-23 = N = 4,000, `--constraint=gpu80`) and
`slurm/rl_maze_suite_spock.slurm` (sizes to 2,000, `--partition=all`).

**Submitted.** Spock, worktree `~/trainRNNbrain_rlmaze` at 9dce02e: `sbatch --array=1-9,13-20
--time=12:00:00 slurm/rl_maze_suite_spock.slurm` = 7383494 (17 runs) and `sbatch --array=10-12
--time=24:00:00 ...` = 7383495 (N = 2,000 x 3); all 20 started within a minute. Della, worktree
`~/trainRNNbrain_rlmaze` moved to 9dce02e: `sbatch --array=21-23 --time=24:00:00
--constraint=gpu80 slurm/rl_maze_suite_della.slurm` = 16883344 (N = 4,000 x 3). About 170 GPU-h;
the maze line is at about 350 GPU-h in all, over the 200 GPU-h package cap, on Pavel's request.
Outputs: Spock `~/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/N=<N>_act=<act>/s<seed>/`;
Della the same path for N = 4,000. Logs `~/trainRNNbrain_logs/MazeSuite.<array>_<i>.out` on each.

**What the earlier runs are.** The 17 runs of the first suite (no warm-up, Della, Progress 12 on
issue #3) are a replication set under the previous protocol; the three quarter-rate N = 2,000
runs (16866951) the learning-rate exception. On the Mac the first suite moves to
`data/trained_RNNs_rl_maze/Maze_suite_v1_nowarmup/` when the uniform series is copied in.
Read-outs and falsifiers as pre-registered (`2026-11-04_0818_...md`, issue #3).
