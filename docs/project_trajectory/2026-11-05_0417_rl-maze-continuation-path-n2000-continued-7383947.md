## 2026-11-05 04:17 — Maze: a continuation path for runs (merged facaf41); N = 2,000 continued to 4,000 updates and seed 3 rerun, Spock 7383947 and 7383948 (issues #3, #4)

Follows `2026-11-05_0414_maze-continuation-path.md` (the implementation, by an independent agent)
and the uniform-series read-out of 03:23.

**Why.** Under the uniform protocol N = 2,000 did not reach the learned criterion in 2,000
updates (probe success 0.77 and 0.84, rising), and seed 3 was stopped by the guard at update 341.
The trainer had no way to continue a run.

**What was merged.** `paths.init_from`, `paths.init_adam`, `trainer.iter_offset` for maze runs:
`run_rl.py` saves `<stem>_AdamState_Maze.pt` at every checkpoint; a continuation loads the
parent's W_inp, W_rec, W_out, y_init and, with init_adam, its moments (then the warm-up is
skipped; otherwise it runs again); records carry absolute update indices;
`common.maze_load_runs` joins the chain so the readers see one run. Bit-identical to main when
unused (N = 40, seed 7, 3 updates); `tests/test_rl_maze.py` 24 of 24. Both suite launchers gain a
continuation mode (`PARENT_ROOT`, `EXTRA_UPDATES`, `INIT_ADAM`).

**Procedure for the large sizes, the same at N = 2,000 and 4,000.** Each finished run is
continued for 2,000 more updates with the optimizer restarted and the warm-up repeated (the
parents were written before the Adam-state file existed), so these sizes get 4,000 updates as two
runs of 2,000 with a restart between them; "learned" is read on the joined chain and "end" is
update 4,000. This is a stated deviation from the one-run protocol of N = 250 to 1,000.

**Submitted** from `~/trainRNNbrain_rlmaze4` at facaf41 (clean): `REPO=... PARENT_ROOT=
$HOME/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite EXTRA_UPDATES=2000 INIT_ADAM=false sbatch
--array=10-11 --time=24:00:00 slurm/rl_maze_suite_spock.slurm` = 7383947 (N = 2,000 seeds 1 and 2,
cells `Maze_suite/N=2000_act=relu_cont/s{1,2}`), and `REPO=... sbatch --array=12
--time=24:00:00 ...` = 7383948 (seed 3 rerun from scratch, 2,000 updates, with the Adam state
saved; to be continued the same way). About 22 GPU-h. N = 4,000 (Della 16883344) follows the same
procedure when its three runs finish.
