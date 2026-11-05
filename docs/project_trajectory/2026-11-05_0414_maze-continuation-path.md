## 2026-11-05 04:14 — Maze: a continuation path for the PPO runs (paths.init_from, paths.init_adam, trainer.iter_offset), branch wp/maze-resume

**Why.** The two uniform-protocol runs at N = 2,000 ended their 2,000 updates below the learned
criterion (probe success 0.77 and 0.84, still rising), and a maze run could not be continued: a
longer budget meant a full rerun. The supervised runs are continued with `paths.init_from` and
their losses are joined along the chain (`common.warm_start_chain`); the maze path now has the same
mechanism. Branch `wp/maze-resume` from main at 0ea1ac7, commits c4e5e54 and 7de30b4; not merged,
not pushed, nothing submitted.

**What each piece does.**
- `trainRNNbrain/training/run_rl.py`: `save_run` writes `<stem>_AdamState_Maze.pt` (the optimizer's
  state_dict) beside the standard files at every checkpoint and at the end, as run_experiment does.
  With `paths.init_from` set, `load_warm_start` copies the parent's W_inp, W_rec, W_out and y_init
  from its `*Params_Maze.npz` into the freshly built RNN (shapes checked); with `paths.init_adam` the
  parent's AdamState is loaded into the optimizer, and the rate and the weight decay in its
  param_groups are set back to this run's config (a restored state carries the parent's last rate).
  `trainer.iter_offset: auto` resolves to the parent's last recorded update (`warm_start_offset`,
  read from the parent's Progress.json; 0 without a parent) and is written back into the saved
  config, as run_experiment records it.
- `trainRNNbrain/trainer/TrainerPPO.py`: two constructor arguments, `iter_offset` and
  `adam_restored`. Every update index in Progress.json and Participation.npz (update, p_update) is
  iter_offset + the run's own count, so a continuation's records start where the parent's end. The
  loop, the probe grid, the checkpoint grid and the warm-up ramp count the run's own updates from 1,
  as the supervised Trainer's track grid does. The ramp is not run when the moments were restored
  and is run again when they were not (`configs/trainer/trainer_ppo.yaml` says why under
  warmup_updates). The rescale rule's state and the dropout participation EMA are not saved, as in
  the supervised trainer, and start afresh in a continuation.
- `trainRNNbrain/experiments_and_analysis/common.py`: `maze_run_chain` follows `paths.init_from`
  oldest-first (a parent missing from its recorded path is looked for under the root by
  `<cell>/s<seed>`, then as a nested `parent_*` folder; a warning when neither exists).
  `maze_load_runs` joins each chain into one record, progress and probe arrays concatenated in
  update order with the parent's first, lists the parents only inside the chain, and stops on a
  chain whose update indices do not increase. The record gains `chain`. rl_maze_readout.py,
  fig_ed_maze.py and fig_paper_F1.py are unchanged and see a continued run as one run;
  `maze_learned_index` runs on the joined probes.
- `slurm/rl_maze_suite_spock.slurm`, `slurm/rl_maze_suite_della.slurm`: with `PARENT_ROOT` (the
  folder holding the cells) and `EXTRA_UPDATES`, index i continues `<PARENT_ROOT>/N=<N>_act=<act>/s<seed>`
  with `paths.init_from=<parent> paths.init_adam=<INIT_ADAM> trainer.n_updates=<EXTRA_UPDATES>
  trainer.iter_offset=auto` into the cell `<tag>_cont` beside the parent. `INIT_ADAM` defaults to
  true; the runs of the uniform series were written before the AdamState file existed, so they are
  continued with `INIT_ADAM=false` (Adam afresh, the ramp again). Guards: a missing EXTRA_UPDATES, a
  missing parent and, with INIT_ADAM=true, a missing AdamState each stop the job. Without PARENT_ROOT
  the command line is the one before. The provenance block is unchanged.
- `configs/trainer/trainer_ppo.yaml` gains `iter_offset: auto` and the warm-up comment;
  `configs/paths/paths_local.yaml` and `paths_DELLA.yaml` say how init_from and init_adam apply to a
  maze run; `docs/AGENT_GUIDE.md` has one clause under its warm-start bullet.

**Equivalence with init_from unset, pass rule set first: every saved number identical.** Main at
0ea1ac7 against the branch at c4e5e54, the earlier merges' recipe (N = 40, seed 7, 3 updates of 8
episodes x 12 decisions, probe of 6 episodes, warm-up 2, CPU): Progress.json has the same keys in
every update with largest difference 0; all 15 arrays of Params_Maze.npz and all 6 of
Participation.npz bitwise equal; the task, model and paths blocks of the saved config equal (paths
differs only in save_to, the two scratch folders); the trainer block gains exactly `iter_offset: 0`.
The branch adds s7_AdamState_Maze.pt. IDENTICAL.

**Checks.** `tests/test_rl_maze.py` 24 of 24: the layout check now lists the AdamState file, and
`test_continuation_from_folder` runs a 2-update parent (N = 40, seed 3, warm-up 2) and three
continuations through the training script. With init_adam: Progress updates 3 and 4, the rate at
both the full rate (no ramp), probe updates 3 and 4 with p_update 3, the saved config's iter_offset
2; `maze_load_runs` returns the parent and the child as one record with updates 1-4 in order, the
parent's probes first, chain [parent, child], and the parent not on its own. Without init_adam: the
rate at the first update is lr / 2, the ramp repeated. With init_adam at rate 0 the saved W_rec,
W_inp, W_out and y_init equal the parent's bitwise, so the continuation starts from the parent's
last weights. `tests/test_no_function_imports.py` 202 files ok; `tests/test_scripts_start.py` 48
scripts ok.

**Launcher dry run** from the worktree with module, conda and srun stubbed (srun appends N = 40, 2
updates), both launchers, index 10 and 11. The first dry run found that every continuation failed in
Hydra's override parser: the parent path holds '=' (the cell name N=2000_act=relu) and an unquoted
override value may not; the path is now passed in literal double quotes, as experiment_tag already
is (7de30b4). After the fix: fresh mode's command line is unchanged and its folder holds the seven
files; the three guards exit 1 with their FATAL lines; four continuations ran (Spock seeds 1 and 2,
Della seed 1, INIT_ADAM true and false), each logging the loaded weights, the restored moments where
asked and "records updates 3-4"; each continuation's saved config differs from its parent's in
exactly experiment_tag, paths.init_from, paths.init_adam (where true) and trainer.iter_offset (0 to
2); `maze_load_runs` on the suite root returns one record per seed with updates 1-4 and a chain of
two folders, and `rl_maze_readout.py --root` on it prints two runs of 4 updates. The two
tracked figures it overwrote in img/internal_figures/ were restored from git.

**Not mirrored, and why.** (1) The offset comes from the parent's Progress.json, not its config as
the supervised `warm_start_offset` does: the maze checkpoint overwrites the run folder in place and
does not stamp n_updates with the updates done, so a parent killed by its wall clock has a config
whose n_updates is the budget while its records hold the count. (2) The maze checkpoint is still
written file by file, not staged and renamed as `write_checkpoint` does; a job killed during a save
can leave Params and Progress from two different updates. (3) The parents that need continuing now
(the uniform series) have no AdamState, so their continuations restart Adam and ramp again; only
runs written from this branch on can be continued with the moments. (4) The joined record keeps the
child's folder as `folder`, so fig_paper_F1's run name for a continued run is the `_cont` cell.
