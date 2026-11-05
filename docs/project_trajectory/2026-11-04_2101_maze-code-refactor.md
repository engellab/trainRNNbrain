## 2026-11-04 21:01 — Maze: the RL code moved into the Task / RNN / Trainer / training-script structure (issue #4); behaviour unchanged

Branch `wp/maze-refactor` (worktree `~/Documents/Projects/paper_push/wt/maze-refactor-code`), from
`main` at 5929579. Not merged and not pushed: Pavel reviews the diff. Pavel, 4 Nov: "it really would be
better if you still use the same conceptual structure: Task, RNN, Trainer, and training script paradigm."

**What moved where.** Every computation is as it was; the files were moved with `git mv` and reshaped.

| before (5929579) | after |
|---|---|
| `trainRNNbrain/rl/maze_env.py` (`MazeEnv`) | `trainRNNbrain/tasks/TaskMaze.py` (`TaskMaze`, a subclass of `Task`; `get_batch` raises, the trainer plays episodes) |
| the `env:` block of `configs/rl/maze_ppo.yaml` | `configs/task/Maze.yaml` (`# @package task`, `taskname: Maze`, `seed: ${seed}`, `n_inputs`/`n_outputs` null = derived from the sensors and recorded in the saved config; every value and comment as it was) |
| `trainRNNbrain/rl/policy.py` (`MazePolicy`), `rl/ppo.py`, the loop of `rl/run_maze_ppo.py` | `trainRNNbrain/trainer/TrainerPPO.py` (`TrainerPPO`: draw_noise, std, initial_state, step, heads, sequence, log_prob, entropy, rollout, safe_rollout, gae, mc_advantages, ppo_update, probe, run_training; the `checkpoint_fn` hook of the supervised Trainer) |
| the `ppo:` block of `configs/rl/maze_ppo.yaml` | `configs/trainer/trainer_ppo.yaml` (`# @package trainer`, `trainer_tag: ""`; every value and comment as it was) |
| `rl/run_maze_ppo.py` (its own Hydra main) | `trainRNNbrain/training/run_rl.py` (`run_rl(cfg)`, save_run, the two figures), called by `run_experiment.run_training` when the trainer's `_target_` ends in `TrainerPPO`; the branch is the first statement of `run_training`, the supervised path below it is untouched |
| `rl/saved.py` `load_run` | `TrainerPPO.load_run(folder, device) -> (cfg, task, trainer)`, beside the class; reads both run layouts |
| `rl/make_maze.py` | `tasks/make_maze.py` (a generator script, kept beside the task it serves; nothing imports it) |
| `trainRNNbrain/rl/`, `configs/rl/` | deleted |

**Entry point.** `python trainRNNbrain/training/run_experiment.py task=Maze trainer=trainer_ppo
model=rnn_relu_standard model.N=500 seed=1`. The three launchers (`slurm/rl_maze_suite_della.slurm`,
`rl_maze_suite_spock.slurm`, `rl_maze_ppo_della.slurm`) call it with their grids and provenance blocks
unchanged; their `SAVE_TO` now defaults to `.../trained_RNNs_rl_maze` and is passed as `paths.save_to`,
so the launcher chooses the tree and the script no longer appends `_rl_maze` in code.

**Save layout.** `<paths.save_to>/Maze_<experiment_tag>/s<seed>/`: `s<seed>_config.yaml` (resolved; the
seed, `task.n_inputs`/`n_outputs` and `model.n_inputs`/`n_outputs` written back), `s<seed>_Params_Maze.npz`
(RNN_torch.get_params(); named Params, not LastParams, because the cell inventories glob `*LastParams*`),
`s<seed>_Progress.json`, `s<seed>_Participation.npz`, `progress.png`, `trajectories.png`. `PolicyState.pt`
is gone: it held the same three weight matrices as the npz. The folder is overwritten at every
`trainer.checkpoint_every` updates through `checkpoint_fn` and once more at the end, as before, so a
killed run leaves its last checkpoint readable. The readers (`common.maze_load_runs`,
`TrainerPPO.load_run`, `fig_paper_F1.maze_ablation`) read the earlier layout (`config.yaml`,
`Progress.json`, `Participation.npz`, the task under `env:`) as well; nothing on disk was converted.

**Behaviour check, pass rule set first: every saved number identical.** The old script
(`rl/run_maze_ppo.py` at 5929579) and the new path, both at N = 40, 3 updates of 8 episodes x 12
decisions, probe of 6 episodes, warm-up 2, seed 7, CPU: Progress.json equal in every key of every
update (largest difference 0), W_rec, W_inp, W_out and y_init equal (0), every Participation array
equal (0), and the task and trainer blocks of the saved config equal key by key to the old env and
ppo blocks. `load_run` on `Maze_suite/N=1000_act=relu/s1` (old layout) gives the same weights and the
same outcome and step count in each of the 128 probe episodes as the old `rl/saved.py` loader run from
the unchanged main checkout, both on the CPU. (The success recorded during training on Della's GPU
differs from the CPU rebuild by 2 episodes of 128 under either loader: rounding over 2,500 RNN steps,
not the code.)

**Checks.** `tests/test_rl_maze.py` 13 of 13: the 12 checks kept with their imports and constructors
renamed (the trained-parameter check now reads the names `W_inp`, `W_out`, `W_rec` without the `rnn.`
prefix, since no Module wraps the RNN any more, and also checks that the optimizer holds exactly those
parameters) plus one new check that runs the training script at N = 40 for 2 updates into the
scratchpad and reads the folder back with `maze_load_runs` and `load_run`.
`tests/test_no_function_imports.py` 202 files ok; `tests/test_scripts_start.py` 48 scripts ok.
`rl_maze_readout.py`, `fig_ed_maze.py` and `fig_supp_tasks.py` ran on `data/trained_RNNs_rl_maze/Maze_suite`
(the tables as before; the regenerated figure files were restored, so the committed figures are
untouched); `rl_maze_animate.py` ran on `Maze_suite/N=250_act=relu/s1`. Second-array runs
(`Maze_second`, no `action_feedback`, a separate value head) load as before; first-array runs
(`Maze_first`, no `substeps` key: one RNN step per decision, clipped actions, discounted shaping) are
refused with a message saying why, where the old loader failed on them with a bare missing-key error.

**Decisions for the review.**
1. `n_inputs`/`n_outputs` are `null` in `Maze.yaml`: TaskMaze derives them from the sensors (8 rays + 2
   efference-copy inputs + 2 with the goal cue; 2 actions); a number given there is checked against the
   derivation and a mismatch raises. The run script records the derived numbers in the saved config, as
   it records the seed.
2. `T` (decisions per episode) and `substeps` stay in the trainer block, as the issue's table has them,
   so the maze task has no trial length and `training_utils.prepare_task_arguments` is not used for it;
   `run_rl.py` builds the task with `filter_kwargs(TaskMaze, cfg.task)`.
3. TaskMaze seeds its drop generator with `seed + 1` (the RNN gets `seed`), as the old script did, so a
   rerun reproduces the suite's drops; the config comment and the docstring say so.
4. `lr`, `lr_scale_exp` and `weight_decay` sit in `trainer_ppo.yaml` and are read by `run_rl.py`, which
   builds the Adam optimizer with the size rule, as `run_experiment` does for the supervised Trainer;
   TrainerPPO takes the base rate from the optimizer for its warm-up ramp.
5. The run sub-folder is `s<seed>` and the file stem `s<seed>`: `utils.make_subfolder_tag` puts a
   validation score first in the name, which a maze run does not have; the supervised checkpoint tree
   names its run folders `s<seed>` too.
6. The suite stays in `<save_to>_rl_maze` by the launchers' `SAVE_TO` default: `fig_paper_F1._cell_index`
   opens every `<cell>/*/*_config.yaml` under the supervised root and skips a maze config only because
   `_signature` fails on it, and every maze run so far is in that tree, as are the Spock and Della jobs
   running now. A local run with `paths_local` lands under `data/trained_RNNs/Maze_<tag>/`.
7. `cfg.n_nets` is not used on the maze path (one network per run, as before).
