## 2026-11-11 23:29 — Maze long training submitted: nine default ReLU networks continued from update 4,000 to 12,000 (Della 17379992-17379995)

Rule: `2026-11-11_2322_rules-maze-long-training-and-retrain-at-250.md`, section 1, committed (3a0cd93) before
any job below was submitted. Stage protocol: `2026-11-07_1008_round19-train-all-rules.md`. The entry time is
the Mac's clock (Pacific); cluster times below are Della's (US Eastern, three hours later).

**Terms.** *Maze*: the egocentric maze task trained by PPO (proximal policy optimisation, a
reinforcement-learning algorithm). *Update*: one batch of 256 episodes followed by 8 optimiser steps.
*Stage*: 2,000 updates. *Continuation*: a run that starts from the saved weights of the previous stage
(`paths.init_from`), with the update numbering continued (`trainer.iter_offset: auto`). *Probe*: a noise-free
test of 128 episodes after every 10th update, which also records each unit's participation. *GPU-h*: hours of
one GPU.

### What was submitted

Nine networks of the Maze size series (`Maze_suite`), default settings, ReLU: N = 1,000 seeds 1-6 and
N = 500 seeds 1-3. Each is continued from its weights at update 4,000 for four stages of 2,000 updates
(stages 3 to 6, updates 4,001-12,000). Each stage starts from the previous stage's end weights with a fresh
Adam state and the 50-update learning-rate warm-up (`INIT_ADAM=false`), as their stage 2 did. The learning
rate is constant after the warm-up (0.001 × (100/N)^(1/3); 4.64e-4 at N = 1,000).

### Where the earlier stages ran, and which trainer version they used

**The earlier stages of these nine networks ran on Spock, not on Della.** Della's
`Maze_suite/N=1000_act=relu/s1-s3` holds Della's own first uniform series (array 16856486), which are other
networks with the same folder names. Della has no `N=1000_act=relu_cont` or `N=500_act=relu_cont`.

| Networks | Stage 2 job (updates 2,001-4,000) | Checkout and commit, from the job logs' `code:` line | Stage 2 wall time |
|---|---|---|---|
| N = 1,000 seeds 1-3; N = 500 seeds 1-3 | Spock 7397701, indices 7-9 and 4-6 | `~/trainRNNbrain_r18train` @ 6317f30 | 4.6-5.3 h; 4.0-4.2 h |
| N = 1,000 seeds 4-6 | Spock 7398076, indices 29-31 | `~/trainRNNbrain_r19` @ 1a80f5e | 5.4-5.6 h |

**Trainer version: the original trainer, without `freeze_after_end`.** Three checks:
1. The saved stage-2 configs of all nine networks (`N=<N>_act=relu_cont/s<seed>/s<seed>_config.yaml`) have no
   `trainer.freeze_after_end` key. A config without the key ran with the switch off (entry
   `2026-11-08_0953`).
2. The fix that added the switch, d2d0b3f, is not an ancestor of 1a80f5e
   (`git merge-base --is-ancestor d2d0b3f 1a80f5e` fails), and `freeze_after_end` does not occur in
   `TrainerPPO.py` or `trainer_ppo.yaml` at 1a80f5e. Code at 1a80f5e cannot hold an ended episode's state.
3. `git diff --stat 6317f30 1a80f5e` lists no file under `trainRNNbrain/trainer`, `trainRNNbrain/training`,
   `trainRNNbrain/rnns` or `configs/{trainer,model,paths}`, and neither `TaskMaze.py` nor
   `configs/task/Maze.yaml`. Under `trainRNNbrain/tasks` and `configs/task` only `make_maze.py` (a mirror
   option for the generator script) and `Maze_transfer.yaml` changed, and these runs use neither. So the two
   stage-2 commits have the same trainer, run script, model, task and configs.

All nine networks are therefore continued from one checkout at **1a80f5e**. No network changes trainer version.

### Code and launcher on Della

- Checkout: `~/trainRNNbrain_mazelong`, a new worktree detached at 1a80f5ea05d6c823d19bebd17776bfc6ed9373a8,
  clean (`git status --porcelain` empty). The existing `~/trainRNNbrain_r19` (same commit) was not touched.
- Launcher: `slurm/rl_maze_long_della.slurm` (new, committed with this entry; md5
  b4e325b751e8c9d4d9301dc824e38818 on the Mac and on Della). It was copied into the Della worktree with `scp`;
  `slurm/*.slurm` is gitignored, so the worktree stays clean. A new launcher was needed because the Della
  suite launcher at 1a80f5e has no index for N = 1,000 seeds 4-6. The new file is
  `rl_maze_suite_della.slurm` of 1a80f5e with five changes: indices 26-28 (N = 1,000 seeds 4-6); the job name
  `MazeLong` and default array; default `REPO` and `SAVE_TO`; a refusal to run from any commit other than
  1a80f5e; a refusal to run without `PARENT_ROOT`. The training command line is unchanged.
- The launcher passes `paths=paths_DELLA`; stage 2 on Spock passed `paths=paths_local`. The two files differ
  only in `save_to`, which the launcher sets.

### Parent folders and weights

The stage-2 folders were copied from the Mac (`data/trained_RNNs_rl_maze/Maze_suite/`, itself copied from
Spock) into a new tree on Della, `~/trainRNNbrain/data/trained_RNNs_rl_maze_long/Maze_suite/`. A tree of its
own keeps the Spock networks apart from Della's first uniform series. Nothing existing on Della was changed.

```
cd ~/Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite
rsync -a --ignore-existing --exclude='*.png' -R N=1000_act=relu_cont/s[1-6] N=500_act=relu_cont/s[1-3] \
  della:trainRNNbrain/data/trained_RNNs_rl_maze_long/Maze_suite/
```

Each parent's `s<seed>_Progress.json` holds updates 2,001-4,000. The md5 of each `s<seed>_Params_Maze.npz`
(the weights at update 4,000) is the same on Spock, the Mac and Della:

| Parent | md5 |
|---|---|
| `N=1000_act=relu_cont/s1` | b0d26ccefa41f33cc28bb27513f53838 |
| `N=1000_act=relu_cont/s2` | 14586aef0ad4689e397cf71b68db1c8f |
| `N=1000_act=relu_cont/s3` | ad7490c1f40a6479dd52b2ad45350fbd |
| `N=1000_act=relu_cont/s4` | 35b711027bd4faa28296f02476f50bf0 |
| `N=1000_act=relu_cont/s5` | 50d1db036cfe36bab9d4d06bb6f5b2b9 |
| `N=1000_act=relu_cont/s6` | 12655d6ae3d90e478eb06880236ab123 |
| `N=500_act=relu_cont/s1` | 0ef6d61bffb8d52465811ecd24e8e882 |
| `N=500_act=relu_cont/s2` | a59adeba880845a92e4f766e4db40177 |
| `N=500_act=relu_cont/s3` | 0ad12bd3ab12cc946d4dfe832c73e865 |

### Checks before submission (no job was queued as a test)

**Local launcher runs** (Mac, CPU): the launcher itself, from a clean detached worktree at 1a80f5e, with
`module`, `conda` and `srun` stubbed; the `srun` stub appends `trainer.n_episodes=4 trainer.T=6
trainer.probe_episodes=4`, and `EXTRA_UPDATES=2`. Parents: copies of the real stage-1 and stage-2 folders.
Pass condition, written before the diffs: each new saved config differs from its real stage-2 parent's only in
`experiment_tag`, `paths.init_from`, `paths.save_to`, `trainer.iter_offset` and the stub's four keys, and has
no `freeze_after_end` key. All passed.

| Run | Result |
|---|---|
| index 7 (N = 1,000 seed 1), `STAGE=3` | weights loaded from `N=1000_act=relu_cont/s1`; "this run records updates 4001-4002"; lr 4.64e-04; config differs from the parent's only in the allowed keys (`iter_offset` 2,000 → 4,000, tag `_cont3`) |
| index 7, `STAGE=4`, from the `_cont3` folder just written | updates 4003-4004; differs from the `_cont3` config only in tag, `init_from`, `iter_offset` |
| index 27 (N = 1,000 seed 5), `STAGE=3` | seed 5 parent loaded; same allowed keys only |
| index 5 (N = 500 seed 2), `STAGE=3` | same |
| index 7, `STAGE=3` again | exits 1: refuses to overwrite `_cont3/s1` |
| index 27, `STAGE=5` with no stage-4 folder | exits 1: no parent weights |
| index 8 without `PARENT_ROOT` | exits 1 |
| `common.maze_load_runs` on the resulting tree | each network read as one record: seed 1 through four folders, updates 1-4,004 with no gap or repeat; seed 5 and N = 500 seed 2 through three folders, updates 1-4,002 |

**On Della's login node:** index 26 (N = 1,000 seed 4), `STAGE=3`, with only `srun` stubbed the same way,
`SAVE_TO=/tmp/pt1290_mazelong_smoke/tree` and the real parent tree: `code: /home/pt1290/trainRNNbrain_mazelong
@ 1a80f5e`, weights loaded from the real `N=1000_act=relu_cont/s4`, updates 4001-4002, exit 0. This run wrote
only under `/tmp`.

**Not tested:** a full 2,000-update stage on a GPU, and the `aftercorr` hand-over between real stages.

### Submission (Della, 12 Nov 02:29 Eastern; `squeue -u pt1290` was empty before)

```
export REPO=$HOME/trainRNNbrain_mazelong SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_long
export PARENT_ROOT=$SAVE_TO/Maze_suite EXTRA_UPDATES=2000 INIT_ADAM=false
STAGE=3 sbatch --parsable --chdir=$REPO --array=4-9,26-28 --time=12:00:00 slurm/rl_maze_long_della.slurm                                   # 17379992
STAGE=4 sbatch --parsable --chdir=$REPO --array=4-9,26-28 --time=12:00:00 --dependency=aftercorr:17379992 slurm/rl_maze_long_della.slurm   # 17379993
STAGE=5 sbatch --parsable --chdir=$REPO --array=4-9,26-28 --time=12:00:00 --dependency=aftercorr:17379993 slurm/rl_maze_long_della.slurm   # 17379994
STAGE=6 sbatch --parsable --chdir=$REPO --array=4-9,26-28 --time=12:00:00 --dependency=aftercorr:17379994 slurm/rl_maze_long_della.slurm   # 17379995
```

`aftercorr` starts index i of a stage only after index i of the previous stage has ended with exit code 0.
One GPU per task (`--gres=gpu:1 --constraint=nomig`), at most 9 at once; the gpu-short limit is 24 GPUs per
user.

| Network | Index | Stage 3 (4,001-6,000) → `_cont3` | Stage 4 (6,001-8,000) → `_cont4` | Stage 5 (8,001-10,000) → `_cont5` | Stage 6 (10,001-12,000) → `_cont6` |
|---|---|---|---|---|---|
| N = 500 seed 1 | 4 | 17379992_4 | 17379993_4 | 17379994_4 | 17379995_4 |
| N = 500 seed 2 | 5 | 17379992_5 | 17379993_5 | 17379994_5 | 17379995_5 |
| N = 500 seed 3 | 6 | 17379992_6 | 17379993_6 | 17379994_6 | 17379995_6 |
| N = 1,000 seed 1 | 7 | 17379992_7 | 17379993_7 | 17379994_7 | 17379995_7 |
| N = 1,000 seed 2 | 8 | 17379992_8 | 17379993_8 | 17379994_8 | 17379995_8 |
| N = 1,000 seed 3 | 9 | 17379992_9 | 17379993_9 | 17379994_9 | 17379995_9 |
| N = 1,000 seed 4 | 26 | 17379992_26 | 17379993_26 | 17379994_26 | 17379995_26 |
| N = 1,000 seed 5 | 27 | 17379992_27 | 17379993_27 | 17379994_27 | 17379995_27 |
| N = 1,000 seed 6 | 28 | 17379992_28 | 17379993_28 | 17379994_28 | 17379995_28 |

**State at 02:30 Eastern.** The nine stage-3 tasks were pending on priority, with start estimates from Slurm
of 02:58-03:01. The 27 tasks of stages 4-6 were pending on their dependencies. No task had started, so no log
of a real stage has been read yet.

**Cost and time.** N = 1,000: 6 networks × 4 stages × 5.5 GPU-h = 132 GPU-h. N = 500: 3 × 4 × 4.2 GPU-h (the
Spock stage-2 wall time) = 50 GPU-h; at the rule entry's 5.3 GPU-h per stage, 64. Total about 182 GPU-h, at
most 196, under the cap of about 200. Wall time per stage: about 5.5 h at N = 1,000 and about 4.2 h at
N = 500, against a limit of 12 h. Four stages in a row take about 22 h at N = 1,000, so if every stage starts
as soon as its parent ends, the last jobs end around 01:00 Eastern on 13 Nov; any wait in the queue between
stages adds to that.

### Where the outputs land and how to read them

Della: `~/trainRNNbrain/data/trained_RNNs_rl_maze_long/Maze_suite/N=<500|1000>_act=relu_cont<3|4|5|6>/s<seed>/`.
Logs: `~/trainRNNbrain_logs/MazeLong.<job>_<index>.out`. None of the folder names
`N=1000_act=relu_cont3..6` and `N=500_act=relu_cont3..6` exists on the Mac, on Spock or in Della's
`trained_RNNs_rl_maze` tree.

Pull back to the Mac (adds only new folders; the Adam state and the PNGs are left on Della):

```
rsync -a --ignore-existing --exclude='*AdamState_Maze.pt' --exclude='*.png' \
  --include='N=1000_act=relu_cont[3-6]/***' --include='N=500_act=relu_cont[3-6]/***' --exclude='*' \
  della:trainRNNbrain/data/trained_RNNs_rl_maze_long/Maze_suite/ \
  ~/Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/
```

On the Mac, `common.maze_load_runs(".../Maze_suite")` joins each network's six folders into one record of
updates 1-12,000. Each new config records `paths.init_from` as a Della path; `common.maze_run_chain` then
finds the parent under the Mac root by the path's last two components (`<cell>/s<seed>`), as it does for the
Spock paths of stage 2.

**If a stage fails.** The task of the next stage for that network stays pending with
`DependencyNeverSatisfied` and has to be cancelled by hand; the other networks are not affected. The launcher
refuses to write into a folder that already holds a run, so a failed stage's partial folder must be moved
before that stage is resubmitted. I did not check which exit code a run stopped by the rollback guard
returns, so read each log's last lines (guard restores, a stop) before reading a record.
