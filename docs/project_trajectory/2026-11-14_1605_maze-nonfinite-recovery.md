# Maze non-finite stops: diagnosis, a recovery fix (branch wp/maze-finite), its tests, and the resumed networks

Follows `2026-11-14_1107_rules-maze-flat-return-by-stage-and-stopped-networks.md` (the rule for resuming
the stopped networks). The pass conditions in section 3 were committed to main before any of those tests
was run; the results were added to this file afterwards (git history of this file shows both).

**Terms.** *Maze*: the egocentric maze task trained by PPO (proximal policy optimisation). *Update*: one
rollout of 256 episodes (up to 500 decisions each, 5 RNN steps per decision, so up to 2,505 RNN steps)
followed by 8 optimizer steps (4 epochs x 2 minibatches of 128 episodes). *Non-finite*: infinite or NaN
(not a number). *Adam moments*: the two running averages of the gradient (`exp_avg`) and of its square
(`exp_avg_sq`) that the Adam optimizer keeps per weight. *Snapshot*: a stored copy of the weights the
trainer falls back on. *Checkpoint*: every 100th update, when the run folder is rewritten.

## 1. Diagnosis

### What the trainer at 1a80f5e does (read from the code, then checked)

- After every update, `good` is set to the current weights. `safe_rollout` first replays from `good`, but
  `good` is the current weights, so the first three "restores" only replay with fresh noise.
- At every checkpoint, `snapshot` is set to `good`, without checking that those weights give a finite
  rollout.
- In the snapshot loop of `safe_rollout`, the first try still rolls out the current weights, so the
  snapshot gets two tries, not three. A record with `restored = 4` means: four failures from the current
  weights, then a finite rollout from the snapshot.
- Adam's moments are never restored.
- `ppo_update` skips a minibatch whose loss is not finite, but applies a step whose loss is finite and
  whose gradient is not.

### The three stops

Data: the stopped stages' folders copied from Della (`data/trained_RNNs_rl_maze/Maze_suite/`).

| Network | Stage, job | Records at the end of the stage | Parameter file written by the last checkpoint |
|---|---|---|---|
| N = 1,000 seed 2 | 3, 17379992_8 | no restore and no skipped step in updates 4,001-5,900; update 5,900 has an approximate KL divergence of 0.19 against about 0.01 before; the run stopped in update 5,901 | finite, the weights after update 5,900 |
| N = 1,000 seed 6 | 5, 17379994_28 | update 8,434: 5 of 8 steps skipped, then a restore at 8,435 that worked; 8,463: 1 skipped; 8,464-8,500 (37 updates): every update `restored 4`, `skipped 7` | every entry of W_rec, W_inp and W_out NaN (update 8,500) |
| N = 500 seed 2 | 6, 17379995_5 | 10,833: 2 skipped; 10,834-10,900 (67 updates): every update `restored 4`, `skipped 7` | every entry NaN (update 10,900) |

**N = 1,000 seed 2: the snapshot was the unstable weights.** Update 5,900 is a checkpoint, so `good` and
`snapshot` were both the weights after update 5,900. Rolling those weights out on the Mac (CPU, the
1a80f5e trainer, fresh noise) gave a non-finite rollout 4 times in 4. In each rollout 1 to 3 of the 256
episodes diverged. Every diverging episode was one that had not yet reached the cheese. Its largest state
grew about 1,000-fold every 25 decisions and overflowed between RNN steps 1,600 and 2,100. In the other
episodes the largest |state| stayed below 5. All six tries of the stop rolled out these weights, so the
stop is reproduced: no restore could help, because the fallback was the same weights.

**N = 1,000 seed 6 and N = 500 seed 2: Adam's moments became NaN.** In every looping update the
snapshot's rollout was finite, one optimizer step was applied, and the other seven minibatch losses were
non-finite. The checkpoint at the end of each loop wrote all-NaN weights. A step with finite gradients and
finite moments cannot write NaN into every weight, so the moments were NaN. Once they are NaN, every step
writes NaN into every weight, whichever weights were restored (checked in torch: a NaN gradient entry makes
`clip_grad_norm_` multiply every gradient by NaN, the Adam step makes both moments and every weight NaN,
and a later step with a clean gradient writes NaN again). The loop ended at the next checkpoint, which
copied the NaN weights into `snapshot`; one update later all six tries failed.

**How the moments became NaN: a finite loss with a non-finite gradient.** The log-ratio clamp at ±20
keeps the PPO loss finite when an episode's recomputed log-probability is minus infinity. Test on real
weights: a batch was rolled out with the update-4,000 weights of N = 1,000 seed 2, and the PPO loss of
1a80f5e was evaluated at that network's update-5,900 weights. Minibatches containing an episode that
overflowed had a NaN loss (3 of 6; correctly skipped). Episodes whose states rose above 1e19 but stayed
finite gave a finite loss and a finite gradient. Then each overflowing episode was cut to T' decisions so
that its overflow fell at or just after the end of the sequence (the recompute is causal, so the earlier
states are unchanged); this is an episode that diverges late. In 15 of the 16 cuts with a finite loss the
gradient was not finite. The exact operation that produces the non-finite gradient was not isolated.

**Verified versus inferred.**
- Verified: the seed 2 stop (its weights, its snapshot, non-finite rollouts 4 of 4); the NaN-moment loop
  and its end at a checkpoint, from the records and the parameter files; a finite loss with a non-finite
  gradient exists on this network's real weights; torch's clip-and-step turns one NaN gradient entry into
  NaN everywhere.
- Inferred, not verified: that a finite-loss, non-finite-gradient step is what poisoned the moments at
  updates 8,463 (seed 6) and 10,833 (N = 500 seed 2). The AdamState files written at those stages'
  last checkpoints, on Della, would show NaN moments if so (check in section 4).
- Not the cause: the spectral radius of W_rec. The stable update-4,000 weights of seed 2 have 6.1, the
  unstable update-5,900 weights 5.0.

## 2. The fix (branch `wp/maze-finite`, from 1a80f5e; only `TrainerPPO.py`, `trainer_ppo.yaml` and tests)

1. `ppo_update`: a step whose clipped gradient norm is not finite is skipped and counted in a new
   statistic, `skipped_grad`.
2. A snapshot is the weights and the optimizer state together (`capture_state`), taken at the start of
   the run and at every checkpoint. It is kept only if the next rollout from those weights is finite on
   its first try and every tensor in it is finite.
3. Up to `snapshot_keep` verified snapshots are kept. `safe_rollout` replays from the current weights up to
   `rollout_tries` times, then restores the newest snapshot not yet used, weights and optimizer state
   together, and gives it `rollout_tries` tries; then the next older one. A restored snapshot is marked
   used until a newer one is verified. With no unused snapshot left the run stops, as before. After a
   restore the learning rate is set back to the current update's.
4. New config keys, `configs/trainer/trainer_ppo.yaml`: `rollout_tries: 3` (the old hard-coded value) and
   `snapshot_keep: 3`. New progress keys per update: `skipped_grad` (0 normally), `snapshot_restores` (0
   normally), `snapshot_from` (the absolute update of the restored snapshot, -1 when none). `restored`
   keeps its meaning: the number of non-finite rollouts discarded.

None of these changes draws a random number or changes a computed value while every rollout and gradient
is finite. Nothing is clipped, and the activation, the learning rate and the loss are unchanged.

## 3. Tests, with pass conditions set before running them

**P1, bit-identity, small.** `run_experiment.py` with the 1a80f5e code (an export of that commit) and with
the new code, the same overrides: N = 50, seed 3, 8 episodes of 30 decisions, 12 updates, checkpoint every
5 updates, probe every 5. Pass: in every progress record, every key of the 1a80f5e record has an equal
value (`==` on the floats) in the new record; the new keys are `skipped_grad = 0`, `snapshot_restores = 0`,
`snapshot_from = -1` throughout; every array of `Participation.npz` and `Params_Maze.npz` is equal
element for element; every tensor of the AdamState file is equal. Any difference fails.

**P2, bit-identity on the production input.** The same comparison for N = 1,000 seed 2 continued from its
real stage-2 folder `N=1000_act=relu_cont/s2` (update 4,000), full size (256 episodes, T = 500), fresh Adam
and warm-up as in the stages, 3 updates, `checkpoint_every=1` and `probe_every=1` so the snapshot code runs
at every update. Same pass rule.

**P3, the seed 2 stop.** Start state: the weights after update 5,900. 1a80f5e's `safe_rollout` with `good`
and `snapshot` both set to those weights (the state at the stop). Pass for the reproduction: it raises
`RuntimeError`. New trainer: the same current weights and one verified snapshot, the update-4,000 weights
of the same network (the nearest finite state on disk), then `run_training` for 3 updates. Pass: update 1
records `restored = 3`, `snapshot_restores = 1`, `snapshot_from = 4000`; all 3 updates complete; the
final weights and moments are finite; no snapshot with update 5,900 or its weights is ever kept.

**P4, the poisoning, on real weights.** A batch made of the overflowing episodes from section 1, cut so
the loss is finite and the gradient is not. One `ppo_update` (1 epoch, 1 minibatch) at the update-5,900
weights with each trainer, from the same Adam state. Pass: with 1a80f5e every entry of W_rec, W_inp and
W_out is NaN afterwards, and both Adam moments are NaN; with the new trainer `skipped_grad = 1`, and the
weights and moments are unchanged bit for bit. If 1a80f5e's weights stay finite, the poisoning mechanism
is not shown on real weights, and this entry says so.

**P5, unit tests.** `tests/test_maze_nonfinite_recovery.py` (4 checks: the skipped step and what the
unguarded step would do; unstable weights never kept as a snapshot; NaN moments replaced by the snapshot's
on a restore, with the current learning rate; used snapshots skipped and the stop when none is left) and
the whole of `tests/test_rl_maze.py` and `tests/test_no_function_imports.py` pass.

**P6, launcher and reader.** The new launcher run locally with `module`, `conda` and `srun` stubbed, with
tiny overrides, on copies of the real parents. Pass: the first resumed stage of each network reads its
parent from the old tree and writes `N=<N>_act=relu_resume_cont<k>/s<seed>` into the new tree; the next
stage reads from the new tree; the saved config has `rollout_tries: 3`, `snapshot_keep: 3` and differs from
the parent's otherwise only in the keys the stage launcher changes; the launcher refuses an existing
folder; with the two trees copied into one folder, `common.maze_load_runs` reads each resumed network as
one run with updates 1 to the last, without gap or repeat.

## 4. Results

Code: commit `8c6b36e9ea20c449aad550ac1b544d318b8c0631` on branch `wp/maze-finite` (pushed to origin), whose
parent is 1a80f5e. The test scripts are in the session's scratch folder and are not committed; each is
described here.

| Test | Result |
|---|---|
| P1 | **Pass.** 12 records, 144 values equal; new keys at 0, 0, -1 throughout; Participation, Params and AdamState equal. Controls: two runs of the 1a80f5e code agree (pass); a run with the learning rate changed by 0.01% fails with 47 differences, so the comparison can fail. Re-run at commit 8c6b36e (which changed only docstrings after the first run): pass. |
| P2 | **Pass.** N = 1,000 seed 2 from its real update-4,000 folder, full size, 3 updates (4,001-4,003), snapshot capture at every update: 36 values equal, weights and Adam state equal. The saved config records `rollout_tries: 3`, `snapshot_keep: 3`. |
| P3 | **Pass.** 1a80f5e: `RuntimeError: the rollout was not finite 6 times in a row`, so the stop is reproduced. New trainer: update 5,901 has `restored 3`, `snapshot_restores 1`, `snapshot_from 4000`; updates 5,901-5,903 complete with probe success 1.00 and success 0.98-0.99; weights and moments finite; the only snapshot kept is update 4,000, and none holds the update-5,900 weights. |
| P4 | **Pass on the fourth run; the first three failed in the test script, before any trainer step was compared** (details below). With 1a80f5e, one update on a minibatch with a finite loss (0.35) and a non-finite gradient left every entry of W_rec, W_inp and W_out NaN and both Adam moments NaN, and the update recorded `skipped 0`, like updates 8,463 and 10,833 of the stopped runs. With the new trainer: `skipped_grad 1`, weights and moments unchanged bit for bit. |
| P5 | **Pass.** `tests/test_maze_nonfinite_recovery.py`: 4 of 4. `tests/test_rl_maze.py`: all 36 checks, exit 0. `tests/test_no_function_imports.py`: 221 files clean. |
| P6 | **Pass.** Launcher runs (Mac, stubs, real parents copied): index 8 stages 3 and 4, index 28 stage 5, index 5 stages 6 and 7 exit 0, each first stage reading the old tree and each later stage the new one; refusals for an existing folder, a stage below the first (28, stage 4), a stage above 10, and an index not resumed (9). Saved configs differ from the stopped stages' configs only in `experiment_tag`, `paths.init_from`, `paths.save_to`, `trainer.iter_offset` (later stages), the stub's keys and the two new keys. Reader: in one folder with the local chains, the stopped partial folders and the resumed folders, with the recorded `init_from` paths no longer on disk, `common.maze_load_runs` reads each resumed network as one run (N = 1,000 seed 2: updates 1-4,004 through 4 folders; seed 6: 1-8,002 through 5; N = 500 seed 2: 1-10,004 through 7), every step +1. Each stopped chain stays a record of its own (ending at 5,900, 8,500, 10,900). |

**The three failed P4 runs (test script, not trainer).** (1) The advantages were computed from the one-episode
batch, whose Monte-Carlo baseline equals the episode's own return, so every advantage was 0; in training,
advantages come from the 256-episode rollout. Fixed by slicing the full-batch advantages. (2) A clean update,
run first so that Adam's moments exist, moved the weights by one Adam step after the overflowing episodes had
been located; (3) located again after that step, no episode of the batch overflowed any more: one Adam step
(about 5e-4 per weight) moved the network out of the unstable region. Fixed by loading the update-5,900
weights again after the clean update, keeping the moments it created. The pass condition was not changed.

**Not tested.** A full 2,000-update stage with the new trainer, on any device. The bit-identity was checked on
the Mac's CPU (torch 2.10); Della runs torch 2.7 on a GPU, where the added code only reads the gradient norm
and copies state, but no comparison was run there. No real run has yet met a non-finite event with the new
trainer; P3 and P4 are built from saved weights. That a non-finite gradient poisoned the moments at updates
8,463 and 10,833 is still an inference: step 2 of the submission script prints the share of NaN entries in the
AdamState files written at those stages' last checkpoints. If the explanation is right, the seed 6 and N = 500
seed 2 files are entirely NaN and the seed 2 file has none.

**The same change on main.** Main's TrainerPPO has the same `good`/`snapshot` recovery, plus the
`freeze_after_end` option and the `restore_kinds` bookkeeping inside `safe_rollout` and `run_training`. The
gradient guard in `ppo_update` applies without conflict; `safe_rollout` and `run_training` need a hand merge
that keeps `restore_kinds`. Not applied to main.

## 5. Deployment and submission: not done, Della refused the connection

`ssh -o BatchMode=yes della true` failed (Duo session expired) at 16:05 and again after the tests. Everything
is prepared so that submission is one command once Pavel has run `ssh -fN della`:

```
ssh della 'bash -s' < slurm/rl_maze_resume_submit_della.sh
```

The script (committed with this entry) makes the new worktree `~/trainRNNbrain_mazefinite` at 8c6b36e
(from `~/trainRNNbrain_mazelong`, whose working tree and jobs it does not touch), copies the launcher
`slurm/rl_maze_resume_della.slurm` from origin/main into it, prints the AdamState check, refuses if any
resumed folder exists, runs the launcher once on the login node with `srun` stubbed (index 8, stage 3,
output under /tmp), and submits 19 tasks as three chains, each stage one task with
`--dependency=aftercorr:<previous stage>`, job name MazeResume, `INIT_ADAM=false` (fresh Adam and the
50-update warm-up at every stage, as in every earlier stage):

| Network | Index | From (update) | Stages | Tasks |
|---|---|---|---|---|
| N = 1,000 seed 2 | 8 | `N=1000_act=relu_cont/s2` (4,000) | 3-10 → `N=1000_act=relu_resume_cont3..10` | 8 |
| N = 1,000 seed 6 | 28 | `N=1000_act=relu_cont4/s6` (8,000) | 5-10 → `N=1000_act=relu_resume_cont5..10` | 6 |
| N = 500 seed 2 | 5 | `N=500_act=relu_cont5/s2` (10,000) | 6-10 → `N=500_act=relu_resume_cont6..10` | 5 |

Outputs: `~/trainRNNbrain/data/trained_RNNs_rl_maze_long_resumed/Maze_suite/`. Cost: 14 stages at
N = 1,000 x about 5.5 GPU-h + 5 at N = 500 x 4.2-5.3 GPU-h = about 100 GPU-h. Wall time if every stage starts
when its parent ends: 8 stages x 5.5 h = about 44 h for N = 1,000 seed 2, about 33 h for seed 6, about 21-27 h
for N = 500 seed 2.

**Pull to the Mac** (adds new folders only; the parents are already in the local folder):

```
rsync -a --ignore-existing --exclude='*AdamState_Maze.pt' --exclude='*.png' \
  --include='N=1000_act=relu_resume_cont*/***' --include='N=500_act=relu_resume_cont*/***' --exclude='*' \
  della:trainRNNbrain/data/trained_RNNs_rl_maze_long_resumed/Maze_suite/ \
  ~/Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/
```

There, `common.maze_load_runs` returns two records for each of the three networks: the stopped chain (ending at
5,900, 8,500 or 10,900; `maze_series` lists it as unfinished) and the resumed chain (to 20,000). A reader marks
a record as resumed by `_resume_` in its folder name, as the rule entry requires. Every recovery event is in the
resumed stages' progress records: `restored`, `snapshot_restores`, `snapshot_from`, `skipped_grad`.
