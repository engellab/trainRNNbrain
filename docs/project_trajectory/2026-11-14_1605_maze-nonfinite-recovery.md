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

(added after the tests ran)
