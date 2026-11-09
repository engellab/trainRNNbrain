## 2026-11-09 15:13 — Round 19: Maze read-out dropout and rescaling rerun with freeze_after_end=true (rules, before submission)

**Why.** Entry `2026-11-08_0953_round19-freeze-default-rules.md` ("Known caveat, not rerun now") lists Maze arms
whose per-unit rules were driven by statistics that included post-episode steps, and which were never rerun with
`trainer.freeze_after_end=true`: read-out dropout (dropout0.2_beta4, dropout0.05_beta4), rescaling (rescale,
rescale_mild, rescale_few, rescale_tgt4) and re-initialisation (redraw, zeroout). The paper's headline
(prune-and-duplicate is the only technique that meets the criterion on all three tasks, read at update 2,000 on
Maze, decision dead-relu-paper#185) carries that caveat, and the independent review of PR #207 named it. Read-out
dropout already meets the criterion on 3BFF (rate 0.05) and CDDM (rate 0.2) under the one-value rule, so a Maze
pass would change the headline. Tracking: ptolmachev/dead-relu-paper#201.

**What the switch changes for these rules.** With `freeze_after_end=true` an ended episode's state is held at its
value at the end, in the rollout and in the PPO recompute. The rule statistics are computed as before, over every
RNN step of the batch: the dropout participation EMA (`TrainerPPO.rollout`, `p_bar`) and the rescale rule's
participation, silent test and graduation statistic (`prune_and_reinit`, `frm_activity`). The post-episode rows
therefore still enter those statistics, but as copies of the state at the episode's end, not as free-running
dynamics. Without the switch those dynamics reached |x| of about 1e21 in one met0.01 episode while every running
episode stayed below 2 (entry `2026-11-08_0943`). This is the same treatment the prune-and-duplicate rerun had
(entry `2026-11-09_0136`), on which the headline now rests.

### What runs

Six arms, seeds 1-3, N = 1,000, 2,000 updates (stage 1), every arm key as in the original runs plus
`trainer.freeze_after_end=true`. Launcher `slurm/rl_maze_interventions_della.slurm` (wrapping
`rl_maze_interventions_spock.slurm`) from the reruns' pinned worktree on Della, `~/trainRNNbrain_r19fz` at
fd564f2, clean. Its trainer code equals main's (`git diff --stat fd564f2 HEAD` touches only
`configs/trainer/trainer_ppo.yaml`, where `freeze_after_end` became true by default, and the launcher's
continuation check). Output under `~/trainRNNbrain/data/trained_RNNs_rl_maze_fz/Maze_interventions/`, the tree of
the other fz reruns.

| Priority | Arm | Indices | The arm's rule | Verdict at 2,000 before the switch (per seed: active units, success) |
|---|---|---|---|---|
| 1 | dropout0.2_beta4 | 19-21 | read-out dropout, rate 0.2, β 4 (the matched value) | fails: 413, 461, 436 active; success 0.84, 0.79, 0.76 (mean 0.104 below the controls') |
| 1 | dropout0.05_beta4 | 121-123 | read-out dropout, rate 0.05, β 4 (3BFF's passing value) | fails on the count: 455, 373, 420 active; success 0.93, 0.96, 1.00 |
| 2 | rescale | 22-24 | rescaling at the paper's values (target 10 × the rate cap; the matched value) | no read: all three stopped by the rollback guard at updates 52-60 |
| 2 | rescale_mild | 25-27 | the same with α 1.0005 and cap 8 | seed 1 stopped at 800; 314 and 321 active at success 0.75 and 0.63 |
| 2 | rescale_few | 28-30 | the paper's values with at most 5% of units growing at once | 255, 231, 234 active; success 0.66, 0.75, 0.81 |
| 2 | rescale_tgt4 | 124-126 | the paper's values with target 4.0 (3BFF's passing value) | seed 2 stopped at 800; 352 active at 0.81 and 197 at 0.01 |

The before-switch numbers are from `data/round19/round19_results.json`. That file quotes, for dropout0.05_beta4,
the verdict string of dropout0.2_beta4 (its reader falls back to the class key `dropout`). The verdict in the
table is computed from dropout0.05_beta4's own numbers: seeds 2 and 3 are not above 449.

**Skipped: re-initialisation (redraw, zeroout).** Re-initialisation is not one of the six techniques of the
paper's cross-task comparison (Results 5, Fig. 5). It fails on both supervised tasks: on 3BFF the redrawn networks
have 205-274 active units against the controls' 236-288, and on CDDM the re-initialised networks have 330-346
against 332-340 (status board, C6 summary). A Maze pass
would therefore change no cross-task claim on the status board, and its Maze verdict keeps the caveat. One detail
for whoever decides later: before the switch, redraw failed on one seed by 2 units (447 active against the bar of
449; the other seeds had 457 and 483, success 0.98-1.00). A rerun would cost about 16 GPU-h at stage 1.

**Why all four rescaling arms.** Rescaling meets the criterion on 3BFF at one of 26 values (target 4.0, protect
0.20; R² 0.0097 below the controls, 0.0003 inside the 0.01 tolerance). Under the one-value rule any Maze value that
works changes the Results 5 sentence "noisy synapses and rescaling meet it in 3BFF networks only". The arm
`rescale` carries the matched value, so it alone can change the matched-values sentence "rescaling in none".
Rescaling fails on CDDM at both values run (target 10: 169-196 active against 333-342; target 4.0: 224-231), so
no Maze result can make rescaling meet the criterion on all three tasks.

### Read-out (set before any job of this entry runs)

- **Learned:** `common.maze_learned_index`: clean probe success at or above 0.9 for five probes in a row, with the
  last probe also at or above 0.9.
- **Update 2,000, the stated read (decision #185):** issue #5's criterion as written (decision #194), against
  the three uniform-series ReLU N = 1,000 controls (426, 449 and 415 active units; success 0.99, 0.98, 0.73;
  mean 0.898). An arm works when every seed has more than 449 active units, the arm's mean success is no more
  than 0.05 below 0.898, and no seed ends below 0.9.
- **Update 4,000, for arms continued:** the part 2 rules (`2026-11-07_1111_round19-settings-stage2-rules.md`)
  against ReLU N = 1,000 seeds 1-6 at 4,000: every seed above 404 active units, mean success within 0.05 of 1.00,
  no seed below 0.9.
- **Stopped networks:** an arm with a seed whose record ends before the read point has no verdict there
  ("incomplete"). An incomplete arm does not meet the criterion at that point. Stopped networks are not restarted.
- **Reader:** `trainRNNbrain/experiments_and_analysis/round19_read_fz.py`, unchanged; it reads every arm in the fz
  tree and applies the criterion above at 2,000 and `round19_read.read_settings` at 4,000. As a check, its numbers
  are compared with each run's training log at the last probe.

**Continuation (the round 19 "one stage, report both points" practice, entries `2026-11-08_1758` and
`2026-11-09_1221`).** An arm with any non-stopped seed not learned at update 2,000 continues one stage, to 4,000,
with every non-stopped seed of the arm, `INIT_ADAM=false` and `FREEZE=true` (the parents' setting). The rule is
the same for all six arms. Each continued arm is reported at 2,000 and 4,000. The 2,000 read stays the stated
one: a pass at 4,000 alone is reported as such and changes no verdict in the paper, as with the rate penalty at
0.3 (entry `2026-11-09_1221`).

### What each outcome changes in the paper

| Outcome at update 2,000 | Change |
|---|---|
| dropout0.05_beta4 or dropout0.2_beta4 works | Under the one-value rule, read-out dropout meets the criterion on all three tasks (3BFF at rate 0.05 with β 1, 2 and 4 and rate 0.10 with β 1; CDDM at rate 0.2 with β 4; Maze at the passing value). "Prune-and-duplicate is the only technique that meets the criterion on all three tasks, under both rules" becomes false under the one-value rule, and the one-value tolerance range for "only" (0.0032-0.0161) is recomputed. Under the matched-values rule (rate 0.2, β 4 on every task) read-out dropout still fails on 3BFF (R² 0.017 below the controls at 75,000 iterations), so the matched-values headline stands. |
| both dropout arms fail | Read-out dropout's Maze verdict "fails" stands on runs with the switch on. The caveat is removed for read-out dropout, and the headline stands under both rules. |
| any rescaling arm works | Under the one-value rule rescaling meets the criterion on 3BFF and Maze. If `rescale` works, the matched-values sentence becomes "rescaling in Maze networks only". The headline is unchanged, because rescaling fails on CDDM. |
| every rescaling arm fails or is incomplete | Rescaling's Maze verdict "fails" stands on runs with the switch on; the caveat is removed for rescaling. |

The status board is updated in every case. A difference between the old and the new runs is reported as the effect
of running with the switch on; no mechanism is inferred from it.

### Smoke tests (local, before submission)

A clean detached worktree at fd564f2 on the Mac; `module`, `conda` and `srun` stubbed, with `srun` appending 2
updates of 4 episodes × 6 decisions; `FREEZE=true`, seed 1 of each arm (indices 19, 121, 22, 25, 28, 124). Pass
condition, set before the diffs: each saved config differs from the original arm's seed-1 config only in
`paths.save_to`, `trainer.freeze_after_end` (absent → true), the stub's keys, and keys added to `trainer_ppo.yaml`
after the original ran whose options are off. `rescale` has no saved original (its runs stopped before writing),
so it was compared with `rescale_few` seed 1, expecting also `experiment_tag` and `rescale_protect_frac` 0.05 →
0.20.

| Index | Arm | Result | Keys that differ besides save_to, freeze_after_end and the stub's |
|---|---|---|---|
| 19 | dropout0.2_beta4 | exit 0 | added later and off: the `prune_args` block with `prune_reinit` false, `lambda_met` 0, `lambda_rwm` 0 with `rwm_args`, `optimizer` adam, `momentum` 0, `checkpoint_keep` false, `iter_offset` 0 |
| 121 | dropout0.05_beta4 | exit 0 | none |
| 22 | rescale | exit 0 | `experiment_tag`, `rescale_protect_frac` 0.05 → 0.20; added later and off as for index 25 |
| 25 | rescale_mild | exit 0 | added later and off: `prune_args.copy_noise` and `copy_norm_preserve` (used only in mode copy), `lambda_met`, `lambda_rwm` with `rwm_args`, `optimizer` adam, `momentum` 0, `checkpoint_keep` false |
| 28 | rescale_few | exit 0 | as for index 25 |
| 124 | rescale_tgt4 | exit 0 | none |

All pass. A stage-2 continuation of index 121 from its smoke output (`PARENT_ROOT`, `EXTRA_UPDATES=2`,
`INIT_ADAM=false`, `FREEZE=true`) ran updates 3-4 with `freeze_after_end` true; its config differs from the parent's
only in `experiment_tag` (`_cont`), `paths.init_from` and `trainer.iter_offset` (0 → 2). The same continuation of
index 124 with `FREEZE=false` exits 1 (the parent ran true).

### Commands (Della)

```
R=$HOME/trainRNNbrain_r19fz      # fd564f2, clean
REPO=$R SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_fz FREEZE=true \
  sbatch --job-name=R19fzDrop --array=19-21,121-123 --time=12:00:00 $R/slurm/rl_maze_interventions_della.slurm
REPO=$R SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_fz FREEZE=true \
  sbatch --job-name=R19fzResc --array=22-30,124-126 --time=12:00:00 $R/slurm/rl_maze_interventions_della.slurm
```

Stage 2, for the arms the continuation rule selects:

```
REPO=$R SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_fz FREEZE=true INIT_ADAM=false EXTRA_UPDATES=2000 \
  PARENT_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_fz/Maze_interventions \
  sbatch --job-name=R19fzC2 --array=<indices> --time=12:00:00 $R/slurm/rl_maze_interventions_della.slurm
```

**Cost.** Stage 1: read-out dropout 6 runs × about 6 h (the round 18 dropout runs took 5.0-6.1 h on Della) = 36
GPU-h; rescaling 12 runs × about 5.3 h = 64 GPU-h; about 100 GPU-h in all, less for runs the rollback guard stops.
Stage 2, if every arm continues: about 100 GPU-h more. At most about 200 GPU-h, the package limit. Job ids are
recorded in a follow-up entry.
