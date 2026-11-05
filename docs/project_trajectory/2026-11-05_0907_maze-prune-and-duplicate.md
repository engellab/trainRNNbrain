## 2026-11-05 09:07 — Maze: prune-and-duplicate as a TrainerPPO option, the paper's copy rule mirrored from the supervised trainer (issue #6; launcher arm 17, indices 52-54)

Code commit f7a33f6 on `main` (trainer, config, checks, launcher); this entry follows it. Nothing
submitted: Spock is unreachable from this Mac, and the submission is the coordinator's.

**Why.** Issue #5 listed prune-and-duplicate as the one Figure 5 per-unit rule TrainerPPO did not
implement ("not built"). Pavel's instruction of 5 Nov (issue #6) is the same rules on all three
tasks, so the maze gets the paper's duplication arm, the rule that recruits the most units on the
3-bit flip-flop (2.77 times the control's active count at N = 1,000; `docs/paper_audit.md`,
remedies catalogue).

**What was built.** `TrainerPPO.duplicate_units` and the copy branch of
`TrainerPPO.prune_and_reinit` (`trainRNNbrain/trainer/TrainerPPO.py`) are the supervised
`Trainer.prune_and_reinit_` in reinit_mode "copy", line for line, under the same keys
`trainer.prune_reinit` and `trainer.prune_args` (`configs/trainer/trainer_ppo.yaml`, which gains
`copy_noise` and `copy_norm_preserve` at the supervised defaults, 0.05 and false). The head of the
check is the one the rescale rule already had: participation p = std + q0.9 of the rate on the
states recomputed for the minibatch; silent if p < 0.05 x q95(p); the contribution utility u <-
0.99 u + 0.01 p x sum|outgoing weight|; strikes while silent, reset otherwise; doomed = strikes >=
patience and mature; at most round(max_replace_frac x N) doomed per check, lowest score first. Then,
for every doomed unit in ascending order: a donor is drawn from the live units with replacement, in
proportion to participation, with the RNN's generator; the donor's rows W_rec[j, :], W_inp[j, :]
and its self-weight are read; the donor's outgoing columns W_rec[:, j] and W_out[:, j] are halved and
written to the copy's columns, so every downstream unit receives the same drive as before the copy;
the copy's incoming rows are the donor's with every entry multiplied by (1 + copy_noise x N(0, 1))
and, with copy_norm_preserve, scaled back to the donor's row norms (the jitter then sets the copy's
direction only, at an expected cosine of 1 / sqrt(1 + copy_noise^2) = 0.71 to the donor at the
paper's copy_noise 1.0); the 2 x 2 block of W_rec spanning the pair is set to half the donor's
self-weight in all four entries (with self-connections, the standard model's case; zero without).
Adam's moments on the copies' incoming rows (W_rec, W_inp) and outgoing columns (W_rec, W_out) are
set to 0; the donors' halved columns keep theirs, as in the supervised code. strikes[doomed] = 0,
last_replaced[doomed] = the step. Refused at construction, as the other unmirrored arms are:
copy_iid and copy_permute (the screen's variants of the copy).

**The clock.** The rule runs where the rescale rule runs, once per optimizer step inside
`ppo_update`, so check_every, patience and maturity count optimizer steps (8 per update). At the
paper's values (check_every 100, patience 3, maturity 1,000) a 2,000-update run makes 160 checks
(the flip-flop arm at 40,000 iterations makes 400), a unit is doomed after 300 silent steps (37.5
updates) and is protected for 125 updates after a replacement; at most round(0.025 x 1,000) = 25
units are replaced per check.

**Progress.json** gains, only with the copy rule on, two integers per update: `copy_replaced`
(replacements made, summed over the update's checks) and `copy_units_ever` (distinct units replaced
so far, after the last check; the supervised trainer's `reinit_units_ever`, which with the event
count detects a treadmill: units replaced again and again with the active count unmoved). The probe
line printed during training shows `replaced <copy_units_ever>`.

**Launcher.** `slurm/rl_maze_interventions_spock.slurm`, arm 17 `duplicate`, indices 52-54, the
array widened to 1-54. The swept keys are the paper's duplication arm
(`slurm/SilentReLU_paper_grid_della.slurm` lines 126-133, with the cadence that arm inherits from
`trainer_ptrack_freshbatch.yaml`): `trainer.prune_reinit=true trainer.prune_args.reinit_mode=copy
trainer.prune_args.copy_noise=1.0 trainer.prune_args.copy_norm_preserve=true
trainer.prune_args.max_replace_frac=0.025 trainer.prune_args.maturity=1000
trainer.prune_args.utility=contribution trainer.prune_args.check_every=100
trainer.prune_args.patience=3`. Output `Maze_interventions/duplicate/s<seed>` under the maze suite's
tree. To run, from a worktree on Spock at commit f7a33f6 or later:
`REPO=$HOME/trainRNNbrain_rlmaze<N> sbatch --export=ALL --array=52-54 slurm/rl_maze_interventions_spock.slurm`.
Cost: 3 runs at about 5.5 h each on a 40 GB card (160 checks per run add little), about 17 GPU-h.
Read-out as issue #5's pre-registration: "works" = more active units than every learned control
(449 is the largest) in all three seeds at an end success within 0.05 of the learned controls' mean.

**Checks**, `tests/test_rl_maze.py` 27 of 27 (three new, one widened), against a numpy
re-implementation of the copy (`duplicate_by_hand`) fed the draws replayed from the RNN's saved
generator state:
- `test_copy_rule_duplicates_live_units`: N = 32, three units forced silent (12 silent at this init
  in all), a check every step, patience 1, no ceiling, copy_noise 1.0 with the norm preserved,
  Adam's moments set to 1 by hand. All 12 silent units are overwritten; the replayed donors are live
  (two of them, units 11 and 5, are drawn more than once; the oracle handles a shared donor
  sequentially, as the trainer does); every weight matrix matches the oracle to 1e-6; the copies'
  input rows sit at the donors' row norms at cosines 0.26 to 0.99 to the donor, mean 0.77 (a
  10-entry row, so single pairs vary); for the donors drawn once the copy's and the donor's columns
  over the untouched units are half the donor's old column, the donor's row over those units is
  unchanged and the block spanning the pair is half the old self-weight; the untouched units keep
  their rows and columns; Adam's moments are 0 on exactly the copies' rows and columns and 1
  elsewhere, the donors' halved columns included; the counts read copy_replaced 12,
  copy_units_ever 12; the step clock reads 1.
- `test_copy_ceiling_and_maturity`: at most round(0.0625 x 32) = 2 replacements per check by
  contribution utility (three forced-silent units A, B, C at hand-set utilities 0.3, 0.2, 0.1, every
  other unit at 10 + its index), maturity 4: check 1 replaces C and B, check 2 replaces A (its strike
  from check 1 was kept) and neither B nor C, checks 3 and 4 none of the three.
- `test_copy_counts_in_progress`: `run_training` with a check every 2 steps and patience 2 records
  the two counts as integers in every update, the three forced units replaced in the first update,
  copy_units_ever non-decreasing, no rescale key, the clock at 32 after 4 updates of 8 steps;
  without the option no copy key.
- `test_other_modes_refused` (was `test_rescale_other_modes_refused`): "random" refused in place of
  "copy", which is now accepted; copy_iid and copy_permute refused.
`tests/test_no_function_imports.py` 205 files; `tests/test_scripts_start.py` 52 scripts start.

**Smoke runs through the launcher's python line** (N = 100, seed 1, 3 updates of 16 episodes,
probe 8, warm-up 1, CPU, `paths.save_to` in the scratchpad):
1. The paper's values: the six standard files; the saved config records prune_reinit true and
   reinit_mode copy with copy_noise 1.0, copy_norm_preserve true, max_replace_frac 0.025, maturity
   1000, utility contribution, check_every 100, patience 3; 24 optimizer steps hold one check (step
   0, a first strike), so copy_replaced is 0, 0, 0, as the cadence says; 61 of 100 active at the
   first probe; nothing skipped, no restore; 2.6 s per update.
2. The same with a check every step and patience 1: round(0.025 x 100) = 2 replacements per check,
   8 checks per update, so copy_replaced 16, 16, 6 and copy_units_ever 16, 32, 38 (the third update
   finds fewer silent units outside their maturity); the same first rollout as run 1 and 76 of 100
   active at the first probe, after the first update's 16 replacements, against 61; nothing skipped.

**Not mirrored, and why.** (1) The supervised train_step re-applies structural masks, Dale's sign
and I/O positivity after the rule; TrainerPPO has no post-step projections, so on a model without
self-connections or with Dale's law the copied columns would not be cleaned up. The standard model
needs none, as for the rescale rule. (2) The rule's state (strikes, utilities, the replaced set, the
step clock) is not saved; a continuation restarts it, as the supervised trainer does. (3) The
supervised copy's screen variants copy_iid and copy_permute are refused, as the rescale rule's
unmirrored arms are.
