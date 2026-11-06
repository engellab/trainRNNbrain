## 2026-11-05 23:06 — Maze package 3: the input-weight cap, two penalties, the optimiser switch and the redraw modes mirrored into TrainerPPO (issue #6; launcher arms 25-38, indices 73-114)

Code commit 64c3293 on `main` (trainer, run script, config, two analysis scripts, checks, launcher);
this entry follows it. Nothing submitted: the submission is the coordinator's.

**Why.** `docs/interventions_matrix.md` (section "Maze ... Code first") lists five settings of the
3BFF and CDDM ladders that the maze could not run because TrainerPPO lacked the code: the input-weight
cap, the recurrent-weight magnitude penalty, the metabolic penalty, any optimiser but Adam, and the
re-initialisation of silent units (modes random and zero_out). Pavel authorised the package on 5 Nov
("Go ahead on additional jobs", entry 2026-11-05_2247). The threshold lift (bias_kick) is left out:
the standard network's bias is fixed at zero, so the rule has nothing to act on.

**What was built**, each a mirror of its supervised definition under the same config keys, quoted in
the docstrings of `trainRNNbrain/trainer/TrainerPPO.py`:

1. *The input-weight cap*, `model.inp_weight_cap` (`TrainerPPO.enforce_inp_cap`, the supervised
   `Trainer.enforce_inp_cap_`): `|W_inp|` is clamped to the cap in place after every optimizer step
   of `ppo_update`, after the per-unit rule's check, the supervised order (step, rule, projections,
   cap). The trainer reads the cap from the RNN at construction, as the supervised Trainer reads it
   (`getattr(self.RNN, "inp_weight_cap", None)`), because the key is the model's and the RNN already
   stores it; `None` (every model config's value) leaves the clamp out.
2. *The recurrent-weight magnitude penalty*, `trainer.lambda_rwm` and `rwm_args`
   (`TrainerPPO.rec_weights_magnitude_penalty`, the supervised `Penalties.rec_weights_magnitude_penalty`
   in its branch without Dale's law; an RNN with a `dale_mask` is refused at construction): every
   entry `|W_ij|` of the raw W_rec is compared with the cap `cap100 x log1p(100) / log1p(N)`; an entry
   above the cap pays `(|W_ij| / cap)^gamma - 1`, one below it pays nothing; the mean over the N^2
   entries is multiplied by `N / (N_ref x k_ref)`. `rwm_args` in `configs/trainer/trainer_ppo.yaml`
   spells out the supervised defaults the CDDM arm ran with `+trainer.rwm_args={}`: cap100 0.07,
   N_ref 100, k_ref 20, gamma 5, eps 1e-12. *The metabolic penalty*, `trainer.lambda_met` and
   `met_args` (`TrainerPPO.metabolic_penalty`, the supervised `Penalties.metabolic_penalty`): the mean
   squared firing rate over every unit, RNN step and episode of the states recomputed for the
   minibatch, the tensor the rate penalty uses; `met_args` is `{}`, the penalty takes no argument.
   Both terms enter the loss of every minibatch as the rate and sparsity penalties do, `loss +=
   lambda x P`, are skipped with the minibatch when the loss is not finite, and are recorded in the
   update statistics (Progress.json keys `rec_weights_magnitude` and `metabolic`, the raw values).
3. *The optimiser switch*, `trainer.optimizer` (adam, adamw, sgd) and `trainer.momentum` (sgd only),
   the B5 comparison's keys (`configs/experiment/b5_optimizer_ff.yaml`,
   `slurm/SilentReLU_b5_optimizer_spock.slurm`): `run_rl.build_optimizer` is the switch of
   `run_experiment.py` lines 388-397 with the same comment; the size rule
   `lr x (100 / N)^(1/3)` and the warm-up ramp apply to every optimiser (the trainer reads its base
   rate from `param_groups[0]["lr"]` of whatever it is given). The state file keeps the name
   `s<seed>_AdamState_Maze.pt` whichever optimiser wrote it, so the readers' `*AdamState*.pt` glob
   finds it, and the saved config's `trainer.optimizer` records which; a run saved without the key
   ran Adam. A continuation with `paths.init_adam` must name its parent's optimiser: torch loads an
   Adam state into an SGD optimizer without complaint and the first step then fails on the missing
   `momentum` key (checked under torch 2.10), so `run_rl.py` reads the parent's config and refuses
   the mismatch before training; after the load, lr, weight decay and (for SGD) momentum are set
   back to this run's config. The config key collides with the trainer's own `optimizer` parameter
   when a trainer is rebuilt from a saved config, so `TrainerPPO.load_run`, `fig_supp_tasks.py` and
   `rl_maze_retrain.py` drop the key before the call (`rl_maze_retrain` builds a fresh Adam for the
   retrain whichever optimiser the run used, as its docstring says).
4. *The redraw modes*, `trainer.prune_args.reinit_mode` random and zero_out
   (`TrainerPPO.redraw_units`, the last branch of the supervised `Trainer.prune_and_reinit_`): a
   doomed unit's incoming rows W_rec[i, :] and W_inp[i, :] are redrawn from N(0, 1/sqrt(N)) with the
   RNN's generator, the W_rec rows first; in mode zero_out (the rule of Dohare et al. 2024) its
   outgoing columns W_rec[:, i] and W_out[:, i] are then set to 0; in mode random the outgoing weights
   stay as training left them. Adam's moments are zeroed on the redrawn rows and kept on the
   columns in both modes, as in the supervised code. The head of the check (participation,
   strikes, patience, the contribution utility, the ceiling `max_replace_frac`, maturity) is the
   copy mode's, so the cadence counts optimizer steps (8 per update) as for the duplicate arm.
   Progress.json gains `reinit_replaced` (summed over the update's checks) and `reinit_units_ever`
   (after the last check); the probe line prints `redrawn <reinit_units_ever>`.

**Checks**, `tests/test_rl_maze.py` 33 of 33 (six new, three widened), each against an oracle written
in numpy without importing Trainer.py:
- `test_inp_weight_cap_holds_after_step`: two trainers from one seed, W_inp set to one draw with
  about half its entries beyond 0.3, one with the cap 0.3 and one without, the same rollout and one
  optimizer step; the capped W_inp equals the uncapped one clamped to [-0.3, 0.3] entry by entry
  (1e-7), with entries at the cap on both sides, and W_rec is the same in both.
- `test_rec_weights_magnitude_penalty_matches_formula`: a hand-made W_rec with entries at the cap
  (pays 0), at twice the cap (pays 31) and at three times it; the value matches the formula to 1e-5
  relative; a W_rec with every entry under the cap pays exactly 0; gamma 3 changes the value.
- `test_metabolic_penalty_matches_formula`: a signed state tensor; the mean squared ReLU rate to
  1e-6 relative; all-ones pays 1, all-negative pays 0.
- `test_penalties_in_update_statistics` (widened): all four penalties reported in the update
  statistics equal their direct values on the pre-update recompute; a trainer without them reports
  no such key.
- `test_optimizer_switch`: `build_optimizer` returns Adam, AdamW and SGD (momentum 0.9) at the rate
  given, refuses "rmsprop", and the trainer reads the rate from each; through the script at N = 40,
  an SGD parent at lr 0.1 records 0.068 at update 1 and 0.136 at update 2 (the ramp, then the size
  rule's rate), "sgd" in its config and SGD param groups in its state file; a continuation with
  `init_adam` and SGD runs at 0.136 from its first update (no ramp) and records updates 3 and 4; a
  continuation asking for Adam is refused before training and writes no folder; an AdamW run's
  state file carries `decoupled_weight_decay` true where an Adam run's carries false.
- `test_redraw_modes_rewrite_doomed_units`: N = 32, three units forced silent (the silent set at
  this init is the trainer's), a check every step, patience 1, no ceiling, Adam's moments set to 1
  by hand; in both modes the generator replayed from its saved state gives the two draws, and every
  doomed unit's rows equal the draws times 1/sqrt(32) (1e-7; the sample standard deviation of the
  draw within 20% of 1/sqrt(32)); the untouched block of W_rec and the other W_inp rows are
  unchanged; in mode random the doomed units' columns over the other units and all of W_out are
  unchanged; in mode zero_out those columns are 0 and the other W_out columns unchanged; the
  moments are 0 on the doomed rows of W_rec and W_inp and 1 everywhere else, columns included; the
  counts read `reinit_replaced` = `reinit_units_ever` = the number of silent units.
- `test_rule_ceiling_and_maturity` (was `test_copy_ceiling_and_maturity`, widened to the three
  modes): the ceiling of 2 per check by contribution utility and the maturity protection replace
  the three forced units in the same order in copy, random and zero_out.
- `test_redraw_counts_in_progress`: `run_training` with zero_out, a check every 2 steps and patience
  2 records the two counts as integers in every update, no copy or rescale key, the three forced
  units redrawn in the first update, the clock at 32 after 4 updates.
- `test_other_modes_refused` (widened): orth and bias_kick refused; the two redraw arms accepted.
`tests/test_no_function_imports.py` 206 files; `tests/test_scripts_start.py` 52 scripts start.

**Launcher.** `slurm/rl_maze_interventions_spock.slurm`, arms 25-38 appended after `rws0.05` (arm
24, indices 70-72), the array widened to 1-114, the GRID comment extended; each arm passes only its
swept keys:

| indices | arm | overrides |
|---|---|---|
| 73-75 | cap0.3 | `model.inp_weight_cap=0.3` |
| 76-78, 79-81 | rwm0.05, rwm0.2 | `trainer.lambda_rwm=0.05` / `0.2` |
| 82-84, 85-87, 88-90, 91-93 | met0.01, met0.1, met1, met10 | `trainer.lambda_met=0.01` / `0.1` / `1` / `10` |
| 94-96 | adamw | `trainer.optimizer=adamw` (weight decay the config's 1e-6) |
| 97-99, 100-102, 103-105, 106-108 | sgd0.03, sgd0.1, sgd0.3, sgd1 | `trainer.optimizer=sgd trainer.lr=0.03` / `0.1` / `0.3` / `1` (momentum 0; the rate at N = 1,000 is 0.464 x the value) |
| 109-111 | redraw | `trainer.prune_reinit=true trainer.prune_args.reinit_mode=random trainer.prune_args.max_replace_frac=0.025 trainer.prune_args.maturity=1000 trainer.prune_args.utility=contribution trainer.prune_args.check_every=100 trainer.prune_args.patience=3` |
| 112-114 | zeroout | the same with `reinit_mode=zero_out` |

The values are the supervised ladders of the matrix: CDDM's cap 0.3, rwm 0.05 and 0.2, met 0.01 to
10; the B5 optimiser ladder's AdamW and the four SGD rates that learned the flip-flop (3 and 10 did
not); the paper's re-initialisation arms (`slurm/SilentReLU_replace_variants_della.slurm`, modes
random and zero_out, rate 0.025, maturity 1,000, utility contribution, with the cadence of
`trainer_ptrack_freshbatch.yaml`, check every 100 steps, patience 3, as the duplicate arm). Output
`Maze_interventions/<arm>/s<seed>` under the maze suite's tree. To run, from a worktree on Spock at
commit 64c3293 or later:
`REPO=$HOME/trainRNNbrain_rlmaze<N> sbatch --export=ALL --array=73-114 slurm/rl_maze_interventions_spock.slurm`
(42 runs; about 5.5 h each at N = 1,000 on a 40 GB card, about 230 GPU-h; the 12 h limit holds).
Read-out as issue #5's pre-registration and the matrix: "works" = more active units than every
learned control (449 is the largest) in all three seeds at an end success within 0.05 of the
learned controls' mean; the SGD rates are also read for whether the task is learned at all.

**Smoke runs** (the launcher's python line by hand, its `ARM_TAG` / `ARM_OVERRIDES` arrays sourced
from the file so the strings tested are the launcher's; N = 100, seed 1, 2 updates of 16 episodes,
probe 8, warm-up 1, CPU, `paths.save_to` in the scratchpad): all 14 arms ran, every saved config
records its swept key at the value passed (`trainer.optimizer` sgd in the four SGD arms, adamw in
the AdamW arm), every folder holds the seven standard files, nothing skipped or restored, 2.4-2.5 s
per update. The two penalties are recorded (rec_weights_magnitude 3.2 and 3.1 at update 2 for rwm
0.05 and 0.2; metabolic 0.052, 0.038, 0.019 and 0.014 for met 0.01, 0.1, 1 and 10, the mean squared
rate already lower at the larger weights after 16 steps). The redraw arms hold one check in 16 steps
(step 0, a first strike), so `reinit_replaced` is 0 in both updates, as the cadence says. SGD at
lr 1 (the size rule gives 1.0 at N = 100) has a first-update KL of 1.09 against 0.005 for Adam; at
N = 1,000 the rate is 0.464. The cap: the cap0.3 run's saved W_inp has a largest |entry| of exactly
0.3000, and a further run at a cap of 0.05 (below the initial scale) for one update has 254 of its
1,000 W_inp entries at exactly 0.05, so the clamp acts in the production path.

**Not mirrored, and why.** (1) bias_kick: the standard network's bias is fixed at zero (not in the
package's scope). (2) The two-cap branch of the magnitude penalty for Dale's law (`account4dale`):
the standard network has no E/I split; an RNN with a `dale_mask` is refused when lambda_rwm is on.
(3) The supervised train_step's post-rule projections (masks, Dale, I/O positivity) are not applied
after a redraw, as for the copy and rescale rules; the standard model needs none. (4) The rule's
state is not saved; a continuation restarts it, as before. Two things noticed: the comment on
`inp_weight_cap: null` in `configs/model/rnn_relu_standard.yaml` (line 58, "T5 knob: rescale every
W_inp row to this norm at init") describes `input_row_norm`, not the cap, while the block above it
describes the cap correctly; and `rl_maze_retrain.build_trainer` builds Adam for every retrain, so a
retrain of an SGD-trained network changes the optimiser (its docstring says "a fresh Adam").
