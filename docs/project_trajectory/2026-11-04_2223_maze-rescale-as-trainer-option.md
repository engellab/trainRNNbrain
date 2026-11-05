## 2026-11-04 22:23 — Maze: the paper's rescaling as a TrainerPPO option, the three-state row-tilt rule mirrored from the supervised trainer (issue #5, stage 2)

Branch `wp/maze-rescale` (worktree `~/Documents/Projects/paper_push/wt/maze-rescale-code`), from
`main` at 9a49c9e. Commits 420d2e1 (trainer, config, checks) and 6857916 (launcher). Not merged, not
pushed, nothing submitted. Follows `2026-11-04_2127_maze-interventions-as-trainer-options.md` (the
first four options) and `2026-11-04_2201_rl-maze-interventions-ladder-7383696-7383697.md` (the
ladder on Spock, which names rescaling as stage 2).

**What was built.** `TrainerPPO.prune_and_reinit` (`trainRNNbrain/trainer/TrainerPPO.py`) is the
supervised `Trainer.prune_and_reinit_` in reinit_mode "rescale" with rescale_target_frac > 0, step for
step and in the same order, under the keys `trainer.prune_reinit` and `trainer.prune_args` of
`configs/trainer/trainer_ppo.yaml` (off by default; the block's values are the paper's arm,
`slurm/SilentReLU_size_series_della.slurm` lines 124-133, with the supervised default noted where it
differs). Per check: participation p = std + q0.9 of the rate; silent if p < 0.05 x q95(p); the
contribution utility updated; strikes while silent, reset otherwise; doomed = strikes >= patience and
mature; at most max_replace_frac x N doomed per check, lowest score first; then GRADUATE (a growing
unit whose soft-max rate, `frm_activity`, reaches rescale_target_frac x the rate cap,
`frm_activity_cap`, stops growing), OPEN (every doomed unit not growing, up to rescale_protect_frac x
N growing at once, lowest score first: W_rec[:, i] and W_out[:, i] redrawn once from N(0, 1/sqrt(N))
with the RNN's generator, cumulative factor 1), SPEND (a growing unit whose factor reached rescale_cap
stops growing), BOOST (`rescale_rows`: positive incoming entries x alpha, negative / alpha, the W_rec
row only in the columns of firing sources with rescale_active_only, W_inp in full, each row scaled
back to its previous norm with rescale_normalize; factor x alpha); Adam's moments on the outgoing
columns of every doomed unit set to 0 (as the supervised block does, every check the unit is doomed,
not only at the opening); strikes[doomed] = 0. `zero_protected_grads` blanks the task gradient on the
growing units' incoming rows in every minibatch of `ppo_update`, after the clip and before the step
(the supervised trainer: once per iteration, between its clip and its step).

**Where the supervised rule counts iterations, this one counts updates.** The rule runs once per
update, after the update's n_epochs x n_minibatches optimizer steps, on the states of that update's
training rollout (noisy; every RNN step of every decision, including the (T + 1)-th decision's
substeps and the steps after an episode ended, the same pooling as the rate penalty and the dropout
participation); the rollout keeps the states for it (`batch["states"]`, 2.6 GB at N = 1,000 and 256
episodes, held through the update). check_every and maturity are in updates; patience counts checks,
one per update; rescale_alpha is a per-update factor; the first check is at update_n = 0, as the
supervised first check is at iter_n = 0. Issue #5's table said "on the probe's participation"; the
port uses the training rollout's, because the supervised rule reads the noisy training pass, and the
task statement for this stage said so.

**Not mirrored to the letter, and why.** (1) The clock: at alpha 1.002 per update a 2,000-update run
tilts a unit at most 2,000 times, a cumulative factor of 1.002^2000 = 54.6, below the cap of 64, so
no episode is ever spent and no unit is re-opened within a run; the supervised arm at 200,000
iterations reaches the cap after 2,083 tilts and re-opens a never-reviving unit about 96 times. The
per-update rule therefore has about a hundredth of the supervised rule's tilting budget per run. The
parameters are the paper's as asked; whether alpha should be raised for the maze is a decision for the
read-out. (2) `frm_activity` is computed 512 units at a time (the supervised code takes the whole
tensor); the rows are independent, so the values are the same up to the kernel's summation order.
(3) The supervised train_step re-applies structural masks, Dale's sign and I/O positivity after the
rule; TrainerPPO has no post-step projections, so the redrawn columns are not cleaned up by them. The
standard model (no Dale, self-connections on, no I/O positivity) needs none. (4) Only the paper's arm
is mirrored: reinit_mode other than "rescale", rescale_target_frac 0 (the legacy path), revive_op
other than "rescale", rescale_norm_mode "median_active" and rescale_refractory > 0 are refused at
construction. (5) The rule's state (strikes, utility, growing set, factors, update clock) is not
saved, as the supervised state is not; a warm start would reset it.

**Progress.json** gains, only with the option on, four integers per update: rescale_growing (units
growing after the check), rescale_opened, rescale_graduated, rescale_spent (counts at that check); the
probe line printed during training shows the growing count.

**Equivalence with the option off, pass rule set first: every saved number identical.** Main at
9a49c9e (a detached worktree in the scratchpad) and the branch, both from scratch through
`run_experiment.py task=Maze trainer=trainer_ppo model=rnn_relu_standard model.N=40 seed=7
trainer.n_updates=3 trainer.n_episodes=8 trainer.T=12 trainer.probe_episodes=6 trainer.warmup_updates=2`,
CPU: Progress.json has the same keys in every update with largest difference 0; all 15 arrays of
Params_Maze.npz and all 6 of Participation.npz bitwise equal; the saved config's task and model blocks
equal and the trainer block gains exactly prune_reinit and prune_args.

**Checks**, `tests/test_rl_maze.py` 23 of 23 (the 19 earlier ones unchanged) plus: the three states on
two units forced silent at N = 32 (A revivable, B with every incoming entry negative), the rule called
once per rollout with no optimizer step between, patience 2, alpha 2, target 0.3 x the cap, against
numpy re-implementations (common.participation for the silent set, a hand-written tilt, a hand-written
soft-max rate): check 1 one strike and nothing touched; check 2 both opened, the W_rec and W_out
columns of every opened unit equal to the next draws of the RNN's generator at 1/sqrt(N) and every
other column untouched, the input rows equal to the hand-tilted rows at the same norm within 1e-6,
factor alpha; check 3 the recurrent rows equal to the hand-tilted rows over the live columns only at
the same norm, factor alpha^2, no second redraw; A graduated at check 5, the first at which its
soft-max rate reached the target, untouched from then on; B's factor doubled per check to 64 and never
above, B stopped growing at check 8 with its rows unchanged and was re-opened at check 10 with a fresh
factor and a redrawn column. The blanking: the most active unit marked growing by hand takes exactly
zero gradient on its incoming rows in an update and those rows do not move, while its outgoing column
and other rows do; the same rollout through a trainer without the option moves the row. The counts in
run_training's records (ints, at least one episode opened, growing never above round(0.2 N), the clock
at n_updates) and their absence without the option. The refusals. `tests/test_no_function_imports.py`
202 files ok; `tests/test_scripts_start.py` 48 scripts ok.

**Smoke run through the script**, `run_experiment.py task=Maze trainer=trainer_ppo model.N=40 seed=3
trainer.prune_reinit=true`, 3 updates of 16 episodes x 30 decisions, into the scratchpad: the six
standard files written; the saved config records prune_reinit true and the 15 prune_args at the
paper's values; the counts per update were growing 8, 8, 8 (the ceiling round(0.2 x 40) = 8), opened
8, 0, 0, graduated 0, 0, 0, spent 0, 0, 0; no minibatch skipped, no restore.

**Launcher.** `slurm/rl_maze_interventions_spock.slurm` gains arm `rescale` at indices 22-24 (N = 1,000,
seeds 1-3, `experiment_tag="interventions/rescale"`), its command line the paper's eleven keys
(prune_reinit, reinit_mode, norm_mode, target_frac 10, active_only, normalize, alpha 1.002, cap 64,
protect_frac 0.2, check_every 1, patience 1); indices 1-21 unchanged; the array header is 1-24. Dry run
from the worktree with module, conda and srun stubbed (N = 40, 2 updates) for indices 1 and 22, pass
rule set first: the two saved configs differ in exactly experiment_tag and trainer.prune_reinit (the
swept prune_args equal the config's values); provenance line 6857916. Not submitted: 3 x about 5.5 h on
Spock, after the ladder's read-out as pre-registered.
