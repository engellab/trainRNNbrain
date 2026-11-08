"""The PPO trainer for the maze: the standard RNN as the whole agent, run one step at a time, trained
by rollouts, advantage estimates and the clipped PPO update, with a clean probe as its participation
tracking. The reinforcement-learning counterpart of trainRNNbrain/trainer/Trainer.py, built from
configs/trainer/trainer_ppo.yaml by trainRNNbrain/training/run_rl.py. Until 5 Nov 2026 this was
trainRNNbrain/rl/policy.py (MazePolicy), trainRNNbrain/rl/ppo.py and the loop of
trainRNNbrain/rl/run_maze_ppo.py; every computation is as it was.

THE RNN IS THE AGENT AND NOTHING ELSE IS TRAINED (Pavel, 4 Nov 2026). The RNN is `RNN_torch`
unchanged: its own W_inp, W_rec, W_out, activation, alpha and training noise, and its parameters
are the only parameters the optimizer holds. Its read-out rows are the action mean, turn and speed
(rows 0 and 1), and, when the RNN was built with 3 outputs, the critic's value (row 2), so the
value is one more read-out of the same network, trained by the value loss through the same W_out
and recurrent weights; with 2 outputs there is no critic (mc_advantages). The action is a sample
from a Gaussian around the mean with a FIXED standard deviation `action_std` (a number in the
config, not a parameter: a learned log-std collapsed to -3.4 in the first runs).

One environment step (a decision) is `substeps` RNN steps, x <- x + alpha * rhs(x, u, noise),
through the RNN's own `rhs`, with the observation held; the action is read from the state after
the last of them (with alpha = 0.1 the network needs several steps to respond to what it sees, see
configs/trainer/trainer_ppo.yaml).

THE NOISE IS THE SAME AS IN RNN_torch.forward. Recurrent and input noise are fresh at every step,
with the scale sqrt(2 / alpha) * sigma. Output noise is drawn ONCE per episode and unit, shape
(N, 1, B), and held constant over the episode, which is what forward does (its out_noise has the
shape of the initial state and broadcasts over time). The maze network therefore runs under the
very noise model the supervised networks were trained with, and the dormant-unit counts can be
compared. Synaptic noise (sigma_w) is one of the interventions below.

THE PAPER'S INTERVENTIONS (issue #5, since 5 Nov 2026) are options of this trainer, off by default,
each a mirror of its supervised definition in trainRNNbrain/trainer/Trainer.py or rnns/RNN_torch.py
under the same config keys (configs/trainer/trainer_ppo.yaml): the rate penalty `lambda_frm` and
`frm_args` (`fr_magnitude_penalty`, on the states of the recompute pass of every minibatch), the
sparsity penalty `lambda_rws` and `rws_args` (`rec_weights_sparsity_penalty`), noisy synapses (the
RNN's own `sigma_w`: a draw per RNN step in the rollout, regenerated from a stored seed in the
recompute, `synaptic_weights` and `advance`) and read-out dropout `dropout` and `dropout_args` (kind
"mute" only: one mask per rollout from a participation EMA, applied to W_out in the rollout and in
the recompute, `draw_dropout_mask` and `readout_weights`). With every option off no line added for
them draws a random number or adds a term to the loss, and a run is bit-identical to one made
before they existed (checked 5 Nov 2026 at N = 40, 3 updates, seed 7, against main at 9cc1aac).

RESCALING (issue #5, stage 2) is the fifth option, `prune_reinit` and `prune_args`: the supervised
`Trainer.prune_and_reinit_` in reinit_mode "rescale" with a fixed target, the three-state rule
dormant -> growing -> mature that the paper's rescaling arm runs
(slurm/SilentReLU_size_series_della.slurm). `prune_and_reinit` runs once per gradient step, inside
`ppo_update` after every optimizer step, on the states `sequence` recomputed for that minibatch
(the noisy rollout states of its episodes under the weights before the step, the counterpart of
the supervised noisy training pass), so the rule and the optimizer share one clock as they do in
the supervised train_step. `zero_protected_grads` blanks the task gradient on the growing units'
incoming rows in every minibatch of `ppo_update`. Where the supervised rule counts iterations,
this one counts optimizer steps, n_epochs x n_minibatches per update (the docstring of
`prune_and_reinit` lists each quantity). With the option off the run is bit-identical to one made
before it existed (checked at N = 40, 3 updates, seed 7, against main at 9a49c9e).

PRUNE-AND-DUPLICATE (issue #6, since 5 Nov 2026) is the sixth option, the same `prune_reinit` and
`prune_args` with reinit_mode "copy": the supervised `Trainer.prune_and_reinit_` in reinit_mode
"copy", the paper's duplication arm (slurm/SilentReLU_paper_grid_della.slurm: a check every 100
steps, a unit doomed after 3 consecutive silent checks, at most 0.025 N replaced per check, the
lowest contribution utility first, maturity 1,000 steps, copy_noise 1.0 with the row norm
preserved). A doomed unit is overwritten with a copy of a live unit drawn in proportion to
participation (`duplicate_units`): the donor's outgoing columns of W_rec and W_out are halved and
written to the copy's, so the network computes the same function at the instant of the copy; the
copy's incoming rows of W_rec and W_inp are the donor's rows, each entry times
(1 + copy_noise x N(0, 1)), scaled back to the donor's row norms (copy_norm_preserve, true in the
paper's arm); the 2 x 2 block of W_rec spanning
the pair is set from the donor's self-weight; Adam's moments on the copy's incoming rows and
outgoing columns are set to 0. The check cadence, patience and maturity count optimizer steps, as
for the rescale rule. Progress.json records per update the replacements made (copy_replaced, summed
over the update's checks) and the distinct units replaced so far (copy_units_ever).

THE UNIFORM SETTINGS (issue #6, package 3, since 5 Nov 2026) are four more mirrors, each off by
default. The input-weight cap is the model's `inp_weight_cap` (model.inp_weight_cap, read from the
RNN as the supervised Trainer reads it): after every optimizer step of ppo_update, |W_inp| is
clamped to the cap in place (`enforce_inp_cap`, the supervised Trainer.enforce_inp_cap_). The
recurrent-weight magnitude penalty `lambda_rwm` and `rwm_args` (`rec_weights_magnitude_penalty`, on
the raw W_rec) and the metabolic penalty `lambda_met` and `met_args` (`metabolic_penalty`, the mean
squared rate over the states of the recompute pass) are added to the loss of every minibatch as
the rate and the sparsity penalties are, and recorded in the update statistics as they are. The
redraw modes of `prune_reinit`, reinit_mode "random" and "zero_out" (`redraw_units`), are the
paper's re-initialisation arms (slurm/SilentReLU_replace_variants_della.slurm): a doomed unit's
incoming rows of W_rec and W_inp are redrawn from N(0, 1/sqrt(N)) and, in mode "zero_out", its
outgoing columns of W_rec and W_out are set to 0; the check cadence, patience, ceiling and maturity
are the copy mode's. The optimiser is the run script's choice (run_rl.py, trainer.optimizer: adam,
adamw or sgd, the switch of run_experiment.py): this trainer reads its base rate from whichever
optimizer it is given, ramps it the same way, and zeroes moments only where they exist (SGD without
momentum keeps none).

A CONTINUATION (paths.init_from, since 5 Nov 2026) is a run built by run_rl.py from a finished maze
run's weights and, with paths.init_adam, its Adam moments; this trainer takes two facts about it,
`iter_offset` (the parent's last recorded update, added to every update index it records, so the
continuation's records start where the parent's end and common.maze_load_runs reads the two as one
run) and `adam_restored` (the moments were loaded, so the warm-up ramp is not run again). Nothing
else of the parent's trainer is carried over: the rescale rule's state and the dropout
participation EMA are not saved, as in the supervised trainer, and start afresh.

WHY THE NOISE IS DRAWN UP FRONT. PPO recomputes the log-probability of the rollout's actions under
the updated parameters by re-running the RNN over the stored observations. The noise is part of
that forward pass, so a recompute with fresh noise would give a different mean even at the old
parameters, and the probability ratio would not start at 1. The rollout therefore draws its noise
tensors first, stores them, and the recompute replays them; tests/test_rl_maze.py checks that the
ratio is 1.

THE ALGORITHM, in words. A rollout plays a batch of episodes with the current policy and records what
it saw and did. The critic's VALUE of a state is its estimate of the discounted reward still to come
from there. The ADVANTAGE of an action is how much better the reward that followed it was than the
value predicted, so a positive advantage says "do that more"; generalised advantage estimation
(GAE, Schulman et al. 2016) blends one-step and many-step versions of that comparison with the
weight lambda to trade bias against variance. An episode cut off at T steps is not over, so its
last value BOOTSTRAPS the missing tail: it stands in for the reward after the cut. The PPO update
(Schulman et al. 2017) raises the probability of actions with positive advantage and lowers the
others, but through a CLIPPED SURROGATE: the probability ratio new/old is clipped to
[1 - clip, 1 + clip], so one update cannot move the policy far from the one that collected the data.
The critic is fitted to the observed returns at the same time (value loss).

Shapes follow the RNN's convention, batch last: observations u (n_inputs, T + 1, B), actions
(2, T, B), everything per step (T, B). A step here is one decision; the RNN runs `substeps`
of its own steps per decision. At step t the RNN consumes observation u_t and emits action a_t;
the environment returns reward r_t, observation u_{t+1} and whether the episode ended. The RNN is
stepped once more on the final observation u_T, which gives the bootstrap value (`values` has
T + 1 rows). `alive[t]` is 1 while the episode had not ended before step t (so alive[0] = 1 and
alive has T + 1 rows), and every loss is a mean over alive steps only.

THE PROBE is the trainer's participation tracking: a noise-free rollout with the policy mean on a
fixed batch of drops (the same drops at every probe of every run, `probe_seed`), the RL counterpart
of the noise-free forward pass the supervised Trainer probes with. Participation is taken over the
RNN steps before the cheese is reached. `run_training` records, per probe, the active count under
the project's rule (common.active_mask on p = std + q0.9), the count under the modulation
criterion (std alone: the count to read for tanh and sigmoid networks, whose resting rate clears
the p bar on its own), the probe's success rate, and every store_participation_every-th probe's
full participation vector.

`load_run` at the end of this file rebuilds a trained agent (task, RNN, trainer) from a run folder,
for the analysis scripts.
"""
import glob
import math
import os
import time
import hydra
import numpy as np
import torch
from torch.utils.checkpoint import checkpoint
from omegaconf import OmegaConf
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.tasks.TaskMaze import TaskMaze
from trainRNNbrain.utils import filter_kwargs
from trainRNNbrain.experiments_and_analysis.common import participation, active_mask

LOG_2PI = math.log(2 * math.pi)
# the generator seed of RNN step s of a rollout with synaptic-noise seed w_seed is
# w_seed * SEED_STRIDE + s (synaptic_weights). w_seed is drawn below 2^20 and the stride is 2^12
# (4,096 steps, more than any episode has), so every seed stays below 2^32: the CPU generator
# keeps only the low 32 bits of a manual seed (review of 4 Nov 2026), and the CUDA one 64
SEED_STRIDE = 2 ** 12
W_SEED_MAX = 2 ** 20
# the supervised Penalties.UpV ("N units per unit of volume, hard constant!"), the 100 in the rate
# cap cap_fr * log1p(100) / log1p(N)
UNITS_PER_VOLUME = 100.0
# units per chunk of the participation computation, the supervised trainer's value
# (Trainer.participation_from_states_); torch.quantile's limit is on the length of the reduced
# dimension (2^24 entries), which 2,505 steps x 256 episodes stays under
PARTICIPATION_CHUNK = 512
# the counts of prune_and_reinit that an update's record takes from its last check (a level), where
# every other count is summed over the update's checks (events)
RULE_LEVEL_KEYS = ("rescale_growing", "copy_units_ever", "reinit_units_ever")


class TrainerPPO:
    """PPO on one TaskMaze with one RNN_torch as the agent (actor rows 0, 1; critic row 2 if built with it)."""

    def __init__(self, RNN, Task, optimizer,
                 n_episodes=256, T=500, substeps=5, n_updates=2000,
                 gamma=0.99, gae_lambda=0.95, clip=0.2, n_epochs=4, n_minibatches=2,
                 value_coef=0.5, entropy_coef=0.0, critic=False, action_std=0.5,
                 max_grad_norm=0.5, warmup_updates=50,
                 probe_every=10, probe_episodes=128, probe_seed=20261103,
                 store_participation_every=5, checkpoint_every=100,
                 lambda_frm=0.0, frm_args=None, lambda_rws=0.0, rws_args=None,
                 lambda_rwm=0.0, rwm_args=None, lambda_met=0.0, met_args=None,
                 dropout=False, dropout_args=None, prune_reinit=False, prune_args=None,
                 iter_offset=0, adam_restored=False, freeze_after_end=False):
        """Args:
            RNN: RNN_torch with n_outputs 2 + critic (turn, speed, and the value row with a
                critic), on the task's device. Its sigma_w (model.sigma_w) switches synaptic noise
                on in the training rollouts, read here as the supervised Trainer reads it; its
                inp_weight_cap (model.inp_weight_cap, None for no cap) is the cap on |W_inp| that
                `enforce_inp_cap` applies after every optimizer step, read the same way.
            Task: TaskMaze.
            optimizer: torch optimizer over RNN.parameters(), Adam, AdamW or SGD as run_rl.py
                builds it from trainer.optimizer; its learning rate is the base rate the warm-up
                ramps to. None for a rebuilt agent that is only played (load_run).
            n_episodes: int, parallel episodes per update; T: int, decisions per episode;
            substeps: int >= 1, RNN steps per decision (the observation is held over them);
            n_updates: int, PPO updates in the run.
            gamma: float, discount; gae_lambda: float, GAE lambda (with a critic);
            clip: float, the ratio clip; n_epochs, n_minibatches: ints, passes over a rollout and
            the episodes split per pass; value_coef, entropy_coef: floats, loss weights.
            critic: bool, the RNN's third read-out row is the value (GAE advantages); False uses
                Monte-Carlo advantages with no parameters.
            action_std: float > 0, the fixed standard deviation of both action outputs, in raw
                output units (the task squashes them).
            max_grad_norm: float, gradient-norm clip per minibatch step.
            warmup_updates: int, updates over which the learning rate ramps from 0; 0 for none.
            probe_every: int, updates between clean probes (the first update is always probed);
            probe_episodes: int, the probe batch; probe_seed: int, seeds the probe's drops.
            store_participation_every: int, probes between stored full participation vectors.
            checkpoint_every: int, updates between calls of `checkpoint_fn`; 0 for none.
            lambda_frm: float >= 0, weight of the rate penalty (`fr_magnitude_penalty`) in the PPO
                loss of every minibatch; 0 (the default) leaves the term out. frm_args: dict with
                cap_fr, tau, g_top, g_bot and eps, the keys of the supervised frm_args block; None
                for the paper's values (0.3, 0.1, 3, 3, 1e-12).
            lambda_rws: float >= 0, weight of the sparsity penalty (`rec_weights_sparsity_penalty`);
                0 leaves it out. rws_args: dict with tg_deg and eps; None for 20 and 1e-12.
            lambda_rwm: float >= 0, weight of the recurrent-weight magnitude penalty
                (`rec_weights_magnitude_penalty`); 0 leaves it out. rwm_args: dict with cap100,
                N_ref, k_ref, gamma and eps, the keyword arguments of the supervised
                Penalties.rec_weights_magnitude_penalty; None or {} for its defaults (0.07, 100, 20,
                5.0, 1e-12), which the CDDM arm ran (+trainer.rwm_args={}).
            lambda_met: float >= 0, weight of the metabolic penalty (`metabolic_penalty`); 0 leaves
                it out. met_args: dict, the supervised key; the penalty takes no argument, so None
                or {}.
            dropout: bool, read-out dropout on every training rollout (`draw_dropout_mask`);
                dropout_args: dict with the keys of the supervised dropout_args block (dropout_kind,
                which must be "mute"; sampling_method; drop_rate; dropout_beta; eta; active_rel;
                rescale; p_max; activity_q), required when dropout is True.
            prune_reinit: bool, the per-unit rule (`prune_and_reinit`) once per optimizer step;
                prune_args: dict with the keys of the supervised prune_args block that the rule
                reads: reinit_mode, "rescale", "copy", "random" or "zero_out"; the keys every mode
                reads (check_every and maturity, in optimizer steps; patience; active_rel; utility;
                utility_decay; max_replace_frac); in mode "rescale" rescale_alpha,
                rescale_target_frac (> 0), rescale_cap, rescale_protect_frac, rescale_active_only,
                rescale_normalize and rescale_norm_mode ("self"); in mode "copy" copy_noise and
                copy_norm_preserve; modes "random" and "zero_out" read no key of their own.
                Required when prune_reinit is True.
            iter_offset: int >= 0, the absolute update count at which this run's clock starts: 0
                for a fresh run, the parent's last recorded update for a continuation
                (paths.init_from; run_rl.py resolves trainer.iter_offset "auto" to it). Added to
                every update index written to progress and probes, so a continuation's records
                start where its parent's end. The loop, the probe grid, the checkpoint grid and
                the warm-up ramp count this run's own updates from 1, as the supervised Trainer's
                track grid does.
            adam_restored: bool, True when the run script loaded the parent's optimizer state (the
                AdamState file, whichever optimiser wrote it) into `optimizer` (paths.init_adam).
                The warm-up ramp is then not run: it exists to tame Adam's first steps before its
                moment estimates exist, and a restored state has them
                (configs/trainer/trainer_ppo.yaml, warmup_updates). A continuation without the
                moments ramps again.
            freeze_after_end: bool (configs/trainer/trainer_ppo.yaml; False in every run before 8 Nov
                2026). False: the RNN keeps stepping on the last observation after an episode has
                ended, in the rollout and in the recompute, and the rate and metabolic penalties
                average over those steps too. True: an ended episode's state is held at its value
                at the end (rollout and recompute alike, so the ratio still starts at 1), and the
                two penalties average over the RNN steps before the end only (`live_steps`). Why: with
                False, post-episode dynamics, which no task term constrains, can grow without bound;
                in Maze_interventions/met0.01/s1 they reached |x| ~ 1e21 in one episode of 16 while
                every running episode stayed below 2, the metabolic penalty's float32 square overflowed
                and every optimiser step from update 1,501 on was skipped (entry
                2026-11-08_*_round19-met0.01-frozen-cause.md). The per-unit rules' statistics
                (prune_and_reinit, the dropout EMA) still take every step.
        Raises:
            ValueError if RNN.n_outputs is not 2 + critic; if lambda_rwm is above 0 on an RNN with
            Dale's law (RNN.dale_mask set: the penalty's two-cap branch is not mirrored); if dropout
            is on without dropout_args or with a dropout_kind other than "mute" ("dead" also
            removes the unit from the recurrent dynamics through the RNN's rhs; its replay in the
            recompute is not mirrored yet); if prune_reinit is on without prune_args, with a
            reinit_mode other than "rescale", "copy", "random" or "zero_out" (orth, mix and
            bias_kick are not mirrored; bias_kick could not act on the standard network's fixed
            zero bias); in mode "rescale"
            with rescale_target_frac <= 0 (the legacy stop-at-the-floor path, not mirrored), or
            with a revive_op other than "rescale", a rescale_norm_mode other than "self" or a
            rescale_refractory above 0 (the arms of the supervised rule the paper does not run;
            not mirrored); in mode "copy" with copy_iid or copy_permute true (the screen's variants
            of the copy, slurm/SilentReLU_replace_variants_della.slurm; not mirrored).
        """
        if RNN.n_outputs != 2 + int(bool(critic)):
            raise ValueError(f"the RNN needs {2 + int(bool(critic))} outputs with critic={critic} "
                             f"(turn, speed{', value' if critic else ''}); it has {RNN.n_outputs}")
        self.lambda_frm, self.lambda_rws = float(lambda_frm), float(lambda_rws)
        self.frm_args = {} if frm_args is None else dict(frm_args)
        self.rws_args = {} if rws_args is None else dict(rws_args)
        self.lambda_rwm, self.lambda_met = float(lambda_rwm), float(lambda_met)
        self.rwm_args = {} if rwm_args is None else dict(rwm_args)
        self.met_args = {} if met_args is None else dict(met_args)
        if self.lambda_rwm > 0 and getattr(RNN, "dale_mask", None) is not None:
            raise ValueError("TrainerPPO mirrors rec_weights_magnitude_penalty without Dale's law only (one cap for "
                             "every entry); this RNN has a dale_mask")
        self.sigma_w = float(RNN.sigma_w)
        # the cap on |W_inp|, the model's inp_weight_cap, read as the supervised Trainer reads it
        # (Trainer.enforce_inp_cap_: getattr(self.RNN, "inp_weight_cap", None)); None for no cap
        self.inp_weight_cap = getattr(RNN, "inp_weight_cap", None)
        self.dropout = bool(dropout)
        self.dropout_args = None if dropout_args is None else dict(dropout_args)
        if self.dropout:
            if self.dropout_args is None:
                raise ValueError("dropout=True needs dropout_args (the keys of the supervised dropout_args block)")
            kind = self.dropout_args.get("dropout_kind")
            if kind != "mute":
                raise ValueError(f"TrainerPPO mirrors read-out dropout of kind 'mute' only; dropout_kind is "
                                 f"{kind!r} ('dead' also removes the unit from the recurrent dynamics, and its "
                                 f"replay in the recompute is not mirrored yet)")
        # THE PARTICIPATION EMA the dropout mask is drawn from, the supervised Trainer's
        # `self.participation`, started at 1e-6 as there (so the first mask is drawn with every unit
        # in the pool at equal participation); updated after every training rollout (run_training)
        self.p_bar = (1e-6 * torch.ones(RNN.N, device=RNN.device)) if self.dropout else None
        self.prune_reinit = bool(prune_reinit)
        self.prune_args = None if prune_args is None else dict(prune_args)
        if self.prune_reinit:
            if self.prune_args is None:
                raise ValueError("prune_reinit=True needs prune_args (the keys of the supervised prune_args block)")
            args = self.prune_args
            mode = args.get("reinit_mode")
            if mode not in ("rescale", "copy", "random", "zero_out"):
                raise ValueError(f"TrainerPPO mirrors reinit_mode 'rescale', 'copy', 'random' and 'zero_out' only; "
                                 f"reinit_mode is {mode!r}")
            if mode == "rescale":
                if float(args.get("rescale_target_frac", 0.0)) <= 0.0:
                    raise ValueError("TrainerPPO mirrors the three-state rescale rule only, which needs "
                                     "rescale_target_frac > 0 (the legacy path at 0 is not mirrored)")
                for key, only in (("revive_op", "rescale"), ("rescale_norm_mode", "self"), ("rescale_refractory", 0)):
                    if args.get(key, only) != only:
                        raise ValueError(f"TrainerPPO mirrors the rescale rule with {key}={only!r} only; "
                                         f"it is {args.get(key)!r}")
            else:
                for key in ("copy_iid", "copy_permute"):
                    if bool(args.get(key, False)):
                        raise ValueError(f"TrainerPPO mirrors the paper's copy only, with {key} false; it is "
                                         f"{args.get(key)!r} (the screen's variant is not mirrored)")
        # THE RULE'S STATE, the supervised Trainer's tensors of the same names, one entry per unit:
        # consecutive silent checks; the running contribution utility; the optimizer step at which
        # the unit was last doomed (maturity), -1e9 for never; whether the unit was ever replaced
        # (mode "copy"); the cumulative tilt factor of the current episode and whether the unit is
        # growing (tilted and gradient-protected; mode "rescale"). step_n counts the optimizer steps
        # completed before the current check, as the supervised iter_n counts the iterations
        # completed before its check: 0 at the first check.
        self._reinit_strikes = torch.zeros(RNN.N, device=RNN.device)
        self._unit_utility = torch.zeros(RNN.N, device=RNN.device)
        self._last_replaced = torch.full((RNN.N,), -1e9, device=RNN.device)
        self._reinit_ever = torch.zeros(RNN.N, dtype=torch.bool, device=RNN.device)
        self._rescale_cum = torch.ones(RNN.N, device=RNN.device)
        self._rescale_growing = torch.zeros(RNN.N, dtype=torch.bool, device=RNN.device)
        self.step_n = 0
        self.RNN = RNN
        self.Task = Task
        self.optimizer = optimizer
        self.device = RNN.device
        self.critic = bool(critic)
        self.action_std = float(action_std)
        self.substeps = int(substeps)
        self.n_episodes, self.T, self.n_updates = int(n_episodes), int(T), int(n_updates)
        self.gamma, self.gae_lambda, self.clip = float(gamma), float(gae_lambda), float(clip)
        self.n_epochs, self.n_minibatches = int(n_epochs), int(n_minibatches)
        self.value_coef, self.entropy_coef = float(value_coef), float(entropy_coef)
        self.max_grad_norm = float(max_grad_norm)
        self.warmup_updates = int(warmup_updates or 0)
        self.iter_offset = int(iter_offset)
        self.adam_restored = bool(adam_restored)
        self.freeze_after_end = bool(freeze_after_end)
        # the base learning rate, as the run script set it (the size rule); the warm-up ramps to it
        self.lr = None if optimizer is None else float(optimizer.param_groups[0]["lr"])
        self.probe_every, self.probe_episodes = int(probe_every), int(probe_episodes)
        self.probe_seed = int(probe_seed)
        self.store_participation_every = int(store_participation_every)
        # PERIODIC CHECKPOINT, updates between writes; 0 disables it. As in the supervised Trainer
        # the trainer only decides WHEN: `checkpoint_fn` is set from outside to a callable
        # fn(updates_done, params), and reads this trainer's progress, probes and last_probe.
        self.checkpoint_every = int(checkpoint_every or 0)
        self.checkpoint_fn = None
        # filled by run_training: one dict per update; dict of lists, one entry per probe; the
        # last probe's rollout (for the trajectory figure)
        self.progress, self.probes, self.last_probe = [], None, None

    # ------------------------------------------------------------------ the RNN as the policy

    def draw_noise(self, T, B, w_noise, generator=None):
        """The RNN's noise for T steps of B episodes, as RNN_torch.forward draws it.

        Args:
            T, B: int; w_noise: bool, False gives zeros (a noise-free pass);
            generator: torch.Generator on the RNN's device, or None for the RNN's own.
        Returns:
            dict with "rec" (N, T, B) and "inp" (n_inputs, T, B), fresh per step, and "out"
            (N, 1, B), one draw per episode held over its T steps; all scaled by sqrt(2/alpha)*sigma.
        """
        rnn = self.RNN
        g = rnn.random_generator if generator is None else generator
        shapes = {"rec": (rnn.N, T, B), "inp": (rnn.n_inputs, T, B), "out": (rnn.N, 1, B)}
        sigmas = {"rec": rnn.sigma_rec, "inp": rnn.sigma_inp, "out": rnn.sigma_out}
        noise = {}
        for key, shape in shapes.items():
            if w_noise:
                scale = torch.sqrt((2 / rnn.alpha) * sigmas[key] ** 2)
                noise[key] = scale * torch.randn(*shape, generator=g, device=rnn.device)
            else:
                noise[key] = torch.zeros(shape, device=rnn.device)
        return noise

    def std(self):
        """The policy's standard deviation per action output, (2,) on the RNN's device."""
        return torch.full((2,), self.action_std, device=self.RNN.device)

    def initial_state(self, B):
        """The RNN state every episode starts from: y_init repeated, (N, B)."""
        return self.RNN.y_init.to(self.RNN.device).reshape(-1, 1).repeat(1, B)

    def step(self, x, u, rec_t, inp_t, W_rec=None, W_inp=None):
        """One RNN step through the RNN's own right-hand side.

        Args:
            x: (N, B) state; u: (n_inputs, B) observation; rec_t: (N, B) and inp_t: (n_inputs, B),
            this step's recurrent and input noise; W_rec (N, N) and W_inp (N, n_inputs), the
            weights of this step (`synaptic_weights`), or both None for the RNN's constrained weights.
        Returns:
            (N, B) next state.
        """
        if W_rec is None:
            W_rec, W_inp, _ = self.RNN._constrained_weights()
        return x + self.RNN.alpha * self.RNN.rhs(x=x, I=u, i_noise=inp_t, r_noise=rec_t,
                                                 W_rec=W_rec, W_inp=W_inp)

    # ------------------------------------------------------------------ noisy synapses (sigma_w)

    def draw_synaptic_seed(self, generator=None):
        """The synaptic-noise seed of one rollout: one integer in [0, 2^31 - 1) from the RNN's
        generator (or `generator`). Called only when sigma_w > 0 and the pass is noisy, so a run
        without synaptic noise consumes the generator exactly as before.

        Args:
            generator: torch.Generator on the RNN's device, or None for the RNN's own.
        Returns:
            int.
        """
        g = self.RNN.random_generator if generator is None else generator
        return int(torch.randint(0, W_SEED_MAX, (1,), generator=g, device=self.device).item())

    def synaptic_weights(self, w_seed, s):
        """W_rec and W_inp with the synaptic noise of RNN step s, as RNN_torch.forward applies it
        (rnns/RNN_torch.py, the loop of forward):
            Wr_t = Wrec_c * (1.0 + sw * torch.randn(Wrec_c.shape, generator=self.random_generator, ...))
            Wi_t = Winp_c * (1.0 + sw * torch.randn(Winp_c.shape, generator=self.random_generator, ...))
        with sw = sigma_w, "a fresh multiplicative perturbation of BOTH weight matrices at every
        step", "ONE DRAW PER TIMESTEP, SHARED ACROSS THE BATCH" (an (N, N, B) draw does not fit).

        forward draws from the RNN's running generator. Here the draws of step s come from a
        generator seeded with w_seed * SEED_STRIDE + s, so that any one step's draws can be
        regenerated on their own: the recompute of ppo_update replays the rollout's weights from
        the stored seed (an (N, N) tensor per step cannot be stored for 2,500 steps), and the
        backward pass regenerates them as well (`advance`) instead of holding them. Cost: one
        (N, N) and one (N, n_inputs) normal draw per RNN step per pass, 4 MB at N = 1,000, made and
        released step by step.

        Args:
            w_seed: int, the rollout's seed (`draw_synaptic_seed`); s: int, the RNN step index
            within the rollout (0 to (T + 1) * substeps - 1).
        Returns:
            (W_rec, W_inp): (N, N) and (N, n_inputs) tensors, differentiable in the RNN's parameters.
        """
        W_rec, W_inp, _ = self.RNN._constrained_weights()
        g = torch.Generator(device=self.device)
        g.manual_seed(w_seed * SEED_STRIDE + s)
        W_rec = W_rec * (1.0 + self.sigma_w * torch.randn(W_rec.shape, generator=g, device=self.device))
        W_inp = W_inp * (1.0 + self.sigma_w * torch.randn(W_inp.shape, generator=g, device=self.device))
        return W_rec, W_inp

    def synaptic_step(self, x, u, rec_t, inp_t, w_seed, s):
        """One RNN step under synaptic noise: `step` with the weights of synaptic_weights(w_seed, s).

        Args:
            x, u, rec_t, inp_t: as in `step`; w_seed: int; s: int, the RNN step index.
        Returns:
            (N, B) next state.
        """
        W_rec, W_inp = self.synaptic_weights(w_seed, s)
        return self.step(x, u, rec_t, inp_t, W_rec, W_inp)

    def advance(self, x, u, rec_t, inp_t, w_seed, s):
        """One RNN step of a rollout or of the recompute: `step` when w_seed is None (no synaptic
        noise: the option is off, or the pass is the clean probe, which runs without it as the
        supervised clean pass does), `synaptic_step` otherwise.

        Under autograd (the recompute) the noisy step runs through torch.utils.checkpoint, so the
        backward pass recomputes it (one more pair of draws and one more step per RNN step) instead
        of keeping the perturbed W_rec and its multiplier of every step: at N = 1,000 and 2,505
        steps those are 2 x 4 MB x 2,505 = 20 GB per minibatch. Memory is then that of a run
        without synaptic noise; time is the recompute's forward pass once more.

        Args:
            x, u, rec_t, inp_t: as in `step`; w_seed: int or None; s: int, the RNN step index.
        Returns:
            (N, B) next state.
        """
        if w_seed is None:
            return self.step(x, u, rec_t, inp_t)
        if torch.is_grad_enabled():
            return checkpoint(self.synaptic_step, x, u, rec_t, inp_t, w_seed, s,
                              use_reentrant=False, preserve_rng_state=False)
        return self.synaptic_step(x, u, rec_t, inp_t, w_seed, s)

    # ------------------------------------------------------------------ read-out dropout (mute)

    def participation_from_states(self, states, q=0.9):
        """Per-unit participation std(r) + q-quantile(|r|) of a state tensor, pooled over every step
        and episode: the supervised Trainer.participation_from_states_, "THE SINGLE DEFINITION OF
        PARTICIPATION" behind its logged trace and its dropout sampler:
            out[i:i + chunk] = (blk.std(dim=1, unbiased=False) + torch.quantile(blk.abs(), q, dim=1))
        on the firing rates (the activation applied for equation "h"), 512 units per chunk as in
        the supervised code (torch.quantile's 2^24-entry limit applies to the reduced dimension,
        the steps x episodes, which stays under it).

        Args:
            states: (N, T, B) RNN states; q: float in (0, 1), the quantile
                (dropout_args["activity_q"]; 0.9 is the project's definition).
        Returns:
            (N,) tensor >= 0, in the states' dtype and on their device.
        """
        N = states.size(0)
        chunk = PARTICIPATION_CHUNK
        out = torch.empty(N, device=states.device, dtype=states.dtype)
        for i in range(0, N, chunk):
            blk = self.RNN.get_firing_rates(states[i:i + chunk].reshape(min(chunk, N - i), -1))
            out[i:i + chunk] = blk.std(dim=1, unbiased=False) + torch.quantile(blk.abs(), q, dim=1)
        return out

    def draw_dropout_mask(self):
        """The read-out mask of one training rollout, drawn by the supervised rule
        (RNN_torch.get_dropout_mask with kind "mute") from the participation EMA p_bar. With
        sampling_method "participation": the pool is the units with p_bar >= active_rel x q95(p_bar)
        (`live = v >= active_rel * torch.quantile(v, 0.95)`); within the pool of M units each is
        ranked by p_bar and weighted `w = torch.softmax(beta * rank / max(M - 1, 1), dim=0)`; the
        drop probabilities are `drop_probabilities(w, drop_rate * M, p_max)`, the weights rescaled
        and clipped so that they sum to drop_rate x M with none above p_max; each unit is dropped
        by an independent Bernoulli draw from the RNN's generator; with rescale a kept unit's factor
        is `1.0 / (1.0 - p)`, clipped at 1 / (1 - p_max). "uniform" and "output_weights" sample as
        there, without the EMA.

        Returns:
            (N, 1) tensor, DropoutMask.outgoing: 0 for a dropped unit, 1 / (1 - p_i) (1 without
            rescale, and for units outside the pool) for a kept one.
        """
        part = self.p_bar if self.dropout_args.get("sampling_method") == "participation" else None
        return self.RNN.get_dropout_mask(self.dropout_args, part).outgoing

    def readout_weights(self, outgoing=None):
        """W_out as the read-out uses it: the RNN's constrained W_out, with a dropout mask applied as
        RNN_torch.forward applies it (`W_out = Wout_c * dropout_mask.outgoing.reshape(1, -1)`), so
        a dropped unit's column is zero and a kept unit's column carries its rescaling factor.

        Args:
            outgoing: (N, 1) tensor from `draw_dropout_mask`, or None for the plain W_out.
        Returns:
            (n_outputs, N) tensor.
        """
        W_out = self.RNN._constrained_weights()[2]
        if outgoing is None:
            return W_out
        return W_out * outgoing.reshape(1, -1).to(W_out.dtype)

    def heads(self, x, out_noise, outgoing=None):
        """Policy mean and value from RNN states: the RNN's read-out of the noisy rates.

        Args:
            x: (N, B) or (N, T, B) states; out_noise: (N, B) or (N, 1, B), the episode's output
            noise (broadcast over time); outgoing: (N, 1) read-out dropout mask or None
            (`readout_weights`).
        Returns:
            (mean, value): mean (2, B) or (2, T, B), read-out rows 0 and 1; value (B,) or (T, B),
            read-out row 2, or zeros without a critic.
        """
        r = self.RNN.get_firing_rates(x)
        W_out = self.readout_weights(outgoing)
        out = torch.einsum("oj,j...->o...", W_out, r + out_noise)
        value = out[2] if self.critic else torch.zeros_like(out[0])
        return out[:2], value

    def sequence(self, u, noise, w_seed=None, outgoing=None, alive=None):
        """Re-run the RNN over a stored observation sequence with stored noise (the PPO recompute).

        Args:
            u: (n_inputs, T, B) observations, one per decision;
            noise: dict from `draw_noise` for T * substeps RNN steps;
            w_seed: int or None, the rollout's synaptic-noise seed, whose per-step draws are
                regenerated here (`advance`); outgoing: (N, 1) or None, the rollout's read-out mask;
            alive: (T, B) or None, the rollout's alive rows for these decisions; with
                freeze_after_end on it holds an ended episode's state as the rollout did (required
                then: without it the recompute would not match the rollout).
        Returns:
            (states, means, values): states (N, T * substeps, B), the state after every RNN step;
            means (2, T, B) and values (T, B), read from the state after the last substep of each
            decision.
        """
        T, B = u.shape[1], u.shape[2]
        if self.freeze_after_end and alive is None:
            raise ValueError("sequence: freeze_after_end is on, so the rollout's alive rows are needed")
        x = self.initial_state(B)
        states = []
        for t in range(T):
            for s in range(t * self.substeps, (t + 1) * self.substeps):
                x_new = self.advance(x, u[:, t], noise["rec"][:, s], noise["inp"][:, s], w_seed, s)
                x = torch.where(alive[t].bool()[None, :], x_new, x) if self.freeze_after_end else x_new
                states.append(x)
        states = torch.stack(states, dim=1)
        means, values = self.heads(states[:, self.substeps - 1::self.substeps], noise["out"], outgoing)
        return states, means, values

    # ------------------------------------------------------------------ the paper's penalties

    def live_steps(self, alive):
        """The RNN steps before each episode's end: the decision rows of `alive` repeated `substeps`
        times, the rule the probe uses for participation.

        Args:
            alive: (T, B) float or bool, the rollout's alive rows for T decisions.
        Returns:
            (T * substeps, B) bool tensor.
        """
        return alive.bool().repeat_interleave(self.substeps, dim=0)

    def fr_magnitude_penalty(self, states, mask=None):
        """The rate penalty of the supervised trainer (Trainer.Penalties.fr_magnitude_penalty), re-typed
        line for line at the paper's frm_args (configs/trainer/trainer_ptrack_freshbatch.yaml: cap_fr
        0.3, tau 0.1, g_top 3, g_bot 3, aggregation "mean", alpha = beta = 1).

        Per unit, a soft maximum of its rate over every step and episode, tau x (logsumexp(r / tau)
        - log(n)), is compared with a cap that falls with the size of the network, cap_fr x
        log1p(100) / log1p(N) (100 is the supervised Penalties.UpV, "N units per unit of volume").
        The penalty is ((cap - a) / cap) ** g_bot for a unit below the cap and ((a - cap) / cap)
        ** g_top above it, the powers taken in float64 (the supervised guard against overflow),
        averaged over units. The supervised lines:
            activity = tau * (torch.logsumexp(a, dim=1) - torch.log(a.size(1)))    # a = x / tau
            over = torch.relu(activity - cap);  under = torch.relu(cap - activity)
            p_over = torch.pow(over.double() / (cap64 + eps), g_top)
            p_under = torch.pow(under.double() / (cap64 + eps), g_bot)
            return (alpha * p_under + beta * p_over).mean()
        "Do NOT take abs(x)": the signed rate enters, as there.

        Args:
            states: (N, T, B) RNN states (pre-activations for equation "h", rates for "s"): in
                ppo_update the states `sequence` returns, every RNN step of every decision of the
                minibatch, including the steps after an episode ended (the noisy rollout states).
            mask: (T, B) bool or None, the steps to keep (`live_steps`, with freeze_after_end on);
                None keeps every step, as every run before 8 Nov 2026 did.
        Returns:
            scalar float64 tensor P >= 0 on the states' device; the loss adds lambda_frm x P.
        """
        args = self.frm_args
        cap_fr, tau, eps = float(args.get("cap_fr", 0.3)), float(args.get("tau", 0.1)), float(args.get("eps", 1e-12))
        g_top, g_bot = float(args.get("g_top", 3.0)), float(args.get("g_bot", 3.0))
        dev, dt = states.device, states.dtype
        x = self.RNN.get_firing_rates(states.reshape(states.size(0), -1))        # (N, T * B)
        if mask is not None:
            x = x[:, mask.reshape(-1)]
        scale = (torch.log1p(torch.as_tensor(UNITS_PER_VOLUME, device=dev, dtype=dt))
                 / torch.log1p(torch.as_tensor(float(self.RNN.N), device=dev, dtype=dt)))
        cap = torch.as_tensor(cap_fr, device=dev, dtype=dt) * scale
        a = x / tau
        activity = tau * (torch.logsumexp(a, dim=1) - torch.log(torch.as_tensor(a.size(1), device=dev, dtype=dt)))
        over = torch.relu(activity - cap)
        under = torch.relu(cap - activity)
        cap64 = cap.double()
        p_over = torch.pow(over.double() / (cap64 + eps), g_top)
        p_under = torch.pow(under.double() / (cap64 + eps), g_bot)
        return (p_under + p_over).mean()

    def rec_weights_sparsity_penalty(self):
        """The sparsity penalty of the supervised trainer (Trainer.Penalties.rec_weights_sparsity_penalty)
        on the raw W_rec parameter, as there. Per row i of W_rec the effective number of synapses is
        S_i = (sum_j |W_ij|)^2 / sum_j W_ij^2 (1 for one synapse, N for N equal ones); the penalty
        is relu(S - tg_deg)^2 averaged over rows and divided by tg_deg^2. The supervised lines:
            l1 = W.abs().sum(dim=1);  l2 = (W.square().sum(dim=1) + eps).sqrt()
            S = (l1 * l1) / (l2 * l2);  over = torch.relu(S - tg_deg)
            return (over ** 2).mean() / (tg_deg ** 2)

        Returns:
            scalar tensor P >= 0 in W_rec's dtype; the loss adds lambda_rws x P.
        """
        tg_deg, eps = float(self.rws_args.get("tg_deg", 20)), float(self.rws_args.get("eps", 1e-12))
        W = self.RNN.W_rec
        l1 = W.abs().sum(dim=1)
        l2 = (W.square().sum(dim=1) + eps).sqrt()
        S = (l1 * l1) / (l2 * l2)
        over = torch.relu(S - tg_deg)
        return (over ** 2).mean() / (tg_deg ** 2)

    def rec_weights_magnitude_penalty(self):
        """The recurrent-weight magnitude penalty of the supervised trainer
        (Trainer.Penalties.rec_weights_magnitude_penalty) on the raw W_rec parameter, in its branch
        without Dale's law (the standard network's; the two-cap branch for an E/I split is not
        mirrored, and an RNN with a dale_mask is refused at construction). Every entry |W_ij| is
        compared with a cap that falls with the size of the network, cap100 x log1p(100) / log1p(N)
        (100 is the supervised Penalties.UpV): an entry above the cap pays (|W_ij| / cap)^gamma - 1,
        an entry below it pays nothing, and the mean over the N^2 entries is multiplied by
        N / (N_ref x k_ref). The supervised lines:
            scale = log1p(UpV) / log1p(N);  cap = cap100 * scale;  W = R.W_rec.abs()
            r = (W + eps) / (cap + eps)
            p = (torch.pow(torch.relu(r - 1.0) + 1.0, gamma) - 1.0)
            return p.mean() * (N / (N_ref * k_ref))
        with the defaults cap100 0.07, N_ref 100, k_ref 20, gamma 5.0 and eps 1e-12 unless rwm_args
        sets them (the CDDM arm, slurm/SilentReLU_std_recinit_della.slurm, passed an empty dict).

        Returns:
            scalar tensor P >= 0 in W_rec's dtype; the loss adds lambda_rwm x P.
        """
        args = self.rwm_args
        cap100, gamma, eps = float(args.get("cap100", 0.07)), float(args.get("gamma", 5.0)), float(args.get("eps", 1e-12))
        N_ref, k_ref = float(args.get("N_ref", 100)), float(args.get("k_ref", 20))
        W = self.RNN.W_rec.abs()
        dev, dt = W.device, W.dtype
        N = self.RNN.N
        scale = (torch.log1p(torch.as_tensor(UNITS_PER_VOLUME, device=dev, dtype=dt))
                 / torch.log1p(torch.as_tensor(float(N), device=dev, dtype=dt)))
        cap = torch.as_tensor(cap100, device=dev, dtype=dt) * scale
        r = (W + eps) / (cap + eps)
        p = torch.pow(torch.relu(r - 1.0) + 1.0, gamma) - 1.0
        return p.mean() * (N / (N_ref * k_ref))

    def metabolic_penalty(self, states, mask=None):
        """The metabolic penalty of the supervised trainer (Trainer.Penalties.metabolic_penalty): the
        mean squared firing rate over every unit, step and episode. The supervised lines:
            fr = self.RNN.activation(states)    # equation "h"; the states themselves for "s"
            return torch.mean(fr ** 2)
        (get_firing_rates applies the same rule). met_args is not read: the penalty takes no
        argument there either.

        Args:
            states: (N, T, B) RNN states; in ppo_update the states `sequence` returns for the
                minibatch, every RNN step of every decision, as for the rate penalty.
            mask: (T, B) bool or None, the steps to keep (`live_steps`, with freeze_after_end on);
                None keeps every step, as every run before 8 Nov 2026 did.
        Returns:
            scalar tensor P >= 0 in the states' dtype; the loss adds lambda_met x P.
        """
        fr = self.RNN.get_firing_rates(states)
        if mask is not None:
            fr = fr[:, mask]
        return torch.mean(fr ** 2)

    # ------------------------------------------------------------------ the per-unit rule (prune_reinit)

    def duplicate_units(self, idx, p, silent):
        """Overwrite each doomed unit with a jittered copy of a live unit: the copy branch of the
        supervised Trainer.prune_and_reinit_ (reinit_mode "copy"), line for line, without the
        screen's copy_iid and copy_permute variants (refused at construction).

        THE DONOR of each doomed unit is drawn from the live units (not silent at this check) with
        replacement, in proportion to participation, with the RNN's generator:
            live_idx = torch.nonzero(~silent, as_tuple=True)[0]
            weights = p[live_idx].clamp_min(1e-12)
            donors = live_idx[torch.multinomial(weights, n, replacement=True, generator=...)]
        THE COPY, one pair (copy_i, donor_j) at a time, in the order of idx:
          the donor's rows W_rec[j, :] and W_inp[j, :] and its self-weight W_rec[j, j] are read
          first; then the donor's outgoing columns are halved and shared,
            W_rec[:, j] *= 0.5;  W_out[:, j] *= 0.5;  W_rec[:, i] = W_rec[:, j];  W_out[:, i] = W_out[:, j]
          so every downstream unit receives (W[k, j] / 2 + W[k, i] / 2) r = W[k, j]_old r, the
          same drive as before; the copy's incoming rows are the donor's with a relative jitter
          on every entry (copy_noise > 0),
            row = row * (1.0 + jitter * torch.randn(row.shape, generator=...))
          and, with copy_norm_preserve, scaled back to the donor's row norm,
            row = row * (len / row.norm().clamp_min(1e-12))
          so the jitter sets the copy's direction only (expected cosine 1 / sqrt(1 + jitter^2) to
          the donor); W_rec[i, :] = row_rec; W_inp[i, :] = row_inp. THE 2 x 2 BLOCK spanning the
          pair: with self-connections the donor drew self_w x r from itself, and the pair keeps
          drawing that once both fire alike, so W_rec[i, i] = W_rec[i, j] = W_rec[j, i] = self_w / 2
          (W_rec[j, j] is self_w / 2 from the halving); without self-connections the three entries
          are 0 (the masked diagonal must not reappear as a live connection between the pair).
        A doomed unit whose draws fall on a donor already copied in this call takes the donor's
        rows and self-weight as the earlier copy left them, as the supervised loop does.

        Args:
            idx: (n,) long tensor of the doomed units (ascending); p: (N,) participation of this
                check; silent: (N,) bool, the silent units of this check.
        Returns:
            int, the units overwritten: n, or 0 when no unit is live (then nothing is changed, and
            the supervised rule returns without resetting the strikes). Call under torch.no_grad().
        """
        args = self.prune_args
        live_idx = torch.nonzero(~silent, as_tuple=True)[0]
        if live_idx.numel() == 0:
            return 0
        n = int(idx.numel())
        weights = p[live_idx].clamp_min(1e-12)
        donors = live_idx[torch.multinomial(weights, n, replacement=True, generator=self.RNN.random_generator)]
        jitter = float(args.get("copy_noise", 0.05))
        norm_preserve = bool(args.get("copy_norm_preserve", False))
        for copy_i, donor_j in zip(idx.tolist(), donors.tolist()):
            row_rec = self.RNN.W_rec[donor_j, :].clone()
            row_inp = self.RNN.W_inp[donor_j, :].clone()
            self_w = self.RNN.W_rec[donor_j, donor_j].clone()
            len_rec, len_inp = row_rec.norm(), row_inp.norm()
            self.RNN.W_rec[:, donor_j] *= 0.5
            self.RNN.W_out[:, donor_j] *= 0.5
            self.RNN.W_rec[:, copy_i] = self.RNN.W_rec[:, donor_j]
            self.RNN.W_out[:, copy_i] = self.RNN.W_out[:, donor_j]
            if jitter > 0:
                row_rec = row_rec * (1.0 + jitter * torch.randn(row_rec.shape, device=row_rec.device,
                                                                generator=self.RNN.random_generator))
                row_inp = row_inp * (1.0 + jitter * torch.randn(row_inp.shape, device=row_inp.device,
                                                                generator=self.RNN.random_generator))
                if norm_preserve:
                    row_rec = row_rec * (len_rec / row_rec.norm().clamp_min(1e-12))
                    row_inp = row_inp * (len_inp / row_inp.norm().clamp_min(1e-12))
            self.RNN.W_rec[copy_i, :] = row_rec
            self.RNN.W_inp[copy_i, :] = row_inp
            if self.RNN.self_connections:
                half = 0.5 * self_w
                self.RNN.W_rec[copy_i, copy_i] = half
                self.RNN.W_rec[copy_i, donor_j] = half
                self.RNN.W_rec[donor_j, copy_i] = half
            else:
                self.RNN.W_rec[copy_i, donor_j] = 0.0
                self.RNN.W_rec[donor_j, copy_i] = 0.0
                self.RNN.W_rec[copy_i, copy_i] = 0.0
        return n

    def redraw_units(self, idx, mode):
        """Overwrite each doomed unit's incoming weights with a fresh draw: the supervised
        Trainer.prune_and_reinit_ in reinit_mode "random" and "zero_out" (its last branch), the
        paper's re-initialisation arms (slurm/SilentReLU_replace_variants_della.slurm, modes random
        and zero_out). The rows W_rec[idx, :] and W_inp[idx, :] are redrawn from N(0, 1 / sqrt(N)),
        the scale the network was initialised at, with the RNN's generator, the W_rec rows first.
        In mode "zero_out" (the rule of Dohare et al. 2024) the outgoing columns W_rec[:, idx] and
        W_out[:, idx] are then set to 0, so the new unit changes nothing the network computes until
        the gradient has grown its output back; in mode "random" the outgoing weights stay as
        training left them. The supervised lines:
            std = 1.0 / np.sqrt(self.RNN.N)
            self.RNN.W_rec[idx, :] = torch.randn(n, self.RNN.N, generator=...) * std
            self.RNN.W_inp[idx, :] = torch.randn(n, self.RNN.W_inp.shape[1], generator=...) * std
            if mode == "zero_out":
                self.RNN.W_rec[:, idx] = 0.0;  self.RNN.W_out[:, idx] = 0.0

        Args:
            idx: (n,) long tensor of the doomed units; mode: str, "random" or "zero_out".
        Returns:
            int, the units overwritten (n). Call under torch.no_grad().
        """
        n = int(idx.numel())
        std = 1.0 / np.sqrt(self.RNN.N)
        self.RNN.W_rec[idx, :] = torch.randn(n, self.RNN.N, device=self.RNN.device,
                                             generator=self.RNN.random_generator) * std
        self.RNN.W_inp[idx, :] = torch.randn(n, self.RNN.W_inp.shape[1], device=self.RNN.device,
                                             generator=self.RNN.random_generator) * std
        if mode == "zero_out":
            self.RNN.W_rec[:, idx] = 0.0
            self.RNN.W_out[:, idx] = 0.0
        return n

    def frm_activity_cap(self):
        """The rate cap of the rate penalty for this network's size, cap_fr x log1p(100) / log1p(N)
        (the supervised Trainer.frm_activity_cap_, with Penalties.UpV = 100 and cap_fr from
        frm_args). The rescale rule's target is rescale_target_frac times this number.

        Returns:
            float, in the units of `frm_activity`.
        """
        cap_fr = float(self.frm_args.get("cap_fr", 0.3))
        return cap_fr * float(np.log1p(UNITS_PER_VOLUME) / np.log1p(self.RNN.N))

    def frm_activity(self, states):
        """Per-unit soft maximum of the rate over every step and episode, the statistic of the rate
        penalty (the supervised Trainer.frm_activity_):
            x = states.detach().reshape(states.shape[0], -1);  x = self.RNN.activation(x)   # equation "h"
            return tau * (torch.logsumexp(x / tau, dim=1) - torch.log(n))
        with tau from frm_args and n the number of steps x episodes. The rescale rule graduates a
        unit when this reaches its target, so "reached the target" means what the penalty means.
        Computed PARTICIPATION_CHUNK units at a time (the supervised code takes the whole tensor at
        once); the rows are independent, so the values are the same.

        Args:
            states: (N, S, B) RNN states, S RNN steps of B episodes.
        Returns:
            (N,) tensor in the states' dtype on their device.
        """
        tau = float(self.frm_args.get("tau", 0.1))
        N, chunk = states.size(0), PARTICIPATION_CHUNK
        x = states.detach()
        n = torch.as_tensor(float(x.shape[1] * x.shape[2]), device=x.device, dtype=x.dtype)
        out = torch.empty(N, device=x.device, dtype=x.dtype)
        for i in range(0, N, chunk):
            blk = self.RNN.get_firing_rates(x[i:i + chunk].reshape(min(chunk, N - i), -1))
            out[i:i + chunk] = tau * (torch.logsumexp(blk / tau, dim=1) - torch.log(n))
        return out

    def rescale_rows(self, idx, alpha, normalize, live=None):
        """Tilt the incoming rows of the given units from inhibition toward excitation: the
        supervised Trainer.rescale_rows_ with rescale_norm_mode "self". In W_rec[idx, :] and
        W_inp[idx, :] the entries above 0 are multiplied by alpha and the others divided by it, the
        recurrent row only in the columns `live` marks when it is given, and with `normalize` each
        row is then scaled back to the L2 norm it had before, so the tilt moves weight from
        inhibition to excitation at a fixed total synaptic weight. The supervised lines:
            before = blk.norm(dim=1, keepdim=True)
            tilted = torch.where(blk > 0, blk * alpha, blk / alpha)
            blk = torch.where(restrict.unsqueeze(0), tilted, blk)      # with `live`; else tilted
            blk = blk * (before / blk.norm(dim=1, keepdim=True).clamp_min(1e-12))   # with normalize
        The input row is always tilted in full: every input is a live source.

        Args:
            idx: (k,) long tensor of unit indices; alpha: float > 1; normalize: bool;
            live: (N,) bool tensor over source units, or None for the whole recurrent row.
        Returns:
            None; W_rec and W_inp are changed in place (call under torch.no_grad()).
        """
        for W, restrict in ((self.RNN.W_rec, live), (self.RNN.W_inp, None)):
            blk = W[idx, :]
            before = blk.norm(dim=1, keepdim=True)
            tilted = torch.where(blk > 0, blk * alpha, blk / alpha)
            blk = tilted if restrict is None else torch.where(restrict.unsqueeze(0), tilted, blk)
            if normalize:
                blk = blk * (before / blk.norm(dim=1, keepdim=True).clamp_min(1e-12))
            W[idx, :] = blk
        return None

    def zero_protected_grads(self):
        """Blank the task gradient on the incoming rows W_rec[i, :] and W_inp[i, :] of the units the
        rescale rule is growing: the supervised Trainer.zero_protected_grads_. ppo_update calls it
        in every minibatch after backward() and the gradient clip and before the optimizer step,
        where the supervised train_step calls it once per iteration between its clip and its step.
        The outgoing weights W_rec[:, i] and W_out[:, i] keep their gradient, and Adam's weight
        decay is not suspended (step() adds it after this). Nothing happens with prune_reinit off.

        Returns:
            None; .grad of W_rec and W_inp changed in place.
        """
        if not self.prune_reinit:
            return None
        idx = torch.nonzero(self._rescale_growing, as_tuple=True)[0]
        if idx.numel() == 0:
            return None
        for W in (self.RNN.W_rec, self.RNN.W_inp):
            if W.grad is not None:
                W.grad[idx, :] = 0.0
        return None

    def enforce_inp_cap(self):
        """Clamp |W_inp| to the model's inp_weight_cap: the supervised Trainer.enforce_inp_cap_,
            cap = getattr(self.RNN, "inp_weight_cap", None);  if cap is None: return None
            with torch.no_grad(): self.RNN.W_inp.clamp_(min=-cap, max=cap)
        which the supervised train_step calls last in every iteration, after the optimizer step,
        the per-unit rule and the projections; ppo_update calls it after every optimizer step and
        the rule's check. Nothing happens without a cap (model.inp_weight_cap null, every model
        config's value).

        Returns:
            None; W_inp changed in place.
        """
        cap = self.inp_weight_cap
        if cap is None:
            return None
        with torch.no_grad():
            self.RNN.W_inp.clamp_(min=-cap, max=cap)
        return None

    def prune_and_reinit(self, states):
        """One check of the per-unit rule: the supervised Trainer.prune_and_reinit_ in reinit_mode
        "rescale" with rescale_target_frac > 0 (the three-state rule dormant -> growing -> mature),
        in reinit_mode "copy" (prune-and-duplicate), or in reinit_mode "random" or "zero_out" (the
        redraw of a silent unit's incoming weights), step for step and in the same order. The
        supervised rule runs once per iteration, after the optimizer step, on the noisy training
        batch of that iteration; this one runs once per gradient step, in ppo_update after every
        optimizer step, on the states `sequence` recomputed for that minibatch (noisy, every RNN
        step of every decision of the minibatch's episodes, under the weights before the step).

        WHAT COUNTS OPTIMIZER STEPS HERE AND ITERATIONS THERE. check_every (a check every
        check_every-th step, the first at step_n = 0); patience (consecutive silent checks, one
        check per step); maturity (steps since the unit was last doomed, from _last_replaced); the
        participation p and the soft-max activity (one minibatch's recomputed states, n_episodes /
        n_minibatches episodes, in place of one iteration's batch); the tilt (once per step, as
        there, so rescale_alpha is a per-step factor and rescale_cap is reached after
        log(cap) / log(alpha) steps, 2,083 at the paper's values); the gradient blanking (once per
        step, zero_protected_grads, as there). A minibatch whose loss is not finite is skipped
        without an optimizer step and without a check (the supervised rule also runs after a
        skipped step, but the PPO skip means the states themselves are not finite). The random
        draws of the outgoing redraw, of the donors and of the jitter come from the RNN's
        generator, as there. At the paper's copy values (check_every 100, patience 3, maturity
        1,000) a 2,000-update run of 8 steps per update makes 160 checks, a unit is doomed after
        300 silent steps (37.5 updates) and is protected for 125 updates after a replacement.

        The steps common to every mode, with the supervised lines:
          participation p = participation_from_states(states) (std + q0.9 of the rate);
            silent = p < active_rel * torch.quantile(p, 0.95)
          the contribution utility, every check:
            out_w = W_rec.abs().sum(dim=0) + W_out.abs().sum(dim=0)
            u <- utility_decay * u + (1 - utility_decay) * p * out_w;  score = u if utility ==
            "contribution" else p
          strikes = where(silent, strikes + 1, 0); mature = (step_n - last_replaced) >= maturity;
            doomed = (strikes >= patience) & mature; return if none is doomed (and, in mode
            "rescale", none is growing);
          at most round(max_replace_frac * N) doomed per check, the lowest score first.
        Mode "copy":
          every doomed unit is overwritten with a jittered copy of a live unit (`duplicate_units`,
            which quotes the lines); when no unit is live nothing is changed and the strikes are
            not reset, as there;
          Adam's exp_avg and exp_avg_sq on the incoming rows W_rec[i, :] and W_inp[i, :] and on
            the outgoing columns W_rec[:, i] and W_out[:, i] of every doomed unit set to 0 (the
            supervised blocks `mode != "rescale"` and `mode in ("copy", "rescale")`); the donor's
            halved columns keep their moments, as there;
          strikes[doomed] = 0; last_replaced[doomed] = step_n; ever[doomed] = True.
        Modes "random" and "zero_out":
          every doomed unit's incoming rows are redrawn and, in mode "zero_out", its outgoing
            columns zeroed (`redraw_units`, which quotes the lines);
          Adam's exp_avg and exp_avg_sq on the incoming rows W_rec[i, :] and W_inp[i, :] of every
            doomed unit set to 0 (the supervised block `mode != "rescale"`); the outgoing columns
            keep their moments in both modes, as there;
          strikes[doomed] = 0; last_replaced[doomed] = step_n; ever[doomed] = True.
        Mode "rescale":
          1) GRADUATE: a growing unit whose frm_activity reaches target = rescale_target_frac *
             frm_activity_cap() stops growing;
          2) OPEN an episode for every doomed unit not growing, up to round(rescale_protect_frac *
             N) growing at once, the lowest score first: W_rec[:, i] and W_out[:, i] redrawn from
             N(0, 1/sqrt(N)) once, cum[i] = 1, growing[i] = True;
          3) SPEND: a growing unit with cum >= rescale_cap stops growing (tested before the boost,
             so cum ends at most rescale_cap * rescale_alpha); then BOOST every growing unit:
             rescale_rows(grow, rescale_alpha, rescale_normalize, live = ~silent if
             rescale_active_only else None); cum *= rescale_alpha;
          Adam's exp_avg and exp_avg_sq on the outgoing columns W_rec[:, i] and W_out[:, i] of every
            doomed unit set to 0 (every check the unit is doomed, not only the one that opens its
            episode: the supervised block zeroes `idx`, the doomed indices);
          strikes[doomed] = 0; last_replaced[doomed] = step_n.

        Args:
            states: (N, S, B) RNN states under the training noise, detached: in ppo_update the
                states `sequence` returned for the minibatch, every RNN step of every decision of
                its B episodes including the (T + 1)-th decision's bootstrap steps and the steps
                after an episode ended, as the rate penalty pools them; S = (T + 1) * substeps.
        Returns:
            dict of ints. Mode "rescale": rescale_growing (units growing after this check),
            rescale_opened (episodes opened at this check), rescale_graduated (units that reached
            the target at this check) and rescale_spent (units whose cumulative factor reached the
            cap at this check); zeros apart from rescale_growing when the check is skipped. Mode
            "copy": copy_replaced (units overwritten at this check; 0 when the check is skipped)
            and copy_units_ever (distinct units overwritten so far in the run). Modes "random" and
            "zero_out": reinit_replaced and reinit_units_ever, the same two counts. The call
            advances step_n by one.
        """
        args = self.prune_args
        mode = args["reinit_mode"]
        n_done = self.step_n
        self.step_n += 1
        if mode == "copy":
            counts = {"copy_replaced": 0, "copy_units_ever": int(self._reinit_ever.sum())}
        elif mode in ("random", "zero_out"):
            counts = {"reinit_replaced": 0, "reinit_units_ever": int(self._reinit_ever.sum())}
        else:
            counts = {"rescale_growing": int(self._rescale_growing.sum()), "rescale_opened": 0,
                      "rescale_graduated": 0, "rescale_spent": 0}
        if n_done % int(args["check_every"]) != 0:
            return counts
        p = self.participation_from_states(states).detach()
        silent = p < float(args["active_rel"]) * torch.quantile(p, 0.95)
        with torch.no_grad():
            out_w = self.RNN.W_rec.abs().sum(dim=0) + self.RNN.W_out.abs().sum(dim=0)
            d = float(args.get("utility_decay", 0.99))
            self._unit_utility = d * self._unit_utility + (1.0 - d) * p * out_w
        score = (self._unit_utility if args.get("utility", "participation") == "contribution" else p)
        self._reinit_strikes = torch.where(silent, self._reinit_strikes + 1, torch.zeros_like(self._reinit_strikes))
        mature = (n_done - self._last_replaced) >= int(args.get("maturity", 0))
        doomed = (self._reinit_strikes >= int(args["patience"])) & mature
        n = int(doomed.sum())
        if n == 0 and not (mode == "rescale" and bool(self._rescale_growing.any())):
            return counts
        cap = max(1, int(round(float(args.get("max_replace_frac", 1.0)) * self.RNN.N)))
        if n > cap:
            cand = torch.nonzero(doomed, as_tuple=True)[0]
            keep = cand[torch.argsort(score[cand])[:cap]]
            doomed = torch.zeros_like(doomed)
            doomed[keep] = True
            n = cap
        idx = torch.nonzero(doomed, as_tuple=True)[0]
        if mode == "copy":
            with torch.no_grad():
                if self.duplicate_units(idx, p, silent) == 0:
                    return counts
                # Adam's moments on the copies' incoming rows and outgoing columns
                for prm, where in ((self.RNN.W_rec, "rows"), (self.RNN.W_inp, "rows"),
                                   (self.RNN.W_rec, "cols"), (self.RNN.W_out, "cols")):
                    st = self.optimizer.state.get(prm, None)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                if where == "rows":
                                    st[key][idx, :] = 0.0
                                else:
                                    st[key][:, idx] = 0.0
            self._reinit_strikes[doomed] = 0
            self._last_replaced[doomed] = float(n_done)
            self._reinit_ever[doomed] = True
            counts.update(copy_replaced=n, copy_units_ever=int(self._reinit_ever.sum()))
            return counts
        if mode in ("random", "zero_out"):
            with torch.no_grad():
                self.redraw_units(idx, mode)
                # Adam's moments on the redrawn incoming rows only (the supervised block
                # `mode != "rescale"`); the outgoing columns keep theirs in both modes, as there
                for prm in (self.RNN.W_rec, self.RNN.W_inp):
                    st = self.optimizer.state.get(prm, None)
                    if st:
                        for key in ("exp_avg", "exp_avg_sq"):
                            if key in st:
                                st[key][idx, :] = 0.0
            self._reinit_strikes[doomed] = 0
            self._last_replaced[doomed] = float(n_done)
            self._reinit_ever[doomed] = True
            counts.update(reinit_replaced=n, reinit_units_ever=int(self._reinit_ever.sum()))
            return counts
        std = 1.0 / np.sqrt(self.RNN.N)
        a = float(args.get("rescale_alpha", 1.0005))
        cum_cap = float(args.get("rescale_cap", 8.0))
        protect_cap = int(round(float(args.get("rescale_protect_frac", 0.20)) * self.RNN.N))
        with torch.no_grad():
            target = float(args["rescale_target_frac"]) * self.frm_activity_cap()
            activity = self.frm_activity(states)
            # 1) GRADUATE
            graduated = self._rescale_growing & (activity >= target)
            self._rescale_growing[graduated] = False
            # 2) OPEN an episode for doomed units not already growing, up to the ceiling
            room = max(0, protect_cap - int(self._rescale_growing.sum()))
            fresh = doomed & ~self._rescale_growing
            cand = torch.nonzero(fresh, as_tuple=True)[0]
            if cand.numel() > room:
                cand = cand[torch.argsort(score[cand])[:room]]
            live_mask = ~silent if bool(args.get("rescale_active_only", False)) else None
            if cand.numel() > 0:
                self.RNN.W_rec[:, cand] = torch.randn(self.RNN.N, cand.numel(), device=self.RNN.device,
                                                      generator=self.RNN.random_generator) * std
                self.RNN.W_out[:, cand] = torch.randn(self.RNN.W_out.shape[0], cand.numel(), device=self.RNN.device,
                                                      generator=self.RNN.random_generator) * std
                self._rescale_cum[cand] = 1.0
                self._rescale_growing[cand] = True
            # 3) SPEND, then BOOST every unit still growing
            spent = self._rescale_growing & (self._rescale_cum >= cum_cap)
            self._rescale_growing[spent] = False
            grow = torch.nonzero(self._rescale_growing, as_tuple=True)[0]
            if grow.numel() > 0:
                self.rescale_rows(grow, a, bool(args.get("rescale_normalize", True)), live=live_mask)
                self._rescale_cum[grow] *= a
            # Adam's moments on the outgoing columns of the doomed units
            for prm in (self.RNN.W_rec, self.RNN.W_out):
                st = self.optimizer.state.get(prm, None)
                if st:
                    for key in ("exp_avg", "exp_avg_sq"):
                        if key in st:
                            st[key][:, idx] = 0.0
        self._reinit_strikes[doomed] = 0
        self._last_replaced[doomed] = float(n_done)
        counts.update(rescale_growing=int(self._rescale_growing.sum()), rescale_opened=int(cand.numel()),
                      rescale_graduated=int(graduated.sum()), rescale_spent=int(spent.sum()))
        return counts

    def log_prob(self, action, mean):
        """Log density of actions under the diagonal Gaussian N(mean, action_std^2).

        Args:
            action, mean: (2, ...) tensors of the same shape.
        Returns:
            (...) tensor, the log density summed over the two action outputs.
        """
        z = (action - mean) / self.action_std
        return (-0.5 * z ** 2 - math.log(self.action_std) - 0.5 * LOG_2PI).sum(dim=0)

    def entropy(self):
        """Entropy of the Gaussian policy, a constant of the fixed std: a scalar tensor."""
        return torch.full((), 2 * (math.log(self.action_std) + 0.5 * (LOG_2PI + 1.0)), device=self.RNN.device)

    # ------------------------------------------------------------------ rollouts and advantages

    def rollout(self, n_episodes, T, clean=False, generator=None):
        """Play n_episodes episodes of at most T steps in parallel, without gradients.

        Args:
            n_episodes: int, the batch B; T: int, steps per episode.
            clean: bool, True for a read-out pass: the RNN's noise off and the policy mean as the
                action; False (training) draws the noise and samples the actions.
            generator: torch.Generator on the task's device for the drops, the RNN noise and the
                action noise, or None for the task's and the RNN's own generators and torch's
                global one.
        Returns:
            dict: u (n_inputs, T+1, B); noise, the dict from draw_noise for (T+1) * substeps RNN
            steps; actions (2, T, B); logp (T, B); values (T+1, B); rewards (T, B); alive (T+1, B)
            float; success (B,) bool, reached the cheese; steps (B,) float, steps taken until the
            cheese (T if never); xy (T+1, B, 2) positions and heading (T+1, B) radians, for figures
            and animations; w_seed, the synaptic-noise seed (int) of this rollout, None when
            sigma_w is 0 or the pass is clean; outgoing, the read-out dropout mask (N, 1) of this
            rollout, None when dropout is off or the pass is clean; participation (N,), std +
            q(activity_q) of every unit's rate over every RNN step of every decision of this
            rollout, the new value for the EMA p_bar, None unless a mask was drawn. The RNN states
            are not kept (they are large; with dropout on they are held until the participation is
            taken and released); sequence(u, noise, w_seed, outgoing) recomputes them exactly.

        THE TWO INTERVENTIONS THAT ACT HERE are drawn before anything else, in this order: the
        synaptic-noise seed (one integer from the RNN's generator) and the dropout mask (Bernoulli
        draws from the RNN's generator), each only when its option is on and the pass is noisy.
        The policy that acts is then the dropped-out, noisy-synapse policy, and ppo_update replays
        both, so the probability ratio starts at 1 as it does without them.
        """
        task, k = self.Task, self.substeps
        with torch.no_grad():
            w_seed = self.draw_synaptic_seed(generator) if (self.sigma_w > 0 and not clean) else None
            outgoing = self.draw_dropout_mask() if (self.dropout and not clean) else None
            keep_states = outgoing is not None
            u = [task.reset(n_episodes, generator)]
            noise = self.draw_noise((T + 1) * k, n_episodes, not clean, generator)
            x = self.initial_state(n_episodes)
            alive = [torch.ones(n_episodes, device=task.device)]
            xy, heading = [task.xy.clone()], [task.heading.clone()]
            actions, logp, values, rewards, states = [], [], [], [], []
            for t in range(T):
                live = alive[-1].bool()[None, :]                  # (1, B): the episode has not ended
                for s in range(t * k, (t + 1) * k):
                    x_new = self.advance(x, u[-1], noise["rec"][:, s], noise["inp"][:, s], w_seed, s)
                    x = torch.where(live, x_new, x) if self.freeze_after_end else x_new
                    if keep_states:
                        states.append(x)
                mean, value = self.heads(x, noise["out"][:, 0], outgoing)
                if clean:
                    action = mean
                else:
                    action = mean + self.std()[:, None] * torch.randn(
                        mean.shape, generator=generator, device=mean.device)
                obs, reward, done = task.step(action)
                actions.append(action)
                logp.append(self.log_prob(action, mean))
                values.append(value)
                rewards.append(reward)
                u.append(obs)
                alive.append((~done).float())
                xy.append(task.xy.clone())
                heading.append(task.heading.clone())
            live = alive[-1].bool()[None, :]
            for s in range(T * k, (T + 1) * k):
                x_new = self.advance(x, u[-1], noise["rec"][:, s], noise["inp"][:, s], w_seed, s)
                x = torch.where(live, x_new, x) if self.freeze_after_end else x_new
                if keep_states:
                    states.append(x)
            values.append(self.heads(x, noise["out"][:, 0], outgoing)[1])
            p_new = None
            if keep_states:
                p_new = self.participation_from_states(torch.stack(states, dim=1),
                                                       float(self.dropout_args["activity_q"]))
        alive = torch.stack(alive)
        return {"u": torch.stack(u, dim=1), "noise": noise, "actions": torch.stack(actions, dim=1),
                "logp": torch.stack(logp), "values": torch.stack(values), "rewards": torch.stack(rewards),
                "alive": alive, "success": alive[-1] == 0, "steps": alive[:-1].sum(dim=0),
                "xy": torch.stack(xy), "heading": torch.stack(heading),
                "w_seed": w_seed, "outgoing": outgoing, "participation": p_new}

    def safe_rollout(self, good, n_episodes, T, tries=3, snapshot=None):
        """A rollout that is finite, or the last finite weights restored and the rollout replayed,
        or, failing that, the last checkpoint's weights restored.

        The maze counterpart of the supervised trainer's rollback: at N = 2,000 the recurrent dynamics
        blew up (states to infinity over the 2,500-step episode) after a few updates at the paper's
        learning rate (arrays 16856487 and 16859389, 4 Nov 2026), and every later loss was skipped.
        Here a rollout whose log-probabilities or rewards are not finite is discarded, the weights are
        restored from `good` (the RNN's state_dict after the last update whose rollout was finite),
        and the rollout is played again with fresh noise. When `tries` replays from `good` all fail,
        the weights after the last update are themselves unstable (the N = 2,000 seed-3 runs of
        5 Nov died this way at updates 341 and 1,225); then, as the supervised trainer restores its
        last snapshot after repeated skips, the weights of the last checkpoint (`snapshot`, up to
        checkpoint_every updates earlier) are restored and `tries` more replays are made. The
        optimizer's moments are kept as they are in both cases.

        Args:
            good: dict, RNN.state_dict() tensors (copied); n_episodes, T: ints; tries: int,
            attempts from `good` and then from `snapshot`; snapshot: dict like `good` from the
            last checkpoint, or None.
        Returns:
            (batch, restored): the finite rollout dict and the number of restores; a restore from
            the snapshot counts tries + 1 or more, so a reader can tell the two apart.
        Raises:
            RuntimeError if no finite rollout was obtained.
        """
        restored = 0
        # WHERE each discarded rollout went non-finite (8 Nov 2026): "post_episode" when every step of a
        # running episode was finite and only steps after an episode's end were not (the RNN keeps
        # stepping after the cheese unless freeze_after_end), "running" otherwise; read by run_training
        self.restore_kinds = {"post_episode": 0, "running": 0}
        for source in ([good] if snapshot is None else [good, snapshot]):
            for _ in range(tries):
                batch = self.rollout(n_episodes, T)
                if torch.isfinite(batch["logp"]).all() and torch.isfinite(batch["rewards"]).all():
                    return batch, restored
                live = batch["alive"][:-1] > 0
                ok_live = bool(torch.isfinite(batch["logp"][live]).all() and torch.isfinite(batch["rewards"][live]).all())
                self.restore_kinds["post_episode" if ok_live else "running"] += 1
                self.RNN.load_state_dict(source)
                restored += 1
        raise RuntimeError(f"the rollout was not finite {restored} times in a row; the run is stopped")

    @staticmethod
    def mc_advantages(rewards, alive, gamma):
        """Critic-free advantages: the discounted reward-to-go minus its mean over the alive episodes
        at the same step (a baseline with no parameters). An episode cut off at T gets no bootstrap.

        Args:
            rewards: (T, B); alive: (T+1, B) float, see the module docstring; gamma: float.
        Returns:
            (advantages, returns), both (T, B), zero at steps that are not alive.
        """
        T = rewards.shape[0]
        ret = torch.zeros_like(rewards)
        last = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            last = rewards[t] + gamma * alive[t + 1] * last
            ret[t] = last * alive[t]
        m = alive[:T]
        baseline = (ret * m).sum(dim=1, keepdim=True) / m.sum(dim=1, keepdim=True).clamp_min(1.0)
        return (ret - baseline) * m, ret

    @staticmethod
    def gae(rewards, values, alive, gamma, lam):
        """Generalised advantage estimation over a batch of episodes.

        Args:
            rewards: (T, B); values: (T+1, B), the last row bootstraps episodes still running at T;
            alive: (T+1, B) float, see the module docstring; gamma, lam: floats, discount and GAE lambda.
        Returns:
            (advantages, returns), both (T, B), zero at steps that are not alive.
        """
        T = rewards.shape[0]
        adv = torch.zeros_like(rewards)
        last = torch.zeros_like(rewards[0])
        for t in reversed(range(T)):
            cont = alive[t + 1]                      # the episode goes on after step t
            delta = rewards[t] + gamma * cont * values[t + 1] - values[t]
            last = delta + gamma * lam * cont * last
            adv[t] = last * alive[t]
        return adv, adv + values[:T] * alive[:T]

    # ------------------------------------------------------------------ the update

    def ppo_update(self, batch, adv, ret):
        """The clipped PPO update on one rollout: n_epochs passes over the episodes in n_minibatches
        minibatches, each a full-sequence recompute (backprop through time) with the stored noise.

        Args:
            batch: the dict from `rollout`; adv, ret: (T, B) from `gae` or `mc_advantages`.
        Returns:
            dict of floats averaged over the minibatch steps: policy_loss, value_loss, entropy,
            approx_kl (mean of old minus new log-probability), clip_frac (share of steps with the
            ratio outside the clip range), skipped, the number of minibatch steps whose loss was
            not finite and which were therefore not applied, and, only when the option is on, the
            raw penalty values fr_magnitude (lambda_frm > 0), rec_weights_sparsity
            (lambda_rws > 0), rec_weights_magnitude (lambda_rwm > 0) and metabolic
            (lambda_met > 0), before their lambda; with the rescale rule on, four ints:
            rescale_opened, rescale_graduated and rescale_spent summed over this update's checks
            (one per optimizer step) and rescale_growing, the number growing after the last check;
            with the copy rule on, two ints: copy_replaced summed over the update's checks and
            copy_units_ever, the distinct units replaced so far, after the last check; with a
            redraw mode on, two ints the same way: reinit_replaced and reinit_units_ever.

        THE PENALTIES are added to the loss of every minibatch as the supervised train_step adds
        them, loss += lambda_frm x P_frm + lambda_rws x P_rws + lambda_rwm x P_rwm + lambda_met x
        P_met, with P_frm and P_met on the states of this minibatch's recompute (the noisy rollout
        states) and P_rws and P_rwm on W_rec. The recompute replays the rollout's synaptic noise
        (batch["w_seed"]) and read-out mask (batch["outgoing"]). With the rescale rule on, every
        minibatch's gradient is blanked on the incoming rows of the units growing at that time
        (zero_protected_grads) after the clip and before the step, and after the step the rule
        checks once (prune_and_reinit) on this minibatch's recomputed states, detached; after the
        check, |W_inp| is clamped to the model's cap when it has one (enforce_inp_cap): the
        supervised order of one iteration, forward, backward, blank, step, rule, cap, at every
        optimizer step.

        TWO GUARDS, both inert while everything is finite (added 4 Nov 2026 after two N = 2,000 runs
        died in their first update, array 16856487): the log-ratio is clamped to [-20, 20] before it
        is exponentiated, so an action that the updated policy finds very unlikely gives a ratio of
        e^-20 instead of 0 or inf; and a minibatch whose loss is not finite is skipped, so one bad
        pass cannot write NaN into the weights for the rest of the run.
        """
        clip = self.clip
        T, B = adv.shape
        alive = batch["alive"][:T]
        adv = (adv - adv[alive > 0].mean()) / (adv[alive > 0].std() + 1e-8)
        sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
        if self.lambda_frm:
            sums["fr_magnitude"] = 0.0
        if self.lambda_rws:
            sums["rec_weights_sparsity"] = 0.0
        if self.lambda_rwm:
            sums["rec_weights_magnitude"] = 0.0
        if self.lambda_met:
            sums["metabolic"] = 0.0
        n = 0
        skipped = 0
        rule = {}
        for _ in range(self.n_epochs):
            for idx in torch.randperm(B, device=adv.device).chunk(self.n_minibatches):
                noise = {k: v[:, :, idx] for k, v in batch["noise"].items()}
                states, means, values = self.sequence(batch["u"][:, :, idx], noise,
                                                      batch["w_seed"], batch["outgoing"],
                                                      batch["alive"][:, idx])
                live = self.live_steps(batch["alive"][:, idx]) if self.freeze_after_end else None
                logp = self.log_prob(batch["actions"][:, :, idx], means[:, :T])
                m = alive[:, idx]
                ratio = (logp - batch["logp"][:, idx]).clamp(-20.0, 20.0).exp()
                a = adv[:, idx]
                surrogate = torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
                policy_loss = -(surrogate * m).sum() / m.sum()
                value_loss = (0.5 * ((values[:T] - ret[:, idx]) ** 2 * m).sum() / m.sum()
                              if self.critic else torch.zeros((), device=adv.device))
                entropy = self.entropy()
                loss = policy_loss + self.value_coef * value_loss - self.entropy_coef * entropy
                penalties = {}
                if self.lambda_frm:
                    penalties["fr_magnitude"] = self.fr_magnitude_penalty(states, live)
                    loss = loss + self.lambda_frm * penalties["fr_magnitude"]
                if self.lambda_rws:
                    penalties["rec_weights_sparsity"] = self.rec_weights_sparsity_penalty()
                    loss = loss + self.lambda_rws * penalties["rec_weights_sparsity"]
                if self.lambda_rwm:
                    penalties["rec_weights_magnitude"] = self.rec_weights_magnitude_penalty()
                    loss = loss + self.lambda_rwm * penalties["rec_weights_magnitude"]
                if self.lambda_met:
                    penalties["metabolic"] = self.metabolic_penalty(states, live)
                    loss = loss + self.lambda_met * penalties["metabolic"]
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.RNN.parameters(), self.max_grad_norm)
                self.zero_protected_grads()
                self.optimizer.step()
                if self.prune_reinit:
                    counts = self.prune_and_reinit(states.detach())
                    rule = {k: (v if k in RULE_LEVEL_KEYS else rule.get(k, 0) + v) for k, v in counts.items()}
                self.enforce_inp_cap()
                with torch.no_grad():
                    sums["policy_loss"] += policy_loss.item()
                    sums["value_loss"] += value_loss.item()
                    sums["entropy"] += entropy.item()
                    sums["approx_kl"] += ((batch["logp"][:, idx] - logp) * m).sum().item() / m.sum().item()
                    sums["clip_frac"] += (((ratio - 1).abs() > clip).float() * m).sum().item() / m.sum().item()
                    for key, val in penalties.items():
                        sums[key] += val.item()
                n += 1
        out = {k: v / max(n, 1) for k, v in sums.items()}
        out["skipped"] = float(skipped)
        out.update(rule)
        return out

    # ------------------------------------------------------------------ the probe and the run

    def probe(self):
        """A noise-free, mean-action rollout on the fixed probe drops (probe_episodes, T, probe_seed).

        Returns:
            (n_active, n_active_mod, p, out): n_active int, active units under common.active_mask on
            p; n_active_mod int, the same bar on the std alone (the modulation criterion); p (N,)
            float participation over every RNN step of the probe before the cheese; out, the rollout.
        """
        g = torch.Generator(device=self.Task.device)
        g.manual_seed(self.probe_seed)
        T, k = self.T, self.substeps
        out = self.rollout(self.probe_episodes, T, clean=True, generator=g)
        with torch.no_grad():
            states = self.sequence(out["u"], out["noise"], alive=out["alive"])[0][:, :T * k]   # (N, T * k, B)
        rates = self.RNN.get_firing_rates(states).cpu().numpy()
        alive = out["alive"][:-1].bool().repeat_interleave(k, dim=0).cpu().numpy()  # (T * k, B)
        p = participation(rates[:, alive])
        n_active_mod = int(active_mask(rates[:, alive].std(axis=1)).sum())
        return int(active_mask(p).sum()), n_active_mod, p, out

    def run_training(self):
        """Train for n_updates PPO updates, probing and checkpointing on the way.

        Every update: the learning rate for this update (the warm-up ramp, unless the Adam moments
        were restored), a finite rollout
        (safe_rollout, rolling back to the last finite weights if needed), with dropout on the
        participation EMA p_bar updated from that rollout's states as the supervised train_step
        updates it (`self.participation = (1 - eta) * self.participation + eta * new_part`, after the
        mask of the same batch was drawn from the old value), advantages (GAE with a critic,
        Monte-Carlo without), the clipped update (with the per-unit rule on in any mode, one check
        of prune_and_reinit after each of its optimizer steps), and one progress record. The first
        update and every probe_every-th one are followed by a probe. Every checkpoint_every-th
        update, except the last (the caller saves the finished run), calls
        checkpoint_fn(updates_done, params) if one was set.

        Returns:
            (RNN, progress, probes, last_probe): the trained RNN; progress, a list with one dict per
            update: update (iter_offset + this run's own count, so a continuation's records continue
            its parent's), discounted_return (mean over the batch of sum_t gamma^t r_t, the quantity
            PPO maximises), success (share of the batch that reached the cheese), steps (mean steps
            to the cheese, T if never), restored (rollouts discarded and weights rolled back), lr
            (the rate used, after the warm-up ramp), the statistics of ppo_update (including
            skipped minibatches, when a penalty is on its raw value, and with the rescale rule on
            its counts of that update: rescale_opened, rescale_graduated and rescale_spent summed
            over the update's checks, rescale_growing after the last one; with the copy rule on
            copy_replaced summed over the checks and copy_units_ever after the last one; with a
            redraw mode on reinit_replaced and reinit_units_ever the same way); probes,
            a dict of lists
            over probes: update (the update index, offset the same way),
            active, active_mod, probe_success, and p_update and p, the stored participation vectors
            (N,) with their update index; last_probe, the last probe's rollout dict.
        """
        hp = self
        progress = []
        probes = {"update": [], "active": [], "active_mod": [], "probe_success": [], "p_update": [], "p": []}
        self.progress, self.probes, self.last_probe = progress, probes, None
        discount = hp.gamma ** torch.arange(hp.T, device=self.device, dtype=torch.float32)[:, None]
        good = {k: v.detach().clone() for k, v in self.RNN.state_dict().items()}   # last finite weights
        snapshot = {k: v.clone() for k, v in good.items()}                         # the last checkpoint's
        tic = time.perf_counter()
        for it in range(1, hp.n_updates + 1):
            ramp = hp.warmup_updates and not self.adam_restored
            lr_now = self.lr * min(1.0, it / hp.warmup_updates) if ramp else self.lr
            for group in self.optimizer.param_groups:
                group["lr"] = lr_now
            batch, restored = self.safe_rollout(good, hp.n_episodes, hp.T, snapshot=snapshot)
            if restored:
                print(f"update {it}: rollout not finite, weights restored ({restored} tries"
                      f"{', from the last checkpoint' if restored > 3 else ''}; non-finite after an episode's "
                      f"end only in {self.restore_kinds['post_episode']}, within a running episode in "
                      f"{self.restore_kinds['running']})", flush=True)
            if batch["participation"] is not None:
                eta = float(self.dropout_args.get("eta", 0.5))
                self.p_bar = (1 - eta) * self.p_bar + eta * batch["participation"]
            if hp.critic:
                adv, ret = self.gae(batch["rewards"], batch["values"], batch["alive"], hp.gamma, hp.gae_lambda)
            else:
                adv, ret = self.mc_advantages(batch["rewards"], batch["alive"], hp.gamma)
            stats = self.ppo_update(batch, adv, ret)
            progress.append({"update": hp.iter_offset + it,
                             "discounted_return": (batch["rewards"] * discount).sum(dim=0).mean().item(),
                             "success": batch["success"].float().mean().item(),
                             "steps": batch["steps"].mean().item(),
                             "restored": restored,
                             "restored_post_episode": self.restore_kinds["post_episode"],
                             "lr": lr_now,
                             **stats})
            good = {k: v.detach().clone() for k, v in self.RNN.state_dict().items()}
            if it == 1 or it % hp.probe_every == 0:
                n_active, n_active_mod, p, out = self.probe()
                self.last_probe = out
                probes["update"].append(hp.iter_offset + it)
                probes["active"].append(n_active)
                probes["active_mod"].append(n_active_mod)
                probes["probe_success"].append(out["success"].float().mean().item())
                if (len(probes["update"]) - 1) % hp.store_participation_every == 0:
                    probes["p_update"].append(hp.iter_offset + it)
                    probes["p"].append(p)
                r = progress[-1]
                print(f"update {it:5d}/{hp.n_updates}  disc. return {r['discounted_return']:7.3f}"
                      f"  success {r['success']:.2f}  steps {r['steps']:6.1f}"
                      f"  probe success {probes['probe_success'][-1]:.2f}"
                      f"  active {n_active} (mod {n_active_mod})/{self.RNN.N}  kl {r['approx_kl']:.4f}  clip {r['clip_frac']:.2f}"
                      + (f"  growing {r['rescale_growing']}" if "rescale_growing" in r else "")
                      + (f"  replaced {r['copy_units_ever']}" if "copy_units_ever" in r else "")
                      + (f"  redrawn {r['reinit_units_ever']}" if "reinit_units_ever" in r else "")
                      + f"  {(time.perf_counter() - tic) / it:.2f} s/update", flush=True)
            if self.checkpoint_every and it % self.checkpoint_every == 0:
                snapshot = {k: v.clone() for k, v in good.items()}      # the fallback of safe_rollout
                if self.checkpoint_fn is not None and it < hp.n_updates:
                    self.checkpoint_fn(it, self.RNN.get_params())
        print(f"done: {hp.n_updates} updates in {time.perf_counter() - tic:.0f} s", flush=True)
        return self.RNN, progress, probes, self.last_probe


def load_run(folder, device=None):
    """Rebuild the task, the trained RNN and a trainer (to play it) from a maze run folder.

    Reads both layouts: the one run_rl.py writes (`<stem>_config.yaml`, `<stem>_Params_Maze.npz`,
    task and trainer under `task:` and `trainer:`) and the one trainRNNbrain/rl/run_maze_ppo.py wrote
    before 5 Nov 2026 (`config.yaml`, `Params_Maze.npz`, the task under `env:` and the algorithm
    under `ppo:`). The weights come from the parameter file in both (RNN_torch.get_params(): W_rec,
    W_inp, W_out and y_init; for the standard model's "sticky" weight boundary these are the raw
    parameters). Runs written before 4 Nov 2026 (no `action_feedback` key; a learned log-std and a
    separate value head, which were never part of the RNN) load too: the key defaults to False and
    the RNN's own read-out is what the clean rollout uses, as their training probe did (the maze has
    since gained the slide along walls and the bump cost on every hit, so such an agent plays a
    maze that differs from its training maze at the walls). Runs of the first array (no `substeps`
    key: one RNN step per decision, clipped actions, discounted shaping) are refused, as before.

    Args:
        folder: str, a run folder; device: torch.device, or None for the CPU.
    Returns:
        (cfg, task, trainer): the saved DictConfig (as saved: `task`/`trainer` or `env`/`ppo`
        blocks), the TaskMaze, and a TrainerPPO without an optimizer holding the trained RNN
        (trainer.rollout(...) plays it; trainer.T, probe_episodes, probe_seed, substeps carry the
        run's values whichever layout it has).
    """
    device = torch.device("cpu") if device is None else device
    cfg = OmegaConf.load(sorted(glob.glob(os.path.join(folder, "*config.yaml")))[0])
    old_layout = "env" in cfg
    task_cfg, hp = (cfg.env, cfg.ppo) if old_layout else (cfg.task, cfg.trainer)
    task_args = OmegaConf.to_container(filter_kwargs(TaskMaze, task_cfg), resolve=True)
    task_args.pop("_target_", None)
    task_args.setdefault("action_feedback", False)
    task_args["seed"] = int(cfg.seed)
    task = TaskMaze(**task_args, device=device)
    args = filter_kwargs(RNN_torch, cfg.model)
    args.seed = int(cfg.seed)
    rnn = hydra.utils.instantiate(args)
    rnn.to(device)
    params = np.load(sorted(glob.glob(os.path.join(folder, "*Params_*.npz")))[0], allow_pickle=True)
    with torch.no_grad():
        for name, prm in rnn.named_parameters():
            prm.copy_(torch.as_tensor(np.asarray(params[name], dtype=np.float32), device=prm.device))
    rnn.y_init = torch.as_tensor(np.asarray(params["y_init"], dtype=np.float32), device=device)
    hp_args = OmegaConf.to_container(filter_kwargs(TrainerPPO, hp), resolve=True)
    hp_args.pop("_target_", None)
    # trainer.optimizer (adam, adamw or sgd, since 5 Nov 2026) names the optimiser run_rl.py builds;
    # the trainer's own `optimizer` parameter is the object, None for a played agent
    hp_args.pop("optimizer", None)
    hp_args["critic"] = bool(hp.get("critic", rnn.n_outputs == 3))
    if "substeps" not in hp:
        # the constructor's default (5) would otherwise play these runs silently with the wrong clock
        raise ValueError(f"{folder}: no `substeps` in the saved config, so this is a run of the first "
                         f"array (16790695, 3 Nov 2026: one RNN step per decision, clipped actions, "
                         f"discounted shaping), trained in a task version the current TaskMaze does "
                         f"not reproduce; it cannot be rebuilt from current code")
    trainer = TrainerPPO(rnn, task, None, **hp_args)
    return cfg, task, trainer
