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
compared. Synaptic noise (sigma_w) is not implemented here and is refused at construction.

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
from omegaconf import OmegaConf
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.tasks.TaskMaze import TaskMaze
from trainRNNbrain.utils import filter_kwargs
from trainRNNbrain.experiments_and_analysis.common import participation, active_mask

LOG_2PI = math.log(2 * math.pi)


class TrainerPPO:
    """PPO on one TaskMaze with one RNN_torch as the agent (actor rows 0, 1; critic row 2 if built with it)."""

    def __init__(self, RNN, Task, optimizer,
                 n_episodes=256, T=500, substeps=5, n_updates=2000,
                 gamma=0.99, gae_lambda=0.95, clip=0.2, n_epochs=4, n_minibatches=2,
                 value_coef=0.5, entropy_coef=0.0, critic=False, action_std=0.5,
                 max_grad_norm=0.5, warmup_updates=50,
                 probe_every=10, probe_episodes=128, probe_seed=20261103,
                 store_participation_every=5, checkpoint_every=100):
        """Args:
            RNN: RNN_torch with n_outputs 2 + critic (turn, speed, and the value row with a
                critic) and sigma_w == 0, on the task's device.
            Task: TaskMaze.
            optimizer: torch optimizer over RNN.parameters(); its learning rate is the base rate
                the warm-up ramps to. None for a rebuilt agent that is only played (load_run).
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
        Raises:
            ValueError if RNN.n_outputs is not 2 + critic, or if RNN.sigma_w is not 0 (`step` runs
            the RNN's rhs with fixed weights and would silently drop the synaptic noise that
            RNN_torch.forward applies).
        """
        if RNN.n_outputs != 2 + int(bool(critic)):
            raise ValueError(f"the RNN needs {2 + int(bool(critic))} outputs with critic={critic} "
                             f"(turn, speed{', value' if critic else ''}); it has {RNN.n_outputs}")
        if RNN.sigma_w != 0.0:
            raise ValueError(f"TrainerPPO does not implement synaptic noise; model.sigma_w is {RNN.sigma_w}")
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

    def step(self, x, u, rec_t, inp_t):
        """One RNN step through the RNN's own right-hand side.

        Args:
            x: (N, B) state; u: (n_inputs, B) observation; rec_t: (N, B) and inp_t: (n_inputs, B),
            this step's recurrent and input noise.
        Returns:
            (N, B) next state.
        """
        W_rec, W_inp, _ = self.RNN._constrained_weights()
        return x + self.RNN.alpha * self.RNN.rhs(x=x, I=u, i_noise=inp_t, r_noise=rec_t,
                                                 W_rec=W_rec, W_inp=W_inp)

    def heads(self, x, out_noise):
        """Policy mean and value from RNN states: the RNN's read-out of the noisy rates.

        Args:
            x: (N, B) or (N, T, B) states; out_noise: (N, B) or (N, 1, B), the episode's output
            noise (broadcast over time).
        Returns:
            (mean, value): mean (2, B) or (2, T, B), read-out rows 0 and 1; value (B,) or (T, B),
            read-out row 2, or zeros without a critic.
        """
        r = self.RNN.get_firing_rates(x)
        W_out = self.RNN._constrained_weights()[2]
        out = torch.einsum("oj,j...->o...", W_out, r + out_noise)
        value = out[2] if self.critic else torch.zeros_like(out[0])
        return out[:2], value

    def sequence(self, u, noise):
        """Re-run the RNN over a stored observation sequence with stored noise (the PPO recompute).

        Args:
            u: (n_inputs, T, B) observations, one per decision;
            noise: dict from `draw_noise` for T * substeps RNN steps.
        Returns:
            (states, means, values): states (N, T * substeps, B), the state after every RNN step;
            means (2, T, B) and values (T, B), read from the state after the last substep of each
            decision.
        """
        T, B = u.shape[1], u.shape[2]
        x = self.initial_state(B)
        states = []
        for t in range(T):
            for s in range(t * self.substeps, (t + 1) * self.substeps):
                x = self.step(x, u[:, t], noise["rec"][:, s], noise["inp"][:, s])
                states.append(x)
        states = torch.stack(states, dim=1)
        means, values = self.heads(states[:, self.substeps - 1::self.substeps], noise["out"])
        return states, means, values

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
            and animations. The RNN states are not kept (they are large); sequence(u, noise)
            recomputes them exactly.
        """
        task, k = self.Task, self.substeps
        with torch.no_grad():
            u = [task.reset(n_episodes, generator)]
            noise = self.draw_noise((T + 1) * k, n_episodes, not clean, generator)
            x = self.initial_state(n_episodes)
            alive = [torch.ones(n_episodes, device=task.device)]
            xy, heading = [task.xy.clone()], [task.heading.clone()]
            actions, logp, values, rewards = [], [], [], []
            for t in range(T):
                for s in range(t * k, (t + 1) * k):
                    x = self.step(x, u[-1], noise["rec"][:, s], noise["inp"][:, s])
                mean, value = self.heads(x, noise["out"][:, 0])
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
            for s in range(T * k, (T + 1) * k):
                x = self.step(x, u[-1], noise["rec"][:, s], noise["inp"][:, s])
            values.append(self.heads(x, noise["out"][:, 0])[1])
        alive = torch.stack(alive)
        return {"u": torch.stack(u, dim=1), "noise": noise, "actions": torch.stack(actions, dim=1),
                "logp": torch.stack(logp), "values": torch.stack(values), "rewards": torch.stack(rewards),
                "alive": alive, "success": alive[-1] == 0, "steps": alive[:-1].sum(dim=0),
                "xy": torch.stack(xy), "heading": torch.stack(heading)}

    def safe_rollout(self, good, n_episodes, T, tries=3):
        """A rollout that is finite, or the last finite weights restored and the rollout replayed.

        The maze counterpart of the supervised trainer's rollback: at N = 2,000 the recurrent dynamics
        blew up (states to infinity over the 2,500-step episode) after a few updates at the paper's
        learning rate (arrays 16856487 and 16859389, 4 Nov 2026), and every later loss was skipped.
        Here a rollout whose log-probabilities or rewards are not finite is discarded, the weights are
        restored from `good` (the RNN's state_dict after the last update whose rollout was finite),
        and the rollout is played again with fresh noise.

        Args:
            good: dict, RNN.state_dict() tensors (copied); n_episodes, T: ints; tries: int,
            attempts before giving up.
        Returns:
            (batch, restored): the finite rollout dict and the number of restores (0 normally).
        Raises:
            RuntimeError if no finite rollout was obtained in `tries` attempts.
        """
        restored = 0
        for _ in range(tries):
            batch = self.rollout(n_episodes, T)
            if torch.isfinite(batch["logp"]).all() and torch.isfinite(batch["rewards"]).all():
                return batch, restored
            self.RNN.load_state_dict(good)
            restored += 1
        raise RuntimeError(f"the rollout was not finite {tries} times in a row; the run is stopped")

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
            ratio outside the clip range), and skipped, the number of minibatch steps whose loss was
            not finite and which were therefore not applied.

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
        n = 0
        skipped = 0
        for _ in range(self.n_epochs):
            for idx in torch.randperm(B, device=adv.device).chunk(self.n_minibatches):
                noise = {k: v[:, :, idx] for k, v in batch["noise"].items()}
                _, means, values = self.sequence(batch["u"][:, :, idx], noise)
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
                if not torch.isfinite(loss):
                    skipped += 1
                    continue
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.RNN.parameters(), self.max_grad_norm)
                self.optimizer.step()
                with torch.no_grad():
                    sums["policy_loss"] += policy_loss.item()
                    sums["value_loss"] += value_loss.item()
                    sums["entropy"] += entropy.item()
                    sums["approx_kl"] += ((batch["logp"][:, idx] - logp) * m).sum().item() / m.sum().item()
                    sums["clip_frac"] += (((ratio - 1).abs() > clip).float() * m).sum().item() / m.sum().item()
                n += 1
        out = {k: v / max(n, 1) for k, v in sums.items()}
        out["skipped"] = float(skipped)
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
            states = self.sequence(out["u"], out["noise"])[0][:, :T * k]          # (N, T * k, B)
        rates = self.RNN.get_firing_rates(states).cpu().numpy()
        alive = out["alive"][:-1].bool().repeat_interleave(k, dim=0).cpu().numpy()  # (T * k, B)
        p = participation(rates[:, alive])
        n_active_mod = int(active_mask(rates[:, alive].std(axis=1)).sum())
        return int(active_mask(p).sum()), n_active_mod, p, out

    def run_training(self):
        """Train for n_updates PPO updates, probing and checkpointing on the way.

        Every update: the learning rate for this update (the warm-up ramp), a finite rollout
        (safe_rollout, rolling back to the last finite weights if needed), advantages (GAE with a
        critic, Monte-Carlo without), the clipped update, and one progress record. The first update
        and every probe_every-th one are followed by a probe. Every checkpoint_every-th update, except
        the last (the caller saves the finished run), calls checkpoint_fn(updates_done, params) if one
        was set.

        Returns:
            (RNN, progress, probes, last_probe): the trained RNN; progress, a list with one dict per
            update: update, discounted_return (mean over the batch of sum_t gamma^t r_t, the quantity
            PPO maximises), success (share of the batch that reached the cheese), steps (mean steps
            to the cheese, T if never), restored (rollouts discarded and weights rolled back), lr
            (the rate used, after the warm-up ramp) and the statistics of ppo_update (including
            skipped minibatches); probes, a dict of lists over probes: update (the update index),
            active, active_mod, probe_success, and p_update and p, the stored participation vectors
            (N,) with their update index; last_probe, the last probe's rollout dict.
        """
        hp = self
        progress = []
        probes = {"update": [], "active": [], "active_mod": [], "probe_success": [], "p_update": [], "p": []}
        self.progress, self.probes, self.last_probe = progress, probes, None
        discount = hp.gamma ** torch.arange(hp.T, device=self.device, dtype=torch.float32)[:, None]
        good = {k: v.detach().clone() for k, v in self.RNN.state_dict().items()}   # last finite weights
        tic = time.perf_counter()
        for it in range(1, hp.n_updates + 1):
            lr_now = self.lr * min(1.0, it / hp.warmup_updates) if hp.warmup_updates else self.lr
            for group in self.optimizer.param_groups:
                group["lr"] = lr_now
            batch, restored = self.safe_rollout(good, hp.n_episodes, hp.T)
            if restored:
                print(f"update {it}: rollout not finite, weights restored ({restored} tries)", flush=True)
            if hp.critic:
                adv, ret = self.gae(batch["rewards"], batch["values"], batch["alive"], hp.gamma, hp.gae_lambda)
            else:
                adv, ret = self.mc_advantages(batch["rewards"], batch["alive"], hp.gamma)
            stats = self.ppo_update(batch, adv, ret)
            progress.append({"update": it,
                             "discounted_return": (batch["rewards"] * discount).sum(dim=0).mean().item(),
                             "success": batch["success"].float().mean().item(),
                             "steps": batch["steps"].mean().item(),
                             "restored": restored,
                             "lr": lr_now,
                             **stats})
            good = {k: v.detach().clone() for k, v in self.RNN.state_dict().items()}
            if it == 1 or it % hp.probe_every == 0:
                n_active, n_active_mod, p, out = self.probe()
                self.last_probe = out
                probes["update"].append(it)
                probes["active"].append(n_active)
                probes["active_mod"].append(n_active_mod)
                probes["probe_success"].append(out["success"].float().mean().item())
                if (len(probes["update"]) - 1) % hp.store_participation_every == 0:
                    probes["p_update"].append(it)
                    probes["p"].append(p)
                r = progress[-1]
                print(f"update {it:5d}/{hp.n_updates}  disc. return {r['discounted_return']:7.3f}"
                      f"  success {r['success']:.2f}  steps {r['steps']:6.1f}"
                      f"  probe success {probes['probe_success'][-1]:.2f}"
                      f"  active {n_active} (mod {n_active_mod})/{self.RNN.N}  kl {r['approx_kl']:.4f}  clip {r['clip_frac']:.2f}"
                      f"  {(time.perf_counter() - tic) / it:.2f} s/update", flush=True)
            if (self.checkpoint_fn is not None and self.checkpoint_every
                    and it % self.checkpoint_every == 0 and it < hp.n_updates):
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
    hp_args["critic"] = bool(hp.get("critic", rnn.n_outputs == 3))
    if "substeps" not in hp:
        # the constructor's default (5) would otherwise play these runs silently with the wrong clock
        raise ValueError(f"{folder}: no `substeps` in the saved config, so this is a run of the first "
                         f"array (16790695, 3 Nov 2026: one RNN step per decision, clipped actions, "
                         f"discounted shaping), trained in a task version the current TaskMaze does "
                         f"not reproduce; it cannot be rebuilt from current code")
    trainer = TrainerPPO(rnn, task, None, **hp_args)
    return cfg, task, trainer
