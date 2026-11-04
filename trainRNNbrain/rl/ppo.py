"""Rollout, generalised advantage estimation and the clipped PPO update for the maze policy.

In words. A rollout plays a batch of episodes with the current policy and records what it saw and
did. The critic's VALUE of a state is its estimate of the discounted reward still to come from
there. The ADVANTAGE of an action is how much better the reward that followed it was than the
value predicted, so a positive advantage says "do that more"; generalised advantage estimation
(GAE, Schulman et al. 2016) blends one-step and many-step versions of that comparison with the
weight lambda to trade bias against variance. An episode cut off at T steps is not over, so its
last value BOOTSTRAPS the missing tail: it stands in for the reward after the cut. The PPO update
(Schulman et al. 2017) raises the probability of actions with positive advantage and lowers the
others, but through a CLIPPED SURROGATE: the probability ratio new/old is clipped to
[1 - clip, 1 + clip], so one update cannot move the policy far from the one that collected the data.
The critic is fitted to the observed returns at the same time (value loss).

Shapes follow the RNN's convention, batch last: observations u (n_inputs, T + 1, B), actions
(2, T, B), everything per step (T, B). A step here is one decision; the RNN runs policy.substeps
of its own steps per decision. At step t the RNN consumes observation u_t and emits action a_t;
the environment returns reward r_t, observation u_{t+1} and whether the episode ended. The RNN is
stepped once more on the final observation u_T, which gives the bootstrap value (`values` has
T + 1 rows). `alive[t]` is 1 while the episode had not ended before step t (so alive[0] = 1 and
alive has T + 1 rows), and every loss is a mean over alive steps only.
"""
import torch


def rollout(env, policy, n_episodes, T, clean=False, generator=None):
    """Play n_episodes episodes of at most T steps in parallel, without gradients.

    Args:
        env: MazeEnv; policy: MazePolicy; n_episodes: int, the batch B; T: int, steps per episode.
        clean: bool, True for a read-out pass: the RNN's noise off and the policy mean as the
            action; False (training) draws the noise and samples the actions.
        generator: torch.Generator on the env's device for the drops, the RNN noise and the action
            noise, or None for the env's and the RNN's own generators and torch's global one.
    Returns:
        dict: u (n_inputs, T+1, B); noise, the dict from policy.draw_noise for (T+1) * substeps
        RNN steps;
        actions (2, T, B); logp (T, B); values (T+1, B); rewards (T, B); alive (T+1, B) float;
        success (B,) bool, reached the cheese; steps (B,) float, steps taken until the cheese
        (T if never); xy (T+1, B, 2) positions and heading (T+1, B) radians, for figures and
        animations. The RNN states are not kept (they are large);
        policy.sequence(u, noise) recomputes them exactly.
    """
    k = policy.substeps
    with torch.no_grad():
        u = [env.reset(n_episodes, generator)]
        noise = policy.draw_noise((T + 1) * k, n_episodes, not clean, generator)
        x = policy.initial_state(n_episodes)
        alive = [torch.ones(n_episodes, device=env.device)]
        xy, heading = [env.xy.clone()], [env.heading.clone()]
        actions, logp, values, rewards = [], [], [], []
        for t in range(T):
            for s in range(t * k, (t + 1) * k):
                x = policy.step(x, u[-1], noise["rec"][:, s], noise["inp"][:, s])
            mean, value = policy.heads(x, noise["out"][:, 0])
            if clean:
                action = mean
            else:
                action = mean + policy.std()[:, None] * torch.randn(
                    mean.shape, generator=generator, device=mean.device)
            obs, reward, done = env.step(action)
            actions.append(action)
            logp.append(policy.log_prob(action, mean))
            values.append(value)
            rewards.append(reward)
            u.append(obs)
            alive.append((~done).float())
            xy.append(env.xy.clone())
            heading.append(env.heading.clone())
        for s in range(T * k, (T + 1) * k):
            x = policy.step(x, u[-1], noise["rec"][:, s], noise["inp"][:, s])
        values.append(policy.heads(x, noise["out"][:, 0])[1])
    alive = torch.stack(alive)
    return {"u": torch.stack(u, dim=1), "noise": noise, "actions": torch.stack(actions, dim=1),
            "logp": torch.stack(logp), "values": torch.stack(values), "rewards": torch.stack(rewards),
            "alive": alive, "success": alive[-1] == 0, "steps": alive[:-1].sum(dim=0),
            "xy": torch.stack(xy), "heading": torch.stack(heading)}


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


def safe_rollout(env, policy, good, n_episodes, T, tries=3):
    """A rollout that is finite, or the last finite weights restored and the rollout replayed.

    The maze counterpart of the supervised trainer's rollback: at N = 2,000 the recurrent dynamics
    blew up (states to infinity over the 2,500-step episode) after a few updates at the paper's
    learning rate (arrays 16856487 and 16859389, 4 Nov 2026), and every later loss was skipped.
    Here a rollout whose log-probabilities or rewards are not finite is discarded, the weights are
    restored from `good` (the state_dict after the last update whose rollout was finite), and the
    rollout is played again with fresh noise.

    Args:
        env: MazeEnv; policy: MazePolicy; good: dict, policy.state_dict() tensors (copied);
        n_episodes, T: ints; tries: int, attempts before giving up.
    Returns:
        (batch, restored): the finite rollout dict and the number of restores (0 normally).
    Raises:
        RuntimeError if no finite rollout was obtained in `tries` attempts.
    """
    restored = 0
    for _ in range(tries):
        batch = rollout(env, policy, n_episodes, T)
        if torch.isfinite(batch["logp"]).all() and torch.isfinite(batch["rewards"]).all():
            return batch, restored
        policy.load_state_dict(good)
        restored += 1
    raise RuntimeError(f"the rollout was not finite {tries} times in a row; the run is stopped")


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


def ppo_update(policy, optimizer, batch, adv, ret, clip, n_epochs, n_minibatches,
               value_coef, entropy_coef, max_grad_norm):
    """The clipped PPO update on one rollout: n_epochs passes over the episodes in n_minibatches
    minibatches, each a full-sequence recompute (backprop through time) with the stored noise.

    Args:
        policy: MazePolicy; optimizer: torch optimizer over policy.parameters();
        batch: the dict from `rollout`; adv, ret: (T, B) from `gae`;
        clip: float, the ratio clip (0.2 in the paper); n_epochs, n_minibatches: ints;
        value_coef, entropy_coef: floats, loss weights; max_grad_norm: float, gradient clip.
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
    T, B = adv.shape
    alive = batch["alive"][:T]
    adv = (adv - adv[alive > 0].mean()) / (adv[alive > 0].std() + 1e-8)
    sums = {"policy_loss": 0.0, "value_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}
    n = 0
    skipped = 0
    for _ in range(n_epochs):
        for idx in torch.randperm(B, device=adv.device).chunk(n_minibatches):
            noise = {k: v[:, :, idx] for k, v in batch["noise"].items()}
            _, means, values = policy.sequence(batch["u"][:, :, idx], noise)
            logp = policy.log_prob(batch["actions"][:, :, idx], means[:, :T])
            m = alive[:, idx]
            ratio = (logp - batch["logp"][:, idx]).clamp(-20.0, 20.0).exp()
            a = adv[:, idx]
            surrogate = torch.min(ratio * a, ratio.clamp(1 - clip, 1 + clip) * a)
            policy_loss = -(surrogate * m).sum() / m.sum()
            value_loss = (0.5 * ((values[:T] - ret[:, idx]) ** 2 * m).sum() / m.sum()
                          if policy.critic else torch.zeros((), device=adv.device))
            entropy = policy.entropy()
            loss = policy_loss + value_coef * value_loss - entropy_coef * entropy
            if not torch.isfinite(loss):
                skipped += 1
                continue
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(policy.parameters(), max_grad_norm)
            optimizer.step()
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
