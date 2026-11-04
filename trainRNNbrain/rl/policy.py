"""The project's standard RNN as the whole agent, run one step at a time.

THE RNN IS THE AGENT AND NOTHING ELSE IS TRAINED (Pavel, 4 Nov 2026). The RNN is `RNN_torch`
unchanged: its own W_inp, W_rec, W_out, activation, alpha and training noise, and its parameters
are the only parameters of this module. Its read-out rows are the action mean, turn and speed
(rows 0 and 1), and, when the RNN was built with 3 outputs, the critic's value (row 2), so the
value is one more read-out of the same network, trained by the value loss through the same W_out
and recurrent weights; with 2 outputs there is no critic (ppo.mc_advantages). The action is a
sample from a Gaussian around the mean with a FIXED standard deviation `action_std` (a number in
the config, not a parameter: a learned log-std collapsed to -3.4 in the first runs).

One environment step (a decision) is `substeps` RNN steps, x <- x + alpha * rhs(x, u, noise),
through the RNN's own `rhs`, with the observation held; the action is read from the state after
the last of them (with alpha = 0.1 the network needs several steps to respond to what it sees, see
configs/rl/maze_ppo.yaml).

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
"""
import math
import torch

LOG_2PI = math.log(2 * math.pi)


class MazePolicy(torch.nn.Module):
    """The RNN as actor (read-out rows 0, 1) and, with 3 outputs, critic (row 2); no other parameters."""

    def __init__(self, rnn, action_std, substeps=1):
        """Args:
            rnn: RNN_torch with n_outputs 3 (turn, speed, value: PPO with a critic) or 2 (turn,
                speed: no critic), and sigma_w == 0.
            action_std: float > 0, the fixed standard deviation of both action outputs, in raw
                output units (the env squashes them).
            substeps: int >= 1, RNN steps per decision (the observation is held over them).
        Raises:
            ValueError if rnn.n_outputs is not 2 or 3, or if rnn.sigma_w is not 0 (`step` runs the
            RNN's rhs with fixed weights and would silently drop the synaptic noise that
            RNN_torch.forward applies).
        """
        super().__init__()
        if rnn.n_outputs not in (2, 3):
            raise ValueError(f"the RNN needs 2 outputs (no critic) or 3 (value row); it has {rnn.n_outputs}")
        if rnn.sigma_w != 0.0:
            raise ValueError(f"MazePolicy does not implement synaptic noise; model.sigma_w is {rnn.sigma_w}")
        self.rnn = rnn
        self.critic = rnn.n_outputs == 3
        self.action_std = float(action_std)
        self.substeps = int(substeps)

    def draw_noise(self, T, B, w_noise, generator=None):
        """The RNN's noise for T steps of B episodes, as RNN_torch.forward draws it.

        Args:
            T, B: int; w_noise: bool, False gives zeros (a noise-free pass);
            generator: torch.Generator on the RNN's device, or None for the RNN's own.
        Returns:
            dict with "rec" (N, T, B) and "inp" (n_inputs, T, B), fresh per step, and "out"
            (N, 1, B), one draw per episode held over its T steps; all scaled by sqrt(2/alpha)*sigma.
        """
        rnn = self.rnn
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
        return torch.full((2,), self.action_std, device=self.rnn.device)

    def initial_state(self, B):
        """The RNN state every episode starts from: y_init repeated, (N, B)."""
        return self.rnn.y_init.to(self.rnn.device).reshape(-1, 1).repeat(1, B)

    def step(self, x, u, rec_t, inp_t):
        """One RNN step through the RNN's own right-hand side.

        Args:
            x: (N, B) state; u: (n_inputs, B) observation; rec_t: (N, B) and inp_t: (n_inputs, B),
            this step's recurrent and input noise.
        Returns:
            (N, B) next state.
        """
        W_rec, W_inp, _ = self.rnn._constrained_weights()
        return x + self.rnn.alpha * self.rnn.rhs(x=x, I=u, i_noise=inp_t, r_noise=rec_t,
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
        r = self.rnn.get_firing_rates(x)
        W_out = self.rnn._constrained_weights()[2]
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
        return torch.full((), 2 * (math.log(self.action_std) + 0.5 * (LOG_2PI + 1.0)), device=self.rnn.device)
