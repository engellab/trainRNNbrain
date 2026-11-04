"""The project's standard RNN as a stochastic policy with a value head, run one step at a time.

The RNN is `RNN_torch` unchanged: its own W_inp, W_rec, W_out, activation, alpha and training
noise. One environment step (a decision) is `substeps` RNN steps, x <- x + alpha * rhs(x, u, noise),
through the RNN's own `rhs`, with the observation held; the action is read from the state after
the last of them (with alpha = 0.1 the network needs several steps to respond to what it sees, see
configs/rl/maze_ppo.yaml). The policy mean is the RNN's ordinary read-out, W_out @ (r + output noise), with two
outputs (turn, speed); the action is a sample from a Gaussian around it with one learned log-std
per output, floored at `log_std_min` so exploration cannot vanish (it fell from -0.7 to -3.4 in the
first runs, which had converged to standing still); the value is a separate linear read-out of the
rates, so the RNN's W_out is trained by the policy gradient only, as a task read-out should be.

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
    """Actor (the RNN and a log-std) and critic (a linear read-out of the rates) in one module."""

    def __init__(self, rnn, log_std_init, log_std_min, substeps=1):
        """Args:
            rnn: RNN_torch built with n_outputs == 2 (turn, speed) and sigma_w == 0.
            log_std_init: float, the initial log standard deviation of both action outputs.
            log_std_min: float, the floor of the log standard deviation (`log_std` is clamped to
                it wherever it is used, so the parameter can fall below it but the policy cannot).
            substeps: int >= 1, RNN steps per decision (the observation is held over them).
        Raises:
            ValueError if rnn.sigma_w is not 0: `step` runs the RNN's rhs with fixed weights and
            would silently drop the synaptic noise that RNN_torch.forward applies.
        """
        super().__init__()
        if rnn.sigma_w != 0.0:
            raise ValueError(f"MazePolicy does not implement synaptic noise; model.sigma_w is {rnn.sigma_w}")
        self.rnn = rnn
        self.log_std = torch.nn.Parameter(torch.full((2,), float(log_std_init), device=rnn.device))
        self.log_std_min = float(log_std_min)
        self.substeps = int(substeps)
        self.value_w = torch.nn.Parameter(torch.zeros(rnn.N, device=rnn.device))
        self.value_b = torch.nn.Parameter(torch.zeros((), device=rnn.device))

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
        """The policy's standard deviation per action output, (2,), with the floor applied."""
        return self.log_std.clamp(min=self.log_std_min).exp()

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
        """Policy mean and value from RNN states.

        Args:
            x: (N, B) or (N, T, B) states; out_noise: (N, B) or (N, 1, B), the episode's output
            noise (broadcast over time).
        Returns:
            (mean, value): mean (2, B) or (2, T, B), the RNN's read-out of the noisy rates;
            value (B,) or (T, B), the linear read-out of the noise-free rates.
        """
        r = self.rnn.get_firing_rates(x)
        W_out = self.rnn._constrained_weights()[2]
        mean = torch.einsum("oj,j...->o...", W_out, r + out_noise)
        value = torch.einsum("j,j...->...", self.value_w, r) + self.value_b
        return mean, value

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
        """Log density of actions under the diagonal Gaussian N(mean, exp(log_std)^2).

        Args:
            action, mean: (2, ...) tensors of the same shape.
        Returns:
            (...) tensor, the log density summed over the two action outputs.
        """
        log_std = self.std().log().reshape((2,) + (1,) * (mean.ndim - 1))
        z = (action - mean) / log_std.exp()
        return (-0.5 * z ** 2 - log_std - 0.5 * LOG_2PI).sum(dim=0)

    def entropy(self):
        """Entropy of the Gaussian policy, a function of the (floored) log_std alone: a scalar."""
        return (self.std().log() + 0.5 * (LOG_2PI + 1.0)).sum()
