"""The entropy and correlation set-point terms of Trainer.Penalties (lambda_ent, lambda_cor).

1. Known answers: each term is recomputed in numpy from its definition (math.erf for the Gaussian
   on-fraction, np.corrcoef over every pair for the correlations), not with the torch calls it uses.
2. A unit that is below threshold on every step of the batch gets a nonzero gradient on its incoming
   weights from each term. The same unit gets exactly zero gradient from the task loss, and from the
   same term computed on rates relu(h) instead of h, which shows the test can fail.
3. Off by default and bit-identical: with lambda 0 a Trainer trains exactly as one whose Penalties
   table does not hold the two entries (the code before they existed); with lambda > 0 it does not.

Run:  python tests/test_setpoint_penalties.py   (or under pytest)
"""
import math

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Penalties, Trainer

K = 0                  # the silenced unit


def _rnn(seed=1, N=30):
    """A small standard ReLU network (h equation, no Dale, self-connections on).

    Args:
        seed: int; N: int, number of units.
    Returns:
        RNN_torch on the CPU.
    """
    return RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False, n_inputs=3,
                     n_outputs=2, equation_type="h", seed=seed, spectral_rad=1.2, self_connections=True)


def _inputs(T=60, B=16, seed=2):
    """Gaussian inputs, (3, T, B)."""
    return torch.randn(3, T, B, generator=torch.Generator().manual_seed(seed))


def test_entropy_known_answer():
    """mean_i ((H_b(Phi(mu_i/sd_i)) - t)/t)^2 recomputed with math.erf and plain logs."""
    x = torch.randn(7, 40, 5, generator=torch.Generator().manual_seed(0), dtype=torch.float64)
    x = x + torch.linspace(-3, 1, 7, dtype=torch.float64).reshape(-1, 1, 1)
    t = 0.75
    got = float(Penalties(_rnn()).entropy_setpoint_penalty(x, target_bits=t, eps=0.0))
    a = x.reshape(7, -1).numpy()
    ref = []
    for row in a:
        z = row.mean() / row.std(ddof=1)
        p = 0.5 * (1 + math.erf(z / math.sqrt(2)))
        H = -(p * math.log2(p) + (1 - p) * math.log2(1 - p))
        ref.append(((H - t) / t) ** 2)
    assert abs(got - np.mean(ref)) < 1e-10, (got, np.mean(ref))


def test_correlation_known_answer():
    """mean_i ((mean_{j != i} corr_ij - c)/s)^2 recomputed with np.corrcoef over every pair."""
    g = torch.Generator().manual_seed(1)
    common = torch.randn(1, 50, 4, generator=g, dtype=torch.float64)
    x = torch.randn(9, 50, 4, generator=g, dtype=torch.float64) + torch.linspace(0, 2, 9, dtype=torch.float64).reshape(-1, 1, 1) * common
    c0, s = 0.1, 0.4
    got = float(Penalties(_rnn()).correlation_setpoint_penalty(x, target_corr=c0, scale=s, eps=0.0))
    C = np.corrcoef(x.reshape(9, -1).numpy())
    ci = (C.sum(1) - 1) / 8
    assert abs(got - np.mean(((ci - c0) / s) ** 2)) < 1e-10


def _silent_row_grads(term):
    """Gradients on the incoming recurrent weights of a unit that is off on every step.

    Unit K gets non-positive recurrent weights and no input, so with the noise off its h never rises
    above its start at 0 and relu(h) is 0 on every step and trial.

    Args:
        term: "entropy" or "correlation".
    Returns:
        (task, pre, rate): each the (N,) gradient on W_rec[K, :] from the task loss, from the term on
        pre-activations h (the implementation), and from the same term on rates relu(h).
    """
    rnn = _rnn()
    with torch.no_grad():
        rnn.W_rec[K] = -rnn.W_rec[K].abs() - 0.05
        rnn.W_inp[K] = 0.0
    states, out = rnn(_inputs(), w_noise=False)
    assert (states[K] <= 0).all(), "unit K must be off on every step"
    assert (states[torch.arange(rnn.N) != K] > 0).any(), "the other units must fire"
    pen = Penalties(rnn)
    fn = pen.entropy_setpoint_penalty if term == "entropy" else pen.correlation_setpoint_penalty
    grads = []
    for loss in ((out ** 2).mean(), fn(states), fn(torch.relu(states))):
        g, = torch.autograd.grad(loss, [rnn.W_rec], retain_graph=True)
        grads.append(g[K])
    return grads


def test_entropy_gradient_reaches_a_silent_unit():
    """The entropy set point moves the incoming weights of a unit that never fires."""
    task, pre, rate = _silent_row_grads("entropy")
    assert torch.count_nonzero(task) == 0 and torch.count_nonzero(rate) == 0
    assert pre.abs().max() > 1e-6, pre.abs().max()


def test_correlation_gradient_reaches_a_silent_unit():
    """The correlation set point moves the incoming weights of a unit that never fires."""
    task, pre, rate = _silent_row_grads("correlation")
    assert torch.count_nonzero(task) == 0 and torch.count_nonzero(rate) == 0
    assert pre.abs().max() > 1e-6, pre.abs().max()


def _train(steps=5, drop_new_entries=False, **lam):
    """Train a small network for a few steps and return its parameters and losses.

    Args:
        steps: int; drop_new_entries: bool, delete the two set-point entries from the Penalties table,
        as the code was before they were added; lam: lambda_ent / lambda_cor overrides.
    Returns:
        (list of parameter tensors, list of float losses).
    """
    rnn = _rnn(seed=5, N=20)
    opt = torch.optim.Adam(rnn.parameters(), lr=1e-3, weight_decay=1e-6)
    tr = Trainer(rnn, None, opt, lambda_rws=0.0, lambda_orth=0.0, lambda_frm=0.0, monitor=False,
                 task_safe_gradients=False, **lam)
    if drop_new_entries:
        del tr.penalty_map["entropy_setpoint"], tr.penalty_map["correlation_setpoint"]
    u = _inputs(T=40, B=8, seed=6)
    y = torch.randn(2, 40, 8, generator=torch.Generator().manual_seed(7))
    losses = [tr.train_step(u, y, np.arange(40))[0] for _ in range(steps)]
    return [p.detach().clone() for p in rnn.parameters()], losses


def test_off_by_default_and_bit_identical():
    """lambda 0 is the default, and a lambda-0 run equals the run without the entries, bit for bit."""
    tr = Trainer(_rnn(), None, None, lambda_rws=0.0, lambda_orth=0.0, lambda_frm=0.0)
    assert tr.penalty_map["entropy_setpoint"][1] == 0 and tr.penalty_map["correlation_setpoint"][1] == 0
    p_old, l_old = _train(drop_new_entries=True)
    p_new, l_new = _train()
    assert l_old == l_new
    assert all(torch.equal(a, b) for a, b in zip(p_old, p_new))
    for lam in ({"lambda_ent": 0.1}, {"lambda_cor": 0.1}):   # the comparison can fail
        p_on, _ = _train(**lam)
        assert not all(torch.equal(a, b) for a, b in zip(p_old, p_on)), lam


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
