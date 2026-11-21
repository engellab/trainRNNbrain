"""The entropy and correlation set-point terms of Trainer.Penalties (lambda_ent, lambda_cor).

1. Known answers: each term is recomputed in numpy from its definition (math.erf for the Gaussian
   on-fraction, np.corrcoef over every pair for the correlations), not with the torch calls it uses.
2. A unit that is below threshold on every step of the batch gets a nonzero gradient on its incoming
   weights from each term. The same unit gets exactly zero gradient from the task loss, and from the
   same term computed on rates relu(h) instead of h, which shows the test can fail.
3. Off by default and bit-identical: with lambda 0 a Trainer trains exactly as one whose Penalties
   table does not hold the two entries (the code before they existed); with lambda > 0 it does not.
4. The maze mirrors (TrainerPPO, Stage 2): each PPO term equals the supervised term on the same kept
   steps; a unit that is off on every step of a maze recompute gets a nonzero gradient on its incoming
   weights from each term and exactly zero from the rate-based metabolic penalty; with lambda 0 neither
   term is computed in ppo_update and the update statistics carry no key for it.

Run:  python tests/test_setpoint_penalties.py   (or under pytest)
"""
import math

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.tasks.TaskMaze import TaskMaze
from trainRNNbrain.trainer.Trainer import Penalties, Trainer
from trainRNNbrain.trainer.TrainerPPO import TrainerPPO

K = 0                  # the silenced unit
# the small test maze of tests/test_rl_maze.py
MAZE = ["#######", "#.....#", "#.###.#", "#.#C#.#", "#.#.#.#", "#.....#", "#######"]
REWARD = {"goal": 5.0, "step": -0.01, "wall": -0.05, "shape": 0.2}
ENT_ARGS = {"target_bits": 0.75, "eps": 1e-6}               # configs/trainer/trainer_ppo.yaml
COR_ARGS = {"target_corr": 0.0, "scale": 0.4, "eps": 1e-6}


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


def _maze_trainer(seed=0, N=24, **hp):
    """A small standard ReLU agent (h equation, no Dale, no critic) in a TrainerPPO on the test maze.

    Args:
        seed: int; N: int, number of units; hp: further TrainerPPO arguments (lambda_ent, ...).
    Returns:
        TrainerPPO on the CPU, two RNN steps per decision, one epoch of one minibatch per update.
    """
    task = TaskMaze(MAZE, n_rays=8, ray_max=3.0, ray_step=0.1, v_max=0.2, turn_max=0.3, goal_radius=0.4,
                    goal_cue=False, action_feedback=False, reward=REWARD, device=torch.device("cpu"), seed=seed)
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, equation_type="h", dt=1, tau=10,
                    dale=False, self_connections=True, bias_range=[0.0, 0.0], sigma_rec=0.05, sigma_inp=0.05,
                    sigma_out=0.03, n_inputs=task.n_inputs, n_outputs=2, seed=seed, init_draws="independent")
    opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    return TrainerPPO(rnn, task, opt, action_std=0.5, substeps=2, critic=False, n_epochs=1, n_minibatches=1,
                      ent_args=ENT_ARGS, cor_args=COR_ARGS, **hp)


def test_ppo_terms_mirror_the_supervised_terms():
    """On states with one episode ended early, each PPO term with the live-step mask equals the
    supervised term on the kept steps alone; without a mask it equals the supervised term on every step."""
    tr = _maze_trainer(lambda_ent=0.1, lambda_cor=0.1)
    g = torch.Generator().manual_seed(3)
    T, B, N = 6, 4, tr.RNN.N
    states = torch.randn(N, T * 2, B, generator=g) + torch.linspace(-2, 1, N).reshape(-1, 1, 1)
    alive = torch.ones(T, B)
    alive[4:, 2] = 0
    mask = tr.live_steps(alive)
    kept = states.reshape(N, -1)[:, mask.reshape(-1)].reshape(N, -1, 1)
    pen = Penalties(tr.RNN)
    for ppo, sup, args in ((tr.entropy_setpoint_penalty, pen.entropy_setpoint_penalty, ENT_ARGS),
                           (tr.correlation_setpoint_penalty, pen.correlation_setpoint_penalty, COR_ARGS)):
        assert torch.equal(ppo(states, mask), sup(kept, **args))
        assert torch.equal(ppo(states), sup(states, **args))
        assert not torch.equal(ppo(states, mask), ppo(states))       # the mask matters here


def _maze_silent_row_grads(term):
    """Gradients on the incoming recurrent weights of a unit that is off on every step of a maze recompute.

    Unit K gets non-positive recurrent weights and no input, and the rollout's stored noise is set to 0,
    so its h never rises above its start at 0.

    Args:
        term: "entropy" or "correlation".
    Returns:
        (rate, pre): the (N,) gradients on W_rec[K, :] from the metabolic penalty (on rates) and from
        the set-point term (on pre-activations).
    """
    tr = _maze_trainer(lambda_ent=0.1, lambda_cor=0.1, lambda_met=0.1)
    with torch.no_grad():
        tr.RNN.W_rec[K] = -tr.RNN.W_rec[K].abs() - 0.05
        tr.RNN.W_inp[K] = 0.0
    torch.manual_seed(4)
    out = tr.rollout(6, 12)
    noise = {k: torch.zeros_like(v) for k, v in out["noise"].items()}
    states = tr.sequence(out["u"], noise, alive=out["alive"][:-1])[0]
    assert (states[K] <= 0).all(), "unit K must be off on every step"
    assert (states[torch.arange(tr.RNN.N) != K] > 0).any(), "the other units must fire"
    fn = tr.entropy_setpoint_penalty if term == "entropy" else tr.correlation_setpoint_penalty
    grads = []
    for loss in (tr.metabolic_penalty(states), fn(states)):
        g, = torch.autograd.grad(loss, [tr.RNN.W_rec], retain_graph=True)
        grads.append(g[K])
    return grads


def test_ppo_entropy_gradient_reaches_a_silent_unit():
    """The maze entropy set point moves the incoming weights of a unit that never fires."""
    rate, pre = _maze_silent_row_grads("entropy")
    assert torch.count_nonzero(rate) == 0
    assert pre.abs().max() > 1e-6, pre.abs().max()


def test_ppo_correlation_gradient_reaches_a_silent_unit():
    """The maze correlation set point moves the incoming weights of a unit that never fires."""
    rate, pre = _maze_silent_row_grads("correlation")
    assert torch.count_nonzero(rate) == 0
    assert pre.abs().max() > 1e-6, pre.abs().max()


def _refuse(*args, **kwargs):
    """Stands in for a penalty that must not be computed."""
    raise AssertionError("a set-point term was computed with its lambda at 0")


def test_ppo_off_by_default():
    """Both lambdas default to 0, and at 0 ppo_update never calls the terms and reports no key for them
    (the update is the one made before the terms existed; the end-to-end comparison with main is in the
    Stage 2 rules entry). With a lambda above 0 the term enters: its key appears and the weights differ."""
    tr0 = _maze_trainer()
    assert tr0.lambda_ent == 0 and tr0.lambda_cor == 0
    torch.manual_seed(5)
    out = tr0.rollout(6, 12)
    adv, ret = tr0.mc_advantages(out["rewards"], out["alive"], 0.99)
    results = {}
    for name, hp in (("off", {}), ("ent", {"lambda_ent": 0.1}), ("cor", {"lambda_cor": 0.1})):
        tr = _maze_trainer(**hp)                     # same seed: the same initial weights as tr0
        if name == "off":
            tr.entropy_setpoint_penalty = tr.correlation_setpoint_penalty = _refuse
        torch.manual_seed(6)                         # the same minibatch permutation
        stats = tr.ppo_update(out, adv, ret)
        results[name] = (stats, [p.detach().clone() for p in tr.RNN.parameters()])
    off_stats, off_params = results["off"]
    assert not ({"entropy_setpoint", "correlation_setpoint"} & set(off_stats))
    for name, key in (("ent", "entropy_setpoint"), ("cor", "correlation_setpoint")):
        stats, params = results[name]
        assert key in stats and np.isfinite(stats[key]) and stats[key] > 0
        assert not all(torch.equal(a, b) for a, b in zip(off_params, params)), name


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
