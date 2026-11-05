"""Checks for the maze task and the PPO trainer (trainRNNbrain/tasks/TaskMaze.py,
trainRNNbrain/trainer/TrainerPPO.py): the maze tables, the sensors, a blocked move and the goal,
GAE on a known case, that the PPO recompute reproduces the rollout exactly (probability ratio 1 at
the old parameters), which is what the stored-noise replay is for, the update's guards and the
rollback, that the training script writes the standard run layout that common.maze_load_runs
reads back, and the paper's interventions as trainer options (issue #5): the two penalties against
the supervised formulas re-typed here, the replay of synaptic noise and of the read-out dropout
mask in the recompute (ratio 1, dropped columns zero in both passes), the penalty values in the
update statistics, and the refusal of dropout kind "dead". The 13 checks before the options were
added run with every option off, as the default is.
Run: python tests/test_rl_maze.py   (CPU, about a minute)
"""
import glob
import math
import os
import subprocess
import sys
import tempfile
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.tasks.TaskMaze import TaskMaze, parse_maze, bfs_distance
from trainRNNbrain.trainer.TrainerPPO import TrainerPPO, load_run
from trainRNNbrain.experiments_and_analysis.common import maze_load_runs, participation

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MAZE = ["#######",
        "#.....#",
        "#.###.#",
        "#.#C#.#",
        "#.#.#.#",
        "#.....#",
        "#######"]
REWARD = {"goal": 5.0, "step": -0.01, "wall": -0.05, "shape": 0.2}
CPU = torch.device("cpu")
FULL = torch.tensor([[0.0], [20.0]])      # raw action: no turn, sigmoid(20) = 1 -> full speed


def make_task(goal_cue=False, action_feedback=False, seed=0):
    """The small test maze on CPU: ray range 3, speed 0.2, turn 0.3, goal radius 0.4."""
    return TaskMaze(MAZE, n_rays=8, ray_max=3.0, ray_step=0.1, v_max=0.2, turn_max=0.3, goal_radius=0.4,
                    goal_cue=goal_cue, action_feedback=action_feedback, reward=REWARD, device=CPU, seed=seed)


def make_trainer(task, N=32, seed=0, substeps=1, critic=True, sigma_w=0.0, **hp):
    """A small standard-style RNN (ReLU, equation h, no Dale, fixed zero bias) as the agent, with
    the value as its third read-out row (critic=True) or without one, and synaptic noise sigma_w
    (0: none), inside a TrainerPPO with an Adam optimizer at 1e-3; `hp` sets the trainer's other
    hyperparameters (every intervention off unless given)."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, equation_type="h", dt=1, tau=10,
                    dale=False, io_nonnegativity=False, self_connections=True, gamma=0,
                    bias_range=[0.0, 0.0], sigma_rec=0.05, sigma_inp=0.05, sigma_out=0.03, sigma_w=sigma_w,
                    n_inputs=task.n_inputs, n_outputs=3 if critic else 2, seed=seed, init_draws="independent")
    opt = torch.optim.Adam(rnn.parameters(), lr=1e-3)
    return TrainerPPO(rnn, task, opt, action_std=0.5, substeps=substeps, critic=critic, **hp)


DROPOUT_ARGS = {"dropout_kind": "mute", "sampling_method": "participation", "eta": 0.5, "active_rel": 0.05,
                "drop_rate": 0.5, "rescale": True, "p_max": 0.9, "dropout_beta": 4.0, "activity_q": 0.9}


def test_only_the_rnn_is_trained():
    """The optimizer's parameters are exactly the RNN's W_out, W_rec and W_inp."""
    task = make_task()
    for critic in (True, False):
        trainer = make_trainer(task, critic=critic)
        assert sorted(n for n, _ in trainer.RNN.named_parameters()) == ["W_inp", "W_out", "W_rec"]
        held = {id(p) for g in trainer.optimizer.param_groups for p in g["params"]}
        assert held == {id(p) for p in trainer.RNN.parameters()}
        assert trainer.RNN.W_out.shape[0] == (3 if critic else 2)


def test_mc_advantages_known_case():
    """Two episodes, rewards 1 and 0 throughout, no termination: returns are the geometric sums
    and the advantages are plus and minus half of them."""
    T = 10
    rewards = torch.stack([torch.ones(T), torch.zeros(T)], dim=1)
    alive = torch.ones(T + 1, 2)
    adv, ret = TrainerPPO.mc_advantages(rewards, alive, gamma=0.5)
    g0 = sum(0.5 ** k for k in range(T))
    assert abs(ret[0, 0].item() - g0) < 1e-6 and ret[0, 1].item() == 0
    assert abs(adv[0, 0].item() - g0 / 2) < 1e-6 and abs(adv[0, 1].item() + g0 / 2) < 1e-6
    alive[3:, 0] = 0                                     # the first episode ended after step 2
    adv, ret = TrainerPPO.mc_advantages(rewards, alive, gamma=0.5)
    assert abs(ret[0, 0].item() - (1 + 0.5 + 0.25)) < 1e-6 and ret[3:, 0].abs().sum() == 0


def test_maze_tables():
    """Every free cell reaches the cheese; the cheese is at 0; walls are inf; known distances hold."""
    wall, goal = parse_maze(MAZE)
    dist = bfs_distance(wall, goal)
    assert goal == (3, 3) and dist[goal] == 0
    assert np.isfinite(dist[~wall]).all() and np.isinf(dist[wall]).all()
    assert dist[4, 3] == 1 and dist[5, 1] == 4 and dist[1, 1] == 8 and dist[1, 3] == 10


def test_sensors():
    """An agent in the top corridor facing +x reads the walls at the measured distances."""
    task = make_task()
    task.reset(1)
    task.xy[:] = torch.tensor([[3.5, 1.5]])
    task.heading[:] = 0.0
    obs = task.observe()[:, 0]
    expect = {0: 1 - 2.5 / 3, 2: 1 - 0.5 / 3, 4: 1 - 2.5 / 3, 6: 1 - 0.5 / 3}   # +x, +y (down), -x, -y
    for ray, val in expect.items():
        assert abs(obs[ray].item() - val) < 0.04, (ray, obs[ray].item(), val)
    assert task.n_inputs == 8 and make_task(goal_cue=True).n_inputs == 10
    task = make_task(action_feedback=True)
    obs = task.reset(1)
    assert task.n_inputs == 10 and obs[8:, 0].abs().sum() == 0          # no previous action yet
    obs, _, _ = task.step(torch.tensor([[0.0], [20.0]]))               # straight, full speed
    assert abs(obs[8, 0].item()) < 1e-6 and abs(obs[9, 0].item() - 1.0) < 1e-6


def test_step_wall_and_goal():
    """A move into a wall is cancelled and costs the wall penalty; standing still earns only the
    step cost; a cell of progress pays the shaping; the cheese ends the episode."""
    task = make_task()
    task.reset(1)
    task.xy[:] = torch.tensor([[3.5, 1.95]])
    task.heading[:] = math.pi / 2                          # +y, into the wall of row 2
    _, r, done = task.step(FULL)
    assert torch.allclose(task.xy, torch.tensor([[3.5, 1.95]]))
    assert abs(r.item() - (-0.01 - 0.05)) < 1e-6 and not done.item()
    _, r, _ = task.step(torch.tensor([[0.0], [-20.0]]))    # sigmoid(-20) = 0: standing still
    assert abs(r.item() - (-0.01)) < 1e-6 and torch.allclose(task.xy, torch.tensor([[3.5, 1.95]]))
    task.xy[:] = torch.tensor([[3.5, 4.2]])
    task.heading[:] = -math.pi / 2                         # -y, towards the cheese at (3.5, 3.5)
    _, r, done = task.step(FULL)                           # y = 4.0: cell (4,3), one cell closer
    assert not done.item() and abs(r.item() - (-0.01 + 0.2)) < 1e-6
    _, r, done = task.step(FULL)                           # y = 3.8: 0.3 away, reached
    assert done.item() and r.item() > 4.5
    _, r, done = task.step(FULL)                           # finished agents do not move or earn
    assert done.item() and r.item() == 0 and abs(task.xy[0, 1].item() - 3.8) < 1e-6


def test_corner_is_not_cut_but_slides():
    """A diagonal step from cell (4,3) into (5,2) passes the corner of the wall cell (4,2): the x
    part is blocked by that wall, the y part is free, so the agent slides down into (5,3); it
    bumped, so it pays the wall cost."""
    task = make_task()
    task.reset(1)
    task.xy[:] = torch.tensor([[3.05, 4.95]])
    task.heading[:] = 3 * math.pi / 4                      # (-x, +y): end point (2.91, 5.09)
    _, r, done = task.step(FULL)
    assert torch.allclose(task.xy, torch.tensor([[3.05, 4.95 + 0.2 * math.sin(3 * math.pi / 4)]]), atol=1e-5)
    assert not done.item()
    assert abs(r.item() - (-0.01 - 0.05 - 0.2)) < 1e-5    # step, bump, one cell further from the cheese


def test_slide_along_wall():
    """Heading 45 degrees into the top wall from the top corridor: the y part is blocked, the x part
    is free, so the agent moves along the corridor at the x component of its speed, and pays the
    bump."""
    task = make_task()
    task.reset(1)
    task.xy[:] = torch.tensor([[2.5, 1.05]])
    task.heading[:] = -math.pi / 4                         # (+x, -y), -y is into the border wall
    _, r, _ = task.step(FULL)
    assert abs(task.xy[0, 0].item() - (2.5 + 0.2 * math.cos(math.pi / 4))) < 1e-5
    assert abs(task.xy[0, 1].item() - 1.05) < 1e-6 and abs(r.item() - (-0.01 - 0.05)) < 1e-5


def test_gae_known_case():
    """Constant reward 1, no termination, zero values: the advantage is the geometric sum."""
    T = 10
    rewards, values, alive = torch.ones(T, 1), torch.zeros(T + 1, 1), torch.ones(T + 1, 1)
    adv, ret = TrainerPPO.gae(rewards, values, alive, gamma=0.5, lam=1.0)
    assert abs(adv[0, 0].item() - sum(0.5 ** k for k in range(T))) < 1e-6
    assert torch.allclose(ret, adv)
    alive[3:] = 0                                         # the episode ended after step 2
    adv, _ = TrainerPPO.gae(rewards, values, alive, gamma=0.5, lam=1.0)
    assert abs(adv[0, 0].item() - (1 + 0.5 + 0.25)) < 1e-6 and adv[3:].abs().sum() == 0


def test_recompute_matches_rollout():
    """With the stored noise, the sequence recompute gives the rollout's means, values and
    log-probabilities exactly, so the PPO ratio starts at 1; with one and with three RNN steps
    per decision."""
    for substeps in (1, 3):
        torch.manual_seed(0)
        task = make_task(goal_cue=True, action_feedback=True)
        trainer = make_trainer(task, substeps=substeps)
        T, B = 25, 6
        out = trainer.rollout(B, T)
        states, means, values = trainer.sequence(out["u"], out["noise"])
        logp = trainer.log_prob(out["actions"], means[:, :T])
        assert states.shape == (32, (T + 1) * substeps, B)
        assert out["noise"]["rec"].shape == (32, (T + 1) * substeps, B)
        assert torch.allclose(logp, out["logp"], atol=1e-5)
        assert torch.allclose(values, out["values"], atol=1e-5)
        assert out["noise"]["out"].shape == (32, 1, B)  # one output-noise draw per episode, as forward
        assert torch.allclose((logp - out["logp"]).exp(), torch.ones_like(logp), atol=1e-4)


def test_ppo_update_runs():
    """One update changes the recurrent weights and returns finite statistics, with and without
    a critic."""
    for critic in (True, False):
        torch.manual_seed(1)
        task = make_task()
        trainer = make_trainer(task, critic=critic, clip=0.2, n_epochs=2, n_minibatches=2,
                               value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5)
        out = trainer.rollout(8, 30)
        if critic:
            adv, ret = trainer.gae(out["rewards"], out["values"], out["alive"], 0.99, 0.95)
        else:
            adv, ret = trainer.mc_advantages(out["rewards"], out["alive"], 0.99)
        before = trainer.RNN.W_rec.detach().clone()
        stats = trainer.ppo_update(out, adv, ret)
        assert all(np.isfinite(v) for v in stats.values()), stats
        assert not torch.allclose(before, trainer.RNN.W_rec)
        assert (stats["value_loss"] == 0) == (not critic)
        assert stats["skipped"] == 0


def test_ppo_update_guards():
    """A stored log-probability far below the recomputed one gives a finite, clamped ratio and
    a finite update; a non-finite advantage makes the loss non-finite, and that minibatch is
    skipped with the weights untouched."""
    torch.manual_seed(2)
    task = make_task()
    trainer = make_trainer(task, critic=False, clip=0.2, n_epochs=1, n_minibatches=1,
                           value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5)
    out = trainer.rollout(8, 20)
    adv, ret = trainer.mc_advantages(out["rewards"], out["alive"], 0.99)
    out["logp"] = out["logp"] - 1000.0                     # ratio would be e^1000 without the clamp
    stats = trainer.ppo_update(out, adv, ret)
    assert np.isfinite(stats["policy_loss"]) and stats["skipped"] == 0
    assert all(torch.isfinite(p).all() for p in trainer.RNN.parameters())
    before = trainer.RNN.W_rec.detach().clone()
    adv[0, 0] = float("nan")
    stats = trainer.ppo_update(out, adv, ret)
    assert stats["skipped"] == 1 and torch.equal(before, trainer.RNN.W_rec)


def test_safe_rollout_restores():
    """With NaN written into W_rec the rollout is not finite; safe_rollout restores the saved
    weights and returns a finite rollout, counting one restore."""
    torch.manual_seed(3)
    task = make_task()
    trainer = make_trainer(task, critic=False)
    good = {k: v.detach().clone() for k, v in trainer.RNN.state_dict().items()}
    with torch.no_grad():
        trainer.RNN.W_rec[0, 0] = float("nan")
    batch, restored = trainer.safe_rollout(good, 4, 10)
    assert restored == 1 and torch.isfinite(batch["logp"]).all()
    assert torch.equal(trainer.RNN.W_rec, good["W_rec"])
    batch, restored = trainer.safe_rollout(good, 4, 10)
    assert restored == 0


def test_run_script_writes_standard_layout():
    """The training script, dispatched to the maze by trainer=trainer_ppo, writes
    <save_to>/Maze_<tag>/s<seed>/ with s<seed>_config.yaml (seed and channel counts recorded),
    s<seed>_Params_Maze.npz (no *LastParams* file, so no cell inventory counts the run),
    s<seed>_Progress.json, s<seed>_Participation.npz and the two PNGs, at N = 40 for 2 updates;
    common.maze_load_runs reads it back and TrainerPPO.load_run rebuilds the agent with the
    saved weights."""
    scratch = os.path.join(os.environ.get("CLAUDE_SCRATCHPAD",
                                          os.path.join(tempfile.gettempdir(), "trainRNNbrain_scratch")),
                           "maze-refactor", "layout_check")
    save_to = os.path.join(scratch, "trained_RNNs")
    cmd = [sys.executable, os.path.join(REPO, "trainRNNbrain", "training", "run_experiment.py"),
           "task=Maze", "trainer=trainer_ppo", "model=rnn_relu_standard", "model.N=40", "seed=3",
           "trainer.n_updates=2", "trainer.n_episodes=4", "trainer.T=6", "trainer.probe_episodes=4",
           "trainer.probe_every=2", "trainer.checkpoint_every=1", "trainer.warmup_updates=1",
           f"paths.save_to={save_to}", "experiment_tag=check", f"hydra.run.dir={os.path.join(scratch, 'hydra')}"]
    env = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    folder = os.path.join(save_to, "Maze_check", "s3")
    names = sorted(os.listdir(folder))
    assert names == ["progress.png", "s3_Params_Maze.npz", "s3_Participation.npz", "s3_Progress.json",
                     "s3_config.yaml", "trajectories.png"], names
    assert not glob.glob(os.path.join(save_to, "**", "*LastParams*"), recursive=True)
    runs = maze_load_runs(save_to)
    assert len(runs) == 1 and runs[0]["N"] == 40 and runs[0]["seed"] == 3 and runs[0]["cue"] is False
    assert [p["update"] for p in runs[0]["progress"]] == [1, 2]
    assert list(runs[0]["probes"]["update"]) == [1, 2] and len(runs[0]["probes"]["p"]) == 1
    cfg, task, trainer = load_run(folder)
    assert cfg.task.n_inputs == 10 and cfg.model.n_inputs == 10 and cfg.model.n_outputs == 2
    assert cfg.task.seed == 3 and cfg.seed == 3
    assert task.n_inputs == 10 and trainer.T == 6 and trainer.probe_episodes == 4
    saved = np.load(os.path.join(folder, "s3_Params_Maze.npz"))
    assert np.array_equal(trainer.RNN.W_rec.detach().numpy(), saved["W_rec"])


def test_fr_magnitude_penalty_matches_formula():
    """The rate penalty of a hand-made state tensor equals the supervised formula re-typed here in
    numpy float64 (nothing imported from Trainer.py): per unit the soft maximum of its ReLU rate
    over steps and episodes, tau x (logsumexp(r / tau) - log(n)) with tau 0.1; the cap 0.3 x
    log1p(100) / log1p(N); ((cap - a) / cap)^3 below the cap and ((a - cap) / cap)^3 above it;
    the mean over units. A silent unit pays exactly 1, a unit held at the cap pays nothing, and
    the value comes back in float64."""
    task = make_task()
    trainer = make_trainer(task, N=32, lambda_frm=0.1)
    N, T, B, tau = 32, 7, 3, 0.1
    cap = 0.3 * math.log1p(100) / math.log1p(N)
    rng = np.random.default_rng(0)
    states = rng.normal(0.0, 0.5, size=(N, T, B)).astype(np.float32)   # signed: the ReLU matters
    states[0] = 10.0                                                    # far above the cap
    states[1] = -1.0                                                    # silent: rate 0
    states[2] = cap                                                     # at the cap
    r = np.maximum(states.reshape(N, -1).astype(np.float64), 0.0)
    a = r / tau
    soft = tau * (np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - math.log(T * B))
    per_unit = np.where(soft > cap, ((soft - cap) / cap) ** 3, ((cap - soft) / cap) ** 3)
    expect = per_unit.mean()
    got = trainer.fr_magnitude_penalty(torch.tensor(states))
    assert got.dtype == torch.float64
    assert abs(got.item() - expect) < 1e-5 * expect, (got.item(), expect)
    assert per_unit[1] == 1.0 and per_unit[2] < 1e-12 and per_unit[0] > 1e4


def test_rec_weights_sparsity_penalty_matches_formula():
    """The sparsity penalty of a hand-made W_rec equals the supervised formula re-typed here: per
    row the effective number of synapses S = (sum |W|)^2 / sum W^2, relu(S - 20)^2 averaged over
    rows and divided by 400. A row with one synapse (S = 1) and a row of 20 equal synapses (S = 20)
    pay nothing; a row of 32 equal synapses (S = 32) pays 144 / 400 before the mean."""
    task = make_task()
    trainer = make_trainer(task, N=32, lambda_rws=0.05)
    rng = np.random.default_rng(1)
    W = rng.normal(size=(32, 32)) * rng.random((32, 32)) ** 3            # uneven rows, most under 20
    W[0] = 0.0
    W[0, 5] = 0.7                                                       # S = 1
    W[1] = 0.0
    W[1, :20] = -0.3                                                    # S = 20
    W[2] = 0.1                                                          # S = 32
    W[3:10] = 0.2 + 0.01 * rng.normal(size=(7, 32))                     # S near 32
    with torch.no_grad():
        trainer.RNN.W_rec.copy_(torch.tensor(W, dtype=torch.float32))
    S = np.abs(W).sum(1) ** 2 / (W ** 2).sum(1)
    expect = (np.maximum(S - 20.0, 0.0) ** 2).mean() / 400.0
    got = trainer.rec_weights_sparsity_penalty().item()
    assert abs(S[0] - 1) < 1e-9 and abs(S[1] - 20) < 1e-9 and abs(S[2] - 32) < 1e-9 and expect > 0
    assert abs(got - expect) < 1e-4 * expect, (got, expect)


def test_synaptic_noise_replay():
    """With sigma_w = 1 the recompute regenerates every step's weight draws from the rollout's
    stored seed: its log-probabilities equal the rollout's within float rounding (ratio 1), with
    and without autograd (the checkpointed path, whose gradient reaches W_rec); another seed, or no
    noise, gives other means, so the noise acts; a clean probe rollout draws no seed."""
    torch.manual_seed(4)
    task = make_task(goal_cue=True)
    trainer = make_trainer(task, substeps=2, sigma_w=1.0)
    T, B = 15, 5
    out = trainer.rollout(B, T)
    assert isinstance(out["w_seed"], int) and out["outgoing"] is None and out["participation"] is None
    with torch.no_grad():
        _, means, _ = trainer.sequence(out["u"], out["noise"], out["w_seed"])
        logp = trainer.log_prob(out["actions"], means[:, :T])
        _, other, _ = trainer.sequence(out["u"], out["noise"], out["w_seed"] + 1)
        _, plain, _ = trainer.sequence(out["u"], out["noise"])
    assert torch.allclose(logp, out["logp"], atol=1e-5)
    assert torch.allclose((logp - out["logp"]).exp(), torch.ones_like(logp), atol=1e-4)
    assert not torch.allclose(other, means, atol=1e-3) and not torch.allclose(plain, means, atol=1e-3)
    _, means_grad, _ = trainer.sequence(out["u"], out["noise"], out["w_seed"])
    assert torch.allclose(means_grad, means, atol=1e-6)
    (means_grad ** 2).sum().backward()
    grad = trainer.RNN.W_rec.grad
    assert grad is not None and torch.isfinite(grad).all() and grad.abs().sum() > 0
    probe = trainer.rollout(B, T, clean=True, generator=torch.Generator().manual_seed(1))
    assert probe["w_seed"] is None


def test_dropout_replay():
    """With read-out dropout on (mute, half the pool, beta 4, rescale) a training rollout draws one
    mask: dropped units' W_out columns are zero and kept pool units' columns are scaled above 1 in
    the read-out that both passes use (readout_weights); the recompute's log-probabilities equal
    the rollout's (ratio 1); the mask changes the means, so the dropped-out policy is the one that
    acted; the by-hand masked read-out of the recomputed rates gives the means; the rollout's
    participation equals the project's numpy definition on the recomputed states; a clean probe
    rollout draws no mask; run_training moves the EMA off its 1e-6 start."""
    torch.manual_seed(5)
    task = make_task()
    T, B = 10, 6
    trainer = make_trainer(task, substeps=2, dropout=True, dropout_args=DROPOUT_ARGS, n_updates=2,
                           n_episodes=B, T=T, probe_episodes=4, probe_every=1, warmup_updates=1, checkpoint_every=0)
    assert torch.all(trainer.p_bar == 1e-6)
    out = trainer.rollout(B, T)
    outgoing = out["outgoing"]
    assert outgoing.shape == (32, 1) and out["w_seed"] is None
    dropped = outgoing.flatten() == 0
    assert dropped.any() and (outgoing.flatten()[~dropped] >= 1).all() and outgoing.max() > 1
    W = trainer.readout_weights(outgoing)
    assert (W[:, dropped] == 0).all() and (W[:, ~dropped] != 0).any()
    with torch.no_grad():
        states, means, _ = trainer.sequence(out["u"], out["noise"], None, outgoing)
        logp = trainer.log_prob(out["actions"], means[:, :T])
        _, unmasked, _ = trainer.sequence(out["u"], out["noise"])
    assert torch.allclose(logp, out["logp"], atol=1e-5)
    assert torch.allclose((logp - out["logp"]).exp(), torch.ones_like(logp), atol=1e-4)
    assert not torch.allclose(unmasked, means, atol=1e-3)
    rates = trainer.RNN.get_firing_rates(states)
    by_hand = torch.einsum("oj,jtk->otk", W, rates[:, 1::2] + out["noise"]["out"])[:2]
    assert torch.allclose(by_hand, means, atol=1e-5)
    assert np.allclose(out["participation"].numpy(), participation(rates.numpy()), atol=1e-5)
    probe = trainer.rollout(B, T, clean=True, generator=torch.Generator().manual_seed(1))
    assert probe["outgoing"] is None and probe["participation"] is None
    trainer.run_training()
    assert torch.isfinite(trainer.p_bar).all() and (trainer.p_bar > 1e-6).any()


def test_penalties_in_update_statistics():
    """With lambda_frm and lambda_rws on, ppo_update reports the raw penalty values under
    fr_magnitude and rec_weights_sparsity: finite, equal to the penalties computed directly on the
    pre-update recompute (one epoch, one minibatch), with the weights changed and nothing skipped.
    With both at 0 the statistics have no such keys, so Progress.json of a run without the options
    is as before."""
    torch.manual_seed(6)
    task = make_task()
    trainer = make_trainer(task, critic=False, lambda_frm=0.1, lambda_rws=0.05, n_epochs=1, n_minibatches=1)
    out = trainer.rollout(6, 12)
    adv, ret = trainer.mc_advantages(out["rewards"], out["alive"], 0.99)
    with torch.no_grad():
        expect_frm = trainer.fr_magnitude_penalty(trainer.sequence(out["u"], out["noise"])[0]).item()
        expect_rws = trainer.rec_weights_sparsity_penalty().item()
    before = trainer.RNN.W_rec.detach().clone()
    stats = trainer.ppo_update(out, adv, ret)
    assert all(np.isfinite(v) for v in stats.values()) and stats["skipped"] == 0
    assert abs(stats["fr_magnitude"] - expect_frm) < 1e-5 * expect_frm and expect_frm > 0
    assert abs(stats["rec_weights_sparsity"] - expect_rws) < 1e-5 * max(expect_rws, 1e-12)
    assert not torch.allclose(before, trainer.RNN.W_rec)
    plain = make_trainer(task, critic=False, n_epochs=1, n_minibatches=1)
    stats = plain.ppo_update(out, adv, ret)
    assert not ({"fr_magnitude", "rec_weights_sparsity"} & set(stats))


def test_dead_dropout_refused():
    """dropout_kind "dead" (the unit removed from the recurrent dynamics as well) is not mirrored
    and is refused at construction with a message naming "mute"."""
    task = make_task()
    try:
        make_trainer(task, dropout=True, dropout_args={**DROPOUT_ARGS, "dropout_kind": "dead"})
    except ValueError as err:
        assert "mute" in str(err)
    else:
        raise AssertionError("dropout_kind 'dead' was accepted")


def main():
    """Run every check and print one line each."""
    for fn in (test_maze_tables, test_sensors, test_step_wall_and_goal, test_corner_is_not_cut_but_slides,
               test_slide_along_wall, test_only_the_rnn_is_trained, test_gae_known_case,
               test_mc_advantages_known_case, test_recompute_matches_rollout, test_ppo_update_runs,
               test_ppo_update_guards, test_safe_rollout_restores, test_run_script_writes_standard_layout,
               test_fr_magnitude_penalty_matches_formula, test_rec_weights_sparsity_penalty_matches_formula,
               test_synaptic_noise_replay, test_dropout_replay, test_penalties_in_update_statistics,
               test_dead_dropout_refused):
        fn()
        print(f"ok: {fn.__name__}")


if __name__ == "__main__":
    main()
