"""Checks for the maze task and the PPO trainer (trainRNNbrain/tasks/TaskMaze.py,
trainRNNbrain/trainer/TrainerPPO.py): the maze tables, the sensors, a blocked move and the goal,
GAE on a known case, that the PPO recompute reproduces the rollout exactly (probability ratio 1 at
the old parameters), which is what the stored-noise replay is for, the update's guards and the
rollback, that the training script writes the standard run layout that common.maze_load_runs
reads back, and the paper's interventions as trainer options (issue #5): the two penalties against
the supervised formulas re-typed here, the replay of synaptic noise and of the read-out dropout
mask in the recompute (ratio 1, dropped columns zero in both passes), the penalty values in the
update statistics, and the refusal of dropout kind "dead"; and the rescale rule (stage 2 of
issue #5): the three states of two units forced silent against numpy re-implementations of the
tilt, the participation and the soft-max rate, the blanked gradient of a growing unit, the rule's
counts in the progress records and the refusal of the modes that are not mirrored; the copy rule
(prune-and-duplicate, issue #6): every silent unit overwritten with a jittered copy of a live unit
against a numpy re-implementation fed the replayed generator draws, Adam's moments zeroed on the
copies' rows and columns, the ceiling on replacements per check and the maturity protection, and
the rule's counts in the progress records; and the
continuation of a run from its folder (paths.init_from, paths.init_adam, trainer.iter_offset): the
weights, the absolute update indices, the joined record of common.maze_load_runs, and the warm-up
ramp skipped with the Adam moments and repeated without them. The 13 checks before the options were
added run with every option off, as the default is.
Run: python tests/test_rl_maze.py   (CPU, about a minute and a half)
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import numpy as np
import torch
from omegaconf import OmegaConf
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
# the paper's rescaling arm (slurm/SilentReLU_size_series_della.slurm), the prune_args block of trainer_ppo.yaml
RESCALE_ARGS = {"reinit_mode": "rescale", "check_every": 1, "patience": 1, "active_rel": 0.05,
                "utility": "participation", "utility_decay": 0.99, "maturity": 0, "max_replace_frac": 1.0,
                "rescale_alpha": 1.002, "rescale_target_frac": 10.0, "rescale_cap": 64.0, "rescale_protect_frac": 0.20,
                "rescale_active_only": True, "rescale_normalize": True, "rescale_norm_mode": "self"}
# the paper's prune-and-duplicate arm (slurm/SilentReLU_paper_grid_della.slurm, with the cadence it
# inherits from trainer_ptrack_freshbatch.yaml), as the maze launcher passes it
COPY_ARGS = {"reinit_mode": "copy", "check_every": 100, "patience": 3, "active_rel": 0.05,
             "utility": "contribution", "utility_decay": 0.99, "maturity": 1000, "max_replace_frac": 0.025,
             "copy_noise": 1.0, "copy_norm_preserve": True}


def duplicate_by_hand(W_rec, W_inp, W_out, pairs, draws, jitter, norm_preserve):
    """The copy rule applied by hand in numpy float64 on a network with self-connections, one
    (copy, donor) pair after another: the donor's W_rec and W_out columns halved and written to the
    copy's columns; the copy's W_rec and W_inp rows the donor's rows as they are when the pair is
    reached, each entry times (1 + jitter x z), scaled back to the donor's row norms when
    norm_preserve; the 2 x 2 block of W_rec spanning the pair at half the donor's self-weight as it
    was when the pair is reached.

    Args:
        W_rec: (N, N), W_inp: (N, n_in), W_out: (n_out, N) float arrays, left unchanged;
        pairs: list of (copy_i, donor_j) int pairs, in order; draws: list of (z_rec (N,),
        z_inp (n_in,)) float arrays, one per pair; jitter: float; norm_preserve: bool.
    Returns:
        (W_rec, W_inp, W_out) float64 arrays after the rule.
    """
    W_rec, W_inp, W_out = (np.array(W, dtype=np.float64) for W in (W_rec, W_inp, W_out))
    for (i, j), (z_rec, z_inp) in zip(pairs, draws):
        row_rec, row_inp, self_w = W_rec[j].copy(), W_inp[j].copy(), float(W_rec[j, j])
        W_rec[:, j] *= 0.5
        W_out[:, j] *= 0.5
        W_rec[:, i], W_out[:, i] = W_rec[:, j], W_out[:, j]
        new_rec, new_inp = row_rec * (1.0 + jitter * z_rec), row_inp * (1.0 + jitter * z_inp)
        if norm_preserve:
            new_rec *= np.linalg.norm(row_rec) / np.linalg.norm(new_rec)
            new_inp *= np.linalg.norm(row_inp) / np.linalg.norm(new_inp)
        W_rec[i], W_inp[i] = new_rec, new_inp
        W_rec[i, i] = W_rec[i, j] = W_rec[j, i] = 0.5 * self_w
    return W_rec, W_inp, W_out


def tilt_by_hand(row, alpha, live=None):
    """The rescale tilt of one incoming row, in numpy float64: entries above 0 times alpha, the
    rest divided by alpha, only in the columns `live` marks when it is given, and the row then
    scaled back to the norm it had.

    Args:
        row: (n,) float array; alpha: float; live: (n,) bool array, or None for every column.
    Returns:
        (n,) float64 array.
    """
    row = row.astype(np.float64)
    tilted = np.where(row > 0, row * alpha, row / alpha)
    if live is not None:
        tilted = np.where(live, tilted, row)
    return tilted * (np.linalg.norm(row) / np.linalg.norm(tilted))


def soft_max_rate(states, tau=0.1):
    """Per-unit soft maximum of the ReLU rate over every step and episode, tau x (logsumexp(r / tau)
    - log(n)), in numpy float64 (the statistic the rate penalty and the rescale target use).

    Args:
        states: (N, S, B) array of pre-activations; tau: float.
    Returns:
        (N,) float64 array.
    """
    a = np.maximum(states.reshape(states.shape[0], -1).astype(np.float64), 0.0) / tau
    m = a.max(axis=1, keepdims=True)
    return tau * (np.log(np.exp(a - m).sum(axis=1)) + m[:, 0] - math.log(a.shape[1]))


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
    # the weights after the last update are themselves unstable: every replay from `good` fails
    # and the last checkpoint's weights are restored instead (counted beyond the three tries)
    snapshot = {k: v.clone() for k, v in good.items()}
    bad = {k: v.clone() for k, v in good.items()}
    bad["W_rec"][0, 0] = float("nan")
    with torch.no_grad():
        trainer.RNN.W_rec[0, 0] = float("nan")
    batch, restored = trainer.safe_rollout(bad, 4, 10, snapshot=snapshot)
    assert restored == 4 and torch.isfinite(batch["logp"]).all()
    assert torch.equal(trainer.RNN.W_rec, snapshot["W_rec"])


def test_run_script_writes_standard_layout():
    """The training script, dispatched to the maze by trainer=trainer_ppo, writes
    <save_to>/Maze_<tag>/s<seed>/ with s<seed>_config.yaml (seed and channel counts recorded),
    s<seed>_Params_Maze.npz (no *LastParams* file, so no cell inventory counts the run),
    s<seed>_Progress.json, s<seed>_Participation.npz, s<seed>_AdamState_Maze.pt and the two PNGs,
    at N = 40 for 2 updates;
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
    assert names == ["progress.png", "s3_AdamState_Maze.pt", "s3_Params_Maze.npz", "s3_Participation.npz",
                     "s3_Progress.json", "s3_config.yaml", "trajectories.png"], names
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


def run_maze_script(save_to, tag, extra, seed=3):
    """Run the training script on the maze at N = 40, seed 3, 2 updates of 4 episodes x 6 decisions,
    probe every 2 updates, warm-up 2, into <save_to>/Maze_<tag>/s<seed>, with `extra` overrides.

    Args:
        save_to: str, paths.save_to; tag: str, experiment_tag; extra: list of str, further
        command-line overrides; seed: int.
    Returns:
        str, the run folder.
    """
    cmd = [sys.executable, os.path.join(REPO, "trainRNNbrain", "training", "run_experiment.py"),
           "task=Maze", "trainer=trainer_ppo", "model=rnn_relu_standard", "model.N=40", f"seed={seed}",
           "trainer.n_updates=2", "trainer.n_episodes=4", "trainer.T=6", "trainer.probe_episodes=4",
           "trainer.probe_every=2", "trainer.checkpoint_every=0", "trainer.warmup_updates=2",
           f"paths.save_to={save_to}", f"experiment_tag={tag}",
           f"hydra.run.dir={os.path.join(os.path.dirname(save_to), 'hydra')}"] + extra
    env = dict(os.environ, PYTHONPATH=REPO, CUDA_VISIBLE_DEVICES="")
    r = subprocess.run(cmd, cwd=REPO, env=env, capture_output=True, text=True, timeout=600)
    assert r.returncode == 0, r.stderr[-2000:]
    return os.path.join(save_to, f"Maze_{tag}", f"s{seed}")


def test_continuation_from_folder():
    """A 2-update run, then continuations from its folder (paths.init_from), each 2 updates with the
    warm-up at 2 updates: with init_adam, the first update's rate is the full rate (no ramp), the
    progress and probe updates are 3 and 4 (trainer.iter_offset resolved to 2 and recorded), and
    maze_load_runs returns the parent and the child as one record with 4 updates in order, the
    parent's probes first, the parent not listed on its own; without init_adam the ramp is repeated
    (the first update's rate is lr / warmup_updates); and with init_adam at rate 0 the saved
    weights equal the parent's last bitwise, so the continuation starts from them."""
    scratch = os.path.join(os.environ.get("CLAUDE_SCRATCHPAD",
                                          os.path.join(tempfile.gettempdir(), "trainRNNbrain_scratch")),
                           "maze-resume", "continuation_check")
    shutil.rmtree(scratch, ignore_errors=True)
    save_to = os.path.join(scratch, "trained_RNNs")
    parent = run_maze_script(save_to, "parent", [])
    with open(os.path.join(parent, "s3_Progress.json")) as fh:
        parent_progress = json.load(fh)
    lr = parent_progress[-1]["lr"]                       # the full rate: the parent's ramp is over at update 2
    assert [p["update"] for p in parent_progress] == [1, 2] and parent_progress[0]["lr"] == lr / 2
    assert os.path.isfile(os.path.join(parent, "s3_AdamState_Maze.pt"))

    child = run_maze_script(save_to, "child", [f"paths.init_from={parent}", "paths.init_adam=true"])
    with open(os.path.join(child, "s3_Progress.json")) as fh:
        child_progress = json.load(fh)
    assert [p["update"] for p in child_progress] == [3, 4], child_progress
    assert [p["lr"] for p in child_progress] == [lr, lr], [p["lr"] for p in child_progress]
    child_probes = np.load(os.path.join(child, "s3_Participation.npz"))
    assert list(child_probes["update"]) == [3, 4] and list(child_probes["p_update"]) == [3]
    saved_cfg = OmegaConf.load(os.path.join(child, "s3_config.yaml"))
    assert saved_cfg.trainer.iter_offset == 2 and saved_cfg.paths.init_from == parent and saved_cfg.paths.init_adam

    fresh = run_maze_script(save_to, "fresh_adam", [f"paths.init_from={parent}", "paths.init_adam=false"])
    with open(os.path.join(fresh, "s3_Progress.json")) as fh:
        fresh_progress = json.load(fh)
    assert [p["update"] for p in fresh_progress] == [3, 4]
    assert fresh_progress[0]["lr"] == lr / 2 and fresh_progress[1]["lr"] == lr, [p["lr"] for p in fresh_progress]

    still = run_maze_script(save_to, "still", [f"paths.init_from={parent}", "paths.init_adam=true", "trainer.lr=0"])
    a, b = np.load(os.path.join(parent, "s3_Params_Maze.npz")), np.load(os.path.join(still, "s3_Params_Maze.npz"))
    for key in ("W_rec", "W_inp", "W_out", "y_init"):
        assert np.array_equal(a[key], b[key]), key

    runs = maze_load_runs(save_to)
    assert [os.path.basename(os.path.dirname(r["folder"])) for r in runs] == ["Maze_child", "Maze_fresh_adam", "Maze_still"]
    joined = runs[0]
    assert joined["chain"] == [os.path.abspath(parent), os.path.abspath(child)]
    assert [p["update"] for p in joined["progress"]] == [1, 2, 3, 4]
    assert list(joined["probes"]["update"]) == [1, 2, 3, 4] and list(joined["probes"]["p_update"]) == [1, 3]
    assert joined["probes"]["p"].shape == (2, 40) and len(joined["probes"]["active"]) == 4
    parent_probes = np.load(os.path.join(parent, "s3_Participation.npz"))
    assert np.array_equal(joined["probes"]["probe_success"][:2], parent_probes["probe_success"])
    assert np.array_equal(joined["probes"]["probe_success"][2:], child_probes["probe_success"])
    assert joined["progress"][:2] == parent_progress and joined["progress"][2:] == child_progress


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
    states[0] = 2.0 * cap                                               # above the cap, penalty 1
    states[1] = -1.0                                                    # silent: rate 0, penalty 1
    states[2] = cap                                                     # at the cap: nothing
    r = np.maximum(states.reshape(N, -1).astype(np.float64), 0.0)
    a = r / tau
    soft = tau * (np.log(np.exp(a - a.max(1, keepdims=True)).sum(1)) + a.max(1) - math.log(T * B))
    per_unit = np.where(soft > cap, ((soft - cap) / cap) ** 3, ((cap - soft) / cap) ** 3)
    expect = per_unit.mean()
    got = trainer.fr_magnitude_penalty(torch.tensor(states))
    assert got.dtype == torch.float64
    # no unit far above the cap, so the units below it carry the mean (the torch soft max runs in
    # float32, hence the 1e-5 tolerance); a wrong exponent on either side is then visible
    assert abs(got.item() - expect) < 1e-5 * expect, (got.item(), expect)
    assert per_unit[1] == 1.0 and per_unit[2] < 1e-12 and abs(per_unit[0] - 1.0) < 1e-6
    other = make_trainer(task, N=32, lambda_frm=0.1, frm_args={"cap_fr": 0.3, "tau": 0.1, "g_top": 3.0, "g_bot": 5.0})
    assert abs(other.fr_magnitude_penalty(torch.tensor(states)).item() - expect) > 1e-3 * expect


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


def test_rescale_rule_three_states():
    """The rescale rule on two units forced silent, A (incoming rows negative but for one input and
    a few live sources, so the tilt can revive it) and B (every incoming entry negative, so nothing
    can), the rule called once per check on the states `sequence` recomputes for a fresh rollout
    (what ppo_update hands it after each optimizer step), with no optimizer step in between, at the
    paper's values but patience 2, alpha 2, target 0.3 x the rate cap and no ceiling on the growing
    set. The oracles are numpy: common.participation for the silent set, tilt_by_hand, soft_max_rate.
    Check 1: both silent, one strike, nothing opened, no weight touched. Check 2 (patience): both
    open; the W_rec and W_out columns of every opened unit are the next draws of the RNN's generator
    at 1/sqrt(N) and every other column is untouched; the input rows of A and B are the hand-tilted
    rows at the same norm (1e-6); their factor is alpha. Check 3: the recurrent row of A and of B
    is the hand-tilted row over the live columns only, at the same norm; the factor is alpha^2; the
    columns opened at check 2 are not redrawn again. A graduates at the first check at which its
    soft-max rate reaches the target, untouched by that check and by every later one. B never
    fires: its factor doubles per check up to the cap (64 = 2^6) and never above it; at the cap B
    stops growing with its rows left as they are, and two checks later (patience) a new episode
    opens with a fresh factor of alpha and a redrawn column."""
    torch.manual_seed(8)
    task = make_task()
    N, B, T, alpha, cap = 32, 6, 10, 2.0, 64.0
    args = {**RESCALE_ARGS, "patience": 2, "rescale_alpha": alpha, "rescale_target_frac": 0.3, "rescale_protect_frac": 1.0}
    trainer = make_trainer(task, N=N, critic=False, prune_reinit=True, prune_args=args)
    rnn = trainer.RNN
    A, Bu = 3, 5
    g1 = torch.Generator().manual_seed(1)
    with torch.no_grad():
        rnn.W_inp[A, :] = -0.3
        rnn.W_inp[A, 0] = 0.05
        rnn.W_rec[A, :] = -0.3 * torch.rand(N, generator=g1)
        rnn.W_rec[A, 10:20] = 0.02
        rnn.W_inp[Bu, :] = -0.3
        rnn.W_rec[Bu, :] = -0.3 * torch.rand(N, generator=g1)
    target = 0.3 * 0.3 * math.log1p(100) / math.log1p(N)
    std = 1 / math.sqrt(N)
    graduated_at, spent_at = None, None
    for k in range(1, 15):
        out = trainer.rollout(B, T)
        with torch.no_grad():
            st = trainer.sequence(out["u"], out["noise"])[0]
        assert st.shape == (N, T + 1, B)                     # T decisions plus the bootstrap one, as ppo_update has them
        p = participation(rnn.get_firing_rates(st).numpy())
        silent = p < 0.05 * np.quantile(p, 0.95)
        act = soft_max_rate(st.numpy())
        W_rec0, W_inp0, W_out0 = rnn.W_rec.detach().clone(), rnn.W_inp.detach().clone(), rnn.W_out.detach().clone()
        growing0, cum0 = trainer._rescale_growing.clone(), trainer._rescale_cum.clone()
        gstate = rnn.random_generator.get_state()
        counts = trainer.prune_and_reinit(st)
        growing, cum = trainer._rescale_growing, trainer._rescale_cum
        assert counts["rescale_growing"] == int(growing.sum()) and trainer.step_n == k
        if k == 1:
            assert silent[A] and silent[Bu] and counts["rescale_opened"] == 0
            assert not growing[A] and not growing[Bu] and trainer._reinit_strikes[A] == 1 and trainer._reinit_strikes[Bu] == 1
            assert torch.equal(rnn.W_rec, W_rec0) and torch.equal(rnn.W_inp, W_inp0) and torch.equal(rnn.W_out, W_out0)
        elif k == 2:
            opened = torch.nonzero(growing & ~growing0).flatten()
            assert growing[A] and growing[Bu] and counts["rescale_opened"] == opened.numel() >= 2
            g = torch.Generator()
            g.set_state(gstate)
            exp_rec = torch.randn(N, opened.numel(), generator=g) * std
            exp_out = torch.randn(2, opened.numel(), generator=g) * std
            free = ~growing                                      # rows the tilt did not touch
            assert torch.equal(rnn.W_rec[free][:, opened], exp_rec[free]) and torch.equal(rnn.W_out[:, opened], exp_out)
            kept = torch.ones(N, dtype=torch.bool)
            kept[opened] = False
            assert torch.equal(rnn.W_rec[free][:, kept], W_rec0[free][:, kept]) and torch.equal(rnn.W_out[:, kept], W_out0[:, kept])
            for i in (A, Bu):
                assert abs(float(cum[i]) - alpha) < 1e-6
                assert np.allclose(rnn.W_inp[i].detach().numpy(), tilt_by_hand(W_inp0[i].numpy(), alpha), atol=1e-6)
                assert abs(float(rnn.W_inp[i].detach().norm()) - float(W_inp0[i].norm())) < 1e-6
        elif k == 3:
            for i in (A, Bu):
                assert growing0[i] and growing[i] and abs(float(cum[i]) - alpha ** 2) < 1e-5
                assert np.allclose(rnn.W_rec[i].detach().numpy(), tilt_by_hand(W_rec0[i].numpy(), alpha, live=~silent), atol=1e-6)
                assert abs(float(rnn.W_rec[i].detach().norm()) - float(W_rec0[i].norm())) < 1e-6
                assert np.allclose(rnn.W_inp[i].detach().numpy(), tilt_by_hand(W_inp0[i].numpy(), alpha), atol=1e-6)
                assert torch.equal(rnn.W_rec[~growing][:, i], W_rec0[~growing][:, i]) and torch.equal(rnn.W_out[:, i], W_out0[:, i])
        if graduated_at is None and growing0[A]:
            if act[A] >= target:
                graduated_at = k
                assert not growing[A] and counts["rescale_graduated"] >= 1
                assert torch.equal(rnn.W_rec[A], W_rec0[A]) and torch.equal(rnn.W_inp[A], W_inp0[A])
            else:
                assert growing[A]
        elif graduated_at is not None:
            assert not growing[A] and not silent[A] and torch.equal(rnn.W_inp[A], W_inp0[A])
        assert act[Bu] == 0 and silent[Bu] and float(cum[Bu]) <= cap
        if growing0[Bu]:
            if float(cum0[Bu]) >= cap:
                spent_at = k
                assert not growing[Bu] and counts["rescale_spent"] >= 1 and float(cum[Bu]) == cap
                assert torch.equal(rnn.W_rec[Bu], W_rec0[Bu]) and torch.equal(rnn.W_inp[Bu], W_inp0[Bu])
            else:
                assert growing[Bu] and abs(float(cum[Bu]) - alpha * float(cum0[Bu])) < 1e-4
        elif spent_at is not None and k == spent_at + 2:
            assert growing[Bu] and abs(float(cum[Bu]) - alpha) < 1e-6 and not torch.equal(rnn.W_out[:, Bu], W_out0[:, Bu])
    assert graduated_at is not None and 3 < graduated_at < spent_at == 8, (graduated_at, spent_at)
    assert float(trainer._rescale_cum[A]) == alpha ** (graduated_at - 2)


def test_rescale_protection_blanks_the_incoming_gradient():
    """A growing unit's incoming rows take no task gradient and do not move in an update while its
    outgoing column trains: with the most active unit j marked growing by hand, one update (one
    epoch, one minibatch) leaves W_rec.grad[j, :] and W_inp.grad[j, :] at exactly 0 after the
    blanking, W_rec[j, :] and W_inp[j, :] unchanged (Adam's moments start at 0), W_rec[:, j] and
    W_out[:, j] changed with a gradient, and other rows changed; the same rollout through a trainer
    without the option moves row j, so the row's gradient is not zero on its own. The rule's check
    after the step (patience 10^6, target 1e-6 x the cap) dooms no unit and graduates j without a
    tilt, so the weights after the update are the optimizer's alone; the step clock reads 1."""
    torch.manual_seed(9)
    task = make_task()
    args = {**RESCALE_ARGS, "patience": 10 ** 6, "rescale_target_frac": 1e-6}
    trainer = make_trainer(task, critic=False, n_epochs=1, n_minibatches=1, prune_reinit=True, prune_args=args)
    plain = make_trainer(task, critic=False, n_epochs=1, n_minibatches=1)
    assert torch.equal(trainer.RNN.W_rec, plain.RNN.W_rec)
    out = trainer.rollout(6, 12)
    adv, ret = trainer.mc_advantages(out["rewards"], out["alive"], 0.99)
    with torch.no_grad():
        j = int(trainer.participation_from_states(trainer.sequence(out["u"], out["noise"])[0]).argmax())
    trainer._rescale_growing[j] = True
    W_rec0, W_inp0 = trainer.RNN.W_rec.detach().clone(), trainer.RNN.W_inp.detach().clone()
    stats = trainer.ppo_update(out, adv, ret)
    assert trainer.step_n == 1 and not trainer._rescale_growing[j]
    assert stats["rescale_graduated"] == 1 and stats["rescale_opened"] == 0 and stats["rescale_growing"] == 0
    g_rec, g_inp, g_out = trainer.RNN.W_rec.grad, trainer.RNN.W_inp.grad, trainer.RNN.W_out.grad
    others = [i for i in range(32) if i != j]
    assert g_rec[j].abs().max() == 0 and g_inp[j].abs().max() == 0
    assert g_rec[others][:, j].abs().sum() > 0 and g_out[:, j].abs().sum() > 0 and g_rec[others].abs().sum() > 0
    assert torch.equal(trainer.RNN.W_rec[j], W_rec0[j]) and torch.equal(trainer.RNN.W_inp[j], W_inp0[j])
    assert not torch.equal(trainer.RNN.W_rec[others][:, j], W_rec0[others][:, j])
    assert (trainer.RNN.W_rec[others] != W_rec0[others]).any()
    plain.ppo_update(out, adv, ret)
    assert not torch.equal(plain.RNN.W_rec[j], W_rec0[j]) and not torch.equal(plain.RNN.W_inp[j], W_inp0[j])


def test_rescale_counts_in_progress():
    """run_training with the rule on (the paper's values, three units forced silent) records in
    every update the four counts rescale_growing, rescale_opened, rescale_graduated and
    rescale_spent as integers (opened, graduated and spent summed over the update's checks, one per
    optimizer step; growing after the last), with at least one episode opened, the growing set
    never above round(0.2 N) = 6, and the step clock at n_updates x n_epochs x n_minibatches =
    4 x 4 x 2 = 32 at the end; without the option the records have no such keys."""
    torch.manual_seed(10)
    task = make_task()
    hp = dict(n_updates=4, n_episodes=6, T=10, probe_episodes=4, probe_every=2, warmup_updates=1, checkpoint_every=0)
    trainer = make_trainer(task, critic=False, prune_reinit=True, prune_args=RESCALE_ARGS, **hp)
    with torch.no_grad():
        trainer.RNN.W_inp[:3, :] = -0.5
        trainer.RNN.W_rec[:3, :] = -0.3
    _, progress, _, _ = trainer.run_training()
    keys = {"rescale_growing", "rescale_opened", "rescale_graduated", "rescale_spent"}
    assert all(keys <= set(r) for r in progress)
    assert all(isinstance(r[k], int) and r[k] >= 0 for r in progress for k in keys)
    assert sum(r["rescale_opened"] for r in progress) >= 1 and max(r["rescale_growing"] for r in progress) <= 6
    assert trainer.step_n == 32 and all(r["skipped"] == 0 for r in progress)
    plain = make_trainer(task, critic=False, **hp)
    _, progress, _, _ = plain.run_training()
    assert not any(keys & set(r) for r in progress)


def test_copy_rule_duplicates_live_units():
    """The copy rule at N = 32 with three units forced silent (every incoming entry negative; at
    this init 12 units are silent in all), a check every step, patience 1, no ceiling, copy_noise
    1.0 with the norm preserved, called once on the states `sequence` recomputes for a fresh
    rollout, with Adam's moments set to 1 by hand beforehand. The silent set (common.participation
    in numpy) is the trainer's; every silent unit is overwritten and the donors are live units;
    replaying the RNN's generator from its saved state gives the donors (a multinomial over the
    live units' participation, with replacement, so a donor can serve twice) and the jitter draws,
    and the weights after the check equal duplicate_by_hand on the weights before it (1e-6). For
    every pair: the copy's W_inp row is the donor's times (1 + z) at the donor's row norm, at a
    cosine to the donor below 0.999 (changed) and below 0.9 on average over the pairs (not twins:
    the expected cosine at jitter 1 is 0.71), and the donor's W_inp row is unchanged. For a
    donor drawn once, over the units that are neither copies nor donors (whose rows and columns
    no pair touches): the copy's and the donor's W_rec columns are half the donor's old column,
    the W_out columns likewise, and the donor's W_rec row is unchanged; the block spanning the
    pair is half the donor's old self-weight. Every unit that is neither a copy nor a donor keeps
    its rows and columns. Adam's moments are 0
    on the copies' rows (W_rec, W_inp) and columns (W_rec, W_out) and 1 everywhere else, the
    donors' halved columns included. The counts: copy_replaced = copy_units_ever = the number of
    silent units; the strikes of the copies are reset; the step clock reads 1."""
    torch.manual_seed(11)
    task = make_task()
    N, B, T = 32, 6, 10
    args = {**COPY_ARGS, "check_every": 1, "patience": 1, "maturity": 0, "max_replace_frac": 1.0}
    trainer = make_trainer(task, N=N, critic=False, prune_reinit=True, prune_args=args)
    rnn = trainer.RNN
    with torch.no_grad():
        rnn.W_inp[:3, :] = -0.5
        rnn.W_rec[:3, :] = -0.3
    for prm in rnn.parameters():
        trainer.optimizer.state[prm] = {"step": torch.tensor(1.0), "exp_avg": torch.ones_like(prm),
                                        "exp_avg_sq": torch.ones_like(prm)}
    out = trainer.rollout(B, T)
    with torch.no_grad():
        st = trainer.sequence(out["u"], out["noise"])[0]
    p = participation(rnn.get_firing_rates(st).numpy())
    silent = p < 0.05 * np.quantile(p, 0.95)
    assert silent[:3].all() and 3 <= silent.sum() < N
    doomed = np.flatnonzero(silent)
    p_t = trainer.participation_from_states(st)
    assert np.array_equal((p_t < 0.05 * torch.quantile(p_t, 0.95)).numpy(), silent)
    W_rec0, W_inp0, W_out0 = (W.detach().clone().numpy() for W in (rnn.W_rec, rnn.W_inp, rnn.W_out))
    gstate = rnn.random_generator.get_state()
    counts = trainer.prune_and_reinit(st)
    n = len(doomed)
    assert counts == {"copy_replaced": n, "copy_units_ever": n} and trainer.step_n == 1
    assert torch.equal(trainer._reinit_strikes[doomed], torch.zeros(n)) and trainer._reinit_ever[doomed].all()
    # replay the generator: the donors, then z_rec and z_inp per pair in the order of the doomed units
    g = torch.Generator()
    g.set_state(gstate)
    live = torch.nonzero(~torch.as_tensor(silent), as_tuple=True)[0]
    donors = live[torch.multinomial(p_t[live].clamp_min(1e-12), n, replacement=True, generator=g)].tolist()
    assert not silent[donors].any()
    draws = [(torch.randn(N, generator=g).numpy(), torch.randn(task.n_inputs, generator=g).numpy()) for _ in doomed]
    pairs = list(zip(doomed.tolist(), donors))
    exp_rec, exp_inp, exp_out = duplicate_by_hand(W_rec0, W_inp0, W_out0, pairs, draws, 1.0, True)
    assert np.allclose(rnn.W_rec.detach().numpy(), exp_rec, atol=1e-6)
    assert np.allclose(rnn.W_inp.detach().numpy(), exp_inp, atol=1e-6)
    assert np.allclose(rnn.W_out.detach().numpy(), exp_out, atol=1e-6)
    W_rec, W_inp, W_out = (W.detach().numpy() for W in (rnn.W_rec, rnn.W_inp, rnn.W_out))
    once = [(i, j) for i, j in pairs if donors.count(j) == 1]
    assert len(once) >= 1
    cosines = []
    for i, j in pairs:
        assert abs(np.linalg.norm(W_inp[i]) - np.linalg.norm(W_inp0[j])) < 1e-5
        cosines.append(W_inp[i] @ W_inp0[j] / (np.linalg.norm(W_inp[i]) * np.linalg.norm(W_inp0[j])))
        assert cosines[-1] < 0.999 and np.array_equal(W_inp[j], W_inp0[j])
    assert np.mean(cosines) < 0.9, cosines
    rest = np.ones(N, dtype=bool)
    rest[doomed] = False
    rest[donors] = False
    for i, j in once:
        assert np.array_equal(W_rec[j][rest], W_rec0[j][rest])
        assert np.allclose(W_rec[rest, i], 0.5 * W_rec0[rest, j]) and np.array_equal(W_rec[rest, i], W_rec[rest, j])
        assert np.allclose(W_out[:, i], 0.5 * W_out0[:, j]) and np.array_equal(W_out[:, i], W_out[:, j])
        assert W_rec[i, i] == W_rec[i, j] == W_rec[j, i] == W_rec[j, j] and abs(W_rec[j, j] - 0.5 * W_rec0[j, j]) < 1e-7
    assert np.array_equal(W_rec[rest][:, rest], W_rec0[rest][:, rest]) and np.array_equal(W_inp[rest], W_inp0[rest])
    assert np.array_equal(W_out[:, rest], W_out0[:, rest])
    for key in ("exp_avg", "exp_avg_sq"):
        m_rec, m_inp, m_out = (trainer.optimizer.state[prm][key].numpy() for prm in (rnn.W_rec, rnn.W_inp, rnn.W_out))
        want_rec = np.ones((N, N))
        want_rec[doomed, :] = 0.0
        want_rec[:, doomed] = 0.0
        want_inp, want_out = np.ones((N, task.n_inputs)), np.ones((2, N))
        want_inp[doomed, :] = 0.0
        want_out[:, doomed] = 0.0
        assert np.array_equal(m_rec, want_rec) and np.array_equal(m_inp, want_inp) and np.array_equal(m_out, want_out)


def test_copy_ceiling_and_maturity():
    """The ceiling on replacements per check and the maturity protection, with the contribution
    utility: three units A, B, C forced silent, their utilities set by hand to 0.3, 0.2, 0.1 and
    every other unit's to 10 + its index (so no other silent unit ranks below them), at most
    round(0.0625 x 32) = 2 replacements per check, maturity 4 steps, patience 1, a check every step,
    four checks on the states of fresh rollouts with no optimizer step between (a unit's W_inp row
    changes exactly when it is replaced). Check 1 replaces C and B, the two lowest, and not A;
    check 2 replaces A (its strike from check 1 was kept) and neither B nor C, whose maturity has
    not passed; checks 3 and 4 replace none of the three (A's maturity); copy_replaced is at most 2
    at every check and copy_units_ever never decreases."""
    torch.manual_seed(12)
    task = make_task()
    N, B, T = 32, 6, 10
    A, Bu, C = 0, 1, 2
    args = {**COPY_ARGS, "check_every": 1, "patience": 1, "maturity": 4, "max_replace_frac": 0.0625}
    trainer = make_trainer(task, N=N, critic=False, prune_reinit=True, prune_args=args)
    rnn = trainer.RNN
    with torch.no_grad():
        rnn.W_inp[:3, :] = -0.5
        rnn.W_rec[:3, :] = -0.3
    trainer._unit_utility = 10.0 + torch.arange(N, dtype=torch.float32)
    trainer._unit_utility[[A, Bu, C]] = torch.tensor([0.3, 0.2, 0.1])
    replaced_at, ever = {}, 0
    for k in range(1, 5):
        out = trainer.rollout(B, T)
        with torch.no_grad():
            st = trainer.sequence(out["u"], out["noise"])[0]
        W_inp0 = rnn.W_inp.detach().clone()
        counts = trainer.prune_and_reinit(st)
        changed = [i for i in range(N) if not torch.equal(rnn.W_inp[i], W_inp0[i])]
        assert counts["copy_replaced"] == len(changed) <= 2 and counts["copy_units_ever"] >= ever
        ever = counts["copy_units_ever"]
        for i in changed:
            replaced_at.setdefault(i, k)
        three = {A, Bu, C} & set(changed)
        assert three == ({Bu, C} if k == 1 else {A} if k == 2 else set()), (k, changed)
    assert replaced_at[A] == 2 and replaced_at[Bu] == 1 and replaced_at[C] == 1


def test_copy_counts_in_progress():
    """run_training with the copy rule on (the paper's values but a check every 2 steps and
    patience 2, three units forced silent) records in every update the two counts copy_replaced
    (summed over the update's checks, one per optimizer step) and copy_units_ever (after the last
    check) as integers, with the three forced units replaced in the first update (their W_inp rows
    no longer all negative at the end) and copy_units_ever non-decreasing and at least 3; no
    rescale key is recorded; the step clock reads 4 x 4 x 2 = 32 at the end; without the option
    the records have no copy key."""
    torch.manual_seed(13)
    task = make_task()
    hp = dict(n_updates=4, n_episodes=6, T=10, probe_episodes=4, probe_every=2, warmup_updates=1, checkpoint_every=0)
    args = {**COPY_ARGS, "check_every": 2, "patience": 2, "maturity": 0, "max_replace_frac": 1.0}
    trainer = make_trainer(task, critic=False, prune_reinit=True, prune_args=args, **hp)
    with torch.no_grad():
        trainer.RNN.W_inp[:3, :] = -0.5
        trainer.RNN.W_rec[:3, :] = -0.3
    _, progress, _, _ = trainer.run_training()
    keys = {"copy_replaced", "copy_units_ever"}
    assert all(keys <= set(r) for r in progress) and not any("rescale" in k for r in progress for k in r)
    assert all(isinstance(r[k], int) and r[k] >= 0 for r in progress for k in keys)
    ever = [r["copy_units_ever"] for r in progress]
    assert progress[0]["copy_replaced"] >= 3 and ever == sorted(ever) and ever[-1] >= 3
    assert ever[-1] <= sum(r["copy_replaced"] for r in progress)
    assert (trainer.RNN.W_inp[:3] > 0).any(dim=1).all() and trainer._reinit_ever[:3].all()
    assert trainer.step_n == 32 and all(r["skipped"] == 0 for r in progress)
    plain = make_trainer(task, critic=False, **hp)
    _, progress, _, _ = plain.run_training()
    assert not any(keys & set(r) for r in progress)


def test_other_modes_refused():
    """prune_reinit without prune_args, a reinit_mode other than "rescale" or "copy", the legacy
    rescale_target_frac 0, rescale_norm_mode "median_active" and rescale_refractory above 0 in mode
    "rescale", and copy_iid or copy_permute in mode "copy" are refused at construction."""
    task = make_task()
    for bad in (None, {**RESCALE_ARGS, "reinit_mode": "random"}, {**RESCALE_ARGS, "rescale_target_frac": 0.0},
                {**RESCALE_ARGS, "rescale_norm_mode": "median_active"}, {**RESCALE_ARGS, "rescale_refractory": 5},
                {**COPY_ARGS, "copy_iid": True}, {**COPY_ARGS, "copy_permute": True}):
        try:
            make_trainer(task, prune_reinit=True, prune_args=bad)
        except ValueError as err:
            assert "rescale" in str(err) or "prune_args" in str(err) or "copy" in str(err)
        else:
            raise AssertionError(f"prune_args {bad} were accepted")
    make_trainer(task, prune_reinit=True, prune_args=COPY_ARGS)       # the paper's copy arm is accepted


def main():
    """Run every check and print one line each."""
    for fn in (test_maze_tables, test_sensors, test_step_wall_and_goal, test_corner_is_not_cut_but_slides,
               test_slide_along_wall, test_only_the_rnn_is_trained, test_gae_known_case,
               test_mc_advantages_known_case, test_recompute_matches_rollout, test_ppo_update_runs,
               test_ppo_update_guards, test_safe_rollout_restores, test_run_script_writes_standard_layout,
               test_continuation_from_folder,
               test_fr_magnitude_penalty_matches_formula, test_rec_weights_sparsity_penalty_matches_formula,
               test_synaptic_noise_replay, test_dropout_replay, test_penalties_in_update_statistics,
               test_dead_dropout_refused, test_rescale_rule_three_states,
               test_rescale_protection_blanks_the_incoming_gradient, test_rescale_counts_in_progress,
               test_copy_rule_duplicates_live_units, test_copy_ceiling_and_maturity, test_copy_counts_in_progress,
               test_other_modes_refused):
        fn()
        print(f"ok: {fn.__name__}")


if __name__ == "__main__":
    main()
