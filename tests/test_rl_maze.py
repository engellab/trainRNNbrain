"""Checks for the maze task and the PPO path (trainRNNbrain/rl): the maze tables, the sensors, a
blocked move and the goal, GAE on a known case, and that the PPO recompute reproduces the rollout
exactly (probability ratio 1 at the old parameters), which is what the stored-noise replay is for.
Run: python tests/test_rl_maze.py   (CPU, a few seconds)
"""
import math
import os
import sys
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.rl.maze_env import MazeEnv, parse_maze, bfs_distance
from trainRNNbrain.rl.policy import MazePolicy
from trainRNNbrain.rl.ppo import rollout, gae, ppo_update

MAZE = ["#######",
        "#.....#",
        "#.###.#",
        "#.#C#.#",
        "#.#.#.#",
        "#.....#",
        "#######"]
REWARD = {"goal": 5.0, "step": -0.01, "wall": -0.05, "shape": 0.2}
CPU = torch.device("cpu")


def make_env(goal_cue=False, seed=0):
    """The small test maze on CPU: ray range 3, speed 0.2, turn 0.3, goal radius 0.4."""
    return MazeEnv(MAZE, n_rays=8, ray_max=3.0, ray_step=0.1, v_max=0.2, turn_max=0.3, goal_radius=0.4,
                   goal_cue=goal_cue, reward=REWARD, gamma=0.99, device=CPU, seed=seed)


def make_policy(n_inputs, N=32, seed=0):
    """A small standard-style RNN (ReLU, equation h, no Dale, fixed zero bias) wrapped as a policy."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, equation_type="h", dt=1, tau=10,
                    dale=False, io_nonnegativity=False, self_connections=True, gamma=0,
                    bias_range=[0.0, 0.0], sigma_rec=0.05, sigma_inp=0.05, sigma_out=0.03,
                    n_inputs=n_inputs, n_outputs=2, seed=seed, init_draws="independent")
    return MazePolicy(rnn, log_std_init=-0.7)


def test_maze_tables():
    """Every free cell reaches the cheese; the cheese is at 0; walls are inf; known distances hold."""
    wall, goal = parse_maze(MAZE)
    dist = bfs_distance(wall, goal)
    assert goal == (3, 3) and dist[goal] == 0
    assert np.isfinite(dist[~wall]).all() and np.isinf(dist[wall]).all()
    assert dist[4, 3] == 1 and dist[5, 1] == 4 and dist[1, 1] == 8 and dist[1, 3] == 10


def test_sensors():
    """An agent in the top corridor facing +x reads the walls at the measured distances."""
    env = make_env()
    env.reset(1)
    env.xy[:] = torch.tensor([[3.5, 1.5]])
    env.heading[:] = 0.0
    obs = env.observe()[:, 0]
    expect = {0: 1 - 2.5 / 3, 2: 1 - 0.5 / 3, 4: 1 - 2.5 / 3, 6: 1 - 0.5 / 3}   # +x, +y (down), -x, -y
    for ray, val in expect.items():
        assert abs(obs[ray].item() - val) < 0.04, (ray, obs[ray].item(), val)
    assert env.n_inputs == 8 and make_env(goal_cue=True).n_inputs == 10


def test_step_wall_and_goal():
    """A move into a wall is cancelled and costs the wall penalty; the cheese ends the episode."""
    env = make_env()
    env.reset(1)
    env.xy[:] = torch.tensor([[3.5, 1.95]])
    env.heading[:] = math.pi / 2                          # +y, into the wall of row 2
    _, r, done = env.step(torch.tensor([[0.0], [1.0]]))
    assert torch.allclose(env.xy, torch.tensor([[3.5, 1.95]]))
    assert abs(r.item() - (-0.01 - 0.05 + 0.2 * 10 * (1 - 0.99))) < 1e-5 and not done.item()
    env.xy[:] = torch.tensor([[3.5, 4.2]])
    env.heading[:] = -math.pi / 2                         # -y, towards the cheese at (3.5, 3.5)
    _, r, done = env.step(torch.tensor([[0.0], [1.0]]))   # y = 4.0: 0.5 away, not yet
    assert not done.item() and r.item() < 5
    _, r, done = env.step(torch.tensor([[0.0], [1.0]]))   # y = 3.8: 0.3 away, reached
    assert done.item() and r.item() > 4.5
    _, r, done = env.step(torch.tensor([[0.0], [1.0]]))   # finished agents do not move or earn
    assert done.item() and r.item() == 0 and abs(env.xy[0, 1].item() - 3.8) < 1e-6


def test_corner_is_not_cut():
    """A diagonal step from cell (4,3) into (5,2) passes the corner of the wall cell (4,2): blocked."""
    env = make_env()
    env.reset(1)
    env.xy[:] = torch.tensor([[3.05, 4.95]])
    env.heading[:] = 3 * math.pi / 4                      # (-x, +y): end point (2.91, 5.09)
    _, r, done = env.step(torch.tensor([[0.0], [1.0]]))
    assert torch.allclose(env.xy, torch.tensor([[3.05, 4.95]])) and r.item() < -0.05 and not done.item()
    env.heading[:] = math.pi / 2                          # straight down into (5,3): free
    env.step(torch.tensor([[0.0], [1.0]]))
    assert abs(env.xy[0, 1].item() - 5.15) < 1e-6


def test_gae_known_case():
    """Constant reward 1, no termination, zero values: the advantage is the geometric sum."""
    T = 10
    rewards, values, alive = torch.ones(T, 1), torch.zeros(T + 1, 1), torch.ones(T + 1, 1)
    adv, ret = gae(rewards, values, alive, gamma=0.5, lam=1.0)
    assert abs(adv[0, 0].item() - sum(0.5 ** k for k in range(T))) < 1e-6
    assert torch.allclose(ret, adv)
    alive[3:] = 0                                         # the episode ended after step 2
    adv, _ = gae(rewards, values, alive, gamma=0.5, lam=1.0)
    assert abs(adv[0, 0].item() - (1 + 0.5 + 0.25)) < 1e-6 and adv[3:].abs().sum() == 0


def test_recompute_matches_rollout():
    """With the stored noise, the sequence recompute gives the rollout's means, values and
    log-probabilities exactly, so the PPO ratio starts at 1."""
    torch.manual_seed(0)
    env = make_env(goal_cue=True)
    policy = make_policy(env.n_inputs)
    T, B = 25, 6
    out = rollout(env, policy, B, T)
    _, means, values = policy.sequence(out["u"], out["noise"])
    logp = policy.log_prob(out["actions"], means[:, :T])
    assert torch.allclose(logp, out["logp"], atol=1e-5)
    assert torch.allclose(values, out["values"], atol=1e-5)
    assert out["noise"]["out"].shape == (32, 1, B)      # one output-noise draw per episode, as forward
    assert torch.allclose((logp - out["logp"]).exp(), torch.ones_like(logp), atol=1e-4)


def test_ppo_update_runs():
    """One update changes the recurrent weights and returns finite statistics."""
    torch.manual_seed(1)
    env = make_env()
    policy = make_policy(env.n_inputs)
    opt = torch.optim.Adam(policy.parameters(), lr=1e-3)
    out = rollout(env, policy, 8, 30)
    adv, ret = gae(out["rewards"], out["values"], out["alive"], 0.99, 0.95)
    before = policy.rnn.W_rec.detach().clone()
    stats = ppo_update(policy, opt, out, adv, ret, clip=0.2, n_epochs=2, n_minibatches=2,
                       value_coef=0.5, entropy_coef=0.0, max_grad_norm=0.5)
    assert all(np.isfinite(v) for v in stats.values()), stats
    assert not torch.allclose(before, policy.rnn.W_rec)


def main():
    """Run every check and print one line each."""
    for fn in (test_maze_tables, test_sensors, test_step_wall_and_goal, test_corner_is_not_cut,
               test_gae_known_case, test_recompute_matches_rollout, test_ppo_update_runs):
        fn()
        print(f"ok: {fn.__name__}")


if __name__ == "__main__":
    main()
