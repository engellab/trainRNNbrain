"""Animate trained maze agents: a GIF of a clean rollout of a saved policy.

Question. What does a trained agent do in the maze, step by step?

Method. The policy is rebuilt from a run folder (config.yaml + PolicyState.pt, the full
state_dict written by run_maze_ppo.py) on the CPU and played on fixed probe drops with the RNN's
noise off and the policy mean as the action, the same clean rollout as the training probe. Each
frame is one decision: every agent is a dot with a short heading tick and a trail of its last
`trail` positions; an agent that reached the cheese turns into a hollow marker and stops. The
first agent's range sensors are drawn as thin grey rays (their length is the measured distance
to the first wall). Frames are subsampled by `--every`.

Output. img/internal_figures/rl_maze_<name>.gif, and the last frame as rl_maze_<name>_last.png.
Run: python trainRNNbrain/experiments_and_analysis/rl_maze_animate.py --run <folder> --name <name>
"""
import argparse
import math
import os
import hydra
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from omegaconf import OmegaConf
from common import IMG_DIR
from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.utils import filter_kwargs
from trainRNNbrain.rl.maze_env import MazeEnv
from trainRNNbrain.rl.policy import MazePolicy
from trainRNNbrain.rl.ppo import rollout


def load_run(folder):
    """Rebuild the env and the trained policy of a run folder on the CPU.

    Args:
        folder: str, a run folder holding config.yaml and PolicyState.pt.
    Returns:
        (cfg, env, policy): the saved DictConfig, the MazeEnv, the MazePolicy with trained weights.
    """
    cfg = OmegaConf.load(os.path.join(folder, "config.yaml"))
    env = MazeEnv(**cfg.env, device=torch.device("cpu"), seed=cfg.seed + 1)
    args = filter_kwargs(RNN_torch, cfg.model)
    args.seed = cfg.seed
    policy = MazePolicy(hydra.utils.instantiate(args), cfg.ppo.log_std_init, cfg.ppo.log_std_min,
                        cfg.ppo.substeps)
    policy.load_state_dict(torch.load(os.path.join(folder, "PolicyState.pt"), map_location="cpu"))
    return cfg, env, policy


def animate(cfg, env, out, path, every, trail, fps):
    """Write the GIF of one rollout.

    Args:
        cfg: the run's DictConfig; env: MazeEnv; out: the rollout dict (xy, heading, u, steps,
        success); path: str, the .gif to write; every: int, decisions per frame; trail: int,
        trail length in decisions; fps: int, frames per second.
    Returns:
        the matplotlib Figure of the last frame.
    """
    wall = env.wall.numpy()
    xy, heading, u = out["xy"].numpy(), out["heading"].numpy(), out["u"].numpy()
    steps, success = out["steps"].numpy().astype(int), out["success"].numpy()
    T1, B = xy.shape[0], xy.shape[1]
    colors = plt.cm.tab10(np.arange(B) % 10)
    fig, ax = plt.subplots(figsize=(5, 5.3))
    ax.imshow(wall, cmap="gray_r", origin="upper", extent=(0, wall.shape[1], wall.shape[0], 0))
    goal = env.goal_xy.numpy()
    ax.plot(goal[0], goal[1], "*", color="gold", ms=16, mec="k", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    trails = [ax.plot([], [], color=colors[b], lw=1, alpha=0.6)[0] for b in range(B)]
    dots = [ax.plot([], [], "o", color=colors[b], ms=6, mec="k", mew=0.5, zorder=4)[0] for b in range(B)]
    ticks = [ax.plot([], [], color="k", lw=1, zorder=5)[0] for b in range(B)]
    rays = [ax.plot([], [], color="grey", lw=0.5, alpha=0.7)[0] for _ in range(env.n_rays)]
    title = ax.set_title("", fontsize=9)
    label = (f"N = {cfg.model.N}, {'cue' if cfg.env.goal_cue else 'blind'}, seed {cfg.seed}: "
             f"{success.mean():.2f} of {B} reached the cheese")
    offsets = env.ray_offsets.numpy()

    def draw(frame):
        """Draw decision t = frame * every; returns the artists that changed."""
        t = min(frame * every, T1 - 1)
        for b in range(B):
            k = min(t, steps[b])                 # a finished agent stays where it stopped
            lo = max(0, k - trail)
            trails[b].set_data(xy[lo:k + 1, b, 0], xy[lo:k + 1, b, 1])
            dots[b].set_data([xy[k, b, 0]], [xy[k, b, 1]])
            done = success[b] and t >= steps[b]
            dots[b].set_markerfacecolor("white" if done else colors[b])
            dx, dy = 0.35 * math.cos(heading[k, b]), 0.35 * math.sin(heading[k, b])
            ticks[b].set_data([xy[k, b, 0], xy[k, b, 0] + dx], [xy[k, b, 1], xy[k, b, 1] + dy])
        k = min(t, steps[0])
        dist = (1.0 - u[:env.n_rays, k, 0]) * env.ray_max
        for i, ray in enumerate(rays):
            a = heading[k, 0] + offsets[i]
            ray.set_data([xy[k, 0, 0], xy[k, 0, 0] + dist[i] * math.cos(a)],
                         [xy[k, 0, 1], xy[k, 0, 1] + dist[i] * math.sin(a)])
        title.set_text(f"{label}\ndecision {t} of {T1 - 1}")
        return trails + dots + ticks + rays + [title]

    n_frames = (T1 - 1) // every + 1
    anim = FuncAnimation(fig, draw, frames=n_frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    draw(n_frames - 1)
    return fig


def main():
    """Parse arguments, play the clean rollout, write the GIF and the last frame."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder with config.yaml and PolicyState.pt")
    ap.add_argument("--name", required=True, help="output name: rl_maze_<name>.gif")
    ap.add_argument("--agents", type=int, default=8, help="agents shown (probe drops, fixed seed)")
    ap.add_argument("--every", type=int, default=2, help="decisions per frame")
    ap.add_argument("--trail", type=int, default=60, help="trail length in decisions")
    ap.add_argument("--fps", type=int, default=15)
    args = ap.parse_args()
    cfg, env, policy = load_run(args.run)
    g = torch.Generator()
    g.manual_seed(int(cfg.ppo.probe_seed))
    out = rollout(env, policy, args.agents, int(cfg.ppo.T), clean=True, generator=g)
    os.makedirs(IMG_DIR, exist_ok=True)
    gif = os.path.join(IMG_DIR, f"rl_maze_{args.name}.gif")
    fig = animate(cfg, env, out, gif, args.every, args.trail, args.fps)
    fig.savefig(os.path.join(IMG_DIR, f"rl_maze_{args.name}_last.png"), dpi=100)
    print(f"{gif}: {out['success'].float().mean():.2f} of {args.agents} reached the cheese, "
          f"mean steps {out['steps'].float().mean():.0f}")


if __name__ == "__main__":
    main()
