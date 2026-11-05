"""Animate trained maze agents: a GIF of a clean rollout of a saved policy.

Question. What does a trained agent do in the maze, step by step?

Method. The agent is rebuilt from a run folder (its saved config and parameter file, written by
trainRNNbrain/training/run_rl.py or, before 5 Nov 2026, trainRNNbrain/rl/run_maze_ppo.py;
trainRNNbrain.trainer.TrainerPPO.load_run) on the CPU and played on fixed probe drops with the RNN's
noise off and the policy mean as the action, the same clean rollout as the training probe. Each
frame is one decision: every agent is a dot with a short heading tick and a trail of its last
`trail` positions; an agent that reached the cheese turns into a hollow marker and stops. The
first agent's range sensors are drawn as thin grey rays (their length is the measured distance
to the first wall). Frames are subsampled by `--every`. The animation ends when the last agent
has reached the cheese (held for `--hold` frames, then the GIF loops and starts anew), and runs
to the episode's end only if an agent never arrives (Pavel, 4 Nov 2026).

Output. img/internal_figures/rl_maze_<name>.gif, and the last frame as rl_maze_<name>_last.png.
Run: python trainRNNbrain/experiments_and_analysis/rl_maze_animate.py --run <folder> --name <name>
"""
import argparse
import math
import os
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from common import IMG_DIR
from trainRNNbrain.trainer.TrainerPPO import load_run


def animate(cfg, task, out, path, every, trail, fps, hold=15):
    """Write the GIF of one rollout.

    Args:
        cfg: the run's DictConfig; task: TaskMaze; out: the rollout dict (xy, heading, u, steps,
        success); path: str, the .gif to write; every: int, decisions per frame; trail: int,
        trail length in decisions; fps: int, frames per second; hold: int, frames the last
        picture is held before the GIF loops.
    Returns:
        the matplotlib Figure of the last frame.
    """
    wall = task.wall.numpy()
    xy, heading, u = out["xy"].numpy(), out["heading"].numpy(), out["u"].numpy()
    steps, success = out["steps"].numpy().astype(int), out["success"].numpy()
    T1, B = xy.shape[0], xy.shape[1]
    colors = plt.cm.tab10(np.arange(B) % 10)
    fig, ax = plt.subplots(figsize=(5, 5.3))
    ax.imshow(wall, cmap="gray_r", origin="upper", extent=(0, wall.shape[1], wall.shape[0], 0))
    goal = task.goal_xy.numpy()
    ax.plot(goal[0], goal[1], "*", color="gold", ms=16, mec="k", zorder=3)
    ax.set_xticks([])
    ax.set_yticks([])
    trails = [ax.plot([], [], color=colors[b], lw=1, alpha=0.6)[0] for b in range(B)]
    dots = [ax.plot([], [], "o", color=colors[b], ms=6, mec="k", mew=0.5, zorder=4)[0] for b in range(B)]
    ticks = [ax.plot([], [], color="k", lw=1, zorder=5)[0] for b in range(B)]
    rays = [ax.plot([], [], color="grey", lw=0.5, alpha=0.7)[0] for _ in range(task.n_rays)]
    title = ax.set_title("", fontsize=9)
    label = (f"N = {cfg.model.N}, {'cue' if task.goal_cue else 'walls-only'}, seed {cfg.seed}: "
             f"{success.mean():.2f} of {B} reached the cheese")
    offsets = task.ray_offsets.numpy()

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
        dist = (1.0 - u[:task.n_rays, k, 0]) * task.ray_max
        for i, ray in enumerate(rays):
            a = heading[k, 0] + offsets[i]
            ray.set_data([xy[k, 0, 0], xy[k, 0, 0] + dist[i] * math.cos(a)],
                         [xy[k, 0, 1], xy[k, 0, 1] + dist[i] * math.sin(a)])
        title.set_text(f"{label}\ndecision {t} of {T1 - 1}")
        return trails + dots + ticks + rays + [title]

    end = int(steps.max()) + 1 if success.all() else T1 - 1     # the last arrival, or the episode's end
    last = end // every
    frames = list(range(last + 1)) + [last] * hold
    anim = FuncAnimation(fig, draw, frames=frames, blit=False)
    anim.save(path, writer=PillowWriter(fps=fps))
    draw(last)
    return fig


def main():
    """Parse arguments, play the clean rollout, write the GIF and the last frame."""
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run", required=True, help="run folder with a saved config and parameter file")
    ap.add_argument("--name", required=True, help="output name: rl_maze_<name>.gif")
    ap.add_argument("--agents", type=int, default=8, help="agents shown (probe drops, fixed seed)")
    ap.add_argument("--every", type=int, default=2, help="decisions per frame")
    ap.add_argument("--trail", type=int, default=60, help="trail length in decisions")
    ap.add_argument("--fps", type=int, default=15)
    ap.add_argument("--hold", type=int, default=15, help="frames to hold the last picture before the loop restarts")
    args = ap.parse_args()
    cfg, task, trainer = load_run(args.run)
    g = torch.Generator()
    g.manual_seed(trainer.probe_seed)
    out = trainer.rollout(args.agents, trainer.T, clean=True, generator=g)
    os.makedirs(IMG_DIR, exist_ok=True)
    gif = os.path.join(IMG_DIR, f"rl_maze_{args.name}.gif")
    fig = animate(cfg, task, out, gif, args.every, args.trail, args.fps, args.hold)
    fig.savefig(os.path.join(IMG_DIR, f"rl_maze_{args.name}_last.png"), dpi=100)
    print(f"{gif}: {out['success'].float().mean():.2f} of {args.agents} reached the cheese, "
          f"mean steps {out['steps'].float().mean():.0f}")


if __name__ == "__main__":
    main()
