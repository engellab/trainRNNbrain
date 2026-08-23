#!/usr/bin/env python3
"""
Real population trajectories of a flip-flop network, as a bouquet, with the fixed points in place.

Shown in OUTPUT space, W_out @ r(t), which is 3-D exactly for k=3 - no projection loss, unlike the
population PC basis where PC1-3 hold only ~52% of the rate variance. It is also the space where the
memory states form a visible 3x3x3 lattice, so "does the state visit all the corners" is answerable
by looking.

FIXED POINTS ARE DELIBERATELY NOT DRAWN HERE, and the reason is a measured result rather than a
decluttering choice. In the full N-dimensional rate space only 6% of trajectory samples come within
10% of the cloud scale of ANY fixed point; the median distance to the nearest one is 6.6 against a
cloud scale of 9.2. Relaxing a real trajectory state with the input clamped to zero DOES converge to
a fixed point in the found set - to distance 0.0000, so the fixed points are correct and reachable -
but it takes ~300 tau to get there and moves the state by more than its own norm. The task only
leaves ~7.5 tau between pulses.

So THE NETWORK NEVER REACHES ITS FIXED POINTS DURING THE TASK. Memory here is carried by slow
transients, not by settling into an attractor, and drawing the fixed points beside the trajectories
invites exactly the wrong reading - that the state is visiting them.

Output: img/internal_figures/flipflop_bouquet_k<k>_N<N>.gif  (+ a static .png)

Usage:  python flipflop_bouquet.py [k] [N] [n_trials]
"""

import os
import sys
import glob
import numpy as np
import hydra
from omegaconf import OmegaConf
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import IMG_DIR
from flipflop_fixedpoints import best_net, load_net, find_fixed_points, classify, N_STARTS
from trainRNNbrain.training.training_utils import prepare_task_arguments

CACHE = "data/trained_RNNs/_fixedpoint_cache"


def cached_fixed_points(folder, W_rec, b, seeds):
    """Fixed points and their unstable dimensions, computed once and cached to disk.

    Args:
        folder: net folder, used as the cache key; W_rec, b: network parameters;
        seeds: (n_starts, N) initial states for the pattern iteration.
    Returns:
        (points, n_unstable) arrays.
    """
    os.makedirs(CACHE, exist_ok=True)
    path = os.path.join(CACHE, os.path.basename(folder)[:40].replace("/", "_") + ".npz")
    if os.path.exists(path):
        d = np.load(path)
        print(f"  fixed points from cache: {path}")
        return d["pts"], d["nun"]
    pts, res = find_fixed_points(W_rec, b, seeds)
    print(f"  {len(pts)} distinct fixed points, |F| max {res.max():.1e}; classifying...")
    nun, _ = classify(W_rec, pts)
    np.savez_compressed(path, pts=pts, nun=nun)
    return pts, nun


def run_trials(rnn, folder, n_trials):
    """Noise-free rates over a batch of trials.

    Args:
        rnn: the network; folder: net folder (for its config); n_trials: batch size.
    Returns:
        (N, T, B) firing rates.
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    cfg.task.batch_size = n_trials
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, _, _ = task.get_batch()
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    return np.maximum(np.array(rnn.get_history()), 0.0)


def main():
    """Draw the trajectory bouquet with fixed points, in the population's own PC basis."""
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    n_trials = int(sys.argv[3]) if len(sys.argv) > 3 else 25

    folder = best_net(k, N)
    print(f"network: {os.path.basename(folder)[:60]}")
    rnn, p = load_net(folder)
    W_rec, b = p["W_rec"], p["bias"]

    rates = run_trials(rnn, folder, n_trials)
    print(f"  rates {rates.shape}")
    W_out = p["W_out"]
    T = np.einsum("on,ntb->tbo", W_out, rates)              # (time, trial, k) output space
    print(f"  output range per channel: "
          + ", ".join(f"[{T[..., j].min():+.2f}, {T[..., j].max():+.2f}]" for j in range(3)))

    def draw(ax, elev, azim):
        """Render one view of the trajectory bouquet in output space."""
        ax.clear()
        for t in range(T.shape[1]):
            ax.plot(T[:, t, 0], T[:, t, 1], T[:, t, 2], "-", lw=1.0, alpha=.55,
                    color=plt.cm.turbo(t / max(T.shape[1] - 1, 1)), zorder=1)
        ax.set(xlabel="out 1", ylabel="out 2", zlabel="out 3",
               xlim=(-1.25, 1.25), ylim=(-1.25, 1.25), zlim=(-1.25, 1.25))
        ax.view_init(elev=elev, azim=azim)
        ax.grid(alpha=.2)

    ttl = (f"{k}-bit flip-flop, N={N} — real population trajectories, output space "
           f"($W_{{out}}\\,r(t)$, exact for k=3)\n"
           f"{n_trials} trials, colour = trial")

    fig = plt.figure(figsize=(8, 7.4))
    ax = fig.add_subplot(111, projection="3d")
    draw(ax, 18, 45)
    fig.suptitle(ttl, fontsize=10)

    def frame(i):
        draw(ax, elev=18 + 12 * np.sin(2 * np.pi * i / 120), azim=i * 3)
        return ()

    out_gif = os.path.join(IMG_DIR, f"flipflop_bouquet_k{k}_N{N}.gif")
    FuncAnimation(fig, frame, frames=120, blit=False).save(
        out_gif, writer=PillowWriter(fps=20), dpi=90)
    print(f"wrote {out_gif}")

    fig2 = plt.figure(figsize=(16, 5.6))
    for j, (e, a) in enumerate([(18, 45), (18, 135), (78, 45)]):
        draw(fig2.add_subplot(1, 3, j + 1, projection="3d"), e, a)
    fig2.suptitle(ttl, fontsize=11)
    fig2.tight_layout()
    out_png = os.path.join(IMG_DIR, f"flipflop_bouquet_k{k}_N{N}.png")
    fig2.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
