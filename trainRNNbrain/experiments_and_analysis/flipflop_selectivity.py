#!/usr/bin/env python3
"""
Selectivity configuration of a k-bit flip-flop network, animated.

Companion to flipflop_fixedpoints.py. That one asks where the network's MEMORY STATES sit in the
3-dimensional output space; this asks where its UNITS sit in the 3-dimensional selectivity space.
Both are 3-D for k=3 without any dimensionality reduction, so neither picture involves a projection
choice that could be doing the work.

SELECTIVITY, DEFINED. The bits are generated independently, so the design is near-orthogonal and a
single linear regression per unit recovers each bit's contribution cleanly:

    r_i(t, trial)  ~  beta_i0 + sum_j beta_ij * b_j(t, trial)

with b_j the TARGET value of bit j (-1, 0 or +1) at that moment - i.e. what the network is supposed
to be remembering, not what it was just shown. beta_i = (beta_i1, ..., beta_ik) is unit i's
selectivity vector, in units of firing rate per unit of remembered bit. Fitted for all units at once
by one least-squares solve.

WHAT THE GEOMETRY MEANS.
    at the origin          the unit carries no bit information - where silent units must land
    along an axis          pure selectivity: the unit encodes one bit and ignores the others
    off-axis, in a corner  mixed selectivity: the unit encodes a combination

The silent/active split is shown by colour, using the same two criteria as everywhere else in the
project, because "how many units carry task information" and "how many units fire" are different
questions and the whole point of this project is that they come apart.

TRAJECTORIES, AND WHAT SHARING AXES WITH THEM DOES AND DOES NOT MEAN. Units sit at ENCODING weights;
a trajectory is a STATE moving in time. They are different quantities and only the axis LABELS (bit
1, 2, 3) are shared. The trajectory is placed by the OPTIMAL LINEAR DECODE of each bit from
population activity - w_j = argmin |r^T w - b_j|, which is the decoder, not the transpose of the
encoder - so its natural units are remembered-bit values in [-1, +1]. It is then rescaled by the
median |beta| of active units so the two occupy a common scale; the factor is printed and stated on
the figure. Nothing about the SHAPE of either is changed by that rescaling, but do not read a
trajectory passing through a unit as the trajectory "visiting" that unit.

Output: img/internal_figures/flipflop_selectivity_k<k>_N<N>.gif  (+ a static .png)

Usage:  python flipflop_selectivity.py [k] [N]
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
from common import IMG_DIR, active_count, participation
from flipflop_fixedpoints import best_net, load_net
from trainRNNbrain.training.training_utils import prepare_task_arguments


def rates_and_targets(rnn, folder, batch=64):
    """Noise-free rates and the target bit values that generated them.

    Args:
        rnn: the network; folder: net folder (for its config); batch: trials to run.
    Returns:
        (r, b) with r (N, T*B) firing rates and b (k, T*B) target bit values in {-1, 0, +1}.
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    cfg.task.batch_size = batch
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, targets, _ = task.get_batch()
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    states = np.array(rnn.get_history())                       # (N, T, B) pre-activation
    r = np.maximum(states, 0.0).reshape(states.shape[0], -1)   # relu -> firing rate
    b = targets.reshape(targets.shape[0], -1)
    return r, b


def selectivity(r, b):
    """Per-unit regression coefficients of rate on the remembered bits.

    One least-squares solve for all units: the design matrix is shared, so this is a single
    (T*B, k+1) \\ (T*B, N) solve rather than N separate regressions.

    Args:
        r: (N, S) firing rates; b: (k, S) target bit values.
    Returns:
        (beta, r2) - (N, k) selectivity vectors and the per-unit variance explained.
    """
    X = np.column_stack([np.ones(b.shape[1]), b.T])            # (S, k+1)
    coef, *_ = np.linalg.lstsq(X, r.T, rcond=None)             # (k+1, N)
    resid = r.T - X @ coef
    ss_res = (resid ** 2).sum(axis=0)
    ss_tot = ((r.T - r.T.mean(axis=0)) ** 2).sum(axis=0)
    return coef[1:].T, 1 - ss_res / np.maximum(ss_tot, 1e-12)


def decode(r, b):
    """Optimal linear decoder of each bit from population rates, and the decoded trajectory.

    The decoder is not the encoder's transpose: beta_j is how strongly unit i responds to bit j,
    whereas w_j is the readout that best reconstructs bit j from all units at once. Correlated units
    make the two differ substantially, so the decode is solved for explicitly.

    Args:
        r: (N, S) firing rates; b: (k, S) target bit values.
    Returns:
        (traj, r2) - (k, S) decoded bit values and the per-bit variance explained.
    """
    X = np.column_stack([np.ones(r.shape[1]), r.T])            # (S, N+1)
    w, *_ = np.linalg.lstsq(X, b.T, rcond=None)                # (N+1, k)
    traj = (X @ w).T                                           # (k, S)
    ss_res = ((b - traj) ** 2).sum(axis=1)
    ss_tot = ((b - b.mean(axis=1, keepdims=True)) ** 2).sum(axis=1)
    return traj, 1 - ss_res / np.maximum(ss_tot, 1e-12)


def main():
    """Fit and animate the selectivity configuration of the best network of one cell."""
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
    if k != 3:
        raise SystemExit("selectivity space is 3-D only for k=3; pass k=3")

    folder = best_net(k, N)
    print(f"network: {os.path.basename(folder)[:60]}")
    rnn, _ = load_net(folder)
    r, b = rates_and_targets(rnn, folder)
    print(f"rates {r.shape}, targets {b.shape}")

    beta, r2 = selectivity(r, b)
    p = participation(r)
    hard = p < 1e-6
    sf = p < 0.05 * np.quantile(p, 0.95)
    norm = np.linalg.norm(beta, axis=1)

    print(f"  silent: hard {100 * hard.mean():.1f}%   scale-free {100 * sf.mean():.1f}%")
    print(f"  |beta| : silent(sf) median {np.median(norm[sf]):.4f}   "
          f"active median {np.median(norm[~sf]):.4f}")
    print(f"  variance explained by the 3 bits, active units: median {np.median(r2[~sf]):.3f}")
    # How concentrated is each active unit's selectivity on ONE bit? 1 = pure, 1/sqrt(k) = uniform.
    conc = np.max(np.abs(beta), axis=1) / np.maximum(norm, 1e-12)
    print(f"  selectivity concentration (max|beta|/|beta|), active: median {np.median(conc[~sf]):.3f}"
          f"   (1.000 = pure single-bit, {1 / np.sqrt(k):.3f} = equal across all {k})")
    for j in range(k):
        dom = (np.argmax(np.abs(beta), axis=1) == j) & ~sf
        print(f"    bit {j + 1} dominant in {int(dom.sum())} active units")

    traj, dec_r2 = decode(r, b)
    print(f"  linear decode of each bit from the population: R^2 = "
          + ", ".join(f"{v:.3f}" for v in dec_r2))
    # Rescale the decoded trajectory (natural range [-1, +1]) onto the unit-cloud scale so both are
    # visible together. Scale to the cloud's EXTENT, not its median: a decoded bit of +-1 should
    # reach the outer shell of the star, otherwise the trajectory collapses into the origin cluster
    # and the memory states it visits are unreadable. Shape is untouched either way.
    scale = float(np.percentile(norm, 99.5))
    # 3 trials, not 6: the decoded state jumps between memory corners on every pulse, so each trial
    # contributes ~300 long segments and the orange swamps the unit cloud well before it adds
    # information. Three is enough to read the corner-to-corner structure.
    n_show = 3
    S_per = traj.shape[1] // 64                                # samples per trial
    tr = traj[:, :n_show * S_per].reshape(3, n_show, S_per) * scale
    print(f"  trajectory rescaled by the 99.5th pct of |beta| = {scale:.4f}; "
          f"showing {n_show} trials")

    lim = float(np.percentile(norm, 99.5))
    act = ~sf

    def draw(ax, elev, azim):
        """Render one view of the selectivity cloud."""
        ax.clear()
        for a in (-1, 1):
            ax.plot([-lim * a, lim * a], [0, 0], [0, 0], color="0.8", lw=.8, zorder=0)
            ax.plot([0, 0], [-lim * a, lim * a], [0, 0], color="0.8", lw=.8, zorder=0)
            ax.plot([0, 0], [0, 0], [-lim * a, lim * a], color="0.8", lw=.8, zorder=0)
        for t in range(tr.shape[1]):
            ax.plot(tr[0, t], tr[1, t], tr[2, t], "-", color="tab:orange", lw=0.8, alpha=.28,
                    zorder=1)
        ax.plot([], [], [], "-", color="tab:orange", lw=1.5, alpha=.8,
                label=f"decoded state, {tr.shape[1]} trials")
        ax.scatter(*beta[sf].T, c="0.55", s=9, alpha=.45, depthshade=False, zorder=2,
                   label=f"silent, scale-free ({int(sf.sum())})")
        sc = ax.scatter(*beta[act].T, c=conc[act], cmap="viridis", s=26, alpha=.9,
                        depthshade=False, zorder=3, vmin=1 / np.sqrt(k), vmax=1.0,
                        label=f"active ({int(act.sum())})")
        ax.set(xlabel=r"$\beta$ bit 1", ylabel=r"$\beta$ bit 2", zlabel=r"$\beta$ bit 3",
               xlim=(-lim, lim), ylim=(-lim, lim), zlim=(-lim, lim))
        ax.view_init(elev=elev, azim=azim)
        ax.grid(alpha=.2)
        return sc

    ttl = (f"{k}-bit flip-flop, N={N} — selectivity configuration with decoded trajectories\n"
           f"points: units at their encoding weights, colour = concentration on one bit.  "
           f"orange: population state, optimally decoded\n"
           f"{int(act.sum())} active, {int(sf.sum())} scale-free silent (grey, at the origin).  "
           f"decoded bit $\\pm1$ maps to the cloud's outer shell (x{scale:.3f})")

    fig = plt.figure(figsize=(7.8, 7.2))
    ax = fig.add_subplot(111, projection="3d")
    sc = draw(ax, 18, 45)
    fig.colorbar(sc, ax=ax, shrink=.6, pad=.10, label="max|$\\beta$| / |$\\beta$|  (1 = pure)")
    ax.legend(fontsize=8, loc="upper left")
    fig.suptitle(ttl, fontsize=10)

    def frame(i):
        draw(ax, elev=18 + 12 * np.sin(2 * np.pi * i / 120), azim=i * 3)
        return ()

    out_gif = os.path.join(IMG_DIR, f"flipflop_selectivity_traj_k{k}_N{N}.gif")
    FuncAnimation(fig, frame, frames=120, blit=False).save(
        out_gif, writer=PillowWriter(fps=20), dpi=90)
    print(f"wrote {out_gif}")

    fig2 = plt.figure(figsize=(16, 5.5))
    for j, (e, a) in enumerate([(18, 45), (18, 135), (78, 45)]):
        draw(fig2.add_subplot(1, 3, j + 1, projection="3d"), e, a)
    fig2.suptitle(ttl, fontsize=11)
    fig2.tight_layout()
    out_png = os.path.join(IMG_DIR, f"flipflop_selectivity_traj_k{k}_N{N}.png")
    fig2.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
