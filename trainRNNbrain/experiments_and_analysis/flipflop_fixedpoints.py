#!/usr/bin/env python3
"""
Fixed-point structure of the best 3-bit flip-flop network, animated.

A k-bit flip-flop is usually described as having 2^k stable states, one per corner of the hypercube
of remembered bit-signs (Sussillo & Barak 2013). THAT PREDICTION IS WRONG FOR THIS TASK AS DEFINED.
The target on a channel is 0 until that channel's FIRST pulse, so each bit rests at one of THREE
values, -1, 0 or +1, and the attractor set is a 3^k grid rather than a 2^k hypercube. Measured on the
best k=3 net: 21 distinct stable points, every one of them on the 3x3x3 lattice, 15 of the 21 having
at least one bit parked at 0. The 6 absent lattice sites are most likely unsampled rather than
missing - 220 starts, 143 converged.

This is the one task in the project where ground truth is known in advance, which is what makes it
worth looking at; the correction above is a reminder to derive that truth from the task definition
rather than from the task's name.

WHAT IS SOLVED. For equation_type "h" the flow is

    F(y) = -y + W_rec @ relu(y) + W_inp @ u + b        (RNN_numpy.rhs, noise off)

and a fixed point is F(y) = 0. Input is held at u = 0: the flip-flop's memory states are what the
network does BETWEEN pulses, so that is the condition whose fixed points constitute the memory.

SOLVED EXACTLY, NOT BY OPTIMISATION. ReLU dynamics are PIECEWISE LINEAR: on the region where the
activation pattern d = 1[y > 0] is fixed, relu(y) = d * y and the fixed-point condition becomes the
linear system

    (I - W_rec * d) y = b

So the solver iterates a pattern rather than descending a loss: guess d from a starting state, solve
the linear system, recompute d from the solution, repeat until d stops changing. A converged pattern
gives a fixed point exact to machine precision, and consistency is verifiable - the returned y must
satisfy 1[y > 0] == d, or the solution lies outside the region whose linearisation produced it.

This replaces a least_squares search that was intractable here: Levenberg-Marquardt on 1000
dimensions costs an O(N^3) factorisation per step, hundreds of steps per start, hundreds of starts.
The pattern iteration converges in a handful of solves per start and is exact rather than approximate.

CLASSIFICATION. The Jacobian of the flow is exactly

    J = -I + W_rec * 1[y > 0]        (column-scaled; relu' is an indicator)

so a point is stable when every eigenvalue has negative real part. The count of eigenvalues with
positive real part is the saddle's unstable dimension, which is what distinguishes "1-saddle between
two neighbouring corners" from higher-index points.

PROJECTION. No PCA needed: the readout W_out is (k, N), so W_out @ relu(y*) puts every fixed point
directly into the 3-dimensional output space the task is defined in. The 2^k = 8 stable states should
land on a 3x3x3 lattice, which is a prediction that can be checked rather than a layout choice.

Output: img/internal_figures/flipflop_fixedpoints.gif  (+ a static .png of three viewing angles)

Usage:  python flipflop_fixedpoints.py [k] [N]
"""

import os
import sys
import glob
import numpy as np
import hydra
from omegaconf import OmegaConf
from scipy.linalg import eigvals, solve
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from plot_drift_curves import IMG_DIR
from trainRNNbrain.rnns.RNN_numpy import RNN_numpy
from trainRNNbrain.training.training_utils import prepare_task_arguments
from trainRNNbrain.utils import filter_kwargs

SWEEP = "data/trained_RNNs/NBitFlipFlop_std_ksweep"
N_STARTS = 220         # initial guesses drawn off trajectories
TOL_DUP = 1e-2         # relative distance below which two solutions are the same point
TOL_FP = 1e-6          # |F| below which a solution counts as a fixed point rather than a slow point


def best_net(k, N):
    """Folder of the lowest-loss network for one (k, N) cell.

    Args:
        k: number of bits; N: network size.
    Returns:
        path to the net folder, chosen by the R^2 encoded in the folder name (the training script
        prefixes each net folder with its final score).
    """
    pat = os.path.join(SWEEP, f"EqType=h_k={k}_N={N}_iters=*", "*")
    cands = [d for d in glob.glob(pat) if os.path.isdir(d)]
    if not cands:
        raise SystemExit(f"no networks under {pat}")
    return max(cands, key=lambda d: float(os.path.basename(d).split("_")[0]))


def load_net(folder):
    """Instantiate RNN_numpy from a saved parameter file.

    Args:
        folder: net folder holding `*LastParams*.npz`.
    Returns:
        (rnn, params dict).
    """
    d = np.load(glob.glob(os.path.join(folder, "*LastParams*.npz"))[0], allow_pickle=True)
    p = {kk: d[kk] for kk in d.files}
    p["activation_name"] = "relu"
    p.pop("activation_args", None)
    return RNN_numpy(**filter_kwargs(RNN_numpy, p), seed=0), p


def trajectory_states(rnn, folder, k):
    """States visited on real trials, used both as fixed-point seeds and as plotted trajectories.

    Args:
        rnn: the network; folder: net folder (for its config); k: number of bits.
    Returns:
        (states, outputs) with states (N, T, B) and outputs (k, T, B).
    """
    cfg = OmegaConf.load(glob.glob(os.path.join(folder, "*_config.yaml"))[0])
    cfg.task.batch_size = 24
    task = hydra.utils.instantiate(prepare_task_arguments(cfg_task=cfg.task, dt=cfg.model.dt))
    inputs, _, _ = task.get_batch()
    rnn.clear_history()
    rnn.y = rnn.y_init
    rnn.run(input_timeseries=inputs, sigma_rec=0.0, sigma_inp=0.0)
    return np.array(rnn.get_history()), np.array(rnn.get_output())


def find_fixed_points(W_rec, b, y0s, max_pattern_iters=60):
    """Exact fixed points by iterating the ReLU activation pattern to consistency.

    On the region where d = 1[y > 0] is constant the flow is linear, so the fixed point there solves
    (I - W_rec * d) y = b. Iterate: pattern -> linear solve -> new pattern, until the pattern repeats.

    Args:
        W_rec: (N, N) recurrent weights; b: (N,) bias; y0s: (n_starts, N) initial states;
        max_pattern_iters: cap, since the pattern map can cycle rather than converge.
    Returns:
        (points, residuals) - distinct solutions verified to satisfy F(y) = 0, and their |F|.
    """
    N = W_rec.shape[0]
    eye = np.eye(N)
    found, res, n_cycle = [], [], 0

    for y0 in y0s:
        d = (y0 > 0).astype(float)
        seen = set()
        y = None
        for _ in range(max_pattern_iters):
            key = d.tobytes()
            if key in seen:            # pattern cycles: no fixed point in this basin
                y = None
                break
            seen.add(key)
            try:
                y = solve(eye - W_rec * d[None, :], b, assume_a="gen")
            except np.linalg.LinAlgError:
                y = None
                break
            d_new = (y > 0).astype(float)
            if np.array_equal(d_new, d):
                break
            d = d_new
        else:
            y = None
        if y is None:
            n_cycle += 1
            continue
        r = float(np.linalg.norm(-y + W_rec @ np.maximum(y, 0.0) + b))
        if r > TOL_FP:
            continue
        scale = max(np.linalg.norm(y), 1e-9)
        if any(np.linalg.norm(y - f) / scale < TOL_DUP for f in found):
            continue
        found.append(y)
        res.append(r)
    print(f"  {n_cycle}/{len(y0s)} starts gave no consistent pattern (cycled or singular)")
    return np.array(found), np.array(res)


def classify(W_rec, pts):
    """Unstable dimension of each fixed point, from the exact Jacobian, on the ACTIVE block only.

    J = -I + W_rec * d with d = 1[y > 0]. For an INACTIVE unit j the whole column is just -e_j, so
    ordering active units first makes J block lower-triangular:

        J = [[-I + W_AA, 0],
             [W_IA,     -I]]

    Its spectrum is therefore exactly eig(-I + W_AA) together with -1 repeated once per inactive
    unit. Those -1s are trivially stable and carry no information, so only the active block needs
    decomposing. With ~200 of 1000 units active that is a 200x200 problem instead of 1000x1000 -
    around 125x less work at O(n^3), and it is an identity, not an approximation.

    Args:
        W_rec: (N, N); pts: (n_fp, N) fixed points.
    Returns:
        (n_unstable, max_real_eig) arrays - 0 unstable directions means a stable attractor.
    """
    nun, mx = [], []
    for y in pts:
        a = np.flatnonzero(y > 0)
        if a.size == 0:                      # every unit off: J = -I, unconditionally stable
            nun.append(0)
            mx.append(-1.0)
            continue
        ev = eigvals(-np.eye(a.size) + W_rec[np.ix_(a, a)])
        nun.append(int((ev.real > 1e-9).sum()))
        # -1 from the inactive block can only lower the max, so the active block decides it.
        mx.append(float(max(ev.real.max(), -1.0)))
    return np.array(nun), np.array(mx)


def main():
    """Find, classify, and animate the fixed points of the best network of one cell."""
    k = int(sys.argv[1]) if len(sys.argv) > 1 else 3
    N = int(sys.argv[2]) if len(sys.argv) > 2 else 1000

    folder = best_net(k, N)
    print(f"network: {os.path.basename(folder)[:60]}")
    rnn, p = load_net(folder)
    W_rec, b, W_out = p["W_rec"], p["bias"], p["W_out"]

    states, outs = trajectory_states(rnn, folder, k)
    print(f"trajectories: states {states.shape}, outputs {outs.shape}")

    # Seeds: real visited states plus jitter, so the search covers what the network actually uses.
    rng = np.random.default_rng(0)
    flat = states.reshape(states.shape[0], -1).T
    idx = rng.choice(len(flat), size=N_STARTS, replace=False)
    y0s = flat[idx] + 0.15 * rng.standard_normal((N_STARTS, states.shape[0]))

    pts, res = find_fixed_points(W_rec, b, y0s)
    print(f"fixed points: {len(pts)} distinct (from {N_STARTS} starts), "
          f"|F| max {res.max():.2e}" if len(pts) else "no fixed points found")
    if not len(pts):
        return
    nun, mx = classify(W_rec, pts)
    print(f"  stable (0 unstable dims): {int((nun == 0).sum())}   "
          f"saddles: {int((nun > 0).sum())}   "
          f"unstable-dim counts: {dict(zip(*np.unique(nun, return_counts=True)))}")
    print(f"  lattice prediction for k={k}: 3^{k} = {3 ** k} stable states "
          f"(each bit rests at -1, 0 or +1; the usual 2^k counts only the two pulsed signs)")

    proj = (W_out @ np.maximum(pts, 0.0).T).T            # (n_fp, k) in output space
    traj = outs.reshape(outs.shape[0], -1)               # (k, T*B)

    for i, (nu, m) in enumerate(zip(nun, mx)):
        tag = "STABLE" if nu == 0 else f"saddle({nu})"
        print(f"    {tag:>12}  max Re(lambda) = {m:+.4f}   out = "
              + " ".join(f"{v:+.2f}" for v in proj[i][:3]))

    if k < 3:
        print("k < 3: output space is not 3-D, skipping the animation")
        return

    stable, saddle = nun == 0, nun > 0

    def draw(ax, elev, azim):
        """Render one view of the output-space fixed-point structure."""
        ax.clear()
        ax.plot(traj[0, ::3], traj[1, ::3], traj[2, ::3], ".", color="0.6", ms=.6, alpha=.35,
                zorder=1)
        if saddle.any():
            s = ax.scatter(*proj[saddle, :3].T, c=nun[saddle], cmap="autumn", s=55, marker="^",
                           edgecolors="k", linewidths=.5, depthshade=False, zorder=3)
            s.set_clim(1, max(2, nun.max()))
        if stable.any():
            ax.scatter(*proj[stable, :3].T, c="tab:blue", s=170, marker="o", edgecolors="k",
                       linewidths=1.0, depthshade=False, zorder=4)
        ax.set(xlabel="out 1", ylabel="out 2", zlabel="out 3")
        ax.view_init(elev=elev, azim=azim)
        ax.grid(alpha=.2)

    fig = plt.figure(figsize=(7.5, 7))
    ax = fig.add_subplot(111, projection="3d")
    ttl = (f"{k}-bit flip-flop, N={N} — fixed points with input held at 0\n"
           f"{int(stable.sum())} stable (blue) of $3^{k}$={3 ** k} lattice sites — each bit rests at "
           f"$-1,0,+1$, not just $\\pm1$\n"
           f"{int(saddle.sum())} saddles (triangles, colour = unstable dimension)")
    fig.suptitle(ttl, fontsize=10)

    def frame(i):
        draw(ax, elev=18 + 12 * np.sin(2 * np.pi * i / 120), azim=i * 3)
        return ()

    out_gif = os.path.join(IMG_DIR, f"flipflop_fixedpoints_k{k}_N{N}.gif")
    FuncAnimation(fig, frame, frames=120, blit=False).save(
        out_gif, writer=PillowWriter(fps=20), dpi=90)
    print(f"wrote {out_gif}")

    fig2 = plt.figure(figsize=(16, 5.5))
    for j, (e, a) in enumerate([(20, 45), (20, 135), (75, 45)]):
        draw(fig2.add_subplot(1, 3, j + 1, projection="3d"), e, a)
    fig2.suptitle(ttl, fontsize=11)
    fig2.tight_layout()
    out_png = os.path.join(IMG_DIR, f"flipflop_fixedpoints_k{k}_N{N}.png")
    fig2.savefig(out_png, dpi=150)
    print(f"wrote {out_png}")


if __name__ == "__main__":
    main()
