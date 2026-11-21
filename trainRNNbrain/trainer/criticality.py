"""Criticality of a leaky ReLU RNN, measured on its actual dynamics, and the W_rec projections built on it.

THE UPDATE. With the h equation (configs/model/rnn_relu_standard.yaml) one step is
    h_{t+1} = h_t + alpha * (-h_t + W_rec relu(h_t) + W_inp u_t + noise),     alpha = dt / tau = 0.1,
so its Jacobian at the operating point h_t is
    J_t = (1 - alpha) I + alpha W_rec D_t,      D_t = diag(relu'(h_t)) = diag(1[h_t > 0]).
A unit below threshold zeroes its COLUMN of W_rec D_t (in the r equation, D_t W_rec, it is the row):
its perturbation reaches nobody, and in the transposed (backward) map the gradient arriving at it
only decays by (1 - alpha) per step. With every unit silent, J_t = (1 - alpha) I, whose exponent
log(1 - alpha) = -0.105 per step is the floor of every exponent below.

THE THREE MEASURES (entry docs/project_trajectory/*_rules-criticality-projection.md):
  lyapunov_max       the largest finite-time Lyapunov exponent along a trajectory: a tangent vector
                     pushed through J_t and renormalised every step; mean log growth per step after a
                     burn-in. 0 is the edge of chaos; < 0 contracting; > 0 chaotic.
  backward_gain      gradient-norm propagation through time: the read-out's sensitivity at the last
                     step, d y_o(T) / d h(T) = W_out[o] * relu'(h_T), pulled back k steps through J^T,
                     log10 of its norm relative to the start.
  operating_radius   spectral radius of J_t at sampled (time, trial) points. eig(W D) is eig of the
                     active block W[A, A] plus zeros, so only that block is diagonalised.

THE FOUR PROJECTIONS (Trainer `wrec_projection`, off by default) each map W_rec to a target and are
applied with a strength eta in (0, 1]: the multiplicative change is the eta-th power of the change
that would reach the target exactly, so eta = 1 lands on it.
"""

import math

import numpy as np
import torch


def relu_gates(states):
    """relu'(h) for a stack of hidden states: 1 where h > 0, else 0 (torch's relu gradient at 0 is 0).

    Args:
        states: tensor (N, T, B) of hidden states h.
    Returns:
        float tensor of the same shape with entries in {0, 1}.
    """
    return (states > 0).to(states.dtype)


def lyapunov_max(W, gates, alpha, burn_in=50, generator=None, lengths=None):
    """Largest finite-time Lyapunov exponent of the leaky update along given trajectories, per trial.

    A unit-norm tangent vector v is pushed through v <- (1 - alpha) v + alpha W (g_t * v) for every
    transition t -> t + 1 of the trajectory and renormalised after each step; the exponent is the mean
    log of the per-step growth from step `burn_in` on.

    Args:
        W: tensor (N, N), the recurrent matrix.
        gates: tensor (N, T, B), relu'(h_t) along each trajectory (relu_gates of the states).
        alpha: float, dt / tau.
        burn_in: int, transitions skipped before averaging (the start of a trial sits at y_init = 0,
            where every gate is 0).
        generator: torch.Generator for the initial tangent vectors, or None.
        lengths: None (every trial has T valid states), or int tensor (B,) of each trial's number of
            valid states L_b <= T (a maze episode's live states): only transitions t -> t + 1 with
            t + 1 < L_b are averaged, and a trial with no transition after the burn-in gives nan.
    Returns:
        tensor (B,) of exponents in units of 1 / step.
    """
    N, T, B = gates.shape
    v = torch.randn(N, B, generator=generator, device=W.device, dtype=W.dtype)
    v = v / v.norm(dim=0, keepdim=True)
    acc = torch.zeros(B, device=W.device, dtype=W.dtype)
    for t in range(T - 1):
        v = (1.0 - alpha) * v + alpha * (W @ (gates[:, t, :] * v))
        n = v.norm(dim=0)
        if t >= burn_in:
            step = torch.log(n)
            acc = acc + (step if lengths is None else torch.where(t + 1 < lengths, step, torch.zeros_like(step)))
        v = v / n
    if lengths is None:
        return acc / max(1, T - 1 - burn_in)
    count = (lengths.to(acc.device) - 1 - burn_in).to(acc.dtype)
    return torch.where(count >= 1, acc / count.clamp_min(1), torch.full_like(acc, float("nan")))


def backward_gain(W, gates, alpha, W_out, lags=(10, 50, 100, 200), lengths=None):
    """Gradient-norm propagation through time: how much of the read-out's sensitivity at the last
    step reaches the hidden state k steps earlier.

    Seeds g_T = W_out[o] * relu'(h_T) for every output o and trial, then g_t = J_t^T g_{t+1}
    = (1 - alpha) g_{t+1} + alpha g_t-gates * (W^T g_{t+1}). A seed that is all zero (no unit above
    threshold at the last step) gives nan.

    Args:
        W: tensor (N, N); gates: tensor (N, T, B); alpha: float; W_out: tensor (n_out, N);
        lags: iterable of int steps back from the last state, each < T.
        lengths: None (every trial's last state is T - 1), or int tensor (B,) of each trial's number
            of valid states L_b <= T: the seed sits at the trial's own last valid state L_b - 1 (a maze
            episode's last live state), and a lag k >= L_b gives nan for that trial.
    Returns:
        dict lag -> tensor (n_out * B,) of log10(||g_{last-lag}|| / ||g_{last}||), column o * B + b.
    """
    N, T, B = gates.shape
    reps = W_out.shape[0]
    cols = torch.arange(B, device=gates.device)
    last = torch.full((B,), T - 1, device=gates.device) if lengths is None else lengths.to(gates.device) - 1
    g = (W_out[:, :, None] * gates[:, last, cols][None]).permute(1, 0, 2).reshape(N, -1)  # (N, n_out*B)
    n0 = g.norm(dim=0)
    out = {}
    lags = sorted(int(k) for k in lags)
    for k in range(1, lags[-1] + 1):
        idx = (last - k).clamp_min(0)
        gt = gates[:, idx, cols].repeat(1, reps)            # column order matches g: (o, b) -> o*B + b
        g = (1.0 - alpha) * g + alpha * gt * (W.T @ g)
        if k in lags:
            val = torch.log10(g.norm(dim=0) / n0)          # n0 = 0 gives nan (0/0) or -inf; see caller
            if lengths is not None:
                val = torch.where((last - k >= 0).repeat(reps), val, torch.full_like(val, float("nan")))
            out[k] = val
    return out


def operating_radius(W, gates, alpha, points):
    """Spectral radius of the step Jacobian J = (1 - alpha) I + alpha W D at given operating points.

    Args:
        W: tensor (N, N); gates: tensor (N, T, B); alpha: float;
        points: iterable of (t, b) index pairs into the gates.
    Returns:
        numpy array of radii, one per point (1 - alpha where no unit is above threshold).
    """
    W64 = W.detach().double().cpu()
    out = []
    for t, b in points:
        A = gates[:, t, b].detach().cpu() > 0
        if not bool(A.any()):
            out.append(1.0 - alpha)
            continue
        mu = torch.linalg.eigvals(W64[A][:, A])
        out.append(float(torch.abs((1.0 - alpha) + alpha * mu).max()))
    return np.asarray(out)


def spectral_radius(W):
    """Largest eigenvalue modulus of a square matrix, computed in float64 on the CPU.

    Args:
        W: tensor (N, N).
    Returns:
        float.
    """
    return float(torch.linalg.eigvals(W.detach().double().cpu()).abs().max())


def project_spectral_radius(W, target, eta=1.0):
    """Variant (a): rescale W so that its spectral radius moves toward `target`.

    Args:
        W: tensor (N, N); target: float, the spectral radius to restore; eta: float in (0, 1].
    Returns:
        (new W tensor, radius before, factor applied).
    """
    rho = spectral_radius(W)
    factor = (target / max(rho, 1e-12)) ** eta
    return W * factor, rho, factor


def project_singular_values(W, s_target, eta=1.0):
    """Variant (b): W = U S V^T -> U S' V^T with S' = S^(1-eta) * s_target^eta, rank by rank.

    The singular vectors (what training learned about directions) are kept; the singular values are
    put back to the reference spectrum (eta = 1) or moved toward it geometrically.

    Args:
        W: tensor (N, N); s_target: tensor (N,), the reference singular values in descending order;
        eta: float in (0, 1].
    Returns:
        (new W tensor, largest singular value before, mean log ratio |log(s / s_target)| before).
    """
    U, S, Vh = torch.linalg.svd(W.double(), full_matrices=False)
    tgt = s_target.to(S)
    S_new = tgt if eta == 1.0 else S.clamp_min(1e-30) ** (1.0 - eta) * tgt ** eta
    dist = float(torch.log(S.clamp_min(1e-30) / tgt.clamp_min(1e-30)).abs().mean())
    return ((U * S_new) @ Vh).to(W.dtype), float(S[0]), dist


def project_row_norms(W, n_target, eta=1.0):
    """Variant (c): rescale each unit's incoming recurrent row W[i, :] toward its reference norm.

    A row of norm 0 (every incoming weight exactly zero) is left at zero: there is no direction to
    scale.

    Args:
        W: tensor (N, N); n_target: tensor (N,), reference row norms; eta: float in (0, 1].
    Returns:
        (new W tensor, mean row norm before, mean |log(norm / target)| before over nonzero rows).
    """
    n = W.norm(dim=1)
    ok = n > 0
    f = torch.ones_like(n)
    f[ok] = (n_target.to(n)[ok] / n[ok]) ** eta
    dist = float(torch.log(n[ok] / n_target.to(n)[ok]).abs().mean()) if bool(ok.any()) else float("nan")
    return W * f[:, None], float(n.mean()), dist


def lyapunov_gain_factor(lam, alpha, eta=1.0, max_step=0.05):
    """Variant (d): the scalar W_rec gain that moves the largest exponent `lam` toward 0.

    For the leaky map, lam ~ alpha (mu - 1) when the effective gain mu of W D along the trajectory is
    near 1, and multiplying W by c multiplies mu by c (with the gates held fixed). c = exp(-lam / alpha)
    is that Newton step; it is raised to eta and clipped to exp(+-max_step) per event so one bad
    estimate cannot move W far.

    Args:
        lam: float, measured exponent (1 / step); alpha: float; eta: float in (0, 1];
        max_step: float, largest |log c| per event.
    Returns:
        float factor c.
    """
    step = max(-max_step, min(max_step, -eta * lam / alpha))
    return math.exp(step)
