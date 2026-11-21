"""The criticality measures and the four W_rec projections (trainRNNbrain/trainer/criticality.py).

Known answers for the measures: with W = c I and every gate on, the step Jacobian is (1 - a + a c) I,
so the Lyapunov exponent is log(1 - a + a c), the gradient pulled back k steps shrinks by
(1 - a + a c)^k, and the operating radius is |1 - a + a c|; with every gate off it is the leak alone,
log(1 - a). The projections are checked with numpy (eigvals, svd, norms), not with the torch calls they
use, and through a real Trainer step on a small network.

Run:  python tests/test_criticality.py   (or under pytest)
"""
import inspect
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer import criticality as C
from trainRNNbrain.trainer.Trainer import Trainer

A = 0.1


def test_measures_known_answers():
    """Scaled identity, all gates on or off: exponent, backward gain and radius have closed forms."""
    N, T, B, c = 20, 120, 3, 1.3
    W = c * torch.eye(N, dtype=torch.float64)
    on = torch.ones(N, T, B, dtype=torch.float64)
    m = 1 - A + A * c
    lam = C.lyapunov_max(W, on, A, burn_in=10, generator=torch.Generator().manual_seed(0))
    assert torch.allclose(lam, torch.full((B,), np.log(m), dtype=torch.float64)), lam
    lam0 = C.lyapunov_max(W, torch.zeros_like(on), A, burn_in=10)
    assert torch.allclose(lam0, torch.full((B,), np.log(1 - A), dtype=torch.float64)), lam0
    W_out = torch.randn(2, N, dtype=torch.float64)
    g = C.backward_gain(W, on, A, W_out, lags=(5, 50))
    for k in (5, 50):
        assert torch.allclose(g[k], torch.full_like(g[k], k * np.log10(m))), (k, g[k])
    r = C.operating_radius(W, on, A, [(3, 0), (50, 2)])
    assert np.allclose(r, m), r


def test_backward_gain_matches_autograd():
    """backward_gain equals the norm of d y_o(T) / d h(T - k) computed by autograd through the update."""
    N, T, B = 15, 40, 2
    g0 = torch.Generator().manual_seed(1)
    W = torch.randn(N, N, generator=g0, dtype=torch.float64) * 1.5 / np.sqrt(N)
    W_out = torch.randn(2, N, generator=g0, dtype=torch.float64)
    u = torch.randn(N, T, B, generator=g0, dtype=torch.float64)
    hs = [torch.zeros(N, B, dtype=torch.float64, requires_grad=True)]
    for t in range(T - 1):
        hs.append(hs[-1] + A * (-hs[-1] + W @ torch.relu(hs[-1]) + u[:, t]))
    for h in hs:
        h.retain_grad()
    gates = C.relu_gates(torch.stack(hs, 1).detach())
    k = 25
    got = C.backward_gain(W, gates, A, W_out, lags=(k,))[k]
    y = (W_out[0] @ torch.relu(hs[-1]))[1]          # output 0, trial 1
    y.backward()
    want = torch.log10(hs[T - 1 - k].grad[:, 1].norm() / (W_out[0] * gates[:, T - 1, 1]).norm())
    assert torch.allclose(got[0 * B + 1], want), (got, want)


def test_lengths_option():
    """The maze option `lengths` (each trial's valid states): identical results when every length is T;
    an episode cut to L gives the unmasked result on its first L states; states after L are ignored."""
    N, T, B = 15, 90, 3
    g0 = torch.Generator().manual_seed(3)
    W = torch.randn(N, N, generator=g0, dtype=torch.float64) * 1.5 / np.sqrt(N)
    W_out = torch.randn(2, N, generator=g0, dtype=torch.float64)
    gates = (torch.rand(N, T, B, generator=g0, dtype=torch.float64) > 0.5).to(torch.float64)
    full = torch.full((B,), T)
    gen = lambda: torch.Generator().manual_seed(0)
    assert torch.equal(C.lyapunov_max(W, gates, A, 10, gen()), C.lyapunov_max(W, gates, A, 10, gen(), lengths=full))
    a, b = C.backward_gain(W, gates, A, W_out, (5, 40)), C.backward_gain(W, gates, A, W_out, (5, 40), lengths=full)
    assert all(torch.equal(a[k], b[k]) for k in (5, 40))
    # the same length L for every trial: the same initial tangent vectors as the call on gates[:, :L]
    L = 60
    cut = torch.full((B,), L)
    assert torch.allclose(C.lyapunov_max(W, gates, A, 10, gen(), lengths=cut),
                          C.lyapunov_max(W, gates[:, :L], A, 10, gen()))
    # different lengths: each trial's backward gain equals the unmasked call on its own first L_b states
    lens = torch.tensor([90, 50, 30])
    got = C.backward_gain(W, gates, A, W_out, (5, 40), lengths=lens)
    for bb, Lb in enumerate(lens.tolist()):
        want = C.backward_gain(W, gates[:, :Lb, bb:bb + 1], A, W_out, (5, 29))
        assert torch.allclose(got[5][[bb, B + bb]], want[5])
        if Lb > 40:
            assert torch.allclose(got[40][[bb, B + bb]], C.backward_gain(W, gates[:, :Lb, bb:bb + 1], A, W_out, (40,))[40])
        else:
            assert torch.isnan(got[40][[bb, B + bb]]).all()
    # known answer with garbage after L: W = c I, gates on before L, off after
    c, Lk = 1.3, 70
    on = torch.zeros(N, T, B, dtype=torch.float64)
    on[:, :Lk] = 1.0
    m = 1 - A + A * c
    lam = C.lyapunov_max(c * torch.eye(N, dtype=torch.float64), on, A, 10, gen(), lengths=torch.full((B,), Lk))
    assert torch.allclose(lam, torch.full((B,), np.log(m), dtype=torch.float64)), lam
    short = C.lyapunov_max(W, gates, A, 10, gen(), lengths=torch.tensor([90, 11, 5]))
    assert torch.isfinite(short[0]) and torch.isnan(short[1:]).all()


def test_projections_hit_their_targets():
    """Each projection lands on its target at eta = 1, checked with numpy; eta = 0.5 goes halfway (in log)."""
    N = 60
    g0 = torch.Generator().manual_seed(2)
    W = torch.randn(N, N, generator=g0, dtype=torch.float64) * 2.0 / np.sqrt(N)
    Wa, rho, f = C.project_spectral_radius(W, 1.2)
    assert abs(np.abs(np.linalg.eigvals(Wa.numpy())).max() - 1.2) < 1e-9
    Wh, _, _ = C.project_spectral_radius(W, 1.2, eta=0.5)
    assert abs(np.abs(np.linalg.eigvals(Wh.numpy())).max() - np.sqrt(1.2 * rho)) < 1e-9

    s0 = torch.linspace(3.0, 0.1, N, dtype=torch.float64)
    Wb, _, _ = C.project_singular_values(W, s0)
    U, S, Vt = np.linalg.svd(W.numpy())
    assert np.allclose(np.linalg.svd(Wb.numpy(), compute_uv=False), s0.numpy())
    # the singular vectors are kept: V^T Wb^T Wb V is diagonal with s0^2 on it
    M = Vt @ Wb.numpy().T @ Wb.numpy() @ Vt.T
    assert np.allclose(M, np.diag(s0.numpy() ** 2), atol=1e-9)

    n0 = torch.linspace(0.5, 2.0, N, dtype=torch.float64)
    W[7] = 0.0
    Wc, _, _ = C.project_row_norms(W, n0)
    norms = np.linalg.norm(Wc.numpy(), axis=1)
    keep = np.arange(N) != 7
    assert np.allclose(norms[keep], n0.numpy()[keep]) and norms[7] == 0.0
    # direction of every row unchanged
    cos = (Wc.numpy()[keep] * W.numpy()[keep]).sum(1) / (norms[keep] * np.linalg.norm(W.numpy()[keep], axis=1))
    assert np.allclose(cos, 1.0)


def test_lyapunov_gain_factor_sign_and_clip():
    """A contracting network (lam < 0) is scaled up, a chaotic one down, never more than exp(0.05)."""
    assert C.lyapunov_gain_factor(-0.002, A) > 1.0
    assert C.lyapunov_gain_factor(0.002, A) < 1.0
    assert np.isclose(C.lyapunov_gain_factor(-0.5, A), np.exp(0.05))
    assert np.isclose(C.lyapunov_gain_factor(0.5, A), np.exp(-0.05))
    assert np.isclose(C.lyapunov_gain_factor(-0.002, A), np.exp(0.02))


def _trainer(kind, period=2):
    """A Trainer stand-in with a small ReLU RNN, exposing what wrec_projection_ touches.

    Args:
        kind: one of Trainer.PROJECTIONS; period: iterations between events.
    Returns:
        (rnn, tr): the network and the stand-in, with references read from the initial W_rec.
    """
    rnn = RNN_torch(N=40, activation_args={"name": "relu", "slope": 1.0}, dale=False, n_inputs=3,
                    n_outputs=2, equation_type="h", seed=3, spectral_rad=1.2, self_connections=True,
                    init_draws="independent")
    tr = types.SimpleNamespace(RNN=rnn, iter_n=0, wrec_projection=kind, _proj_last=None,
                               wrec_projection_args={"period": period, "eta": 1.0, "lyap_trials": 8,
                                                     "lyap_burn_in": 5, "lyap_max_step": 0.05},
                               _proj_s0=torch.linalg.svdvals(rnn.W_rec.detach().double()),
                               _proj_n0=rnn.W_rec.detach().norm(dim=1).clone(),
                               _proj_gen=torch.Generator().manual_seed(0))
    return rnn, tr


def test_trainer_event_restores_the_reference():
    """Through Trainer.wrec_projection_: after a perturbation of W_rec, an event puts each target back,
    and an iteration that is not a multiple of the period changes nothing."""
    u = torch.randn(3, 30, 8, generator=torch.Generator().manual_seed(4))
    for kind in ("spectral", "singular", "rownorm"):
        rnn, tr = _trainer(kind)
        s0, n0 = tr._proj_s0.numpy(), tr._proj_n0.numpy()
        with torch.no_grad():
            rnn.W_rec.mul_(1.7).add_(0.05 * torch.randn(40, 40, generator=torch.Generator().manual_seed(5)))
        before = rnn.W_rec.detach().clone()
        tr.iter_n = 0                                   # (0 + 1) % 2 != 0: no event
        Trainer.wrec_projection_(tr, u)
        assert torch.equal(rnn.W_rec.detach(), before), kind
        tr.iter_n = 1                                   # event
        Trainer.wrec_projection_(tr, u)
        W = rnn.W_rec.detach().double().numpy()
        if kind == "spectral":
            assert abs(np.abs(np.linalg.eigvals(W)).max() - 1.2) < 1e-4
        elif kind == "singular":
            assert np.allclose(np.linalg.svd(W, compute_uv=False), s0, rtol=1e-4, atol=1e-6)
        else:
            assert np.allclose(np.linalg.norm(W, axis=1), n0, rtol=1e-5)
    rnn, tr = _trainer("lyapunov")
    before = rnn.W_rec.detach().clone()
    tr.iter_n = 1
    Trainer.wrec_projection_(tr, u)
    lam, factor = tr._proj_last
    assert np.isfinite(lam) and np.exp(-0.05) - 1e-9 <= factor <= np.exp(0.05) + 1e-9
    assert torch.allclose(rnn.W_rec.detach(), before * factor)


def test_off_by_default():
    """The Trainer's default is off."""
    sig = inspect.signature(Trainer.__init__)
    assert sig.parameters["wrec_projection"].default is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
