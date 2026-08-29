"""Why the frm gradient-spike defences are shaped the way they are.

Pins three claims that the fix rests on. If any assertion fails, the corresponding design
decision in Trainer.train_step / Penalties.fr_magnitude_penalty is no longer justified.
"""
import math
import types
import torch

from trainRNNbrain.trainer.Trainer import Penalties

CAP = 0.3 * math.log1p(100) / math.log1p(2000)      # N=2000 cap, as configured
ARGS = dict(cap_fr=0.3, tau=0.1, g_top=3.0, g_bot=3.0, alpha=1.0, beta=1.0)


def test_clip_converts_inf_into_nan():
    """clip_grad_norm_ does not guard non-finite grads: total_norm=inf -> coef=0 -> inf*0=nan."""
    W1 = torch.nn.Parameter(torch.ones(3))
    W2 = torch.nn.Parameter(torch.ones(3))
    opt = torch.optim.Adam([W1, W2], lr=1e-3)
    W1.grad, W2.grad = torch.ones(3), torch.full((3,), float("inf"))
    torch.nn.utils.clip_grad_norm_([W1, W2], max_norm=50.0)
    opt.step()
    assert torch.isfinite(W1).all(), "clean param should survive (its grad is scaled to 0)"
    assert not torch.isfinite(W2).all(), "the overflowing param must go NaN — else the guard is moot"


def _displacement(max_norm, spike=1e27, normal=1e-3, lr=3.7e-4, tail=300):
    """Net displacement from one spike, in units of a normal Adam step."""
    def run(with_spike):
        p = torch.nn.Parameter(torch.zeros(1))
        opt = torch.optim.Adam([p], lr=lr)
        def step(g):
            p.grad = torch.tensor([g])
            torch.nn.utils.clip_grad_norm_([p], max_norm=max_norm)
            opt.step()
        for _ in range(200):
            step(normal)
        anchor = p.detach().clone()
        step(spike if with_spike else normal)
        for _ in range(tail):
            step(normal)
        return (p.detach() - anchor).item()
    return (run(True) - run(False)) / lr


def test_clipping_barely_limits_a_spike_because_adam_is_scale_invariant():
    """A 500x stricter clip must NOT meaningfully reduce the damage — that is why we SKIP."""
    loose, tight = _displacement(50.0), _displacement(0.1)
    assert loose > 100, f"expected a large displacement at max_norm=50, got {loose:.1f}"
    assert tight > 0.5 * loose, (
        f"tightening the clip 500x cut damage from {loose:.0f} to {tight:.0f} steps — if clipping "
        "really were this effective, skipping would be unnecessary")


def test_penalty_is_finite_where_float32_would_overflow():
    """fp64 powers: the fp32 build returned inf for over/cap ~1e13 (reached at gain 1.10, T=300)."""
    rnn = types.SimpleNamespace(N=2000, equation_type="r", activation=torch.relu,
                                device=torch.device("cpu"))
    x = torch.full((2000, 4096), CAP, dtype=torch.float32)
    x[0, 0] = 1e13
    v = Penalties(RNN=rnn).fr_magnitude_penalty(x, **ARGS)
    assert torch.isfinite(v), "penalty overflowed — the float64 powers are not in effect"
    assert v.item() > 1e30, f"expected a huge but finite value, got {v.item():.3g}"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
