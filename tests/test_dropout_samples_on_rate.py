"""The dropout sampler must score FIRING RATES, not pre-activations.

The bug this pins down: until 2026-09-21 the sampler used its own scorer, `get_participation_`, on
the raw states, so for equation_type "h" it read q(|h|) + std(|h|). A unit held far below threshold
on every trial has a large |h| and therefore scored HIGH, so the sampler kept dropping units that
were already silent -- a no-op -- and 51 +- 4% of the drop mass was wasted that way. The sampler now
shares `participation_from_states_` with the logged trace, which applies the activation first.

Run:  python tests/test_dropout_samples_on_rate.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N = 40
SILENT = np.arange(0, N, 2)          # every other unit is forced permanently below threshold


def _fake_trainer():
    """Minimal stand-in exposing only what participation_from_states_ touches."""
    return types.SimpleNamespace(RNN=types.SimpleNamespace(equation_type="h", activation=torch.relu))


def _states(seed=1):
    """(N, T, B) pre-activations in which the units in SILENT never cross threshold."""
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(N, 25, 8, generator=g)
    x[SILENT] = -torch.abs(x[SILENT]) - 5.0
    return x


def test_silent_units_score_zero():
    """A unit that never fires must get participation exactly 0, however negative its h is."""
    x = _states()
    v = Trainer.participation_from_states_(_fake_trainer(), x, q=0.9).numpy()
    assert np.allclose(v[SILENT], 0.0), f"silent units scored up to {v[SILENT].max():.3g}"
    assert (v[1::2] > 0).all(), "units that do fire must score above zero"

    # and the point of the change: the OLD scorer ranked those same silent units ABOVE the live ones
    a = x.reshape(N, -1).abs()
    v_old = (torch.quantile(a, 0.9, dim=1) + a.std(dim=1, unbiased=False)).numpy()
    assert v_old[SILENT].mean() > v_old[1::2].mean(), \
        "sanity check failed: the pre-activation scorer should favour the deeply-silent units"


def test_activity_q_is_honoured():
    """The q argument must reach the quantile -- it was hard-coded to 0.9 before."""
    g = torch.Generator().manual_seed(3)
    x = torch.rand(N, 25, 8, generator=g) * 2.0 - 0.5
    t = _fake_trainer()
    v50 = Trainer.participation_from_states_(t, x, q=0.5).numpy()
    v99 = Trainer.participation_from_states_(t, x, q=0.99).numpy()
    assert not np.allclose(v50, v99), "activity_q had no effect on the score"
    assert (v99 >= v50 - 1e-6).all(), "a higher quantile cannot lower the score"


def test_exactly_k_units_are_dropped_at_every_beta():
    """The dose must not depend on beta. It used to: the clamped Bernoulli lost most of the budget
    as the softmax concentrated (50 -> 7.9 dropped at beta=4, N=1000, drop_rate=0.05), which made
    the ladder's beta=4 arm a dose experiment masquerading as a targeting one. k counts the LIVE
    pool, which is half of the units here."""
    v = Trainer.participation_from_states_(_fake_trainer(), _states(), q=0.9)
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(5))
    n_live = int((v >= 0.05 * torch.quantile(v, 0.95)).sum())
    assert n_live == N - len(SILENT), f"expected {N - len(SILENT)} live units, found {n_live}"
    for beta in (0.0, 1.0, 4.0, 16.0):
        args = {"dropout_kind": "dead", "sampling_method": "participation",
                "drop_rate": 0.25, "dropout_beta": beta}
        for _ in range(20):
            keep = RNN_torch.get_dropout_mask(rnn, args, v)
            assert keep.shape == (N, 1), f"mask must be (N, 1), got {tuple(keep.shape)}"
            n_dropped = int((keep == 0).sum())
            assert n_dropped == round(0.25 * n_live), \
                f"beta={beta}: dropped {n_dropped}, expected exactly {round(0.25 * n_live)}"


def test_drops_land_on_units_that_fire():
    """Silent units are outside the pool entirely, so NO draw may ever land on one -- at any
    beta, including beta=0 where the weights within the pool are uniform."""
    v = Trainer.participation_from_states_(_fake_trainer(), _states(), q=0.9)
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(6))
    args = {"dropout_kind": "dead", "sampling_method": "participation",
            "drop_rate": 0.2, "dropout_beta": 0.0}
    silent_hits = total = 0
    for _ in range(200):
        dropped = (RNN_torch.get_dropout_mask(rnn, args, v).squeeze(1) == 0).numpy()
        silent_hits += dropped[SILENT].sum()
        total += dropped.sum()
    assert silent_hits == 0, f"{silent_hits} of {total} drawn units were already silent"


def test_unknown_sampling_method_is_named():
    """A typo in the config must say so, not fall through to an UnboundLocalError."""
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(7))
    try:
        RNN_torch.get_dropout_mask(rnn, {"sampling_method": "participaton", "drop_rate": 0.1}, None)
    except ValueError as e:
        assert "participaton" in str(e), f"error should name the bad value, got: {e}"
    else:
        raise AssertionError("an unknown sampling_method must raise")


def test_beta_is_scale_free():
    """beta must act on RANK, not on the raw score. Multiplying every participation by 100 -- which
    is what a runaway does, and did: a beta=4 run reached participation 579 where healthy runs sit
    at 0.8-1.3 -- must not change how concentrated the sampling is. Under the raw-score softmax the
    busiest unit went from 4.5x uniform to 375x, i.e. drawn every iteration, which hides it from the
    task loss (scored on the dropout pass) and lets it run away."""
    v = Trainer.participation_from_states_(_fake_trainer(), _states(), q=0.9)
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(11))
    top = int(torch.argmax(v))
    args = {"dropout_kind": "dead", "sampling_method": "participation",
            "drop_rate": 0.2, "dropout_beta": 4.0}
    hits = {}
    for scale in (1.0, 100.0):
        vs = v * scale
        hits[scale] = sum(int(RNN_torch.get_dropout_mask(rnn, args, vs)[top, 0] == 0)
                          for _ in range(400)) / 400.0
    assert abs(hits[1.0] - hits[100.0]) < 0.10, \
        f"p(drop) for the busiest unit moved {hits[1.0]:.3f} -> {hits[100.0]:.3f} when the score " \
        "was rescaled; beta is not scale-free"
    assert hits[1.0] < 0.95, f"busiest unit drawn {hits[1.0]:.0%} of iterations -- it would be " \
        "hidden from the task loss every step and could run away unopposed"



if __name__ == "__main__":
    test_silent_units_score_zero()
    test_activity_q_is_honoured()
    test_exactly_k_units_are_dropped_at_every_beta()
    test_drops_land_on_units_that_fire()
    test_unknown_sampling_method_is_named()
    test_beta_is_scale_free()
    print("all six checks passed")
