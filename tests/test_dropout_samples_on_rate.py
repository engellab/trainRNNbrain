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


def test_expected_drop_count_hits_the_budget():
    """Units are drawn INDEPENDENTLY now, so the count fluctuates -- but its MEAN must be the
    budget at every beta. The old code lost most of the budget as beta rose (50 nominal drops
    became 7.9 at beta=4) because clipping at 0.999 discarded whatever a saturated unit could not
    absorb. drop_probabilities redistributes it instead."""
    v = Trainer.participation_from_states_(_fake_trainer(), _states(), q=0.9)
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(5))
    n_live = int((v >= 0.05 * torch.quantile(v, 0.95)).sum())
    for beta in (0.0, 1.0, 4.0, 16.0):
        args = {"dropout_kind": "dead", "sampling_method": "participation",
                "drop_rate": 0.25, "dropout_beta": beta, "p_max": 0.9}
        counts = []
        for _ in range(400):
            mask = RNN_torch.get_dropout_mask(rnn, args, v)
            assert mask.silence.shape == (N, 1), f"mask fields must be (N,1) i.e. shared across the batch"
            counts.append(int((mask.silence == 0).sum()))
        mean, target = sum(counts) / len(counts), 0.25 * n_live
        assert abs(mean - target) < 0.1 * target, \
            f"beta={beta}: mean drops {mean:.1f}, budget {target:.1f}"


def test_silence_is_binary_and_outgoing_is_unbiased():
    """The two fields do different jobs and must not be conflated.

    silence gates the unit's OWN drive, so it is binary: a surviving unit receives exactly its
    normal input, never an amplified one. outgoing carries the inverted-dropout rescaling, so its
    expectation is 1 for every live unit -- PER UNIT, because an over-scaled quiet unit and an
    under-scaled busy one cancel almost exactly in any aggregate. That cancellation is precisely
    how a uniform M/(M-k) factor looked correct while being wrong at both ends.
    """
    v = Trainer.participation_from_states_(_fake_trainer(), _states(), q=0.9)
    live = (v >= 0.05 * torch.quantile(v, 0.95)).numpy()
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(6))
    args = {"dropout_kind": "dead", "sampling_method": "participation",
            "drop_rate": 0.25, "dropout_beta": 4.0, "rescale": True, "p_max": 0.9}
    draws = 6000
    acc = np.zeros(N)
    for _ in range(draws):
        m = RNN_torch.get_dropout_mask(rnn, args, v)
        assert set(np.unique(m.silence.numpy())) <= {0.0, 1.0}, "silence must never be rescaled"
        assert not (m.outgoing.numpy()[m.silence.numpy() == 0] != 0).any(), \
            "a dropped unit must be zero in BOTH fields"
        acc += m.outgoing.squeeze(1).numpy()
    e = acc[live] / draws
    assert np.abs(e - 1.0).max() < 0.12, \
        f"E[outgoing] should be 1 per live unit; worst is {e[np.argmax(np.abs(e-1))]:.3f}"
    assert m.outgoing.max() <= 1.0 / (1.0 - args["p_max"]) + 1e-6, \
        "p_max must cap the rescaling factor"


def test_rescale_off_reproduces_a_plain_binary_mask():
    """Every completed run used rescale=false and must remain exactly reproducible."""
    v = Trainer.participation_from_states_(_fake_trainer(), _states(), q=0.9)
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(7))
    m = RNN_torch.get_dropout_mask(rnn, {"dropout_kind": "mute", "sampling_method": "participation",
                                         "drop_rate": 0.2, "dropout_beta": 2.0, "rescale": False}, v)
    assert torch.equal(m.outgoing, m.silence), "with rescale off the two fields must coincide"
    assert set(np.unique(m.outgoing.numpy())) <= {0.0, 1.0}


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


if __name__ == "__main__":
    test_silent_units_score_zero()
    test_activity_q_is_honoured()
    test_expected_drop_count_hits_the_budget()
    test_silence_is_binary_and_outgoing_is_unbiased()
    test_rescale_off_reproduces_a_plain_binary_mask()
    test_unknown_sampling_method_is_named()
    print("all five checks passed")
