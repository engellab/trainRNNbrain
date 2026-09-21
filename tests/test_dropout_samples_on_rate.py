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


def test_drop_mass_avoids_silent_units():
    """End to end: p_drop built from rate scores must not spend itself on silent units."""
    v = Trainer.participation_from_states_(_fake_trainer(), _states(), q=0.9)
    rnn = types.SimpleNamespace(N=N, device=torch.device("cpu"),
                                random_generator=torch.Generator().manual_seed(4))
    args = {"dropout_kind": "dead", "sampling_method": "participation",
            "drop_rate": 0.2, "dropout_beta": 4.0}
    keep = RNN_torch.get_dropout_mask(rnn, args, v)
    assert keep.shape == (N, 1), f"mask must be (N, 1) i.e. shared across the batch, got {tuple(keep.shape)}"

    p_drop = torch.clamp(args["drop_rate"] * N * torch.softmax(args["dropout_beta"] * v, dim=0),
                         0.0, 0.999).numpy()
    wasted = p_drop[SILENT].sum() / p_drop.sum()
    assert wasted < 0.05, f"{wasted:.1%} of the drop mass still lands on already-silent units"


if __name__ == "__main__":
    test_silent_units_score_zero()
    test_activity_q_is_honoured()
    test_drop_mass_avoids_silent_units()
    print("all three checks passed")
