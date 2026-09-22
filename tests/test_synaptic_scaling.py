"""Homeostatic scaling must RAISE the drive of a unit that inhibition has silenced.

The trap this pins down: naive multiplicative scaling (multiply the whole incoming row by
alpha > 1) has the WRONG SIGN of effect on exactly the case that matters. A unit silenced because
its net drive is negative gets a MORE negative drive when its row is scaled up, and is buried
deeper. Biology scales excitatory and inhibitory synapses in opposite directions; this tests that
the implemented rule does too, and that the naive rule would have failed.

Run:  python tests/test_synaptic_scaling.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 40, 15, 4
INHIBITED = 0          # the unit we silence with net-negative recurrent drive


def _setup(eta=0.5, scale_q=0.5, every=1):
    """A small non-Dale ReLU RNN plus a Trainer stand-in exposing what synaptic_scaling_ touches."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    with torch.no_grad():
        # net-negative drive: mostly strong inhibition, a little excitation left to scale up
        rnn.W_rec[INHIBITED, :] = -1.0
        rnn.W_rec[INHIBITED, 1:6] = 0.1
        rnn.W_inp[INHIBITED, :] = -1.0
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0,
        scaling_args={"every": every, "eta": eta, "scale_q": scale_q},
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))
    return rnn, tr


def _drive(rnn, r_pop, u):
    """Net input h to the INHIBITED unit given a population rate vector and an input vector."""
    return float(rnn.W_rec[INHIBITED] @ r_pop + rnn.W_inp[INHIBITED] @ u)


def test_sign_split_raises_an_inhibited_unit_while_naive_scaling_buries_it():
    """The core claim: E-up/I-down raises h for a net-inhibited unit; uniform scaling lowers it."""
    rnn, tr = _setup()
    r_pop = torch.abs(torch.randn(N, generator=torch.Generator().manual_seed(7)))
    u = torch.abs(torch.randn(2, generator=torch.Generator().manual_seed(8)))

    h_before = _drive(rnn, r_pop, u)
    assert h_before < 0, f"setup failed: the unit is not net-inhibited (h={h_before:.3f})"

    # what NAIVE uniform scaling would do to the same row
    alpha = 1.5
    naive = float((rnn.W_rec[INHIBITED] * alpha) @ r_pop + (rnn.W_inp[INHIBITED] * alpha) @ u)
    assert naive < h_before, "premise check: uniform scaling should make a negative drive worse"

    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(9))),
                    w_noise=False)
    Trainer.synaptic_scaling_(tr, states)
    h_after = _drive(rnn, r_pop, u)

    assert h_after > h_before, \
        f"sign-split scaling failed to raise the drive: {h_before:.4f} -> {h_after:.4f}"
    print(f"      h: {h_before:.4f} -> {h_after:.4f} (sign-split), naive would give {naive:.4f}")


def test_signs_and_zeros_are_preserved():
    """Scaling must not flip any weight's sign or fill in a structural zero (Dale/mask safety)."""
    rnn, tr = _setup()
    with torch.no_grad():
        rnn.W_rec[5, :10] = 0.0                       # a structural-looking zero block
    before = rnn.W_rec.clone()
    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(11))),
                    w_noise=False)
    Trainer.synaptic_scaling_(tr, states)

    assert torch.equal(torch.sign(before), torch.sign(rnn.W_rec)), "a weight changed sign"
    assert (rnn.W_rec[5, :10] == 0).all(), "a structural zero was filled in"


def test_scaling_is_two_sided():
    """Over-active units must be scaled DOWN -- that is what homogenises, not just a floor lift."""
    rnn, tr = _setup(eta=0.5)
    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(13))),
                    w_noise=False)
    p = Trainer.participation_from_states_(tr, states).detach()
    live = p >= 0.05 * torch.quantile(p, 0.95)
    target = torch.quantile(p[live], 0.5)
    hot = int(torch.argmax(p))                         # the busiest unit: must shrink
    assert p[hot] > target, "setup failed: no unit above the set-point"

    pos_before = rnn.W_rec[hot][rnn.W_rec[hot] > 0].clone()
    Trainer.synaptic_scaling_(tr, states)
    pos_after = rnn.W_rec[hot][rnn.W_rec[hot] > 0]
    assert (pos_after < pos_before).all(), "an over-active unit's excitation was not scaled down"


def test_purely_inhibited_unit_cannot_be_rescued_by_scaling_alone():
    """The documented limit: with no excitatory input, h rises toward 0 but never crosses it."""
    rnn, tr = _setup(eta=0.9, every=1)
    with torch.no_grad():
        rnn.W_rec[INHIBITED, :] = -1.0                 # no excitation at all
        rnn.W_inp[INHIBITED, :] = -1.0
    r_pop = torch.abs(torch.randn(N, generator=torch.Generator().manual_seed(15)))
    u = torch.abs(torch.randn(2, generator=torch.Generator().manual_seed(16)))

    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(17))),
                    w_noise=False)
    for _ in range(30):
        Trainer.synaptic_scaling_(tr, states)
    h = _drive(rnn, r_pop, u)
    assert h < 0, f"a purely inhibited unit should not cross zero by scaling alone (h={h:.4f})"
    print(f"      after 30 scalings h = {h:.4f} — still below zero, as documented")


if __name__ == "__main__":
    for fn in [test_sign_split_raises_an_inhibited_unit_while_naive_scaling_buries_it,
               test_signs_and_zeros_are_preserved,
               test_scaling_is_two_sided,
               test_purely_inhibited_unit_cannot_be_rescued_by_scaling_alone]:
        fn()
        print(f"PASS  {fn.__name__}")
