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


def _target_for(p, mode, sigma_log=1.2, scale_q=0.5):
    """The per-unit target that synaptic_scaling_ would use, extracted for direct comparison.

    Args:
        p: (N,) participation tensor.
        mode: 'median' (single set-point) or 'lognormal' (rank-matched).
    Returns:
        (N,) tensor of targets.
    """
    live = p >= 0.05 * torch.quantile(p, 0.95)
    if mode == "median":
        return torch.full_like(p, float(torch.quantile(p[live], scale_q)))
    N = p.numel()
    mu = torch.log(torch.quantile(p[live], 0.5).clamp_min(1e-8))
    ranks = torch.argsort(torch.argsort(p)).to(p.dtype)
    z = torch.erfinv(2.0 * ((ranks + 0.5) / N) - 1.0) * float(np.sqrt(2.0))
    return torch.exp(mu + sigma_log * z)


def test_rank_matched_target_preserves_spread_where_a_set_point_destroys_it():
    """The claim behind the change: a single set-point homogenises, a rank-matched one does not."""
    g = torch.Generator().manual_seed(21)
    p = torch.exp(torch.randn(1000, generator=g) * 1.2)        # a lognormal population
    p[:300] = 0.0                                              # plus a 30% silent atom, as measured

    t_med = _target_for(p, "median")
    t_rank = _target_for(p, "lognormal")

    def decades(x):
        """Spread of the positive part, q01 to q99, in log10 units."""
        nz = x[x > 1e-12]
        return float(torch.log10(torch.quantile(nz, 0.99)) - torch.log10(torch.quantile(nz, 0.01)))

    assert decades(t_med) == 0.0, "a single set-point should have zero spread by construction"
    assert decades(t_rank) > 1.5, f"rank-matched target too narrow: {decades(t_rank):.2f} decades"
    print(f"      target spread: set-point {decades(t_med):.1f} decades, "
          f"rank-matched {decades(t_rank):.1f} decades (cortex ~2)")


def test_rank_matched_asks_silent_units_to_rejoin_the_tail_not_the_middle():
    """A silent unit's target must be small but NONZERO -- the tail, not the median."""
    g = torch.Generator().manual_seed(22)
    p = torch.exp(torch.randn(1000, generator=g) * 1.2)
    p[:300] = 0.0

    t_rank = _target_for(p, "lognormal")
    med = float(torch.quantile(p[p > 0], 0.5))
    silent = p <= 1e-12

    # The 300 tied-at-zero units are spread over ranks 0..299 by argsort, so they COLLECTIVELY fill
    # the bottom 30% of the target lognormal; which silent unit lands in which slot is arbitrary
    # and does not matter. The contract is on the set, not on any one unit.
    assert float(t_rank.min()) > 0, "a lognormal target must never be exactly zero"
    assert float(t_rank.min()) < 0.1 * med, \
        f"lowest target is {float(t_rank.min())/med:.2f}x median -- the tail is not being asked for"
    assert float(t_rank[silent].max()) < med, \
        "a silent unit is being asked to exceed the median, i.e. to become above-average"
    print(f"      silent units' targets span {float(t_rank[silent].min())/med:.3f}x to "
          f"{float(t_rank[silent].max())/med:.2f}x the median -- the lower tail, all nonzero")


def test_rank_matched_does_not_drag_the_busiest_unit_down_to_the_middle():
    """A set-point scales the busiest unit DOWN; the rank-matched target leaves it near the top."""
    g = torch.Generator().manual_seed(23)
    p = torch.exp(torch.randn(1000, generator=g) * 1.2)

    hot = int(torch.argmax(p))
    t_med, t_rank = _target_for(p, "median"), _target_for(p, "lognormal")

    assert t_med[hot] < p[hot], "premise: a set-point should pull the busiest unit down"
    assert t_rank[hot] > t_med[hot] * 5, \
        "rank-matched target for the busiest unit is not meaningfully above the set-point"
    print(f"      busiest unit: p={float(p[hot]):.2f}, set-point target={float(t_med[hot]):.2f}, "
          f"rank-matched target={float(t_rank[hot]):.2f}")


def test_row_norms_are_exactly_preserved():
    """Each unit's total input weight must be unchanged; only its E/I balance may move."""
    rnn, tr = _setup(eta=0.3)
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(43)))
    states, _ = rnn(inp, w_noise=False)
    before = rnn.W_rec.norm(dim=1).clone()
    Trainer.synaptic_scaling_(tr, states)
    after = rnn.W_rec.norm(dim=1)
    err = (after - before).abs().max().item()
    assert err < 1e-4 * before.max().item(), f"row norms moved by {err:.3g}"
    print(f"      max row-norm change {err:.2e} on norms up to {before.max():.2f}")


def test_the_sign_split_pumps_row_magnitude_unless_it_is_renormalised():
    """The defect that destroyed the first two attempts, isolated.

    Multiplying a row's positive weights by alpha and dividing its negative ones by alpha changes
    the row's magnitude by about (alpha + 1/alpha), which exceeds 2 for every alpha except 1. Over
    300 events that compounds. The test asserts the pump exists without renormalisation and is
    exactly cancelled with it.
    """
    moved = {}
    for preserve in (False, True):
        rnn, tr = _setup(eta=0.3)
        tr.scaling_args["preserve_row_norm"] = preserve
        inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(45)))
        states, _ = rnn(inp, w_noise=False)
        before = rnn.W_rec.norm(dim=1).clone()
        Trainer.synaptic_scaling_(tr, states)
        moved[preserve] = float(((rnn.W_rec.norm(dim=1) - before) / before).abs().max())

    assert moved[False] > 0.01, \
        f"premise check failed: unrenormalised scaling moved row norms by only {moved[False]:.4f}"
    assert moved[True] < 1e-5, f"renormalised scaling still moved a row norm by {moved[True]:.3g}"
    print(f"      largest row-norm change: unrenormalised {moved[False]:.3f}, "
          f"renormalised {moved[True]:.1e}")


def test_redistribution_saturates_instead_of_running_away():
    """Repeated events must converge, not compound -- the property that makes a large eta safe.

    A redistribution at fixed row norm has a fixed point (all of a unit's input weight excitatory),
    so the drive it can deliver is bounded whatever eta is. Falsifier: if the drive after 50 events
    keeps growing with eta, the operation is still unbounded and eta must be tuned rather than
    chosen freely.
    """
    ends = []
    for eta in (0.2, 0.5, 0.9):
        rnn, tr = _setup(eta=eta)
        r_pop = torch.abs(torch.randn(N, generator=torch.Generator().manual_seed(7)))
        u = torch.abs(torch.randn(2, generator=torch.Generator().manual_seed(8)))
        inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(9)))
        for _ in range(50):
            states, _ = rnn(inp, w_noise=False)
            Trainer.synaptic_scaling_(tr, states)
        ends.append(_drive(rnn, r_pop, u))

    spread = max(ends) - min(ends)
    assert spread < 0.01 * max(abs(e) for e in ends) + 1e-6, \
        f"drive still depends on eta ({ends}) -- redistribution has not saturated"
    print(f"      drive after 50 events at eta 0.2/0.5/0.9: "
          f"{ends[0]:.3f} / {ends[1]:.3f} / {ends[2]:.3f} -- saturated")


def test_redistribution_still_raises_an_inhibited_unit():
    """Preserving the norm must not cost the mechanism: drive on a silenced unit must still rise."""
    rnn, tr = _setup(eta=0.5)
    r_pop = torch.abs(torch.randn(N, generator=torch.Generator().manual_seed(7)))
    u = torch.abs(torch.randn(2, generator=torch.Generator().manual_seed(8)))
    h_before = _drive(rnn, r_pop, u)
    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(9))),
                    w_noise=False)
    Trainer.synaptic_scaling_(tr, states)
    h_after = _drive(rnn, r_pop, u)
    assert h_after > h_before, \
        f"redistribution at fixed norm failed to raise the drive: {h_before:.4f} -> {h_after:.4f}"
    print(f"      h {h_before:.4f} -> {h_after:.4f} at unchanged total synaptic weight")


if __name__ == "__main__":
    for fn in [test_sign_split_raises_an_inhibited_unit_while_naive_scaling_buries_it,
               test_signs_and_zeros_are_preserved,
               test_scaling_is_two_sided,
               test_purely_inhibited_unit_cannot_be_rescued_by_scaling_alone,
               test_rank_matched_target_preserves_spread_where_a_set_point_destroys_it,
               test_rank_matched_asks_silent_units_to_rejoin_the_tail_not_the_middle,
               test_rank_matched_does_not_drag_the_busiest_unit_down_to_the_middle,
               test_row_norms_are_exactly_preserved,
               test_the_sign_split_pumps_row_magnitude_unless_it_is_renormalised,
               test_redistribution_saturates_instead_of_running_away,
               test_redistribution_still_raises_an_inhibited_unit]:
        fn()
        print(f"PASS  {fn.__name__}")
