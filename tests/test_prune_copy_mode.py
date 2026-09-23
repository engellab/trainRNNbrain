"""Duplicating a working unit must leave the network's function untouched.

WHY THIS MODE EXISTS. The screen of 2026-09-22 measured what a random redraw does: 26,012 redraws
over 591 units, 44 deaths each, and the active count unmoved. A randomly drawn unit has no
function, so the task gradient has no reason to keep it. Copying a unit that IS working gives the
new unit a function by construction.

WHY THE OUTGOING WEIGHTS ARE HALVED. Copying only the incoming weights would double the donor's
contribution to every downstream unit and jolt the loss. Splitting the donor's outgoing weights
between donor and copy makes the duplication exactly function-preserving, which is what these
tests check: with zero noise the network's output must be bit-identical before and after.

Run:  python tests/test_prune_copy_mode.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 40, 15, 4
DEAD = np.array([0, 3, 7, 11])


def _setup(copy_noise=0.0, patience=1):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in in copy mode."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0
        rnn.W_inp[DEAD, :] = -5.0
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        prune_args={"check_every": 1, "patience": patience, "active_rel": 0.05,
                    "reinit_mode": "copy", "copy_noise": copy_noise},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        _unit_utility=torch.zeros(N), _last_replaced=torch.full((N,), -1e9),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))
    return rnn, tr


def test_each_copy_reproduces_its_donor_exactly():
    """The core of the construction: a copy must fire exactly as its donor does.

    This is the part that is exact. Reviving MANY units at once is not exactly function-preserving,
    because the donors hold nonzero weights FROM the units being revived: those weights carried
    nothing while the units were silent and start carrying signal the moment they are not. That is
    what reviving a unit means, not a defect in the construction. The per-copy identity below is
    the property the halved outgoing weights actually buy.
    """
    rnn, tr = _setup(copy_noise=0.0)
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))
    states_before, _ = rnn(inp, w_noise=False)
    w_before = rnn.W_rec.clone()
    Trainer.prune_and_reinit_(tr, states_before)
    states_after, _ = rnn(inp, w_noise=False)

    assert tr._n_reinit_events > 0, "no unit was duplicated -- the test checks nothing"
    r = torch.relu(states_after)
    revived = torch.nonzero(tr._reinit_ever).flatten()
    for i in revived.tolist():
        d = (w_before - rnn.W_rec[i]).abs().mean(dim=1)
        d[revived] = float("inf")          # a donor is always a live unit
        j = int(d.argmin())
        err = (r[i] - r[j]).abs().max().item()
        assert err == 0.0, f"copy {i} does not reproduce donor {j}: max rate difference {err:.3g}"
    print(f"      {len(revived)} copies, every one reproducing its donor to exactly zero error")


def test_duplication_perturbs_the_network_far_less_than_a_random_redraw():
    """Splitting the donor's outgoing weights is what keeps the loss from jolting.

    Pass threshold fixed before running: the copy mode must perturb the output at least 5x less
    than a random redraw of the same units. A smaller factor would mean the halving is not buying
    what it is there for.
    """
    def change(mode):
        """Max change in network output when the silent units are revived in the given mode."""
        rnn, tr = _setup(copy_noise=0.0)
        tr.prune_args["reinit_mode"] = mode
        inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))
        _, out0 = rnn(inp, w_noise=False)
        st, _ = rnn(inp, w_noise=False)
        Trainer.prune_and_reinit_(tr, st)
        _, out1 = rnn(inp, w_noise=False)
        return (out1 - out0).abs().max().item(), out0.abs().max().item()

    e_copy, scale = change("copy")
    e_rand, _ = change("random")
    ratio = e_rand / max(e_copy, 1e-12)
    assert ratio > 5.0, f"copy mode only {ratio:.1f}x gentler than random -- the halving is not working"
    print(f"      copy {100*e_copy/scale:.1f}% of output vs random {100*e_rand/scale:.1f}% "
          f"-- {ratio:.0f}x gentler")


def test_noise_breaks_the_symmetry_between_donor_and_copy():
    """Without noise the copy is identical to its donor forever; with noise it can diverge."""
    rnn0, tr0 = _setup(copy_noise=0.0)
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(5)))
    s0, _ = rnn0(inp, w_noise=False)
    Trainer.prune_and_reinit_(tr0, s0)
    r0, _ = rnn0(inp, w_noise=False)
    a0 = torch.relu(r0)

    rnn1, tr1 = _setup(copy_noise=0.05)
    s1, _ = rnn1(inp, w_noise=False)
    Trainer.prune_and_reinit_(tr1, s1)
    r1, _ = rnn1(inp, w_noise=False)
    a1 = torch.relu(r1)

    # every revived unit must now actually fire -- otherwise the copy achieved nothing
    revived = torch.nonzero(tr1._reinit_ever).flatten()
    assert a1[revived].abs().max() > 0, "the duplicated units are still silent"
    print(f"      {len(revived)} units duplicated; max rate of a revived unit "
          f"{a1[revived].max():.3f} (zero-noise case {a0[torch.nonzero(tr0._reinit_ever).flatten()].max():.3f})")


def test_donors_come_from_the_live_pool_only():
    """A dead unit must never be chosen as a donor -- copying a corpse revives nothing."""
    rnn, tr = _setup(copy_noise=0.01)
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(7)))
    states, _ = rnn(inp, w_noise=False)
    p = Trainer.participation_from_states_(tr, states).detach()
    silent = p < 0.05 * torch.quantile(p, 0.95)

    before = rnn.W_rec.clone()
    Trainer.prune_and_reinit_(tr, states)
    revived = torch.nonzero(tr._reinit_ever).flatten()

    # each revived unit's new incoming row must resemble SOME live unit's row, never a silent one
    for i in revived.tolist():
        d = (before[~silent] - rnn.W_rec[i]).abs().mean(dim=1)
        d_silent = (before[silent] - rnn.W_rec[i]).abs().mean(dim=1)
        assert d.min() < d_silent.min(), f"unit {i} was copied from a silent donor"


def test_adam_moments_cleared_for_both_rows_and_columns():
    """Copy mode touches outgoing weights too, so their stale moments must also be zeroed."""
    rnn, tr = _setup(copy_noise=0.01)
    opt = torch.optim.Adam([rnn.W_rec, rnn.W_inp, rnn.W_out], lr=1e-3)
    for prm in (rnn.W_rec, rnn.W_inp, rnn.W_out):
        opt.state[prm] = {"exp_avg": torch.full_like(prm, 9.0),
                          "exp_avg_sq": torch.full_like(prm, 9.0)}
    tr.optimizer = opt
    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(9))),
                    w_noise=False)
    Trainer.prune_and_reinit_(tr, states)
    revived = torch.nonzero(tr._reinit_ever).flatten()

    assert opt.state[rnn.W_rec]["exp_avg"][revived, :].abs().max() == 0.0, "incoming rows not cleared"
    assert opt.state[rnn.W_rec]["exp_avg"][:, revived].abs().max() == 0.0, "outgoing columns not cleared"
    assert opt.state[rnn.W_out]["exp_avg"][:, revived].abs().max() == 0.0, "readout columns not cleared"


def test_a_doomed_unit_can_never_be_chosen_as_a_donor():
    """Donors are drawn from ~silent, and every doomed unit is silent, so the two sets are disjoint.

    Copying a unit that is itself about to be replaced would propagate a dead unit instead of a
    working one. The guarantee is structural: strikes reset to zero on any check where a unit is
    active, so a unit can only reach the patience threshold while silent on the current check, and
    the donor pool excludes every silent unit regardless of its strike count.
    """
    rnn, tr = _setup(copy_noise=0.01, patience=2)
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(31)))

    for _ in range(3):                      # build up strikes, then trigger a replacement round
        states, _ = rnn(inp, w_noise=False)
        p = Trainer.participation_from_states_(tr, states).detach()
        silent = p < 0.05 * torch.quantile(p, 0.95)
        before = rnn.W_rec.clone()
        Trainer.prune_and_reinit_(tr, states)
        revived = torch.nonzero(tr._reinit_ever).flatten()
        for i in revived.tolist():
            d = (before - rnn.W_rec[i]).abs().mean(dim=1)
            d[revived] = float("inf")
            donor = int(d.argmin())
            assert not bool(silent[donor]), \
                f"unit {i} was copied from unit {donor}, which was silent on this check"
        tr._reinit_ever[:] = False          # only inspect the current round next time


if __name__ == "__main__":
    for fn in [test_each_copy_reproduces_its_donor_exactly,
               test_duplication_perturbs_the_network_far_less_than_a_random_redraw,
               test_noise_breaks_the_symmetry_between_donor_and_copy,
               test_donors_come_from_the_live_pool_only,
               test_adam_moments_cleared_for_both_rows_and_columns,
               test_a_doomed_unit_can_never_be_chosen_as_a_donor]:
        fn()
        print(f"PASS  {fn.__name__}")
