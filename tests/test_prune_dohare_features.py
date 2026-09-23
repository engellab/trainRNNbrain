"""The four features borrowed from Dohare et al., Nature 632:768-774 (2024).

Their algorithm, continual backpropagation, replaces low-utility units during training. Four of its
choices differ from ours and each is tested here:

  1. `zero_out` reinit -- resample incoming weights, ZERO the outgoing ones, so a new unit cannot
     disturb what the network has learned. Our `random` mode leaves outgoing weights as training
     left them, which is a DIFFERENT intervention, so our measured failure of `random` says nothing
     about their published method.
  2. Contribution utility -- |r| * sum|outgoing weight| rather than firing rate alone, so a unit
     whose output goes nowhere scores as useless rather than as healthy.
  3. Maturity -- unconditional protection after replacement, whatever the unit does next.
  4. A replacement-rate ceiling, with the lowest-utility candidates going first.

Run:  python tests/test_prune_dohare_features.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 40, 15, 4
DEAD = np.array([0, 3, 7, 11])


def _setup(**over):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0
        rnn.W_inp[DEAD, :] = -5.0
    args = {"check_every": 1, "patience": 1, "active_rel": 0.05, "reinit_mode": "random",
            "copy_noise": 0.05, "utility": "participation", "utility_decay": 0.99,
            "maturity": 0, "max_replace_frac": 1.0}
    args.update(over)
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}), prune_args=args,
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        _unit_utility=torch.zeros(N), _last_replaced=torch.full((N,), -1e9))
    tr.participation_from_states_ = lambda s, **k: Trainer.participation_from_states_(tr, s, **k)
    return rnn, tr


def _states(rnn, seed=3):
    """One noise-free forward pass."""
    return rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(seed))),
               w_noise=False)[0]


def test_zero_out_mode_leaves_no_outgoing_weight():
    """Their rule: a replaced unit must send nothing until the gradient rebuilds its output."""
    rnn, tr = _setup(reinit_mode="zero_out")
    Trainer.prune_and_reinit_(tr, _states(rnn))
    revived = torch.nonzero(tr._reinit_ever).flatten()
    assert len(revived) > 0, "nothing was replaced -- the test checks nothing"
    assert float(rnn.W_rec[:, revived].abs().max()) == 0.0, "recurrent outgoing weights not zeroed"
    assert float(rnn.W_out[:, revived].abs().max()) == 0.0, "readout weights not zeroed"
    assert float(rnn.W_rec[revived, :].abs().max()) > 0.0, "incoming weights should be resampled"
    print(f"      {len(revived)} units replaced, all outgoing weights exactly zero")


def test_our_random_mode_is_not_their_method():
    """The distinction that makes our negative result on `random` inapplicable to their paper."""
    rnn_r, tr_r = _setup(reinit_mode="random")
    before = rnn_r.W_rec.clone()
    Trainer.prune_and_reinit_(tr_r, _states(rnn_r))
    revived = torch.nonzero(tr_r._reinit_ever).flatten()
    # Only rows that were NOT themselves redrawn: a redrawn row rewrites its own entries in every
    # column, including the columns of other revived units.
    kept = torch.nonzero(~tr_r._reinit_ever).flatten()
    assert torch.equal(rnn_r.W_rec[kept][:, revived], before[kept][:, revived]), \
        "`random` should leave outgoing weights untouched -- that is the whole point"
    assert float(rnn_r.W_rec[kept][:, revived].abs().max()) > 0, \
        "premise check: the outgoing weights should be nonzero, unlike zero_out"

    rnn_z, tr_z = _setup(reinit_mode="zero_out")
    Trainer.prune_and_reinit_(tr_z, _states(rnn_z))
    rev_z = torch.nonzero(tr_z._reinit_ever).flatten()
    assert float(rnn_z.W_rec[:, rev_z].abs().max()) == 0.0, "zero_out did not clear outgoing weights"
    print("      `random` keeps outgoing weights; `zero_out` clears them. Different interventions.")


def test_contribution_utility_demotes_a_loud_unit_that_sends_nothing():
    """A unit firing into severed outgoing weights must score as useless, not as healthy."""
    rnn, tr = _setup(utility="contribution")
    with torch.no_grad():
        rnn.W_rec[:, 5] = 0.0          # unit 5 fires but reaches nobody
        rnn.W_out[:, 5] = 0.0
    states = _states(rnn)
    p = Trainer.participation_from_states_(tr, states).detach()
    Trainer.prune_and_reinit_(tr, states)

    assert float(p[5]) > 0, "setup failed: unit 5 is not firing"
    live = p > 0.05 * torch.quantile(p, 0.95)
    assert float(tr._unit_utility[5]) == 0.0, "a unit reaching nobody must have zero contribution"
    assert float(tr._unit_utility[live].max()) > 0, "no live unit has any contribution utility"
    print(f"      unit 5 fires at {float(p[5]):.3f} and has contribution utility "
          f"{float(tr._unit_utility[5]):.3g}")


def test_maturity_protects_a_replacement_that_stays_silent():
    """The treadmill fix: a just-replaced unit is protected even if it does not start firing."""
    rnn, tr = _setup(reinit_mode="zero_out", maturity=500)
    states = _states(rnn)
    Trainer.prune_and_reinit_(tr, states)
    first = tr._n_reinit_events
    assert first > 0, "nothing was replaced on the first pass"

    tr.iter_n = 100                     # still inside the 500-iteration protection window
    Trainer.prune_and_reinit_(tr, states)
    assert tr._n_reinit_events == first, "a protected unit was replaced again inside its window"

    tr.iter_n = 900                     # window has expired
    Trainer.prune_and_reinit_(tr, states)
    assert tr._n_reinit_events > first, "protection never expired"
    print(f"      {first} replaced, none again at iter 100, more at iter 900")


def test_replacement_rate_cap_limits_the_count_and_takes_the_worst_first():
    """The rate ceiling, and the rule that the lowest-utility candidates go first."""
    rnn, tr = _setup(max_replace_frac=1.0)
    Trainer.prune_and_reinit_(tr, _states(rnn))
    uncapped = tr._n_reinit_events

    cap_frac = 0.05                     # 5% of 40 units = 2 per check
    rnn2, tr2 = _setup(max_replace_frac=cap_frac)
    states2 = _states(rnn2)
    p2 = Trainer.participation_from_states_(tr2, states2).detach()
    Trainer.prune_and_reinit_(tr2, states2)
    capped = tr2._n_reinit_events

    assert capped == max(1, int(round(cap_frac * N))), f"cap not honoured: {capped} replaced"
    assert capped < uncapped, f"cap ({capped}) did not reduce the count ({uncapped})"
    chosen = torch.nonzero(tr2._reinit_ever).flatten()
    assert float(p2[chosen].max()) <= float(p2[p2 > 0].min()) + 1e-6 or float(p2[chosen].max()) == 0.0, \
        "the cap did not take the lowest-scoring candidates first"
    print(f"      uncapped {uncapped} replaced, capped {capped}, lowest-scoring first")


if __name__ == "__main__":
    for fn in [test_zero_out_mode_leaves_no_outgoing_weight,
               test_our_random_mode_is_not_their_method,
               test_contribution_utility_demotes_a_loud_unit_that_sends_nothing,
               test_maturity_protects_a_replacement_that_stays_silent,
               test_replacement_rate_cap_limits_the_count_and_takes_the_worst_first]:
        fn()
        print(f"PASS  {fn.__name__}")
