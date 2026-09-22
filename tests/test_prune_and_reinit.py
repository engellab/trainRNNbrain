"""Prune-and-reinitialise must actually revive frozen units.

The premise: a silent ReLU unit is FROZEN, not merely quiet. With r_i = 0 at every timestep,
dL/dW_rec[i,j] ~ relu'(h_i) * r_j = 0 and dL/dW_rec[j,i] ~ r_i = 0, so every weight into and out
of it has exactly zero gradient and no penalty on the loss can move it. If that premise is false
the intervention is pointless, so the first test checks it directly rather than assuming it.

Run:  python tests/test_prune_and_reinit.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 30, 12, 4
DEAD = np.array([0, 3, 7, 11])      # units forced permanently below threshold


def _rnn(seed=0):
    """A small non-Dale ReLU RNN with the units in DEAD wired to never fire."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=seed)
    with torch.no_grad():
        # strongly negative incoming drive -> h_i < 0 for every input, so r_i = 0 always
        rnn.W_rec[DEAD, :] = -5.0
        rnn.W_inp[DEAD, :] = -5.0
    return rnn


def test_dead_units_have_exactly_zero_gradient():
    """The premise: a permanently silent ReLU unit gets zero gradient on ALL its weights."""
    rnn = _rnn()
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(2)))
    states, out = rnn(inp, w_noise=False)

    r = torch.relu(states) if rnn.equation_type == "h" else states
    assert r[DEAD].abs().max() == 0.0, "setup failed: the DEAD units are not actually silent"

    out.pow(2).mean().backward()
    g_in = rnn.W_rec.grad[DEAD, :].abs().max().item()     # incoming
    g_out = rnn.W_rec.grad[:, DEAD].abs().max().item()    # outgoing
    assert g_in == 0.0, f"incoming gradient not zero: {g_in:.3g}"
    assert g_out == 0.0, f"outgoing gradient not zero: {g_out:.3g}"

    live = np.setdiff1d(np.arange(N), DEAD)
    assert rnn.W_rec.grad[live, :].abs().max() > 0, "live units should have nonzero gradient"


def test_reinit_fires_only_after_patience_and_redraws_incoming():
    """Strikes must accumulate to `patience` before a redraw, and only incoming weights change."""
    rnn = _rnn()
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        prune_args={"check_every": 1, "patience": 3, "active_rel": 0.05},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))

    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3))),
                    w_noise=False)
    before_in = rnn.W_rec[DEAD, :].clone()
    before_out = rnn.W_rec[:, DEAD].clone()
    p0 = Trainer.participation_from_states_(tr, states).detach()
    active = torch.nonzero(p0 >= 0.05 * torch.quantile(p0, 0.95)).flatten()

    for step in range(2):                      # strikes 1 and 2 -> below patience, no redraw
        Trainer.prune_and_reinit_(tr, states)
        assert tr._n_reinit_events == 0, f"redrew after only {step+1} strikes"
    Trainer.prune_and_reinit_(tr, states)      # third strike -> redraw

    # An untrained N=30 ReLU net already has ~10 units at exactly zero participation, so the
    # redraw set is a SUPERSET of DEAD. The contract is that DEAD is included and that units
    # which are clearly firing are left alone -- not an exact count.
    assert tr._reinit_ever[DEAD].all(), "the forced-dead units were not all redrawn"
    assert not torch.equal(rnn.W_rec[DEAD, :], before_in), "incoming weights were not redrawn"
    # Only rows that were NOT themselves redrawn: a redrawn row j legitimately overwrites its own
    # entry W_rec[j, DEAD] as part of unit j's incoming draw.
    kept = torch.nonzero(~tr._reinit_ever).flatten()
    assert torch.equal(rnn.W_rec[:, DEAD][kept], before_out[kept]), \
        "outgoing weights of untouched units must not change"
    assert not tr._reinit_ever[active].any(), "units that were firing must not be redrawn"
    assert (tr._reinit_strikes[DEAD] == 0).all(), "strikes must reset after a redraw"
    # redrawn from N(0, 1/sqrt(N)): nothing like the -5.0 that killed them
    assert rnn.W_rec[DEAD, :].abs().max() < 1.0, "redraw is not at initialisation scale"


def test_redrawn_units_become_unfrozen():
    """After a redraw the revived units must carry nonzero gradient again -- the whole point."""
    rnn = _rnn()
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        prune_args={"check_every": 1, "patience": 1, "active_rel": 0.05},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))

    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(4)))
    states, _ = rnn(inp, w_noise=False)
    Trainer.prune_and_reinit_(tr, states)
    assert tr._reinit_ever[DEAD].all(), "the forced-dead units were not redrawn"

    rnn.zero_grad(set_to_none=True)
    _, out2 = rnn(inp, w_noise=False)
    out2.pow(2).mean().backward()
    g = rnn.W_rec.grad[DEAD, :].abs().max().item()
    assert g > 0, "redrawn units are still frozen -- the redraw did not restore drive"


def test_adam_moments_are_cleared():
    """Stale Adam momentum would push a redrawn unit straight back to dead; it must be zeroed."""
    rnn = _rnn()
    opt = torch.optim.Adam([rnn.W_rec, rnn.W_inp], lr=1e-3)
    opt.state[rnn.W_rec] = {"exp_avg": torch.full((N, N), 9.0),
                            "exp_avg_sq": torch.full((N, N), 9.0)}
    opt.state[rnn.W_inp] = {"exp_avg": torch.full_like(rnn.W_inp, 9.0),
                            "exp_avg_sq": torch.full_like(rnn.W_inp, 9.0)}
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=opt,
        prune_args={"check_every": 1, "patience": 1, "active_rel": 0.05},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))

    states, _ = rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(5))),
                    w_noise=False)
    Trainer.prune_and_reinit_(tr, states)

    kept = torch.nonzero(~tr._reinit_ever).flatten()
    assert len(kept) > 0, "every unit was redrawn -- the test cannot check the untouched case"
    for prm in (rnn.W_rec, rnn.W_inp):
        st = opt.state[prm]
        for key in ("exp_avg", "exp_avg_sq"):
            assert st[key][DEAD, :].abs().max() == 0.0, f"{key} not cleared for redrawn units"
            assert st[key][kept, :].abs().min() == 9.0, f"{key} wrongly cleared for kept units"


if __name__ == "__main__":
    for fn in [test_dead_units_have_exactly_zero_gradient,
               test_reinit_fires_only_after_patience_and_redraws_incoming,
               test_redrawn_units_become_unfrozen,
               test_adam_moments_are_cleared]:
        fn()
        print(f"PASS  {fn.__name__}")
