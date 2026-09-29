"""The three-state rescale rule: dormant -> growing -> mature, with a fixed activity target.

WHY IT LOOKS LIKE THIS. The first version stopped boosting the moment a unit cleared the silence
floor and every cell landed at the control's active count. The weights said why: the units it
failed on ended with an excitation/inhibition ratio far BELOW the control's, which the rule cannot
produce directly, so they revived, the boost stopped, and the gradient a revived unit now has put
them back. Three changes follow from that.

  FIXED TARGET, NOT A QUANTILE. A quantile of the live pool is a goalpost defined on the population
  the rule is modifying; at the median, half that pool is below it by construction, so no unit is
  ever released by reaching it. The target is a fraction of frm's cap, cap_fr*log1p(UpV)/log1p(N),
  on frm's own soft-max-over-time activity.
  ONE OUTGOING REDRAW PER EPISODE, at the reset, instead of every step -- redrawing repeatedly
  throws away the gradient a marginally firing unit has accumulated on the weights that are its
  only route to earning a role.
  INCOMING ROWS PROTECTED from the gradient while growing, capped in number, since a protected row
  is a row removed from training.

CONTRACT, fixed before running:
  1. Opening an episode redraws the outgoing column; the next boost step does not redraw it again.
  2. A growing unit has its incoming gradient zeroed and its outgoing gradient untouched.
  3. Reaching the target graduates the unit: it stops growing and stops being protected.
  4. A mature unit that goes dormant again opens a NEW episode with a fresh budget.
  5. The number of units growing at once never exceeds rescale_protect_frac * N.

Run:  python tests/test_prune_rescale_states.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 200, 20, 4
DEAD = np.arange(0, 60)


def _setup(target_frac=0.3, protect_frac=0.20, alpha=1.5, cap=8.0):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in in rescale mode.

    Args:
        target_frac: float, maturity target as a fraction of frm's cap.
        protect_frac: float, ceiling on the fraction of units growing at once.
        alpha: float, per-event boost (large here so a few events do visible work).
        cap: float, per-episode cumulative boost ceiling.
    Returns: (RNN_torch, trainer stand-in).
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=3, equation_type="h", self_connections=True, seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0 * torch.rand(len(DEAD), N,
                                               generator=torch.Generator().manual_seed(1))
        rnn.W_inp[DEAD, :] = -5.0
        rnn.W_rec[DEAD[:, None], np.arange(80, 120)[None, :]] = 0.02
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        Penalties=types.SimpleNamespace(UpV=100), prune_reinit=True,
        frm_args={"cap_fr": 0.3, "tau": 0.1},
        prune_args={"check_every": 1, "patience": 1, "active_rel": 0.05,
                    "reinit_mode": "rescale", "rescale_alpha": alpha,
                    "rescale_normalize": True, "rescale_cap": cap,
                    "rescale_target_frac": target_frac,
                    "rescale_protect_frac": protect_frac,
                    "maturity": 0, "max_replace_frac": 1.0},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool), _unit_utility=torch.zeros(N),
        _last_replaced=torch.full((N,), -1e9), _rescale_cum=torch.ones(N),
        _rescale_growing=torch.zeros(N, dtype=torch.bool), _rescale_episodes=torch.zeros(N))
    tr.participation_from_states_ = lambda s, **k: Trainer.participation_from_states_(tr, s, **k)
    tr.rescale_rows_ = lambda i, al, nm, live=None: Trainer.rescale_rows_(tr, i, al, nm, live)
    tr.frm_activity_cap_ = lambda: Trainer.frm_activity_cap_(tr)
    tr.frm_activity_ = lambda s: Trainer.frm_activity_(tr, s)
    tr.zero_protected_grads_ = lambda: Trainer.zero_protected_grads_(tr)
    return rnn, tr


def _inputs():
    """The fixed probe batch."""
    return torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))


def _step(rnn, tr, inp):
    """One forward pass and one rescale event."""
    tr.iter_n += 1
    states, _ = rnn(inp, w_noise=False)
    Trainer.prune_and_reinit_(tr, states)
    return states


def test_outgoing_is_redrawn_once_per_episode_not_every_step():
    """The redraw opens an episode; boosting afterwards leaves the column to train."""
    rnn, tr = _setup()
    inp = _inputs()
    _step(rnn, tr, inp)                                  # opens episodes
    grew = torch.nonzero(tr._rescale_growing).flatten()
    assert grew.numel() > 0, "no episode opened"
    col_after_reset = rnn.W_rec[:, grew].detach().clone()
    out_after_reset = rnn.W_out[:, grew].detach().clone()
    cum_before = tr._rescale_cum.clone()
    _step(rnn, tr, inp)                                  # a pure boost step
    still = tr._rescale_growing[grew]
    assert bool(still.any()), "every unit graduated in one step; the test checks nothing"
    g = grew[still]
    assert torch.equal(rnn.W_out[:, g].detach(), out_after_reset[:, still]), \
        "the readout column was redrawn on a boost step"
    assert (tr._rescale_cum[g] > cum_before[g]).all(), "the boost did not fire"
    print(f"      {g.numel()} growing units: readout column untouched on the boost step, "
          f"cumulative boost advanced")


def test_growing_units_have_incoming_gradient_zeroed_and_outgoing_left_alone():
    """Protection covers the rows the rule tilts, and nothing else."""
    rnn, tr = _setup()
    inp = _inputs()
    _step(rnn, tr, inp)
    g = torch.nonzero(tr._rescale_growing).flatten()
    assert g.numel() > 0, "no unit is growing"
    free = torch.tensor([k for k in range(N) if not bool(tr._rescale_growing[k])])
    rnn.W_rec.grad = torch.ones(N, N)
    rnn.W_inp.grad = torch.ones(N, rnn.W_inp.shape[1])
    rnn.W_out.grad = torch.ones(rnn.W_out.shape[0], N)
    tr.zero_protected_grads_()
    assert float(rnn.W_rec.grad[g, :].abs().max()) == 0.0, "a growing unit's incoming row kept gradient"
    assert float(rnn.W_inp.grad[g, :].abs().max()) == 0.0, "a growing unit's input row kept gradient"
    assert float(rnn.W_rec.grad[free, :].abs().min()) == 1.0, "a free unit's row lost gradient"
    assert float(rnn.W_out.grad[:, g].abs().min()) == 1.0, \
        "the readout column was protected; it must stay free to find a use for the unit"
    print(f"      {g.numel()} protected: incoming gradient zeroed, readout gradient intact, "
          f"{free.numel()} free units untouched")


def test_reaching_the_target_graduates_the_unit():
    """Maturity ends both the boost and the protection."""
    rnn, tr = _setup()
    inp = _inputs()
    _step(rnn, tr, inp)
    g = torch.nonzero(tr._rescale_growing).flatten()
    assert g.numel() > 0
    i = int(g[0])
    with torch.no_grad():                       # make one unit unambiguously active
        rnn.W_rec[i, :] = 0.0
        rnn.W_inp[i, :] = 8.0
    states = _step(rnn, tr, inp)
    act = Trainer.frm_activity_(tr, states)
    cap = Trainer.frm_activity_cap_(tr)
    assert float(act[i]) >= 0.3 * cap, \
        f"fixture failed: unit {i} at activity {float(act[i]):.4f}, target {0.3*cap:.4f}"
    assert not bool(tr._rescale_growing[i]), "a unit past the target is still growing"
    print(f"      unit {i} reached activity {float(act[i]):.3f} against target "
          f"{0.3*cap:.3f} and graduated")


def test_a_mature_unit_that_dies_again_gets_a_fresh_budget():
    """Episodes are per-episode, not per-lifetime."""
    # the ceiling is lifted here on purpose: with it binding, a re-dormant unit competes for the
    # one freed slot against every other dormant unit and usually loses, which is correct behaviour
    # but makes the test blind to the thing it is checking
    rnn, tr = _setup(protect_frac=1.0)
    inp = _inputs()
    _step(rnn, tr, inp)
    i = int(torch.nonzero(tr._rescale_growing).flatten()[0])
    with torch.no_grad():                       # graduate it
        rnn.W_rec[i, :] = 0.0
        rnn.W_inp[i, :] = 8.0
    _step(rnn, tr, inp)
    assert not bool(tr._rescale_growing[i])
    eps_before, spent = float(tr._rescale_episodes[i]), float(tr._rescale_cum[i])
    with torch.no_grad():                       # kill it again
        rnn.W_rec[i, :] = -5.0
        rnn.W_inp[i, :] = -5.0
    _step(rnn, tr, inp)                         # re-reset fires, and boosts once in the same event
    alpha = tr.prune_args["rescale_alpha"]
    assert float(tr._rescale_episodes[i]) > eps_before, "no new episode was opened"
    # the budget restarts at 1.0 and takes exactly one boost in the event that opened the episode,
    # so it lands on alpha -- not on the previous episode's spend times alpha
    assert abs(float(tr._rescale_cum[i]) - alpha) < 1e-5, \
        (f"budget is {float(tr._rescale_cum[i]):.4f}, expected {alpha:.4f} from a fresh start; "
         f"the previous episode had spent {spent:.4f}")
    print(f"      unit {i}: episodes {eps_before:.0f} -> {float(tr._rescale_episodes[i]):.0f}, "
          f"budget restarted at 1.0 and stands at {float(tr._rescale_cum[i]):.3f} "
          f"(previous episode spent {spent:.3f})")


def test_the_protected_population_never_exceeds_its_ceiling():
    """A protected row is a row out of training, so the count has to be bounded."""
    rnn, tr = _setup(protect_frac=0.10)
    inp = _inputs()
    ceiling = int(round(0.10 * N))
    seen = []
    for _ in range(8):
        _step(rnn, tr, inp)
        seen.append(int(tr._rescale_growing.sum()))
    assert max(seen) <= ceiling, f"growing count reached {max(seen)}, ceiling {ceiling}"
    print(f"      growing count over 8 events: {seen}, ceiling {ceiling}")


def test_active_only_moves_synapses_from_firing_units_and_leaves_the_rest():
    """The boost goes where it can raise h, and nowhere else.

    Drive is sum_k W[i,k]*r_k, so a synapse from a silent source delivers nothing however large it
    is made. With rescale_active_only the recurrent boost touches only the columns of firing units;
    the columns of silent units are left bitwise alone. Input weights are always rescaled in full,
    because the task input is driving on every step.

    Contract, fixed before running: with the option on, a growing unit's weights from ACTIVE
    sources move and its weights from SILENT sources do not; with it off, both move.
    """
    for active_only, expect_silent_cols_move in ((True, False), (False, True)):
        # a small ceiling on purpose: with it lifted, every silent unit opens an episode and has
        # its column redrawn, leaving no silent column untouched to compare against
        rnn, tr = _setup(protect_frac=0.05)
        tr.prune_args["rescale_active_only"] = active_only
        tr.prune_args["rescale_normalize"] = False      # isolate the restriction from renormalising
        inp = _inputs()
        states, _ = rnn(inp, w_noise=False)
        p = tr.participation_from_states_(states).detach()
        silent = p < 0.05 * torch.quantile(p, 0.95)
        live_cols = torch.nonzero(~silent).flatten()
        dead_cols = torch.nonzero(silent).flatten()
        assert live_cols.numel() > 5 and dead_cols.numel() > 5, "fixture has no two-sided split"
        before = rnn.W_rec.detach().clone()
        was_growing = tr._rescale_growing.clone()
        Trainer.prune_and_reinit_(tr, states)
        g = torch.nonzero(tr._rescale_growing).flatten()
        assert g.numel() > 0, "nothing is growing"
        i = int(g[0])
        # Every unit that OPENED an episode this event had its outgoing column redrawn, which
        # writes into row i at that unit's position. Those columns say nothing about whether row i
        # was boosted, so they come out of both comparisons -- along with i's own column.
        opened = (tr._rescale_growing & ~was_growing)
        opened[i] = True
        keep = ~opened
        lc = torch.nonzero(~silent & keep).flatten()
        dc = torch.nonzero(silent & keep).flatten()
        assert lc.numel() > 5 and dc.numel() > 5, "too few untouched columns to compare"
        moved_live = not torch.allclose(rnn.W_rec[i, lc].detach(), before[i, lc], atol=1e-9)
        moved_dead = not torch.allclose(rnn.W_rec[i, dc].detach(), before[i, dc], atol=1e-9)
        assert moved_live, f"active_only={active_only}: weights from firing units did not move"
        assert moved_dead == expect_silent_cols_move, \
            (f"active_only={active_only}: weights from silent units "
             f"{'moved' if moved_dead else 'did not move'}")
        print(f"      active_only={active_only}: from firing units moved={moved_live}, "
              f"from silent units moved={moved_dead}")


if __name__ == "__main__":
    for t in (test_outgoing_is_redrawn_once_per_episode_not_every_step,
              test_growing_units_have_incoming_gradient_zeroed_and_outgoing_left_alone,
              test_reaching_the_target_graduates_the_unit,
              test_a_mature_unit_that_dies_again_gets_a_fresh_budget,
              test_the_protected_population_never_exceeds_its_ceiling,
              test_active_only_moves_synapses_from_firing_units_and_leaves_the_rest):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
