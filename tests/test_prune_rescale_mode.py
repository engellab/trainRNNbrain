"""Gradual unsuppression: keep a silent unit's trained weights, nudge them until it fires.

WHY THIS MODE EXISTS. Across the revival rules measured at N=1000, every rule that threw away a
unit's trained incoming weights landed between 262 and 333 active units (random 262, zero_out 289,
orth 294, mix 333), while the two that kept trained weights reached 373 (bias_kick) and 702 (copy).
This rule keeps the unit's own incoming weights and only rescales them -- excitation up by alpha,
inhibition down by alpha -- so the drive rises from both directions while the unit's learned
selectivity is preserved within the excitatory set and within the inhibitory set separately.

WHAT SEPARATES IT FROM synaptic_scaling_, WHICH BLEW UP. That rule scaled every unit toward a
population set-point at every event and never stopped, so a unit it could not rescue was scaled at
all 400 events; measured afterwards, the inflated rows in its one surviving configuration belong to
the units it FAILED on. Here the rule stops on revival, never scales an active unit down, and caps
the cumulative boost per unit.

CONTRACT, fixed before running:
  1. Signs are preserved and zeros stay zero: excitation multiplied by alpha, inhibition divided.
  2. With rescale_normalize, the incoming row's length is unchanged, so the event is a pure
     redistribution from inhibition to excitation.
  3. Without it, the row changes by exactly sqrt((alpha^2 E + I/alpha^2)/(E + I)), where E and I
     are the summed squares of the excitatory and inhibitory weights. That factor exceeds 1 only
     when the two are comparable -- the pump that killed synaptic_scaling_. An inhibition-dominated
     row, which is what a silenced unit has, SHRINKS under the same rule.
  4. The outgoing column is redrawn at 1/sqrt(N), not zeroed, so a revived unit is visible to the
     loss immediately.
  5. Active units are never touched.
  6. The cumulative cap stops the boost for a unit that never revives.
  7. The point of it all: the unit's net recurrent drive goes UP.

Run:  python tests/test_prune_rescale_mode.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 200, 20, 4
DEAD = np.arange(0, 20)


def _setup(alpha, normalize=True, cap=8.0, one_at_a_time=False):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in in rescale mode.

    Args:
        alpha: float, per-event multiplicative boost.
        normalize: bool, whether each incoming row keeps its length.
        cap: float, cumulative boost ceiling per unit.
        one_at_a_time: bool, boost a single unit per event. Needed wherever a test inspects one
            row: redrawing a boosted unit's outgoing COLUMN overwrites that column in every row,
            so with several units boosted at once a row carries redrawn entries that were never
            scaled, and comparing it against its former self measures the wrong thing.
    Returns: (RNN_torch, trainer stand-in).
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=3, equation_type="h", self_connections=True, seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0 * torch.rand(len(DEAD), N, generator=torch.Generator().manual_seed(1))
        rnn.W_inp[DEAD, :] = -5.0
        # give the dead units some excitation too, or there is nothing for alpha to act on
        rnn.W_rec[DEAD[:, None], np.arange(40, 80)[None, :]] = 0.02
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        prune_args={"check_every": 1, "patience": 1, "active_rel": 0.05,
                    "reinit_mode": "rescale", "rescale_alpha": alpha,
                    "rescale_normalize": normalize, "rescale_cap": cap,
                    "maturity": 0,
                    "max_replace_frac": 1e-6 if one_at_a_time else 1.0},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool), _unit_utility=torch.zeros(N),
        _last_replaced=torch.full((N,), -1e9), _rescale_cum=torch.ones(N),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))
    return rnn, tr


def _step(rnn, tr, inp):
    """Run one forward pass and one rescale event. Returns the silent mask used."""
    states, _ = rnn(inp, w_noise=False)
    p = tr.participation_from_states_(states).detach()
    silent = p < 0.05 * torch.quantile(p, 0.95)
    Trainer.prune_and_reinit_(tr, states)
    return silent


def _inputs():
    """The fixed probe batch."""
    return torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))


def test_signs_preserved_and_split_applied():
    """Excitation up by alpha, inhibition down by alpha, zeros untouched."""
    a = 1.5
    rnn, tr = _setup(a, one_at_a_time=True)
    inp = _inputs()
    before = rnn.W_rec.detach().clone()
    silent = _step(rnn, tr, inp)
    i = int(torch.nonzero(silent).flatten()[0])
    off = [k for k in range(N) if k != i]           # the diagonal is redrawn with the column
    b, c = before[i, off], rnn.W_rec[i, off].detach()
    assert torch.all(torch.sign(b) == torch.sign(c)), "a weight changed sign"
    assert torch.all(c[b == 0] == 0), "a zero weight became nonzero"
    pos, neg = b > 0, b < 0
    # normalize is on by default, so the split shows up as a RATIO between the two groups
    ratio = (c[pos] / b[pos]).median() / (c[neg] / b[neg]).median()
    assert abs(float(ratio) - a * a) < 0.05 * a * a, \
        f"excitatory/inhibitory gain ratio {float(ratio):.4f}, expected {a*a:.4f}"
    print(f"      unit {i}: excitatory gain / inhibitory gain = {float(ratio):.4f}, "
          f"expected alpha^2 = {a*a:.4f}")


def test_normalize_holds_the_row_length():
    """With normalization the incoming row is redistributed at fixed total synaptic weight."""
    rnn, tr = _setup(1.5, normalize=True, one_at_a_time=True)
    inp = _inputs()
    before = rnn.W_rec.detach().clone()
    silent = _step(rnn, tr, inp)
    i = int(torch.nonzero(silent).flatten()[0])
    off = [k for k in range(N) if k != i]
    r = float(rnn.W_rec[i, off].detach().norm() / before[i, off].norm())
    assert abs(r - 1.0) < 1e-4, f"row length changed by {r:.5f}x with normalization on"
    print(f"      row length ratio {r:.6f} with normalization on")


def test_without_normalize_the_row_follows_the_pump_formula():
    """The row changes by exactly sqrt((a^2 E + I/a^2)/(E + I)) -- in whichever direction."""
    a = 1.5
    rnn, tr = _setup(a, normalize=False, one_at_a_time=True)
    inp = _inputs()
    before = rnn.W_rec.detach().clone()
    silent = _step(rnn, tr, inp)
    i = int(torch.nonzero(silent).flatten()[0])
    off = [k for k in range(N) if k != i]
    b = before[i, off]
    E, I = float(b[b > 0].pow(2).sum()), float(b[b < 0].pow(2).sum())
    expect = float(np.sqrt((a * a * E + I / (a * a)) / (E + I)))
    got = float(rnn.W_rec[i, off].detach().norm() / b.norm())
    assert abs(got - expect) < 1e-3, f"row changed {got:.4f}x, the formula gives {expect:.4f}x"
    print(f"      inhibition-dominated row (I/E = {I/max(E,1e-12):.0f}): changed {got:.4f}x, "
          f"formula {expect:.4f}x -- it SHRINKS, it does not pump")


def test_the_pump_grows_a_balanced_row():
    """Where excitation and inhibition are comparable, the same rule inflates the row."""
    a = 1.5
    rnn, tr = _setup(a, normalize=False, one_at_a_time=True)
    with torch.no_grad():
        # a silenced unit whose excitation and inhibition are matched in magnitude: the regime the
        # alpha + 1/alpha argument describes, and the one the old set-point rule kept every unit in
        row = torch.zeros(N)
        row[:80] = 0.10
        row[80:160] = -0.10
        rnn.W_rec[DEAD[0], :] = row
        rnn.W_inp[DEAD[0], :] = -5.0
    inp = _inputs()
    before = rnn.W_rec.detach().clone()
    silent = _step(rnn, tr, inp)
    i = int(torch.nonzero(silent).flatten()[0])
    off = [k for k in range(N) if k != i]
    b = before[i, off]
    E, I = float(b[b > 0].pow(2).sum()), float(b[b < 0].pow(2).sum())
    expect = float(np.sqrt((a * a * E + I / (a * a)) / (E + I)))
    got = float(rnn.W_rec[i, off].detach().norm() / b.norm())
    assert got > 1.0, f"a balanced row should grow, it changed {got:.4f}x"
    assert abs(got - expect) < 1e-3, f"row grew {got:.4f}x, the formula gives {expect:.4f}x"
    print(f"      balanced row (I/E = {I/max(E,1e-12):.2f}): grew {got:.4f}x, "
          f"formula {expect:.4f}x -- this is the pump")


def test_outgoing_is_redrawn_not_zeroed():
    """A revived unit must be visible to the loss, which a zeroed column would prevent."""
    rnn, tr = _setup(1.5)
    inp = _inputs()
    silent = _step(rnn, tr, inp)
    idx = torch.nonzero(silent).flatten()
    col = rnn.W_rec[:, idx].detach()
    assert float(col.abs().max()) > 0, "outgoing column was zeroed"
    got, want = float(col.std()), 1.0 / np.sqrt(N)
    assert abs(got - want) / want < 0.2, f"outgoing draw std {got:.4f}, expected {want:.4f}"
    assert float(rnn.W_out[:, idx].detach().abs().max()) > 0, "readout column was zeroed"
    print(f"      {idx.numel()} outgoing columns redrawn at std {got:.4f} "
          f"(1/sqrt(N) = {want:.4f}), readout nonzero")


def test_active_units_are_never_touched():
    """One-sided by construction: an active unit is not scaled down."""
    rnn, tr = _setup(1.5)
    inp = _inputs()
    before = rnn.W_rec.detach().clone()
    silent = _step(rnn, tr, inp)
    live = torch.nonzero(~silent).flatten()
    d = (rnn.W_rec[live, :].detach() - before[live, :]).abs().max()
    # live rows may still change in the columns of revived units, so compare only live-to-live
    ll = (rnn.W_rec[live][:, live].detach() - before[live][:, live]).abs().max()
    assert float(ll) == 0.0, f"an active unit's weights from other active units moved by {float(ll):.3g}"
    print(f"      {live.numel()} active units unchanged among themselves (max diff {float(ll):.1e}); "
          f"their columns from revived units changed by up to {float(d):.3f}, as intended")


def test_cumulative_cap_stops_a_unit_that_never_revives():
    """`until it revives` has to be bounded when it never does."""
    a, cap = 1.5, 4.0
    rnn, tr = _setup(a, normalize=False, cap=cap)
    inp = _inputs()
    norms = []
    for _ in range(12):
        tr.iter_n += 1
        _step(rnn, tr, inp)
        norms.append(float(rnn.W_rec[DEAD[0], :].detach().norm()))
    reached = float(tr._rescale_cum[DEAD[0]])
    assert reached <= cap * a, f"cumulative boost {reached:.2f} ran past the cap {cap}"
    assert abs(norms[-1] - norms[-2]) < 1e-6, "the row was still growing after the cap"
    print(f"      cumulative boost stopped at {reached:.2f} (cap {cap}); "
          f"row norm flat at {norms[-1]:.4f} over the last events")


def test_the_drive_on_a_silent_unit_goes_up():
    """The mechanism: moving weight from inhibition to excitation raises the unit's input."""
    rnn, tr = _setup(1.5)
    inp = _inputs()
    states, _ = rnn(inp, w_noise=False)
    r = torch.relu(states).detach().mean(dim=(1, 2))
    before = rnn.W_rec.detach().clone()
    silent = _step(rnn, tr, inp)
    idx = torch.nonzero(silent).flatten()
    off = torch.tensor([k for k in range(N) if k not in set(idx.tolist())])
    d0 = (before[idx][:, off] @ r[off])
    d1 = (rnn.W_rec[idx][:, off].detach() @ r[off])
    rose = float((d1 > d0).float().mean())
    assert rose > 0.9, f"drive rose for only {100*rose:.0f}% of the boosted units"
    print(f"      drive rose for {100*rose:.0f}% of {idx.numel()} boosted units "
          f"(median change {float((d1 - d0).median()):+.4f})")


if __name__ == "__main__":
    for t in (test_signs_preserved_and_split_applied,
              test_normalize_holds_the_row_length,
              test_without_normalize_the_row_follows_the_pump_formula,
              test_the_pump_grows_a_balanced_row,
              test_outgoing_is_redrawn_not_zeroed,
              test_active_units_are_never_touched,
              test_cumulative_cap_stops_a_unit_that_never_revives,
              test_the_drive_on_a_silent_unit_goes_up):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
