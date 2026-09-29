"""A boosted row must GROW toward the firing population's median, not stay pinned at its own length.

WHY. `rescale_normalize=True` holds each boosted row at the L2 norm it already had, so the rule
can only redistribute a budget it never changes. Measured on unpenalised controls at N=1000, an
active unit's incoming recurrent row is 4.9x longer than a silent unit's (1.818 against 0.373) and
its input row 22.8x longer (0.769 against 0.034) -- so a revived unit is redistributing a budget
that is several-fold too small, and the magnitude it eventually gains arrives from the gradient
after maturity, by draining the units that had it (active-unit input rows fell 4.4x under the
target-6.0 arm while silent-unit rows rose 2.9x).

`rescale_norm_mode="median_active"` grows a short row toward the median row norm of the firing
population instead. Turning normalisation OFF entirely is the other way to add magnitude and it
diverged (r2 -44.8), so growth here is bounded twice: never faster than `alpha` per step, and never
past the median.

CONTRACT, fixed before running:
  1. With mode "self" the boosted row's norm is unchanged, to 1e-5 relative. The historical
     behaviour, which must not move.
  2. With mode "median_active" a SHORT row's norm grows by exactly `alpha` per step, to 1e-5
     relative, for as long as it stays below the median.
  3. Growth stops at the median: after enough steps the row norm sits at the median and does not
     exceed it, to 1e-5 relative.
  4. A row ALREADY longer than the median is left at its own length -- the mode grows short rows,
     it does not shrink long ones.
  5. The excitation/inhibition tilt is unaffected: with either mode EACH ROW's ratio of positive
     to |negative| weight changes by alpha^2 per step, to 1e-4 relative. Magnitude and balance are
     separate axes and this test fails if the patch coupled them. Per row, not summed over rows:
     normalisation scales each row by its own factor, so a block-summed ratio is a reweighted
     average and misses alpha^2 by about 1e-3 even when every row is exact.

Run:  python tests/test_rescale_norm_mode.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, ALPHA = 200, 1.05
SHORT = np.arange(0, 10)        # units whose rows are scaled down, standing in for silent units
LONG = np.arange(10, 20)        # units whose rows are scaled up, already past the median


def _setup(short_scale=0.2, long_scale=4.0):
    """An RNN whose rows span a range of lengths, plus a Trainer stand-in holding only the RNN.

    Args:
        short_scale: float, factor applied to the SHORT units' incoming rows.
        long_scale: float, factor applied to the LONG units' incoming rows.
    Returns: (RNN_torch, trainer stand-in, (N,) bool tensor marking the reference population).
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=2, equation_type="h", seed=0)
    with torch.no_grad():
        rnn.W_rec[SHORT, :] *= short_scale
        rnn.W_inp[SHORT, :] *= short_scale
        rnn.W_rec[LONG, :] *= long_scale
        rnn.W_inp[LONG, :] *= long_scale
    tr = types.SimpleNamespace(RNN=rnn)
    ref = torch.zeros(N, dtype=torch.bool)
    ref[20:] = True                      # the "firing population": everything left at its own scale
    return rnn, tr, ref


def _step(tr, idx, mode_ref):
    """Apply one boost to the given units and return their (W_rec norm, W_inp norm).

    Args:
        tr: the trainer stand-in.
        idx: LongTensor of unit indices to boost.
        mode_ref: (N,) bool tensor for "median_active", or None for "self".
    Returns: (recurrent row norms, input row norms), each a numpy array over idx.
    """
    with torch.no_grad():
        Trainer.rescale_rows_(tr, idx, ALPHA, True, live=None, norm_ref=mode_ref)
    return (tr.RNN.W_rec[idx, :].norm(dim=1).detach().numpy(),
            tr.RNN.W_inp[idx, :].norm(dim=1).detach().numpy())


def _median_of(W, ref):
    """Median L2 row norm of the reference population of one weight matrix."""
    return float(W[ref, :].norm(dim=1).detach().median())


def test_self_mode_holds_the_row_length():
    """Mode "self" must not change any row's norm -- the historical behaviour."""
    rnn, tr, _ = _setup()
    idx = torch.tensor(SHORT)
    before = rnn.W_rec[idx, :].norm(dim=1).detach().clone().numpy()
    rec, _ = _step(tr, idx, None)
    assert np.allclose(rec, before, rtol=1e-5), \
        f"mode 'self' changed the row norm: {before[0]:.6f} -> {rec[0]:.6f}"
    print(f"      mode 'self': row norm held at {before[0]:.6f} through a boost")


def test_median_active_grows_a_short_row_by_alpha():
    """A row below the median grows by exactly alpha per step, on BOTH weight matrices."""
    rnn, tr, ref = _setup()
    idx = torch.tensor(SHORT)
    b_rec = rnn.W_rec[idx, :].norm(dim=1).detach().clone().numpy()
    b_inp = rnn.W_inp[idx, :].norm(dim=1).detach().clone().numpy()
    rec, inp = _step(tr, idx, ref)
    assert np.allclose(rec, b_rec * ALPHA, rtol=1e-5), \
        f"W_rec grew by {rec[0] / b_rec[0]:.6f}, expected {ALPHA}"
    assert np.allclose(inp, b_inp * ALPHA, rtol=1e-5), \
        f"W_inp grew by {inp[0] / b_inp[0]:.6f}, expected {ALPHA}"
    print(f"      median_active: short row grew {rec[0] / b_rec[0]:.6f}x (alpha = {ALPHA})")


def test_growth_stops_at_the_median():
    """Repeated boosts take a short row to the median and no further."""
    rnn, tr, ref = _setup()
    idx = torch.tensor(SHORT)
    med_rec = _median_of(rnn.W_rec, ref)
    # 0.2x the starting length needs log(5)/log(1.05) ~ 33 steps; 80 is comfortably past that
    for _ in range(80):
        rec, _ = _step(tr, idx, ref)
    assert rec.max() <= med_rec * (1 + 1e-5), \
        f"row norm {rec.max():.6f} overshot the median {med_rec:.6f}"
    assert rec.min() >= med_rec * (1 - 1e-5), \
        f"row norm {rec.min():.6f} stalled short of the median {med_rec:.6f}"
    print(f"      median_active: row settled at {rec.mean():.6f}, median {med_rec:.6f}")


def test_a_long_row_is_not_shrunk():
    """The mode grows short rows; it must leave a row already past the median alone."""
    rnn, tr, ref = _setup()
    idx = torch.tensor(LONG)
    med_rec = _median_of(rnn.W_rec, ref)
    before = rnn.W_rec[idx, :].norm(dim=1).detach().clone().numpy()
    assert before.min() > med_rec, "test setup is wrong: the LONG rows are not above the median"
    rec, _ = _step(tr, idx, ref)
    assert np.allclose(rec, before, rtol=1e-5), \
        f"a long row was shrunk: {before[0]:.6f} -> {rec[0]:.6f} (median {med_rec:.6f})"
    print(f"      median_active: long row held at {before[0]:.6f} > median {med_rec:.6f}")


def _ei_per_row(W, idx):
    """Excitation/inhibition ratio of each of the given rows: sum of positives over sum of |negatives|.

    Args:
        W: a weight matrix.
        idx: LongTensor of row indices.
    Returns: numpy array, one ratio per row.
    """
    blk = W[idx, :].detach()
    return (blk.clamp(min=0).sum(dim=1) / blk.clamp(max=0).abs().sum(dim=1)).numpy()


def test_the_tilt_is_unchanged_by_the_mode():
    """Both modes change EACH ROW's excitation/inhibition by alpha^2; only magnitude differs.

    Measured per row, not summed over rows: normalisation scales every row by its own factor, so
    the ratio of block-summed excitation to block-summed inhibition is a reweighted average and is
    NOT alpha^2 even when every row is exactly alpha^2. Checking the block hides a per-row error and
    reports one that does not exist.
    """
    out = {}
    for name, use_ref in (("self", False), ("median_active", True)):
        rnn, tr, ref = _setup()
        idx = torch.tensor(SHORT)
        before = _ei_per_row(rnn.W_rec, idx)
        _step(tr, idx, ref if use_ref else None)
        r = _ei_per_row(rnn.W_rec, idx) / before
        out[name] = r
        assert np.allclose(r, ALPHA ** 2, rtol=1e-4), \
            (f"{name}: per-row E/I changed by {r.min():.6f}-{r.max():.6f}, "
             f"expected {ALPHA ** 2:.6f}")
    print(f"      per-row E/I changed by {out['self'].mean():.6f} (self) and "
          f"{out['median_active'].mean():.6f} (median_active); alpha^2 = {ALPHA ** 2:.6f}")


if __name__ == "__main__":
    for t in (test_self_mode_holds_the_row_length,
              test_median_active_grows_a_short_row_by_alpha,
              test_growth_stops_at_the_median,
              test_a_long_row_is_not_shrunk,
              test_the_tilt_is_unchanged_by_the_mode):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
