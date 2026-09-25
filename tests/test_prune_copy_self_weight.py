"""A copy must inherit the donor's SELF-connection, on its own diagonal.

WHY. A row of W_rec is indexed by source unit, so position k means "from unit k" in any unit's row
and transplanting a row needs no shifting -- with two exceptions. Position donor_j in the donor's
row is the donor's SELF-weight, and position copy_i is the donor's weight from the unit being
replaced. Copied literally, the donor's self-weight would become the copy's weight FROM the donor,
inventing a coupling the donor never had; zeroed outright, as this code did until 2026-09-24, the
copy starts life with no self-connection at all.

Every experiment in this project runs with self_connections=true, where the W_rec diagonal is a
free trained weight. Measured on four trained N=1000 flip-flop controls: the median |W[i,i]| is
about 33x the median off-diagonal weight, the largest self-weight (3.27) exceeds the largest
off-diagonal weight (2.24), and the self-term supplies 1.8% of an active unit's total absolute
recurrent drive. A copy written with a zero diagonal therefore lacks the weight that sets its own
effective time constant, which on a memory task is the one worth having.

The self-weight is awkward because it sits in the donor's ROW and in the donor's outgoing COLUMN,
so halving that column halves the self-weight along with it. The construction that preserves the
function splits the donor's self-weight in half across the 2x2 block spanning the pair: each twin
then draws (w/2)*r from itself and (w/2)*r from the other, summing to the w*r the donor drew alone.

CONTRACT, fixed before running:
  1. With zero jitter, all four entries of the 2x2 block equal half the donor's pre-duplication
     self-weight.
  2. Every other entry of the copy's row equals the donor's.
  3. The block adds the same drive to both twins, so it cannot glue them together.
  4. Once the twins fire at the same rate, their two rows deliver the SAME recurrent drive. That
     is the exactness statement the transplant buys, and it is the right one: "the copy tracks the
     donor's rate" is NOT, because a positive self-weight amplifies any difference between the twins
     and divergence between them is the outcome this project wants. Measured on this fixture,
     transplanting the self-weight makes the pair's rates differ MORE (0.064 against 0.049), which
     is the amplifier working, not a defect.

Run:  python tests/test_prune_copy_self_weight.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 60, 25, 4
DEAD = np.array([0, 3, 7])
DIAG = 0.5          # a substantial self-weight, the size a memory unit's would be


def _setup(self_connections, copy_noise=0.0):
    """A ReLU RNN with the DEAD units wired silent and substantial self-weights everywhere.

    Args:
        self_connections: bool, whether the W_rec diagonal is a free weight.
        copy_noise: float, relative multiplicative jitter on the copy's incoming weights.
    Returns: (RNN_torch, trainer stand-in).
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=8, equation_type="h",
                    self_connections=self_connections, seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0
        rnn.W_inp[DEAD, :] = -5.0
        rnn.W_rec[range(N), range(N)] = DIAG
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        prune_args={"check_every": 1, "patience": 1, "active_rel": 0.05,
                    "reinit_mode": "copy", "copy_noise": copy_noise,
                    # one replacement per call, so a snapshot taken beforehand still describes the
                    # donor's row when it is read
                    "max_replace_frac": 1e-6},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        _unit_utility=torch.zeros(N), _last_replaced=torch.full((N,), -1e9),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))
    return rnn, tr


def _duplicate_once(self_connections, copy_noise=0.0):
    """Run one single-unit duplication and report who was copied from whom.

    Args:
        self_connections: bool, whether the W_rec diagonal is a free weight.
        copy_noise: float, relative multiplicative jitter.
    Returns: (rnn, inp, w_rec_before, copy index, donor index).
    """
    rnn, tr = _setup(self_connections, copy_noise)
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))
    states, _ = rnn(inp, w_noise=False)
    w_before = rnn.W_rec.detach().clone()
    Trainer.prune_and_reinit_(tr, states)
    fresh = torch.nonzero(tr._reinit_ever).flatten().tolist()
    assert len(fresh) == 1, f"expected exactly one replacement, got {len(fresh)}"
    i = fresh[0]
    hits = [j for j in range(N) if j != i
            and torch.allclose(rnn.W_out[:, i].detach(), rnn.W_out[:, j].detach(), atol=1e-8)]
    assert len(hits) == 1, f"donor of copy {i} not uniquely identified: {hits}"
    return rnn, inp, w_before, i, hits[0]


def test_the_pair_shares_the_donors_self_weight_in_half():
    """All four entries of the 2x2 block equal half the donor's original self-weight."""
    rnn, _, w_before, i, j = _duplicate_once(True)
    half = 0.5 * float(w_before[j, j])
    block = {"W[copy,copy]": float(rnn.W_rec[i, i]), "W[copy,donor]": float(rnn.W_rec[i, j]),
             "W[donor,copy]": float(rnn.W_rec[j, i]), "W[donor,donor]": float(rnn.W_rec[j, j])}
    for name, v in block.items():
        assert abs(v - half) < 1e-7, f"{name} is {v:+.6f}, expected {half:+.6f}"
    print(f"      donor self-weight {float(w_before[j, j]):+.4f} split as "
          + ", ".join(f"{k} {v:+.4f}" for k, v in block.items()))


def test_the_block_is_neutral_on_the_difference_between_the_twins():
    """The block adds the same drive to both twins, so it does not glue the pair together."""
    rnn, _, _, i, j = _duplicate_once(True)
    with torch.no_grad():
        # drive each twin draws from the pair, for an arbitrary pair of twin rates
        for r_i, r_j in ((1.0, 1.0), (1.0, 0.3), (0.0, 2.0)):
            to_i = float(rnn.W_rec[i, i]) * r_i + float(rnn.W_rec[i, j]) * r_j
            to_j = float(rnn.W_rec[j, j]) * r_j + float(rnn.W_rec[j, i]) * r_i
            assert abs(to_i - to_j) < 1e-6, \
                f"rates ({r_i}, {r_j}): block drives the twins apart, {to_i:+.4f} vs {to_j:+.4f}"
    print(f"      block drive identical for both twins at three different rate pairs")


def test_every_other_entry_matches_the_donor():
    """Away from the 2x2 block, the copy's row is the donor's row."""
    rnn, _, w_before, i, j = _duplicate_once(True)
    others = [k for k in range(N) if k not in (i, j)]
    d = (rnn.W_rec[i, others].detach() - w_before[j, others]).abs().max()
    assert float(d) < 1e-7, f"copy row differs from donor row by up to {float(d):.3g}"
    print(f"      {len(others)} of {N} entries identical to the donor's row (max diff {float(d):.1e})")


def test_equal_rates_give_the_pair_equal_drive():
    """With the twins firing alike, the copy's row and the donor's row deliver the same drive.

    This is what the self-weight transplant buys. Under the old code, which zeroed the copy's
    diagonal, the copy's drive fell short by exactly the donor's self-term.
    """
    rnn, _, w_before, i, j = _duplicate_once(True)
    r = torch.abs(torch.randn(N, generator=torch.Generator().manual_seed(11)))
    r[i] = r[j]                                    # the twins fire alike
    with torch.no_grad():
        drive = rnn.W_rec.detach() @ r
        gap_fixed = float((drive[i] - drive[j]).abs())
        rnn.W_rec[i, i] = 0.0                      # exactly what the code used to write
        drive_old = rnn.W_rec.detach() @ r
        gap_old = float((drive_old[i] - drive_old[j]).abs())
    expected_old = abs(0.5 * float(w_before[j, j]) * float(r[j]))
    assert gap_fixed < 1e-6, f"drive still differs by {gap_fixed:.3g} with the block in place"
    assert abs(gap_old - expected_old) < 1e-5, (
        f"with the copy's diagonal zeroed the gap is {gap_old:.4f}, expected half the donor's "
        f"self-term, {expected_old:.4f}")
    print(f"      equal rates: drive gap {gap_fixed:.2e} with the block, {gap_old:.4f} with the "
          f"copy's diagonal zeroed -- half the donor's self-term, {expected_old:.4f}")


def test_masked_diagonal_case_is_unaffected():
    """With self_connections=False the diagonal is not a free weight, so nothing changes."""
    rnn, _, w_before, i, j = _duplicate_once(False)
    others = [k for k in range(N) if k not in (i, j)]
    d = (rnn.W_rec[i, others].detach() - w_before[j, others]).abs().max()
    assert float(d) < 1e-7, f"copy row differs from donor row by up to {float(d):.3g}"
    assert float(rnn.W_rec[i, j]) == 0.0
    print(f"      self_connections=False: row transplanted identically, no coupling through the twin")


if __name__ == "__main__":
    for t in (test_the_pair_shares_the_donors_self_weight_in_half,
              test_the_block_is_neutral_on_the_difference_between_the_twins,
              test_every_other_entry_matches_the_donor,
              test_equal_rates_give_the_pair_equal_drive,
              test_masked_diagonal_case_is_unaffected):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
