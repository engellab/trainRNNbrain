"""Scrambling a copied row: the donor's exact weights, in the wrong places.

WHY. The jitter sweep found that rotating a copy away from its donor does not cost recruitment --
at jitter 3.0 the new unit shares a cosine of 0.32 with its donor, 37% of its signs are flipped,
and it recruits 752 active units against 685 with no jitter at all. Jitter keeps the donor's wiring
POSITIONS and scrambles the values. A permutation is the complementary control: it keeps every
value, sign and zero of the donor's row and scrambles only which unit each weight comes from.

Between the two, "what the weights look like" and "which units it listens to" come apart.

CONTRACT, fixed before running:
  1. The permuted row holds exactly the donor's multiset of weights over the non-pair positions.
  2. Its cosine to the donor's row is near zero -- the alignment is what the permutation destroys.
  3. The 2x2 self-weight block spanning donor and copy is untouched by the permutation.
  4. With copy_permute off, the row is the donor's, position for position.

Run:  python tests/test_prune_copy_permute.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B, N_OUT = 400, 15, 4, 8
DEAD = np.arange(0, 40)


def _setup(permute):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in in copy mode.

    Args:
        permute: bool, whether the copied row is scrambled across source units.
    Returns: (RNN_torch, trainer stand-in).
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=N_OUT, equation_type="h", self_connections=True, seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0
        rnn.W_inp[DEAD, :] = -5.0
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        prune_args={"check_every": 1, "patience": 1, "active_rel": 0.05,
                    "reinit_mode": "copy", "copy_noise": 0.0, "copy_permute": permute,
                    # one replacement per call, so a snapshot taken beforehand still describes the
                    # donor's row when it is read
                    "max_replace_frac": 1e-6},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        _unit_utility=torch.zeros(N), _last_replaced=torch.full((N,), -1e9),
        _rescale_cum=torch.ones(N),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))
    return rnn, tr


def _duplicate_once(permute):
    """Run one single-unit duplication and report who was copied from whom.

    Args:
        permute: bool, whether the copied row is scrambled.
    Returns: (rnn, w_rec_before, copy index, donor index).
    """
    rnn, tr = _setup(permute)
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
    return rnn, w_before, i, hits[0]


def test_the_permuted_row_holds_the_donors_weights():
    """Every value, sign and zero of the donor's row survives; only the places change."""
    rnn, w_before, i, j = _duplicate_once(True)
    off = [k for k in range(N) if k not in (i, j)]
    got = torch.sort(rnn.W_rec[i, off].detach()).values
    want = torch.sort(w_before[j, off]).values
    d = float((got - want).abs().max())
    assert d < 1e-7, f"the multiset changed: largest sorted-value difference {d:.3g}"
    print(f"      {len(off)} weights, sorted values identical to the donor's (max diff {d:.1e})")


def test_the_permutation_destroys_the_alignment():
    """Same weights, different sources: the cosine to the donor collapses."""
    rnn, w_before, i, j = _duplicate_once(True)
    off = [k for k in range(N) if k not in (i, j)]
    c, d = rnn.W_rec[i, off].detach(), w_before[j, off]
    cos = float(torch.dot(c, d) / (c.norm() * d.norm()))
    assert abs(cos) < 0.2, f"cosine to the donor is {cos:.3f}, the permutation did not scramble"
    print(f"      cosine to donor {cos:+.4f} after permuting (1.000 without it)")


def test_the_self_weight_block_survives_the_permutation():
    """The 2x2 block spanning donor and copy is set after the row, so it is unaffected."""
    rnn, w_before, i, j = _duplicate_once(True)
    half = 0.5 * float(w_before[j, j])
    for name, v in (("W[copy,copy]", float(rnn.W_rec[i, i])),
                    ("W[copy,donor]", float(rnn.W_rec[i, j])),
                    ("W[donor,copy]", float(rnn.W_rec[j, i])),
                    ("W[donor,donor]", float(rnn.W_rec[j, j]))):
        assert abs(v - half) < 1e-7, f"{name} is {v:+.6f}, expected {half:+.6f}"
    print(f"      2x2 block still at half the donor's self-weight ({half:+.4f})")


def test_without_the_flag_the_row_is_the_donors():
    """copy_permute off leaves the existing behaviour untouched."""
    rnn, w_before, i, j = _duplicate_once(False)
    off = [k for k in range(N) if k not in (i, j)]
    d = float((rnn.W_rec[i, off].detach() - w_before[j, off]).abs().max())
    assert d < 1e-7, f"row differs from the donor's by {d:.3g} with permutation off"
    print(f"      permutation off: row matches the donor position for position (max diff {d:.1e})")


def test_iid_keeps_the_donors_outgoing_column_and_nothing_else():
    """copy_iid: incoming weights owe the donor nothing, the outgoing column is still the donor's.

    Contract, fixed before running: the copy's incoming row has the initialisation scale and no
    relation to the donor's, while its outgoing column equals the donor's halved one -- which is
    what makes this a single-variable test of the outgoing projection.
    """
    rnn, tr = _setup(False)
    tr.prune_args["copy_iid"] = True
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))
    states, _ = rnn(inp, w_noise=False)
    w_before = rnn.W_rec.detach().clone()
    Trainer.prune_and_reinit_(tr, states)
    i = torch.nonzero(tr._reinit_ever).flatten().tolist()[0]
    hits = [j for j in range(N) if j != i
            and torch.allclose(rnn.W_out[:, i].detach(), rnn.W_out[:, j].detach(), atol=1e-8)]
    j = hits[0]
    off = [k for k in range(N) if k not in (i, j)]
    c, d = rnn.W_rec[i, off].detach(), w_before[j, off]
    cos = float(torch.dot(c, d) / (c.norm() * d.norm()))
    assert abs(cos) < 0.2, f"incoming row still resembles the donor's, cosine {cos:+.3f}"
    got, want = float(c.std()), 1.0 / np.sqrt(N)
    assert abs(got - want) / want < 0.2, f"incoming draw std {got:.4f}, expected {want:.4f}"
    assert torch.allclose(rnn.W_rec[:, i].detach(), rnn.W_rec[:, j].detach(), atol=1e-7), \
        "the outgoing column is not the donor's halved one"
    print(f"      incoming iid at std {got:.4f} (cosine to donor {cos:+.3f}); "
          f"outgoing column identical to the donor's halved one")


if __name__ == "__main__":
    for t in (test_the_permuted_row_holds_the_donors_weights,
              test_the_permutation_destroys_the_alignment,
              test_the_self_weight_block_survives_the_permutation,
              test_without_the_flag_the_row_is_the_donors,
              test_iid_keeps_the_donors_outgoing_column_and_nothing_else):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
