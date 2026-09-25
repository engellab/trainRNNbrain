"""A perturbed copy must change DIRECTION without changing magnitude.

WHY. Duplication recruits units (702 active against a control's 275 at N=1000) but the units it
recruits carry little independent activity: dimensions per active unit fall from 0.020 to 0.013.
The obvious fix is to jitter the copy harder, which raises the question of how hard, and the
existing knob confounds the answer. Multiplicative jitter grows the copied row's norm by
sqrt(1 + jitter^2) -- 3.2x at jitter = 3 -- so a sweep over the noise level would vary the copy's
weight MAGNITUDE and its DIRECTION at once. Magnitude is the axis that has already destabilised
this network twice: the uncapped duplication runs, and synaptic scaling in every configuration.

`copy_norm_preserve` rescales the jittered row back to the donor row's original length. The noise
level then sets one quantity, the angle between the copy's weight vector and the donor's, whose
expected cosine is 1 / sqrt(1 + jitter^2).

CONTRACT, fixed before running:
  1. With copy_norm_preserve, the copy's incoming row norm sits in [0.97, 1.0] times its donor's
     at every noise level. Not exactly 1.0: the rescale pins the length at the donor's, and the
     construction then zeroes the two cross terms to avoid inventing a self-loop through the twin,
     which can only shorten the row. Two entries of a 400-long row at up to 3 sigma bound the loss
     at about 2% of the norm, so 3% is the tolerance and it is one-sided.
  2. Without it, the norm grows by sqrt(1 + jitter^2), give or take the same 3% -- the historical
     behaviour, unchanged.
  3. The mean cosine between copy and donor follows 1 / sqrt(1 + jitter^2) to within 0.05.
  4. At jitter = 0 the flag does nothing: norm and cosine are both within the same one-sided 3%
     of the donor's. Removing a component of relative length e leaves cosine sqrt(1 - e^2), so the
     2% of norm the zeroed cross terms take also costs about 2% of cosine -- the same bound, not a
     separate allowance.

Run:  python tests/test_prune_copy_norm_preserve.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

# n_outputs is 8 so that a W_out column identifies a unit. Duplication sets the copy's W_out column
# equal to the donor's halved one and nothing afterwards touches W_out, which makes it the only
# fingerprint that survives arbitrary jitter on the incoming row. W_rec columns do not work: the
# construction zeroes the two cross terms afterwards to avoid inventing a self-loop through the twin.
N, T, B, N_OUT = 400, 15, 4, 8
DEAD = np.arange(0, 40)


def _setup(copy_noise, norm_preserve):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in in copy mode.

    Args:
        copy_noise: float, relative multiplicative jitter on the copy's incoming weights.
        norm_preserve: bool, whether to rescale the jittered row to the donor row's length.
    Returns: (RNN_torch, trainer stand-in).
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=N_OUT, equation_type="h", seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0
        rnn.W_inp[DEAD, :] = -5.0
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        prune_args={"check_every": 1, "patience": 1, "active_rel": 0.05,
                    "reinit_mode": "copy", "copy_noise": copy_noise,
                    "copy_norm_preserve": norm_preserve,
                    # ONE replacement per call. Each replacement halves its donor's outgoing
                    # column, so when many happen in one event the rows read later in the loop no
                    # longer match a snapshot taken before the event, and the copy's row comes out
                    # about 0.6x its donor's for reasons that have nothing to do with the jitter.
                    "max_replace_frac": 1e-6},
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        _unit_utility=torch.zeros(N), _last_replaced=torch.full((N,), -1e9),
        participation_from_states_=lambda s, **k: Trainer.participation_from_states_(tr, s, **k))
    return rnn, tr


def _pairs(rnn, tr, w_out_after):
    """Match revived units to the donors they were copied from, via the W_out fingerprint.

    A donor drawn twice has its W_out column halved twice, so the first copy no longer matches it;
    such pairs are dropped rather than guessed.

    Args:
        rnn: the RNN after duplication.
        tr: the trainer stand-in, whose _reinit_ever marks the revived units.
        w_out_after: clone of W_out taken after duplication.
    Returns: list of (copy index, donor index).
    """
    revived = set(torch.nonzero(tr._reinit_ever).flatten().tolist())
    out = []
    for i in sorted(revived):
        hits = [j for j in range(rnn.N) if j not in revived
                and torch.allclose(w_out_after[:, i], w_out_after[:, j], atol=1e-8)]
        if len(hits) == 1:
            out.append((i, hits[0]))
    return out


def _run(copy_noise, norm_preserve, n_events=30):
    """Duplicate one unit per call, repeatedly, and collect copy/donor row norms and cosines.

    Args:
        copy_noise: float, relative multiplicative jitter.
        norm_preserve: bool, whether to rescale to the donor's row length.
        n_events: int, how many single-unit replacement events to run.
    Returns: (copy norms, donor norms, cosines) as numpy arrays, one entry per event.
    """
    rnn, tr = _setup(copy_noise, norm_preserve)
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))
    cn, dn, cos = [], [], []
    for _ in range(n_events):
        states, _ = rnn(inp, w_noise=False)
        w_rec_before = rnn.W_rec.detach().clone()
        seen = torch.nonzero(tr._reinit_ever).flatten().tolist()
        Trainer.prune_and_reinit_(tr, states)
        fresh = [i for i in torch.nonzero(tr._reinit_ever).flatten().tolist() if i not in seen]
        if len(fresh) != 1:
            continue
        i = fresh[0]
        hits = [j for j in range(rnn.N) if j != i
                and torch.allclose(rnn.W_out[:, i].detach(), rnn.W_out[:, j].detach(), atol=1e-8)]
        if len(hits) != 1:
            continue
        c, d = rnn.W_rec[i, :].detach(), w_rec_before[hits[0], :]
        cn.append(float(c.norm()))
        dn.append(float(d.norm()))
        cos.append(float(torch.dot(c, d) / (c.norm() * d.norm()).clamp_min(1e-12)))
    assert len(cn) >= 15, f"only {len(cn)} copy/donor pairs recovered -- too few to test"
    return np.array(cn), np.array(dn), np.array(cos)


def test_norm_preserve_holds_the_row_length_at_the_donors():
    """With the flag on, the copy's incoming row has exactly the donor's length."""
    for jitter in (0.3, 1.0, 3.0):
        cn, dn, _ = _run(jitter, True)
        ratio = cn / dn
        assert ratio.max() <= 1.0 + 1e-6, \
            f"jitter {jitter}: row norm EXCEEDS the donor's, up to {ratio.max():.4f}x"
        assert ratio.min() > 0.97, \
            f"jitter {jitter}: row norm as low as {ratio.min():.4f}x the donor's"
        print(f"      jitter {jitter}: {len(cn)} pairs, row norm {ratio.min():.4f}-"
              f"{ratio.max():.4f}x the donor's")


def test_without_the_flag_the_norm_grows_as_before():
    """The historical behaviour is untouched: the norm grows by sqrt(1 + jitter^2)."""
    for jitter in (0.3, 1.0, 3.0):
        cn, dn, _ = _run(jitter, False)
        ratio = (cn / dn).mean()
        expected = np.sqrt(1.0 + jitter ** 2)
        assert abs(ratio - expected) / expected < 0.05, \
            f"jitter {jitter}: norm ratio {ratio:.3f}, expected about {expected:.3f}"
        print(f"      jitter {jitter}: norm ratio {ratio:.3f} against sqrt(1+jitter^2) = "
              f"{expected:.3f} (flag off)")


def test_cosine_to_the_donor_follows_the_noise_level():
    """The noise level sets the angle: expected cosine 1 / sqrt(1 + jitter^2)."""
    for jitter in (0.3, 1.0, 3.0):
        _, _, cos = _run(jitter, True)
        expected = 1.0 / np.sqrt(1.0 + jitter ** 2)
        assert abs(cos.mean() - expected) < 0.05, \
            f"jitter {jitter}: mean cosine {cos.mean():.3f}, expected about {expected:.3f}"
        print(f"      jitter {jitter}: mean cosine to donor {cos.mean():.3f} "
              f"(spread {cos.std():.3f}), theory {expected:.3f}")


def test_zero_jitter_leaves_the_copy_all_but_exact():
    """The flag must be inert when there is no noise to rescale."""
    cn, dn, cos = _run(0.0, True)
    assert (cn / dn).min() > 0.97, "zero jitter shortened the row by more than the zeroed entries"
    assert (cn / dn).max() <= 1.0 + 1e-6, "zero jitter lengthened the row"
    assert cos.min() > 0.97, f"zero jitter rotated the row: min cosine {cos.min():.4f}"
    print(f"      jitter 0: {len(cn)} pairs, cosine to donor {cos.min():.4f}-{cos.max():.4f}, "
          f"norm {(cn/dn).min():.4f}-{(cn/dn).max():.4f}x the donor's")


if __name__ == "__main__":
    for t in (test_norm_preserve_holds_the_row_length_at_the_donors,
              test_without_the_flag_the_norm_grows_as_before,
              test_cosine_to_the_donor_follows_the_noise_level,
              test_zero_jitter_leaves_the_copy_all_but_exact):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
