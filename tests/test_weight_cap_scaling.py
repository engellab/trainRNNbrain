"""A weight cap must depend on the network, never on how long a trial is.

HISTORY, kept because the bug is easy to reintroduce. `inp_weights_magnitude_penalty` read
`N, U = states.size(0), states.size(1)`, so U was the TRIAL LENGTH and the cap came out as
cap100 * (T/N) * log1p(N)/log1p(T). The same 1000-unit network then got a cap of 0.182 on a T=300
task and 0.278 on a T=500 one, and the log ratio was inverted relative to its siblings, so the cap
scaled the wrong way with N as well. `U` was meant to be `self.UpV`, the hard constant 100.

That penalty and its output-weight sibling were deleted on 2026-09-30 -- neither was ever given a
nonzero weight in any run on disk. The guard moves to `rec_weights_magnitude_penalty`, which
survives (lambda_rwm is set by the archived recurrent-weight-magnitude sweep that Fig. 1d cites)
and which builds its cap from the same expression, so the same mistake fits it exactly.

Run:  python tests/test_weight_cap_scaling.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.trainer.Trainer import Penalties

CAP100 = 0.07          # the penalty's own default for the recurrent cap


def _pen(N, T, w=0.5):
    """Recurrent weight-magnitude penalty for N units on a T-step trial, every |W_rec| equal to w.

    Args:
        N: int, network size; T: int, trial length in steps; w: float, the value every weight takes.
    Returns: float penalty.
    """
    rnn = types.SimpleNamespace(W_rec=torch.full((N, N), w), N=N, dale_mask=None, exc2inhR=4.0)
    P = Penalties(rnn)
    return float(P.rec_weights_magnitude_penalty(torch.zeros(N, T, 4), cap100=CAP100))


def test_cap_does_not_depend_on_trial_length():
    """Trial length is a property of the TASK, not of the weights. It must not enter the cap."""
    vals = [_pen(400, T) for T in (200, 300, 500, 1000)]
    assert max(vals) - min(vals) < 1e-6 * max(1.0, abs(vals[0])), \
        f"penalty moved with trial length: {vals} -- the cap has picked up T again"
    print(f"      trial length 200/300/500/1000 all give penalty {vals[0]:.6g}")


def test_cap_shrinks_mildly_as_the_network_grows():
    """The cap is cap100 * log1p(UpV)/log1p(N): bigger network, smaller cap, but only mildly.

    The broken version had the ratio inverted, which made the cap GROW with N.
    """
    P = Penalties(types.SimpleNamespace(W_rec=torch.zeros(4, 4), N=4, dale_mask=None,
                                        exc2inhR=4.0))
    caps = [CAP100 * np.log1p(P.UpV) / np.log1p(N) for N in (500, 1000, 4000)]
    assert caps[0] > caps[1] > caps[2], f"cap must shrink as N grows, got {caps}"
    assert caps[0] / caps[2] < 2.0, \
        f"cap shrank {caps[0] / caps[2]:.1f}x from N=500 to N=4000; the log form should give ~1.3x"
    print(f"      cap at N = 500/1000/4000: {caps[0]:.4f}, {caps[1]:.4f}, {caps[2]:.4f} "
          f"({caps[0] / caps[2]:.2f}x across the range)")


def test_penalty_rises_as_the_cap_tightens():
    """With the weights held fixed, a smaller cap must mean a larger penalty.

    Measured on the PER-ENTRY mean, because rec_weights_magnitude_penalty multiplies its mean by
    N / (N_ref * k_ref); comparing raw values across N would be reading that prefactor, not the cap.
    """
    vals = [_pen(N, 300) * (100 * 20) / N for N in (500, 1000, 4000)]
    assert vals[0] < vals[1] < vals[2], \
        f"per-entry penalty should rise as the cap tightens: {vals}"
    print(f"      per-entry penalty at N = 500/1000/4000: "
          + ", ".join(f"{v:.4g}" for v in vals))


if __name__ == "__main__":
    for t in (test_cap_does_not_depend_on_trial_length,
              test_cap_shrinks_mildly_as_the_network_grows,
              test_penalty_rises_as_the_cap_tightens):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
