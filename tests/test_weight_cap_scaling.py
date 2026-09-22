"""A weight cap must depend on the network, never on how long a trial is.

`inp_weights_magnitude_penalty` read `N, U = states.size(0), states.size(1)`, so U was the TRIAL
LENGTH and the cap came out as cap100 * (T/N) * log1p(N)/log1p(T). The same 1000-unit network then
got a cap of 0.182 on a T=300 task and 0.278 on a T=500 one, and the log ratio was inverted relative
to out_weights_magnitude_penalty so the cap scaled the wrong way with N as well. `U` was meant to be
`self.UpV`, the hard constant 100 its siblings use.

Run:  python tests/test_weight_cap_scaling.py
"""
import types

import numpy as np
import torch

from trainRNNbrain.trainer.Trainer import Penalties


def _pen(N, T, w=0.5, n_inputs=3):
    """Penalty value for a network of N units on a T-step trial, with every |W_inp| equal to w."""
    P = Penalties(types.SimpleNamespace(W_inp=torch.full((N, n_inputs), w)))
    return float(P.inp_weights_magnitude_penalty(torch.zeros(N, T, 4), cap100=0.5))


def test_cap_does_not_depend_on_trial_length():
    """Trial length is a property of the TASK, not of the weights. It must not enter the cap."""
    vals = [_pen(1000, T) for T in (200, 300, 500, 1000)]
    assert max(vals) - min(vals) < 1e-9, \
        f"penalty moved with trial length: {vals} -- the cap has picked up T again"


def test_cap_scales_with_N_the_same_way_as_its_sibling():
    """Both weight-magnitude penalties cap at cap100 * log1p(UpV)/log1p(N), so a bigger network
    gets a SMALLER cap, mildly. The broken version had the ratio inverted."""
    P = Penalties(types.SimpleNamespace(W_inp=torch.full((1000, 3), 0.5),
                                        W_out=torch.full((3, 1000), 0.5), N=1000))
    caps = [0.5 * np.log1p(P.UpV) / np.log1p(N) for N in (500, 1000, 4000)]
    assert caps[0] > caps[1] > caps[2], "cap must shrink as N grows"
    assert caps[0] / caps[2] < 2.0, \
        f"cap shrank {caps[0]/caps[2]:.1f}x from N=500 to N=4000; the log form should give ~1.3x"
    # and the penalty must rise as the cap tightens, with the weights held fixed
    rising = [_pen(N, 300) for N in (500, 1000, 4000)]
    assert rising[0] < rising[1] < rising[2], f"penalty should rise as the cap tightens: {rising}"


if __name__ == "__main__":
    test_cap_does_not_depend_on_trial_length()
    test_cap_scales_with_N_the_same_way_as_its_sibling()
    print("both checks passed")
