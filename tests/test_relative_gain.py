"""Gain must mean the same thing for ReLU, tanh and sigmoid, or it cannot compare them.

WHY THE MEASURE EXISTS. The project counts a unit active when its rate clears a threshold, which is
a definition built around an activation whose dead state is 0. A dead ReLU unit sits at 0 and is
counted out; a dead tanh unit sits at -1 or +1 and is counted as maximally loud, so the same rule
reports a fully saturated tanh network as 1000 of 1000 working. What a dead ReLU unit and a
saturated tanh unit share is that neither passes gradient, so `Trainer.relative_gain_` measures the
activation's slope at each unit's operating point, normalised by that activation's steepest slope.

CONTRACT, fixed before running:
  1. ReLU: a unit held above threshold scores 1, a unit held at or below it scores 0, and a unit
     above threshold on a known fraction of its samples scores that fraction. This is the case
     where the right answer is arithmetic, not approximate.
  2. tanh: a unit held at h = 0 scores 1, and a unit driven far out scores ~0. The score falls
     monotonically as the drive grows.
  3. sigmoid: the same two ends, with the normaliser taking care of the fact that a sigmoid's
     steepest slope is a quarter of its own slope parameter.
  4. The normaliser is per-activation, so the MAXIMUM achievable score is 1 for every activation
     tested. Without it a sigmoid unit at its most responsive would score 0.25 and read as
     three-quarters dead next to a ReLU unit.
  5. Leaky ReLU's dead end is its leak, not zero, and the measure says so rather than reporting 0.

Run:  python tests/test_relative_gain.py
"""

import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 6, 50, 4
TOL = 0.02


def gain_of(activation_args, states):
    """Run Trainer.relative_gain_ with a given activation over given drives.

    Args:
        activation_args: dict passed to RNN_torch, e.g. {"name": "tanh", "slope": 1.0};
        states: (N, T, B) tensor of hidden drives.
    Returns:
        (N,) tensor of relative gain.
    """
    rnn = RNN_torch(N=N, activation_args=activation_args, dale=False, n_inputs=2, n_outputs=1,
                    equation_type="h", seed=0)
    tr = Trainer.__new__(Trainer)          # only relative_gain_ is exercised
    tr.RNN = rnn
    return Trainer.relative_gain_(tr, states)


def test_relu_counts_the_fraction_of_time_above_threshold():
    """CONTRACT 1 and 4. The one case with an exact arithmetic answer."""
    h = torch.zeros(N, T, B)
    h[0] = 1.0                                     # always on
    h[1] = -1.0                                    # always off
    h[2, : T // 2] = 1.0                           # on half the samples
    h[2, T // 2:] = -1.0
    h[3, : T // 4] = 3.0                           # on a quarter
    h[3, T // 4:] = -3.0
    g = gain_of({"name": "relu", "slope": 1.0}, h)
    assert abs(g[0] - 1.00) < TOL, f"always-on unit scored {g[0]:.3f}, expected 1"
    assert abs(g[1] - 0.00) < TOL, f"always-off unit scored {g[1]:.3f}, expected 0"
    assert abs(g[2] - 0.50) < TOL, f"half-on unit scored {g[2]:.3f}, expected 0.5"
    assert abs(g[3] - 0.25) < TOL, f"quarter-on unit scored {g[3]:.3f}, expected 0.25"
    assert g.max() <= 1.0 + TOL, f"a score exceeded 1: {g.max():.3f}"
    print(f"1,4 OK  relu: always-on {g[0]:.2f}, always-off {g[1]:.2f}, "
          f"half {g[2]:.2f}, quarter {g[3]:.2f}")


def test_tanh_falls_monotonically_as_the_drive_saturates():
    """CONTRACT 2 and 4. A tanh unit at 0 is maximally responsive; far out it passes nothing."""
    drives = [0.0, 0.5, 1.0, 2.0, 4.0]
    h = torch.zeros(N, T, B)
    for i, d in enumerate(drives):
        h[i] = d
    g = gain_of({"name": "tanh", "slope": 1.0}, h)
    vals = [float(g[i]) for i in range(len(drives))]
    assert abs(vals[0] - 1.0) < TOL, f"tanh at h=0 scored {vals[0]:.3f}, expected 1"
    assert vals[-1] < 0.01, f"tanh at h=4 scored {vals[-1]:.4f}, expected ~0"
    assert all(a > b for a, b in zip(vals, vals[1:])), f"not monotone: {vals}"
    print("2,4 OK  tanh at drive " + ", ".join(f"{d}:{v:.3f}" for d, v in zip(drives, vals)))


def test_sigmoid_is_rescaled_to_its_own_best_slope():
    """CONTRACT 3 and 4. Without the normaliser a sigmoid's best unit would score 0.25."""
    h = torch.zeros(N, T, B)
    h[1] = 6.0
    g = gain_of({"name": "sigmoid", "slope": 1.0}, h)
    assert abs(g[0] - 1.0) < TOL, f"sigmoid at h=0 scored {g[0]:.3f}, expected 1"
    assert g[1] < 0.02, f"sigmoid at h=6 scored {g[1]:.3f}, expected ~0"
    print(f"3,4 OK  sigmoid: at h=0 {g[0]:.3f}, at h=6 {g[1]:.4f}")


def test_leaky_relu_reports_its_leak_not_zero():
    """CONTRACT 5. A leaky unit below threshold still passes its leak, and the measure shows it.

    The activation is "leaky_relu" with a "leak_slope", NOT relu with a "leak". The first version
    of this test passed {"name": "relu", "leak": 0.01}, which configure_activation_ reads as a
    plain ReLU: it scored exactly 0 and the assertion "far below 1" passed on a premise that was
    never set up. A test that cannot tell a leaky unit from a dead one is not testing the leak.
    """
    h = torch.full((N, T, B), -1.0)
    g = gain_of({"name": "leaky_relu", "slope": 1.0, "leak_slope": 0.01}, h)
    plain = gain_of({"name": "relu", "slope": 1.0}, h)
    assert abs(float(g[0]) - 0.01) < 0.002, \
        f"leaky unit below threshold scored {float(g[0]):.4f}, expected its leak of 0.01"
    assert float(plain[0]) == 0.0, f"plain relu below threshold scored {float(plain[0]):.4f}"
    print(f"5 OK    below threshold: leaky {float(g[0]):.4f} (its leak), plain relu "
          f"{float(plain[0]):.4f}")


if __name__ == "__main__":
    test_relu_counts_the_fraction_of_time_above_threshold()
    test_tanh_falls_monotonically_as_the_drive_saturates()
    test_sigmoid_is_rescaled_to_its_own_best_slope()
    test_leaky_relu_reports_its_leak_not_zero()
    print("relative gain means the same thing for every activation tested")
