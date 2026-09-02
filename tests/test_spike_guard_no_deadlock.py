"""Calibration of the gradient-spike guard: it must catch catastrophes without eating normal steps.

Models the ACCEPT/SKIP decision only — the properties under test are properties of the
reference-tracking rule, not of train_step.

⚠️ HISTORY. The first guard tracked an EMA updated ONLY on accepted steps, with spike_factor=100.
In production it skipped 99.8% of updates (798 rollbacks x 500 consecutive skips out of 400,000
iterations) across 11 runs that all finished with negative r2 — including `none` cells with no
penalty at all, and frm k=1 which is the easiest cell in the grid. A synthetic i.i.d. model does
NOT reproduce that exact 99.8% figure, so the precise trigger is not fully characterised; what IS
established is that a reference which updates only on acceptance can go stale indefinitely, and
that factor=100 fires on a large fraction of ordinary gradients once they are heavy-tailed.
Both are fixed here: the median updates on EVERY step, and the factor only admits catastrophes.
"""
import numpy as np
from collections import deque

WINDOW, MIN_SAMPLES = 1000, 100
FACTOR = 1e6                      # must match Trainer's spike_factor default


def skip_rate(norms, factor=FACTOR):
    """Fraction of steps the shipped rule would skip."""
    win, sk = deque(maxlen=WINDOW), 0
    for n in norms:
        win.append(n)                                     # EVERY step, accepted or not
        ref = float(np.median(win)) if len(win) >= MIN_SAMPLES else None
        if ref is not None and ref > 0 and n > factor * ref:
            sk += 1
    return sk / len(norms)


def test_does_not_eat_normal_steps_even_when_heavy_tailed():
    """An frm run near instability produces gradients spanning many orders of magnitude."""
    rng = np.random.default_rng(0)
    for sigma in (1.0, 3.0, 5.0):
        r = skip_rate(rng.lognormal(0.0, sigma, 20000))
        assert r < 0.01, f"sigma={sigma}: skipped {r:.2%} of ordinary steps (factor={FACTOR:g})"


def test_still_catches_catastrophes():
    """Observed catastrophes were ~1e27 against a normal ~1e-3. 1e12 here is deliberately mild."""
    rng = np.random.default_rng(1)
    x = rng.lognormal(0.0, 3.0, 20000)
    idx = rng.choice(len(x), 50, replace=False)
    x[idx] *= 1e12
    win, caught, ids = deque(maxlen=WINDOW), 0, set(idx)
    for i, n in enumerate(x):
        win.append(n)
        ref = float(np.median(win)) if len(win) >= MIN_SAMPLES else None
        if ref is not None and n > FACTOR * ref and i in ids:
            caught += 1
    assert caught >= 48, f"only {caught}/50 catastrophes caught — guard too permissive"


def test_reference_cannot_go_stale():
    """A sustained 1000x regime shift is the new normal, not a spike. The OLD rule could stick."""
    rng = np.random.default_rng(2)
    norms = np.concatenate([rng.lognormal(0.0, 0.3, 5000),
                            rng.lognormal(0.0, 0.3, 15000) * 1000.0])
    assert skip_rate(norms) < 0.01, "median failed to follow a sustained regime shift"


def test_a_single_tiny_gradient_cannot_poison_the_run():
    """The historical seeding hazard: one anomalously small norm arriving first."""
    rng = np.random.default_rng(3)
    norms = np.concatenate([[1e-12], rng.lognormal(0.0, 0.3, 20000)])
    assert skip_rate(norms) < 0.01, "a single tiny gradient still poisons the reference"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn(); print(f"PASS {name}")
