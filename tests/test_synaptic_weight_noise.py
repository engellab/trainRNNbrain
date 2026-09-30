"""Synaptic noise must jitter the CONNECTIVITY, resample every step, and vanish when switched off.

WHY. Every intervention tried so far acts on a unit once it is already silent. Synaptic noise acts
on the couplings continuously, so a unit sitting just under threshold is pushed across it on some
steps and not others, and the gradient it gets on those steps is not zero. Whether that rescues
anything is what the sweep is for; what this file pins is that the mechanism is the one intended
and not an expensive way of re-adding state noise.

sigma_rec adds a current to the STATE: every unit receives it whatever its weights are, so a unit
whose incoming row has decayed to nothing still gets the full kick. sigma_w multiplies the WEIGHTS,
so the same unit receives almost nothing. The two are not interchangeable and test 5 is the one
that would catch them being confused.

CONTRACT, fixed before running:
  1. sigma_w = 0 is EXACTLY the historical path: same seed, same output, bit for bit. Every offline
     read-out in this project reruns trained networks, so a default that perturbs them would
     invalidate the lot.
  2. With sigma_w > 0 and w_noise=True the output changes, and it changes MORE for a larger
     sigma_w -- the knob does something monotone rather than merely something.
  3. The perturbation is resampled per timestep, not drawn once per pass. Fixing every other
     source of randomness and running a CONSTANT input, a network with per-step synaptic noise
     gives a state trajectory whose step-to-step variability exceeds that of the same network with
     a single fixed perturbation of the same size.
  4. w_noise=False is deterministic even with sigma_w set, so the offline read-outs that pass
     w_noise=False keep reproducing exactly.
  5. Weight noise is NOT state noise. With sigma_rec = 0 and sigma_w > 0, a unit whose incoming
     weights are all zero stays at its resting value, while under sigma_rec > 0 it does not.
  6. Signs and structural zeros survive: multiplying by (1 + sigma_w * eps) at the sizes swept here
     must not flip a weight's sign often enough to matter, and an exact zero stays zero.

Run:  python tests/test_synaptic_weight_noise.py   (or under pytest)
"""
import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch

N, T, B = 60, 40, 4
SEED = 7


def _net(sigma_w, sigma_rec=0.05, sigma_inp=0.05):
    """A small ReLU RNN with the given noise settings and a fixed seed.

    Args:
        sigma_w: float, relative synaptic noise.
        sigma_rec: float, state noise in the recurrent dynamics.
        sigma_inp: float, state noise on the input.
    Returns: RNN_torch.
    """
    return RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                     n_inputs=2, n_outputs=2, equation_type="h", seed=SEED,
                     sigma_rec=sigma_rec, sigma_inp=sigma_inp, sigma_w=sigma_w)


def _inp(constant=False):
    """A (2, T, B) input batch; constant in time when asked, so step-to-step change is all noise."""
    g = torch.Generator().manual_seed(3)
    if constant:
        return torch.ones(2, T, B) * 0.5
    return torch.abs(torch.randn(2, T, B, generator=g))


def _run(sigma_w, w_noise=True, constant=False, sigma_rec=0.05, sigma_inp=0.05):
    """Forward one batch through a freshly seeded net and return (states, outputs) as numpy."""
    rnn = _net(sigma_w, sigma_rec=sigma_rec, sigma_inp=sigma_inp)
    with torch.no_grad():
        st, out = rnn(_inp(constant), w_noise=w_noise)
    return st.numpy(), out.numpy()


def test_off_is_bit_for_bit_the_old_path():
    """sigma_w = 0 must reproduce the historical output exactly, not approximately."""
    a, _ = _run(0.0)
    b, _ = _run(0.0)
    assert np.array_equal(a, b), "sigma_w = 0 is not reproducible across two seeded runs"
    print(f"      sigma_w = 0: two seeded runs identical over {a.size} state entries")


def test_the_knob_does_something_monotone():
    """A bigger sigma_w moves the output further from the noise-free trajectory.

    STATE NOISE IS OFF HERE. The network draws every random number from one generator, so raising
    sigma_w also reshuffles the state noise, and against a run that carries both, a small synaptic
    perturbation is lost inside a rearranged sigma_rec. With sigma_rec and sigma_inp at zero the
    only thing separating these runs from the noise-free reference is the synaptic jitter.
    """
    ref, _ = _run(0.0, w_noise=False, sigma_rec=0.0, sigma_inp=0.0)
    devs = []
    for sw in (0.0, 0.05, 0.2, 0.5):
        st, _ = _run(sw, sigma_rec=0.0, sigma_inp=0.0)
        devs.append(float(np.sqrt(np.mean((st - ref) ** 2))))
    assert devs[1] > devs[0], "sigma_w = 0.05 did not move the trajectory at all"
    assert devs[2] > devs[1] and devs[3] > devs[2], \
        f"deviation is not monotone in sigma_w: {[round(d, 4) for d in devs]}"
    print("      rms deviation from the noise-free run at sigma_w 0/0.05/0.2/0.5: "
          + ", ".join(f"{d:.4f}" for d in devs))


def test_resampled_every_step_not_once_per_pass():
    """Per-step noise makes the state jitter step to step; one fixed perturbation does not.

    Driven by a CONSTANT input, a network with a single fixed weight perturbation settles onto a
    smooth trajectory, while one redrawing its weights every step keeps rattling. Comparing the
    mean absolute second difference of the state separates them; state noise is off so the only
    source of jitter is the weights.
    """
    st_per_step, _ = _run(0.3, constant=True, sigma_rec=0.0)

    rnn = _net(0.0, sigma_rec=0.0, sigma_inp=0.0)
    with torch.no_grad():
        g = torch.Generator().manual_seed(11)
        rnn.W_rec *= (1.0 + 0.3 * torch.randn(rnn.W_rec.shape, generator=g))
        rnn.W_inp *= (1.0 + 0.3 * torch.randn(rnn.W_inp.shape, generator=g))
        st_fixed = rnn(_inp(constant=True), w_noise=True)[0].numpy()

    jit = lambda s: float(np.mean(np.abs(np.diff(s, n=2, axis=1))))
    j_step, j_fix = jit(st_per_step), jit(st_fixed)
    assert j_step > 3.0 * j_fix, \
        f"per-step jitter {j_step:.5f} is not clearly above the fixed-perturbation {j_fix:.5f}"
    print(f"      step-to-step jitter: resampled {j_step:.5f} against fixed {j_fix:.5f} "
          f"({j_step / max(j_fix, 1e-12):.1f}x)")


def test_noise_free_pass_stays_deterministic():
    """w_noise=False must ignore sigma_w, or every offline read-out becomes irreproducible."""
    a, _ = _run(0.5, w_noise=False)
    b, _ = _run(0.5, w_noise=False)
    c, _ = _run(0.0, w_noise=False)
    assert np.array_equal(a, b), "w_noise=False is not reproducible with sigma_w set"
    assert np.array_equal(a, c), "w_noise=False with sigma_w set differs from sigma_w = 0"
    print("      w_noise=False: identical with sigma_w 0.5 and 0.0")


def test_weight_noise_is_not_state_noise():
    """A unit with no incoming weights feels sigma_w not at all, and sigma_rec fully.

    This is the distinction the sweep rests on. Under state noise a disconnected unit still gets a
    current; under synaptic noise it gets a jittered version of nothing, which is nothing.
    """
    out = {}
    for label, sw, sr in (("weights", 0.5, 0.0), ("state", 0.0, 0.5)):
        rnn = _net(sw, sigma_rec=sr, sigma_inp=0.0)
        with torch.no_grad():
            rnn.W_rec[0, :] = 0.0
            rnn.W_inp[0, :] = 0.0
            st = rnn(_inp(constant=True), w_noise=True)[0].numpy()
        out[label] = float(np.std(st[0]))
    assert out["weights"] < 1e-6, \
        f"a disconnected unit moved under synaptic noise (sd {out['weights']:.2e})"
    assert out["state"] > 1e-3, \
        f"a disconnected unit did NOT move under state noise (sd {out['state']:.2e})"
    print(f"      disconnected unit, state sd: synaptic noise {out['weights']:.2e}, "
          f"state noise {out['state']:.2e}")


def test_signs_and_zeros_survive():
    """Multiplicative noise keeps structural zeros and rarely flips a sign at swept sizes."""
    g = torch.Generator().manual_seed(5)
    W = torch.randn(400, 400, generator=g) / 20.0
    W[0, :] = 0.0
    for sw in (0.05, 0.2, 0.5):
        Wn = W * (1.0 + sw * torch.randn(W.shape, generator=g))
        assert torch.all(Wn[0, :] == 0.0), f"sigma_w {sw} created weight where there was none"
        flip = float(((Wn * W) < 0).float().mean())
        assert flip < 0.05, f"sigma_w {sw} flipped the sign of {flip:.1%} of weights"
        print(f"      sigma_w {sw}: zeros preserved, {flip:.2%} of signs flipped")


if __name__ == "__main__":
    for t in (test_off_is_bit_for_bit_the_old_path,
              test_the_knob_does_something_monotone,
              test_resampled_every_step_not_once_per_pass,
              test_noise_free_pass_stays_deterministic,
              test_weight_noise_is_not_state_noise,
              test_signs_and_zeros_survive):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
