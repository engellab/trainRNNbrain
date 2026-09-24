"""Three ways to revive a dead unit without making it a copy of a live one.

WHY THESE EXIST. Measured on 2026-09-24, duplication raises the active count 2.5x at N=1000
(277 -> 705) while activity dimensionality rises only 1.4x (6.3 -> 9.1), so dimensions per active
unit FALL from 0.020 to 0.013. `dead` dropout does the reverse: 32.2 dimensions on 913 units,
0.034 each. The target is therefore added DIMENSIONS, which an exact copy cannot provide.

  orth       a random incoming row with everything the live units already span removed, so the
             new unit reads an input combination no existing unit reads
  mix        a uniformly random blend of several live units -- a twin of nobody, but still placed
             where the network is active so it fires and the gradient can hold it
  bias_kick  no donor at all: a measured bias offset unfreezes the unit and leaves its weights
             where they are, so it must find its own function

Run:  python tests/test_prune_noncopy_modes.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 40, 15, 4
DEAD = np.array([0, 3, 7, 11])


def _setup(mode, **over):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in in the given mode."""
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=1, equation_type="h", seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0
        rnn.W_inp[DEAD, :] = -5.0
    args = {"check_every": 1, "patience": 1, "active_rel": 0.05, "reinit_mode": mode,
            "copy_noise": 0.05, "utility": "participation", "utility_decay": 0.99,
            "maturity": 0, "max_replace_frac": 1.0, "mix_k": 4}
    args.update(over)
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}), prune_args=args,
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool),
        _unit_utility=torch.zeros(N), _last_replaced=torch.full((N,), -1e9))
    tr.participation_from_states_ = lambda s, **k: Trainer.participation_from_states_(tr, s, **k)
    return rnn, tr


def _states(rnn, seed=3):
    """One noise-free forward pass."""
    return rnn(torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(seed))),
               w_noise=False)[0]


def test_orth_units_read_an_unused_input_combination():
    """The new row must be orthogonal to every live unit's row IN THE NETWORK AS IT NOW STANDS.

    The comparison is against the post-replacement rows, not the pre-replacement ones. Replacement
    zeroes the revived units' outgoing columns, which changes every live unit's row at those
    entries, so a pre-replacement comparison measures orthogonality to a network that no longer
    exists and reports a spurious 0.28.
    """
    rnn, tr = _setup("orth")
    st = _states(rnn)
    Trainer.prune_and_reinit_(tr, st)
    revived = torch.nonzero(tr._reinit_ever).flatten()
    assert len(revived) > 0, "nothing was replaced"

    p = Trainer.participation_from_states_(tr, _states(rnn)).detach()
    live = torch.nonzero(~torch.isin(torch.arange(N), revived)).flatten()
    L = rnn.W_rec[live, :]
    worst = 0.0
    for i in revived.tolist():
        v = rnn.W_rec[i, :]
        cos = (L @ v).abs() / (L.norm(dim=1) * v.norm()).clamp_min(1e-12)
        worst = max(worst, float(cos.max()))
    assert worst < 1e-4, f"not orthogonal to the live population (max cos {worst:.3g})"
    print(f"      {len(revived)} units, max cosine against any surviving row {worst:.2e}")


def test_orth_falls_back_when_no_unused_direction_exists():
    """A complement exists only while the live rows do not already span R^N.

    With n_live >= N there is nothing left to be orthogonal to, and projecting anyway leaves
    numerical noise that normalisation inflates into a full-size weight row. The rule must fall
    back to a plain draw rather than emit rounding error as connectivity.
    """
    rnn, tr = _setup("orth")
    st = _states(rnn)
    # force the degenerate case: pretend almost everything is live
    tr.prune_args["active_rel"] = 0.0          # nothing counts as silent...
    tr._reinit_strikes = torch.zeros(N)
    Trainer.prune_and_reinit_(tr, st)
    assert tr._n_reinit_events == 0, "with nothing silent, nothing should be replaced"
    print("      degenerate case handled: no silent units, no replacements")


def test_mix_units_are_a_twin_of_nobody():
    """A blend must resemble the population without matching any single donor."""
    rnn, tr = _setup("mix")
    st = _states(rnn)
    before = rnn.W_rec.clone()
    p = Trainer.participation_from_states_(tr, st).detach()
    live = torch.nonzero(p >= 0.05 * torch.quantile(p, 0.95)).flatten()
    Trainer.prune_and_reinit_(tr, st)
    revived = torch.nonzero(tr._reinit_ever).flatten()
    assert len(revived) > 0, "nothing was replaced"

    worst = 0.0
    for i in revived.tolist():
        v = rnn.W_rec[i, :]
        cos = (before[live] @ v) / (before[live].norm(dim=1) * v.norm()).clamp_min(1e-12)
        worst = max(worst, float(cos.max()))
    assert worst < 0.98, f"a blended unit is effectively a copy (max cos {worst:.3f})"
    assert worst > 0.05, "blends should still resemble the live population, not be orthogonal to it"
    print(f"      max cosine against any single donor {worst:.3f} -- similar, not identical")


def test_bias_kick_unfreezes_without_touching_any_weight():
    """No donor and no weight change: only the bias moves, and the unit must start firing."""
    rnn, tr = _setup("bias_kick")
    inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))
    st = rnn(inp, w_noise=False)[0]
    w_rec_before, w_inp_before = rnn.W_rec.clone(), rnn.W_inp.clone()
    Trainer.prune_and_reinit_(tr, st)
    revived = torch.nonzero(tr._reinit_ever).flatten()
    assert len(revived) > 0, "nothing was replaced"

    assert torch.equal(rnn.W_rec, w_rec_before), "bias_kick must not change recurrent weights"
    assert torch.equal(rnn.W_inp, w_inp_before), "bias_kick must not change input weights"
    b = rnn.bias.data if isinstance(rnn.bias, torch.nn.Parameter) else rnn.bias
    assert float(b[revived].abs().min()) > 0, "no bias offset was written"

    r = torch.relu(rnn(inp, w_noise=False)[0])
    assert float(r[revived].max()) > 0, "the kicked units are still silent"
    print(f"      {len(revived)} units kicked, weights untouched, max rate now "
          f"{float(r[revived].max()):.3f}")


def test_every_mode_leaves_the_revived_units_unfrozen():
    """A revived unit must have gradient SOMEWHERE -- incoming or outgoing.

    A dead ReLU unit has exactly zero gradient on every weight, in both directions, which is what
    makes it unreachable by any loss term. Escaping that state does not require the INCOMING
    weights to have gradient immediately: a mode that zeroes the outgoing weights (orth, and
    Dohare et al.'s zero_out) leaves the unit reaching nothing, so no gradient flows back into it
    on the first step. Its outgoing weights do have gradient, because the unit now fires, so they
    grow and the incoming gradient follows. That one-step delay is exactly why Dohare et al. add a
    maturity threshold: without it, a zeroed-output unit has zero utility and is chosen again
    immediately.
    """
    for mode in ("orth", "mix", "bias_kick"):
        rnn, tr = _setup(mode)
        inp = torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(4)))
        Trainer.prune_and_reinit_(tr, rnn(inp, w_noise=False)[0])
        revived = torch.nonzero(tr._reinit_ever).flatten()
        rnn.zero_grad(set_to_none=True)
        rnn(inp, w_noise=False)[1].pow(2).mean().backward()
        g_in = rnn.W_rec.grad[revived, :].abs().max().item()
        g_out = rnn.W_rec.grad[:, revived].abs().max().item()
        assert max(g_in, g_out) > 0, f"mode {mode}: revived units are still completely frozen"
        route = "incoming" if g_in > 0 else "outgoing only (zeroed output, unfreezes next step)"
        print(f"      {mode:10s} in {g_in:.2e}  out {g_out:.2e}  -> {route}")


if __name__ == "__main__":
    for fn in [test_orth_units_read_an_unused_input_combination,
               test_orth_falls_back_when_no_unused_direction_exists,
               test_mix_units_are_a_twin_of_nobody,
               test_bias_kick_unfreezes_without_touching_any_weight,
               test_every_mode_leaves_the_revived_units_unfrozen]:
        fn()
        print(f"PASS  {fn.__name__}")
