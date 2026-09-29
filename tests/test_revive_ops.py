"""Two biologically-motivated revival operations, against the measurements that motivate them.

WHY THESE TWO. Every rule tested so far rebalances or rescales the weights a silent unit already
has, and every one of them collapsed dimensionality: control 5.53 directions, synaptic scaling
5.14 (1.62 pushed hard), rescale-at-floor 4.34, rescale-with-target 2.0-2.4. They homogenise,
because they drive every unit toward a common activity level using whatever weak, unstructured
input it happens to have. Two measurements say what such rules cannot fix:

  - a silent unit is NOT strongly inhibited. Its excitation AND its inhibition from the firing
    population are both 3-9x weaker than an active unit's, so rebalancing the whole row spends most
    of the effort on synapses that carry no current.
  - its INPUT weights are 21x smaller than an active unit's, so in the place that matters most
    there is often nothing left to rescale.

  disinhibit      weaken only the synapses actually holding the unit down, ranked by their
                  contribution to the drive W[i,j]*r_j rather than by weight magnitude. Sparse and
                  unit-specific, so it should homogenise less than a whole-row rebalance.
  synaptogenesis  GROW new excitatory synapses from firing units. Structural rather than
                  multiplicative: it creates input where the unit has none. With syn_novel it
                  connects to the firing units a unit listens to LEAST, so different revived units
                  read different parts of the active population -- aimed squarely at the
                  homogenisation that sank the rescaling family.

CONTRACT, fixed before running:
  1. disinhibit weakens exactly the strongest inhibitory DRIVE contributors, not the largest
     inhibitory weights, and touches at most k synapses per unit.
  2. disinhibit never touches excitatory weights.
  3. synaptogenesis adds positive weight onto firing units only, and to at most syn_m of them.
  4. its partners are fixed within an episode and redrawn when a new episode opens.
  5. syn_novel picks partners with weaker existing weights than the default rule does.

Run:  python tests/test_revive_ops.py   (or under pytest)
"""
import types

import numpy as np
import torch

from trainRNNbrain.rnns.RNN_torch import RNN_torch
from trainRNNbrain.trainer.Trainer import Trainer

N, T, B = 200, 20, 4
DEAD = np.arange(0, 60)


def _setup(op, **over):
    """A ReLU RNN with the DEAD units wired silent, plus a Trainer stand-in running `op`.

    Args:
        op: str, one of "rescale", "disinhibit", "synaptogenesis".
        **over: prune_args overrides.
    Returns: (RNN_torch, trainer stand-in).
    """
    rnn = RNN_torch(N=N, activation_args={"name": "relu", "slope": 1.0}, dale=False,
                    n_inputs=2, n_outputs=3, equation_type="h", self_connections=True, seed=0)
    with torch.no_grad():
        rnn.W_rec[DEAD, :] = -5.0 * torch.rand(len(DEAD), N,
                                               generator=torch.Generator().manual_seed(1))
        rnn.W_inp[DEAD, :] = -5.0
        # the dead units need SOME excitation, or "excitation is untouched" has nothing to check
        # and "grow onto firing units" has no existing weights to compare novelty against
        rnn.W_rec[DEAD[:, None], np.arange(80, 140)[None, :]] = 0.02
    args = {"check_every": 1, "patience": 1, "active_rel": 0.05, "reinit_mode": "rescale",
            "revive_op": op, "rescale_alpha": 1.5, "rescale_normalize": True,
            "rescale_cap": 1e9, "rescale_target_frac": 2.5, "rescale_protect_frac": 1.0,
            "disinhibit_k": 10, "syn_m": 8, "syn_step": 0.05, "syn_novel": False,
            "maturity": 0, "max_replace_frac": 1.0}
    args.update(over)
    tr = types.SimpleNamespace(
        RNN=rnn, iter_n=0, optimizer=types.SimpleNamespace(state={}),
        Penalties=types.SimpleNamespace(UpV=100), prune_reinit=True,
        frm_args={"cap_fr": 0.3, "tau": 0.1}, prune_args=args,
        _reinit_strikes=torch.zeros(N), _n_reinit_events=0,
        _reinit_ever=torch.zeros(N, dtype=torch.bool), _unit_utility=torch.zeros(N),
        _last_replaced=torch.full((N,), -1e9), _rescale_cum=torch.ones(N),
        _syn_partners=torch.full((N, 32), -1, dtype=torch.long),
        _rescale_growing=torch.zeros(N, dtype=torch.bool), _rescale_episodes=torch.zeros(N),
        _rescale_refract_until=torch.zeros(N))
    tr.participation_from_states_ = lambda s, **k: Trainer.participation_from_states_(tr, s, **k)
    tr.rescale_rows_ = lambda i, al, nm, live=None, norm_ref=None: Trainer.rescale_rows_(
        tr, i, al, nm, live, norm_ref)
    tr.disinhibit_rows_ = lambda i, al, r, k: Trainer.disinhibit_rows_(tr, i, al, r, k)
    tr.grow_synapses_ = lambda i, st, r, lv, m, nv: Trainer.grow_synapses_(tr, i, st, r, lv, m, nv)
    tr.frm_activity_cap_ = lambda: Trainer.frm_activity_cap_(tr)
    tr.frm_activity_ = lambda s: Trainer.frm_activity_(tr, s)
    return rnn, tr


def _inputs():
    """The fixed probe batch."""
    return torch.abs(torch.randn(2, T, B, generator=torch.Generator().manual_seed(3)))


def _step(rnn, tr, inp):
    """One forward pass and one revival event; returns the pre-event weights and the rates."""
    tr.iter_n += 1
    states, _ = rnn(inp, w_noise=False)
    before = rnn.W_rec.detach().clone()
    rbar = torch.relu(states.detach()).reshape(N, -1).mean(dim=1)
    Trainer.prune_and_reinit_(tr, states)
    return before, rbar


def test_disinhibit_weakens_the_strongest_drive_contributors():
    """Ranked by W*r, not by |W| -- a big weight from a silent partner holds nothing down."""
    rnn, tr = _setup("disinhibit")
    inp = _inputs()
    before, rbar = _step(rnn, tr, inp)
    g = torch.nonzero(tr._rescale_growing).flatten()
    assert g.numel() > 0, "nothing is growing"
    i = int(g[0])
    changed = torch.nonzero(~torch.isclose(rnn.W_rec[i, :].detach(), before[i, :],
                                           atol=1e-9)).flatten()
    changed = changed[changed != i]                      # own column is redrawn at episode open
    changed = torch.tensor([c for c in changed.tolist() if not bool(tr._rescale_growing[c])])
    assert 0 < changed.numel() <= tr.prune_args["disinhibit_k"], \
        f"{changed.numel()} synapses changed, cap is {tr.prune_args['disinhibit_k']}"
    contrib = before[i, :] * rbar
    rank_by_drive = set(torch.topk(-contrib, 40).indices.tolist())
    hit = sum(1 for c in changed.tolist() if c in rank_by_drive)
    assert hit == changed.numel(), \
        f"only {hit} of {changed.numel()} changed synapses are top drive contributors"
    print(f"      unit {i}: {changed.numel()} synapses weakened, all among the strongest "
          f"inhibitory DRIVE contributors")


def test_disinhibit_never_touches_excitation():
    """It is one-sided by construction: only inhibitory synapses are weakened."""
    rnn, tr = _setup("disinhibit")
    inp = _inputs()
    before, _ = _step(rnn, tr, inp)
    g = torch.nonzero(tr._rescale_growing).flatten()
    i = int(g[0])
    pos = (before[i, :] > 0)
    pos[i] = False
    for c in torch.nonzero(tr._rescale_growing).flatten().tolist():
        pos[c] = False                                   # their columns were redrawn
    d = float((rnn.W_rec[i, pos].detach() - before[i, pos]).abs().max())
    assert d == 0.0, f"an excitatory weight moved by {d:.3g}"
    print(f"      unit {i}: {int(pos.sum())} excitatory synapses untouched")


def test_synaptogenesis_grows_onto_firing_units_only():
    """New input comes from units that actually emit something."""
    rnn, tr = _setup("synaptogenesis")
    inp = _inputs()
    states, _ = rnn(inp, w_noise=False)
    p = tr.participation_from_states_(states).detach()
    silent = p < 0.05 * torch.quantile(p, 0.95)
    before, _ = _step(rnn, tr, inp)
    g = torch.nonzero(tr._rescale_growing).flatten()
    i = int(g[0])
    grown = torch.nonzero(rnn.W_rec[i, :].detach() - before[i, :] > 1e-9).flatten()
    grown = torch.tensor([c for c in grown.tolist() if not bool(tr._rescale_growing[c])])
    assert 0 < grown.numel() <= tr.prune_args["syn_m"], \
        f"{grown.numel()} synapses grown, cap is {tr.prune_args['syn_m']}"
    assert bool((~silent[grown]).all()), "a synapse was grown from a silent unit"
    print(f"      unit {i}: {grown.numel()} synapses grown, every source firing")


def test_synaptogenesis_partners_are_fixed_within_an_episode():
    """A stable identity, not a new random projection every step."""
    rnn, tr = _setup("synaptogenesis")
    inp = _inputs()
    _step(rnn, tr, inp)
    g = torch.nonzero(tr._rescale_growing).flatten()
    i = int(g[0])
    first = tr._syn_partners[i, :tr.prune_args["syn_m"]].clone()
    assert int((first >= 0).sum()) == tr.prune_args["syn_m"], "partners were not recorded"
    _step(rnn, tr, inp)
    assert torch.equal(tr._syn_partners[i, :tr.prune_args["syn_m"]], first), \
        "partners changed inside an episode"
    print(f"      unit {i}: partners {first[:4].tolist()}... unchanged across two events")


def test_novel_partners_are_ones_the_unit_listens_to_least():
    """The anti-homogenisation variant connects where there is no connection."""
    rnn_d, tr_d = _setup("synaptogenesis", syn_novel=False)
    rnn_n, tr_n = _setup("synaptogenesis", syn_novel=True)
    inp = _inputs()
    out = {}
    for tag, rnn, tr in (("loudest", rnn_d, tr_d), ("novel", rnn_n, tr_n)):
        before, _ = _step(rnn, tr, inp)
        g = torch.nonzero(tr._rescale_growing).flatten()
        i = int(g[0])
        cols = tr._syn_partners[i, :tr.prune_args["syn_m"]]
        out[tag] = float(before[i, cols].abs().median())
    assert out["novel"] < out["loudest"], \
        f"novel picked |w| {out['novel']:.4g}, loudest picked {out['loudest']:.4g}"
    print(f"      median existing |weight| onto chosen partners: novel {out['novel']:.4g} "
          f"< loudest {out['loudest']:.4g}")


def test_refractory_tail_keeps_protection_after_graduation():
    """The unit is carried through the window where it is otherwise lost.

    The units this rule fails on end with MORE inhibition from firing sources than the control's
    silent units, although the rule can only divide that inhibition down -- so the gradient added it
    after graduation, when protection lifts. A refractory tail keeps the incoming rows out of the
    gradient for a while longer, while the outgoing weights, never protected, go on looking for a
    use for the unit.

    Contract, fixed before running:
      1. With rescale_refractory = 0 a graduated unit loses protection immediately.
      2. With it > 0 the unit keeps protection for that many steps, then loses it.
      3. A unit inside its tail does not open a new episode.
    """
    for refr, still_protected in ((0, False), (50, True)):
        rnn, tr = _setup("rescale", rescale_refractory=refr, rescale_protect_frac=1.0)
        inp = _inputs()
        _step(rnn, tr, inp)
        g = torch.nonzero(tr._rescale_growing).flatten()
        assert g.numel() > 0, "nothing is growing"
        i = int(g[0])
        with torch.no_grad():                       # push it over the target
            rnn.W_rec[i, :] = 0.0
            rnn.W_inp[i, :] = 8.0
        _step(rnn, tr, inp)
        assert not bool(tr._rescale_growing[i]), "the unit did not graduate"
        rnn.W_rec.grad = torch.ones(N, N)
        Trainer.zero_protected_grads_(tr)
        protected = float(rnn.W_rec.grad[i, :].abs().max()) == 0.0
        assert protected == still_protected, \
            f"refractory={refr}: graduated unit {'is' if protected else 'is not'} protected"
        if still_protected:
            tr.iter_n += refr + 1                   # walk past the end of the tail
            rnn.W_rec.grad = torch.ones(N, N)
            Trainer.zero_protected_grads_(tr)
            assert float(rnn.W_rec.grad[i, :].abs().max()) == 1.0, \
                "protection outlasted the refractory window"
        print(f"      refractory={refr}: protected right after graduation = {protected}"
              + (", released once the window closed" if still_protected else ""))


if __name__ == "__main__":
    for t in (test_disinhibit_weakens_the_strongest_drive_contributors,
              test_disinhibit_never_touches_excitation,
              test_synaptogenesis_grows_onto_firing_units_only,
              test_synaptogenesis_partners_are_fixed_within_an_episode,
              test_novel_partners_are_ones_the_unit_listens_to_least,
              test_refractory_tail_keeps_protection_after_graduation):
        print(f"\n{t.__name__}")
        t()
    print("\nall checks passed")
