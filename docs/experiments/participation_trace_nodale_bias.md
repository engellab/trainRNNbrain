# Experiment: unconstrained RNN with trainable bias — the field-standard architecture

*This file is copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

**Sibling sweeps** (same grid, same seeds procedure — do not mix):

| Folder | Model |
|---|---|
| `CDDM_ptrack_g0` | Dale + non-negative I/O, no bias — the original reference, links to all earlier results |
| `CDDM_ptrack_g0_nodale` | unconstrained, no bias |
| `CDDM_ptrack_g0_nodale_trainablebias` | **this one**: unconstrained + trainable bias |

Trace format and metric definitions: [participation_trace.md](participation_trace.md). Constraint
switches and the E/I motivation: [participation_trace_nodale.md](participation_trace_nodale.md).
Bias range rationale and the DC-offset caveat: [participation_trace_bias.md](participation_trace_bias.md).

## Why this sweep exists

It replaces a cancelled **Dale + trainable bias** sweep (Della array `11610299`, cancelled 2026-07-25
~16:50 after 4 of 40 tasks had run 9–13 min and written no data). The reasoning: Dale's law, the
excitatory-only readout and the I/O sign clamps make the networks a biologically-motivated special
case, and a result stated only for that special case is easy to dismiss. Most of the RNN literature
trains **unconstrained networks with biases** — that is the architecture this sweep uses, so the
finding can be stated for RNNs in general rather than for a niche variant.

## Question

Does the low-activity (silent) population appear in a network with **no** architectural excuse for it?

Together with its two siblings this gives three points on the constraint ladder:

| | Dale + positive I/O | no bias | silent fraction |
|---|---|---|---|
| `CDDM_ptrack_g0` | yes | yes | known regime: 44–55% at N=1000 |
| `CDDM_ptrack_g0_nodale` | no | yes | running |
| `CDDM_ptrack_g0_nodale_trainablebias` | no | **no** (bias trainable) | running |

If silence persists across all three, no architectural constraint explains it and the claim is about
trained RNNs. If it collapses in this sweep, the phenomenon belongs to constrained networks and the
paper narrows accordingly.

## Pre-registered readout — a bias can fake participation

Unchanged from the Dale+bias design: a unit with `b = 0.3` and no task input fires at a **constant**
0.3, so its peak rate and `q_0.9(|fr|)` look healthy while `std(fr) ≈ 0` and it carries no
information. Therefore:

- Report **`std(fr)` over (time, conditions) separately from participation**, for every condition.
- Count a unit as **rescued only if its `std` rises** into the active population's range.
- A "0% silent" number based on peak rate alone is uninterpretable here.

With `frm` on, raising the bias is the cheapest way to satisfy the penalty, so expect exactly this in
the `frm`/`both` cells.

## Prediction

Silence **persists** at a broadly similar level under `none`/`rws` (~40–55%), because the apparent
driver — CDDM needing far fewer units than the network has — is untouched by removing constraints or
adding offsets. The E/I asymmetry seen in Dale nets (53–55% of E units silent vs 3.5–5% of I units)
has no analogue here by construction.

**Falsifier:** a silent fraction under `none` below ~15% would mean the constraints, not the training
dynamics, were producing the silence.

## Grid — 40 jobs

`2 equation types {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds`, N=1000, γ=0, 30000
iterations, participation tracked every 10 iterations.

Launcher: `slurm/SilentReLU_ptrack_nodalebias_gamma0_N1000_della.slurm`, run from its own git worktree
(`$HOME/trainRNNbrain_nodalebias`) with a `PYTHONPATH` guard, so sweeps already in flight keep
executing the code they started with.
