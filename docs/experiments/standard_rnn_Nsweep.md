# Experiment: network-size sweep in STANDARD RNNs

*Copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

Architecture, configs and rationale: [standard_rnn_reference.md](standard_rnn_reference.md). This
sweep varies only **N**, with `λ_rws = 0` throughout.

## Question

In Dale-constrained networks the silent fraction scaled steeply with network size — **0% at N=100,
24–38% at N=500, 44–56% at N=1000**. That scaling is the single most important descriptive fact in
the project, and the one that most constrains interpretation: it is also what a reviewer will point
to when arguing the effect is just **spare capacity** (a 1000-unit network on a low-dimensional task
trivially has units to spare, and unpenalized nets solve CDDM with an effective ~60–150 units at any
N). Does the same scaling hold once every architectural constraint is removed?

The `frm` arm at each size additionally tests whether the rescue is size-independent, as it was in
Dale networks (0% silent at every N).

## Prediction

The scaling persists: near-0% at N=100 rising to ~50% at N=1000. `frm` gives 0% at every size.

**Falsifier:** a flat silent fraction across N would break the "the network recruits what it needs"
reading and mean something size-independent is at work — a different paper.

## Grid — 60 jobs

`3 sizes {100, 250, 500} × 2 equation types {h, s} × 2 λ_frm {0, 0.2} × 5 seeds`, 30000 iterations,
participation logged every 10 iterations. **N = 1000 comes from the `CDDM_std_g0` reference sweep**
(its `λ_rws = 0` cells), completing the curve at four sizes.

`lr` is rescaled at runtime as `lr × (100/N)^0.333`, unchanged from the original 120-net size sweep,
so the new curve is directly comparable to the Dale one. Consequence to state in Methods: N and lr
co-vary by design.

Launcher `slurm/SilentReLU_std_gamma0_Nsweep_della.slurm`, own git worktree
(`$HOME/trainRNNbrain_stdN`).
