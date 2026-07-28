# Experiment: does the field-standard metabolic-cost regularizer rescue silent units?

*Copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

Architecture and configs: [standard_rnn_reference.md](standard_rnn_reference.md). This sweep varies
only `lambda_met`, with all other penalties at zero.

## Why this is the load-bearing experiment

The first objection any referee raises is **"this is already known — activity regularization is
standard practice."** Metabolic-cost terms of the form `mean(fr²)` are routine in RNN-for-cognition
work. So the paper cannot simply propose another penalty; it has to say precisely what the standard
one does.

Mechanistically, `mean(fr²)` penalizes rate **magnitude** — it pushes the whole population down. It
should therefore *deepen* silence rather than relieve it, exactly as the recurrent-weight sparsity
penalty did (worse than baseline in constrained networks; no rescue in standard ones). If that holds,
the finding is not "here is another penalty" but **"the regularizer the field already uses makes this
worse, and nobody has reported it"** — a much stronger claim, and the paper's hook.

## Design — 36 jobs

`4 λ_met × 3 sizes × 3 seeds`, h equation, standard RNNs, 30000 iterations, participation tracked
every 10 iterations.

| Axis | Values |
|---|---|
| `trainer.lambda_met` | 0.01, 0.1, **1.0**, 10.0 |
| `model.N` | 100, 500, 1000 |
| seeds | 3 (`seed="random"`) |

**Sweeping λ rather than picking one is the point.** Reporting the silent fraction *as a function of*
λ removes the "you calibrated it badly" objection entirely. The bracket is set by scale: with rates
~0.3–0.5, `mean(fr²)` ≈ 0.05–0.25 against a task loss of ~0.02–0.09, so **λ = 1 is roughly equal
weight** and the four decades span negligible to task-dominating.

Comparisons already measured at identical settings: **λ=0** (`none`: 3% / 22% / 42% truly silent at
N = 100 / 500 / 1000) and the **firing-rate penalty** (`frm`: 0% at every size).

## Predictions (recorded before the run)

1. **Silence does not decrease at any λ**, and at large λ it *increases* — monotonically in λ at
   N=1000. Pavel's stronger form: at N=1000 under a literature-typical λ, most units end up dead.
2. **The effect grows with N**, as every other version of this phenomenon has.
3. Task performance degrades at large λ, so the interesting comparison is at λ where R² is still
   intact — a λ that kills the task proves nothing.

**Falsifier:** if any λ drives the silent fraction toward 0 while keeping R² ≈ 0.85, then the standard
regularizer *does* solve the problem, §2 of the paper collapses, and the contribution narrows to the
diagnosis plus the population-level analysis.

## Readout

Silent fraction (both the strict `p < 1e-6` and scale-free `p < 5%·p95` metrics) against λ, one line
per N, with `none` and `frm` as horizontal reference levels — plus R² against λ on a separate panel,
since a penalty that destroys the task is not a counterexample to anything.
