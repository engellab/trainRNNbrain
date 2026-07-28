# Experiment: large-N sweep — does the active-unit count saturate?

*Copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

Architecture and configs: [standard_rnn_reference.md](standard_rnn_reference.md). This sweep varies
**N** into the range that decides a specific objection, with no penalties.

## The question

If trained RNNs leave half their units silent, the obvious rebuttal to any rescue method is: *"why
make every unit compute — just train a bigger network and prune the silent ones. Compute is cheap."*
The answer depends entirely on how the number of **active** units grows with N.

Measured up to N=1000 (standard RNNs, h, no penalty): active = 97, 221, 340, 448 at N = 100, 250,
500, 1000. Two models fit those four points and diverge sharply beyond them:

| Model | Fit | N=2000 | N=5000 | N=10000 |
|---|---|---|---|---|
| power law | `active = 4.55·N^0.665` | 710 | 1305 | **2069** |
| saturating | `active = 749·N/(N+672)` | 561 | 660 | **702** |

**Prediction (recorded before the run):** the saturating model is closer — the local exponent already
falls monotonically (0.899 → 0.621 → 0.398), which is what a curve approaching a ceiling does, not a
power law. Expect a **task-determined ceiling of order 750 active units**. Decision thresholds set in
advance: **< ~1200 active at N=10000 confirms saturation; > ~2000 confirms growth.**

If saturation holds, pruning **cannot** deliver a large active population at any network size — the
ceiling is set by the task, not the budget — while `frm` at N=1000 already yields 1000 active units.
That is the paper's answer to the objection.

## Design — 15 jobs, h equation, no penalties, 3 seeds per cell

| Cell | N | iterations | purpose |
|---|---|---|---|
| 0 | 1000 | 5000 | truncation control vs the existing 30000-iteration N=1000 data |
| 1 | 2000 | 5000 | truncation control at a second size |
| 2 | 2000 | 30000 | full-length reference for that control |
| 3 | 5000 | 5000 | |
| 4 | 10000 | 5000 | |

**Why truncated training is legitimate here** — and why it is verified rather than assumed. The
silent fraction is decided in the first ~400–600 iterations and frozen thereafter (measured directly
in the participation traces; the N=1000 test run gave 53% silent at iteration 3000 vs ~55% at
30000). Cells 0–2 test that claim at two network sizes: if 5000 iterations reproduces the 30000-
iteration silent fraction at both N=1000 and N=2000, the N=5000 and N=10000 numbers can be read at
face value. If they do not, these runs measure an upper bound on how early the fraction settles and
must be reported as such.

## Sizing (measured, Della job `11677325`, A100)

| N | s/iter | peak GPU | 30000 iters | 5000 iters |
|---|---|---|---|---|
| 1000 | 0.144 | 6.6 GB | 1.2 h | 0.2 h |
| 2000 | 0.317 | 13.1 GB | 2.6 h | 0.4 h |
| 5000 | 1.595 | 33.1 GB | 13.3 h | 2.2 h |
| 10000 | 5.788 | **65.9 GB** | 48.2 h (infeasible) | **8.0 h** |

N=10000 requires an **80 GB** card (`--constraint=gpu80`), and 30000 iterations there would exceed
the `gpu-short` 24 h limit — hence the truncation plus its controls.

## Why `light_outputs`

At N=10000 the post-training analysis, not the training, dominates the job: `RNN_numpy` runs the
validation batch on CPU single-threaded (`OMP_NUM_THREADS=1`) for ~3 h, builds a 10.8 GB float64
trajectory array, and runs PCA over a 10000 × 135000 matrix — and the parameters would be written as
~4 GB of indented JSON per network, materialised first as Python floats. `+experiment=silent_units_largeN`
sets `light_outputs: true`, which skips every numpy analysis step and writes parameters as `.npz`.
Retained: the config, both parameter sets, the participation trace, the loss curve, and
`participation_trace.png`. The readout for this experiment is the trace, which needs none of the rest.
