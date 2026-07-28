# Experiment: how long until training actually stops changing the network?

*Copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

## The problem this exists to fix

**30000 iterations is not convergence, and we were treating it as if it were.** Measured on the
standard-RNN reference sweep: the training loss is flat over the final 10% of training (−0.4 to
−0.6% at every size), yet the hard-silent fraction is still rising when training stops:

| N | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| hard-silent gained in the final 5000 iterations | +0.2 pp | +0.9 pp | +2.9 pp | **+3.7 pp** | **+3.1 pp** |

The loss has converged; the network has not. It goes on quietly switching units off while task
performance stays constant — drift along a flat loss manifold.

**This is not a bookkeeping detail, it changes a conclusion.** Residual drift *grows with N*, so
larger networks sit further from their asymptote at any fixed iteration count. That undercounts
silence at large N, overcounts active units at large N, and biases the saturation curve toward
"growth" — which is precisely the result the large-N runs produced. The learning-rate scaling
`lr·(100/N)^0.333` is the obvious mechanism: a 10000-unit network needs ~2.2× as many steps to move
its weights as far as a 1000-unit one.

So the horizon gets measured at two tractable sizes **before** committing ~185 GPU-hours to
N=5000/10000 at a length already known to be too short.

## Design — 12 jobs

`2 sizes {1000, 2000} × 2 weight_decay {1e-6, 0} × 3 seeds`, h equation, no penalties,
**100000 iterations** (3.3× the previous horizon), participation tracked every 10.

**Why `weight_decay` is the second axis.** Adam's 1e-6 decay keeps pulling weights down after the
loss plateaus, and is a plausible driver of the late-phase silencing. If `wd=0` removes the drift,
the slow silencing is a regularisation artifact rather than something the task demands — a different
story from the one the paper currently tells, and cheap to check while these jobs run anyway.

## Two measurements

1. **New silent units per 1000 iterations** — derived from the participation trace, which already
   records every unit every 10 iterations. No extra logging; this is simply the derivative of a curve
   we have been saving all along.
2. **Relative parameter drift** `‖W(t) − W(t−Δ)‖_F / ‖W(t)‖_F` per tracked step, for `W_rec`,
   `W_inp`, `W_out` and `bias` — **new**, stored under the `"drift"` key of the participation-trace
   pickle. This is the direct test: the loss can be flat while the weights still move, and only this
   distinguishes the two.

`bias` drift reads 1.0 at the first snapshot as an artifact of starting at exactly zero (relative
change from a zero vector is trivially 1); it settles immediately and can be ignored.

## What the outcome decides

- **Both curves flatten well inside 100000 iterations** → read off the horizon, scale it by N, and
  size the large-N runs from measurement.
- **They are still moving at 100000** → the fixed-iteration protocol is unusable for cross-N
  comparison, and the size sweep must be re-run to a convergence *criterion* (stop when new silent
  units per 1000 iterations falls below a threshold) rather than a fixed count. Every silent-fraction
  number in the project would then carry the caveat "at 30000 iterations", not "at convergence".
- **`wd=0` removes the late drift** → the late-phase silencing is driven by weight decay, and the
  headline phenomenon needs restating in those terms.

Runtime at measured rates: N=1000 at 0.144 s/iter → 4.0 h; N=2000 at 0.317 s/iter → 8.8 h.
