## 2026-07-28 (10:40) — 30000 iterations is not convergence: the horizon experiment (submitted)

### The problem

Every silent-unit number in this project was measured at a **fixed 30000 iterations**, on the
assumption that training had settled. Checking that assumption directly shows it is false, and in a
way that matters for a conclusion we already drew.

**The training loss has converged. The network has not.** Across the standard-RNN reference sweep,
the loss over the last 10% of training differs from the preceding 10% by only −0.4 to −0.6% at every
size — flat. But the hard-silent fraction is still climbing when training stops:

| N | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| loss change (last 10% vs previous 10%) | −0.46% | −0.38% | −0.47% | −0.56% | −0.61% |
| **hard-silent gained in the final 5000 iterations** | +0.20 pp | +0.92 pp | +2.86 pp | **+3.67 pp** | **+3.08 pp** |

Tracing one condition (standard, N=1000, no penalty) across the whole run makes the scale of it
plain — the silent fraction roughly **quintuples** after iteration 5000:

| iteration | 0 | 1000 | 5000 | 10000 | 20000 | 30000 |
|---|---|---|---|---|---|---|
| hard silent (`p < 1e-6`) | 4.4% | 5.4% | 7.4% | 13.9% | 30.9% | **41.5%** |
| `p < 0.01` | 21.2% | 23.6% | 28.9% | 37.3% | 48.0% | **54.7%** |
| scale-free | 21.3% | 29.0% | 39.8% | 47.4% | 52.2% | **58.1%** |

The network solves the task early and then goes on quietly switching units off — drift along a flat
loss manifold.

### Two corrections this forces

1. **"The bifurcation completes by iteration ~400–600 and is frozen thereafter" was wrong.** That
   claim came from the 3000-iteration test run, where the curve does look flat after ~600 *within
   that window*. Extrapolating a 3000-iteration window to 30000 was the error, and it propagated
   into the design of the large-N runs (which is why their truncation control failed). Any statement
   of the form "silencing is decided early" should be struck.
2. **"0% of units are silent at initialisation" is Dale-specific.** In standard unconstrained
   networks, **4.4% are hard-silent and 21% sit below the 0.01 line before a single training step**.
   With signed input weights a unit can start net-negative and never fire; I/O positivity is what
   guaranteed the old result. Training still creates most of the silence (4.4% → 41.5%), but not all
   of it.

### Why it changes the saturation conclusion, not just the bookkeeping

Residual drift **grows with N** (+0.2 pp at N=100 → +3.7 pp at N=1000). Larger networks therefore
sit further from their asymptote at any fixed iteration count, which **undercounts silence at large
N**, **overcounts active units at large N**, and biases the size curve toward "growth" — exactly the
result the large-N runs produced. The learning-rate scaling `lr·(100/N)^0.333` is the obvious
mechanism: a 10000-unit network needs ~2.2× as many steps to move its weights as far as a 1000-unit
one. So the evidence against saturation may be an artifact of undertrained large networks, and
spending ~185 GPU-hours on N=5000/10000 at 30000 iterations would buy a differently-wrong answer.

### The experiment (submitted — Della array `11706899`, commit `f5aa558`)

**12 jobs = 2 sizes × 2 weight-decay values × 3 seeds**, h equation, no penalties, standard RNNs,
**100000 iterations** (3.3× the previous horizon), participation tracked every 10:

| Tasks | N | `weight_decay` | seeds |
|---|---|---|---|
| 1–3 | 1000 | 1e-6 | 3 |
| 4–6 | 1000 | **0** | 3 |
| 7–9 | 2000 | 1e-6 | 3 |
| 10–12 | 2000 | **0** | 3 |

Runtime 4.0 h (N=1000) and 8.8 h (N=2000) at measured rates; ~76 GPU-hours. All 12 started
immediately.

**`weight_decay` is the second axis** because Adam's 1e-6 decay keeps pulling weights down after the
loss plateaus, making it a plausible driver of the late-phase silencing. If `wd=0` removes the drift,
the slow silencing is a regularisation artifact rather than something the task demands.

### What is measured

1. **New silent units per 1000 iterations** — the derivative of the participation trace. Needs no new
   logging: the trace has recorded every unit every 10 iterations since the first sweep.
2. **Relative parameter drift** `‖W(t) − W(t−Δ)‖_F / ‖W(t)‖_F` for `W_rec`, `W_inp`, `W_out`, `bias`
   — **new**, added to `Trainer.track_participation_` and stored under the `"drift"` key of the
   trace pickle (additive, so existing analysis scripts are unaffected). This is the direct test:
   the loss can be flat while the weights still move, and only this separates the two. Verified
   working — `W_rec` drift declines 0.054 → 0.040 over the first 50 iterations. (`bias` drift reads
   1.0 at the first snapshot because the bias starts at exactly zero; it settles immediately.)
3. **R² and the loss curve**, so "converged" is never confused with "stopped learning".

### What each outcome forces

- **Both curves flatten well inside 100000** → read off the horizon, scale it by N, and size the
  large-N runs from measurement instead of assumption.
- **Still moving at 100000** → the fixed-iteration protocol is unusable for cross-N comparison, and
  every silent-fraction number in this project carries the caveat *"at 30000 iterations"* rather than
  *"at convergence"*. The size sweep would have to be re-run to a convergence **criterion** (stop
  when new silent units per 1000 iterations falls below a threshold).
- **`wd=0` removes the late drift** → the late-phase silencing is a weight-decay artifact and the
  headline phenomenon needs restating in those terms. It would not erase the result — the silence is
  present at 30000 iterations either way, and `frm` still removes it — but it would change what the
  phenomenon *is*.

