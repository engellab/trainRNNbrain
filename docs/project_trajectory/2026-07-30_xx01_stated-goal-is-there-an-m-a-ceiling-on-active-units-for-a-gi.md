## 2026-07-30 — STATED GOAL: is there an M*, a ceiling on active units for a given task?

Written as a goal statement rather than a result, because the next block of compute is aimed at
exactly one question and it is worth fixing what that question is, why it matters, and what would
answer it — before spending the hours.

### The question

For a given task, is there an **M\*** — an upper limit on the number of *active* units a trained
network converges to, **regardless of how large N is**? Or does the active count keep growing with N
without bound?

### Why it decides the value of the penalties

This is the argument that makes the whole project matter, so it should be stated plainly:

- **If M\* exists (saturation):** you cannot obtain a network with more than M\* active units by
  standard training at any size. Training bigger and pruning the silent units does not help — the
  ceiling is set by the task, not the budget. In that case the firing-rate + sparsity penalties are
  **the only route** to a network with M > M\* participating units, and they are therefore genuinely
  useful rather than merely tidy. `frm` at N=1000 already delivers 1000 active units; if M\* is ~750,
  no unpenalized network of any size can match that.
- **If there is no M\* (unbounded growth):** pruning works, the penalties are a convenience rather
  than a necessity, and the paper's weight must shift entirely onto the population-level
  consequences (`paper.md` §5) — that silent units distort dimensionality, selectivity distributions
  and cluster structure, and so corrupt the comparison between model and data.

Either answer is publishable. They are different papers.

### Why we cannot answer it yet

The current active-unit curve is measured at a fixed 30000 iterations, and **the networks are not
converged at that point — with the residual drift growing with N**:

| N | 100 | 250 | 500 | 1000 | 2000 |
|---|---|---|---|---|---|
| active units (30k, wd=1e-6) | 97 | 227 | 390 | 580 | 862 |
| local exponent | — | 0.93 | 0.78 | 0.57 | 0.57 |
| silencing rate still running at 30k | ~0 | — | 3.3 ± 1.5 | 9.6 ± 3.5 | ~12 |

Larger networks sit further from their asymptote, which **inflates their active count** and biases
the curve toward "growth" — which is exactly the answer we currently get (a power-law fit
`4.55·N^0.665` beats the saturating `749·N/(N+672)` at the one new point, N=2000). That bias has the
same sign as the effect we are trying to detect, so the current evidence cannot be trusted either
way.

### What has to be measured

**All of the drift channels, per model — not just the silent fraction.** The silent count is one
projection of a network that is still moving, and it can be flat while the network reorganises
underneath (measured: at N=100 the silent count is stationary while the participation vector still
changes 4.5% per 1000 iterations). So each run records:

| channel | quantity |
|---|---|
| weights | `‖W(t) − W(t−L)‖_F / ‖W(t)‖_F` separately for `W_inp`, `W_rec`, `W_out`, at lags 100 / 1000 / 10000 |
| direction | cosine between consecutive displacements, per matrix — ≈0 = jitter, >0 = still marching |
| participation | `‖p(t) − p(t−L)‖ / ‖p(t)‖` at the same lags |
| outcome | silent-unit count (participation < 1e-6) at every probe |

All reduced to scalars during training; no weights are written to disk.

### The design

1. **Characterise the drift curves per model** and extract **T(N)** — the iteration at which motion
   stops being directional. Criterion: the consecutive-displacement **cosine**, not the silencing
   rate, because the rate is noisy and near zero by construction at the end while the cosine is
   well-conditioned. The other channels corroborate.
2. **Report the active-unit count at T(N)** rather than at an arbitrary fixed iteration. This is the
   converged curve, and it is the only one the M\* question can legitimately be asked of.
3. **Fit the converged curve** with the two competing models and decide by the pre-registered rule
   already recorded (2026-07-27): saturating vs power law, discriminated most sharply at large N.

Sizes and lengths, set from measured decay rates (N=2000 decays ~2.5× slower than N=1000 and needs
the longest run):

| N | iterations | h/job | 3 seeds |
|---|---|---|---|
| 100 | 50000 | 1.8 | 5 h |
| 500 | 200000 | 7.2 | 22 h |
| 1000 | 200000 | 8.0 | 24 h |
| 2000 | 300000 | 26.4 | 79 h |

~130 GPU-hours. `weight_decay = 1e-6` throughout — the field default, kept deliberately (2026-07-30)
rather than switching to 0, even though 0 gives a more stationary measurement.

### What would settle it

- **M\* exists** if the converged active count flattens with N — successive local exponents
  continuing to fall toward 0.
- **No M\*** if the exponent stabilises (it currently sits at 0.57 between N=500→1000 and
  N=1000→2000, having fallen from 0.93; a genuine plateau in the exponent is the signature of
  unbounded power-law growth).
- The discrimination is sharpest at large N, which is why the ceiling estimate (~750 active units
  from the saturating fit) has to be tested against a converged N=2000 point rather than an
  extrapolation from N ≤ 1000.

**Caveat to carry:** even a converged curve over N ∈ {100 … 2000} is one decade. A ceiling at ~750
would be visible in that range; unbounded growth with a slowly-falling exponent would not be
distinguishable from a very high ceiling. State the range of validity rather than claiming the
asymptote.

### Submitted 2026-08-17 16:00 EDT — Della arrays `12545109` (9 jobs) and `12545110` (3 jobs)

The drift-characterisation sweep defined by the goal statement above is now running, from its own
git worktree at commit `2f9bcc7`.

| Array | Cells | Jobs | Iterations | Wall request | QOS |
|---|---|---|---|---|---|
| `12545109` | N = 100, 500, 1000 | 9 | 50000 / 200000 / 200000 | 12:00:00 | `gpu-short` |
| `12545110` | N = 2000 | 3 | **300000** | 40:00:00 | `gpu-medium` |

Split into two submissions because the N=2000 cells need ~26.4 h, past the 24 h `gpu-short` ceiling.
**Iteration budgets are per size, set from measured decay rates** rather than uniform: the silencing
rate decays ~2.5× more slowly at N=2000 than at N=1000 (a factor of 4.2 vs 10.8 between iterations
30k and 100k), so a flat budget would leave the largest and most decisive cell still drifting while
over-running N=100, which is already settled by iteration ~3000.

Configuration verified on Della before submitting: `dale: false`, `self_connections: true`,
`weight_decay: 1.0e-06` (the field default, kept deliberately — see 2026-07-30), `track_drift: true`,
`track_every: 10`, `store_participation_every: 100`, `light_outputs: true`, no penalties, h equation.

**Recorded per run** (all reduced to scalars during training; no weights written to disk):
`drift_{W_inp,W_rec,W_out}_lag{100,1000,10000}`, `cos_{W_inp,W_rec,W_out}`,
`dp_lag{100,1000,10000}`, and `silent_1em6` at every probe.

**Planned analysis:** drift curves per model → **T(N)** from the directional-cosine criterion →
active-unit counts *at* T(N) → fit the converged curve against the saturating and power-law models
using the thresholds pre-registered on 2026-07-27, to test for **M\***.

> **Near-miss worth recording.** The task-index decode was verified under `zsh`, which indexes arrays
> from 1, while the launcher runs under `bash`, which indexes from 0 — the check appeared to show
> every cell shifted by one size. Re-checked under `bash` and the mapping is correct (tasks 1–3 →
> N=100, 4–6 → 500, 7–9 → 1000, 10–12 → 2000). A real off-by-one here would have silently mislabelled
> every output folder with the wrong N.

