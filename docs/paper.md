# Paper plan — silent units in trained RNNs

Working document: the *argument*, what supports each claim, and what is missing. The full
experimental record (configs, job IDs, repro instructions) is [`project_trajectory.md`](project_trajectory.md).

**Every claim carries an evidence status** — ✅ measured, 🟡 preliminary, ⬜ planned — so that
interpretation never quietly becomes result.

**Architecture policy — main text vs supplement.** Every main-text result is stated for **standard
unconstrained RNNs**: no Dale's law, no I/O sign constraints, self-connections allowed, trainable
bias, no cubic term, plain multi-objective gradients. **Dale-constrained and I/O-positive networks
are supplementary throughout** (§S1) — a robustness check on the main claims, never the basis of
one. A result stated only for constrained networks reads as niche and is easy to dismiss.

Where a main-text number is currently derived from constrained networks because that is the only
data with the relevant manipulation, it is marked ⚠️ **Dale-derived, standard-RNN version pending**
rather than presented as settled.

**Working titles**

- *Trained RNNs recruit a fixed number of units, not a fixed fraction*
- *Most units in trained RNNs are silent, and the standard regularizer makes it worse*
- *Silent units distort what RNN models tell us about neural populations*

---

## 1. The problem

RNNs are used as models of cortical circuits, and the comparison to data is made at the level of the
**population**: dimensionality, selectivity distributions, functional cell classes, correlation
structure. That comparison assumes the model's population is the thing being modelled.

**It usually isn't**, and the problem has three parts, each of which makes the next one worse:
units go silent (§1.1); enlarging the network does not buy active units, because the active count
saturates (§1.2); and the surviving population has statistics that differ materially from the ones
a recording would produce (§1.3).

### 1.1 Most units in a trained RNN never fire

In standard ReLU RNNs trained on a context-dependent decision task (CDDM), a large fraction of units
never fire ✅ (N = 1000, unconstrained + trainable bias, 5 nets per row):

| | silent (peak rate < 0.01) | scale-free criterion (peak < 5% of p95) |
|---|---|---|
| h equation, no penalty | **45.6%** | 48.0% |
| s equation, no penalty | **41.9%** | 46.2% |

These are not merely quiet units. **31–42% of units are at exactly 0.0** — literally switched off,
never crossing threshold at any timestep in any condition ✅. (In constrained networks this is
hidden: non-negative input weights guarantee every unit a small positive push, so silent units sit
at 1e-9…1e-2 instead of zero. Same population, concealed depth — see §S1.)

**Most of the silence is created by training** ✅ — but, corrected 2026-07-28, **not all of it**. In
standard unconstrained networks **4.4% of units are already hard-silent at initialization and 21% sit
below the 0.01 line** before a single training step: with signed input weights a unit can start
net-negative and never fire. Training takes it from 4.4% to 41.5%. The older "0% silent at init"
result is **Dale-specific** — I/O positivity guarantees every unit a positive push, so nothing can be
exactly zero there. It also isn't a bug: an independently written Euler integrator reproduces the
per-unit peak rates exactly (max abs diff 0.0) ✅.

### 1.2 The active count saturates: enlarging the network does not buy active units

✅ **Measured 2026-08-19–20** (standard RNN, h equation, no penalty, 3 seeds per size,
N ∈ {100, 500, 1000, 2000, 5000}; figures `silent_at_threshold.png`, `saturation_test.png`).

This is the paper's answer to the sharpest objection it faces: *"why make every unit compute — just
train a bigger network and prune the silent ones?"*

**The comparison had to be made fair first**, and this took most of the effort (§6.1). Networks of
different sizes cannot be compared at a fixed iteration count, because the learning rate is rescaled
with N and larger networks train more slowly per step. They are instead compared at **matched
performance**: each size is read at the iteration `T_N` where its smoothed noise-free training loss
*stably* reaches a common level `L*` = 0.023. This is licensed by a separate result — the loss floor
is size-independent to within 1.6%, and what deviation exists is in the conservative direction
(§6.1) ✅.

| N | `T_N` (iterations to reach L*) | silent, hard | silent, scale-free | **active units** |
|---|---|---|---|---|
| 100 | 32.0k ± 2.2k | 0.7% ± 0.5 | 18.3% ± 1.7 | **99.3 ± 0.5** |
| 500 | 37.6k ± 1.1k | 26.1% ± 1.5 | 41.5% ± 1.3 | **369.3 ± 7.7** |
| 1000 | 44.9k ± 0.5k | 50.8% ± 1.9 | 62.6% ± 2.2 | **492.3 ± 19.0** |
| 2000 | 58.5k ± 1.5k | 70.3% ± 0.9 | 76.4% ± 0.9 | **593.3 ± 18.9** |
| 5000 | 91.1k ± 2.7k | 85.4% ± 0.6 | 87.0% ± 0.3 | **731.7 ± 31.1** |

**A 50× larger network buys 7.4× the active units — and that ratio is still shrinking.** Restricting
the fit to `N ≥ 500` (so the N=100 ceiling, where the network is too small to waste units, cannot
drive it), `M ∝ N^0.31` under the hard criterion and `N^0.36` under the scale-free one. **Doubling
the active population therefore costs roughly a 9× larger network.**

**The growth is significantly decelerating** ✅ — three tests, all pre-registered before the N=5000
seeds landed:

1. **Pooled curvature.** Fitting `log M = a + k·log N + c·(log N)²` over all seed-level points at
   `N ≥ 500` gives **c = −0.118 ± 0.033, F(1,42) = 12.79, p = 0.001** under the hard criterion
   (c = −0.034 ± 0.013, p = 0.012 scale-free). Saturation predicts `c < 0`; `c` is negative in both.
2. **Local exponent.** The exponent falls monotonically with the depth of the matched level, from
   0.69 at `L*` = 0.0257 to 0.31 at `L*` = 0.0230 — the better the networks are trained, the more
   sublinear the recruitment.
3. **Model comparison.** At every matched level the **saturating (hyperbolic) fit beats the power
   law** under the hard criterion (ΔAICc = −9.1 to −17.8, decisive). Under the scale-free criterion
   the power law wins at shallow levels and the saturating fit wins at the deepest one — so the
   evidence for a hard ceiling is *criterion-dependent*, while the evidence for deceleration is not.

The asymptote implied by the hyperbolic fit at the deepest matched level is
**M\* ≈ 880 active units (profile-likelihood 95% upper bound 939)** under the hard criterion, 795
(<882) under the scale-free one — i.e. a task-determined ceiling of order 10³, reached by a network
of a few thousand units and not exceeded by making it larger.

⚠️ **State this caveat in the paper.** `M*` is not a single number: it falls as the matched level
gets deeper (3972 at `L*` = 0.0257 → 880 at `L*` = 0.0230). Better-trained networks use *fewer*
units, so the ceiling is not a capacity limit that training fills up — it is a target that training
keeps tightening toward (§6.1: silencing has not stopped even at 10⁵ iterations). Any quoted `M*`
is therefore an upper bound *at the performance level where it was read*, and the honest claim is
the **deceleration**, not a specific asymptote.

**Consequence.** Pruning cannot deliver a large active population at any size, because the ceiling
is set by the task rather than the parameter budget. Activity regularization is the only route:
`frm` at N=1000 gives 1000 active units at the same performance (§3).

### 1.3 The surviving population's statistics are materially distorted

The argument that turns this from tidiness into a validity problem. **Recorded cortical populations
do not contain 45–85% of neurons that never fire during a task** — so an RNN that does is not the
population being claimed as a model, and every population-level comparison inherits the distortion.

✅ **Measured 2026-08-17** (N=1000, 5 nets per cell, all conditions at R² = 0.84–0.87 — see
`project_trajectory.md`, figure `population_distortion.png`). Networks that perform identically
differ by:

- **3.5× in effective dimensionality** (participation ratio 2.22 → 7.74);
- **1.7× in choice selectivity measured over active units** (33.5% → 58.3%). The 3× figure from
  all-unit fractions is inflated by dilution, since silent units are non-selective by construction —
  and context selectivity actually *falls* among active units, 42.3% → 31.1%;
- **2× in total metabolic cost** (31.1 → 14.6 — the concentrated solution being the *more*
  expensive one);
- **100× in energy concentration** (HHI 0.123 → 0.0012, i.e. cost carried by an effective 8 units
  versus ~850).

Separately, within unpenalized networks the same selectivity statistic reads 24.3% over all units
but 42.3% over active units only — and a recording experiment sees only the latter.

| Statistic | Why it is distorted |
|---|---|
| **Dimensionality / participation ratio** | computed over an effectively ~500-unit circuit while reported as N=1000 |
| **Selectivity distributions** (context, motion, colour, choice) | silent units are untuned by construction, diluting every proportion by an arbitrary factor |
| **Functional clustering / cell classes** | a large "unresponsive" cluster with no counterpart in a recorded population |
| **Metabolic cost of the computation** | unpenalized nets concentrate high rates in few units; `frm` nets spread moderate rates over all. Total energy may match while the distribution differs entirely |
| **Correlation structure / eigenspectrum** | inherits the above |

The claim: *your conclusion about the circuit depends on whether you regularized activity, and
nobody reports it* — and by §1.2 the distortion **grows with N**, so it is worst in exactly the
large networks the field is moving toward.

⚠️ **One statistic goes the other way, and belongs in the paper as a limitation.** Rate heterogeneity
across active units, measured as **σ_log** (the std of log₁₀ mean rate — the shape parameter of the
lognormal that cortical rate distributions follow, ~1 in cortex), is **1.20 (h) / 1.01 (s) without
penalties — the biological value — and collapses to 0.26 / 0.15 under `frm`**, a fifth of a decade.
CV (3.62 → 0.49) and the p90/median tail (6.2 → 1.7) agree. So `frm` trades one unrealism (half the
population silent) for another (a population too uniform). See §6.5.

✅ **But the revived units are genuinely modulated, not tonic.** Within-trial temporal CV is 1.29
under `frm` versus 1.43 without — only ~10% lower — so the penalty is *not* satisfied by units
sitting constantly at the cap. This closes a worry open since the first sweeps. (`both` does flatten
modulation, to 0.96.)

⬜ **Does the rescued network function better?** — still open, no new training needed:

- **Noise robustness** — raise σ_rec at evaluation, compare R² decay. (A first attempt found a
  size effect at p=0.084 that **failed to localise** — p=0.31 at |coh|<0.02, p=0.81 at coh=0, and
  the wrong sign — and was **retracted** 2026-08-16. Do not resurrect it without a pre-registered
  localisation test.)
- **Lesion robustness of the top contributors** — ablate the highest-participation units and measure
  degradation. *Design note:* naive random ablation is misleading, because in a half-dead network
  half the lesions hit units that do nothing, making it look artificially robust. Ablate by
  participation rank, or express degradation per *active* unit removed.
- **Generalization** — held-out coherences (the interleaved-midpoint validation batch is now logged
  during training, §6.1) or a shifted stimulus distribution.

---

## 2. Standard activity regularizers do not fix it — and at strength make it worse

Activity regularization in RNN training is routine — metabolic-cost terms are standard practice in
this literature. But the usual penalties act on the wrong side of the distribution: they penalize
*high* rates, pushing the whole population down.

| Penalty | Effect on silence |
|---|---|
| **`rws`** — recurrent-weight sparsity (λ=0.05) | **Does not rescue — and the appearance that it does is a threshold artifact** ✅, see below. |
| **`met`** — metabolic cost, `mean(fr²)`, the field-standard form | ✅ **Never rescues, and at strength makes it worse.** Across λ ∈ {0.01, 0.1, 1, 10} and N ∈ {100, 500, 1000} the silent fraction never falls below baseline. At λ=10 it rises sharply where there is headroom — N=100: **12% → 59%** scale-free silent; N=500: 41% → 69% — while R² stays 0.81–0.87, so this is not a penalty destroying the task. Flat at N=1000, where the baseline is already ~42% and little room remains. |

### 2.1 `rws` produces units that are non-zero but functionally dead ✅

The sharpest single illustration of why **two silence criteria must always be reported**, and a
result that only became visible once N=2000 landed (2026-08-20).

Measured at a **matched iteration budget** (the `none` baseline read from its participation trace at
the iteration where the penalty run stopped, since the unpenalized runs were given up to 2× the
budget and silencing never settles — §6.1):

| N=2000, budget 150k | `none` | `rws` | `frm` | `both` |
|---|---|---|---|---|
| silent, **hard** (`p<1e-6`) | 78.9% | **60.5%** ← looks like a rescue | 0.0% | 0.0% |
| silent, **scale-free** (`p<0.05·q95`) | 80.9% | **86.0%** ← worse than baseline | 0.0% | 0.0% |

`rws` lifts units just across the 1e-6 line while leaving them far below any functional threshold:
it converts hard zeros into a long tail of tiny-but-nonzero rates. The effect **grows with N** —
the hard/scale-free gap is 11.9 pp at N=500, 20.7 at N=1000, **25.5 at N=2000** — and it is absent
in every other condition (`none`'s two criteria agree to within 2 pp at every size).

This is also the resolution of an apparent late-training "resurrection" under `rws`: the hard-silent
count peaks and then falls back (75.5% → 60.5% at N=2000, peaking at iteration 76k), with the drop
scaling steeply in N (3.2 / 10.2 / **15.0** pp at N = 500 / 1000 / 2000) and absent in `none`
(≤1.0 pp). **No units are revived** — they cross 1e-6 without becoming active, which the scale-free
criterion sees straight through. Anyone reporting a single hard threshold would publish this as a
rescue.

`rws` also silences on a **completely different timescale**: median first crossing at iteration 3287
versus 16 for the unpenalized net ✅. It does not participate in the early collapse; it kills units
slowly, late in training.

### 2.2 Task cost: `frm` and `both` cost nothing, and at N=2000 they *help* ✅

⚠️ **Read the loss column carefully — this bit the project twice.** `TrainLosses.json` records what
the optimizer descends: task + λ·penalty, evaluated with **noise on**. That quantity is ~65% noise
floor plus a penalty term, so ordering conditions by it compares regularization strengths rather
than performance. At N=2000 it reads `rws` 0.02432 (apparently the worst condition) while `rws`'s
noise-free task loss is 0.00745 (better than `rws` at N=500 or N=1000). **Every task-performance
number below is the noise-free masked MSE of the final weights on a shared batch**, evaluated
post-hoc by an independent numpy integrator, not the recorded training column.

| noise-free task loss | `none` | `rws` | `frm` | `both` |
|---|---|---|---|---|
| N=500, train / held-out | 0.00851 / 0.01598 | 0.00835 / 0.01569 | 0.00980 / 0.01757 | 0.00844 / 0.01571 |
| N=1000, train / held-out | 0.00907 / 0.01664 | 0.00792 / 0.01532 | 0.00876 / 0.01628 | 0.00828 / 0.01509 |
| N=2000, train / held-out | 0.00888 / 0.01628 | 0.00713 / 0.01442 | 0.00760 / 0.01463 | **0.00578 / 0.01226** |

Held-out = interleaved coherence midpoints, never seen in training. Three things follow:

1. **The activity penalties are not paid for in performance.** `frm` is ~15% worse than `none` at
   N=500, level at N=1000, and **14% better at N=2000**; `both` is level, better, and **35% better**.
   The earlier "+5% task cost of `frm`" figure was computed from the total objective and is
   **retracted**.
2. **The advantage grows with N** — exactly where the unpenalized silence is worst. At N=2000 `both`
   attains the best task loss of any condition *and* keeps all 2000 units active.
3. **Generalization tracks training loss, with a constant ~1.85× gap in every condition.** No penalty
   trades train for held-out performance; the gap is a property of the coherence grid, not of
   regularization.

⬜ **Extension running**: the same grid at N=5000 (Spock array `5670493`), to check whether the
advantage keeps growing into the regime where unpenalized silence reaches 85%.

---

## 3. What we tried, and what worked

Every intervention below was chosen because it *could* plausibly have removed the effect.

| # | Intervention | Result |
|---|---|---|
| 1 | Equation type `h` vs `s` | both affected ✅ |
| 2 | Cubic saturation term γ: 0.1 → 0 | no change ✅ |
| 3 | Dale boundary: sticky → reflective | no change ✅ |
| 4 | **Removing Dale's law entirely** | no change in the silent fraction ✅ |
| 5 | **Removing I/O positivity** | no change in the fraction; converts soft floors into hard zeros ✅ (§S1) |
| 6 | **Trainable bias** (`[-1,1]`, zero init) | no rescue ✅ (55.3% → 54.9%) |
| 7 | **Self-connections** allowed | no rescue ✅ — and the network trains the diagonal into self-*inhibition*: corr(self-weight, log participation) = **−0.51**, active units at −0.060 vs silent at −0.007. The cheapest escape from silence is one it declines to take. |
| 8 | Activation: softplus(β=25), leaky-ReLU | persists, 40–64% (Dale) ✅; ⬜ standard-RNN rerun |
| 9 | Recurrent noise σ_rec ∈ {0, .01, .05, .1} | never helps; σ=0 is the *worst* regime (~80%) ✅; ⬜ rerun |
| 10 | `rws` sparsity penalty | fails (§2) ✅ |
| 11 | `met` metabolic penalty, 4 decades | fails, and at λ=10 makes it worse ✅ |
| 12 | **Longer training** (30k → 300k iterations) | makes it worse, monotonically ✅ — silence is not a transient of under-training |
| 13 | **Larger networks** | makes it worse, monotonically ✅ (§1.2) |
| 14 | **Connectivity scale** (spectral radius, density) | ⬜ planned — is silence simply weak recurrent drive? |
| 15 | **`frm` firing-rate-magnitude penalty** (λ=0.1) | ✅ **works** |

**`frm` is the one that works, and it works completely** ✅:

- Silent fraction → **exactly 0 under both criteria**, in every cell, every architecture (Dale,
  unconstrained, ±bias), and every activation tested — and it stays 0 through 200k iterations, at
  every size up to N=2000, where the unpenalized network has reached 79% ✅.
- No tail whatsoever: the *minimum* participation of any unit in any `frm` net is 4.8e-2 — about a
  sixth of the median, with nothing approaching zero. Contrast `rws` (§2.1), which produces exactly
  the tail `frm` does not.
- **No task cost, and a growing benefit with N** ✅: level with `none` at N=1000 and 14% better at
  N=2000 on the noise-free task loss; `both` is 35% better at N=2000 (§2.2).
- Effective participating units (1/HHI) rise from ~60–150 to ~700–900 of 1000.

> **Name the tautology before a referee does.** "A penalty on sub-cap firing removes sub-cap units"
> is nearly circular. The non-trivial content is: it *improves* performance at large N; unpenalized nets
> concentrate onto a *task-determined absolute number* of units regardless of N; the standard
> regularizer fails or makes it worse; and — §1.3 — the resulting populations differ in ways that
> change scientific conclusions.

---

## 4. Does the rescue prevent, or resurrect?

A per-unit gradient argument says rescue should be impossible: a dead ReLU has zero derivative
everywhere, so no gradient — including from the penalty — reaches its incoming weights.

Answered directly from the participation traces (per-unit participation logged every 10 iterations
through all 30000), h equation, per 1000 units ✅:

| | ever dips below 0.01 | ends below | **silent ≥500 iters, then recovers** |
|---|---|---|---|
| standard, none | 83.4% | 54.9% | 95.8 |
| standard, **frm** | 48.9% | **0%** | **0.6** |
| Dale, none | 96.0% | 53.6% | 85.0 |
| Dale, **frm** | 94.4% | **0%** | **369.4** |

**Both, and which one depends on the architecture.** In standard RNNs `frm` **prevents** — units dip
briefly during the early collapse and are caught within a few hundred iterations; essentially none
endure a long silent episode. In Dale networks it genuinely **resurrects** — 369 units per network
were silent for ≥500 consecutive iterations and returned.

Two by-products of the same analysis ✅: silence is **not strictly irreversible** even without
penalties (~96 units per network recover spontaneously), and the split *begins* within the first few
hundred iterations, preceded by a **global collapse** in which the entire population goes quiet
within ~20 iterations and only the eventual-active subset climbs back out. Note it begins early but
does **not** finish early — see §6.1.

⬜ Optional sharpening, 20 jobs: force a random 25% of units silent at init (bias = −1, frozen) and
follow those specific units. Only needed if a referee insists on "can it revive a unit dead from the
very start". The Dale resurrection number already carries most of that weight.

*(The earlier master-inhibitor / frozen-clamp experiments asked this same question through
hand-built silencing constructions. They are superseded by the trace analysis, which answers it on
the natural initialization in the standard architecture. Keep at most as supplementary.)*

---

## 5. Why the floor exists (mechanism of the matched level)

Needed because §1.2 rests on `L*` = 0.023 being a meaningful common performance level rather than an
arbitrary stopping point.

✅ **98% of the residual loss comes from |coh| < 0.05**, where the target is *discontinuous* at
coh = 0: the correct choice flips sign across an infinitesimal stimulus change, so no smooth network
output can match it. The floor is **task structure, not capacity** — which is exactly why it does
not move with N. Ruled out as alternative causes: training noise, weight decay, and fit bias
(§6.1) ✅.

This also disposes of a reading a referee will try: the networks are not "failing to converge to
different floors because they are different sizes" — they are all converging to the same
task-imposed floor, from different directions and at different rates.

---

## 6. Open questions

In dependency order — 6.4 gates the framing of the whole thing.

### 6.1 Does training converge? — **answered, and the answer forced a new protocol** ✅

**Status: resolved 2026-08-14 → 08-18.** This consumed the most effort of anything in the project
and produced a methodological result worth its own methods paragraph.

**It does not converge in any strict sense.** The loss is flat over the final 10% at every size, but
the silent fraction is still climbing, and *seven* candidate stopping criteria all fail:

- relative parameter change (`‖ΔW‖/‖W‖`) decays as a **power law** in iteration (exponent −0.29 to
  −0.57), so a 1% criterion extrapolates to 0.5–5.6 **million** iterations;
- the drift/jitter distinction via the lag-scaling exponent α is unusable — α wanders below 0.5
  (sub-diffusive/caged) and back, so there is no stable crossing;
- criteria on the raw loss trace fire ~7× too early, because the single-batch loss is noisy and its
  minimum is a **noise lottery** (an earlier "median of 101 lowest losses" statistic was retracted
  for this reason);
- criteria on the silent-unit count never fire, because silencing never stops.

The full negative record — what was tried, why each failed, and the three traps (mismatched fit
ranges manufacture trends; a criterion satisfied only because the run *ended*; noise-lottery
statistics) — is written up self-containedly in `project_trajectory.md`.

**What replaced it: matched performance.** Read each size at the iteration `T_N` where its
**smoothed** clean training loss stably crosses a common level `L*` — "stably" meaning the last
iteration at which the centred 2001-iteration mean is still above threshold, with explicit guards
against a rising trend and against sub-unit resolution. `T_N` scales as ≈ `N^0.27` (32k → 91k over
N = 100 → 5000), i.e. slowly, which is why fixed-iteration comparison is biased and why the bias
runs in the direction of *underestimating* silence at large N.

**What licenses it:** the loss floor must not depend on N, or "the same loss" would not mean "the
same performance". ✅ Fitted with three decay families (power law, exponential, stretched
exponential; the stretched form wins by AICc after fixing the unidentifiable τ), the floor is
**size-independent to within 1.6%**, and the residual deviation is a slight *rise* with N — the
conservative direction, since it means large networks are read at a marginally harder standard, not
an easier one. Cause of the residual rise: **optimization difficulty, not capacity** — ruled out
noise, weight decay, and fit bias.

### 6.2 Does the active-unit count saturate? — **answered: it decelerates significantly** ✅

**Status: answered 2026-08-19–20, moved into §1.2.** Curvature is significantly negative
(p = 0.001 hard, p = 0.012 scale-free); the saturating fit beats the power law at every matched
level under the hard criterion; a 50× network buys 7.4× the active units, and doubling the active
count costs ≈9× the network. Remaining caveats, both in §1.2: the ceiling estimate depends on the
matched level, and the model-comparison verdict (though not the deceleration) depends on the
silence criterion.

⬜ **N = 10000 in flight** (Della `12599054`, 3 seeds, 80k iterations, ETA 2026-08-23 ~22:00). It
adds one point at 2× the current largest size. The power law predicts ~830 active units there and
the hyperbolic fit ~760 — a 9% gap, which three seeds at ±31 units can resolve. This is the last
data the size argument needs.

### 6.3 Is the rescue preventive or genuinely restorative? — **answered, architecture-dependent** ✅

See §4. In standard RNNs `frm` **prevents** (0.6 units per network recover from a long silent
episode); in Dale networks it **resurrects** (369 per network).

What remains open is the harder case: can it revive a unit dead **from initialization and stays
dead**? §1.1's finding that 4.4% *are* hard-silent at init in standard networks means the material
now exists. ⬜ 20 jobs would settle it. Optional.

### 6.4 Is the silence just spare capacity — is the task too easy?

**Status: open, and now the single most important gap.** The deflationary reading, and the one a
referee will default to: CDDM is low-dimensional, unpenalized networks solve it with ~60–150
effective units, and a 5000-unit network trivially has units to spare.

**§1.2 sharpened this objection rather than answering it.** A task-determined ceiling of ~880 active
units is *precisely what the spare-capacity story predicts*. Right now the paper's headline result
and its deflationary explanation are observationally identical.

The test is to scale the **task**, not the network, and ask whether `M*` moves:

- more contexts (CDDM generalises naturally to 3–4 modalities);
- more stimulus dimensions / finer coherence resolution;
- compositional variants; or simply a second, harder task from the repo.

Then measure `M(N)` at matched performance in each, and ask whether the **ceiling** — not the
fraction — tracks task demand while remaining independent of N.

**This converts the result from a curiosity into a law with two axes:** *trained RNNs recruit a
number of units set by the task, not a fraction of the network.* Every answer is publishable, but
they are different papers. If `M*` rises with task complexity, the conclusion is partly a
recommendation (*don't train 1000-unit RNNs on simple tasks and then analyse the population*), and
it composes with §1.2 because you cannot cheaply buy a large active population by enlarging the
network either. If `M*` is flat regardless of task complexity, it is a genuine pathology and the
paper is much stronger.

### 6.5 Can the rescue be made biologically realistic? — the σ_log gap

**Status: open, and it undercuts the recommendation.** `frm` removes the silence but flattens the
rate distribution (σ_log 1.20 → 0.26 against a cortical value of ~1; §1.3). Recommending `frm` to
the field while it produces a population too uniform to be cortex is a real weakness.

⬜ The natural knobs already exist in the implementation and have **never been swept**: `frm`'s
`cap_fr` (the rate above which a unit stops being pushed) and its unused `logsumexp` aggregation,
which penalises the *softmin* of the rates rather than the mean and should therefore act only on the
lowest tail — exactly the shape needed to remove zeros without compressing the top. A single sweep
over `cap_fr` × aggregation at N=1000 would report σ_log, silent fraction, and R² jointly, and turn
this limitation into a tuned recommendation.

### 6.6 Smaller open items

- **Connectivity scale** ⬜ — is silence simply insufficient recurrent drive? Spectral radius was only
  ever checked *at initialization* (0% silent at every radius), never in trained networks.
- **Generality across tasks** ⬜ — everything is CDDM. 17 task configs exist in the repo
  (DMTS / GoNoGo / MemoryAngle / …). Overlaps with 6.4 but is a weaker version of it: showing the
  effect on a second task establishes generality; showing `M*` *moves* with task complexity
  establishes mechanism.
- **Activation and noise reruns** ⬜ — established in Dale networks, not yet repeated in standard ones.
- **Reverse engineering / identifiability** ⬜ — see §7.

---

## 7. Possible framing: RNNs as a testbed for identifiability

A framing worth considering for the introduction, and the one that would make §1.3 land hardest.
RNNs are valuable precisely because, unlike a brain, the ground truth is available: one can subsample
units from a trained RNN exactly as an electrode subsamples a cortex, fit the standard population
models to the subsample, and ask how well the underlying system is recovered.

Under that frame the silent-unit result is a statement about **identifiability**: a network whose
computation is carried by a task-determined ~10³ units, of which a recording sees a random subset,
is a system where the population statistics one measures (§1.3) depend on a training choice nobody
reports. And the `frm`-rescued networks then serve as a **proof of usability**: activity-regularized
networks are easier to reverse-engineer, because the computation is distributed over units that all
actually fire.

⬜ Nothing here is measured yet. The minimal experiment: subsample K units from `none` and `frm`
networks matched on performance, fit the same population model to both, and show that recovery of
the known ground truth is better for `frm` at every K. This closes the loop with the introduction
and is the strongest available answer to "so what should I do differently?".

---

## 8. Methods points that must be stated

- Training noise (σ_rec = σ_inp = 0.05, σ_out = 0.03) — retained deliberately; standard in
  neuroscience RNNs, absent in vanilla ML ones.
- `lr` is rescaled as `lr × (100/N)^0.333`, so **N and learning rate co-vary** in the size sweep.
  This is precisely why the matched-performance protocol (§6.1) is necessary.
- **The matched-performance protocol itself** — the stable-crossing definition, the smoothing
  window, the guards, and the size-independence check on the floor. This is a reusable contribution
  for anyone comparing RNNs across sizes and should be written as such.
- **Participation metric**: `p_i = std_{t,c}(r_i) + q_0.9(|r_i|)`; two silence criteria are reported
  throughout — hard (`p < 1e-6`) and scale-free (`p < 0.05·q95(p)`) — because the silent fraction is
  threshold-dependent (13% at <1e-4, 44% at <1e-2, 49% at <5e-2 in the same networks). Never report
  one threshold alone.
- Loss is reported **noise-free and on the training conditions**, with a held-out set of interleaved
  coherence midpoints logged in parallel. Noisy single-batch loss is not a usable convergence signal.
- Penalized runs use plain multi-objective descent (the earlier task-safe gradient projection is off).
  **Comparing total objectives across penalties is meaningless**; only the task term is comparable.
- **A practical result worth its own paragraph** ✅: strong activity penalties can make training
  diverge, and *not* through gradient explosion. `frm` builds a self-exciting recurrent loop with
  gain > 1 to overcome inhibition; explicit Euler then overflows within the trial while gradient
  norms remain ~1, so clipping is useless. The fix is a smaller integration step plus a bounding
  nonlinearity (dt 1 → 0.5 with γ = 0.1 cut the divergence rate from ~50% to ~10%). This is exactly
  the kind of finding a tools-and-methods paper should carry.
- Per-net spread or CIs, not just means over 3–5 nets.

---

## 9. Venue and priorities

**Target: PLOS Computational Biology (Methods).** It requires correct, complete, useful work rather
than novelty impact — which fits. Fallbacks: eNeuro (Research Methods and New Tools), NBDT (welcomes
careful, unglamorous, control-heavy work). bioRxiv preprint immediately. **eLife is now viable** —
the earlier objection was that §1.3 did not exist; it does as of 2026-08-17. JOSS for the package as
a companion citation.

**Priority order (revised 2026-08-20):**

1. ⬜ **Task scaling** (§6.4) — *the one thing that decides what the paper claims.* Without it, the
   saturation result and the spare-capacity dismissal are indistinguishable.
2. ⬜ **`cap_fr` × `logsumexp` sweep** (§6.5) — cheap, one grid at N=1000, converts the σ_log
   limitation into a tuned recommendation.
3. 🟡 **N=10000** — in flight, ETA 2026-08-23; last point the size argument needs.
4. 🟡 **Penalty grid at N ∈ {2000, 5000}** — in flight; checks `frm` survives the 85%-silence regime.
5. ⬜ **Subsample-and-fit identifiability demo** (§7) — closes the loop with the introduction.
6. ⬜ **Lesion / generalization robustness** (§1.3) — no new training.
7. ⬜ Second task; activation / noise / connectivity-scale reruns.

**Done since the last revision:** metabolic-cost sweep (§2), population-distortion analyses (§1.3),
standard-RNN N sweep to N=5000 (§1.2), matched-performance protocol (§6.1), floor mechanism (§5).

---

## S1. Supplementary: Dale-constrained and I/O-positive networks

**Not part of the main argument.** Every claim in §§1–7 is established in standard unconstrained
RNNs. This section exists for two narrow purposes: to show the phenomenon is *not* an artifact of
the constrained architecture much of this literature uses, and to record the specific places where
the constrained case behaves differently. Readers who do not work with Dale-constrained models can
skip it.

1. **Silence is overwhelmingly excitatory** ✅: 53–55% of excitatory units vs 3.5–5.0% of inhibitory
   units silent (h, none). This **falsifies** the natural hypothesis that the excitatory-only readout
   starves inhibitory units of gradient — the opposite happens. Likely load-bearing redundancy: 200
   inhibitory units carry the whole network's inhibition at 4× weight and are individually
   indispensable, while 800 excitatory units are mutually redundant.
2. **I/O positivity conceals how dead the units are** ✅: with `W_inp ≥ 0` and non-negative inputs,
   every unit gets a positive push at every timestep, so no unit is *exactly* zero (2.2% hard-dead vs
   31.4% unconstrained) — while the total silent fraction is unchanged (44.2% vs 42.7%). Every one of
   the 285 hard-zero units in an unconstrained net has Σ`W_inp` < 0 (100%, no exceptions).
3. **`rws` reverses sign** ✅: worse than baseline under Dale, mildly better unconstrained.
4. **`frm` resurrects rather than prevents** ✅ (§4) — in standard RNNs it prevents.
5. **The `s` equation is more sensitive to constraints** ✅: 55.1% → 32.6% silent when Dale and I/O
   positivity are removed, while `h` is unchanged.

---

## Retracted claims (kept so they are not re-derived)

- **"Median of the 101 lowest losses" as a performance statistic** — a noise-lottery on the noisy
  single-batch loss; the raw trace dips below any threshold ~7× too early.
- **"93–96% of the loss is irreducible"** — model-dependent, and the model is misspecified.
- **A size effect in noise tolerance** — p = 0.084 whole-task but failed to localise (p = 0.31 at
  |coh| < 0.02, p = 0.81 at coh = 0, wrong sign).
- **"Training never reaches a stationary regime" (aging framing)** — overstated. The loss *has*
  converged; the motion is along a flat manifold.
- **"`frm` costs ~5% in task loss"** — computed from the total objective (task + λ·penalty), not the
  task term. On the noise-free task loss `frm` is level with `none` at N=1000 and *better* at
  N=2000 (§2.2). This is the same total-vs-task trap that invalidated an earlier penalty table;
  it has now caught the analysis twice, so **never quote `TrainLosses.json` across penalty
  conditions**.
- **"`rws` partially rescues silent units at large N"** — a hard-threshold artifact. Under the
  scale-free criterion `rws` is *worse* than baseline at every size, and the gap grows with N
  (§2.1).
- **Monotone decay exponent γ with N** — an artifact of mismatched fit ranges (N=2000 fitted over
  300k while others were fitted over 200k). Mismatched fit ranges manufactured a spurious trend
  three separate times in this project.
