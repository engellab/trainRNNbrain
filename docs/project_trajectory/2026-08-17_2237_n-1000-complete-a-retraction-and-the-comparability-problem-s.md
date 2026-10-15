## 2026-08-17 22:2x — N=1000 complete, a retraction, and the comparability problem SOLVED

### Retraction: the N=1000 seed spread was a reading error

An earlier note in this conversation reported N=1000 losses of 0.0178 / 0.0222 / 0.0251, called the
seed spread "~700× wider than N=500", and inferred that the loss floor is not universal across sizes.
**All of that is wrong.** Those numbers were read from the tail of the SLURM log, which prints the
*instantaneous single-batch* loss — pure noise around the true value. The smoothed losses are
0.02225 / 0.02235 / 0.02232, a spread of **0.4%**, and the fitted floors are
0.02135 / 0.02138 / 0.02147.

### The three sizes share one floor

| N | L_∞ | γ | seed spread in L_∞ | L(T) | reducible left |
|---|---|---|---|---|---|
| 100 (50k) | 0.02086 ± 0.00039 | 0.47 ± 0.06 | 4.5% | 0.02266 | 5.3–9.2% |
| 500 (200k) | 0.02135 ± 0.00003 | 0.52 ± 0.01 | 0.3% | 0.02222 | 3.0–3.2% |
| 1000 (200k) | **0.02140 ± 0.00005** | 0.54 ± 0.01 | 0.6% | 0.02225 | 2.9–3.3% |

L_∞ agrees to **0.2%** between N=500 and N=1000. N=100's slightly lower estimate is inside its own
much larger uncertainty and is a short-run artefact. A single task-imposed floor at ≈ **0.0214**
governs every size: 10× the units buys no better performance.

### Loss at matched budgets — the curves nearly coincide

| budget | N=100 | N=500 | N=1000 |
|---|---|---|---|
| 10,000 | 0.02460 ± 0.00008 | 0.02470 ± 0.00017 | 0.02484 ± 0.00005 |
| 20,000 | 0.02350 ± 0.00008 | 0.02354 ± 0.00028 | 0.02369 ± 0.00011 |
| 50,000 | 0.02266 ± 0.00005 | 0.02273 ± 0.00006 | 0.02284 ± 0.00006 |
| 100,000 | — | 0.02237 ± 0.00003 | 0.02250 ± 0.00003 |
| 200,000 | — | 0.02222 ± 0.00004 | 0.02225 ± 0.00002 |

Bigger networks are *very slightly* behind at equal iterations — consistent with their smaller
learning rate — but the gap is under 1%.

### THE ANSWER: match on performance, not on iterations

Comparability does **not** require convergence, and the previous seven attempts failed because they
all tried to answer the harder question. The tractable question is: *when is each size equally good
at the task?* Read the iteration at which each size first reaches a chosen loss:

| target loss | N=100 | N=500 | N=1000 | T(1000)/T(100) |
|---|---|---|---|---|
| 0.0250 | 6,500 ± 707 | 8,333 ± 236 | 9,667 ± 236 | 1.49 |
| 0.0240 | 12,167 ± 471 | 13,000 ± 408 | 16,167 ± 236 | 1.33 |
| 0.0235 | 18,000 ± 1,780 | 19,500 ± 408 | 23,167 ± 943 | 1.29 |
| 0.0230 | 30,000 ± 3,189 | 32,500 ± 1,080 | 37,333 ± 2,321 | 1.24 |
| 0.0228 | 37,167 ± 2,248 | 40,500 ± 408 | 49,000 ± 2,041 | 1.32 |
| 0.0226 | not reached | 59,500 ± 3,342 | 64,333 ± 2,656 | — |
| 0.0224 | not reached | 78,667 ± 4,989 | 106,833 ± 7,663 | — |

**Why this works where everything else failed.** It needs no fit, no L_∞, no asymptote, no threshold
inverted through a power law, and no claim that anything has converged. It is simply: *compare
networks that are equally good at the task.* It also absorbs the learning-rate difference
automatically — a network trained at lower lr just needs more iterations to reach the same loss,
which is exactly what a fair matching should charge it.

**The ratios are stable**, which is the property that matters: T(500)/T(100) = 1.07–1.28 and
T(1000)/T(100) = 1.24–1.49 across the whole usable range of target losses. So the *relative* budgets
are well determined even though the absolute budget depends on which loss level is chosen.

**Recipe for the M\* comparison.**
1. Choose a target loss every size reaches with margin — **0.0230** is a good default (all sizes,
   ~30–37k iterations, comfortably inside every run).
2. Read M(N) — the active-unit count — at each size's own T_N for that target.
3. Repeat at several target losses (0.0240, 0.0235, 0.0230, 0.0228) and confirm the
   saturating-versus-growing **verdict** is unchanged. That is the robustness check that makes the
   arbitrary choice of level harmless.
4. Report alongside each M(N) its movement per budget doubling, as the sensitivity.

Implemented as `matched_performance_budget()` in
`trainRNNbrain/experiments_and_analysis/plot_loss_fit.py`. Every number above comes from traces
already on disk — no re-run, and the 200k budgets are ample, since matching happens at 30–50k.

### Evidence that the loss floor is shared across sizes (and one caveat about its value)

Figure: `img/internal_figures/shared_floor.png`, from
`trainRNNbrain/experiments_and_analysis/plot_shared_floor.py`.

**Why a naive comparison would not have settled it.** L_∞ comes from extrapolating
`L(t) = L_∞ + A·t^(−γ)`, and that extrapolation is biased by how much data it sees — fitting a curve
that has not yet flattened systematically *underestimates* the asymptote. N=100 had 50k iterations
and the others 200k, so their raw L_∞ estimates are not comparable, and the apparent "N=100 has a
lower floor" was entirely this artefact.

**The controlled test.** Refit every size using only its first `t_max` iterations, so the same bias
applies to all of them, and ask whether the estimates agree:

| t_max | N=100 | N=500 | N=1000 | max pairwise diff |
|---|---|---|---|---|
| 20,000 | 0.01954 ± 0.00163 | 0.01905 ± 0.00143 | 0.01952 ± 0.00111 | 0.00048 (0.2 sd) |
| 30,000 | 0.02025 ± 0.00069 | 0.02044 ± 0.00062 | 0.02036 ± 0.00056 | 0.00019 (0.2 sd) |
| 50,000 | 0.02086 ± 0.00039 | 0.02080 ± 0.00013 | 0.02086 ± 0.00016 | 0.00007 (0.2 sd) |
| 75,000 | — | 0.02102 ± 0.00010 | 0.02101 ± 0.00013 | 0.00001 (0.1 sd) |
| 100,000 | — | 0.02111 ± 0.00007 | 0.02118 ± 0.00012 | 0.00007 (0.5 sd) |
| 150,000 | — | 0.02127 ± 0.00003 | 0.02131 ± 0.00008 | 0.00004 (0.5 sd) |
| 200,000 | — | 0.02135 ± 0.00003 | 0.02140 ± 0.00005 | 0.00005 (0.9 sd) |

**At every matched data length the sizes agree to within 0.1–0.9 standard deviations.** A tenfold
change in N moves the floor by at most 0.00005 out of 0.0214 — **under 0.3%**. The falsification test
was explicit: if bigger networks reached a lower floor, the curves in panel (a) would separate as N
increases. They lie on top of each other.

**Honest note on the residual.** At the three largest t_max the N=1000 estimate is above N=500 every
time (+0.00007, +0.00004, +0.00005). Individually all are ≤0.9 sd, and three same-signed differences
from 3 seeds is not significant (p ≈ 0.25 two-tailed), but it is a consistent sign and should not be
hidden. If real it would mean bigger networks are *marginally worse*, not better — the opposite of the
confound we were guarding against. Either way the magnitude (<0.3%) is far too small to affect an
M(N) comparison where the active count changes several-fold.

**The caveat that does matter: the VALUE of the floor is not determined.** Panel (a) shows the
estimate still climbing with t_max — 0.0195 → 0.0202 → 0.0209 → 0.0210 → 0.0211 → 0.0213 → 0.0214 —
with no sign of levelling. So **L_∞ ≳ 0.0214 and rising with fit range**; quoting 0.0214 as a
converged asymptote would be wrong, and any statement of the form "X% of the loss is irreducible"
inherits that uncertainty (the irreducible share is a *lower* bound; the reducible share an upper
bound).

Crucially, **the shared-floor conclusion does not depend on knowing the value**, because the
comparison is controlled: whatever bias remains applies equally to every size. That is the entire
point of matching on t_max.

**Pre-registered prediction for N=2000** (in progress, ~11 h out): refitted at t_max = 200,000 it
should give L_∞ within about 0.0001 of the 0.02135–0.02140 that N=500 and N=1000 give. If it comes in
materially lower, the shared-floor claim fails and the M(N) comparison becomes confounded by
performance. Stated now, before the data exists.

### The loss in power-law coordinates, without assuming the asymptote

Figure: `img/internal_figures/powerlaw_coords.png`, from
`trainRNNbrain/experiments_and_analysis/plot_powerlaw_coords.py`.

**The circularity to avoid.** The conventional linearisation of `L = L_∞ + A·t^(−γ)` plots
`log(L − L_∞)` against `log t`. It does give a straight line — but L_∞ is a fitted parameter chosen
to make the power law fit, so its straightness is partly guaranteed by construction. Panel (c) is
that plot, labelled as such.

**The L_∞-free coordinate.** Differentiating with respect to log t removes the constant entirely:

`−dL/d(log t) = A·γ·t^(−γ)`  ⟹  `log(−dL/dlog t) = log(Aγ) − γ·log t`

so the log-log slope is exactly **−γ** with no asymptote anywhere in it. In practice the pointwise
derivative of a noisy loss is unusable (the first version of this figure scattered over an order of
magnitude), so the same quantity is measured across a whole factor of two instead:

`L(t/2) − L(t) = A·t^(−γ)·(2^γ − 1) ∝ t^(−γ)`

— identical scaling, still no L_∞, far less noise. Panel (b) is straight over ~1.7 decades for every
seed at every size. **The power-law form is supported without assuming the asymptote.**

**γ does not depend on N.** Applying the lesson from the L_∞ analysis, the exponent is compared at
matched fit ranges:

| fit range | N=100 | N=500 | N=1000 |
|---|---|---|---|
| ≤ 50,000 | 0.468 ± 0.060 | 0.507 ± 0.016 | 0.500 ± 0.059 |
| ≤ 100,000 | — | 0.563 ± 0.019 | 0.561 ± 0.025 |
| ≤ 200,000 | — | 0.592 ± 0.025 | 0.609 ± 0.024 |

At matched range the sizes agree within error. The apparent "γ increases with N" in the unmatched
table was the same run-length artefact that produced the apparent L_∞ difference.

**But γ drifts upward with fit range — so it is not a single global power law.** 0.47–0.51 at 50k,
0.56 at 100k, 0.59–0.61 at 200k. A true fixed-exponent power law would give the same slope at every
range. The loss is decaying *faster* than a pure power law at late times, i.e. the approach to the
floor accelerates slightly.

Two consequences.
- It **explains** why L_∞ keeps rising with fit window: a fixed-γ fit forced onto a steepening decay
  compensates by pushing the asymptote down. The two biases are the same phenomenon.
- The three-parameter fit's γ (0.524, 0.541 at N=500, 1000) is systematically **below** the
  L_∞-free estimate (0.592, 0.609) — by ~0.07 in all six long-run seeds. The L_∞-free value is the
  one to trust, since it has no asymptote to trade against.

**Caveat.** Part of the late steepening could be an edge effect: the last points of panel (b) sit
below their fitted line, and they are the ones nearest the end of the trace. The N=2000 run (300k)
gives a longer lever arm and will show whether the drift continues or flattens.

### The local exponent: the loss is NOT a single power law, and earlier extrapolations were pessimistic

Figure: `img/internal_figures/local_exponent.png`, from
`trainRNNbrain/experiments_and_analysis/plot_local_exponent.py`. This measures
`gamma_eff(t) = −d log D / d log t` for `D(t) = L(t/2) − L(t)`, in a sliding factor-4 window — a
local slope, using no L_∞ anywhere.

| N | t=8k | t=15k | t=30k | t=60k | t=120k | t=180k |
|---|---|---|---|---|---|---|
| 100 | 0.411 ± 0.069 | 0.459 ± 0.097 | 0.727 ± 0.041 | — | — | — |
| 500 | 0.315 ± 0.095 | 0.484 ± 0.073 | 0.636 ± 0.083 | 0.672 ± 0.142 | 0.860 ± 0.203 | 0.960 ± 0.265 |
| 1000 | 0.324 ± 0.095 | 0.608 ± 0.032 | 0.601 ± 0.075 | 0.685 ± 0.068 | 0.799 ± 0.116 | 0.668 ± 0.212 |

**It is not an edge effect.** The figure was built to make that distinction: an artefact would show
gamma_eff flat until the last few points, whereas a real steepening shows it rising smoothly through
the middle. It rises monotonically across the whole span, roughly **0.3 → 0.8**, from t = 8k to
t = 120k. Nor can noise produce it: noise in log D adds variance to the slope estimate but no
systematic steepening.

**The sizes agree** (panel c, bands overlap throughout), so the *shape* of the decay is
size-independent — consistent with everything else about the shared trajectory.

**Consequence 1 — this explains two earlier puzzles at once.** A fixed-γ fit forced onto a steepening
decay must compensate by pushing L_∞ down, which is exactly why the estimated floor kept rising with
fit range, and why the three-parameter γ sat ~0.07 below the L_∞-free estimate. Both are one
misspecification: the model `L_∞ + A·t^(−γ)` has a constant exponent and the data do not.

**Consequence 2 — the pessimistic extrapolations in this document should be softened.** Every "you
would need N million iterations" estimate assumed a fixed γ ≈ 0.5, and a decay that steepens beats a
fixed power law by a growing margin. Concretely, halving the remaining reducible loss costs
`2^(1/γ)` in extra training: **4× at γ = 0.5, but only 2.2× at γ = 0.9**. The following claims are
therefore upper bounds, probably substantial ones, and should not be quoted as-is:
- "a 1% criterion extrapolates to 0.5–5.6 M iterations"
- "T(f = 1% of amplitude) = 233k–355k"
- the panel-(c) budget projections in `loss_fit_N*.png`

**What does NOT change.** The shared-floor conclusion, because it was established by a *matched*
comparison in which any misspecification applies equally to every size. The performance-matching
recipe for T(N), because it uses no fit at all. And the fact that participation keeps moving after
the loss has effectively settled, since that rests on measured per-doubling movement, not on
extrapolation.

**Open.** The functional form is unidentified — a steadily rising log-log slope is consistent with a
stretched exponential or similar, but this run does not have the range to pin it down, and it is not
needed for any current conclusion. The N=2000 300k trace extends the lever arm by another 0.2 decades
and will show whether gamma_eff keeps climbing or levels near 1.

### Where did "power law" come from? A correction to the framing used throughout this document

**Direct answer: the loss decay is NOT a power law, and this project's own data rejects it.** The
form `L = L_∞ + A·t^(−γ)` was an assumption introduced during this analysis, not a result. It was
never established here, and the local-exponent measurement falsifies the constant-exponent version:
a power law requires gamma_eff(t) to be flat, and it climbs 0.3 → 0.8.

**Why it was a reasonable first guess.** Near a minimum, gradient descent on a quadratic decays each
Hessian eigendirection as `exp(−η·λ_i·t)`. A network has a *broad spectrum* of curvatures, and a sum
of exponentials with a power-law-distributed set of rates looks like a power law over any window in
which many modes are still decaying. This is the standard spectral account of learning curves
(Bordelon, Canatar & Pehlevan 2020, on spectrum-dependent learning curves in kernel regression and
wide networks; Canatar et al. 2021; Bahri et al., "Explaining Neural Scaling Laws"). *Cited from
memory — verify before use.*

**And that same account predicts the steepening we observe.** The power-law phase is transient: it
lasts while a broad range of modes is still relaxing, and must steepen once only the slowest modes
remain. Our gamma_eff rising at late times is what the spectral picture predicts, not an anomaly.

**What the literature actually established, and where.** The well-known power laws — Hestness et al.
2017 (Baidu); Kaplan et al. 2020 (OpenAI); Rosenfeld et al. 2019; Hoffmann et al. 2022 (Chinchilla) —
are scaling of loss with **model size, dataset size and compute**, in large language models, image
classifiers, machine translation and speech. They are *not* a general law for how loss decays with
training iteration in a small recurrent network. Note also that theory does not uniformly predict
power laws at all: in the NTK/linearised regime (Jacot et al. 2018; Du et al. 2019) the training loss
converges **exponentially**.

**Applicability to this project is weak.** These runs use `same_batch=True` on a fixed 450-condition
batch, so there is no dataset-size scaling whatsoever — it is pure optimisation on a fixed objective
plus injected noise, in a 100–2000 unit RNN. The LLM scaling-law literature is at best a loose
analogy.

**What our data can and cannot distinguish.** A stretched exponential,
`L − L_∞ = A·exp(−(t/τ)^β)`, predicts `gamma_eff = β[(t/τ)^β − 1]`, which rises — matching the
observation. Fitting it to gamma_eff(t) gives R² = 0.25–0.67 against 0.00 for a constant exponent, so
"rising" beats "flat". But β = 0.10–0.25 and τ = 1–696 across sizes: the parameters are wildly
unstable and the form is **not identified**. Supported: *faster than any fixed power law*.
Not supported: any specific alternative.

### Consequence for the "six metrics failed" summary

That summary rests partly on "every quantity here relaxes as a power law, and for a power law the
fraction of remaining distance covered per doubling is constant, so 'converged' has no meaning".
**That argument is weakened for the loss**: with gamma_eff rising from 0.3 to 0.8, the fraction of
remaining reducible loss removed per doubling goes from ~19% to ~43% — increasing, not constant. The
loss converges *faster* than the argument assumed.

**But the participation conclusion is unaffected**, because it never depended on the power-law gloss.
The 28%-per-doubling figure for `p` is a *direct measurement*, flat across the whole late run, with no
fit and no assumed functional form. The metrics failed for the reasons individually documented —
lag-dependent floors, momentum artefacts, magnitude-blindness, denominator choice — and those stand.
What should be dropped is the tidy theoretical story that they all failed *because* of power laws.

---

## THE PROTOCOL: how the sizes are matched (definitive; supersedes attempts 1-7)

**Criterion: train each size until its training loss reaches the same value L\*.**

That is the whole rule. No convergence claim, no fit, no asymptote, no threshold inverted through a
decay law.

### Why equal loss is the right matching variable here

It is right *because the floor is shared*, which was measured rather than assumed (see the
shared-floor section: L_∞ agrees across N=100/500/1000 to within 0.3% at matched fit length, 0.1–0.9
sd). The chain is:

> shared L_∞  ⟹  equal L means equal L − L_∞  ⟹  equal distance from the achievable floor
> ⟹  equal fraction of the trainable improvement completed.

Had the floor differed with N, raw loss would have been the *wrong* variable — a network with a lower
floor sitting at the same raw loss is further from its own optimum. The shared-floor measurement is
what licenses the simple rule, which is why that test was worth doing.

Note what is NOT needed: the *value* of L_∞ (still poorly determined, and rising with fit range), the
functional form of the decay (power law vs stretched — unresolved and irrelevant here), and any
statement about convergence. The rule uses only the raw loss trace.

It also absorbs the learning-rate difference automatically. lr = 1e-3·(100/N)^(1/3), so N=2000 trains
at 0.37× the rate of N=100; a slower network simply needs more iterations to reach L*, which is
exactly what a fair matching should charge it.

### The numbers

| L* | N=100 | N=500 | N=1000 |
|---|---|---|---|
| 0.0240 | 12,167 ± 471 | 13,000 ± 408 | 16,167 ± 236 |
| 0.0235 | 18,000 ± 1,780 | 19,500 ± 408 | 23,167 ± 943 |
| **0.0230** | **30,000 ± 3,189** | **32,500 ± 1,080** | **37,333 ± 2,321** |
| 0.0228 | 37,167 ± 2,248 | 40,500 ± 408 | 49,000 ± 2,041 |

`matched_performance_budget()` in `plot_loss_fit.py`.

### Robustness: the choice of matching method barely matters

Because the loss curves nearly coincide across sizes (within 1% at equal iterations), matching on
loss and matching on iterations give almost the same answer: T(1000)/T(100) = 1.24–1.49 and
T(500)/T(100) = 1.07–1.28. So the M(N) comparison is not sensitive to which is used. Loss-matching is
the principled choice; iteration-matching is a close approximation that happens to be defensible here
but would not be in general.

### Full procedure for the M* result

1. **Primary comparison at L\* = 0.0230**, i.e. ~30k / 32.5k / 37.3k iterations. Read the
   active-unit count M(N) at each size's own T_N from the stored participation traces.
2. **Repeat at L\* = 0.0240, 0.0235, 0.0228** and confirm the saturating-versus-growing *verdict* is
   unchanged. This is what makes the arbitrary choice of level harmless.
3. **Also report the matched-ITERATION comparison at the common 200k budget**, as a complementary
   view at a much later stage of training. The verdict must agree with (1)–(2); if it does not, M* is
   budget-dependent and that is the finding.
4. **Report both silent-unit criteria** (hard `p<1e-6` and scale-free `p<0.05·q95`), since at N=100
   they disagree by an order of magnitude and may not agree about saturation either.
5. **Attach two error bars to every M(N) point**: seed spread, and movement of M over the last budget
   doubling (the sensitivity, since M itself does not converge).

### What is still open

- Whether the deepest matched level can be pushed below 0.0226 for all four sizes. The N=100 top-up
  to 200k (running, ~1.5 h out) will let N=100 reach ≈0.0222 and unlock the deeper levels.
- N=2000 (~11 h out) must satisfy the pre-registered floor prediction (L_∞ within ~0.0001 of
  0.02135–0.02140 at t_max=200k). If it does not, the shared-floor justification fails for that size
  and its point on the M(N) curve becomes confounded by performance.

### How the matching levels L* are chosen (replaces the hand-picked set)

The levels used in the first pass (0.0240 / 0.0235 / 0.0230 / 0.0228) were picked by eye from the
range every size reaches. That is not a principle, and it has a concrete flaw: those four levels span
T ≈ 12k–50k, a factor of 4 — only two doublings — so they cluster at the deep end and leave the early
range unsampled. Four points in two doublings is over-sampling a narrow window.

Two constraints fix the range, one choice fixes the spacing, and the NUMBER of levels then follows
from the data rather than being chosen.

**Constraint 1 — the deepest level is the worst final loss across all seeds.** Every seed of every
size must actually reach L*, otherwise the comparison silently drops the seeds that did not, which is
survivorship. Currently `L_deep = 0.02273`, set by the slowest N=100 seed — i.e. by the shortest run
in the sweep. *(The N=100 top-up to 200k, running, removes exactly this bottleneck: it should reach
≈0.0222, moving L_deep to ≈0.0223 and unlocking one to two deeper rungs.)*

**Constraint 2 — the shallowest level must be past the transient.** Before the solution has formed,
active-unit counts reflect initialisation rather than training. Operationalised as T ≥ 5000 for the
slowest size. This one is a judgement call; the alternative (a fixed fraction of the total loss drop)
was rejected because it reintroduces the arbitrary L(0) reference that sank Attempt 7.

**Spacing — halve the reference size's budget at each rung.** log t is the natural axis, because
every quantity here decays as a power of t, and because it is the same axis as the per-doubling
sensitivity measure already in use. The reference is the largest size present.

| rung | T_ref | L* | T(N=100) | T(N=500) | T(N=1000) |
|---|---|---|---|---|---|
| 0 | 60,400 | 0.02273 | 40,267 ± 1,543 | 46,067 ± 3,866 | 54,867 ± 3,934 |
| 1 | 30,200 | 0.02334 | 18,733 ± 1,636 | 21,467 ± 2,217 | 26,600 ± 1,840 |
| 2 | 15,100 | 0.02417 | 11,400 ± 816 | 12,867 ± 340 | 14,600 ± 993 |
| 3 | 7,550 | 0.02574 | 5,067 ± 94 | 5,667 ± 189 | 6,467 ± 94 |

Four rungs spanning T ≈ 5k–55k, a factor of 11 (3.5 doublings) — considerably better coverage than
the hand-picked set, and the count is determined by the constraints rather than assumed.

**Use:** compute M(N) at every rung. If the saturating-versus-growing verdict is the same across a
factor of 11 in training time, it is not an artefact of where we chose to look. If it changes, that
change *is* the result and must be reported as such — the first pass already showed the ratio
M(1000)/M(500) moving from 1.64 to 1.30 across a narrower range, so this is a live possibility rather
than a formality.

**Finalise after the N=100 top-up lands**, since L_deep is currently pinned by the one short run and
will move once it does.

---

