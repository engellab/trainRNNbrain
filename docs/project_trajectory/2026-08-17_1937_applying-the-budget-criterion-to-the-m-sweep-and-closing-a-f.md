## 2026-08-17 19:37 — Applying the budget criterion to the M* sweep, and closing a fairness gap

The point of the Spock sweep is M*: does the number of ACTIVE units saturate as N grows? That makes
the reported statistic the active count itself, so the per-doubling test from the previous section
applies to it directly — no separate convergence machinery is needed.

**Nothing has to be re-run.** The traces already hold the whole M(N, t) surface: `silent_1em6` is
recorded at every probe (every 10 iterations) and the full per-unit `p` vector every 100 iterations,
so the active count at any threshold can be reconstructed at any t up to each run's length. The
sweep was designed as "train, then count at the end"; it can be analysed instead as a family of
curves, which is strictly more informative and costs nothing.

**Two fairness confounds, both now identified.**

1. *Unequal budgets.* 50k / 200k / 200k / 300k for N = 100 / 500 / 1000 / 2000. These were set from
   extrapolated silencing rates — a reasonable guess at the time, but not a principled matching.
2. *Unequal learning rates.* `run_experiment.py:65` sets `lr = 1e-3 · (100/N)^(1/3)`, so
   lr(N=2000) = 3.68e-4 is **2.7× smaller** than lr(N=100) = 1e-3. Equal iterations therefore are
   not equal optimisation, independently of the budget issue. This is a further argument against
   matching on iteration count and for matching on a measured statistic.

| N | lr | × vs N=100 | budget |
|---|---|---|---|
| 100 | 1.000e-3 | 1.00 | 50k → **now also 200k** |
| 500 | 5.848e-4 | 0.58 | 200k |
| 1000 | 4.642e-4 | 0.46 | 200k |
| 2000 | 3.684e-4 | 0.37 | 300k |

**The analysis that avoids needing convergence at all.** Rather than picking one T and hoping every
size has settled there, evaluate the M(N) curve at several **common** budgets and ask whether the
*verdict* — saturating versus still growing — is the same at each. If M(N) saturates at t = 25k, 50k,
100k and 200k alike, the conclusion is budget-independent, which is stronger than any single matched
comparison and does not require M to have stopped moving. If the verdict flips with t, then M* is
budget-dependent and that is the honest finding.

Each point on the M(N) curve then carries **two** error bars: seed spread, and its own per-doubling
movement (the budget sensitivity). The saturating-vs-power-law fit must be judged against both.

**Gap closed.** N=100 had only 50k, so the largest budget common to all four sizes was 50k — the
smallest network would have set the resolution of the whole comparison. Submitted **array 5660933**
(3 seeds, N=100, 200k iterations, ~7 h each, 12 h wall). Verified running and writing to
`EqType=h_N=100_iters=200000`, a separate folder from the original 50k run, so nothing can mix.
Implemented via a one-line `ITERS_OVERRIDE` env hook in the launcher (the launcher is the only place
allowed to vary swept parameters); the run script still overrides nothing.

All four sizes now share a common 200k budget, with N=2000 additionally reaching 300k.

---

# 2026-08-17 21:02 — SELF-CONTAINED SUMMARY: there is no simple convergence metric, and why

*This section is written to be read on its own. It assumes no other part of this document. All
measurements are from the standard ReLU RNN, no penalties, N=100, 3 seeds, 50,000 iterations
(Spock array 5660818). Figures: `img/internal_figures/drift_N100.png`,
`drift_msd_N100.png`, `drift_budget_N100.png`, `silent_stopping_criterion.png`.*

## 1. Why we needed a convergence criterion

The scientific question is **M\***: does the number of *active* units in a trained RNN saturate as
the network gets bigger? If it saturates, then adding units past M\* buys nothing, and activity
penalties become the only route to a network where every unit does something. To answer it we train
networks at N = 100, 500, 1000, 2000 and count active units at each size.

That comparison is only meaningful if every size is trained "equally far". So we needed a rule for
when a network is trained enough — and, ideally, a rule that transfers across sizes. Note the sizes
do *not* even share a learning rate: `run_experiment.py` sets `lr = 1e-3·(100/N)^(1/3)`, so N=2000
trains at 3.68e-4 versus N=100 at 1e-3, a factor of 2.7. Equal iteration counts were never equal
optimisation.

The naive expectation — the one we started with, and the one most people have — is that a gradient
optimiser near a minimum reaches a stationary regime: the loss flattens, the weights stop going
anywhere, and everything afterwards is small noise-driven jitter. Finding that point should be easy.

**It is not. Six different metrics were tried. Every one of them failed, and each failed for a
different and instructive reason.**

## 2. Attempt 1 — absolute drift magnitude

**Metric.** `d(L) = ‖W(t) − W(t−L)‖_F / ‖W(t)‖_F`: how far the weight matrix moved over a window of
L iterations, as a fraction of its size. Declare convergence when d is small.

**Result.** d falls steeply, then flattens onto a **noise floor that is not zero and whose height
depends on L** — about 0.03 at L=100, 0.07 at L=1000, 0.23 at L=10,000. The L=100 curve is flat and
pure noise from iteration ~1000 onward.

**Why it fails.** Three reasons.
- The magnitude of d conflates *how big the steps are* with *how they combine*. Tiny steps all in
  the same direction (slow but going somewhere) and large steps that cancel (fast, going nowhere)
  give the same d. That distinction is exactly what we care about.
- "d stopped decreasing" is not "motion stopped". The L=100 curve flattened at iteration 10³ while
  directed motion demonstrably continued to 10⁴.
- Any threshold ("converged when d < 0.05") is arbitrary *and does not transfer across N*, because
  the natural scale of d turns out to be `step size / typical weight magnitude`, which depends on N.
  Since the whole point is to compare across N, a non-transferable threshold assumes the answer.

## 3. Attempt 2 — directional persistence (cosine)

**Metric.** The cosine between consecutive weight displacements. For an uncorrelated random walk the
expected cosine is exactly 0; while the trajectory keeps a consistent heading it is positive. So:
declare convergence when the cosine reaches 0. Appealingly, 0 is a *prediction*, not a tuned value.

**Result.** It never reaches 0. It falls from 0.7–0.85 to a floor of **0.32–0.35** by iteration
~850–1300 and sits there for the remaining 48,000 iterations.

**Why it fails.** The floor is an artifact of how it was recorded. Consecutive displacements are
separated by only `track_every = 10` iterations, and Adam's momentum (β₁ = 0.9, roughly a 10-step
memory) *explicitly* correlates successive updates. So even a completely settled network reports
cos ≈ 0.33 here. The null value is not 0; it is an unknown positive number set by the optimiser.

The evidence that the floor is an optimiser property and not a network property: **all three weight
matrices floor at the same value** (W_inp 0.30–0.33, W_rec 0.32–0.35, W_out 0.32–0.33) despite
starting at very different values (0.5 for W_rec, 0.85 for W_out). A floor reflecting real residual
structure would not be identical across three matrices of different shape and role.

Worse than an offset, this is a **dynamic-range** failure: the cosine hits its floor at iteration
~10³, while the exponent below shows directed motion continuing to ~10⁴. Using it would have put T
an order of magnitude too early.

*Fixable in principle* — measure the cosine between displacements separated by much more than the
momentum timescale (e.g. successive 1000-iteration displacements) and the artifact disappears. Only
the lag-10 version was stored, so this would need a re-run.

## 4. Attempt 3 — the lag-scaling exponent α

**Metric.** Measure `d` at several lags *at the same t*, and take the log-log slope
`α = Δlog d / Δlog L`. The physics is unambiguous:

| motion | steps combine | d(L) ∝ | α |
|---|---|---|---|
| every step the same direction | linearly | L | **1.0** |
| independent random steps | in quadrature | √L | **0.5** |
| steps partly cancel / confined | saturates | L⁰ | **→ 0** |

This is mean-squared-displacement analysis, standard in diffusion physics and single-particle
tracking. Its appeal: taking a ratio of d at two lags **cancels both the step size and ‖W‖**, so α
is dimensionless, threshold-free (0.5 and 1.0 are predictions), and comparable across N.

**Result, and the first real finding.** α is **not a number — it is a curve in L**. Estimated from
two lags on W_rec at a matched iteration (t = 40,000):

| seed | α(100→1000) | α(1000→10000) |
|---|---|---|
| 0 | 0.37 | 0.52 |
| 1 | 0.42 | 0.55 |
| 2 | 0.37 | 0.53 |

α *increases* with lag. That rules out the tidy interpretation (a confined basin, which predicts α
falling toward 0 at long lags).

**Two errors we made here, both worth recording.**

*Error 1 — reading one lag pair.* The first write-up reported only α(100→1000) ≈ 0.4, called it
sub-diffusive, and concluded the weights sit in a confined Ornstein–Uhlenbeck basin. Comparing both
pairs at the same iteration killed that.

*Error 2 — using a two-point slope at all.* The participation vector `p` is stored in full every 100
iterations, so its displacement can be time-averaged over **every** lag that is a multiple of 100,
and the local slope read off directly — no re-run needed. Doing that shows α_p rises smoothly from
~0.10 at L=100, through 0.5 at L ≈ 2500–3000, to **0.85–0.95 at L=10⁴**, late in training.

So the two-point estimate over 1000→10⁴ (which gave 0.38–0.65) was **biased, not merely noisy**: it
averages a curve rising from ~0.15 to ~0.9 across that decade, and the mean of a rising function
sits far below its endpoint. *Any conclusion drawn from a two-point α must be re-derived from the
resolved curve.*

**Why α fails as a criterion.** Two reasons.
- It is lag-dependent, so "α < 0.5" is meaningless without naming a timescale. At N=100 the crossing
  is at ~10⁴ for the short lag pair and ~4×10⁴ for the long one — a factor of 4.
- **α is blind to magnitude.** α ≈ 1 says "there is a consistent direction", not "a lot is
  happening". A tiny but consistent motion has α = 1. This is what made the results look alarming:
  the amount of motion is small and shrinking (displacement over a 10⁴ window is 16% of ‖p‖ late in
  training, down from 40% early) while α screamed "still directed".

*(One check worth keeping: repeating everything on `p̂ = p/‖p‖`, which removes the ×2.1 growth of
‖p‖ entirely, gives the same picture — α ≈ 0.85–0.9 at L=10⁴. So the persistent directed motion is
genuine reorganisation of who participates, not merely overall rate inflation.)*

## 5. Attempt 4 — the caging timescale L\*(t), and a wrong story about "aging"

**Metric.** If α depends on lag, invert the question: define **L\*(t) = the lag at which α_p crosses
0.5, measured at training age t**. Below L\* the participation vector is caged (changes cancel);
above it, motion is directed. L\* is "how long must I wait before training takes the network
somewhere new".

**Result.** L\* grows in proportion to training age: L\* ∝ t^0.95 (full `p`) and t^0.89 (`p̂`), with
**L\*/t = 0.081 ± 0.009** and 0.079 ± 0.008 respectively.

**What we concluded, and why it was wrong.** We read this as glassy *aging*: no fixed relaxation
time to wait out, because the relaxation time is set by how long you have already waited; therefore
the network never converges. That framing is wrong and should not be repeated. Two measurements
killed it (next section). It is recorded here because it was written into this document and into
three rounds of analysis, and because L\*/t ≈ 0.08 is still a correct *measurement* — it is only the
interpretation that was overreached.

## 6. Attempt 5 — the loss

**Metric.** The obvious one: stop when the loss stops improving. Made scale-free by asking what one
more **doubling** of the budget buys — compare the loss at t/2 with the loss at t. ("Doubling" is
the natural unit because the loss decays as a power law: improvement from 1k→2k is comparable to
10k→20k, not to 10k→11k. It is also literally the compute question.)

**Result.** Clean and monotone. Loss gain per doubling falls as t^−0.41:

| t | 2,000 | 10,000 | 20,000 | 30,000 | 50,000 |
|---|---|---|---|---|---|
| loss gain / doubling | 7.4% | 5.0% | 4.1% | 3.1% | 2.1% |

A 2%-per-doubling threshold gives T(N=100) ≈ 9.7×10⁴ — i.e. the 50k actually run was about half.
This looked like the answer.

**Why it fails: the denominator.** Fitting `L(t) = L_∞ + A·t^(−γ)`:

| seed | L_∞ | γ | loss at T | irreducible | still reducible |
|---|---|---|---|---|---|
| 0 | 0.02149 | 0.56 | 0.02261 | 95% | 5.0% |
| 1 | 0.02113 | 0.48 | 0.02273 | 93% | 7.0% |
| 2 | 0.02102 | 0.47 | 0.02265 | 93% | 7.2% |

**93–95% of the loss is an irreducible noise floor** — the task's own stochasticity, which no amount
of training removes. Dividing by it makes the gain shrink simply because the floor comes to
dominate. The loss only *looks* convergent.

Divide instead by what is actually still on the table and it inverts: for any power law the fraction
of the **reducible** loss removed per doubling is `1 − 2^(−γ)`, which with γ ≈ 0.5 is a **constant
29% per doubling, forever**. Exactly the behaviour we had called "non-convergent" for participation.

The criterion was not measuring convergence. It was measuring our choice of denominator.

## 7. Attempt 6 — the participation vector per doubling

**Metric.** Same doubling logic, applied to `p` directly: `‖p(t) − p(t/2)‖ / ‖p(t)‖`.

**Result.** It **rises** from 15% and then sits flat at ~28%, from t ≈ 20,000 to the end of the run.
Decomposed: ~26 points of that is genuine reorganisation of the pattern `p̂`, only ~16% is growth of
‖p‖ (which itself goes 8.8 → 18.4, ×2.1, while the *median* participation falls 0.57 → 0.37 — the
rate distribution is stretching, not uniformly inflating).

**Why it fails.** It is not decaying at all, so **no threshold on it is ever met, at any budget**.

## 8. The underlying reason all six failed

Two facts, and together they explain everything above.

**(a) Everything here relaxes as a power law, and no power-law quantity converges in the fractional
sense.** Loss, participation, weight drift — all of them. For a power law, the fraction of the
remaining distance covered per doubling of time is a *constant*. So "has it converged?" has no
answer for any of them; it is the wrong question. Chasing it is what produced attempts 1–6.

**(b) The loss has converged; the configuration has not, and need not, because the minimum is a
manifold rather than a point.** Over the entire second half of training the loss falls 1.8–2.1%
while `p` moves ~37% of its norm. Large configuration change at negligible loss change means the
motion is **along a flat direction of the loss, not down it**.

We measured one such direction explicitly. The network output is `W_out · r`, which is invariant
under `W_out → W_out/c` together with `r → c·r`. Nothing in the loss pins the split. Adding raw
norm tracking to the trainer shows ‖W_out‖ falling **1.35 → 0.66 in the first 300 iterations** while
‖W_inp‖ (2.44 → 2.64) and ‖W_rec‖ (12.03 → 12.12) barely move. Weight decay is *not* the driver —
wd = 1e-6 cannot halve a norm in 300 steps; the task gradient is. This also explains a puzzle in the
drift figure: the apparent late "speed-up" of W_out is a shrinking denominator, not faster updates.

Motion along a flat valley has no restoring force to stop it, and because it is driven by a
systematic bias rather than by noise it is **directed** — which is exactly why α ≈ 1 at long lags.

**The role of Adam.** Adam's update is `lr·m̂/(√v̂+ε)`. Along a direction where the gradient is tiny
but *consistent*, `m̂/√v̂ ≈ ±1`, so Adam takes a **full lr-sized step** where SGD would take one
proportional to the (tiny) gradient. Adam traverses flat directions at roughly constant speed while
SGD would nearly stop. This is likely a substantial part of why directed motion persists.
**Untested, and a cheap decisive experiment:** re-run one N=100 cell with SGD+momentum; if this is
right, the persistent directed component should largely disappear.

## 9. Where we landed

Stop asking whether the network converged. Ask whether **the number that goes in the paper** has
stopped moving, in its own units:

> **T = the first training age after which the reported statistic moves by less than a stated
> precision over a doubling of the budget, and stays there.**

For the M\* question that statistic is the silent fraction (equivalently the active count), and the
precision is **1 percentage point of N**. This is superior to all six attempts above because it is
in interpretable units, needs no fit and no extrapolation, is directly the sensitivity a reader
wants, and cannot be gamed by the choice of denominator.

**And where convergence cannot be reached, the fallback does not need it.** Evaluate the M(N) curve
at several *common* budgets and ask whether the **verdict** — saturating versus still growing — is
the same at each. Budget-independence of the conclusion substitutes for convergence of the quantity,
and is a stronger claim than any single matched comparison.

## 10. The result of applying it — and an unresolved problem

Applied at N=100 (`silent_stopping_criterion.png`), the two silent-unit definitions **disagree
sharply**:

| criterion | silent fraction at 50k | movement per doubling |
|---|---|---|
| hard, `p < 1e-6` | 2.6% | 0.93 pp, still rising |
| scale-free, `p < 0.05·q₉₅(p)` | **29%** | 14.7 pp, accelerating |

Neither reaches T. The mechanism for the gap is the distribution stretch noted above: ‖p‖ grows
while the median falls, so `q₉₅` rises while the bulk sinks, and a *relative* threshold sweeps up
ever more units. The scale-free criterion is partly tracking rate heterogeneity, not only silencing.

Two guards had to be added to the detector, both after it initially gave a wrong answer (T = 3013
for the hard criterion): a curve can sit below threshold merely because **the run ended** while it is
trending back up, and a threshold finer than one unit is **below the measurement resolution**
(at N=100, 1 pp = 1 unit). Both now report "NOT REACHED" with the reason.

**Open items.**
- The scale-free criterion may not support a budget-matched comparison at all if it never settles at
  any N. The multi-budget shape test decides this.
- The M\* verdict must be reported under **both** criteria. If they disagree about saturation the
  way they disagree about magnitude here, that disagreement is the finding.
- N=100 is permanently the weak cell for this test (1 pp = 1 unit). N=1000 and N=2000, where 1 pp is
  10 and 20 units, are where the criterion has real power.
- The Adam-versus-SGD experiment in section 8 has not been run.

## 11. One-paragraph version

We wanted a rule for "trained enough" so that networks of different sizes could be compared fairly.
Six candidate metrics all failed: absolute drift has a lag-dependent noise floor and an untransferable
threshold; the directional cosine has a positive floor set by Adam's momentum, not by the network;
the lag-scaling exponent is a curve rather than a number and is blind to *how much* is moving; the
caging timescale grows in proportion to training age; the loss appears to converge only because 93–95%
of it is an irreducible noise floor sitting in the denominator; and the participation vector's change
per doubling is simply flat at ~28%. The reason none of them work is that every quantity here relaxes
as a power law — for which the fraction of remaining distance covered per doubling is constant, so
"converged" has no meaning — and, more fundamentally, that the loss minimum is a flat manifold rather
than a point: the loss falls 2% over the second half of training while the configuration moves 37%,
because the network slides along directions the loss does not penalise (we measured one, the
`W_out·r` rescaling degeneracy, with ‖W_out‖ halving in 300 iterations). The resolution is to stop
asking about convergence and instead report how much **the specific number being published** moves
when the training budget is doubled — and, where even that does not settle, to show that the
scientific conclusion is the same at every budget.

### Loss fits for N=100, and how well L_inf is actually determined

Figure: `img/internal_figures/loss_fit_N100.png`, from
`trainRNNbrain/experiments_and_analysis/plot_loss_fit.py`. Fit is `L(t) = L_inf + A·t^(−γ)` on
log-binned medians (so late iterations do not dominate), started at t = 2000.

| seed | L_∞ | γ | L(T) at 50k | irreducible | reducible left | L_∞ spread across fit windows |
|---|---|---|---|---|---|---|
| 0 | 0.02135 | 0.55 | 0.02261 | 94.4% | 5.3% | 3.2% |
| 1 | 0.02041 | 0.40 | 0.02273 | 89.8% | 9.2% | 7.1% |
| 2 | 0.02083 | 0.45 | 0.02265 | 91.9% | 7.8% | 6.3% |

**The power law holds cleanly.** Panel (b) plots `L(t) − L_∞` on log-log; all three seeds are
straight lines over the whole fitted range, which is the goodness check for the functional form.

**But L_∞ is not tightly determined, and this must be stated wherever the decomposition is used.**
Re-fitting from t = 2000, 5000 and 10000 moves L_∞ by 3.2–7.1%. Since the reducible part is only
5–9% of the total loss, that uncertainty is **comparable in size to the quantity being estimated**
(e.g. seed 0: L_∞ uncertainty ≈ 0.00068 versus reducible ≈ 0.0012, so ~57% of it). An earlier fit in
this document, run on raw rather than log-binned data, gave L_∞ = 0.02149/0.02113/0.02102,
γ = 0.56/0.48/0.47 and "93–95% irreducible"; the difference between the two procedures is of the
same order as the window spread, which is itself a consistency check.

**Corrected statement:** roughly **90–94%** of the loss at 50k is irreducible, with real uncertainty
on the exact split. The qualitative conclusion — most of the remaining loss cannot be trained away,
so raw loss improvement flatters the apparent convergence — is robust. The precise percentage is not,
from a 50k run. The 200k and 300k cells will constrain L_∞ far better, since the lever arm on the
asymptote grows with run length.

**Seeds genuinely differ**: γ ranges 0.40–0.55 and the reducible remainder 5.3–9.2%. This is not
fit noise; it is real seed-to-seed variation in how fast each network approaches its own floor.

**What a longer budget buys (panel c), reducible loss remaining as a % of total:**

| budget | 50k | 100k | 200k | 300k | 1M |
|---|---|---|---|---|---|
| seed 0 | 5.3% | 3.7% | 2.6% | 2.1% | 0.9% |
| seed 1 | 9.2% | 7.2% | 5.5% | 4.7% | 3.0% |
| seed 2 | 7.8% | 5.8% | 4.3% | 3.6% | 2.1% |

**This confirms the recommendation not to re-run at 300k.** Going 200k → 300k moves the remaining
reducible loss from ~4.1% to ~3.5% on average — a 0.6 percentage point gain for 9 jobs of ~11 h each.
Even 1M iterations, a 5× extension, only reaches ~2%.

---

