## 2026-08-17 17:59 — Drift sweep, N=100 cell complete: motion stops being *directed* by ~10⁴, but does not stop

All three N=100 seeds finished on Spock (array 5660818, 50k iterations each, r² 0.87–0.90).
Figure: `img/internal_figures/drift_N100.png`, produced by
`trainRNNbrain/experiments_and_analysis/plot_drift_curves.py`.

### The criterion that works, and the one that does not

Two readouts were meant to identify T(N), the iteration at which motion stops being systematic.

**The directional cosine fails as a criterion.** `cos(ΔW_t, ΔW_{t-1})` falls from 0.7–0.85 to a
floor of **0.32–0.35** by iteration ~850–1300 and then stays flat for the remaining 48k iterations.
The floor is positive for the reason anticipated when the metric was designed — consecutive
displacements are separated by only `track_every=10` iterations and Adam's momentum correlates them —
so once the curve is at its floor it can no longer distinguish "still marching" from "jittering".
It saturates an order of magnitude earlier than the real transition and must not be used to set T(N).

**The lag-scaling exponent works and needs no threshold.** With
`α = Δlog d / Δlog L` for `d(L) = ‖W(t)−W(t−L)‖_F / ‖W(t)‖_F`, the value 0.5 is the theoretical
random-walk exponent and 1.0 is straight-line motion — both are predictions, not tuning knobs.
Measured on W_rec between lags 100 and 1000:

| seed | α at iter 1000 | T(α = 0.5) | α at 50k |
|---|---|---|---|
| 0 | 1.10 | **12000** | 0.41 |
| 1 | 1.06 | **12000** | 0.40 |
| 2 | 1.05 | **9000** | 0.39 |

So **T(N=100) ≈ 1×10⁴ iterations**, ten times later than the cosine floor would have suggested.

### The finding: α is timescale-dependent — pinned at short lags, still diffusing at long ones

**Correction to the first reading of this figure.** The initial entry reported only the short-lag
exponent (0.39–0.41), read it as sub-diffusive, and concluded the weights sit in a confined
Ornstein–Uhlenbeck basin. Comparing both lag pairs **at the same iteration** (t = 40000, so the two
are not being read off different points of a decaying curve) does not support that:

| seed | α(100→1000) | α(1000→10000) |
|---|---|---|
| 0 | 0.37 | 0.52 |
| 1 | 0.42 | 0.55 |
| 2 | 0.37 | 0.53 |

α *increases* with lag. A confined OU process predicts the opposite — beyond its relaxation time
displacement saturates and α falls toward 0. So the weights are **not** in a closed basin. The
consistent reading is two regimes:

- **10²–10³ iterations: sub-diffusive (α ≈ 0.4).** Displacement over 1000 steps is only ~2.4× that
  over 100, less than the √10 = 3.16 a free walk would give. Successive steps partly cancel. Two
  candidate causes, neither tested here: local confinement at that timescale, or overshoot
  oscillation along sharp curvature directions (the "edge of stability" behaviour reported for
  Adam/SGD at practical learning rates).
- **10³–10⁴ iterations: diffusive (α ≈ 0.53).** Ratio 3.3–3.6 versus √10 = 3.16. At the longest lag
  measured the weights are still executing an unbounded random walk.

**What survives, and what does not.** T(N=100) ≈ 10⁴ stands: that is where the strongly *systematic*
phase (α ≈ 1, straight-line motion) ends, and it is the quantity the sweep was designed to extract.
What does not survive is the stronger claim that training has arrived somewhere. It has not — it has
stopped going anywhere *in particular*. "Undirected" is the defensible word; "converged" is not.

The functional readout is better behaved than the weights. For the participation vector at t = 40000,
**α_p(100→1000) = 0.04–0.08** — essentially no growth with lag, i.e. waiting ten times longer buys
almost no additional change in *p*. But α_p(1000→10000) = 0.58–0.69, mildly *super*-diffusive. So
even at N=100 after 50k iterations there is still a slow systematic component in how participation
evolves at the 10³–10⁴ timescale. The weights wander; *p* is pinned minute-to-minute but not
hour-to-hour.

Caveat: measured at N=100, the size that silences least. It does not license any claim at N=1000 or
N=2000, where the silencing rate is still visibly non-zero at 100k. The larger cells are the test.

### Silent units at N=100: transient silencing, then recruitment back

`silent_1em6` peaks **early** (15, 11, 3 units at iteration ~30–60) and then *declines* for the rest
of training, ending at **4, 3, 1** of 100 units — i.e. 96, 97 and 99 active. The per-unit
participation vectors say the same thing: active counts go 92 → 87 → **96**, 92 → 92 → **97**,
98 → 97 → **99**. Small networks do not exhibit the phenomenon: they transiently mute a few units in
the first ~50 iterations and then recruit them back.

This is the opposite of the N=1000 behaviour (65–76% silent at 100k) and is the first data point of
the M* curve: **M(100) ≈ 97**. One point cannot distinguish saturation from growth — that needs the
N=500/1000/2000 cells, which are still running (arrays 5660819/20/21, ~1:45 elapsed at the time of
writing).

### Secondary observation, not yet explained

In panel (b) the **output** weights drift most, and their relative drift at lag 1000 *rises* after
iteration ~2×10³ while W_inp and W_rec keep falling. The likely cause is the denominator: with
`weight_decay=1e-6` and no penalty holding W_out up, ‖W_out‖ shrinks, so a constant absolute step
becomes a growing relative one. Checking this needs the raw norms, which are not stored — worth a
scalar `norm_W_*` addition to the tracker if it matters later.

### The stopping criterion, stated properly: α_p at the longest lag crossing 0.5

The exponent gives a threshold-free stopping rule — "train until the motion is no longer directed",
i.e. until α falls stably below the random-walk value 0.5. Three things have to be pinned down before
that rule is usable, and the N=100 cell settles all three.

**1. It must name the lag.** α is lag-dependent, so "α < 0.5" is meaningless without saying at what
timescale. At N=100 the W_rec exponent crosses 0.5 at ~10⁴ for the 100→1000 pair but only at
~4–5×10⁴ for the 1000→10⁴ pair. Using the short lag declares victory 4× too early. **Use the longest
resolvable lag.**

**2. It should be measured on the participation vector, not the weights.** The weights are free to
move along functionally degenerate directions — W_out is doing exactly that (see the W_out section
below), sliding along the `W_out·r` rescaling invariance for the whole run without changing what any
unit computes. A weight-based rule therefore reports motion that has no functional consequence.
`p` is the quantity the silent-unit question is about, so `α_p` is the right criterion.

| N=100, α at the end of the run | 100→1000 | 1000→10000 |
|---|---|---|
| W_rec | 0.37–0.42 | 0.52–0.55 |
| **p** | **0.04–0.08** | **0.38–0.65** |

`α_p` at the short lag is ~0 from very early on: over 10²–10³ iterations the participation vector
does not move at all in the sense that matters, which is why it looked "converged" in every earlier
short-window analysis. At the long lag it starts **ballistic** (1.12–1.35 at iteration 10⁴ — still
straight-line motion) and only reaches 0.5 at **30–40k**.

**3. "Stably" is doing real work.** The short-lag α bounces between 0.3 and 0.5 after 2×10⁴, so the
rule needs "stays below for the remainder", not "first touch". At the long lag there are only four
measurements in a 50k run (one per 10⁴), so stability is not even assessable at N=100.

**Consequence: T(N=100) ≈ 3–4×10⁴, not 10⁴.** The earlier figure quoted 10⁴ from the short-lag
weight curve; on the criterion above it is 3–4× larger, i.e. the 50k budget for N=100 was only just
sufficient and the crossing sits at the edge of the run. The larger cells have 200k (N=500, N=1000)
and 300k (N=2000), giving 20–30 points on the criterion curve — enough to assess stability properly,
provided T does not grow faster than ~4× with N. That is now a thing to watch, not an assumption.

### Why the output weights appear to "speed up": a shrinking denominator

`d = ‖ΔW‖_F/‖W‖_F`. Under Adam the per-parameter step is roughly uniform within a matrix, so
`‖ΔW‖_F ≈ step·√P·f(L)` and `‖W‖_F ≈ w·√P` for P parameters and typical weight magnitude w — the
√P cancels and **d ≈ step·f(L)/w**. Parameter count is irrelevant; a rising d at fixed learning rate
means w is falling.

Measured directly, after adding `norm_W_*` to the tracker (commit a11e179): over the first 300
iterations at N=100, **‖W_out‖ 1.35 → 0.66** (halves) while ‖W_inp‖ 2.44 → 2.64 and
‖W_rec‖ 12.03 → 12.12 barely move. Two consistency checks from the existing traces: ‖p‖ grows
monotonically **8.8 → 18.4 (×2.1)** between iteration 1k and 50k, and d_out at L=1000 rises from
~0.20 at its minimum to ~0.40 — the same factor of 2.

Mechanism: the output is `W_out·r`, invariant under `W_out → W_out/c`, `r → c·r`. Nothing in the loss
pins the split, so the network slides along that degenerate direction throughout training. Note
weight decay is *not* the driver — wd=1e-6 cannot halve a norm in 300 steps; the task gradient is.
So W_out is not being updated faster in absolute terms, and panel (b) is a normalisation artifact —
but the artifact reveals a real, never-terminating slide.

**Hypothesis, untested:** median participation *falls* (0.57 → 0.37) while ‖p‖ *rises* (×2.1), so the
rate distribution is stretching — a few units get much louder, the typical unit gets quieter. This is
the same heterogeneity the population-distortion analysis measures, appearing here as a dynamic
process. Whether this slide is what terminates in silencing at large N is testable against the
N=1000/2000 cells and has not been tested.

### α resolved at every lag: it is not a number, it is a curve — and the network never settles

The two-point exponent uses the only lags the online tracker records. But the stored participation
vectors give `p` every 100 iterations, so the displacement can be time-averaged over all pairs in a
window at **every** multiple of 100 and the local slope read off directly. No re-run needed. (The
weight matrices are not stored, only scalar drifts, so the same trick is unavailable for W — see the
design note at the end.) Figure: `img/internal_figures/drift_msd_N100.png`.

**α_p(L) rises smoothly with lag.** Late in training (window 24k–49k) it is ~0.10 at L=100, crosses
0.5 at L ≈ 2500–3000, and reaches **0.85–0.95** by L=10⁴. So the two-point estimate over 1000→10⁴
(0.38–0.65, reported above) is **biased, not merely noisy**: it averages a curve that rises from
~0.15 to ~0.9 across that decade, and the average of a rising function sits far below its endpoint.
Any conclusion drawn from a two-point α should be re-derived from the resolved curve.

**The corrected reading: at long lags the participation vector is still nearly ballistic.** At
N=100, after 50k iterations, over 10⁴-iteration windows the network is still marching in a
consistent direction (α_p ≈ 0.9), not diffusing. It has not converged in any sense.

**And it is not rate inflation.** Repeating everything on `p̂ = p/‖p‖` — which removes the ×2.1
growth of ‖p‖ entirely — gives the same picture (α_p̂ ≈ 0.85–0.9 at L=10⁴). The *pattern* of who
participates is reorganising directionally, not merely scaling up.

### The reframing this forces: L*(t), and aging

If α depends on lag, "the iteration at which α crosses 0.5" is ill-posed. The well-posed quantity is
its inverse: **L\*(t) = the lag at which α_p crosses 0.5, measured at training age t.** Below L* the
participation vector is caged (changes cancel); above it, motion is directed. L* is "how long you
must wait before training takes the network somewhere new".

Measured on sliding 20k windows, 3 seeds, both definitions of the displacement:

| | scaling | L*/t |
|---|---|---|
| full `p` | L\* ∝ t^0.95 | 0.081 ± 0.009 |
| direction only `p̂` | L\* ∝ t^0.89 | 0.079 ± 0.008 |

**L\* grows in proportion to training age, at ≈ 8% of it.** The network is always still directed on
timescales longer than about a twelfth of its own age, however long it has trained. Train 10× longer
and the caging timescale stretches 10× with it. This is the **aging** signature of glassy relaxation:
there is no fixed relaxation time to wait out, because the relaxation time is set by how long you
have already waited.

This supersedes and mechanistically explains the earlier power-law extrapolation ("a 1% criterion
needs 0.5–5.6 M iterations"). The extrapolation is not a long wait — it is unreachable in principle,
because the criterion recedes as fast as you approach it. **"Trained to convergence" is not
available for these networks and should not be claimed anywhere in the paper.** What can be claimed
is a stated training budget plus the measured L*/t, which says exactly how much residual directed
change that budget leaves.

**Caveats, stated before the numbers are used.** The lever arm is short: t spans only 15k–40k, a
factor 2.7, so the exponent 0.89–0.95 is consistent with 1.0 but also with somewhat less; the
windows overlap, so points are not independent; and this is N=100 only. The N=500/1000/2000 cells
(200k–300k iterations) give a 4–6× longer lever arm and are the real test of whether L*/t is
constant, and of whether the constant depends on N.

**Design note for the next sweep.** To resolve α for the weight matrices as well, the cheap fix is
not more entries in `drift_lags` (each costs a full CPU-resident snapshot of every matrix) but
storing a fixed random projection of each matrix at every probe — ~200 floats instead of ~4 M at
N=2000. Johnson–Lindenstrauss keeps pairwise distances, so any lag becomes computable post hoc,
exactly as it already is for `p`.

### Correction: this is not "aging", it is a flat valley — and the loss HAS converged

The previous section framed L*/t ≈ const as glassy aging with no stationary regime. That framing is
wrong and should not be used. Two measurements settle it.

**The loss has essentially converged.** Over the entire second half of training (25k → 50k) the
training loss falls by **1.8–2.1%** (0.0231 → 0.0227, all three seeds). The optimiser is not still
descending in any meaningful sense.

**The participation vector has not.** Fitting the distance from the final state,
`D(t) = ‖p(T) − p(t)‖/‖p(T)‖ = A·t^(−γ)`, over the window [T/12, T/3]:

| seed | γ | remaining change after T=50k | iterations to halve it |
|---|---|---|---|
| 0 | 0.33 | 30.0% of ‖p‖ | ×8.2 |
| 1 | 0.35 | 29.7% | ×7.3 |
| 2 | 0.35 | 31.0% | ×7.3 |

`p` moved ~37% of its norm between iteration 25k and 50k while the loss improved 1.8%.

**Those two facts together are the explanation.** Large configuration change at negligible loss
change means the motion is **along a flat direction of the loss**, not down it. The minimum is a
manifold, not a point: the `W_out·r` rescaling degeneracy is one exactly-flat direction, and it is
measurably being traversed (‖W_out‖ halves in 300 iterations while the other matrices hold). Motion
along a flat valley has no restoring force to stop it, and because it is driven by a systematic bias
(weight decay, plus the small consistent component of the gradient) rather than by noise, it is
**directed** — which is exactly why α ≈ 1 at long lags.

So the network does reach the stationary regime one expects. It just is not a *point*: it is a
valley floor, and it keeps sliding along it.

**Why α made this look alarming: α is blind to magnitude.** α ≈ 1 says "there is a consistent
direction", not "a lot is happening". The amount is small and shrinking — displacement over a 10⁴
window is 16% of ‖p‖ late in training, down from 40% early. Every future report should pair α with
the *amount* of remaining change; α alone is misleading. The number to quote for a training budget
is the ~30% remaining change with γ ≈ 1/3, not L*/t.

**The Adam question, and a testable prediction.** Adam's update is `lr·m̂/(√v̂+ε)`. Along a direction
where the gradient is tiny but *consistent*, `m̂/√v̂ ≈ ±1`, so Adam takes a full `lr`-sized step where
SGD would take one proportional to the (tiny) gradient. Adam therefore traverses flat directions at
roughly constant speed while SGD would nearly stop. This is a substantial part of why directed
motion persists here. **Prediction: re-run one cell with SGD+momentum and the persistent directed
component should largely disappear — γ should rise and the remaining-change estimate should fall.**
That is a cheap, decisive experiment (one N=100 cell, ~2 h) and it has not been run.

### How to choose the budget: T(N) is defined on the LOSS; participation supplies the error bar

Figure: `img/internal_figures/drift_budget_N100.png`. The question is made budget-relevant rather
than metaphysical by asking, at each training age t, **what one more doubling of the budget buys** —
comparing t/2 with t. No fit, no extrapolation to infinity, no estimate of a limit.

| t | loss gain / doubling | Δp / doubling | of which reorganisation (Δp̂) |
|---|---|---|---|
| 2000 | 7.4% | 15.0% | 10.5% |
| 10000 | 5.0% | 19.4% | 17.3% |
| 20000 | 4.1% | 27.6% | 25.1% |
| 30000 | 3.1% | 28.7% | 26.7% |
| 50000 | 2.1% | 28.4% | 26.7% |

**The loss converges and gives a usable criterion.** Gain per doubling falls as t^−0.41, cleanly and
monotonically. Declaring training done when one more doubling buys **< 2%** gives

> **T(N=100) ≈ 9.7 × 10⁴ iterations** — i.e. the 50k budget actually used was about half of it.

**Participation does not converge and cannot supply a criterion.** Its change per doubling *rises*
to ~28% and then sits there. It is not decaying, so no threshold on it will ever be met, at any
budget. This is the flat-valley motion: ~26 of those 28 points are genuine reorganisation of `p̂`,
only ~16% is rate inflation. (This also retires the earlier "30% remaining, halve it with 8× more
training" estimate — that came from fitting a power law over a narrow window to a quantity that is
actually flat in log-time. Do not use it.)

**So the recipe is:**

1. **Set T(N) from the loss**: train until the loss gain over the last doubling drops below a stated
   threshold. Scale-free, interpretable ("doubling my compute buys < 2%"), measurable during the run
   without extrapolation, and comparable across N — which is what the T(N) curve needs.
2. **Do not wait for participation to stabilise.** It does not. Looking for that point is what made
   the last three analyses confusing.
3. **Quote the participation sensitivity as an error bar on every reported statistic.** Not on `p`
   itself, but on the number actually reported: silent fraction, PR, selectivity. For each, report
   its value at T *and* how much it moves between T/2 and T. That converts an unanswerable
   convergence question into a stated, honest sensitivity.

**Robustness check to run before quoting T(N):** compute T at 3%, 2% and 1% and confirm the *shape*
of T(N) versus N is the same. The threshold is a choice; the scaling with N must not depend on it.

**Caveats.** T(N=100) is extrapolated ~2× beyond the data (the fit reaches 2% at 97k, the run ended
at 50k), so it is a projection, not a measurement — the last few points sit slightly below the fit,
which would make the true T somewhat smaller. The N=500/1000 (200k) and N=2000 (300k) cells should
reach the criterion inside their budgets and will give measured rather than extrapolated values.

### Correction to the recipe above: the loss criterion is a denominator artifact

Fitting `L(t) = L_∞ + A·t^(−γ)` on all three N=100 seeds (t > 2000):

| seed | L_∞ | γ | loss at T | irreducible | still reducible |
|---|---|---|---|---|---|
| 0 | 0.02149 | 0.56 | 0.02261 | 95% | 0.00112 (5.0%) |
| 1 | 0.02113 | 0.48 | 0.02273 | 93% | 0.00160 (7.0%) |
| 2 | 0.02102 | 0.47 | 0.02265 | 93% | 0.00163 (7.2%) |

**93–95% of the loss at T is irreducible** — it is the noise floor of the task, not something more
training can remove. The "2% gain per doubling" figure divides by that floor, so it shrinks as the
floor comes to dominate, and the loss *looks* convergent.

Divide instead by what is still on the table and the picture inverts. For any power law, the
fraction of the *reducible* loss removed per doubling is `1 − 2^(−γ)` — with γ ≈ 0.5 that is a
**constant 29% per doubling, forever**. Exactly the behaviour that was called "does not converge"
for participation.

**The general point: no power-law-relaxing quantity ever converges in the fractional sense.** Asking
"when has it converged" has no answer for the loss, for `p`, or for the weights. It is the wrong
question, and chasing it is what generated three rounds of confusing analysis.

### The criterion that actually works: movement of the reported statistic, in its own units

T is not a property of the network. It is a property of **(network, statistic, precision required)**.
So define it that way:

> Train until **the number that goes in the paper** moves by less than a stated amount over the last
> doubling of the budget.

For this project that number is the silent fraction. If it moves less than ~1 percentage point from
T/2 to T, then "X% of units are silent" is budget-independent to the precision claimed, and no
statement about convergence is needed at all. The same test is then run for every other reported
statistic (PR, selectivity, σ_log), each in its own units, and each may give a different T — which is
honest, because a coarse claim needs less training than a fine one.

Advantages over both earlier proposals: it is in interpretable units (percentage points of silent
units, not fractions of a norm), it needs no fit and no extrapolation, it is directly the
sensitivity the reader wants, and it cannot be gamed by the choice of denominator.

At N=100 the silent counts are too small to test this (1–3 units, drifting up ~0.7 pp/doubling late
in training). The N=1000 and N=2000 cells, where the fraction is 65–76%, are where it can actually
be measured. **Supersedes the loss-threshold recipe in the section above; that section is kept for
the record but should not be used.**

---

