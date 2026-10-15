## 2026-08-17 21:5x — N=500 complete: five times the units, identical loss

All 3 N=500 seeds finished (200k iterations, r² 0.867–0.873). Figures:
`img/internal_figures/loss_fit_N500.png` and `loss_fit_compare.png`.

### The loss curves are indistinguishable from N=100

| budget | N=100 | N=500 |
|---|---|---|
| 5,000 | 0.02564 ± 0.00032 | 0.02629 ± 0.00026 |
| 10,000 | 0.02460 ± 0.00008 | 0.02470 ± 0.00017 |
| 20,000 | 0.02350 ± 0.00008 | 0.02354 ± 0.00028 |
| 50,000 | 0.02266 ± 0.00005 | 0.02273 ± 0.00006 |
| 100,000 | — | 0.02237 ± 0.00003 |
| 200,000 | — | 0.02222 ± 0.00004 |

At every matched budget the two sizes agree to within about one standard deviation, and where they
differ at all it is **N=100 that is marginally lower**. r² agrees too (N=100: 0.870/0.873/0.898;
N=500: 0.867/0.871/0.873). Panel (a) of `loss_fit_compare.png` shows the curves lying on top of each
other over the whole overlapping range.

**Five times the units buys exactly nothing in task performance.**

### The fits, and why the N=500 ones are the trustworthy ones

| | L_∞ | γ | seed spread in L_∞ | L_∞ spread across fit windows |
|---|---|---|---|---|
| N=100 (50k) | 0.02086 ± 0.00039 | 0.47 ± 0.06 | 4.5% | 3.2–7.1% |
| N=500 (200k) | **0.02135 ± 0.00003** | 0.52 ± 0.01 | **0.3%** | 1.5–2.4% |

The prediction made when only N=100 was available — that a longer run would pin L_∞ far better —
held: the seed spread is **15× tighter** at N=500 and the fit-window spread roughly halved. The two
L_∞ estimates are consistent with a single common floor of **≈ 0.0213**; the lower N=100 value is
within its own (large) uncertainty and is most likely biased by extrapolating from a 4× shorter run.

At 200k, N=500 has **3.0–3.2%** of its loss still reducible.

### Why this matters for M*

If both sizes sit at the same floor, then **every network in this sweep is over-parameterised** —
the task's capacity requirement is already met at N=100, and probably well below. That reframes the
silent-unit phenomenon: silencing is what training does with capacity it does not need.

But it does *not* imply the network collapses down to the needed capacity. The earlier
population-distortion analysis measured **574 active units at N=1000** (h, no penalty), versus ~97 at
N=100 — so the active count grows several-fold with N while task performance does not improve at all.
If that holds up when the N=1000 drift cell lands (which is the matched-protocol version of the same
measurement, so it must be re-checked rather than assumed), the sharpened statement is:

> the number of active units is not set by what the task requires — it is set by the network's own
> size and its optimisation dynamics, and the surplus units contribute nothing measurable.

That is a stronger and more interesting claim than "some units go silent", and it is testable
against exactly the M(N) curve this sweep produces.

**Caveat to carry:** the loss compared here is the *training* loss, which includes the injected noise,
so L_∞ is a genuine stochastic floor rather than an optimisation failure. That is the correct
interpretation of "both sizes are at the floor", but it also means the loss cannot distinguish
networks once they are both at it — which is precisely why the M(N) comparison has to be made on
active-unit counts and not on performance.

### Attempt 7 — normalise by the trainable amplitude: right instinct, but it inverts a power law

**Proposal.** Estimate the total trainable amplitude `A = L(0) − L_∞` and train until the remaining
reducible loss is below a fraction of it: `L(t) − L_∞ < f·A`, e.g. f = 1%.

**What is right about it.** It fixes the exact flaw identified in Attempt 5: dividing by the *total*
loss flatters convergence because 90–96% of that loss is an irreducible floor. Dividing by the
trainable amplitude instead is the honest normalisation, and as a *descriptive* statement — "this run
covered 99% of the loss the optimiser could ever remove" — it is scale-free, interpretable, and worth
reporting.

**Why it fails as a stopping rule.** Inverting `L(t) − L_∞ = A_fit·t^(−γ)` for t gives
`T ∝ (1/f)^(1/γ)` and `T ∝ A^(1/γ)`. With γ ≈ 0.5 both exponents are ≈ **2**, so every arbitrary
choice is squared.

*Sensitivity to the reference point defining L(0)* — the same seeds, same fit, only the choice of
which early iteration counts as "the start":

| N | seed | L(0) = iter 1 | iter 100 | iter 1000 |
|---|---|---|---|---|
| 100 | 0 | 11,276 | 108,598 | 1,144,691 |
| 100 | 1 | 30,517 | 587,913 | 13,834,023 |
| 100 | 2 | 19,452 | 369,835 | 5,841,236 |
| 500 | 0 | 18,245 | 248,718 | 1,927,711 |
| 500 | 1 | 14,362 | 233,784 | 1,894,168 |
| 500 | 2 | 12,533 | 217,127 | 1,915,397 |

A factor of **100** in the recommended budget, from a choice nothing in the problem determines.
(The loss falls steeply early, so L(0) is not a well-defined quantity: 0.294 at iteration 1, 0.100 by
iteration 100, ~0.05 by iteration 1000.)

*Sensitivity to the threshold f* (N=500, seed 0):

| f | 5% | 2% | 1% | 0.5% | 0.1% |
|---|---|---|---|---|---|
| T | 11,715 | 66,714 | 248,718 | 927,251 | 19,686,006 |

A factor of 20 in f becomes a factor of **1700** in T.

*And it needs a long run to be usable at all.* T(f=1%) is 355,449 ± 195,944 at N=100 (55% seed
spread, because L_∞ and γ are poorly determined from 50k) versus 233,209 ± 12,903 at N=500 (5.5%).
Circular: a reliable estimate of how long to train requires having already trained long.

**The general lesson, which applies to every criterion of this family.** Any rule of the form "train
until quantity X falls below a threshold", where X decays as a power law, must invert that power law
and therefore inherits an exponent of 1/γ ≈ 2 on both the threshold and any scale factor. It
*amplifies* arbitrariness. The per-doubling rule does not invert anything — it reads a local rate —
which is why its output moves roughly linearly with the threshold rather than quadratically. That is
the structural reason to prefer it.

**Verdict.** Keep amplitude-normalised remaining loss as a *reported* number (it is the honest way to
say "the optimisation is essentially complete"). Do not use it to choose the budget. And note that
even a perfectly well-defined loss criterion would still answer the wrong question here, since the
loss finishes long before the statistic being published does.

---

