## 2026-08-25 (night, 3) — Diffusion read-out swept over 5 variables x 5 thresholds

New: [`flipflop_diffusion.py`](../trainRNNbrain/experiments_and_analysis/flipflop_diffusion.py).
`common` gained `drift_alpha_pairwise` (per-decade alpha), `scalar_alpha` (Hurst-type exponent of a
scalar trajectory, self-checked against a synthetic random walk and a pure ramp), and `_drift_key` so
`drift_alpha` also accepts `"p"` — the participation vector, logged as `dp_lag*`.

### The variables settle in a clear ORDER — a result in its own right

| variable | final alpha | seed sd | per-decade disagreement | settles at |
|---|---|---|---|---|
| W_out | 0.011 | 0.015 | 0.009 | ~5k (then frozen) |
| participation vector | 0.318 | 0.355 | **0.204** ⚠️ | ~25k |
| W_rec | 0.252 | 0.089 | 0.091 | ~40k |
| W_inp | 0.198 | 0.206 | **0.302** ⚠️ | ~100k |

**Output weights freeze first, then the participation structure, then the recurrent weights, and the
input weights keep moving longest** — consistent with W_inp being the matrix that grows 54x on this
task while W_rec barely moves.

⚠️ `W_out` and `M(t)` are UNUSABLE as read-outs: their alpha is already below every swept threshold
at the first probe, so `diffusive_onset` correctly returns nan (the "requires a prior directed phase"
guard added earlier). Reported rather than hidden.

⚠️ `W_inp` and the participation vector FAIL the single-power-law check (per-decade disagreement
0.302 and 0.204 against W_rec's 0.091), so a threshold on their alpha is averaging two regimes.
**W_rec is the best-founded drift variable** and should be the default.

### 13 trusted (variable, threshold) combinations, and they agree

Combinations scored by the seed CV of their read-out time; all 13 kept are CV 0.09–0.25.

| variable | thresholds | mean T | b | c |
|---|---|---|---|---|
| W_rec | 0.50–0.80 | 19k–41k | 0.45–0.47 | −0.010 … +0.072 |
| W_inp | 0.50–0.80 | 46k–73k | 0.44–0.46 | −0.031 … −0.018 |
| participation | 0.50–0.60 | 16k–22k | 0.47–0.52 | −0.043 … +0.001 |

**b = 0.465 ± 0.025, range [0.44, 0.52]. c = −0.009 ± 0.030, range [−0.043, +0.072].**

So across 13 independent diffusion read-outs — three different variables, five thresholds, read-out
times spanning 16k to 73k iterations — c is indistinguishable from zero everywhere except
W_rec@0.80 (+0.072, the shallowest read of all). **The k-independence of M at matched dynamics is not
an artifact of the one variable or the one threshold used before.**

⚠️ **An unexplained offset.** Diffusion read-outs give b = 0.465 ± 0.025 while the loss-based ones
give 0.41–0.43. The gap (~0.04) is larger than the spread within either family and does NOT track
read-out depth: W_inp reads DEEPER than the loss criteria (46k–73k vs 29k–41k) yet still returns the
higher b. Not resolved; flagged so it is not quietly averaged away.

### The settling time itself: T = A N^beta k^gamma

`flipflop_diffusion.py` gained `fit_settling_law`. This asks how the DYNAMICS depend on N and k,
rather than how the active-unit count does.

| variable | thresholds | beta (size) | gamma (complexity) | T(k=8)/T(k=1) |
|---|---|---|---|---|
| **W_inp** | 0.50–0.80 | −0.14 … −0.18 | **0.426 … 0.499** | **2.43x … 2.82x** |
| participation | 0.50–0.60 | −0.15 … −0.24 | 0.480 … 0.504 | 2.7x … 2.9x |
| W_rec | 0.50–0.70 | −0.21 … −0.37 | 0.158 … 0.430 | 1.4x … 2.4x |

**W_inp: gamma = 0.458 ± 0.027 across all five thresholds**, every CI excluding zero. Going from 1 to
8 bits lengthens the time the input weights keep moving directionally by **+160%**. The participation
vector independently agrees (0.48–0.50); W_rec is lower and threshold-sensitive.

**beta is NEGATIVE everywhere: larger networks settle FASTER**, matching the earlier finding that
t_half falls with N at fixed k.

### The mechanism test — and it passes quantitatively

If `c > 0` at fixed compute is *only* slower convergence at higher k, reading each network at a
k-compensated iteration `t ∝ k^gamma` must remove it. gamma comes from the WEIGHT drift and never
touches M or the participation vector, so this is not circular.

| read-out | b | c [95% CI] |
|---|---|---|
| fixed compute, t = 150 000 | 0.41 | **0.179 [0.150, 0.208]** |
| k-compensated, t = 50 000·k^0.458 | 0.41 | **0.001 [−0.039, 0.035]** |
| matched performance (reference) | 0.41 | −0.038 [−0.076, 0.004] |

**100% of the fixed-compute k-effect is removed**, and b is untouched at 0.41 in all three.

gamma is BRACKETED, not merely fitted — c is driven positive by too little compensation and negative
by too much, at every base time:

| gamma | c at base 30k | 50k | 80k |
|---|---|---|---|
| 0 (none) | +0.028 | +0.057 | +0.106 |
| 0.34 (W_rec) | −0.024 | +0.029 | +0.049 |
| **0.458 (W_inp)** | **−0.043** | **+0.001** | **+0.025** |
| 0.50 (participation) | −0.053 | −0.010 | +0.014 |
| 0.70 (over) | −0.087 | −0.050 | −0.027 |

⚠️ Note the uncompensated k-effect GROWS with read-out depth (c = 0.028 at 30k → 0.106 at 80k →
0.179 at 150k). Low-k networks converge and silence fully while high-k ones are still silencing, so
the gap widens with training. A shallow fixed-compute read understates the artifact.

⚠️ This is a CONSISTENCY result, not independent evidence. The compensated read-out is "matched
dynamics" implemented through a fitted scaling instead of per-run, so its agreement with the
per-run matched criteria confirms the mechanism is quantitatively sufficient — it does not add a
new independent measurement of c.

---

# ★ HEADLINE RESULT — matched on W_inp diffusion, active units depend on N and not on k

Script: [`flipflop_decisive.py`](../trainRNNbrain/experiments_and_analysis/flipflop_decisive.py)

![Active units depend on size, not complexity](../img/internal_figures/flipflop_decisive.png)

*Same networks, same measure, same silence rule — only the read-out time differs between rows. Top:
read at a fixed budget, where a k effect appears. Bottom: read where W_inp stops moving directionally,
where it does not. Panel (c) divides out the size dependence and looks for any k trend in what is
left: a clear rise at fixed compute (median 13 → 18.7), flat scatter at matched diffusion.*

## The comparison criterion: W_inp diffusion

Networks are compared at the iteration where the lag exponent of the INPUT-weight displacement,
|W_inp(t+L) − W_inp(t)| ~ L^α, falls below 0.6 and stays there — i.e. where the input weights stop
moving directionally and start merely jittering (α = 1 directed, 0.5 unbiased random walk, < 0.5
confined).

**W_inp is chosen because it moves the LONGEST of anything measurable.** Measured settling times:

| variable | settles at | note |
|---|---|---|
| W_out | ~5k | then frozen (α → 0.01) |
| participation vector | ~25k | |
| W_rec | ~19k–41k | |
| **W_inp** | **~35k–122k (mean 63k)** | the last thing still moving |

Matching at the W_inp point is therefore the most conservative dynamical read-out available: every
other variable has already settled, so nothing is being compared mid-flight. It needs no threshold on
the loss, no fitted floor, and no claim that the task has converged. 64/64 runs reach it.

## The result

| read-out | b (size) | c (complexity) | M(k=8)/M(k=1) |
|---|---|---|---|
| fixed compute, t = 150 000 | 0.405 [0.377, 0.434] | **0.179 [0.150, 0.208]** | 1.45× [1.37, 1.54] |
| **matched W_inp diffusion** | **0.441 [0.403, 0.477]** | **−0.026 [−0.078, 0.021]** | **0.95× [0.85, 1.04]** |

**Same networks, same measure, same silence rule — only the read-out time differs.**

- **M depends on N**: b = 0.44, CI comfortably excluding both 0 and 1. Sublinear, so added units are
  increasingly not recruited; the silent count N − M therefore grows with N as well.
- **M does not depend on k**: c = −0.026 with a CI straddling zero. An 8× increase in task complexity
  changes the active-unit count by −5% [−15%, +4%].

Panel (c) of the figure is the decisive one: dividing out the size dependence (plotting M/N^b) leaves
a clear upward trend at fixed compute (median 13 → 18.7 over k=1..8) and a flat scatter at matched
diffusion (median ~16.4, no trend).

## Why the fixed-compute k-effect is not evidence of task demand

The settling time itself scales as **T ~ N^(−0.16) · k^(0.458)** (W_inp, γ = 0.458 ± 0.027 over five
thresholds, every CI excluding zero): 8 bits keep the input weights moving 2.6× longer than 1 bit.
At any fixed budget, high-k networks are therefore proportionally less converged, and less-converged
networks hold more active units. Reading at a k-compensated iteration t ∝ k^0.458 — with γ taken from
the WEIGHT dynamics, never from M — removes **100%** of the fixed-compute effect
(c: 0.179 → 0.001 [−0.039, 0.035]) while leaving b untouched.

## Status

This is the outcome the k-sweep pre-registered: *"if M\* tracks task demand it must move with k...
Flat M\*(k) would mean a genuine pathology rather than spare capacity."* M\*(k) is flat under three
matched-performance criteria, five diffusion thresholds, three drift variables, and both silence
rules. The spare-capacity reading predicted the opposite and fails.

⚠️ Standing caveats: the effect is measured with the scale-free rule (or the task-calibrated absolute
rule, 4e-2 — not the CDDM-calibrated 1e-6, which is degenerate here); the N × k interaction under
matched loss (+7% at N=500, −15% at N=2000) is unresolved at 3 seeds/cell; and diffusion read-outs
give b ≈ 0.465 against the loss-based 0.41–0.43, an unexplained ~0.04 offset.

---

