## 2026-09-24, 15:17 — Activity dimensionality, and three non-copy revival rules built on it

### What duplication actually buys: units, not dimensions

The duplication rule is exactly function-preserving at the instant of replacement, so the obvious
worry is that it raises the active-unit count without changing the computation. Weight-space
evidence was mixed — no unit pair exceeds 0.99 cosine similarity, but 2.26% exceed 0.9 against
**0.00%** in controls. Activity dimensionality settles it.

Participation ratio of the firing-rate covariance over time and trials: 1 if every unit does the
same thing, N if all independent. All 21 networks passed the r² verification gate.

| N | arm | active units | activity dimensions | dims per unit |
|---|---|---|---|---|
| 500 | control | 220 ± 24 | 6.1 | 0.024 |
| 500 | duplication | 441 ± 30 | 7.3 | **0.013** |
| 500 | dead-dropout | 498 ± 1 | 29.2 | **0.058** |
| 1000 | control | 277 ± 26 | 6.3 | 0.020 |
| 1000 | duplication | 705 ± 28 | 9.1 | **0.013** |
| 1000 | dead-dropout | 913 ± 12 | 32.2 | **0.034** |

**Duplication is close to the worst case for this measure.** At N=1000 it takes the count from 277
to 705 (2.5×) while dimensionality goes 6.3 → 9.1 (1.4×), so dimensions per active unit FALL from
0.020 to 0.013. The recruited units largely ride directions the network already used.

**Dropout does the opposite.** `dead` reaches 32.2 dimensions on 913 units — five times the
control's dimensionality, and dimensions per unit RISE to 0.034. Its 12-point r² cost is therefore
not an inflated count; it finds a genuinely higher-dimensional solution that fits the task less
well.

Noise contributes almost nothing to these numbers (noise-on vs noise-off: 6.2/5.9, 9.2/9.1,
28.8/28.4), so this is signal rather than injected variance.

**Consequence: the bar changed.** "Recruit units cheaply" is already solved — duplication buys 428
units for 0.003 of r². The open problem is recruiting units that ADD DIMENSIONS, which an exact
copy cannot do by construction.

### These networks depend on their training noise

Found while debugging the reload. Evaluated without the noise they were trained with, the flip-flop
networks lose 0.15–0.19 of r² (0.944 → 0.75–0.80), consistently across every network. With noise on,
the recomputed values reproduce the stored ones to a few thousandths. They are not tolerating the
noise, they are using it, and any noise-free evaluation of this task family understates performance
by about 17%.

### Three non-copy revival rules (`14381695`, cells 8-11, 12 jobs)

All at rate 0.025 and maturity 1000, matching the `copy` cell so the rule is the only difference.
N=1000, 3-bit flip-flop, 40k iterations, gamma=0.

| cell | rule | property, verified in tests |
|---|---|---|
| 8 | `orth` | row orthogonal to every live unit's — cosine **1.9e-07** |
| 9 | `mix` | blend of 4 live units — max cosine **0.87** to any single donor |
| 10 | `bias_kick` | bias offset of −median(h); **no weight touched**, no donor |
| 11 | `random` | plain redraw at the same rate, as the floor |

`random` has never been run under these settings — its earlier failure (44 deaths per unit, ~1%
survival) was at an UNCAPPED rate, so it is not yet a fair floor.

⚠️ **`bias_kick` carries a prediction recorded before the run (Pavel's): it fails.** The recurrent
input that silenced the unit is still present and still training, so the network should re-suppress
it. If it treadmills where `copy` does not, the donor's FUNCTION is what makes duplication stick
rather than merely escaping the frozen state.

### Three bugs in `orth`, each caught by a test rather than by a wrong-looking number

1. **`torch.linalg.lstsq` assumes full column rank by default** and a live population's rows are
   rank-deficient, leaving cosine 0.155 against a live row. QR gives an orthonormal basis whatever
   the rank.
2. **Zeroing the revived units' outgoing columns after projecting** overwrites entries inside the
   new row. Columns are now zeroed first, then the projection runs.
3. **Restricting the projection to surviving columns is worse, not better.** With n_live ≥ n_keep
   the live rows span the whole restricted space, the residual is numerically zero, and normalising
   it turns rounding error into a full-size weight row. Measured: 23 live rows, 23 surviving
   columns, cosine 0.244. The projection runs over all N columns and falls back to a plain draw
   when the live rows already span R^N.

Two test contracts were also wrong: orthogonality holds against the population AFTER replacement,
not before (replacement changes every live row, giving a spurious 0.28), and a revived unit must
have gradient SOMEWHERE — `orth` and `mix` zero their outgoing weights, so nothing flows back into
them on the first step and they unfreeze via the outgoing route. That is exactly the mechanism
Dohare et al.'s maturity threshold exists to handle.

### 2026-09-24 20:29 — the non-copy revival rules landed: only duplication adds dimensions

Array `14381695` (12 jobs, N=1000, 3-bit flip-flop, replacement rate 0.025, maturity 1000) returned
all three seeds for each of `orth`, `mix`, `bias_kick` and `random`. Every network solved the task.
The read-out fixed before submission was active units AND activity dimensions per active unit, and
both are below. Dimensions are the participation ratio of the firing-rate covariance,
(sum of eigenvalues)^2 / sum of their squares: 1 if every unit does the same thing, N if all
independent. Networks were rebuilt from their own saved `config.yaml` and each one's r2 recomputed
and checked against the stored value; all 30 passed.

| rule | active per seed | mean | dims (clean) | dims / active | r2 |
|---|---|---|---|---|---|
| control | 254, 269, 301 | 275 | 5.6 | 0.0202 | 0.945 |
| copy | 668, 711, 728 | 702 | 9.3 | 0.0130 | 0.943 |
| bias_kick | 368, 370, 380 | 373 | 6.0 | 0.0161 | 0.952 |
| mix | 296, 341, 362 | 333 | 6.2 | 0.0185 | 0.948 |
| orth | 266, 295, 322 | 294 | 4.6 | 0.0159 | 0.948 |
| zero_out | 278, 278, 312 | 289 | 4.7 | 0.0162 | 0.946 |
| random | 250, 261, 275 | 262 | 5.0 | 0.0193 | 0.946 |
| scaling, row-norm eta=0.05 | 374, 417, 432 | 408 | 5.2 | 0.0134 | 0.943 |

**Three rules separate from the control on unit count, and only one adds dimensions.** Duplication
(668-728) and `bias_kick` (368-380) and synaptic scaling (374-432) have seed ranges that do not
touch the control's 254-301. `orth`, `zero_out` and `random` sit inside it, so on three seeds they
recruit nothing. Duplication is the only rule that raises total dimensionality at all, 9.3 against
the control's 5.6; every other rule leaves it between 4.6 and 6.2. `bias_kick` buys 98 units and
0.4 dimensions between them; synaptic scaling buys 133 units and LOSES 0.4.

**`orth` failed, and it was the rule designed to succeed.** Writing a new unit's incoming weights
in the orthogonal complement of the surviving population guarantees the weights are novel — the
test measured cosine 1.9e-07 against every survivor — and the network neither keeps the unit nor
uses more directions when it does. Novel weights are not a function. Whatever makes duplication
stick is the donor's role in the computation, not the escape from the frozen state, because
`bias_kick` escapes the frozen state without touching a weight and gains almost no dimensions.

**Pavel predicted `bias_kick` would fail** ("the suppressed unit with elevated bias will simply be
suppressed further") and it did not. It recruits more units than any rule except duplication, its
seed range clears the control's by 67 units, and it has the HIGHEST r2 of the eight cells, 0.952
against the control's 0.945. It does not treadmill. What it does not do is add dimensions.

**The noise-versus-clean gap flags units that respond only to noise.** Dimensions were measured both
with the network's training noise and without it. The ratio is 1.01 for duplication, 1.20 for the
control, and 1.88 for synaptic scaling (9.8 with noise against 5.2 without) — the largest of any
arm. The units synaptic scaling recruits are largely tracking injected noise, which is why its unit
count rises while its clean dimensionality falls below the control's.

### 2026-09-24 20:29 — synaptic scaling with gamma=0.1: one surviving corner, and it costs r2

Arrays `14365880` and `14365957` (12 cells). Nine runs ended at NaN with 55-67% of gradient updates
discarded, far past the 5% stability gate. The three healthy runs are all one cell, row-norm
preserving with eta=0.05, at 0.02%, 0.07% and 2.5% of updates discarded, r2 0.943, 0.943, 0.944
against the gamma=0.1 control's 0.952. So the cubic saturation rescues one corner of the scaling
rule rather than the rule. With the dimensionality read-out above, that corner recruits 408 units
whose clean dimensionality (5.2) is BELOW the control's (5.6). Synaptic scaling is finished as an
intervention; six configurations and a gamma rescue have not produced one that recruits units
carrying independent activity.

### 2026-09-24 20:29 — weight decay: 1e-6 already removes a quarter of the active units

Pavel asked whether stronger decay had been tested. It had not: the 2026-07-28 horizon experiment
covered 0 and 1e-6 only, and the trajectory recorded its active-unit read-out as never taken (T2).
Taking it now from the participation traces on /scratch, CDDM, 100000 iterations:

| N | weight decay | r2 | active per seed | mean |
|---|---|---|---|---|
| 1000 | 0 | 0.865 | 394, 428, 438 | 420 |
| 1000 | 1e-6 | 0.870 | 301, 314, 321 | 312 |
| 2000 | 0 | 0.863 | 580, 583, 594 | 586 |
| 2000 | 1e-6 | 0.873 | 420, 433, 449 | 434 |

A decay of 1e-6, small enough to read as a rounding choice, removes a quarter of the active units at
both sizes. The seed ranges do not overlap and r2 does not fall. Outside the explicit interventions
this is the largest single lever on the active count found in the project, and it had been sitting
unread in traces already on disk.

`14395238` adds 1e-3 and 1e-2 at the same two sizes, the same 100000 iterations and the same code
commit (`trainRNNbrain_horizon` at f5aa558; the launcher refuses to run from any other commit), so
the four-point series becomes one curve. 12 jobs, capped at 6 concurrent so the DMTS 7-tau array
keeps moving; about 74 GPU-hours from the original runs' measured elapsed times, 3:48-3:52 at
N=1000 and 8:21-8:51 at N=2000.

PREDICTION, fixed before submission: if the late silencing is decay-driven, both new cells sit below
the wd=1e-6 count — under 312 at N=1000 and under 434 at N=2000.
FALSIFIER: if 1e-3 and 1e-2 land inside the wd=1e-6 seed ranges (301-321 and 420-449), decay
strength beyond 1e-6 does nothing, the 0 -> 1e-6 step is a threshold rather than a dose, and weight
decay cannot be described as setting the count.
GATE: any cell with mean r2 below 0.7 has its active count reported but NOT interpreted. Adam here
takes `weight_decay` as coupled L2 added to the gradient (`torch.optim.Adam`, not AdamW) and lr is
scaled by (100/N)^lr_scale_exp, so 1e-2 against a base lr of 1e-3 is a real candidate for collapse —
and a collapsed network's participation distribution is meaningless under a relative criterion.
Measured directly while probing shrink-and-perturb: at r2 0.009, 996 of 1000 units cleared the 5% bar.

### 2026-09-24 20:29 — shrink-and-perturb: never tested here, and not inert as first argued

Pavel asked whether anything resembling shrink-and-perturb (Ash & Adams, NeurIPS 2020: every
parameter replaced by lambda*theta + noise) had been run. It had not. The closest rules are
`copy`, which multiplies a donor's outgoing column by 0.5 and jitters the copied incoming row by 5%;
`random`, which is lambda=0 restricted to flagged dead units; `mix`, whose Dirichlet blend shrinks
the norm without adding noise; and `synaptic_scaling_`, which is multiplicative but scales UP and is
targeted at silent units. Nothing global, nothing touching a working unit's own weights, nothing
applied repeatedly to the whole matrix. Dohare et al. used shrink-and-perturb as a loss-of-plasticity
baseline; this project took the continual-backprop side of that paper and skipped the baseline.

⚠️ MY FIRST ARGUMENT WAS WRONG. I reasoned that ReLU's positive homogeneity makes the shrink half
inert: scale every weight and bias by lambda, every pre-activation scales by lambda, and the set of
silent units cannot change. That holds in a feedforward network and fails in a recurrent one. The
recurrent drive `W_rec @ relu(h)` is bilinear in the weights and the rates, so it shrinks by
lambda^2 while the input drive `W_inp @ u` and the bias shrink by lambda. Checked against the
production forward pass: one step is exact to 2e-9, the second breaks at 1e-3. A unit held silent by
recurrent inhibition therefore sees its suppression fall 1/lambda faster than its drive, which is a
revival channel shrink-and-perturb would not have in a feedforward classifier.

Measured on the four trained N=1000 flip-flop controls, applied once and scored immediately:

| lambda | noise | active | vs base | r2 after | spectral radius |
|---|---|---|---|---|---|
| 1.0 | - | 262 | 1.00x | 0.954 | 8.26 |
| 0.9 | 0.3 | 326 | 1.24x | 0.857 | 7.44 |
| 0.8 | 0.3 | 366 | 1.40x | 0.598 | 6.61 |
| 0.6 | 0.3 | 534 | 2.03x | 0.179 | 4.96 |
| 0.4 | 0.0 | 897 | 3.42x | 0.057 | 3.30 |
| 0.2 | 0.0 | 996 | 3.79x | 0.009 | 1.65 |

⚠️ THE COUNTS AT SMALL LAMBDA ARE AN ARTIFACT, and the useful kind. The silence criterion is
scale-free, so a uniform rescale cannot move it; what moves it is the lambda^2-against-lambda split
flattening the participation distribution. At lambda=0.2, 996 of 1000 units clear the bar while r2
is 0.009. Any arm measured this way must report dimensions per active unit or the count means
nothing — the same trap duplication set, reached from the opposite direction. This row is now the
project's reference case for it, and it supplies the gate used on the weight-decay runs above.

Not yet run as a training intervention. The instantaneous trade at lambda=0.8-0.9 is 1.24-1.40x
recruitment for a 0.10-0.36 r2 debt that retraining would have to pay back, against duplication's
2.5x at no immediate debt — but it recruits by relieving inhibition rather than by cloning, so its
units are not copies by construction.

Two corrections to earlier entries:
* "These networks perform better with their training noise" held for the `ff_revive` networks and
  does NOT generalise. On the `std_dropout` controls the noise-free score is higher, 0.9644 against
  0.9537. The direction depends on the arm.
* Trained networks here end at spectral radius 8.26 from an initialisation of 1.2, measured on the
  four N=1000 flip-flop controls. Earlier entries that reason about stability from the init value of
  1.2 are reasoning about the wrong number.

