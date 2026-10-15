## 2026-08-18 12:30 — ESTABLISHED: the loss floor does not depend on network size

Figure: `img/internal_figures/floor_vs_N.png`, from
`trainRNNbrain/experiments_and_analysis/test_floor_vs_N.py`. Criteria fixed before running: a size
dependence is accepted only if the slope differs from zero at p < 0.05 AND the best model beats the
constant by dAICc > 4 AND both hold at every fit length tested.

**Why this had to be settled first.** Matching sizes at equal training loss is only a valid notion of
"equally trained" if every size is heading for the same floor. If bigger networks could reach a lower
loss, two networks at equal loss would not be equally far from their own optima and every M(N) number
would inherit the confound. There is also a concrete mechanism that could produce a size-dependent
floor, so this is not a formality: lr = 1e-3·(100/N)^(1/3), and a noisy optimiser equilibrates in a
noise ball whose width grows with lr, so **larger N (smaller lr) should give a lower floor**.

**Result — the constant model wins at every fit length.** Four candidate shapes,
`L_inf = c` / `c + bN` / `c + b·log10 N` / `a·N^(-b)`, fitted on per-seed estimates, all sizes fitted
on the same first `t_max` iterations so the extrapolation bias is common:

| t_max | best model | dAICc: linear | log-linear | power law | size-term p |
|---|---|---|---|---|---|
| 50k | **constant** | +3.2 | +2.2 | +2.2 | 0.36–0.90 |
| 100k | **constant** | +3.2 | +2.1 | +2.1 | 0.368 |
| 150k | **constant** | +3.2 | +2.2 | +2.2 | 0.384 |
| 200k | **constant** | +2.4 | +3.2 | +3.2 | 0.429–0.976 |

Every size-dependent model is *worse* than the constant at every fit length, and no size term comes
close to significance. **Size dependence accepted at 0 of 4 fit lengths.**

**Stated as an equivalence bound, not as "no effect".** "Not significant" is not "equal", so the
right statement is how large an effect the data can still hide:

| t_max | slope per decade of N | bound on total change over N=100→2000 |
|---|---|---|
| 100k | −0.000073 ± 0.000149 | < 0.00019 = **0.92%** of L_inf |
| 150k | −0.000043 ± 0.000091 | < 0.00012 = **0.55%** |
| 200k | +0.000001 ± 0.000069 | < 0.00009 = **0.42%** |

So across a 20-fold range of N the floor is constant to better than half a percent, and the bound
tightens as more data enters the fit — the signature of a real null rather than of low power.

**Per-size values at t_max = 200k** (the tightest): 0.02141, 0.02135, 0.02140, 0.02145 for
N = 100, 500, 1000, 2000. The residual scatter is ~2–4× the within-size seed sd but is
**non-monotone** in N, which is why no trend model fits it — it is scatter in the extrapolation, not
a size effect.

**Model-free cross-check (panel c).** Raw measured loss at matched iteration count, no extrapolation
at all: it *increases* slightly with N at every checkpoint (e.g. at 200k: 0.02211, 0.02222, 0.02225,
0.02236). That is the expected consequence of the smaller learning rate at larger N, and it is the
opposite sign from "bigger networks reach a lower floor". Both the extrapolated and the measured
quantity therefore agree that nothing supports a decreasing floor.

**The lr/noise-ball mechanism finds no support.** Despite lr differing by 2.7× across the range, no
corresponding floor difference appears. Either the floor is set by task stochasticity rather than by
optimiser noise, or the effect is below 0.42%.

**Consequence.** Equal training loss = equal distance from a common optimum, so the matched-performance
protocol is licensed. This is now a stated fact of the project rather than an assumption.

**Caveats.** N=2000 rests on one seed (two more ~19:10 today), and the range is 1.3 decades; N=5000
extends it to 1.7 decades. The test re-runs unchanged on both, and the equivalence bound should
tighten further.

### Floor test, extended: in-progress runs recovered from logs, and the one marginal result explained

**Unfinished runs are usable for anything that needs only the loss.** `TrainLosses.json` is written
only on completion, but the per-iteration loss is printed to SLURM stdout as training goes, so an
in-progress curve is fully recoverable. Validated before use: parsing the log of a COMPLETED run and
comparing against its saved JSON gives a maximum difference of **5e-7**, i.e. exactly the 6-decimal
rounding of the printout. Runs that reached their declared max_iter are skipped so the JSON copy is
not double-counted. Implemented as `losses_from_logs()` in `test_floor_vs_N.py` (`--logs=DIR`).

This adds **2 more N=2000 seeds** (222k iterations each) and brings **N=5000 into the test** at 26–49k
iterations, extending the size range from 1.3 to 1.7 decades — today, rather than after Aug 19.

| t_max | N=100 | N=500 | N=1000 | N=2000 | N=5000 | slope p | best model |
|---|---|---|---|---|---|---|---|
| 25k | 0.02093 | 0.02018 | 0.01992 | 0.02003 | **0.01937** | **0.051** | power law |
| 50k | — | — | — | — | — | 0.570 | constant |
| 100k | — | — | — | — | — | 0.633 | constant |
| 150k | — | — | — | — | — | 0.623 | constant |
| 200k | 0.02141 | 0.02135 | 0.02140 | **0.02142** | — | 0.922 | constant |

N=2000 now rests on 3 seeds: **0.02142 ± 0.00016**, which meets the pre-registered prediction
(within ~0.0001 of 0.02135–0.02140) more comfortably than the single-seed 0.02145 did.

**The t_max = 25k anomaly is an artefact, and its mechanism is identified.** Matching the fit *length*
does not match the fit *content*: at a fixed iteration count the sizes are not equally far along,
because lr falls with N.

| loss actually reached at iteration 25,000 | N=100 | N=500 | N=1000 | N=2000 | N=5000 |
|---|---|---|---|---|---|
| | 0.02322 | 0.02339 | 0.02351 | 0.02375 | 0.02406 |

A bigger network is further from its floor at 25k, so its curve has flattened less, so its
extrapolated L_∞ is biased further **downward** — which mimics precisely a floor that decreases with
N. This corrects a claim made earlier in this document: fitting every size on the same t_max makes
the bias common only when all sizes are near their floor. At long t_max that holds; at 25k it does not.

**Control — fit each size on its own window ending at a COMMON LOSS**, so every fit sees the same
amount of curvature rather than the same number of steps:

| L* | N=100 | N=500 | N=1000 | N=2000 | N=5000 | slope p | best |
|---|---|---|---|---|---|---|---|
| 0.02450 | 0.01945 | 0.01767 | 0.01582 | 0.01882 | 0.01858 | 0.765 | constant |
| 0.02420 | 0.02035 | 0.01927 | 0.01820 | 0.01953 | 0.01842 | 0.223 | constant |
| 0.02400 | 0.02095 | 0.01941 | 0.01821 | 0.01959 | 0.01925 | 0.208 | constant |

Constant wins at every level and no slope approaches significance. The estimates are noisy (sd
0.0006–0.0034, since the windows are short), so this control has low power on its own — but it
removes the specific artefact and does not reproduce the trend.

**Verdict unchanged and now better supported: the loss floor is independent of network size.** Size
dependence accepted at 0 of 5 matched-length fits and 0 of 3 matched-progress fits; at t_max=200k any
change over N=100→2000 is bounded below **0.70%**.

### Correction: L_inf is model-dependent and the model is wrong — restate the floor result model-free

Fitting one N=1000 seed at four cut-offs gives L_inf = 0.01887 / 0.02070 / 0.02106 / 0.02135 and
gamma = 0.35 / 0.47 / 0.50 / 0.53 for t_max = 25k / 50k / 100k / 200k. Both parameters drift
**monotonically** with the fit window. A correctly specified model gives estimates that scatter about
the truth; ones that march in one direction mean the functional form is wrong. This is the same
misspecification found independently from the local exponent (gamma_eff rising 0.3 -> 0.8), reached
here from the opposite direction. Figure: `img/internal_figures/tmax_explainer.png`.

**Consequences.**

*Drop:* the absolute value of L_inf, and everything derived from it — including the repeated claim
that "93–96% of the loss is irreducible". That figure is model-dependent and should not appear
anywhere.

*Keep, restated without any fit:* the lowest loss each size actually reached —

| N | 100 | 500 | 1000 | 2000 |
|---|---|---|---|---|
| best loss reached | **0.02211** | 0.02222 | 0.02225 | 0.02227 |

No larger network achieves a lower loss than the smallest. No extrapolation, no functional form.

*Why the fitted comparison also survives:* the bias direction is known — less progress gives a lower
fitted L_inf. At t_max = 200k, N=100 is ahead (loss 0.02211) while every larger size is behind
(0.02222–0.02234), so the larger networks' L_inf estimates are biased **downward relative to N=100**.
They nonetheless come out equal. Correcting the misspecification would raise the large-N floors,
i.e. move further from "bigger reaches lower". The misspecification cannot be manufacturing the null.

*Unresolved:* whether a much longer run would let a large network overtake N=100. Extrapolation
cannot answer it; only training longer can. The residual gaps are 0.0001–0.0002, far too small to
move any M(N) conclusion.

*Unaffected:* the matched-performance protocol, which never used L_inf — it reads raw loss only, and
the shared-floor fact it relies on is now established without the fit.

---

