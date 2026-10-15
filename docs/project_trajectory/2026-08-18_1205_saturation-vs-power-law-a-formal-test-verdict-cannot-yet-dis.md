## 2026-08-18 12:05 — Saturation vs power law: a formal test. VERDICT: cannot yet distinguish.

Figure: `img/internal_figures/saturation_test.png`, from
`trainRNNbrain/experiments_and_analysis/test_saturation.py`. Criteria fixed before running:
curvature significant at p < 0.05; model comparison by AICc with dAICc > 4 substantial, > 10 decisive.

**N=100 is excluded from every fit.** At N=100 the networks sit at 94–99% active, hard against the
M ≤ N ceiling — a 100-unit network cannot show that it "wants" 120 active units. That censoring
flattens M(100), inflates the apparent exponent from 100 to 500, and manufactures exactly the
downward curvature that saturation predicts. Only N ≥ 500 (all below 78% active) is fitted. This is
why the earlier three-size figure looked like it was bending.

### Test 1 — pooled curvature: NOT significant

`log M = a_level + k·log N (+ c·(log N)²)`, one intercept per matching level, curvature shared across
levels (which is what buys the degrees of freedom a single level cannot).

| criterion | k (straight) | c (curvature) | F(1,22) | p | verdict |
|---|---|---|---|---|---|
| hard | 0.389 | −0.081 ± 0.063 | 1.65 | 0.212 | not significant |
| scale-free | 0.384 | −0.030 ± 0.043 | 0.48 | 0.496 | not significant |

Both point estimates of c are **negative**, the direction saturation predicts, but neither is
distinguishable from zero. The data over N ∈ [500, 2000] are consistent with a straight line in
log-log, i.e. a pure power law. **This is a failure to detect curvature, not evidence of its absence**
— the range spans only a factor of 4 and N=2000 has a single seed.

### Test 2 — explicit saturating fits: mixed, no consistent winner

dAICc = AICc(saturating) − AICc(power law); positive favours the power law.

| L* | hard: hyperbolic | hard: exponential | scale-free: hyperbolic | scale-free: exponential |
|---|---|---|---|---|
| 0.02226 | +2.1 | +6.0 | −1.2 | +2.9 |
| 0.02253 | −2.4 | −1.2 | +0.3 | +2.6 |
| 0.02301 | +0.4 | +2.6 | +3.8 | +7.3 |
| 0.02369 | **−11.3** | **−13.6** | +4.0 | +9.0 |

Eight of sixteen comparisons favour each model; one (hard, shallowest level) is decisive *for*
saturation while the same level under the other criterion is substantial *against* it. No consistent
winner.

**The apparent M\* upper limits are an artefact and must not be quoted.** The profile-likelihood
95% upper bounds look reassuringly tight (e.g. 418, upper 480 at the deepest level) but that
tightness is manufactured by the short N range: with three sizes spanning a factor of 4, a saturating
curve can always be fitted with the ceiling placed just above the last data point. A ceiling
extrapolated from a factor-of-4 range is not credible.

### Test 3 — identification of k: clean

Bootstrap over seeds, N ≥ 500:

| L* | k, hard (95% CI) | k, scale-free (95% CI) |
|---|---|---|
| 0.02226 | **0.260** [0.184, 0.328] | **0.315** [0.264, 0.409] |
| 0.02253 | 0.327 [0.265, 0.460] | 0.387 [0.311, 0.511] |
| 0.02301 | 0.420 [0.370, 0.527] | 0.384 [0.297, 0.451] |
| 0.02369 | 0.549 [0.488, 0.669] | 0.450 [0.417, 0.516] |

k is well identified (CI width ≈ 0.15) and **depends systematically on the matching level**: deeper
training gives a smaller exponent, 0.55 → 0.26 under the hard criterion. So "the" exponent is not one
number — it is a function of how far the networks are trained, which is itself a result and matches
the earlier finding that saturation is something training produces rather than an architectural fact.
Pooled across levels, k ≈ 0.39 for both criteria.

### The sharp discriminator: what N=5000 will decide

Both models fitted on N = 500, 1000, 2000 only, then extrapolated. N=5000 is not yet on disk.

| L* | criterion | power law predicts | saturating predicts | gap |
|---|---|---|---|---|
| 0.02226 | hard | **471** | **393** | 78 |
| 0.02253 | hard | 642 | 526 | 116 |
| 0.02301 | hard | 1012 | 797 | 214 |
| 0.02369 | hard | 1513 | 1171 | 342 |
| 0.02226 | scale-free | 463 | 380 | 84 |
| 0.02369 | scale-free | 914 | 715 | 199 |

Seed-to-seed scatter at N=1000–2000 is 9–36 units, so a 78-unit gap at the deepest level is roughly
2–8 sigma — comfortably resolvable, and larger at every shallower level. **N=5000 extends the
ceiling-free range from a factor of 4 to a factor of 10 and will settle this.** First seed ~00:30
tonight; the remaining two ~22:30 on Aug 19. The two outstanding N=2000 seeds (~19:10 today) will
also tighten every k above and give that point real error bars.

---

