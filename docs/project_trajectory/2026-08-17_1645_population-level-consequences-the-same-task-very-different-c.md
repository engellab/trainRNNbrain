## 2026-08-17 (16:45) — population-level consequences: the same task, very different circuits

The analysis that decides whether this project is about tidiness or about model validity. **No new
training** — computed on the existing standard-RNN reference sweep (N=1000, 5 nets per cell) with
[`population_distortion.py`](../trainRNNbrain/experiments_and_analysis/population_distortion.py).

**The premise, verified first:** all four penalty conditions solve CDDM equally well. R² per net:
`none` 0.840–0.875, `rws` 0.850–0.873, `frm` 0.848–0.872, `both` 0.837–0.853. No condition is
trading task performance for anything below.

![Population statistics](../img/internal_figures/population_distortion.png)

| h equation, N=1000 | none | rws | **frm** | **both** |
|---|---|---|---|---|
| silent units | 42.6% | 40.0% | **0%** | **0%** |
| **effective dimensionality (PR)** | 2.22 ± 0.07 | 2.39 ± 0.06 | **7.74 ± 0.21** | 6.14 ± 0.18 |
| **units selective to choice** | 19.2 ± 1.7% | 23.9 ± 1.5% | **58.3 ± 1.5%** | **82.1 ± 2.3%** |
| units selective to context | 24.3 ± 1.8% | 24.0 ± 1.3% | 31.1 ± 3.4% | **61.7 ± 3.2%** |
| **total metabolic cost** | 31.1 | 31.6 | **14.6** | 16.0 |
| **concentration of that cost (HHI)** | 0.123 | 0.140 | **0.0012** | 0.0011 |

The `s` equation gives the same picture (PR 3.03 → 7.06; choice selectivity 27.9% → 73.2%).

### What this establishes

1. **Effective dimensionality differs 3.5×** — 2.22 vs 7.74, with tight non-overlapping intervals.
   "How many dimensions does the circuit use" is *the* standard quantity for comparing model
   population activity against neural recordings, and it depends almost entirely on whether activity
   was regularised, not on how well the network does the task.
2. **Choice selectivity differs 3–4×** — 19% of units carry choice information without penalties,
   58% with `frm`, 82% with `frm`+`rws`. Any claim of the form "X% of units in the model are
   choice-selective, comparable to Y% in the data" is a statement about the penalty, not the circuit.
3. **The concentrated solution costs MORE energy, not less** — total metabolic cost is 31.1 without
   penalties and 14.6 with `frm`, less than half. This inverts the intuitive efficiency argument: a
   network that concentrates its computation into a few high-rate units pays more, because cost goes
   as the square of the rate. Spreading the same computation thinly is cheaper.
4. **Energy concentration differs 100×.** HHI 0.123 vs 0.0012, i.e. the unpenalized network's
   metabolic cost is carried by an effective **8 units** (1/HHI), the penalized one's by **~850**.

### The sampling distortion, separately

Within unpenalized networks, the same statistic computed over all units versus over active units only
(h equation):

| penalty | context: all units → active only | choice: all → active |
|---|---|---|
| none | 24.3% → **42.3%** | 19.2% → **33.5%** |
| rws | 24.0% → 40.0% | 23.9% → 39.8% |
| frm | 31.1% → 31.1% | 58.3% → 58.3% (no silent units) |

A recording experiment does not see silent neurons — it sees the active ones. So the fraction a
modeller reports from the full model population is ~1.7× lower than what the equivalent experiment
would measure from the same circuit. Under `frm` the two numbers are identical by construction.

### A prediction of mine that was wrong, and why the correction matters

Before running this I argued participation ratio would be **insensitive** to silent units, since
appending all-zero units adds zero eigenvalues and changes neither `(ΣΛ)²` nor `ΣΛ²`. That algebra is
correct but irrelevant: `frm` and `none` are not the same network with zeros removed, they are
**different solutions**. The penalized network genuinely spreads its computation over ~3.5× more
dimensions. The distortion is not an artifact of counting dead units — it is a real difference in the
circuit that the training choice produces.

### Consequence for the paper

`paper.md` §5 moves from planned to measured, and its claim strengthens: it is no longer only "silent
units dilute your statistics" but **"two networks that perform identically on the task differ by 3.5×
in dimensionality, 3× in selectivity and 100× in energy concentration, and nothing in a typical
methods section tells the reader which one you trained."** That is a validity argument about
model-based inference, not an aesthetic preference — and it is the strongest case for the penalties
independent of whether M* exists.

### CORRECTION (2026-08-17 17:00) — the selectivity fractions above were diluted, and one reverses

Pavel's objection: a selectivity **percentage** computed over all units is trivially depressed when
42% of them are silent, since silent units are non-selective by construction. Correct — and the
active-units-only comparison, which the analysis already computed but which I failed to headline,
changes the conclusion.

| h, N=1000, **active units only** | none | rws | frm | both |
|---|---|---|---|---|
| context-selective | **42.3 ± 2.9%** | 40.0 ± 2.6% | **31.1 ± 3.4%** | 61.7 ± 3.2% |
| choice-selective | 33.5 ± 2.7% | 39.8 ± 2.9% | **58.3 ± 1.5%** | 82.1 ± 2.3% |

| s, N=1000, **active units only** | none | rws | frm | both |
|---|---|---|---|---|
| context-selective | **65.3 ± 1.9%** | 51.6 ± 2.0% | **40.4 ± 2.3%** | 63.7 ± 0.8% |
| choice-selective | 46.5 ± 1.5% | 47.1 ± 3.1% | **73.2 ± 1.3%** | 85.6 ± 0.6% |

**What changes:**

- **Context selectivity REVERSES.** Among active units `frm` is *lower*, not higher: 42.3% → 31.1%
  (h, ratio 0.74) and 65.3% → 40.4% (s, ratio 0.62). The apparent increase reported above
  (24.3% → 31.1%) was **entirely a dilution artifact**. That claim is withdrawn.
- **Choice selectivity survives but shrinks by half.** 33.5% → 58.3% (h) is 1.74×, not the 3.0×
  claimed from the all-units figures; 46.5% → 73.2% (s) is 1.57×. Still a large, CI-separated effect.

**What is untouched, and why:**

- **Participation ratio.** `pr_active` equals `pr` to the printed digit in every cell — appending
  zero-variance units adds zero eigenvalues, which change neither `(ΣΛ)²` nor `ΣΛ²`. The **3.5×
  dimensionality difference is not dilution**; it is a real difference between the two solutions.
  (This is the same algebra I raised as a caveat before running the analysis, then wrongly set aside
  when the numbers came in.)
- **Total metabolic cost and its HHI concentration.** Zero-energy units contribute nothing to either
  sum, so neither statistic is diluted. The 2× cost difference and 100× concentration difference
  stand.

**A nuance the fractions hide.** Fractions fall while counts rise. Under `none`, 42.3% of 574 active
units = **243** context-selective units; under `frm`, 31.1% of 1000 = **311**. So `frm` produces ~28%
*more* context-selective units in absolute terms while lowering the fraction — the revived units are
disproportionately choice-tuned rather than context-tuned. That is a change in the **composition** of
the population, not a rescaling of it, and it is a more interesting statement than either fraction
alone.

**Methodological rule for the write-up:** every population fraction gets reported over active units,
with the all-units version shown only to make the dilution point explicitly. Scale-free statistics
(PR, HHI) are exempt, and it is worth saying why in the text rather than leaving the reader to check.

**Figure regenerated** with the corrected metric — the version above showed all-unit selectivity and
is superseded:

![Population statistics, corrected](../img/internal_figures/population_distortion.png)

Eight panels, each axis carrying its formula, and the header defining the symbols. Participation
ratio, context- and choice-selectivity **over active units**, total metabolic cost, cost
concentration, **σ_log with the cortical reference line at 1**, within-trial modulation, and the
p90/median rate tail.

**Exact definitions.** `r_i(t,c)` is the noise-free rate of unit *i* at timestep *t* in condition
*c*, over the full 450-condition CDDM batch; `r̄_i` its mean over time and conditions.

| statistic | definition | notes |
|---|---|---|
| **PR** | `(Σᵢλᵢ)² / Σᵢλᵢ²`, where λ are eigenvalues of the units × units covariance of `r` with time and conditions flattened into one sample axis and each unit centred | 1 = all variance in one dimension, N = variance spread equally. **Invariant to appending silent units** (they add zero eigenvalues, changing neither sum), which is why it needs no active-only correction |
| **η²** selectivity | each unit reduced to its mean rate over the decision epoch (t ≥ 200) per condition, giving one value per condition; conditions grouped by factor (context: motion vs colour; choice: ±1); `η² = σ²_between / σ²_total`; a unit is selective if **η² > 0.10**; reported as a fraction of **active** units | ⚠️ the 0.10 threshold is arbitrary and its sensitivity is untested, and there is **no null distribution** — this means "explains >10% of across-condition variance", not "statistically significant". A permutation null would make it a proper test, and should be added before publication |
| **E** | `Σᵢ ⟨r_i²⟩` over time and conditions | zero-rate units contribute nothing, so no dilution |
| **HHI** | `Σᵢ pᵢ²` with `pᵢ = ⟨r_i²⟩ / E` | 1/N = perfectly even, 1 = one unit carries everything; `1/HHI` is the effective number of units bearing the cost |
| **σ_log** | `std_i[log₁₀ r̄_i]` over active units | the lognormal shape parameter; cortex ≈ 1 |
| **within-trial CV** | `med_i [ ⟨σ_t(r_i)⟩_c / r̄_i ]` over active units | temporal std within a trial, averaged over conditions, normalised by the unit's own mean rate |
| **rate tail** | `r̄_(90) / r̄_(50)` across active units | scale-free, assumption-free heterogeneity check | Reading across, `frm` raises dimensionality 3.5×, *lowers*
context selectivity among active units, raises choice selectivity 1.7×, halves total metabolic cost,
spreads that cost over ~100× more units, keeps units genuinely modulated within the trial — and
flattens the across-unit rate distribution far below the biological range, which is the one result
that goes against it.

### The result that cuts against the penalty: rate heterogeneity, measured as sigma_log

**How it is measured.** Each unit is collapsed to one number — its mean firing rate over all 300
timesteps and all 450 conditions of the noise-free batch. Silent units are excluded, then the spread
of that distribution *across units* is summarised three ways:

| statistic | definition | why |
|---|---|---|
| **σ_log** | std of `log10(mean rate)` across active units | cortical rate distributions are close to **lognormal**, and σ_log is the shape parameter that literature reports — about **1 in log10 units**, i.e. roughly a decade of spread between typical slow and fast cells. Scale-free by construction, and stable on heavy tails. Requires positive rates, hence active units only. |
| CV | `std / mean` of the rate across active units | intuitive but **not** scale-free — `frm` raises the mean rate, so part of a CV drop is a mean shift rather than a narrowing |
| p90 / median | tail ratio across active units | scale-free, robust, no distributional assumption |
| **within-trial CV** | per unit, the temporal std of its rate averaged over conditions and divided by its own mean rate; median over active units | asks whether a unit is genuinely **modulated by the task** or merely sitting at a constant rate — the direct test of whether an activity penalty is satisfied by tonic firing |

| N=1000, active units | σ_log | CV | p90/median | within-trial CV |
|---|---|---|---|---|
| h/none | **1.20 ± 0.05** | 3.62 | 6.22 | **1.43 ± 0.05** |
| h/rws | 1.27 ± 0.08 | 3.64 | 7.84 | 1.36 ± 0.08 |
| **h/frm** | **0.26 ± 0.01** | 0.49 | 1.73 | 1.29 ± 0.04 |
| h/both | 0.19 ± 0.01 | 0.43 | 1.66 | **0.96 ± 0.01** |
| s/none | **1.01 ± 0.08** | 2.41 | 6.46 | 1.31 ± 0.06 |
| s/rws | 1.39 ± 0.07 | 2.77 | 7.55 | 1.14 ± 0.06 |
| **s/frm** | **0.15 ± 0.01** | 0.31 | 1.38 | 1.16 ± 0.02 |
| s/both | 0.16 ± 0.00 | 0.35 | 1.52 | **0.89 ± 0.01** |

**Two findings, pointing in opposite directions.**

1. **Unpenalized networks match cortex on rate heterogeneity; penalized ones do not.** σ_log is
   **1.20 (h) and 1.01 (s)** without penalties — essentially the cortical value of ~1, a full decade
   of spread across the active population. Under `frm` it collapses to **0.26 and 0.15**, a fifth of
   a decade. `rws` does not restore it, and `both` is flatter still (0.19, 0.16). All three
   heterogeneity measures agree, which is what makes this believable rather than an artifact of one
   statistic. **This is a real limitation of the penalty and belongs in the paper as one:** `frm`
   trades one unrealism (half the population silent) for another (a population too uniform).
2. **But the revived units are genuinely modulated, not tonic.** Within-trial CV is 1.29 under `frm`
   against 1.43 without — only ~10% lower. So `frm` units are *not* sitting at the cap doing nothing;
   they vary over the trial almost as much as the units of an unpenalized network. **This closes the
   "the penalty is satisfied by tonic firing" worry** that has been open since the earliest sweeps.
   It is `both` that materially flattens modulation (0.96, a 33% drop) — quantifying, at last, the
   old qualitative claim that `rws` makes responses "more sustained".

**Concrete follow-up** (no new machinery): `frm` has a `cap_fr` target and an already-implemented,
never-used `aggregation: logsumexp` option with temperature `tau_n`. Mean aggregation pulls *every*
unit toward the cap, which is exactly what would flatten the rate distribution; logsumexp penalises
the worst offenders instead. A one-axis sweep over those two knobs could plausibly keep every unit
active while preserving σ_log near 1 — turning a stated limitation into a tuning result.

**Figure regenerated** with the corrected metric — the version above showed all-unit selectivity and
is superseded:

![Population statistics, corrected](../img/internal_figures/population_distortion.png)

Eight panels, each axis carrying its formula, and the header defining the symbols. Participation
ratio, context- and choice-selectivity **over active units**, total metabolic cost, cost
concentration, **σ_log with the cortical reference line at 1**, within-trial modulation, and the
p90/median rate tail.

**Exact definitions.** `r_i(t,c)` is the noise-free rate of unit *i* at timestep *t* in condition
*c*, over the full 450-condition CDDM batch; `r̄_i` its mean over time and conditions.

| statistic | definition | notes |
|---|---|---|
| **PR** | `(Σᵢλᵢ)² / Σᵢλᵢ²`, where λ are eigenvalues of the units × units covariance of `r` with time and conditions flattened into one sample axis and each unit centred | 1 = all variance in one dimension, N = variance spread equally. **Invariant to appending silent units** (they add zero eigenvalues, changing neither sum), which is why it needs no active-only correction |
| **η²** selectivity | each unit reduced to its mean rate over the decision epoch (t ≥ 200) per condition, giving one value per condition; conditions grouped by factor (context: motion vs colour; choice: ±1); `η² = σ²_between / σ²_total`; a unit is selective if **η² > 0.10**; reported as a fraction of **active** units | ⚠️ the 0.10 threshold is arbitrary and its sensitivity is untested, and there is **no null distribution** — this means "explains >10% of across-condition variance", not "statistically significant". A permutation null would make it a proper test, and should be added before publication |
| **E** | `Σᵢ ⟨r_i²⟩` over time and conditions | zero-rate units contribute nothing, so no dilution |
| **HHI** | `Σᵢ pᵢ²` with `pᵢ = ⟨r_i²⟩ / E` | 1/N = perfectly even, 1 = one unit carries everything; `1/HHI` is the effective number of units bearing the cost |
| **σ_log** | `std_i[log₁₀ r̄_i]` over active units | the lognormal shape parameter; cortex ≈ 1 |
| **within-trial CV** | `med_i [ ⟨σ_t(r_i)⟩_c / r̄_i ]` over active units | temporal std within a trial, averaged over conditions, normalised by the unit's own mean rate |
| **rate tail** | `r̄_(90) / r̄_(50)` across active units | scale-free, assumption-free heterogeneity check | Reading across, `frm` raises dimensionality 3.5×, *lowers*
context selectivity among active units, raises choice selectivity 1.7×, halves total metabolic cost,
spreads that cost over ~100× more units, keeps units genuinely modulated within the trial — and
flattens the across-unit rate distribution far below the biological range, which is the one result
that goes against it.

### The result that cuts against the penalty: firing-rate heterogeneity

| N=1000 | active units | rate CV (active) | p90 / median rate |
|---|---|---|---|
| h/none | 574 | **3.62 ± 0.51** | 6.22 |
| h/rws | 600 | 3.64 ± 0.28 | 7.84 |
| **h/frm** | 1000 | **0.49 ± 0.01** | **1.73** |
| h/both | 1000 | 0.43 ± 0.02 | 1.66 |
| s/none | 600 | 2.41 ± 0.17 | 6.46 |
| **s/frm** | 1000 | **0.31 ± 0.02** | **1.38** |

Unpenalized networks have **strongly heterogeneous** rates among their active units — CV ≈ 2.4–3.6,
with the 90th percentile 6–8× the median, i.e. a long right tail. `frm` networks are nearly
**uniform**: CV ≈ 0.3–0.5, p90 only 1.4–1.7× the median. `rws` does not restore the spread; `both`
is, if anything, flatter still.

**This is a genuine limitation of the penalty, and it should be reported as one.** Cortical firing
rates are strongly heterogeneous — approximately lognormal with a long tail — so a model population
in which every active unit fires at nearly the same rate is *less* data-like in this respect, not
more. `frm` trades one unrealism (half the population silent) for another (a population that is too
uniform). The honest statement for the paper is that it fixes the silent-unit problem and introduces
a homogeneity problem, and that both matter for the same reason: they change what the model
population looks like relative to a recorded one.

**Concrete follow-up this suggests** (no new machinery needed): `frm` has a `cap_fr` target and an
`aggregation` option (`mean` vs `logsumexp` with temperature `tau_n`, already implemented and
unused). A weaker cap, or logsumexp aggregation that penalises only the worst offenders rather than
pulling every unit toward the target, might keep every unit active *without* collapsing the rate
distribution. That is a one-axis sweep and would turn a limitation into a tuning result.

---

