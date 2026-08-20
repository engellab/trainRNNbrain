# Does `rws` close the transient loophole in `frm`? — pre-registered analysis plan

Written 2026-08-20 13:03, **before any of these metrics were computed**. Directions and thresholds
below are fixed in advance so that a pass means something; anything decided after seeing numbers is
marked as such when the results are written up.

No new training. Everything runs on networks already on disk.

---

## 1. The hypothesis, and why it is not yet supported

`frm` computes its penalty on a **soft maximum over time and conditions**, not on a mean:

```python
activity = tau * (logsumexp(x / tau, dim=1) - log(T*B))    # tau = 0.1, dim=1 spans time x conditions
over  = relu(activity - cap);  under = relu(cap - activity)
penalty = (alpha * (under/cap)**g_bot + beta * (over/cap)**g_top).mean()
```

At τ = 0.1 this is effectively the unit's **peak**. A unit therefore satisfies the penalty by
producing one large transient in one condition and doing nothing else. Nothing in `frm` asks a unit
to contribute *sustainedly* or *across conditions*.

**Hypothesis (H).** `frm` alone drives units above the silence threshold via large transients.
`rws` suppresses those transients and converts the units into sustained, broadly-tuned contributors.
The two penalties are therefore complementary by design: `frm` supplies the upward pressure, `rws`
makes what it produces useful. The joint condition `both` is the method; neither term alone is.

**Why the existing evidence does not establish H.** The only measurement bearing on it is the
within-trial temporal CV (1.29 under `frm` vs 1.43 under `none`, 0.96 under `both`), which was
recorded in `paper.md` §1.3 as showing the revived units are "genuinely modulated, not tonic". That
inference is invalid in the direction that matters here: CV separates *tonic* from *modulated*, but
a single large transient is the **highest**-CV signal a unit can produce. The observation is equally
consistent with H and was over-read. The discriminating statistic is peak-to-mean, not CV.

**The falsifier, stated first.** If `frm` and `both` are indistinguishable on the group-A metrics
below — specifically if the median peak-to-mean ratio over active units differs by less than 1.3x
and the seed-level means overlap — then H is **wrong**, `rws` contributes something other than
transient suppression, and the paper must not claim this mechanism.

---

## 2. Data

| condition | source | note |
|---|---|---|
| `none` | `data/trained_RNNs/CDDM_std_g0_drift` | 200k (N=500/1000), 300k (N=2000) |
| `rws`, `frm`, `both` | `data/trained_RNNs/CDDM_std_g0_penalties` | 200k (N=500/1000), 150k (N=2000) |

N ∈ {500, 1000, 2000}, 3 seeds per cell. Rates come from the saved `*LastParams*.npz` run
**noise-free** through `RNN_numpy` on one shared CDDM batch (450 conditions), giving `r[i, t, c]`
over the masked timepoints — the same path `penalty_matched.py` already uses.

**Two controls are mandatory on every figure**, because both have already reversed a conclusion in
this project:

- **Matched budget.** `none` ran up to 2x the penalty runs' budget; read it from the participation
  trace at the penalty run's stopping iteration, not at its own endpoint.
- **Noise-free task loss, never `TrainLosses.json`.** The recorded column is task + λ·penalty
  evaluated with noise on, and ordering conditions by it inverts the true ordering.

---

## 3. Metrics

Every metric is **per unit**, computed over active units only (both silence criteria reported), and
summarised as a **distribution** — ECDF or histogram — not only as a mean. Aggregate agreement
between two conditions can hide two populations that differ unit by unit.

### Group A — the transient loophole (this is what tests H)

| # | metric | definition | reads as |
|---|---|---|---|
| A1 | **peak-to-mean ratio** | `PMR_i = softmax_τ(r_i) / mean_{t,c}(r_i)`, τ = 0.1 to match the penalty exactly | high = transient, →1 = sustained |
| A2 | **robust peak-to-mean** | `q99_{t,c}(r_i) / mean_{t,c}(r_i)` | A1 without the softmax's sensitivity to a single sample |
| A3 | **temporal duty cycle** | fraction of `(t,c)` with `r_i > 0.5 · softmax_τ(r_i)` | low = the unit is near its peak almost never |
| A4 | **condition breadth** | fraction of conditions `c` with `max_t r_i(t,c) > 0.5 · peak_i` | low = the unit fires in a handful of conditions only |
| A5 | **temporal participation ratio** | `(Σ r_i)² / (T·B·Σ r_i²)` | effective fraction of samples the unit is active; 1 = flat, →0 = single spike |
| A6 | **loophole margin** | evaluate the penalty's own statistic `s_τ(r_i) = τ·(logsumexp(r_i/τ) − log(T·B))` at τ ∈ {0.1, 1, 10, 100}; count units with `s_0.1 ≥ cap` **and** `s_10 < cap` | the direct count of units satisfying the penalty by transient |

A1–A3 and A5 measure concentration **in time**; A4 measures concentration **across conditions**, a
genuinely separate way to be useless, and one that A1–A3 cannot see.

**A6 is the sharpest of these and should be the primary statistic**, because it is not a proxy — it
evaluates the trained network against *the penalty's own objective*, at the τ that admits transients
and at a τ that does not, and counts the units that pass only under the former. τ interpolates the
statistic continuously from max (τ→0) to mean (τ→∞):

    τ·(logsumexp(r/τ) − log n) → max(r)   as τ → 0
                               → mean(r)  as τ → ∞

Measured on two synthetic units with **identical mean rate** — one a transient at 1% duty cycle, one
flat — the configured τ = 0.1 scores the transient **77× higher**; by τ = 10 the ratio is 1.1, and by
τ = 100 the two are indistinguishable. So the loophole is a property of τ alone, and its size is
known before any network is examined. A6 asks only whether trained networks actually walk through it.

**If A6 returns a large count, the natural fix is a τ sweep, and it needs no code change** — τ is
already `trainer.frm_args.tau`. Two cautions for that sweep, if it happens:

- **τ and `cap` are confounded.** At fixed `cap`, raising τ makes the penalty strictly harsher (peak
  ≈ cap becomes mean ≈ cap, roughly a 1/duty-cycle increase in demanded activity), and the harsh end
  is where the documented divergence mode lives — a self-exciting loop with gain > 1 whose Euler
  integration overflows while gradient norms stay ~1. Rescale `cap` per τ, or sweep both.
- **The mean has the opposite loophole.** `mean(r) = cap` is satisfied exactly by a unit sitting
  tonically at the cap with zero modulation. Peak-based admits transients; mean-based admits tonic
  units; neither alone demands *modulated and sustained*. So a τ change may not make `rws`
  redundant — it may only change what `rws` is needed for. Any τ sweep must therefore report A3
  (duty cycle) **and** a modulation statistic, or it will simply trade one failure mode for the
  other and score it as a success.

### Group B — heterogeneity across units

Existing numbers cover `none` and `frm` only; `rws` and `both` are the gap this fills.

| # | metric | definition | current status |
|---|---|---|---|
| B1 | **σ_log** | std of `log10 mean_{t,c}(r_i)` over active units | `none` 1.20 → `frm` 0.26; `rws`/`both` **unmeasured** |
| B2 | **CV of unit mean rates** | `std_i / mean_i` of per-unit mean rate | `none` 3.62 → `frm` 0.49 |
| B3 | **tail ratio** | `p90 / median` of per-unit mean rate | `none` 6.2 → `frm` 1.7 |
| B4 | **energy concentration** | HHI over per-unit metabolic cost, and `1/HHI` as effective unit count | `none` 0.123 → `frm` 0.0012 |

**B1 must be recomputed over group-A-passing units, not merely over non-silent units.** If H holds,
`frm`'s "active" population is contaminated by transient units whose mean rate is near zero, which
would *inflate* its σ_log spread for a reason that has nothing to do with biological heterogeneity.
Report σ_log both ways and say which is which.

### Group C — is the unit doing useful work

| # | metric | definition |
|---|---|---|
| C1 | **readout contribution** | `‖W_out[:, i]‖ · std_{t,c}(r_i)` — the unit's share of output variance under a linear readout |
| C2 | **condition modulation** | per unit, variance across conditions ÷ total variance; the fraction of units with meaningful task modulation |
| C3 | **population dimensionality** | participation ratio of the `r` covariance | `none` 2.22 → `frm` 7.74; `rws`/`both` unmeasured |

### Group D — the headline number

| # | metric | definition |
|---|---|---|
| D1 | **useful units `M_useful`** | count of units that are non-silent **and** pass a sustained-activity bar: `A3 ≥ 0.05` and `A4 ≥ 0.05` |

D1 is the number the paper should report in place of "active units". It reframes the claim from
*`frm` gives you 1000 non-silent units* to *`frm` gives you 1000 non-silent units of which only
`M_useful` do sustained work, and `both` gives you 1000 that nearly all do*. It also plugs directly
into the existing `M(N)` saturation analysis: rerun that fit on `M_useful` and see whether the
ceiling moves.

The 0.05 bars are set now, arbitrarily but in advance. **Report `M_useful` as a curve over the bar
value**, so the conclusion cannot rest on the specific choice.

---

## 4. Pre-registered predictions

Under H, at every N:

| metric | prediction | threshold for "supported" |
|---|---|---|
| A1, A2 | `frm` > `both` | median ratio ≥ 1.3x, seed-level means non-overlapping |
| A3, A4, A5 | `frm` < `both` | same, in the opposite direction |
| A1–A5 | `rws` ≈ `none` | `rws` alone should not create transients, only remove them |
| D1 | `M_useful`: `both` > `frm` | and the gap **widens with N** |
| B1–B4 | `both` ≤ `frm` (more homogeneous still) | directional only — this is a stated cost, not a success criterion |
| C1, C3 | `both` ≥ `frm` | directional |

**Effect-size reporting.** For each group-A metric also report the fraction of `frm` units above
`both`'s 95th percentile. A mechanism claim needs a separated distribution, not a shifted mean.

**Outcomes and what each means:**

- **H supported** → the paper's method is `frm` + `rws`, with a measured mechanism for why both terms
  are needed, and the loophole is disclosed by us rather than found by a referee.
- **H falsified** (`frm` ≈ `both` on group A) → `rws`'s contribution is something else; find it before
  claiming a mechanism, and drop the transient language entirely.
- **Split outcome** (some A metrics separate, others don't) → most likely if `rws` suppresses temporal
  transients but not condition-narrowness, or vice versa. Report which axis it acts on; do not
  average the metrics into a single score to manufacture a clean answer.

---

## 5. Why this matters for the paper's framing

Under the identifiability framing (`paper.md` §7), the homogeneity `rws` adds is **not a cost**. Units
that all carry comparable, sustained, broadly-tuned signal are exactly what makes a random subsample
of K units representative of the population, and therefore what makes the network recoverable from
few sampled units. B1's reduced σ_log is the mechanism of that benefit, not a defect to apologise
for — and that argument only holds if group A shows the units are genuinely sustained rather than
uniformly transient.

This analysis is therefore a prerequisite for the identifiability section, not an aside.

---

## 6. Outputs

- `experiments_and_analysis/transient_loophole.py` → `img/internal_figures/transient_loophole.png`
  - (a) ECDF of A1 per condition, one panel per N
  - (b) A3 vs A4 scatter, per unit, coloured by condition — the two axes of uselessness at once
  - (c) `M_useful` vs N per condition, with the bar-value sweep as an inset
  - (d) B1–B4 table, computed both over non-silent units and over group-A-passing units
- Result appended to `project_trajectory.md` with the pre-registration date cited, and the
  `paper.md` §1.3 CV claim corrected regardless of outcome.
