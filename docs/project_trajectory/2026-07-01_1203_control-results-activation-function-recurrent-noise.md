## 2026-07-01 — control results: activation function & recurrent noise

The activation and noise controls above were re-run at commit `fb2792` (the `f8be3e` submissions were
superseded) and analysed here. All sweeps: N=1000, γ=0, no bias, scored on the **noise-free** CDDM batch
(so we measure the silence baked into the trained weights, not instantaneous noise silencing). Baseline for
comparison is `CDDM_4a031e_g0` (plain ReLU): N=1000 silent (peak<0.01) — h/none ≈47%, h/rws ≈54%, s/none ≈54%,
s/rws ≈56%; fr-only and both = 0.

**Metrics defined.** Every net is reconstructed from its saved (Dale-compliant) weights and run on the
noise-free CDDM batch, giving a firing-rate tensor `fr` of shape `(N units, T time, C conditions)`. From it:

- **peak rate** `peak_i = max over (t, c) of |fr[i]|` — unit `i`'s single most-active moment anywhere in the task.
- **dead<0.01** — fraction of units with `peak_i < 0.01`. An **absolute** floor (the rate cap/target is ~0.3–0.5),
  so "essentially never fires." Fine for ReLU (silent units hard-zero) but *unfair across activations*: softplus
  has a smooth positive floor so no unit is ever exactly 0, and this cut then reads 0% even when half the units are
  ~30× quieter than the active ones.
- **silent<5%p95** — fraction of units with `peak_i < 0.05 · p95`, where `p95` is the net's 95th-percentile peak.
  A **within-network, scale-free** cut: "is this unit's best moment still >20× quieter than the net's active
  population?" (95th percentile, not max, so one outlier unit doesn't set the scale). This is the number to compare
  across activations, because it is invariant to each net's overall activity scale.
- **participation** `p_i = std(fr[i]) + 0.9-quantile(|fr[i]|)` over (t, c) — a graded activity measure (used for the
  histograms, HHI, and least-unit pick) rather than a hard silent/active threshold.
- **HHI / 1/HHI** — Herfindahl–Hirschman index of participation: with shares `s_i = p_i / Σ p_j`, `H = Σ s_i²`.
  `H ∈ [1/N, 1]`; `H = 1/N` is perfectly even participation, larger `H` means a few units dominate. **`1/H` is the
  effective number of participating units** — the intuitive x-axis of the scatter (≈ N=1000 means "all units share
  the work", ≈ 60 means "~60 units do everything, the rest are near-silent").

Counts by
[`count_silent_units.py`](../trainRNNbrain/experiments_and_analysis/count_silent_units.py) (softplus, leaky) and
[`count_silent_units_noise.py`](../trainRNNbrain/experiments_and_analysis/count_silent_units_noise.py) (noise, adapted to the `sigrec=` dir naming).

### Activation function — `CDDM_fb2792_g0_softplus25` (38/40 nets; job 5100397), `CDDM_fb2792_g0_leakyrelu` (40/40; 5100398)

Both softplus(β=25) and leaky-ReLU (leak 0.01) have **nonzero gradient everywhere** — no dead-gradient trap.
Silent fraction at N=1000 (mean over 5 nets/condition, except softplus h/none and s/both = 4 nets — 2 jobs
short, softplus trains slower):

| eq / penalty | criterion | ReLU baseline | softplus25 | leaky-ReLU |
|----|----|----|----|----|
| h / none    | silent<5%p95 | ~47% | 40.6% | 45.2% |
| h / rws     | silent<5%p95 | ~54% | 63.6% | 56.2% |
| s / none    | silent<5%p95 | ~54% | 54.4% | 55.1% |
| s / rws     | silent<5%p95 | ~56% | 61.1% | 61.1% |
| h / none    | dead<0.01    | ~47% | **0.0%** | 32.5% |
| h / rws     | dead<0.01    | ~54% | **0.0%** | 49.5% |
| s / none    | dead<0.01    | ~54% | 52.4% | 54.0% |
| s / rws     | dead<0.01    | ~56% | 53.3% | 54.0% |
| any / fr-only, both | both | 0 | **0** | **0** |

![Participation — softplus25 h](../img/internal_figures/participation_histograms_h_fb2792_g0_softplus25.png)
![Participation — leaky h](../img/internal_figures/participation_histograms_h_fb2792_g0_leakyrelu.png)

**Finding (the key framing test): the large low-activity population is NOT a hard-ReLU / dead-gradient
artifact — it is a general property of trained CDDM RNNs.**

1. On the scale-free criterion, **all three activations concentrate ~40–64% of units into a low-activity
   mode** under `none`/`rws`, statistically indistinguishable from the ReLU baseline. Everywhere-positive
   gradient does **not** keep units active.
2. **Softplus-h is the sharpest demonstration:** `dead<0.01` = **exactly 0%** (softplus's smooth floor keeps
   every unit's peak nominally above 0.01), yet `silent<5%p95` is still 41–64%. The population doesn't
   disappear — it **reorganises from exact zeros into a soft low-activity continuum** at the same ~half-of-N
   mass. The participation histogram shows it directly: red (`none`)/orange (`rws`) still pile near zero, green
   (`fr`)/blue (`both`) collapse to one tight active bump. Softplus-s and both leaky equations keep large
   *hard*-near-zero populations (49–57%), like ReLU.
3. **The fr-magnitude penalty collapses everyone into one active mode in every activation** (fr-only, both = 0
   throughout) — the rescue is activation-independent.

**Performance vs participation spread (R² vs 1/HHI).** Each point is one N=1000 net: x = effective number of
participating units (`1/HHI`, log scale; dashed line = even N=1000), y = validation R² (the score in the net's
folder name). Plotted by
[`plot_r2_vs_hhi.py`](../trainRNNbrain/experiments_and_analysis/plot_r2_vs_hhi.py).

![R² vs 1/HHI — softplus25](../img/internal_figures/r2_vs_hhi_N1000_fb2792_g0_softplus25.png)
![R² vs 1/HHI — leaky](../img/internal_figures/r2_vs_hhi_N1000_fb2792_g0_leakyrelu.png)

The two clouds are cleanly separated on the x-axis and overlapping on the y-axis: `none`/`rws` nets solve the task
using an **effective ~60–150 units** (softplus-h: ~60–90), while `fr`/`both` nets spread the identical task across
**~700–900** — near the even line — at **equal or better R²** (~0.82–0.87 throughout). So concentrating computation
onto a small subset is **not required** for performance; it is what the network does when nothing pushes back.
Two asides visible here: for **softplus-h, `rws`-only actively hurts R²** (several nets at 0.4–0.65, the low orange
points) while barely changing the concentration — sparsifying recurrent weights degrades the solution without
redistributing activity; and the `s` equation concentrates less severely than `h` (clouds sit further right).

**The least-participating unit.** For each net we take the single lowest-participation unit and draw its firing
rate as a heatmap (x = time, y = task condition; `p` = its participation). 5 nets (rows) × 4 penalties (cols),
by [`plot_least_unit_activity.py`](../trainRNNbrain/experiments_and_analysis/plot_least_unit_activity.py).

![Least-unit activity — softplus25 h](../img/internal_figures/least_unit_activity_h_N1000_fb2792_g0_softplus25.png)

Under `none`/`rws` the **worst unit is blank** (softplus-h `p≈0.003` — nominally nonzero because of softplus's
floor, but flat and task-unrelated). Under `fr`/`both` **even the worst unit carries a clear, time- and
condition-locked activity blob** (`p≈0.05–0.12`). This is the same collapse seen in the histograms, at the
single-unit extreme: the penalty doesn't just shift the bulk — it pulls up the very tail.

### Recurrent noise — `CDDM_fb2792_g0_noise` (40/40; job 5100399)

ReLU, `none` penalty, sweeping the recurrent training noise σ_rec ∈ {0, 0.01, 0.05, 0.1}. Silent fraction at
N=1000 (mean over 5 nets/σ):

| σ_rec | h dead<0.01 | h silent<5%p95 | s dead<0.01 | s silent<5%p95 |
|----|----|----|----|----|
| 0.00 | **79.4%** | **81.6%** | 58.2% | 60.1% |
| 0.01 | 46.0% | 47.9% | 55.5% | 57.9% |
| 0.05 | 43.5% | 47.6% | 54.1% | 55.5% |
| 0.10 | 40.7% | 46.0% | 56.4% | 57.0% |

![Silent fraction vs recurrent noise](../img/internal_figures/silent_vs_noise_fb2792_g0_noise.png)

**Finding: the silence is NOT noise-driven.** Adding recurrent noise never *increases* it.

1. For **h**, σ_rec=0 (noise fully off) is a distinct pathological regime — **~80% silent**. Any nonzero noise
   (≥0.01) drops it to ~46–48%, then it is flat-to-slightly-*decreasing* with more noise: recurrent noise mildly
   **regularises against** silence rather than causing it.
2. For **s**, the silent fraction is ~54–60% and essentially **flat across all σ_rec** — noise-independent.
3. The ReLU baseline (h/none ≈47%, s/none ≈54%) matches the **σ_rec≈0.05** column (h 43.5%, s 54.1%), i.e. the
   baseline sweep trained with a nonzero default σ_rec; the σ=0 point is what removing noise entirely exposes.
4. Net: the ~45–60% low-activity population **persists across every realistic noise level** — an intrinsic
   feature of the trained solution, not a recurrent-noise artifact.

### Bottom line

Two candidate "trivial mechanism" explanations are now excluded. The low-activity population is neither a
hard-ReLU dead-gradient trap (survives softplus/leaky, everywhere-differentiable) nor noise-induced silencing
(survives noise-free, mildly reduced by noise). Combined with the earlier controls (not a code bug, emerges in
training not init, a low-activity *mode* not literal dead units), the phenomenon is best framed as
**trained CDDM RNNs concentrate computation onto a subset of units**, and the fr-magnitude penalty is what
redistributes it across the full population.

