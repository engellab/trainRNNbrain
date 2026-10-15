## 2026-08-24 09:42 — k=1 lands: the floor law is pc(k) = a + b·sqrt(k), C(k,2) rejected out of sample

**The pre-registered test.** Both surviving models were fitted on k=2..6 and neither saw k=1. Measured
per-channel floor at k=1, N=500: **0.02369 ± 0.00004** (seeds 0.02375 / 0.02368 / 0.02365).

| model | predicted | error |
|---|---|---|
| `C(k,2)` | 0.02440 | **+2.99%** — 18 seed-SDs |
| `sqrt(k)` | 0.02373 | **+0.16%** — inside seed scatter |

k=7 and k=8 also landed, so the fit now spans k=1..8 (23 seed points, N=500):

| model | ΔAICc | max resid |
|---|---|---|
| `k·a + b·k^1.5`  (pc = a + b√k) | **0.0** | 0.33% |
| `k·a + b·k^(1+q)`, q free | +1.8 | 0.24% |
| `k·f1 + g·C(k,2)^p`, p free | +8.4 | 0.75% |
| `k·f1 + C(k,2)·g`  (p = 1) | **+70.7** | 4.28% |

Per-channel residuals across the whole range: +4, +2, −0, +0, −1, −1, +1, −0 (units of 1e-5), against
a seed SD of 3e-5. The law is exact to the noise floor.

**Free exponent: per-channel ~ k^0.481 ± 0.022.** √k predicts exactly 0.5, which is 0.86σ away — not
rejected. The free exponent on C(k,2) is 0.826 ± 0.008, i.e. 22σ from 1.

**So: `floor_per_channel(k) = a + b·√k` with a = 0.0208, b = 0.0029.** Equivalently the TOTAL floor is
`k·a + b·k^1.5`.

**What √k means.** If interference from the (k−1) other channels were independent and additive in
variance, per-channel excess would grow as k and the total as C(k,2). It grows as √k instead, so the
**effective number of independent interfering partners scales as √k, not k.** That is consistent with
the directly measured effective cross-talk |G_off| ~ k^−0.20 at N=500: (k−1)·⟨G²⟩ with G ~ k^−0.25
gives k^0.5 exactly, and −0.25 vs the measured −0.20 is inside that measurement's noise.

### Separately: the N=10000 CDDM point, all 3 seeds

⚠️ It never reaches L\*=0.023 — smoothed loss bottoms at 0.02334, because the budget was cut 100k →
80k for Della's 6-day QOS while the extrapolated read-point was ~110k. Read at shallower levels:

| L\* | criterion | observed | power law | saturating |
|---|---|---|---|---|
| 0.0240 | hard | 1463 ± 61 | 1851 (+26.5%) | **1511 (+3.3%)** |
| 0.0240 | scale-free | 1221 ± 51 | **1249 (+2.3%)** | 1019 (−16.6%) |
| 0.0235 | hard | 1041 ± 19 | 1317 (+26.5%) | **1094 (+5.1%)** |
| 0.0235 | scale-free | 958 ± 20 | 1092 (+14.1%) | **894 (−6.7%)** |

**The two criteria give opposite verdicts, consistently at both levels.** Hard favours saturation
(+3%, +5% vs the power law's +26%); scale-free favours the power law at the shallower level. Three
seeds with ±4% spread do not close that gap — it is a real criterion dependence, not sampling noise.

**Consequence for `paper.md`:** the saturation claim must be stated as criterion-dependent. The one
out-of-sample point that tests it does not adjudicate, and reading it at the intended depth is not
possible with the budget that was run.

---

