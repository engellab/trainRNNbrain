## 2026-10-16 16:14 — Two claims measured after the silent / inactive split

Pavel: "go ahead and measure both". Pass/fail rules were set before running.

**1. Excluding inactive units barely changes the participation ratio.** Rule: "unchanged" stands
only if the ratio over active units is within 1% of the ratio over all units in every network.
Computed on the Mac for the 12 networks behind Figure 5 (CDDM, N = 1000, 200k; arms none, rws, frm,
both, 3 each) with `fig_paper_F5.cddm_rates` and `participation_ratio`:

| arm | active | inactive (silent) | PR all → active only | inactive units' share of variance |
|---|---|---|---|---|
| none | 259-282 | 718-741 (697-723) | 2.09-2.27, changes −0.001 to −0.002% | ≤ 0.0011% |
| rws | 178-186 | 814-822 (485-751) | 2.13-2.24, changes −0.002 to −0.006% | ≤ 0.0030% |
| frm, both | 1000 | 0 | unchanged | 0 |

The claim passes. The paper's "provably unchanged" argued only from silent units; it now states the
measurement (main.tex Fig. 5b caption and `06_whyitmatters.tex`; paper bc85c59). Figure 5's cache
from 20 Sep gives the same values.

**2. Tanh networks have no inactive unit.** Rule: the claim stands only if the count is 0 in all 9
networks. From each `CDDM_std_g0_tanh` run's participation trace at iteration 149,900 (Della job
PRtanh):
- every one of the 9 tanh networks has **0 inactive units**;
- the least-participating unit sits at 0.09 to 0.89 of the network's 95th percentile, against the
  0.05 criterion;
- the 9 ReLU networks of the same sweep have 702 to 719 inactive units.

The claim passes. "Tanh units are neither silent nor inactive" stays in the paper.
