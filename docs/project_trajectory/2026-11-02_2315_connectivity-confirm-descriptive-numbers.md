## 2026-11-02 23:15 — Confirmatory connectivity test: descriptive numbers by read point and by weight decay (post hoc, for the paper text)

Computed after the verdict of `2026-11-02_2232_connectivity-confirm-result.md`, for the revision of
dead-relu-paper PR #178 (review comment 6597677662). Nothing here changes the verdict. Script:
`trainRNNbrain/experiments_and_analysis/connectivity_confirm_posthoc.py` (commit below), on
`data/connectivity_confirm.json` and `data/connectivity_confirm/`. The reviewer's independent code
gave the same values.

**1. The difference depends on the read point.** −rel of W1 (how much lower the s.d. of the recurrent
weights is over all units than over the active units), by task and by the iteration of the weights
used; "gap" is that iteration minus the floor crossing t*:

| task | iteration | networks | −rel | gap |
|---|---|---|---|---|
| 3BFF | 40,000 | 2 | 57-61% | −17,341 to −16,159 |
| 3BFF | 75,000 | 4 | 67-71% | +12,182 to +16,032 |
| 3BFF | 200,000 | 1 | 79% | +132,129 |
| CDDM | 30,000 | 3 | 52-53% | +1,861 to +7,181 |
| CDDM | 50,000 | 5 | 58-62% | +19,005 to +27,707 |
| CDDM | 75,000 | 7 | 61-65% | +48,060 to +53,592 |
| CDDM | 100,000 | 3 | 66-67% | +72,741 to +76,071 |
| CDDM | 200,000 | 8 | 70-74% | +169,266 to +171,751 |
| DMTS | 60,000 / 63,000 | 2 | 64%, 69% | +982, +945 (kept snapshots at the crossing) |
| DMTS | 150,000 | 4 | 79-83% | +74,279 to +131,401 for two; two have no crossing |

At one read point, networks of one task agree within about 4 percentage points. The 52-83% range of
the result entry is mostly a range of read points. The difference grows with training in CDDM, from
52-53% at 30,000 iterations to 70-74% at 200,000. At the paper's same-size read point, 75,000
iterations (11 networks), it is 61-65% in CDDM and 67-71% in 3BFF.

**2. Without weight decay the difference remains.** In the six CDDM networks trained with weight
decay 0 (#40-45), the s.d. over all units is 0.13-0.16 and the s.d. over the active units is
0.24-0.31, so the value over all units is 43-51% lower (0.435-0.506). The s.d. over all units there is
3.5-4.2 times its initial value. Weight decay therefore explains why the all-unit value in CDDM falls
below its initial value, not why it is lower than the active-unit value.

**3. Every CDDM network with weight decay, not only #11-13.** Over the 26 CDDM networks of the main
test (weight decay 1e-6, read at 30,000-200,000 iterations):
- the slope of the inactive–inactive weights on their initial values is at most 0.008, and at most
  0.0003 from 75,000 iterations on;
- the s.d. of those weights is 3-15% of its initial value (3-4% in #11-13 at 200,000);
- the s.d. over all units is 0.42-0.57 times its initial value.

**Set sizes, named.** Of the 39 networks, 37 have a floor crossing and 2 (DMTS 2946835052 and
3546706548) do not. Of the 37 with a crossing, 2 (DMTS 2548178424 and 2239198786) kept a snapshot
within 1,000 iterations after it, and 35 are read at other saved weights. Methods and the
Supplementary Methods name these sets in place of "37".
