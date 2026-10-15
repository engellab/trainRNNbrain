## 2026-08-21 16:52 — the active COUNT does obey a clean scaling law, unlike the silent fraction

New script: `experiments_and_analysis/flipflop_M_scaling.py` → `img/internal_figures/flipflop_M_scaling.png`.

The silent-FRACTION collapse failed with a divergent exponent. Asking the same question of the
absolute count M is a genuinely different target, since M = N(1-f) carries an explicit N, and it
behaves completely differently: **the exponent is identified, stable across readings, and far from
any search boundary** (grid -6 to +6).

**M = A N^b k^c, scale-free criterion:**

| reading | b (size) | c (complexity) | var. expl. | saturated | implied u = N k^(c/b) |
|---|---|---|---|---|---|
| endpoint | 0.727 [0.65, 0.82] | 0.594 [0.46, 0.72] | 0.882 | 0.919 | N k^0.82 |
| R² = 0.949 | 1.075 [0.96, 1.18] | 1.120 [0.98, 1.31] | 0.931 | 0.963 | N k^1.04 |
| R² = 0.970 | 0.989 [0.89, 1.10] | 0.803 [0.59, 0.95] | 0.904 | 0.956 | N k^0.81 |
| R² = 0.980 | 0.901 [0.83, 0.98] | 0.754 [0.65, 0.86] | 0.955 | 0.980 | N k^0.84 |

The independent non-parametric search for the collapse exponent (ansatz B, no power-law assumption)
returns -0.80, -1.05, -0.80, -0.82 — matching the value implied by c/b to two decimals without being
told it. That agreement is the evidence the collapse is real rather than fitted.

**So at matched performance, on the flip-flop, active units obey roughly**

        M  ~  N * k^0.85          (b ~ 1, c ~ 0.8)

**This reconciles exactly with the failed fraction collapse.** b ~ 1 means 1 - f is N-INDEPENDENT,
i.e. the silent fraction depends only on k — which is precisely what the variance decomposition said
(N explains 1-14%, k explains 65-86%). The two analyses agree; they were asking different questions.

Honest caveats:
- The F-test against the saturated model is marginal at the deeper readings (p = 0.027-0.072), so
  the law captures most but not all of the cell-to-cell structure (0.88-0.96 vs 0.92-0.98).
- b drifts from 0.73 at the endpoint to ~1.0 at matched levels. The endpoint value is the
  training-depth-confounded one and should not be quoted.

**Hard criterion, for contrast:** b = 1.00, c = 0.10 at every matched reading — M is essentially N
with no k-dependence at all, because the hard count is pinned near M = N. It is degenerate as a
measure here and only the scale-free count carries structure. Another entry in the running case for
never reporting a single silence threshold.

**Consequence.** This is the quotable form of the result: recruitment is linear in size and
sublinear-but-strong in task complexity, and the wasted FRACTION is set by task complexity alone.
It is a better statement than either "M saturates" (CDDM-specific, b = 0.31 there) or "silencing
depends on N and k".

---

