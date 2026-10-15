## 2026-07-27 (21:45) — strategy: how the saturation question gets decided (array `11686960` submitted)

Written down explicitly because the answer determines whether the paper has a reply to *"why bother —
just train a bigger network and prune the silent units"*, and because the decision rules must be
fixed before the data lands.

### The measurement

**Active units = N × (1 − silent fraction)**, with the silent fraction read from the **final
participation snapshot** of each network's trace, on **both** metrics:

- **truly silent**, `p < 1e-6` — unambiguous (a ReLU unit that never fires has `p = 0` exactly);
- **scale-free**, `p < 5%` of that network's 95th-percentile participation — the metric that is
  comparable *across network sizes*, which matters here because the activity scale differs with N.

Both curves are reported. If they disagree about saturation, that disagreement is itself the result
and gets reported rather than resolved by picking the more convenient one.

### The two hypotheses, fitted to N ≤ 1000 (standard RNNs, h, no penalty: 97, 221, 340, 448)

| Model | Fit | N=2000 | N=5000 | N=10000 |
|---|---|---|---|---|
| **A — power law** | `active = 4.55·N^0.665` | 710 | 1305 | **2069** |
| **B — saturating** | `active = 749·N/(N+672)` | 561 | 660 | **702** |

They differ by 3× at N=10000, so the experiment discriminates cleanly rather than requiring a
statistical argument.

### Decision rules, fixed in advance

- **< ~1200 active units at N=10000 → saturation (B).** Pruning cannot deliver a large active
  population at *any* size; the ceiling is set by the task, not the budget. `frm` at N=1000 already
  yields 1000 active units, which no amount of scaling could reach.
- **> ~2000 active units at N=10000 → growth (A).** Pruning works; the paper's argument must then
  rest entirely on the population-level differences (`paper.md` §5) rather than on reachability.
- **Between 1200 and 2000 → ambiguous.** Report as such and extend the curve rather than argue.

Secondary evidence, decided the same way: **the local exponent**. It has fallen monotonically so far
(0.899 → 0.621 → 0.398 across N = 100→250→500→1000). Continued decline toward 0 supports B; a
stabilisation near ~0.6–0.7 supports A.

### Two controls without which the answer is not interpretable

1. **Truncation control.** N=5000 and N=10000 train for 5000 iterations rather than 30000, because
   48 h exceeds the queue limit. The justification is that the silent fraction is decided by
   iteration ~400–600 and frozen — but that is a claim, so cells 0–2 of the array measure it: N=1000
   and N=2000 at 5000 iterations, plus N=2000 at 30000. **If 5000 iterations does not reproduce the
   30000-iteration silent fraction at both sizes, the large-N numbers become an upper bound and are
   reported as such.**
2. **Task-performance control.** A large network that simply failed to learn would trivially show
   many silent units, and the active-unit curve would be measuring training failure rather than
   capacity. So R² must be checked at every N before the curve is interpreted: if it degrades
   materially at N=5000 or N=10000, the corresponding points are not comparable to the smaller ones.

### Scope and known limits

3 seeds per cell (not 5) and h equation only — enough to separate a 3× difference, not enough for a
precise exponent. The fit is over four points spanning one decade; extrapolating two decades beyond
it is exactly why the N=10000 point is being measured rather than predicted. `s` equation and 5-seed
confirmation are cheap follow-ups if the answer lands near the ambiguous band.

