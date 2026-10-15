## 2026-07-27 (16:10) — standard-RNN results: size sweep and penalty sweep, all 100 jobs complete

Arrays `11672037` (reference, 40 jobs) and `11672038` (size sweep, 60 jobs) finished **100/100
COMPLETED** — clean exits, confirming the non-Dale clustering fix. Every cell holds exactly 5 nets
and 5 participation traces. Statistics computed on Della from the final trace snapshot
(`collect_stats.py` → `silent_stats_all.csv`, 220 networks including the earlier sweeps); figures by
[`plot_silent_summary.py`](../trainRNNbrain/experiments_and_analysis/plot_silent_summary.py).

**Two metrics throughout**, because a single threshold misleads (participation
`p = std(fr) + q_0.9(|fr|)`):
- **truly silent** — `p < 1e-6`; a ReLU unit that never fires has exactly `p = 0`.
- **scale-free** — `p < 5%` of the network's 95th-percentile participation; comparable across
  conditions whose overall activity scale differs.

### Figure 1 — silent units vs network size (standard RNNs)

![Silent units vs N, standard RNNs](../img/internal_figures/silent_vs_N_std.png)

| unpenalized | N=100 | N=250 | N=500 | N=1000 |
|---|---|---|---|---|
| h — truly silent / scale-free | 3% / 12% | 9% / 27% | 22% / 41% | **42% / 59%** |
| s — truly silent / scale-free | 3% / 7% | 7% / 22% | 19% / 40% | **40% / 59%** |

**Observation.** The size scaling survives de-constraining: silence rises monotonically with N in
both equation types, from a nearly clean N=100 (3% truly silent) to ~40% truly dead and ~59%
functionally silent at N=1000. **`frm` gives exactly 0 on both metrics at every size**, so the
rescue is not a large-N-only effect. The gap between the two metrics (~17 points at N=1000) is the
population that fires, but so weakly it is irrelevant beside the active units — visible only to the
scale-free criterion. The consequence for the "just train big and prune" objection is that the
*active* count (97 → 221 → 340 → 448 for h) grows with a **falling local exponent**
(0.899 → 0.621 → 0.398), pointing at a ceiling near ~750 active units rather than unbounded growth.

### Figure 2 — penalty configurations at N=1000 (standard RNNs)

![Silent units by penalty, N=1000](../img/internal_figures/silent_by_penalty_N1000_std.png)

| N=1000, truly silent / scale-free | none | rws | frm | both |
|---|---|---|---|---|
| h | 42% / 59% | 40% / **64%** | **0% / 0%** | **0% / 0%** |
| s | 40% / 59% | **22%** / 53% | **0% / 0%** | **0% / 0%** |

**Observation.** Only the firing-rate penalty rescues, and it does so completely — zero silent units
on both metrics, in both equation types. `rws` does not: in `h` it leaves the truly-silent count
unchanged while making the scale-free count *worse* (59% → 64%). The `s` case is the more
interesting one and is only visible because two metrics are reported: `rws` nearly halves the
truly-silent count (40% → 22%) while the scale-free count barely moves (59% → 53%). **It lifts units
off exact zero without making them participate** — cosmetic rescue, which a single hard threshold
would have scored as a genuine improvement.

### Figure 3 — constrained vs standard architecture (h equation, N=1000)

![Constrained vs standard](../img/internal_figures/silent_constrained_vs_unconstrained_h.png)

| h, N=1000 | none | rws | frm | both |
|---|---|---|---|---|
| truly silent — constrained vs standard | **6% vs 42%** | 46% vs 40% | 0 vs 0 | 0 vs 0 |
| scale-free — constrained vs standard | **56% vs 59%** | 65% vs 64% | 0 vs 0 | 0 vs 0 |

**Observation, and the reason both metrics are needed.** On the **scale-free** criterion the two
architectures are **indistinguishable** (56% vs 59% unpenalized; 65% vs 64% under `rws`) — the
phenomenon is not an artifact of Dale's law or I/O positivity, which is the point this figure exists
to make. On the **strict** criterion they look wildly different (6% vs 42%), but that gap is entirely
the I/O positivity floor: non-negative input weights with non-negative task inputs guarantee every
unit a small positive push, so no unit in a constrained network can be exactly zero. Same silent
population, different floor. Reporting only the strict metric would have suggested the constrained
architecture is nine times healthier, when it is merely nine times better at hiding it.

### Note on scope

**No networks have been trained at N=2000 or N=5000.** The only runs at those sizes were the
benchmark's 8 timed iterations for memory/speed sizing (job `11677325`). Trained networks exist for
N ∈ {100, 250, 500, 1000} only; the saturation-vs-growth question stays open until the large-N runs.

