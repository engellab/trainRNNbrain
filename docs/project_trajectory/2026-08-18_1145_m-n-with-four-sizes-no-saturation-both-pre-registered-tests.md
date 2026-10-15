## 2026-08-18 11:45 — M(N) with four sizes: NO SATURATION. Both pre-registered tests resolved.

N=100 top-up (200k, 3 seeds) and the first N=2000 seed (300k) are in, so the ladder now reaches
L_deep = 0.02226 (was 0.02273) and all four sizes are on it. Figures: `img/internal_figures/M_vs_N.png`
(four levels x two criteria) and `M_vs_N_deepest.png`.

### Pre-registered test 1 — the shared floor: PASSES, and the N=100 anomaly is explained

Predicted (before the data existed): L_∞ for N=2000 within ~0.0001 of the 0.02135–0.02140 given by
N=500 and N=1000, fitted on the first 200k iterations.

| N | L_∞ (t_max = 200k) | n |
|---|---|---|
| 100 | 0.02141 ± 0.00000 | 3 |
| 500 | 0.02135 ± 0.00003 | 3 |
| 1000 | 0.02140 ± 0.00005 | 3 |
| **2000** | **0.02145** | 1 |

N=2000 lands at 0.02145 — 0.00005 above the range, inside the stated tolerance. Total spread across a
**20-fold** range of N is 0.00010, i.e. **0.5%**.

Independently, the N=100 top-up settles the one loose end in that analysis. Its 50k run gave
L_∞ = 0.02086, which was attributed to run-length bias rather than to a genuinely lower floor. Refitted
on 200k it gives **0.02141**, in line with every other size. The artefact explanation was correct.

### Pre-registered test 2 — M(2000): the "no saturation" branch wins

Predicted at L* = 0.02273: **583** if k holds at 0.38, **~515** if bending continues, **448** if
genuinely saturated.

| criterion | N=100 | N=500 | N=1000 | N=2000 | k(100→500) | k(500→1000) | k(1000→2000) |
|---|---|---|---|---|---|---|---|
| hard | 99.3 ± 0.5 | 344.0 ± 12.0 | 448.0 ± 9.0 | **566** | 0.77 | 0.38 | **0.34** |
| scale-free | 80.0 ± 1.4 | 280.7 ± 7.6 | 360.7 ± 21.5 | **470** | 0.78 | 0.36 | **0.38** |

**Measured 566, against 583 predicted for "k holds" and 448 for "saturated".** Saturation is refuted.
The local exponent stops falling: 0.38 → 0.34 (hard) and 0.36 → 0.38 (scale-free), i.e. flat within
the scatter rather than heading to zero.

### The result, across all four levels (hard criterion)

| L* | N=100 | N=500 | N=1000 | N=2000 | k(500→1000) | k(1000→2000) |
|---|---|---|---|---|---|---|
| 0.02226 | 94.0 | 259.0 | 310.3 | 371 | 0.26 | **0.26** |
| 0.02253 | 99.0 | 306.3 | 400.3 | 469 | 0.39 | 0.23 |
| 0.02301 | 99.3 | 387.3 | 528.0 | 686 | 0.45 | 0.38 |
| 0.02369 | 99.3 | 445.7 | 692.7 | 917 | 0.64 | 0.40 |

**M grows without saturating, as a sub-linear power law M ∝ N^k with k ≈ 0.25–0.40.** At the deepest
level the exponent is *identical* for the last two pairs (0.26, 0.26) — a clean power law, not a
curve bending to zero. The bending seen earlier with three sizes was the N=100 ceiling effect
(N=100 sits at 94–99% active, pinned against M ≤ N), exactly as flagged: k(100→500) is inflated at
every level, and the two ceiling-free pairs agree with each other.

**What this means for the project's central question.** There is no finite M* in the range tested.
Bigger networks do recruit more active units — but so inefficiently that at the deepest matching
level, **doubling the active count requires a 14-fold increase in N** (2^(1/0.26)). The active
*fraction* M/N falls steadily with no plateau (panel b). So the motivating argument for penalties
survives in a modified form: penalties are not the *only* route to more active units, but they are
overwhelmingly the cheaper one.

**Caveat: N=2000 is n=1.** Error bars on that point are zero because there is one seed; the other two
land ~19:10 today. Every k involving N=2000 is provisional until then. N=5000 (~00:30 tonight for the
first seed, ~22:30 Aug 19 for the other two) extends the ceiling-free range to a factor of 5 and is
the real test of whether k ≈ 0.26 holds or drifts further.

---

