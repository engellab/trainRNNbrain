## 2026-11-01 07:41 — Population statistics: post-hoc checks, results

**Post hoc.** These are the checks planned in `2026-11-01_0739_population-stats-posthoc-plan.md`, after the
pre-registered results (`2026-10-31_2220_population-stats-results.md`) and the PR #66 review were seen.
They amend how that results entry is read: the DMTS sign reversal of S2 below belongs to it. The
pre-registered labels do not change. Run on the Mac (CPU, 42 s):
`/opt/miniconda3/envs/rnns/bin/python population_stats_posthoc.py` (code at the commit of this entry).
Output: `data/population_stats_posthoc.json` (gitignored). The numbers agree with the reviewer's
independent code to the printed precision, except for the noise counts of P5, which depend on the noise draw.

### P1. S2 with a soft normaliser

PR of rates divided by (sd + c), c = f x the median rate sd of the network's active units; over all
units with nonzero variance / over active units, (all - active) / active.

| network | f = 0 (the rule's S2) | f = 0.01 | f = 0.05 | f = 0.10 |
|---|---|---|---|---|
| 3BFF s1258717025 | 25.2 / 12.5 (+102%) | 18.2 / 12.4 (+47%) | 16.0 / 12.0 (+33%) | 14.9 / 11.7 (+27%) |
| 3BFF s2461724787 | 27.2 / 13.5 (+102%) | 18.5 / 13.4 (+38%) | 16.7 / 13.1 (+28%) | 15.6 / 12.7 (+22%) |
| 3BFF s4113307577 | 23.0 / 12.5 (+84%) | 17.0 / 12.4 (+37%) | 15.0 / 12.1 (+24%) | 14.0 / 11.7 (+19%) |
| CDDM s1423687489 | 18.7 / 11.3 (+65%) | 15.5 / 11.3 (+38%) | 13.4 / 11.0 (+22%) | 12.4 / 10.8 (+15%) |
| CDDM s2657899762 | 18.4 / 11.1 (+66%) | 15.4 / 11.1 (+39%) | 13.3 / 10.9 (+21%) | 12.4 / 10.7 (+15%) |
| CDDM s2670295995 | 19.5 / 11.4 (+71%) | 16.0 / 11.4 (+40%) | 13.9 / 11.2 (+24%) | 12.9 / 11.0 (+18%) |
| DMTS s198893703 | 5.8 / 8.3 (-31%) | 10.3 / 8.3 (+24%) | 9.0 / 8.1 (+10%) | 8.4 / 7.9 (+6%) |
| DMTS s3322309204 | 5.7 / 7.2 (-21%) | 9.1 / 7.2 (+26%) | 8.0 / 7.0 (+13%) | 7.6 / 6.9 (+10%) |
| DMTS s3753404946 | 3.9 / 7.5 (-48%) | 9.1 / 7.4 (+23%) | 7.8 / 7.1 (+9%) | 7.3 / 6.9 (+6%) |

On 3BFF and CDDM the gap stays above the rule's 10% at every f, so the label *changes* would hold
under any of these normalisers, but most of its size comes from dividing by near-zero standard
deviations: at f = 0.05 it is +24-33% on 3BFF and +21-24% on CDDM, against +84-102% and +65-71% at
f = 0. **On DMTS the sign reverses**: all < active at f = 0, all > active at every f > 0. At f = 0.05
the three networks give +10%, +13% and +9%, which the rule would label *mixed*.

### P2-P4. S4

- **3BFF.** The fourth class over all units has a flat centroid (0.32-0.36 on each bit) and holds
  173-275 units, 146-252 of them inactive and 10-13 selective. The clustered inactive units have a
  median summed partial R^2 of 0.0003; the active units 0.38-0.41. Dividing a profile by its sum
  turns a unit without selectivity into a full-weight equal-mixture profile. Clustering selective
  units only (P3) gives the same k* over all and over active units in all three networks (4/4, 3/3,
  5/5; matched shares within 0.3 points). At fixed k (P4) the matched shares differ by 21-33 points,
  because the flat class takes units from the other classes.
- **CDDM.** The silhouette margins at k* are 0.01-0.07 (results entry). At fixed k = 3 the matched
  shares differ by 5.6, 7.3 and 7.3 points (below the rule's 10); at k = 4 by 15.1, 8.4 and 4.2. The
  inactive units in the classes are selective: two classes dominated by motion or colour coherence
  each hold 43-55 inactive units; the context-dominated class holds 51-72 and the choice-dominated
  class 1. Selective units only: k* 4/3, 4/3 and 3/3.
- **DMTS.** Unchanged under every check: selective-only k* 2/2, 2/2, 6/6; fixed k within 4.6 points.

### P5. Size of the inactive selective units' responses

| task | inactive selective | median epoch-response sd, inactive selective / active selective | inactive selective with noise at 1% / 5% |
|---|---|---|---|
| CDDM | 119, 133, 150 | 0.0036 / 0.038; 0.0034 / 0.033; 0.0037 / 0.034 | 100 / 81; 116 / 92; 136 / 112 |
| DMTS | 43, 99, 14 | 0.014 / 0.32; 0.15 / 0.19; 0.057 / 0.44 | 32 / 23; 93 / 82; 13 / 12 |
| 3BFF | 0, 1, 1 | (none) / 0.50; 0.064 / 0.45; 0.044 / 0.56 | 0, 1, 1 at both levels |

The CDDM inactive selective units respond about a tenth as strongly as the active selective units.
Most of them stay selective under added noise, so they are not a fit to rounding error. On DMTS the
ratio ranges from about 1 in 20 to nearly equal (s3322309204).
