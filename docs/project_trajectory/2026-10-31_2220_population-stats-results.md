## 2026-10-31 22:20 — Population statistics over all units and over active units: results

Work package pop-stats (ptolmachev/dead-relu-paper#56, decision #34). Follows the rule entry
`2026-10-31_2212_population-stats-rule.md` (commit 80f01a1) with no departure. Code:
`trainRNNbrain/experiments_and_analysis/population_stats.py` at b6443c5, run on the Mac (CPU, 43 s
wall time for the oracle tests and all nine networks):

    cd trainRNNbrain/experiments_and_analysis
    /opt/miniconda3/envs/rnns/bin/python population_stats.py run

Weights: the nine Fig. 1d ablation snapshots, copied from Della to `data/pop_stats_nets/` (hashes in
the rule entry). Output: `data/population_stats.json` (gitignored; it holds every number below).

### Oracle tests (thresholds of the rule entry, Part 3): all pass

- V1a: eigenvalue PR and tr(C)^2 / tr(C^2) agree to 2e-16 relative on the random covariance, and to
  at most 1.1e-15 on the nine networks' covariances.
- V1b: orthonormal mixing, k = 3: S1 over 1,000 units 2.9996, over the 200 mixed units 2.9996
  (equal to 1e-15), S2 2.998; k = 10: 9.997, 9.997, 9.93. All within 5% of k. Gaussian mixing (no
  threshold): 2.94 and 9.59.
- V2: on the real CDDM design every planted partial R^2 (0.05, 0.2, 0.5 on each of the four
  regressors) is recovered to 4 decimals, with 0.0000 on the other regressors; 0.05 is not
  selective, 0.2 and 0.5 are selective and not mixed; the two 0.3 + 0.3 units are mixed-selective.
  Sampling version (20,000 samples, noise not orthogonalised): 0.0476, 0.2029, 0.5032.
- V3 (three planted classes of 100 units plus 700 zero units): S1 6.2957 over all and over the 300;
  S2 identical; selective 300 of 300 (share 0.3 of 1,000, 1.0 of 300), mixed 100; k* = 3 in both runs,
  every share 1/3, matched difference 0.0 points.
- V4 (the 700 extra units weak, 1e-3 of the amplitude, each tuned to one variable): S1 changes by
  0.0002% (6.3054 vs 6.3054); S2 doubles (15.3 vs 7.1); all 700 weak units are classified selective;
  k* 5 vs 3.

### Gates on the networks

- CDDM active counts on the rule's batch equal the ablation file's exactly (430, 416, 438): same
  450 conditions, same rule, so the weights and the build are the ones Fig. 1d used.
- DMTS and 3BFF (a different trial draw from the ablation's): 323, 412, 261 against 322, 411, 261;
  270, 340, 285 against 267, 342, 287.
- Noise-free R^2 on the rule's batch: CDDM 0.965-0.974, DMTS 0.985-0.997, 3BFF 0.813-0.819 (3BFF's
  noise-free loss is known to rise after about 10,000 iterations; AGENT_GUIDE "Reading networks").
  A task rebuilt wrongly would score near 0.

### Per-network table

"active" is the active count on the rule's batch; "zero-var." the units whose rate has zero variance
(rate 0 on every step and trial). Differences are (all - active) / active. Counts of selective and
mixed-selective units stand next to their shares; "inactive selective / mixed" counts the inactive
units classified selective and mixed-selective. S4 gives k* over all units with a profile / over
active units with a profile, and the largest difference between matched class shares.

| network | active | zero-var. | S1 all / act (diff) | S2 all / act (diff) | selective all / act (share all / act) | mixed all / act (share all / act) | inactive selective / mixed | S4 k* all / act, largest matched-share diff |
|---|---|---|---|---|---|---|---|---|
| 3BFF s1258717025 | 270 | 0 | 4.94 / 4.89 (+1.1%) | 25.2 / 12.5 (+102%) | 179 / 179 (0.18 / 0.66) | 11 / 11 (0.011 / 0.041) | 0 / 0 | 4 / 3, 14.3 pp |
| 3BFF s2461724787 | 340 | 0 | 5.15 / 5.11 (+0.8%) | 27.2 / 13.5 (+102%) | 245 / 244 (0.24 / 0.72) | 15 / 15 (0.015 / 0.044) | 1 / 0 | 4 / 3, 13.7 pp |
| 3BFF s4113307577 | 285 | 0 | 4.62 / 4.59 (+0.7%) | 23.0 / 12.5 (+84%) | 197 / 196 (0.20 / 0.69) | 8 / 8 (0.008 / 0.028) | 1 / 0 | 4 / 3, 20.2 pp |
| CDDM s1423687489 | 430 | 310 | 2.26 / 2.26 (+0.1%) | 18.7 / 11.3 (+65%) | 543 / 424 (0.54 / 0.99) | 477 / 390 (0.477 / 0.907) | 119 / 87 | 4 / 3, 12.9 pp |
| CDDM s2657899762 | 416 | 288 | 2.17 / 2.16 (+0.1%) | 18.4 / 11.1 (+66%) | 548 / 415 (0.55 / 1.00) | 459 / 370 (0.459 / 0.889) | 133 / 89 | 4 / 3, 15.1 pp |
| CDDM s2670295995 | 438 | 253 | 2.21 / 2.21 (+0.1%) | 19.5 / 11.4 (+71%) | 580 / 430 (0.58 / 0.98) | 501 / 388 (0.501 / 0.886) | 150 / 113 | 4 / 3, 13.4 pp |
| DMTS s198893703 | 323 | 100 | 3.72 / 3.71 (+0.2%) | 5.8 / 8.3 (-31%) | 343 / 300 (0.34 / 0.93) | 324 / 286 (0.324 / 0.885) | 43 / 38 | 2 / 2, 0.5 pp |
| DMTS s3322309204 | 412 | 40 | 3.12 / 3.09 (+1.1%) | 5.7 / 7.2 (-21%) | 470 / 371 (0.47 / 0.90) | 450 / 358 (0.450 / 0.869) | 99 / 92 | 2 / 2, 0.0 pp |
| DMTS s3753404946 | 261 | 65 | 3.91 / 3.90 (+0.3%) | 3.9 / 7.5 (-48%) | 229 / 215 (0.23 / 0.82) | 223 / 210 (0.223 / 0.805) | 14 / 13 | 6 / 6, 1.1 pp |

| task | S1 | S2 | S3 selective share | S3 mixed-selective share | S4 |
|---|---|---|---|---|---|
| 3BFF | unchanged | changes | changes | changes | changes |
| CDDM | unchanged | changes | changes | changes | changes |
| DMTS | unchanged | changes | changes | changes | unchanged |

The second table is the rule's label per statistic and task. In every cell the three networks carry
the same per-network flag, so no label rests on a split between networks.

### Each statistic labelled *changes*: did its value change, or only its denominator?

- **S2 (PR of normalised rates)**: its value changes. Over all units with nonzero variance, PR is
  roughly double the active-only value on 3BFF (23.0-27.2 against 12.5-13.5) and 1.7 times it on
  CDDM (18.4-19.5 against 11.1-11.4). On DMTS it moves the other way: 3.9-5.8 over all units against
  7.2-8.3 over active units.
- **S3 (shares selective and mixed-selective)**: on **3BFF only the denominator changes**: the
  inactive units add 0-1 selective units and no mixed-selective unit (179 / 179, 245 / 244,
  197 / 196 selective), so the share falls from 0.66-0.72 of the active units to 0.18-0.24 of 1,000
  only because the count is divided by 1,000. On **CDDM and DMTS the counts change too**: 119-150
  inactive units are selective on CDDM (87-113 mixed-selective), and 14-99 on DMTS (13-92 mixed).
  This contradicts the rule's prediction, which expected S3 on CDDM to change through the
  denominator alone; the silent CDDM units (253-310 per network) are not selective, but 119-150 of
  the 260-309 inactive units with nonzero variance are.
- **S4 (functional classes)**: its value changes on 3BFF and CDDM: k* is 4 over all units and 3 over
  active units in all six networks, and the largest matched-share difference is 13-20 points. On
  DMTS k* is the same (2, 2, 6) and the matched shares differ by at most 1.1 points, so S4 is
  *unchanged* there.
- **S1 (PR of rates)** is *unchanged* on every task: the largest difference is 1.1% (3BFF s1258717025,
  4.94 vs 4.89; DMTS s3322309204, 3.12 vs 3.09); CDDM differs by 0.1%.

### Against the rule's predictions

- "S1 unchanged on every task": confirmed.
- "S2 and S3 may change on 3BFF": both change; S2 also changes on CDDM and DMTS.
- "On CDDM, S3's shares change through the denominator alone": contradicted (counts change by
  119-150 units). On 3BFF, where no unit has zero variance, the S3 shares do change through the
  denominator alone.

### Facts a reader of these numbers needs

- **Units without an S3 response variance.** Besides the zero-variance units, many units are nonzero
  only during the first steps of a trial (the transient from the initial state) and zero during the
  S3 epoch: units without a profile number 387-433 on CDDM, 515-754 on DMTS and 353-399 on 3BFF.
  On DMTS 19 active units (s198893703) are among them. They have no partial R^2, so they are not
  selective and are not clustered, as Part 2 of the rule entry fixed.
- **S4 margins.** The silhouette at k* exceeds the runner-up by 0.01-0.07 on CDDM (all units: k = 4
  0.471 vs 0.447; 0.500 vs 0.427; 0.458 vs 0.445. Active units: k = 3 0.506 vs 0.495; 0.554 vs 0.509;
  0.517 vs 0.499) and by about 0.02 on 3BFF (0.017-0.025); the CDDM k* difference rests on near ties, but the matched
  shares also differ by 13-15 points there. Silhouettes for every k are in the JSON.
- **3BFF S3 samples.** 39,867 (step, trial) samples per network on the fixed 256-trial batch.
- **Optional N = 4,000 CDDM networks: not measured.** The three default CDDM networks at N = 4,000
  (`CDDM_paper_grid/EqType=h_N=4000_arm=control`) cross their floor at 28,397, 34,015 and 34,943
  iterations (`common.noisy_floor` on their copied `TrainLosses.json`) and kept weights only at
  100,000 iterations, about three times later. By the rule, the Discussion's sentence about
  4,000-unit CDDM networks is cut (for frame-discussion, #57).

### What the paper can say (by the rule)

Statistics labelled *changes* on at least two of three tasks: S2, S3 (both shares) and S4. S1 is
*unchanged* on every task and is named as unchanged. For S3 the paper states that on 3BFF the count
of selective units is unchanged and only the share moves; on CDDM and DMTS the count changes too.
