## 2026-11-02 22:10 — Connectivity statistics over all units and over the active units: no statistic passes the rule

Work package r12-weights (dead-relu-paper #127, Pavel's item 52). Rule: entry
`2026-11-02_2203_connectivity-stats-rule.md`, committed as 4f25b25 before any value was computed.
Code: `connectivity_stats.py`, `connectivity_stats_handcheck.py`, `connectivity_stats_posthoc.py`
(commit 8058d93, branch wp/r12-weights, local). Run on the Mac, CPU, about 4 minutes:

```
PYTHONPATH=<code worktree> python trainRNNbrain/experiments_and_analysis/connectivity_stats.py
PYTHONPATH=<code worktree> python trainRNNbrain/experiments_and_analysis/connectivity_stats_handcheck.py
PYTHONPATH=<code worktree> python trainRNNbrain/experiments_and_analysis/connectivity_stats_posthoc.py
```

Inputs: `data/pop_stats_nets/` (the nine networks of Fig. 1d/1e and Extended Data Fig. 2) and
`data/connectivity_init/<label>/` (new: the kept snapshot at iteration 500 of the first run of each
chain and that run's config, copied from Della on 2 Nov; Della paths in each `source.txt`). Output
`data/connectivity_stats.json`.

**Verdict: no statistic passes.** W1-W4 have |rel| above 0.10 in all nine networks but fail the
size control in all nine. W5 has |rel| above 0.10 in five of nine networks. Under the rule, the
paper does not change. Extended Data Fig. 2a keeps the dimensionality panel, and the Introduction
list gets no connectivity statistic from this analysis.

### Checks

- **M1 (active mask): passed.** The active counts equal the stored `n_active` in all nine networks
  (3BFF 270, 340, 285; CDDM 430, 416, 438; DMTS 323, 412, 261).
- **M2 (rebuilt initial weights): passed.** The correlation between the rebuilt W0 and the
  iteration-500 snapshot is 0.973-0.987, and for the seed-plus-1 twin it is −0.0012 to 0.0026. θ is
  therefore 2 σ0, with σ0 = 0.0364-0.0376 (θ = 0.073-0.075).
- **M3 (independent hand check): passed.** For CDDM s1423687489, θ and W1, W2, W3 and W5, over the
  active units and over all units, agree between the pure-Python code and the script to within 4e-16.
- **M4 (the comparison on the initial weights): failed in one network for one statistic.** |rel| on
  W0 is at most 0.004 for W1 and at most 0.024 for W2 in all nine networks. For W3 and W4 it is at
  most 0.044, except W3 in CDDM s2670295995, where it is 0.060. The median degree changes in steps of
  1/k, so W3 near 0.05 moves by about 5% for a one-step change. M4 was a reported check and does not
  enter the verdict.

### Per-network values

rel = (S_all − S_act) / S_act. "size p95" is the 95th percentile of |(S_all − S_R) / S_R| over 200
random subsets R of |A| units; the rule required it below 0.05.

| network | active | W1 all / active, rel, size p95 | W2 all / active, rel, size p95 | W3 rel, size p95 | W4 rel, size p95 | W5 all / active, rel, size p95 |
|---|---|---|---|---|---|---|
| 3BFF s1258717025 | 270 | 0.040 / 0.129, −0.69, 0.24 | 0.025 / 0.202, −0.87, 0.26 | −1.00, nan | −0.98, 0.64 | 0.039 / 0.090, −0.57, 0.026 |
| 3BFF s2461724787 | 340 | 0.042 / 0.105, −0.60, 0.17 | 0.035 / 0.192, −0.82, 0.23 | −1.00, 0.83 | −0.97, 1.04 | 0.038 / 0.082, −0.53, 0.028 |
| 3BFF s4113307577 | 285 | 0.041 / 0.125, −0.67, 0.20 | 0.030 / 0.220, −0.86, 0.28 | −1.00, 0.86 | −0.97, 0.71 | 0.039 / 0.097, −0.60, 0.026 |
| CDDM s1423687489 | 430 | 0.022 / 0.043, −0.50, 0.086 | 0.017 / 0.081, −0.79, 0.22 | −0.87, 0.51 | −0.94, 0.72 | 0.0371 / 0.0390, −0.049, 0.009 |
| CDDM s2657899762 | 416 | 0.022 / 0.043, −0.50, 0.10 | 0.017 / 0.081, −0.79, 0.22 | −0.85, 0.46 | −0.93, 1.08 | 0.0371 / 0.0397, −0.066, 0.010 |
| CDDM s2670295995 | 438 | 0.022 / 0.042, −0.47, 0.088 | 0.018 / 0.076, −0.77, 0.23 | −0.80, 0.46 | −0.89, 0.53 | 0.0371 / 0.0385, −0.035, 0.010 |
| DMTS s198893703 | 323 | 0.015 / 0.040, −0.63, 0.13 | 0.008 / 0.070, −0.89, 0.32 | −1.00, nan | −1.00, 1.00 | 0.037 / 0.046, −0.20, 0.009 |
| DMTS s3322309204 | 412 | 0.012 / 0.025, −0.53, 0.11 | 0.003 / 0.015, −0.82, 0.40 | −1.00, nan | −1.00, nan | 0.0371 / 0.0363, +0.021, 0.007 |
| DMTS s3753404946 | 261 | 0.015 / 0.045, −0.68, 0.17 | 0.008 / 0.087, −0.91, 0.45 | −1.00, nan | −1.00, nan | 0.038 / 0.054, −0.31, 0.017 |

W3 and W4: the median normalised degree over all units is 0 to 0.010 in every network (0 in four of
nine for W3); over the active units it is 0.007-0.22. "nan" in the size column: S_R is 0 in some
random subsets, so rel is undefined there, and the control fails by the rule. At θ = 1 σ0 and 3 σ0
(secondary), rel for W2-W4 is −0.68 to −1.00 in every network, except W3 and W4 at 3 σ0 in DMTS
s3322309204, where both medians are 0 and rel is undefined.

The prediction in the rule entry (W1, W2 and W5 smaller over all units by more than 10% in every
network) held for W1 and W2 and failed for W5: in the three CDDM networks and in DMTS s3322309204,
the median weight change of a unit is about the same over all units as over the active units.

### Post hoc: why the size control fails (after the results were seen; does not change the verdict)

`connectivity_stats_posthoc.py`, 200 fresh random subsets per network.

1. **The control measures how many active units a random subset contains, not the subset's size.**
   A random subset of |A| units holds 49-215 active units, depending on the network and the draw.
   The trained active–active block has a larger s.d. than every other block, so S_R follows that
   number: the correlation between rel_R and the number of active units in R is −0.46 to −0.79 for
   W1 and W2 in all nine networks. A random subset of the same size as A therefore varies by more
   than 5% because the active units differ from the others, which is the effect being measured. The
   control as written could not pass for any statistic that differs between the active units and the
   rest, so it was badly designed. The rule fixed it before the results; it stays.
2. **No random subset reaches the active set's value.** In all nine networks, none of the 200
   subsets has W1, W2, W3 or W4 as large as over the active units. For W1, the largest subset value
   is 0.014-0.057, compared with 0.025-0.129 over the active units.

A test that would decide the question: a new rule fixed before computing, with the control "S_act
lies outside the range of S_R over 200 random subsets of |A| units", applied to networks this
analysis has not seen. On these nine networks that control has now been seen to pass, so it cannot
be the test.

### Post hoc: the weights among inactive units are near zero, not at their initial values

Measured (`connectivity_stats_posthoc.py`, block s.d., initial s.d. 0.036-0.038 in every network):

| network | W[A,A] | W[A,I] (to active from inactive) | W[I,A] | W[I,I] | slope of W[I,I] on W0[I,I] | W[I,I] with \|w\| < 1e-3 |
|---|---|---|---|---|---|---|
| 3BFF (3 networks) | 0.105-0.129 | 0.031-0.036 | 0.025-0.026 | 0.012-0.013 | 0.005-0.010 | 18-21% |
| CDDM (3) | 0.042-0.043 | 0.014-0.016 | 0.016-0.018 | 0.006-0.007 | 0.010-0.022 | 51-61% |
| DMTS (3) | 0.025-0.045 | 0.008-0.013 | 0.008-0.013 | 0.002-0.004 | 0.001-0.005 | 59-76% |

The regression slope of the trained weights among inactive units on their initial values is at most
0.022, so at most about 2% of the initial values remain. The s.d. of those weights is 6-35% of the
initial s.d. This result contradicts the rule entry's premise that the weights of inactive units
"stay close to W0", and it explains why W5 over all units (0.037) is about σ0: the change from W0 of
a weight that went to zero is −W0.

Interpretation, not tested: the optimiser is `torch.optim.Adam(..., weight_decay=1e-6)`
(`run_experiment.py`), which adds 1e-6·w to the gradient before Adam divides by the root mean square
of the gradient. When the task gradient of a weight is near zero, that division leaves a step of
order lr ≈ 4.6e-4 toward zero, which would remove a weight of 0.037 within about 80 iterations. The
test: train with weight decay 0 and check whether W[I,I] then keeps its initial values (slope near 1).
The CDDM weight-decay sweep of Results 3 has such networks. Opened as a separate issue, outside this
package.

### Files

- `trainRNNbrain/experiments_and_analysis/connectivity_stats.py`, `connectivity_stats_handcheck.py`,
  `connectivity_stats_posthoc.py` (8058d93).
- `data/connectivity_init/` (9 folders: `it0000500_LastParams.npz`, `root_config.yaml`, `source.txt`),
  `data/connectivity_stats.json`; both gitignored, on the Mac only.
