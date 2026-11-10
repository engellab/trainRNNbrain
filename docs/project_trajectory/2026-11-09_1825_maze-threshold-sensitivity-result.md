## 2026-11-09 18:25 — Activity-threshold sensitivity of the Maze size exponent: both checks pass; b = 0.39 / 0.35 / 0.30

Work package r21-text (ptolmachev/dead-relu-paper#218). Rule: entry `2026-11-09_1824_maze-threshold-sensitivity-rule.md`,
written before the run. Script `trainRNNbrain/experiments_and_analysis/maze_threshold_sensitivity.py` (branch
`wp/r21-text`), run on the Mac in a few seconds from the local `data/trained_RNNs_rl_maze/Maze_suite`.

**Checks (set before running): both pass.**
- Same quantity: 1,640 stored participation vectors fall on probe updates in the 16 networks; at f = 0.05 the
  count equals the probe's stored count in all 1,640.
- The exponent at f = 0.05 from the vectors nearest the learned point is 0.353, against 0.343 for the counts of
  Fig. 2a (the probe at the learned point itself): a difference of 0.010, inside the 0.03 set in the rule. The
  stored vector is 0 to 50 updates from the learned point (t\* 1,280 → 1,300, etc.; one vector per 50 updates).

**Result** (16 networks: 3, 6, 3, 4 at N = 500, 1,000, 2,000, 4,000; N ≥ 500)

| factor f | 0.025 | 0.05 | 0.10 |
|---|---|---|---|
| exponent b (bootstrap 95%) | 0.394 (0.333-0.452) | 0.353 (0.299-0.408) | 0.304 (0.260-0.349) |
| count relative to f = 0.05 | +8% to +21% | — | −7% to −21% |

So for Maze, b ranges from 0.30 to 0.39 over the factors 0.025 to 0.10, and differs from its value at 0.05 by at
most 0.05, as for 3BFF (0.43-0.49), CDDM (0.44-0.48) and DMTS (0.54-0.60) in entry `2026-10-31_2211`. Halving the
factor raises the Maze count by 8-21% and doubling it lowers the count by 7-21%, inside the 3-25% and 9-28% of
the other three tasks except the 7% lower end.

**Output (verbatim)**

```
N     seed  t*     p_update  fig2a  f=0.025  f=0.05  f=0.1
  500 s1      1350     1350    281      303      281      249
  500 s2      1280     1300    259      272      253      235
  500 s3      1750     1750    279      306      279      253
 1000 s1      1050     1050    451      502      451      385
 1000 s2       760      750    517      575      514      438
 1000 s3      1750     1750    428      495      428      352
 1000 s4      1530     1550    427      502      438      366
 1000 s5      1180     1200    412      441      407      376
 1000 s6      1330     1350    406      462      413      374
 2000 s1      3000     3000    393      446      393      337
 2000 s2      2480     2500    557      676      560      445
 2000 s3      2840     2850    414      477      424      372
 4000 s1      2860     2850    619      734      622      506
 4000 s2      3900     3900    532      613      532      448
 4000 s3      4120     4100    681      818      694      561
 4000 s5      3970     3950    553      686      574      464

check: 1640 stored vectors at probe updates, 0 with a count different from the probe's
f = 0.025 : b = 0.394 (bootstrap 95% 0.333-0.452); n per size 500: 3, 1000: 6, 2000: 3, 4000: 4
f = 0.05  : b = 0.353 (bootstrap 95% 0.299-0.408); n per size 500: 3, 1000: 6, 2000: 3, 4000: 4
f = 0.1   : b = 0.304 (bootstrap 95% 0.260-0.349); n per size 500: 3, 1000: 6, 2000: 3, 4000: 4
count at f = 0.025 relative to 0.05: +8% to +21%
count at f = 0.1 relative to 0.05: -21% to -7%
```

**Paper.** Methods M04 gives the Maze range 0.30 to 0.39 beside the other tasks; Supplementary Methods
(Participation) names the 37 supervised networks correctly (16 3BFF, 12 CDDM, 9 DMTS) and the 16 Maze networks,
with the Maze count changes (dead-relu-paper PR for #218).

**Also recorded here for the same PR (no computation):** the DMTS size series (`DMTS_paper_grid_d5`, Della job
14830790, entry `2026-10-01_1305`) also holds 15 networks trained with a technique: prune-and-duplicate at
N = 500, 1,000 and 2,000 and read-out dropout (`arm=mute`) at the same sizes, three each. The stored final R² in
each run folder's name: prune-and-duplicate N = 500, 0.996, 0.999, 0.999; N = 1,000, 0.417, 0.418, 0.418;
N = 2,000, 0.417, 0.418, 0.419; read-out dropout N = 500, 0.413, 0.415, 0.415; N = 1,000, 0.414, 0.415, 0.416;
N = 2,000 (listed on Della, not copied to the Mac), 0.424, 0.426, 0.427. Chance level for DMTS is about 0.418
(the not-learned networks of entry `2026-11-02_1822`). So only the three prune-and-duplicate networks of
N = 500 learned the task, as entry `2026-10-16_1151` found ("trained from scratch and never learned").
