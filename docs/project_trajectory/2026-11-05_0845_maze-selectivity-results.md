# Maze selectivity (Figure 1e): 154 to 200 of 1,000 units are selective, almost all of them active

**Date:** 2026-11-05 08:45
**Issue:** #3
**Script:** rl_maze_population_stats.py (efc1edf and the JSON-writer fix that follows); the rule and
the five task variables were fixed in entry 2026-11-05_0832 before any value was computed.
**Output:** data/population_stats_maze.json (committed with the redrawn Figure 1).

The three N = 1,000 ReLU agents of the uniform series at their final weights, the training's
clean probe (128 fixed drops, noise off, mean action), every decision before the cheese as one
sample (24,886 to 27,206 per agent), the S1 to S4 rule of population_stats.py against the turn and
speed commands, the distance to the cheese and the heading (cosine, sine).

| seed | learned | probe success | active | S1 dimensionality all / active | S2 (standardised rates) all / active | selective over all units | selective over active units | inactive units counted selective | k-means classes all / active |
|---|---|---|---|---|---|---|---|---|---|
| 1 | yes | 0.99 | 421 | 10.7 / 10.6 | 46.2 / 20.0 | 154 of 1,000 (0.154) | 154 of 421 (0.366) | 0 | 3 / 3 |
| 2 | yes | 0.98 | 451 | 10.3 / 10.2 | 42.7 / 19.5 | 200 of 1,000 (0.200) | 199 of 451 (0.441) | 1 | 2 / 2 |
| 3 | no (0.73 at the end; 0.82 on this probe) | 0.82 | 420 | 9.3 / 9.3 | 35.5 / 15.8 | 197 of 1,000 (0.197) | 195 of 420 (0.464) | 2 | 3 / 3 |

## Reading

- The fraction of units selective for a task variable is 0.15 to 0.20 over all units and 0.37 to
  0.46 over the active units: the same gap, by the same cause, as on the supervised tasks (Figure
  1e; C8). Of the 154 to 200 selective units, 0 to 2 are inactive, as on 3BFF (0 to 1) and unlike
  CDDM (119 to 150 inactive units counted selective there).
- The participation-ratio dimensionality of the raw rates (S1) is the same over all units and over
  the active units (10.7 against 10.6, 10.3 against 10.2, 9.3 against 9.3). On standardised rates
  (S2) it is more than twice as large over all units (36 to 46) as over the active units (16 to 20):
  the inactive units' small variances are scaled up to one, as on the supervised tasks ("S2
  changes").
- The k-means class count is the same over all and over active units in every agent.
- The not-learned seed 3 looks like the learned ones on every statistic. Its probe success here
  (0.82, 420 active) differs from its last training probe (0.73, 415 active) because the probe's
  drops differ: the probe generator is seeded the same way, but a CUDA generator (training, on the
  GPU) and a CPU generator (here, on the Mac) give different streams for the same seed, so the 128
  drops are not the same set. Seeds 1 and 2 move by 0.00 and 0.00 in success and by 5 and 2 active
  units between the two drop sets; rl_maze_ablate.py's clean score is the same 0.82 on seed 3.

Figure 1e now draws the maze column with open markers (end-of-budget weights; the supervised
networks' markers are filled, their floor-crossing snapshots; dead-relu-paper#83). Panels d and e
draw the two learned agents (seeds 1 and 2); seed 3 is left out of both by the pre-registered
exclusion (a run whose last probe is below 0.9 enters no participation statistic), which panel c
already applied: the maze sub-panel of d reads 579 and 549 inactive units, success 0.889 and 0.916
with every inactive unit removed, 0.311 and 0.249 with as many at random. Figure redrawn
(img/internal_figures/fig_paper_F1.pdf, .svg; no overlapping text; the style test passes).
