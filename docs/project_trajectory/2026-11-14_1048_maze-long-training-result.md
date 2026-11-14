# Maze long training, result: the number of active units keeps decreasing to update 12,000 (6 of 6); the flat-return test still has no verdict

Rule: `2026-11-11_2322_rules-maze-long-training-and-retrain-at-250.md`, section 1. Script: `maze_long_read.py`
(committed before any continued record existed). Jobs: Della arrays 17379992-17379995; stops recorded in the two
entries of 2026-11-12. Output: `data/maze_long_read.json`. Read on 14 Nov; the background waiter did not fire
because the Mac lost its connection to Della.

## Which networks are counted

Nine networks were continued. Three stopped because the rollout was non-finite six times in a row and are not
counted or rerun: N = 1,000 seed 2 (update 5,900), N = 1,000 seed 6 (update 8,500) and, new in this entry,
N = 500 seed 2 (job 17379995_5, update 10,900). Six records reach update 12,000. In the three stopped networks the
logs show fewer active units before the stop than at update 4,000 (seed 2: 405 to 357; N = 500 seed 2: 209 at
update 6,000, 197 at 10,000); these are log lines, not part of the reading.

## Reading A, the decrease over updates 4,000 to 12,000: keeps decreasing (6 of 6)

| N | seed | probe success at the end | active at 4,000 | active at 12,000 | lost per 1,000 updates |
|---|---|---|---|---|---|
| 500 | 1 | 0.97 | 234 | 172 | 6.8 |
| 500 | 3 | 1.00 | 240 | 186 | 7.0 |
| 1,000 | 1 | 1.00 | 355 | 264 | 10.2 |
| 1,000 | 3 | 1.00 | 320 | 242 | 9.7 |
| 1,000 | 4 | 1.00 | 374 | 226 | 23.5 |
| 1,000 | 5 | 1.00 | 345 | 279 | 8.4 |

Active counts are medians of the last ten probes up to that update. Every network passes both conditions (at
least 5 units lost per 1,000 updates, later median below the earlier one). The number of active units is 19 to
40% lower at update 12,000.

Check against a second source: the last `active` value printed in each final-stage job log is 169, 180, 264, 256,
227, 282 for the six rows, within 14 units of the ten-probe medians.

## Reading B, the flat-return test over updates 9,000 to 12,000: no verdict

The return changes by -5.1, +1.8, -1.0, +13.0, +3.2 and +5.4% of the rise per 1,000 updates (same row order). It
is flat (below 3%) in two networks; the rule needs four. Both flat networks lose at least 5 active units per 1,000
updates (6.1 and 5.6), as do the four others (5.9 to 18.0).

## What the paper says as a consequence (pre-set outcomes)

- A "keeps decreasing": Results 2 and Methods state the Maze decrease from update 4,000 to 12,000; the
  abstract's sentence "the number of active units decreases as training progresses" stands for all three tasks.
- B without a verdict: the after-convergence sentence keeps its two tasks (3BFF, CDDM) in the abstract,
  Introduction and Discussion; Results 2 states that the Maze test gives no verdict and why.

Not tested: whether the return would become flat with a decayed learning rate; the rule entry excluded that
because a decayed rate would also stop the weights.
