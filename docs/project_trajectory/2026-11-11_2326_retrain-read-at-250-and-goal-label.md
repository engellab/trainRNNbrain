# Result: the removal-and-retraining curves read at 250 iterations; "cheese" renamed "goal"

Rule: `2026-11-11_2322_rules-maze-long-training-and-retrain-at-250.md`, section 2 (commit 3a0cd93 on
wp/maze-long, before the read). Data: the retraining JSONs of Della `c1_retrain_79932ef` (held-out R^2
every 250 iterations) and `data/rl_maze_retrain` (success rate every 25 updates). No new training.

## Held-out R^2 at 0 / 250 / 3,000 further iterations

| network | intact | inactive units removed | as many random units removed |
|---|---|---|---|
| 3BFF s1258717025 | 0.950 / 0.950 / 0.949 | 0.260 / 0.933 / 0.945 | 0.068 / 0.798 / 0.920 |
| 3BFF s2461724787 | 0.950 / 0.949 / 0.950 | 0.347 / 0.935 / 0.946 | 0.037 / 0.822 / 0.927 |
| 3BFF s4113307577 | 0.950 / 0.950 / 0.950 | 0.517 / 0.935 / 0.948 | 0.043 / 0.825 / 0.930 |
| CDDM s1423687489 | 0.833 / 0.831 / 0.829 | 0.832 / 0.830 / 0.829 | -0.210 / 0.822 / 0.828 |
| CDDM s2657899762 | 0.830 / 0.829 / 0.827 | 0.830 / 0.830 / 0.828 | 0.049 / 0.825 / 0.821 |
| CDDM s2670295995 | 0.827 / 0.829 / 0.825 | 0.824 / 0.829 / 0.825 | -0.284 / 0.819 / 0.828 |
| DMTS s198893703 | 0.999 / 0.999 / 0.999 | 0.944 / 0.999 / 0.999 | -0.098 / 0.923 / 0.998 |
| DMTS s3322309204 | 0.999 / 0.999 / 0.999 | 0.746 / 0.999 / 0.999 | -0.134 / 0.415 / 0.419 |
| DMTS s3753404946 | 0.999 / 0.999 / 0.999 | 0.988 / 0.999 / 0.999 | -0.154 / 0.409 / 0.998 |

Maze, success rate at 0 / 25 / 200 further updates: seed 1 intact 0.892 / 0.920 / 0.913, inactive removed
0.892 / 0.899 / 0.925, random removed 0.369 / 0.634 / 0.812; seed 2 intact 0.903 / 0.926 / 0.903, inactive
removed 0.917 / 0.937 / 0.925, random removed 0.234 / 0.562 / 0.832.

## Verdicts under the rule

- **Dispensable at 250 iterations (bar: within 0.01 of the intact network): fails in 3BFF.** The three 3BFF
  networks without their inactive units are 0.015 to 0.017 below the intact networks at 250 iterations
  (0.002 to 0.005 below at 3,000). CDDM passes (within 0.002) and so does the Maze at 25 updates (0.021
  below and 0.011 above; bar 0.05). As the rule says for this outcome, the paper keeps 3,000 iterations and
  states how fast the recovery is: 3BFF from 0.26-0.52 to about 0.93 in 250 iterations, less than 1% of the
  50,000 to 67,000 iterations before the removal.
- **Specific to inactive units at 250 iterations (bar: random removal more than 0.01 lower, in 5 of 6):
  fails, 3 of 6.** 3BFF: 0.110 to 0.136 lower. CDDM: 0.006 to 0.010 lower, inside the bar. The paper says
  that 3BFF and Maze networks recover more slowly and less completely after random removal, and that CDDM
  networks regain their performance after either removal.

## "Cheese" renamed "goal" (Pavel, 11 Nov)

`fig_supp_tasks.py`: three drawn labels ("find the goal from the walls alone", "goal", "goal reached").
The 110-dpi renders differ only inside the Maze row. The paper's text uses "the goal" throughout.
