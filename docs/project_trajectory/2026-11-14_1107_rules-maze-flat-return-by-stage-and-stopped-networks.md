# Rules added before any record after update 12,000 was read: the flat-return test by stage, and the stopped networks

Adds to `2026-11-14_*_rules-maze-long-training-to-20000.md`. Pavel (14 Nov): the Maze test has to be decided; the
non-finite stops are a technical problem to be solved. Written while stage 7 (job 17546732) was starting; no
record after update 12,000 has been read.

## What the records to update 12,000 show about the return (seen data; motivates the reading below)

Window: updates 9,000 to 12,000. "Slope" is the least-squares slope of the per-update training return, in percent
of the rise per 1,000 updates, with a block-bootstrap standard error (100-update blocks). "D" is the two-block
change of the existing rule.

| Network | D | slope | episode length, steps per 1,000 updates |
|---|---|---|---|
| N = 500 seed 1 | -5.1 | -7.4 ± 2.0 | +10.3 |
| N = 500 seed 3 | +1.8 | +1.2 ± 0.3 | -2.1 |
| N = 1,000 seed 1 | -1.0 | +2.7 ± 1.5 | -1.0 |
| N = 1,000 seed 3 | +13.0 | +8.6 ± 2.5 | -6.7 |
| N = 1,000 seed 4 | +3.2 | +9.1 ± 1.4 | -9.1 |
| N = 1,000 seed 5 | +5.4 | +6.7 ± 1.5 | -6.9 |

- In three networks the return is still rising (7 to 9% of the rise per 1,000 updates, three to six standard
  errors from zero) because the episodes still get shorter (7 to 9 steps per 1,000 updates, from about 165). The
  goal reward is discounted by 0.99 per step, so at 150 steps one step fewer changes the goal term by 1%.
- The 100-update block medians scatter around the trend with a standard deviation of 2 to 11% of the rise. The
  two-block D divides the difference of two such medians by 2.9 (thousand updates), so its own noise is about 1 to
  5% of the rise per 1,000 updates, the size of the 3% bar. D can call a flat return not flat, and the reverse.
- The return moves in plateaus: N = 1,000 seed 3 has block medians 1.33, 1.19, 1.18, 1.18, 1.00 over updates
  4,000 to 11,000 and 1.47 over the last 1,000.

## Reading D: the flat-return test by stage (new, applies only to updates after 12,000)

Unit: one stage of one network (2,000 updates; stages 7-10, so up to four per network). In each stage:
- return slope = least-squares slope of the per-update training return over the stage's 2,000 updates, per 1,000
  updates, divided by the network's rise (as in `maze_flat_return_full.measure`);
- the stage is **flat** if the absolute slope is below 3% (the bar of the existing rule);
- the stage is **decreasing** if the least-squares slope of the number of active units over the stage's probes
  is a loss of at least 5 units per 1,000 updates (the bar of the existing rule).
Verdict: needs at least 8 flat stages coming from at least 4 different networks. "Decreasing while the return is
flat" if at least two thirds of the flat stages are decreasing AND, in at least two thirds of the networks that
have a flat stage, at least half of that network's flat stages are decreasing. "Not decreasing" if at most one
third of the flat stages are decreasing. Anything between: mixed, reported as such. The warm-up of each stage
(50 updates) is inside the stage and is not removed.

Which reading the paper uses: B (last quarter, two-block) stays the reading named first. If B has a verdict, the
paper uses B and reports D beside it. If B has no verdict and D has one, the paper uses D and says that D was
defined after the first continuation, before the records it is applied to existed.

## The stopped networks

The three networks that stopped before update 12,000 are resumed from the end of their last complete stage with
a trainer whose only change is the recovery from a non-finite rollout (to be written and tested on the stopped
networks' saved weights; it must leave a run without non-finite rollouts unchanged). A resumed network is counted
in readings A to D for updates after 12,000 and marked as resumed. Every count is reported twice: over the
networks never stopped, and over all. If one of the six running networks stops, it is resumed the same way.
