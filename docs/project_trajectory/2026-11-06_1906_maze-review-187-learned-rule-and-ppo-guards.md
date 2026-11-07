# Review of dead-relu-paper PR #187: the learned-point rule has two conditions, and six size-series networks used the PPO guards (dead-relu-paper#182)

**Date:** 2026-11-06 1906
**Issue:** ptolmachev/dead-relu-paper#182 (review comment 6701368506 on PR #187)
**Scripts:** fig_ed_maze.py and fig_paper_F2_size.py (this commit); the counts below from the
Progress records under data/trained_RNNs_rl_maze/Maze_suite (recomputed here, equal to the reviewer's).

## The learned-point rule as applied (common.maze_learned_index)

The paper stated the learned point as "the first probe from which the clean probe's success rate
stays at or above 0.9 for five consecutive probes" and then said a network that ends below 0.9 has
no learned point. The code applies a second condition: the last probe must also be at or above 0.9.
Two of the four crossed networks of the size series held the level for six consecutive probes and
then fell below it: N = 1,000 seed 3 (0.90 to 0.98 over updates 1,750 to 1,800; 0.73 at the end)
and N = 4,000 seed 2 (3,900 to 3,950; 0.84 at the end). The paper now states both conditions
wherever the rule appears (Results 1, Methods: read-out, the Extended Data maze caption, the notes
of Supplementary Table 1), and fig_paper_F2_size.py prints the applied reason ("the last probe
(0.73) is below 0.9") instead of "never stayed at or above 0.9 for 5 probes".

## The PPO guards (TrainerPPO.safe_rollout, ppo_update) in the size series

Methods: Training said no update is skipped and no weights are restored; both guards exist. A
rollout whose log-probabilities or rewards are not finite is discarded, the weights are restored to
those after the last finite update and the rollout replayed up to three times, then up to three
more times from the last checkpoint (every 100 updates); a minibatch with a non-finite loss is
skipped. Restores / skipped minibatches summed over each chain: N = 250 seed 4, 1 / 0; N = 2,000
seed 2, 18 / 29 (all in the continuation); seed 3, 5 / 10 (all in the first run); N = 4,000 seed 1,
18 / 32; seed 2, 11 / 9; seed 3, 41 / 45 (all in the continuations); the other ten networks 0 / 0.

## Figure (fig_ed_maze.py)

Panel c: the not-learned crosses sit one marker right of the open markers (N x 1.07^3) and above
them; at the open markers' x two of the four (128 under 132 at N = 250, 415 under 426 at N = 1,000)
were hidden. The mean bars of c and e are ink, 0.8 pt, above the markers (in the hue and under the
markers they were invisible); panel e's axis ends at 1,050 so the tanh markers at 999 to 1,000 are
not cut. Numbers unchanged; no overlapping text; tests/test_paperstyle.py ok; no raster image.
