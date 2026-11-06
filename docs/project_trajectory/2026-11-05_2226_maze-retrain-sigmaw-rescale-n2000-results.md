# Maze results of 5 Nov: prune-and-retrain passes, noisy synapses and the gentler rescaling fail, N = 2,000 learned at updates 2,480 and 3,000

**Date:** 2026-11-05 22:26
**Issues:** #3 (Figure 1, the size series), #5 (interventions)
**Data copied to the Mac** from Spock (7384815, 7384816, 7383697, 7384301, 7384302) and Della
(16883344_22); figures redrawn: fig_paper_F1, fig_ed_maze, fig_ed_maze_interventions.

## Prune-and-retrain (Figure 1d's maze strip): the pre-registered rule passes

Rule (entry 2026-11-05_0832): in every learned network the agent retrained 200 updates without its
inactive units ends within 0.05 of the intact agent retrained the same way. Score = success under
the training noise, 256 fixed drops x 4 draws (Spock GPU; the inactive counts differ by up to 5
units from the Mac ablation's because the probe generator's stream differs between CUDA and the CPU).

| network | inactive removed | intact: after removal -> after 200 updates | inactive removed: after removal -> after 200 | as many at random: after removal -> after 200 | pruned minus intact at the end |
|---|---|---|---|---|---|
| N = 1,000 s1 | 574 | 0.892 -> 0.913 | 0.892 -> 0.925 | 0.369 -> 0.812 | +0.012 |
| N = 1,000 s2 | 551 | 0.903 -> 0.903 | 0.917 -> 0.925 | 0.234 -> 0.832 | +0.021 |
| N = 1,000 s3 (not learned, reported) | 585 | 0.787 -> 0.815 | 0.824 -> 0.861 | 0.266 -> 0.781 | +0.046 |

The agents retrained without their inactive units score as the intact agents retrained the same
way (0.925 against 0.913 and 0.903); the agents with as many units removed at random recover from
0.23-0.37 to 0.81-0.83 in 200 updates, below the other two arms. Figure 1d draws the stars and
triangles with open markers (end-of-budget weights).

## Interventions at N = 1,000 (fig_ed_maze_interventions.py; reference = the two learned controls, 437.5 active, success 0.984)

| arm | active at the end (seeds 1-3) | fold | success at the end | success minus controls' | verdict |
|---|---|---|---|---|---|
| noisy synapses sigma_w 1.0 | 451, 478, 454 | 1.05 | 0.91, 0.96, 0.90 | -0.062 | fails on success (count above every control's) |
| rescaling, the paper's values | stopped at updates 52-60 (three seeds) | - | - | - | stopped |
| rescaling, slow tilt (alpha 1.0005, cap 8) | seed 1 stopped at update 800 (the network went to 0 active units and the guard gave up after six non-finite replays); 314, 321 | 0.73 | 0.75, 0.63 | -0.293 | fails (2 of 3 finished) |
| rescaling, 5% growing (protect 0.05) | 255, 231, 234 | 0.55 | 0.66, 0.75, 0.81 | -0.245 | fails on both |

On the maze the rescaling rule REDUCES the number of active units (231-321 against 426-449 in the
controls) and the success rate, at every setting that ran to the end. Noisy synapses keep the count
(1.03-1.09x) at a success 0.06 below the controls'. Together with the earlier read-out (issue #5):
the rate penalty at 0.03 and 0.3 works; 0.1, the rate + sparsity pair, read-out dropout, noisy
synapses and rescaling fail. Prune-and-duplicate is queued (Spock 7389335_52-54). The three
rescaling arms are now told apart in the figure (lighter shades; the stopped run listed as such).

## Size series (fig_ed_maze): N = 2,000 learned on the 2,000 + 2,000 chain

| N | seed | updates | learned at | active at learned | active at the end | success at the end |
|---|---|---|---|---|---|---|
| 2,000 | 1 | 4,000 | 3,000 | 393 | 400 | 0.98 |
| 2,000 | 2 | 4,000 | 2,480 | 557 | 506 | 0.98 |
| 2,000 | 3 (rerun) | 2,000 | not yet | - | 431 | 0.84 (continued to 4,000, Spock 7389358) |
| 4,000 | 2 | 2,000 | not yet | - | 514 | 0.43 (continued to 4,000, Della 16955490) |

Fit of the end count over the learned runs of N = 250 to 2,000: N^0.55 (bootstrap 95% 0.48-0.61);
size means 152 (n = 2), 261 (3), 438 (2), 453 (2). N = 4,000: seed 1 is at update 2,280 of its
4,000 (success 0.65 at 1,280 of the continuation, 649 active), seeds 2 and 3 continue (16955490,
16955491; seed 3 stopped at update 1,760 on the old code and is continued from its update-1,700
checkpoint by 2,300). The first seed-3 attempt's files at N = 2,000 were moved to
Maze_suite/N=2000_act=relu_s3_first_attempt on Spock and on the Mac so the rerun's continuation
loads the rerun.
