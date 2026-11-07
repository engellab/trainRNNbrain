# Maze: every arm read out (package 3 landed); N = 4,000 seed 1 learned at update 2,860, seed 2 did not by 4,000

**Date:** 2026-11-06 17:23
**Issues:** #6 (the uniform programme), #3 (the size series)
**Scripts:** read_maze_arms.py (data/read_maze_arms.json), fig_ed_maze_settings.py, fig_ed_maze_interventions.py, fig_paper_F3_tasks.py, fig_paper_F5_tasks.py, fig_ed_maze.py, all re-run on the files copied from Spock (7389398, 42 runs) and Della (16907630_21, 16955490_22).

## Package 3 arms at N = 1,000 (reference: the two learned controls, 426 and 449 active, success 0.984; 3 seeds each)

| arm | active at the end | fold | success minus controls' | seeds learned | verdict |
|---|---|---|---|---|---|
| input-weight cap 0.3 | 434, 463, 380 | 0.97 | +0.003 | 3/3 | no change |
| recurrent-weight penalty 0.05 | 481, 459, 394 | 1.02 | -0.055 | 3/3 | no change |
| recurrent-weight penalty 0.2 | 517, 452, 405 | 1.05 | -0.034 | 2/3 | no change |
| metabolic penalty 0.01 | 537, 413, 379 | 1.01 | -0.047 | 2/3 | no change |
| metabolic penalty 0.1 | 431, 487, 387 | 0.99 | -0.036 | 2/3 | no change |
| metabolic penalty 1 | 390, 336, 359 | 0.83 | -0.112 | 2/3 | decreases |
| metabolic penalty 10 | 203, 162, 216 | 0.44 | -0.523 | 0/3 | decreases |
| AdamW (wd 1e-6) | 434, 452, 429 | 1.00 | -0.039 | 2/3 | no change |
| SGD 0.03 | 521, 556, 540 | 1.23 | -0.982 | 0/3 | increases the count; never learns (success 0.00-0.02) |
| SGD 0.1 | 686, 566, 540 | 1.37 | -0.927 | 0/3 | increases; never learns |
| SGD 0.3 | 488, 483, 611 | 1.21 | -0.904 | 0/3 | increases; never learns |
| SGD 1 | 268, 423, 371 | 0.81 | -0.924 | 0/3 | decreases; never learns |
| redraw (random re-initialisation) | 447, 457, 483 | 1.06 | +0.000 | 3/3 | fails: seed 1 (447) not above the largest control (449) |
| zero-out | 445, 440, 420 | 0.99 | -0.031 | 3/3 | fails on the count |

SGD at every rate keeps more units (at 0.03 to 0.3) and solves nothing: PPO under SGD at these
rates does not learn the maze in 2,000 updates (the first-update KL at rate 1 is 0.46 at N = 1,000;
at 0.03 the policy barely moves). The maze therefore cannot test the "SGD keeps more units at
lower performance" pattern of 3BFF: it is not a learned network. Reported as such.

## Across the three tasks (the settings; the rules are in decision dead-relu-paper#185)

Weight decay 1e-5 and 1e-4 lower the count on all three tasks. The input-weight norm raises it on
3BFF and at 0.5 on the maze and lowers it on CDDM. Activity noise 0.1 lowers it on the maze; 0 and
0.01 change nothing there. The sparsity penalty alone lowers it on CDDM and the maze. The metabolic
penalty at 1 and 10 lowers it on the maze (at 10 on CDDM the R^2 falls instead). AdamW and Adam
without weight decay raise it on both supervised tasks at equal R^2 and change nothing on the maze.
Re-initialising silent units (redraw, zero-out) changes nothing on any task. The 3BFF cells of
noise 0.01 and 0.1, spectral radius, the recurrent-weight and metabolic penalties, sparsity, the
activations and the cap finish on Della tonight (16957556 to 16957562; the status board lists them).

## Size series with N = 4,000 (fig_ed_maze; chains of 2,000 + 2,000, or 1,000 + 3,000 for seed 1)

| N | seed | updates | learned at | active at learned | active at the end | success at the end |
|---|---|---|---|---|---|---|
| 2,000 | 3 | 4,000 | 2,840 | 414 | 388 | 1.00 |
| 4,000 | 1 | 4,000 | 2,860 | 619 | 676 | 0.96 |
| 4,000 | 2 | 4,000 | not learned | - | 516 | 0.84 |
| 4,000 | 3 | 2,180 of 4,000 (running) | - | - | - | 0.77 at 2,180 |

Fit of the end count over the learned runs, N = 250 to 4,000: N^0.52 (bootstrap 95% 0.48 to
0.55); size means 146 (n = 3, with the new seed 4), 261 (3), 438 (2), 431 (3), 676 (1). N = 4,000
has one learned network of two finished; seeds 4 and 5 are submitted (Della 17012367, 2,000
updates, to be continued by 2,000 as the others) so the largest size does not rest on one network.
The end-weight ablations of the N = 2,000 chains and of N = 4,000 seed 1 are running on the Mac.
