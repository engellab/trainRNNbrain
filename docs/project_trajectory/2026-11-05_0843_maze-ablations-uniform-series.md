# Maze ablations of the uniform series: removing every inactive unit changes success by at most 0.03

**Date:** 2026-11-05 08:43
**Issue:** #3
**Commit:** 382d488 (data/rl_maze_ablation/*.json, rl_maze_ablate.py on the Mac CPU)

Six agents of the uniform series at their final weights (update 2,000), the ablation of Figure 1d
and the Extended Data maze figure: units removed by zeroing their outgoing weights, scored by the
success rate over 256 fixed drops under the training noise with the mean action, 4 draws. "Inactive
removed" is the score with exactly the inactive units removed, quietest first; "random same k" the
mean over 5 random orders at the same count; "active removed" the busiest-first score at the
active count.

| network | active | full (clean) | inactive removed | random same k | active removed |
|---|---|---|---|---|---|
| N = 250 s1 | 147 of 250 | 0.95 (1.00) | 0.96 | 0.10 | 0.03 |
| N = 500 s1 | 265 of 500 | 0.94 (1.00) | 0.95 | 0.07 | 0.01 |
| N = 1,000 s1 | 421 of 1,000 | 0.90 (0.99) | 0.89 | 0.31 | 0.01 |
| N = 1,000 s2 | 451 of 1,000 | 0.89 (0.98) | 0.92 | 0.25 | 0.01 |
| N = 1,000 s3 (not learned) | 420 of 1,000 | 0.80 (0.82) | 0.79 | 0.22 | 0.01 |
| N = 2,000 s2 (at 2,000; continues) | 560 of 2,000 | 0.76 (0.89) | 0.79 | 0.14 | 0.01 |

Removing every inactive unit changes the noisy success by -0.01 to +0.03 in all six; removing as
many units at random lowers it to 0.07-0.31; removing the active units lowers it to 0.01-0.03. The
maze agrees with claim C1 at every size tested. The N = 2,000 network is ablated at update 2,000;
its continuation to 4,000 (Spock 7384301) will be ablated again at its end. The earlier ablations
of the first suite stay under data/rl_maze_ablation_v1. The rate-penalty networks are being
ablated next into data/rl_maze_ablation_interventions (Figure 5's maze counterpart, panel d).
