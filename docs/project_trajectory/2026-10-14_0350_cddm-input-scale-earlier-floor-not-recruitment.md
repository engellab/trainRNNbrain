### 2026-10-14 03:50 — CDDM input scale: earlier floor, not recruitment

The CDDM input-scale ladder (N = 1000, 3 seeds, 50k, snapshots; R² MEASURED at each floor on
snapshots, 36 draws), read at the floor (beta guard on):

| input-row norm | active at floor | crossing | R² |
|---|---|---|---|
| default draw (claim-1 controls) | 422-437 | 24.9-25.9k | 0.830-0.836 |
| 0.5 | 420 / 585 / 606 | 20.6k / 8.3k / 7.3k | 0.828-0.837 |
| 2 | 695-720 | 4.4-4.8k | 0.827-0.833 |
| 5 | 645-663 | 6.5-6.9k | 0.802-0.831 |
| 20 (2 of 3 read) | 416-451 | 16-17k | 0.828-0.834 |

Check set before looking at the matched counts: if scale 2's advantage over the control at 25k and
50k is ≤ 50 units, the floor gain is an early-read artefact. Result, active at matched iterations
(per seed): 25k control 426-438, scale 2 328-337, scale 5 277-317, scale 20 280-308; 50k control
355-376, scale 2 247-268, scale 5 201-216, scale 20 166-185. **So on CDDM a larger input scale makes
the network converge 5x sooner, while ~700 units are still active, and then shed units FASTER - it
does not recruit.** (On the flip-flop the gain survives to the end of training, 206 → 329 at 150k.)
The paper's knob section now says this (dead-relu-paper main). The trade-off scatter
(`fig_f2_knob_tradeoff`) has all nine ladders; x axis widened to 1040 so the rate penalty's points
at exactly 1000 are not clipped. A general caution for the floor read-out follows: a knob that
speeds convergence is read earlier, when more units are still active; matched-iteration counts
should accompany every floor comparison in Figure 2.

