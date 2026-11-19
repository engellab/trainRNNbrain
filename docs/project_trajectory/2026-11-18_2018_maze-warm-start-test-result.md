# Maze warm-start test: result (Phase A). A warm start adds no jump and keeps performance (R1, R4 pass); the cold restart's jump is not seen in the pre-set window either (R2, R3 fail)

Follows `2026-11-17_2249_rules-maze-warm-start-test.md` (rules, unchanged) and
`2026-11-18_0727_maze-warm-start-test-submitted.md`. All 27 stage tasks finished with exit 0 on Della (warm 11
17851661, warm 12 17851662, cold 11 17851663; 4.98-7.07 h each, 147 GPU-h in all). Records pulled without the Adam
state files; read with `maze_warm_start_read.py` (committed in 7b62310 before any record was pulled):

```
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_warm_start_read.py <reading root>/Maze_suite <out>.json
```

The reading root holds the warm and cold cells and links to every earlier stage of the nine networks, so each arm
is read as one record from update 0. The parent's last 5 probes before update 20,000 are identical in the two arms
of every network (column "same before" of the reader).

## Verdicts (rules as written)

Jump J(U) = median active count of the first 5 probes after update U minus the median of the last 5 probes up to U.
Noise = the largest |J| over updates 18,250, 18,500, ..., 19,750 of the parent stage.

| Network | noise | J warm (20,000) | J warm (22,000) | J cold (20,000) | lowest probe success, warm, 20,001-20,500 |
|---|---|---|---|---|---|
| N=500 s1 | 5 | +3 | 0 | 0 | 1.00 |
| N=500 s2 | 5 | +1 | -1 | +1 | 0.99 |
| N=500 s3 | 4 | -1 | -3 | +1 | 1.00 |
| N=1000 s1 | 7 | -3 | +1 | +2 | 1.00 |
| N=1000 s2 | 9 | 0 | +1 | -1 | 1.00 |
| N=1000 s3 | 8 | +4 | -2 | +3 | 1.00 |
| N=1000 s4 | 10 | -4 | +1 | -2 | 0.98 |
| N=1000 s5 | 6 | -4 | +2 | -5 | 1.00 |
| N=1000 s6 | 5 | +2 | +2 | +3 | 1.00 |

- **R1, the warm start is not disruptive: pass.** |J warm| is at most the noise in 9 of 9 networks at update 20,000
  and in 9 of 9 at update 22,000 (7 of 9 needed at each).
- **R2, the cold restart is disruptive (replication): fail.** J cold (20,000) exceeds the noise in 0 of 9 (5
  needed).
- **R3, paired (J warm < J cold): fail.** 4 of 9 (7 needed).
- **R4, performance: pass.** Probe success stays at or above 0.9 at all 51 probes of updates 20,001-20,500 in every
  network (lowest 0.98).

By the rules, warm starts are adopted for every later Maze continuation (R1 and R4 pass), and Phase B (rules
`2026-11-18_0745_rules-maze-long-warm-continuation.md`) is submitted (entry
`2026-11-18_2019_maze-long-warm-continuation-submitted.md`).

What the four verdicts together say: a warm start adds no jump above noise, but at update 20,000 a cold restart
does not add one either, inside the window J reads. So this test shows that the warm start is safe; it does not
show that the warm start removes a disruption that the cold arm showed here. The earlier records agree with a small
jump per boundary: entry `2026-11-17_2218_maze-long-to-20000-result.md` gives a rise of 3 to 73 units summed over
eight cold boundaries per network, about 0.4 to 9 units per boundary, against noise of 4 to 10.

## R5 (descriptive, no verdict): the warm arm over updates 20,001-24,000

One boundary inside the window (22,000), warm; the last restart is at update 18,000. "Lost" is minus the
least-squares slope of the active count, per 100,000 Adam steps (1 update = 8 Adam steps); positive means units are
lost. The supervised bar is 5. "Flat (old)" is `maze_flat_return_full.measure` from update 20,001 (medians of the
first and last 100 updates; |D| < 3% of the rise per 1,000 updates), D being that change as a share of the rise.

| Network | active, first -> last 5 probes | lost per 100,000 Adam steps | >= 5 | D / rise per 1,000 updates | flat (old) |
|---|---|---|---|---|---|
| N=500 s1 | 152 -> 137 | 43 | yes | -1.7% | yes |
| N=500 s2 | 176 -> 161 | 14 | yes | -0.6% | yes |
| N=500 s3 | 155 -> 159 | -1 | no | -0.9% | yes |
| N=1000 s1 | 219 -> 227 | -36 | no | +3.1% | no |
| N=1000 s2 | 240 -> 204 | 107 | yes | +3.8% | no |
| N=1000 s3 | 228 -> 209 | 100 | yes | +3.0% | no |
| N=1000 s4 | 203 -> 232 | -43 | no | +0.7% | yes |
| N=1000 s5 | 230 -> 245 | -27 | no | +0.4% | yes |
| N=1000 s6 | 254 -> 252 | 37 | yes | +6.3% | no |

5 of 9 lose units at or above the supervised bar; 3 gain units (N=1000 s1, s4, s5) with no restart in the window.
Of the 5 networks flat by the old criterion, 2 lose units at the supervised bar. The gains of N=1000 s4 and s5
therefore do not come from optimizer restarts. (Over 4,000 updates and one window this says nothing about the
paper's claim; Phase B is the test.)

## After the data (descriptive, not a pre-set reading): the cold arm after its ramp

J reads the first 5 probes after the boundary, updates 20,001-20,040. In the cold arm all of them fall inside the
50-update ramp of the learning rate (1/50 to 40/50 of the base rate). The table compares the arms later. Each cell
is the arm's median active count in the window minus the median of the last 5 probes up to 20,000, warm / cold
(`maze_warm_cold_after.py`, written after R1-R4 were read; the same numbers came from an independent scratch
script):

| Network | before | 20,001-20,040 | 20,050-20,100 | 20,100-20,200 | 20,200-20,500 | 20,500-21,000 | 21,000-21,500 | 21,500-22,000 |
|---|---|---|---|---|---|---|---|---|
| N=500 s1 | 149 | +3 / 0 | +4 / +4 | +1 / +2 | -1 / 0 | +1 / -5 | 0 / -6 | -4 / -7 |
| N=500 s2 | 175 | +1 / +1 | -2 / -2 | -4 / -4 | -4 / -4 | -11 / -8 | -16 / -2 | -13 / -1 |
| N=500 s3 | 156 | -1 / +1 | -4 / 0 | +1 / +3 | 0 / +3 | +1 / +10 | -2 / +7 | -3 / +12 |
| N=1000 s1 | 222 | -3 / +2 | -4 / +4 | -4 / -1 | -5 / -6 | -2 / -1 | -3 / 0 | -9 / +1 |
| N=1000 s2 | 240 | 0 / -1 | -6 / +7 | -7 / +9 | -16 / +12 | -9 / -1 | -6 / -5 | -8 / -9 |
| N=1000 s3 | 224 | +4 / +3 | +6 / +8 | +7 / +10 | +4 / +8 | +7 / -5 | -3 / -7 | -8 / -2 |
| N=1000 s4 | 207 | -4 / -2 | -1 / +8 | 0 / +15 | +11 / +29 | +6 / +11 | +10 / +12 | +11 / +2 |
| N=1000 s5 | 234 | -4 / -5 | -4 / 0 | -9 / +5 | +1 / +10 | +2 / +20 | 0 / +20 | +5 / +28 |
| N=1000 s6 | 252 | +2 / +3 | +2 / +4 | +7 / +4 | +4 / -2 | +5 / -6 | +6 / +2 | 0 / -4 |
| cold above warm | | 4 of 9 | 7 of 9 | 7 of 9 | 6 of 9 | 6 of 9 | 6 of 9 | 5 of 9 |

From update 20,050 on, the cold arm holds more active units than the warm arm in 5 to 7 of 9 networks, by up to 28
units (N=1000 s2, updates 20,200-20,500). The reactivation after a cold restart appears once
the rate has ramped up, after the window J reads. This is a reason for Phase B's warm-only design; it changes none
of the verdicts above.
