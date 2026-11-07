## 2026-11-07 04:21 — Round 18, G17: no candidate second maze passes the difficulty check; G17 not submitted

Applies the rule of `2026-11-07_0259_round18-second-maze-calibration.md`. Spock array 7397777, 39 CPU jobs
from `~/trainRNNbrain_r18calib` at 75e8d7d, all finished at update 300. Run folders copied to the Mac,
`data/trained_RNNs_rl_maze_calib` on Spock. The measure is probe success (the share of 128 noise-free test
episodes from fixed starts that reach the cheese) averaged over the last five probes (updates 260-300),
per N = 100 network and then over network seeds 1-3.

| `make_maze.py` seed | Seed 1 | Seed 2 | Seed 3 | Mean | Passes (0.156-0.622) |
|---|---|---|---|---|---|
| 20 (first maze, the reference) | 0.369 | 0.366 | 0.198 | 0.311 | |
| 1446 | 0.012 | 0.016 | 0.002 | 0.010 | no |
| 1610 | 0.016 | 0.000 | 0.011 | 0.009 | no |
| 958 | 0.000 | 0.000 | 0.008 | 0.003 | no |
| 1531 | 0.028 | 0.027 | 0.028 | 0.028 | no |
| 1485 | 0.000 | 0.000 | 0.000 | 0.000 | no |
| 1086 | 0.158 | 0.175 | 0.062 | 0.132 | no |
| 793 | 0.000 | 0.014 | 0.000 | 0.005 | no |
| 1926 | 0.016 | 0.016 | 0.027 | 0.019 | no |
| 1644 | 0.002 | 0.008 | 0.084 | 0.031 | no |
| 861 | 0.000 | 0.000 | 0.008 | 0.003 | no |
| 781 | 0.000 | 0.003 | 0.008 | 0.004 | no |
| 1372 | 0.000 | 0.000 | 0.000 | 0.000 | no |

The reference measured on Spock is 0.311 (Mac, same check: 0.267), so the acceptance range is
0.156-0.622. No candidate falls in it. The closest is seed 1086 at 0.132, 0.024 below the lower bound
(it would also fail against the Mac reference, whose lower bound is 0.134). Every other candidate scores
0.031 or less: after 300 updates the N = 100 networks reach the cheese in at most 4 of 128 probe
episodes, against 25-47 of 128 on the first maze. With the three candidates of the first round
(222, 761, 830: 0.000-0.008), 15 mazes with the first maze's size, loop count and geodesic statistics
have been checked, and none matches its difficulty.

**By the rule, G17 is not submitted.** The choice goes to the coordinator on dead-relu-paper#201. The
options on record, with what each would change:
- **Seed 1086 with its deviation stated.** At 0.132 it is the only candidate within a factor of 2.5 of
  the first maze; after 300 updates N = 100 networks reach the cheese on it in 0.132 of probe episodes,
  against 0.311 on the first maze. The verdict rules need fresh networks to learn the second maze (2 of 3), which
  at N = 1,000 and 2,000 updates is untested for this maze.
- **The first maze mirrored left to right.** Its difficulty equals the first maze's by the symmetry of
  the task (the range sensors cover 360 degrees and turns are symmetric), and every route turns the other
  way, so the policy must change. It is not a make_maze.py draw, which the package asked for, and it
  shares the first maze's corridor geometry.
- **Report the transfer test on 3BFF and CDDM only**, and scope the Maze text (Figure 4, Results 4) as the
  audit's recommendation 4 describes.

The other cells of the package (G11, G12, G18 and the parity runs) are unaffected and running.
