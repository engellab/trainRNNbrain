# Maze long warm continuation: result (Phase B). P1 gives no verdict: by the matched flat bar the return is still rising in 7 of 9 networks; 8 of 9 lose active units at the supervised bar with no optimizer restart

Follows `2026-11-18_0745_rules-maze-long-warm-continuation.md` (rules, unchanged) and
`2026-11-18_2019_maze-long-warm-continuation-submitted.md`. All 18 stage tasks finished with exit 0 on Della
(stage 13 17897469, stage 14 17897470; 12.3-17.7 h each, 242 GPU-h; with Phase A, 389 GPU-h in all). No task
stopped early, so no recovery was run. The last stage-14 task ended on Friday 20 Nov at 10:37 Eastern, before the
reading deadline (22:00 Eastern), so the readings were taken then, over the full window W = updates 24,001-34,000
(80,000 Adam steps); the fallback window W' is not used. All nine networks reach update 34,000 with probe success
1.00 at the last probe: n = 9.

Read with `maze_warm_long_read.py` (committed in 7ebb7f7 before any Phase B record existed):

```
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python maze_warm_long_read.py <reading root>/Maze_suite <out>.json
```

**Independent check.** U, F, the probe count (1,002) and the update count (10,000) were recomputed for every
network by a separate script that reads the stage 13 and 14 files directly, with no repo code
(`/Users/pt1290/firstmate/data/maze-plateau/check_reader/oracle_phaseB.py`; pass set before running: |dU| < 0.01,
|dF| < 0.0001). All nine agree. The rise comes from the reader in both (the same value Phase A used).

## Per network

U: units lost per 100,000 Adam steps (minus the least-squares slope of the active count; positive = lost);
decreasing if U >= 5. F: least-squares slope of the per-update return per 100,000 Adam steps, as a share of the
network's rise (median return over the 100 updates up to the learned point minus the median over the first 100
updates); flat if |F| < 3%. "Old flat" (P2): `maze_flat_return_full.measure` from update 24,001, change between the
medians of the first and last 100 updates as a share of the rise per 1,000 updates, flat if below 3%.

| Network | active, first -> last 5 probes of W | U | decreasing | F | flat (P1) | old D per 1,000 updates | old flat (P2) | stage 14: U | stage 14: F |
|---|---|---|---|---|---|---|---|---|---|
| N=500 s1 | 138 -> 124 | 10 | yes | +14.7% | no | +1.0% | yes | 25 | +14.9% |
| N=500 s2 | 161 -> 153 | 26 | yes | +2.0% | **yes** | +0.7% | yes | 29 | +6.2% |
| N=500 s3 | 157 -> 132 | 29 | yes | +6.3% | no | +0.6% | yes | 25 | +18.9% |
| N=1000 s1 | 229 -> 177 | 50 | yes | +3.6% | no | +2.3% | yes | 60 | -12.6% |
| N=1000 s2 | 203 -> 177 | 58 | yes | +12.8% | no | +0.7% | yes | 63 | -16.5% |
| N=1000 s3 | 206 -> 230 | -33 | no | +17.3% | no | +1.3% | yes | -52 | +24.3% |
| N=1000 s4 | 225 -> 181 | 79 | yes | -14.0% | no | -0.3% | yes | 60 | +1.1% |
| N=1000 s5 | 248 -> 204 | 49 | yes | +25.5% | no | +2.1% | yes | 23 | +43.7% |
| N=1000 s6 | 257 -> 206 | 62 | yes | +35.6% | no | +2.6% | yes | 87 | +61.2% |

The median return over the last 500 updates of W exceeds that over the first 500 in all nine networks, by 1% to 26%
of the rise (N=1000 s4: 1.744 -> 1.763 on a rise of 1.58, although its least-squares slope is negative).

## Verdicts (rules as written)

- **P1 (primary), matched bars: no verdict.** 1 network is flat (N=500 s2, F = +2.0%), 5 needed. The return rises by
  more than 3% of the rise per 100,000 Adam steps in 7 networks (+3.6% to +35.6%) and falls by 14% in one (N=1000
  s4). The one flat network is decreasing (U = 26).
- **P2, the earlier Maze flat criterion: "decreasing while the return is flat".** 9 of 9 flat, 8 of the 9
  decreasing at the supervised bar (all but N=1000 s3, which gains 33 per 100,000 Adam steps). This flat criterion
  is 12.5 times looser per Adam step than the supervised one; over W it calls flat a return that rises by up to 30%
  of the rise.
- **P3, the earlier Maze decrease bar (62.5 units per 100,000 Adam steps) on P1's flat networks: no verdict** (1
  flat network, not decreasing at that bar, U = 26). Over all nine, 1 network reaches 62.5 (N=1000 s4, 79).
- **P4, stage 14 alone (updates 29,001-34,000): no verdict.** 1 flat network (N=1000 s4, F = +1.1%, U = 60,
  decreasing). U >= 5 in 8 of 9 (all but N=1000 s3), the same networks as over W, so the decrease in W does not come
  from a slow recovery after the restart at update 18,000.
- **P5, the warm boundaries: no disruption.** |J| exceeds the network's noise at update 24,000 in 0 of 9 and at
  update 29,000 in 1 of 9 (N=1000 s6, +7 against a noise of 5); more than 2 of 9 would have flagged one.

| Network | noise | J(24,000) | J(29,000) |
|---|---|---|---|
| N=500 s1 | 5 | +1 | 0 |
| N=500 s2 | 5 | 0 | +3 |
| N=500 s3 | 4 | -2 | 0 |
| N=1000 s1 | 7 | +2 | +1 |
| N=1000 s2 | 9 | -1 | -2 |
| N=1000 s3 | 8 | -3 | +1 |
| N=1000 s4 | 10 | -7 | -1 |
| N=1000 s5 | 6 | +3 | +3 |
| N=1000 s6 | 5 | +5 | +7 |

## Consequence for the paper's claim (mapping fixed in the rules entry)

P1 is "no verdict", so **Maze neither supports nor contradicts the claim that the number of active units keeps
decreasing while the loss or return is flat, and the claim stays scoped to the supervised tasks (3BFF and CDDM).**
The reason is that by the supervised flatness standard the Maze return has not stopped rising by update 34,000: the
test needs a flat window and Maze does not supply one.

Two facts sit beside the verdict (descriptive, not readings that decide the claim):

- With no optimizer restart in W (every boundary warm, the last restart 6,000 updates before W), 8 of 9 networks
  lose active units at the supervised bar, 10 to 79 per 100,000 Adam steps, and have 5% to 23% fewer active units at
  update 34,000 than at 24,000; N=1000 s3 gains 24 units. CDDM's flat-loss networks lose 34 to 64 per 100,000
  iterations (M12). The restart artefact therefore does not produce the Maze decrease. The Phase A warm arm
  (updates 20,001-24,000) had N=1000 s1, s4 and s5 gaining units; over W all three lose 49 to 79, so a 4,000-update
  window does not fix a network's sign.
- The paper as merged in dead-relu-paper PR #243 keeps the flat-loss claim as the last paragraph of Results 2 and
  cites Maze as 0 of 7 flat-return networks decreasing at 5 units per 1,000 updates (62.5 per 100,000 Adam steps).
  Under the mapping above, Maze should be described there as not testing the claim (its return keeps rising at
  the supervised flatness bar through update 34,000), not as a task on which it fails. Its headline replacement,
  "the number of active units continues to decrease after the task is learned", gains a restart-free Maze
  observation: 8 of 9 networks have fewer active units at update 34,000 than at 24,000.
