# Maze warm-start test, result: warm starts are exact and not disruptive; the cold restart's effect is small and does not persist; without restarts the late Maze decrease is still inconsistent

Rules: `2026-11-17_2249_rules-maze-warm-start-test.md`. Script: `maze_warm_start_read.py` at 4dcf8da, committed while the
jobs ran and before any record was pulled, run unchanged. Jobs 17851661-17851663 (27 stage tasks, all completed, none
failed). Records in `data/trained_RNNs_rl_maze_warmtest/Maze_suite` with the nine parents linked in. Output
`data/maze_warm_start_read.json`.

## Per network (J: jump in active units across the boundary; noise: largest |J| at seven non-boundary points of the parent stage)

| Network | noise | J warm 20,000 | J warm 22,000 | J cold 20,000 | min. warm probe success 20,001-20,500 | R5 lost per 100k Adam steps, 20,000-24,000 | R5 return change / rise per 1,000 | R5 flat |
|---|---|---|---|---|---|---|---|---|
| N=500 s1 | 5 | +3 | +0 | +0 | 1.00 | 43 | -1.7% | yes |
| N=500 s2 | 5 | +1 | -1 | +1 | 0.99 | 13 | -0.6% | yes |
| N=500 s3 | 4 | -1 | -3 | +1 | 1.00 | -1 | -0.9% | yes |
| N=1000 s1 | 7 | -3 | +1 | +2 | 1.00 | -36 | +3.1% | no |
| N=1000 s2 | 9 | +0 | +1 | -1 | 1.00 | 107 | +3.8% | no |
| N=1000 s3 | 8 | +4 | -2 | +3 | 1.00 | 100 | +3.0% | no |
| N=1000 s4 | 10 | -4 | +1 | -2 | 0.98 | -43 | +0.7% | yes |
| N=1000 s5 | 6 | -4 | +2 | -5 | 1.00 | -27 | +0.4% | yes |
| N=1000 s6 | 5 | +2 | +2 | +3 | 1.00 | 37 | +6.3% | no |

## Verdicts (pre-set)

- **R1, warm start not disruptive: passes** (|J| within noise at 20,000 in 9 of 9 and at 22,000 in 9 of 9).
- **R4, behaviour: passes** (warm probe success 0.98-1.00 throughout updates 20,001-20,500).
- **R2, cold restart disruptive: not shown** (0 of 9 above noise). **R3, warm jump below cold jump: not shown** (4 of 9).
- **R5 (descriptive)**: warm arm over 20,000-24,000 (32,000 Adam steps, one warm boundary): 5 networks have a flat
  return, and 2 of those 5 lose at least 5 units per 100,000 Adam steps. Over all nine, the rates run from a loss of
  107 units to a gain of 43 units per 100,000 Adam steps.

By the rule, warm starts are adopted for every later Maze continuation.

## Why R2 fails: the jump window lies inside the cold warm-up (a flaw in the rule)

J compares the probes at updates U+1 to U+40 with the last 5 before U. The cold arm's learning rate ramps from 0 over its
first 50 updates, so those probes come before the restarted optimizer moves the weights much. My earlier diagnostic
(entry 2026-11-17 result) measured after the ramp.

**Post-hoc paired comparison (not pre-registered).** Both arms start from the same weights and draw the same random
numbers. The table gives the cold arm's median active count minus the warm arm's:

| Window (updates) | 20,050-20,300 | 20,300-20,700 | 20,700-21,300 | 21,300-22,000 |
|---|---|---|---|---|
| median over 9 networks | +3 | +1 | +1.5 | +1.5 |
| cold > warm in | 7 of 9 | 5 of 9 | 5 of 9 | 5 of 9 |
| range | -1 to +20 | -12 to +22 | -10 to +20 | -11 to +22 |

A restart reactivates a few units right after the warm-up in most networks (median 3, at most 20). After that the two
trajectories drift apart in both directions, and the difference is not systematic. The restart is not what keeps late
Maze networks from losing units. The 3-73 units "at stage boundaries" of the earlier diagnostic sum seven boundaries,
each measured with fluctuations of a similar size; they overstate a lasting restart effect.

## Consequences

- Warm starts: adopted for new Maze continuations (`INIT_ADAM=true` where the parent has an AdamState file).
- The paper's Maze statements stand (decrease from update 4,000 to 20,000 in 9 of 9; decrease slows once the return is
  flat). Nothing in the paper changes from this test. Methods already state that stages restarted Adam; the effect of a
  restart (median 3 more active units just after the warm-up, not lasting) can be added there.
- The difference between late Maze and supervised training remains. Without restarts, flat-return Maze networks lose
  units in 2 of 5 at the supervised bar; with restarts (15,000-20,000), 5 of 7 did. Late in Maze training the decrease
  is weak and varies between networks. Untested explanations: the networks near a minimal number of units the task
  needs (N = 500 at 145-170 active, N = 1,000 at 210-270); the policy's action noise and changing episode distribution
  reactivating units in some networks.
