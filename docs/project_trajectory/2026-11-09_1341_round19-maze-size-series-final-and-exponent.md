## 2026-11-09 13:41 — Round 19: the Maze ReLU size series at its final budgets (N = 4,000 seed 5 learned at 8,000; seed 4 stopped) and the size exponent refit

**Rules applied.** `2026-11-07_1008_round19-train-all-rules.md`:
- every network of one size shares one budget, which at N = 4,000 is 8,000 updates;
- the across-size comparison is read at the learned point (the 16 Oct rule);
- the fit on equal 4,000-update budgets is computed at update 4,000, with N = 4,000 represented only by the
  networks learned by then;
- the end read at each size's own final budget has unequal budgets (4,000 updates for N ≤ 2,000, 8,000 for
  N = 4,000) and is labelled so.
Tracking: ptolmachev/dead-relu-paper#201.

*Learned* is `common.maze_learned_index`: clean probe success at or above 0.9 for five probes in a row, with the
last probe also at or above 0.9. *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active
if p ≥ 0.05·q95(p).

### N = 4,000 seed 5, stage 4 (Della 17106119_25)

The job completed at 13:38 Pacific: updates 6,001-8,000, 51,107 s, 25.6 s per update, exit code 0. Its last probe,
at update 8,000, read 0.99 success with 533 of 4,000 units active.
- Stage 4 had 32 guard restore events with 53 discarded rollouts: 21 events by update 7,800 and 11 after it.
  The run did not stop. This code (1a80f5e) does not classify restores as post-episode or running-episode.
  Earlier stages had 2 events (stage 2) and 18 events (stage 3).
- Seed 5 is learned at 8,000. Its learned point stays at update 3,970, with 553 active units there.

**Copy.** Before copying, I checked the checksums of every N = 4,000 progress and config file that the Mac
already had: all 31 matched Della's. The Mac's `N=4000_act=relu_cont4/s5` was an empty folder, so
`rsync -a --ignore-existing` (without the Adam state and the PNGs) added its four files. Their checksums
match Della's.

### The series at its final budgets

Read by `trainRNNbrain/experiments_and_analysis/round19_size_exponent.py` (new; output
`data/round19/size_exponent.json`) from `fig_paper_F2_size.maze_series` on the Mac's `Maze_suite`.

| N = 4,000 seed | Budget reached | Learned point (active there) | End: update, success, active of 4,000 | At update 4,000: learned, active |
|---|---|---|---|---|
| 1 | 8,000 | 2,860 (619) | 8,000, 1.00, 557 | yes, 676 |
| 2 | 8,000 | 3,900 (532) | 8,000, 0.96, 595 | no, 516 |
| 3 | 8,000 | 4,120 (681) | 8,000, 0.99, 702 | no, 693 |
| 4 | stopped at 5,940 (non-finite weights) | none | not read | no, 521 |
| 5 | 8,000 | 3,970 (553) | 8,000, 0.99, 533 | no, 564 |

- Seed 4 stays "stopped at update 5,940 (non-finite weights)", as entry 2026-11-08_2346 reported. It is left out
  of every fit. Its record ends at 6,000 with a last probe of 0.00, so the reader counts it as not learned.
- Seed 2's learned point, 3,900, lies before update 4,000. At 4,000 it is still not learned, because the last
  probe of its cut record, 0.84, is below 0.9. Seed 5 is not learned at 4,000 either: its last probe there is
  0.91, but the five-probe hold that starts at 3,970 ends after update 4,000.
- The other sizes are unchanged from entry 2026-11-08_0915. Every ReLU network at N = 250 (4), 500 (3), 1,000 (6)
  and 2,000 (3) is learned at 4,000 updates.

**Cross-check against the log.** The log's last line for seed 5 reads probe success 0.99 with 533 of 4,000
active. The reader reads the same values. Seeds 1, 2, 3 and 5 have learned points of 2,860, 3,900, 4,120 and
3,970, as the earlier entries stated.

### The exponent refit

Fit as in `round18_training_read.read_exponent`: the least-squares slope of log mean active count on log N
(`power_fit`), with a 95% bootstrap interval that resamples networks within each size (`boot_exponent`,
`BOOT_SEED`).

**Check first.** With the round 18 set, which is N = 1,000 seeds 1-3, every record cut at 4,000, and N = 4,000
seed 1 alone, the script gives 0.351 (0.318-0.384) at the learned point and 0.491 (0.463-0.520) at update 4,000.
Entry 2026-11-08_0859 gave the same numbers.

| Read-out point, N ≥ 500 | Networks at N = 500, 1,000, 2,000, 4,000 | Mean active units at those sizes | b (95% bootstrap) | Before (entry 2026-11-08_0859) |
|---|---|---|---|---|
| **Learned point, each network on its full record** | 3, 6, 3, 4 | 273, 440, 455, 596 | **0.343 (0.291-0.395)** | 0.351 (0.318-0.384), one network at N = 4,000 |
| Learned point, N = 1,000 seeds 1-3 only | 3, 3, 3, 4 | 273, 465, 455, 596 | 0.335 (0.279-0.386) | |
| **Update 4,000, equal budgets** (N = 4,000: the networks learned by 4,000) | 3, 6, 3, 1 | 231, 351, 431, 676 | **0.495 (0.469-0.521)** | 0.491 (0.463-0.520) |
| Update 4,000, N = 1,000 seeds 1-3 only | 3, 3, 3, 1 | 231, 360, 431, 676 | 0.491 (0.463-0.520) | |
| End at each size's final budget (**unequal**: 4,000 for N ≤ 2,000, 8,000 for N = 4,000) | 3, 6, 3, 4 | 231, 351, 431, 597 | 0.441 (0.391-0.495) | |

Every size (N = 250 included): 0.471 (0.426-0.518) at the learned point, 0.554 (0.528-0.580) at update 4,000 and
0.518 (0.481-0.559) at the final budgets.

- **The learned-point exponent changes from 0.351 to 0.343.** N = 4,000 now enters with four learned networks
  instead of one. Their mean, 596 active units, is below seed 1's 619. The interval is wider, 0.29-0.40 against
  0.32-0.38, because the four N = 4,000 networks have learned-point counts from 532 to 681.
- **The equal-budget exponent at update 4,000 is 0.495.** N = 4,000 is still represented only by seed 1, the one
  network learned by update 4,000. The change from 0.491 comes from N = 1,000 seeds 4-6, which now have records
  to 4,000.
- **At the final budgets the exponent is 0.44.** N = 4,000 is read there after 8,000 updates and the smaller
  sizes after 4,000, so the label "unequal budgets" goes with the number. Between update 4,000 and 8,000, seed 1
  went from 676 to 557 active units.

### What changes in the claims

- **C2 (Maze):** read at the learned point, the number of active units grows with network size as N^0.34
  (0.29-0.40), on 16 learned networks at N ≥ 500, four of them at N = 4,000. On equal 4,000-update budgets
  the exponent is 0.50 (0.47-0.52). The equal-budget fit still has one network at N = 4,000.
- Of the five N = 4,000 networks, four are learned within 8,000 updates (learned points 2,860-4,120). One stopped
  at update 5,940 with non-finite weights.
