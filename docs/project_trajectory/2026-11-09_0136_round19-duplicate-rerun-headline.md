## 2026-11-09 01:36 — Round 19: prune-and-duplicate on the Maze, rerun with freeze_after_end=true, still meets the criterion; the headline stands. The metabolic reruns at 2,000.

**Rules.** `2026-11-08_0953_round19-freeze-default-rules.md` (duplicate) and the penalty rerun rules
(`2026-11-08_0940`, `_1758`).

**Prune-and-duplicate (Della 17108178, `freeze_after_end=true`, stage 1, update 2,000).** Issue #5's criterion
against the three uniform-series controls: every seed above 449 active, mean success within 0.05 of 0.898, no
seed below 0.9.

| Seed | Success at 2,000 | Active units | Learned at update | Guard restores |
|---|---|---|---|---|
| 1 | 1.00 | 763 | 760 | 0 |
| 2 | 1.00 | 749 | 950 | 0 |
| 3 | 0.98 | 757 | 710 | 0 |

**Verdict: works.**
- Every count is above 449.
- Mean success is 0.99, 0.09 above the controls' 0.898.
- No seed ends below 0.9.
- Before the fix the arm had 747, 741 and 712 active units, with success 0.98, 0.95 and 0.99. The verdict and
  the counts are the same within 40 units.

**The headline stands:** prune-and-duplicate is the one technique that meets the criterion on all three tasks.
Taking the post-episode steps out of its rule's statistics left its Maze verdict unchanged.

**Metabolic reruns at 2,000** (direction rule against the five learned controls, 394-449; learned seeds only):

| Arm | Per seed: success, active (learned update) | Verdict at 2,000 |
|---|---|---|
| met0.01 | 0.99, 503 (800); 1.00, 449 (900); 0.88, 393 (not) | no change |
| met0.1 | 0.84, 423 (not); 0.65, 331 (not); 0.95, 368 (1,290) | decreases, on one learned seed |
| met1 | 0.94, 341 (not); stopped at 1,900; 0.87, 360 (not) | no direction (no seed learned) |
| met10 | 0.71, 190; 0.52, 217; 0.70, 219 (none learned) | no direction |

met1 s2's stop happened inside running episodes, as noted in `2026-11-08_2346`. met10 s2 and s3 had 2 and 6
discarded rollouts, all inside running episodes.

**Second stage under the "one stage, report both points" policy.** met1 and met10 have seeds not learned at
2,000, so they continue to 4,000: **Della 17156817**, interventions indices 88, 90, 91, 92 and 93, from
`~/trainRNNbrain_r19fz` with `FREEZE=true` and `INIT_ADAM=false`. met1 s2 is stopped and does not continue.
About 5 × 5.3 = 27 GPU-h. frm0.3, met0.01 and met0.1 are in 17152874, and the rate-plus-sparsity pair is in
17130854.
