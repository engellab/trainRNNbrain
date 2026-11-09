## 2026-11-08 23:46 — Round 19: interim reads of the freeze_after_end reruns at 2,000; second-stage submissions; N = 4,000 seed 4 and met1 s2 stopped

**Rules.** `2026-11-08_0940_round19-penalty-reruns-rules.md` and `2026-11-08_1758_round19-penalty-reruns-stage2-rules.md`.

**Reruns at update 2,000.** All runs have `freeze_after_end=true`, read from the copied fz tree.
- Rate penalty, by issue #5's criterion: every seed above 449 active, mean success within 0.05 of 0.898, no seed
  below 0.9.
- Metabolic penalty, by the direction rule against the five learned controls (394-449), learned seeds only.
- No guard restore occurred in any frm rerun or in met0.01 or met0.1.

| Arm | Per seed at 2,000: success, active (learned update) | Verdict at 2,000 |
|---|---|---|
| frm0.03 | 1.00, 998 (770); 0.97, 997 (1,110); 1.00, 952 (460) | works |
| frm0.1 | 1.00, 998 (560); 0.99, 1,000 (760); 1.00, 997 (1,390) | works |
| frm0.3 | 1.00, 1,000 (970); 0.98, 1,000 (830); 0.87, 997 (not learned) | **fails: seed 3 ends at 0.87** (mean success 0.95; every count above 449) |
| frm0.1_rws0.05 | 0.39, 1,000; 0.68, 1,000; 0.34, 1,000 (none learned) | fails (mean success 0.47; all below 0.9) |
| met0.01 | 0.99, 503 (800); 1.00, 449 (900); 0.88, 393 (not learned) | no change (learned seeds 503 and 449; 449 is not above the range) |
| met0.1 | 0.84, 423 (not); 0.65, 331 (not); 0.95, 368 (1,290) | decreases, on its one learned seed (368) |
| met1, met10 | still running, or stopped (below) | not yet read |

**The Maze rate-penalty statement at 2,000 changes.** Before the fix it read "works at 0.03 and 0.3". With the
penalty taken over running steps only it reads **"works at 0.03 and 0.1; fails at 0.3, where one seed ends at
0.87"**. By the rule of the rerun rules entry, that is "works at the values that pass".

**Second stage, submitted under the "one stage, report both points" policy.** Each arm with a seed not learned at
2,000 continues whole to update 4,000:
- frm0.1_rws0.05: Della 17130854, entry 2026-11-08_1758;
- frm0.3, met0.01 and met0.1: **Della 17152874**, interventions indices 10-12 and 82-87, from
  `~/trainRNNbrain_r19fz` (fd564f2), with `FREEZE=true` as their parents had and `INIT_ADAM=false`. About
  9 × 5.3 = 48 GPU-h.
met1 and met10 are decided when all their seeds have finished.

**met1 s2 (freeze on): stopped at update 1,900.** It had 44 restore events, with 176 discarded rollouts, all
non-finite within a running episode and none after an episode's end. That fits the switch: with ended episodes
held, post-episode growth cannot occur. So this stop is instability in the task dynamics, not the
post-episode failure. Reported as stopped, per the rules.

**N = 4,000 seed 4.**
- Probe success held at 0.97-0.99, with about 530 active units, through update 5,940 of its stage 3 (the
  coordinator's job 17105347_24, code 1a80f5e).
- From update 5,941 the weights were non-finite after every update. Each next rollout was restored from the last
  checkpoint (255 restore attempts in stage 3), and the clean probe read success 0.00 with 0 active units from
  update 5,950 to 6,000.
- Its saved weights at update 6,000 are non-finite in all 16 million entries of W_rec. So its stage 4
  (17106118_24) failed within 28 s: every rollout from those weights is non-finite.
- That code does not classify restores, and no finite weights near update 5,941 were kept, so it cannot be told
  whether the first failure began after an episode's end or inside one. The stage-2 weights at update 4,000 are
  finite.

**How seed 4 enters the size series, by the rules.**
- It is **not learned at 6,000**, because its last probe at the stage end is 0.00.
- It cannot be trained further from its own weights, so it is reported as "**stopped at update 5,940
  (non-finite weights)**" and left out of the learned-point and end statistics, as every stopped network is.
- For the record, not as a verdict: on its record cut at update 5,940 the learned rule finds a learned point at
  update 5,010, with 504 active units.

**Options for seed 4, nothing submitted.**
- (i) Leave it stopped.
- (ii) Rerun stage 3 from its finite update-4,000 weights with the same setting. The replicates have shown that a
  same-code, same-seed replay is deterministic, so this would most likely repeat the failure at 5,941.
- (iii) Rerun stage 3 with `freeze_after_end` switched on (`FREEZE_SWITCH`), then stage 4. About 31 GPU-h,
  and the chain would mix settings.

**N = 4,000 by the rules now.** Seeds 1, 2 and 3 are learned at 8,000 (learned points 2,860, 3,900 and 4,120).
Seed 5 is learned at 6,000 (3,970), and its stage 4 (17106119) is running. Seed 4 is stopped. The size series
entry follows when seed 5 lands.
