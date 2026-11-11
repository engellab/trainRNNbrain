## 2026-11-10 20:14 — Round 19: Maze read-out dropout and rescaling with freeze_after_end=true, read at update 2,000 (both dropout arms fail, every rescaling arm fails or is incomplete; the headline stands, and the caveat is removed for both techniques)

**Rules applied:** `2026-11-09_1513_round19-dropout-rescale-fz-rules.md`, as written. Jobs: entry
`2026-11-09_1516_round19-dropout-rescale-fz-submitted.md` (Della 17207738 and 17207739, `~/trainRNNbrain_r19fz` at
fd564f2, `freeze_after_end=true`). Tracking: ptolmachev/dead-relu-paper#201.

**Criterion at update 2,000** (issue #5 as written, decision #194): every seed has more than 449 active units (the
largest of the three uniform-series ReLU N = 1,000 controls: 426, 449, 415), the arm's mean success is no more than
0.05 below the controls' 0.898, and no seed ends below 0.9. An arm with a seed whose record ends before 2,000 is
"incomplete" and does not meet the criterion. Learned: `common.maze_learned_index` (clean probe success at or above
0.9 for five probes in a row, the last also at or above 0.9).

### Data copy

The six arm folders were copied from Della (`~/trainRNNbrain/data/trained_RNNs_rl_maze_fz/Maze_interventions/`) on
10 Nov at 20:12 Pacific, with `rsync -a --ignore-existing`, without the Adam state files and the PNGs. Before the
copy the Mac held no folder of these arms, so no stale file could be kept. After the copy all 72 files have the
same MD5 checksums as on Della (72 of 72; Della holds 72 such files).

### Results

Read with `trainRNNbrain/experiments_and_analysis/round19_read_fz.py` (unchanged; output
`data/round19/fz_reruns.json`). Per seed: L marks a network learned by update 2,000; then clean probe success and
active units of 1,000 at the probe at update 2,000; the learned update and the active units there in brackets.

| Arm | Seed 1 | Seed 2 | Seed 3 | Mean success | Verdict at 2,000 |
|---|---|---|---|---|---|
| dropout0.05_beta4 | L 1.00, 444 (1,200; 474) | L 0.98, 416 (1,490; 433) | L 0.97, 498 (980; 542) | 0.98 | **fails**: seeds 1 and 2 not above 449 |
| dropout0.2_beta4 | L 0.92, 428 (1,520; 460) | 0.84, 456 | L 0.93, 436 (1,090; 497) | 0.898 | **fails**: seeds 1 and 3 not above 449; seed 2 ends at 0.84 |
| rescale | 0.19, 233 | 0.22, 239 | stopped at update 500 | — | **incomplete** (seed 3 stopped); seeds 1 and 2 also fail every condition |
| rescale_mild | 0.62, 252 | stopped at 1,100 | stopped at 1,100 | — | **incomplete** (seeds 2 and 3 stopped); seed 1 fails every condition |
| rescale_few | 0.54, 240 | 0.81, 251 | 0.48, 233 | 0.61 | **fails** every condition (0.29 below 0.898) |
| rescale_tgt4 | 0.62, 370 | 0.36, 251 | 0.38, 375 | 0.45 | **fails** every condition (0.45 below 0.898) |

- Read-out dropout meets both success conditions at rate 0.05 (mean 0.98, every seed at or above 0.97) and fails on
  the count alone: seed 1 has 5 units fewer than the bar of 449, seed 2 has 33 fewer. At rate 0.2 the arm's mean
  success equals the controls' (0.898), and the arm fails on the count in two seeds and on seed 2's success.
- No rescaling network learned the task by update 2,000. Every rescaling network that reached 2,000 has 233-375
  active units, fewer than every control (415-449).
- **Cross-check against the training logs.** At update 2,000 the log's probe success and active count equal the
  reader's for all 15 networks that reached it.

**Stopped networks.** The rollback guard stops a run after six non-finite rollouts in a row. rescale seed 3 stopped
at update 500 (Della 17207739_24); rescale_mild seeds 2 and 3 stopped at 1,100 (_26, _27). Each had 0 of 1,000
units active and 200 units growing (the rule's cap of 0.2 N) at its last probe, with probe success 0.00. Every
discarded rollout of every network in this entry was non-finite within a running episode; none was non-finite
only after an episode's end. Restore events at stage 1: rescale 17, 13 and 45; rescale_mild 46, 60 and 61;
rescale_few 0, 1 and 1; rescale_tgt4 5, 5 and 9; dropout0.2_beta4 1, 0 and 0; dropout0.05_beta4 none.

### Compared with the runs before the switch

| Arm | Before the switch: active units; success (verdict) | With freeze_after_end |
|---|---|---|
| dropout0.05_beta4 | 455, 373, 420; 0.93, 0.96, 1.00 (fails on the count) | 444, 416, 498; 1.00, 0.98, 0.97 (fails on the count) |
| dropout0.2_beta4 | 413, 461, 436; 0.84, 0.79, 0.76, none learned (fails) | 428, 456, 436; 0.92, 0.84, 0.93, two learned (fails) |
| rescale | all three stopped at updates 52-60 (no read) | two reach 2,000 at 0.19 and 0.22; one stopped at 500 (incomplete) |
| rescale_mild | one stopped at 800; 314 and 321 at 0.75 and 0.63 (incomplete) | 252 at 0.62; two stopped at 1,100 (incomplete) |
| rescale_few | 255, 231, 234; 0.66, 0.75, 0.81 (fails) | 240, 251, 233; 0.54, 0.81, 0.48 (fails) |
| rescale_tgt4 | 352 at 0.81; one stopped at 800; 197 at 0.01 (no verdict) | 370, 251, 375; 0.62, 0.36, 0.38 (fails) |

Every verdict is the same as before the switch, or moves from no read to "fails" or "incomplete". As the rules
entry says, each difference is the effect of running with the switch on; no mechanism is inferred from it.

### What changes in the paper

- **Read-out dropout does not meet the criterion on Maze** at either value with the switch on. Under the one-value
  rule it meets the criterion on 3BFF and CDDM only, and under matched values on CDDM only. Both Results 5
  sentences stand.
- **Rescaling does not meet the criterion on Maze** at any of the four values. "Rescaling meets it in 3BFF networks
  only" (one-value rule) and "rescaling in none" (matched values) stand.
- **The headline stands under both rules:** prune-and-duplicate is the only technique that meets the criterion on
  all three tasks at the tolerance of 0.01. The one-value tolerance range (0.0032-0.0161) and the matched-values
  range (0.0046-0.021) are unchanged, because no technique gains a Maze pass.
- **The caveat of entry `2026-11-08_0953` ("Known caveat, not rerun now") is removed for read-out dropout and
  rescaling.** Their Maze verdicts now come from runs with the switch on. The caveat remains for re-initialisation
  (redraw, zeroout), which was not rerun (rules entry: not one of the six techniques of the cross-task comparison,
  and it fails on both supervised tasks).

### Stage 2 (submitted under the rules entry's continuation rule)

An arm with any non-stopped seed not learned at 2,000 continues one stage, to 4,000, with every non-stopped seed,
`INIT_ADAM=false`, `FREEZE=true`. The rule selects:
- dropout0.2_beta4 seeds 1-3 (indices 19-21; seed 2 not learned);
- rescale seeds 1 and 2 (22, 23; seed 3 stopped);
- rescale_mild seed 1 (25; seeds 2 and 3 stopped);
- rescale_few seeds 1-3 (28-30);
- rescale_tgt4 seeds 1-3 (124-126).

dropout0.05_beta4 does not continue: all three seeds learned by 2,000. That makes 12 runs, as **Della 17289218**
(`R19fzC2`), submitted 10 Nov at 20:13 Pacific with the stage-2 command of the rules entry and
`--array=19-23,25,28-30,124-126 --time=12:00:00`. The 4,000 read is reported in a later entry. It changes no verdict
in the paper, because the stated read is at update 2,000.

**Cost.** Stage 1 used 117 GPU-h (`sacct` elapsed time summed over the 18 tasks: dropout runs 6.4-8.4 h, rescaling
2.2-9.0 h), more than the rules entry's estimate of about 100. Stage 2: 12 runs of about 7.5 h, about 90 GPU-h.
The package total is then about 207 GPU-h, 4% over the 200 GPU-h estimate.
