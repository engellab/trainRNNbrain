## 2026-11-11 03:06 — Round 19: Maze read-out dropout 0.2 and the rescaling arms with freeze_after_end=true, read at update 4,000 (no arm meets the criterion; read-out dropout does not join noisy synapses at 4,000)

**Rules applied:** `2026-11-09_1513_round19-dropout-rescale-fz-rules.md`, as written. The update-2,000 read is
`2026-11-10_2014_round19-dropout-rescale-fz-results-2000.md`. Stage 2 is Della 17289218 (12 runs, updates
2,001-4,000, from `~/trainRNNbrain_r19fz` at fd564f2). Every continuation's saved config has
`freeze_after_end: true`, `init_adam: false` and `iter_offset: 2000`. Tracking: ptolmachev/dead-relu-paper#201.

**Criterion at update 4,000.** This is the criterion the paper applies to noisy synapses (σ_w = 1.0) at 4,000:
rules `2026-11-07_1111_round19-settings-stage2-rules.md` ("Techniques"), applied in entry `2026-11-08_0915`. The
controls are ReLU N = 1,000 seeds 1-6, all learned at 4,000, with 314-404 active units and success 1.00. An arm
works when three conditions hold:
- every seed has more than 404 active units, the largest count among the six controls;
- the arm's mean success is within 0.05 of 1.00;
- no seed ends below 0.9.

An arm with a seed whose record ends before 4,000 is "incomplete" and does not meet the criterion. The verdicts
come from `round19_read.read_settings`, run by `round19_read_fz.py` (output `data/round19/fz_reruns.json`).

### Data copy

The five `_cont` folders were copied from Della on 11 Nov at 03:05 Pacific, with `rsync -a --ignore-existing`,
without the Adam state files and the PNGs. Before the copy the Mac held none of these folders. After the copy all
48 files have the same MD5 checksums as on Della (48 of 48).

### Results

Per seed at update 4,000: L marks a network learned by 4,000; then clean probe success and active units of 1,000
at the probe at update 4,000; the learned update in brackets. Update-2,000 values are repeated for comparison.

| Arm | Seed 1: 2,000 → 4,000 | Seed 2: 2,000 → 4,000 | Seed 3: 2,000 → 4,000 | Verdict at 4,000 |
|---|---|---|---|---|
| dropout0.2_beta4 | L 0.92, 428 → L 0.98, 397 (1,520) | 0.84, 456 → L 0.96, 388 (2,240) | L 0.93, 436 → stopped at 2,100 | **incomplete** (seed 3 stopped); seeds 1 and 2 are also below 404 |
| rescale | 0.19, 233 → stopped at 2,800 | 0.22, 239 → stopped at 2,900 | stopped at 500 | **no verdict** (every seed stopped) |
| rescale_mild | 0.62, 252 → 0.92, 297 | stopped at 1,100 | stopped at 1,100 | **incomplete** (seeds 2 and 3 stopped) |
| rescale_few | 0.54, 240 → 0.74, 231 | 0.81, 251 → 0.70, 200 | 0.48, 233 → 0.69, 203 | **fails** every condition (mean success 0.71) |
| rescale_tgt4 | 0.62, 370 → 0.84, 333 | 0.36, 251 → 0.41, 190 | 0.38, 375 → 0.62, 241 | **fails** every condition (mean success 0.62) |

- **Read-out dropout at rate 0.2 does not meet the criterion at 4,000.** Seed 3 stopped, so the arm is
  incomplete. The other two seeds learned and reached 4,000 with 397 and 388 active units, 7 and 16 fewer than
  the bar of 404, so the arm would fail on the count whatever seed 3 had done. Before the switch, the same arm
  also failed at 4,000 on the count (389, 400 and 410 active; success 0.91, 0.98, 0.98; entry `2026-11-08_0915`).
- No rescaling network learned the task by update 4,000. rescale_mild seed 1 ends at 0.92 but has not held 0.9
  for five probes in a row. The seven rescaling networks that reached 4,000 have 190-333 active units. All seven
  are below the bar of 404, and six are below the lowest control (314); rescale_tgt4 seed 1 (333) is the
  exception.
- **Cross-check against the training logs.** At update 4,000 the log's probe success and active count equal the
  reader's for all nine networks that reached it.

**Stopped networks.** All three stage-2 stops came from the rollback guard (six non-finite rollouts in a row).
Every non-finite rollout of stage 2 was within a running episode; none was non-finite only after an episode's end.
- *dropout0.2_beta4 seed 3* (17289218_21) ran without a restore for 56 updates. Its first non-finite rollout was
  at update 2,057, seven updates after the 50-update warm-up ramp of the restarted optimiser (`INIT_ADAM=false`)
  ended. It stopped at update 2,100, its last probes at 0 active units: 41 restore events, 158 discarded
  rollouts.
- *rescale seeds 1 and 2* stopped at 2,800 and 2,900, with 20 and 6 restore events. Seed 1 had 0 active units and
  200 units growing at its last probes.
- Restore events in the networks that reached 4,000: dropout0.2_beta4 seeds 1 and 2 none; rescale_mild seed 1
  none; rescale_few 1, 22 and 0; rescale_tgt4 3, 6 and 5.

### What changes in the paper

- **The update-4,000 sentence on noisy synapses stands as written.** The paper states that at 4,000 noisy synapses
  at σ_w = 1.0 meet the criterion on Maze, and therefore on all three tasks under the one-value rule. Read-out
  dropout does not join it: its only Maze arm read at 4,000, rate 0.2, is incomplete, and its two complete seeds
  are below the bar. Rate 0.05 was not continued, because all three seeds had learned by 2,000 (continuation
  rule of the rules entry). No rescaling arm meets the criterion at 4,000.
- The update-2,000 verdicts and the headline are as in entry `2026-11-10_2014`. The 4,000 read changes none of
  them.

**Cost.** Stage 2 used 65 GPU-h (`sacct` elapsed time summed over the 12 tasks; three stopped early). Package
total: 117 + 65 = 183 GPU-h, within the 200 GPU-h limit.
