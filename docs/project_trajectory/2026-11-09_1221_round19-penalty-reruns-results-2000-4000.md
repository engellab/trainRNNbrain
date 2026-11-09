## 2026-11-09 12:21 — Round 19: the freeze_after_end=true penalty reruns read at updates 2,000 and 4,000 (rate penalty 0.3 works at 4,000; rate + sparsity fails at both points; metabolic penalty)

**Rules applied.** `2026-11-08_0940_round19-penalty-reruns-rules.md` (which arms, the criteria),
`2026-11-08_1758_round19-penalty-reruns-stage2-rules.md` (one stage, report both points; the 2,000 read stays
the stated one) and, for the 4,000 read, `2026-11-07_1111_round19-settings-stage2-rules.md`. Tracking:
ptolmachev/dead-relu-paper#201. Every run here has `trainer.freeze_after_end=true`: an ended episode's state is
held, and the rate and metabolic penalties are averaged over the steps of running episodes only.

**Criteria.**
- *Learned*: `common.maze_learned_index`, clean probe success at or above 0.9 for five probes in a row, with the
  last probe also at or above 0.9.
- *Techniques* (the rate penalty, alone and with the sparsity penalty), issue #5's criterion (decision #194):
  every seed's active count above every control's, the arm's mean success no more than 0.05 below the
  controls' mean, and no seed below 0.9. At 2,000 the controls are the three uniform-series ReLU N = 1,000
  networks (bar 449 active, mean success 0.898). At 4,000 they are ReLU N = 1,000 seeds 1-6 (bar 404, mean
  success 1.00).
- *Settings* (the metabolic penalty), the direction rule on learned networks only: every learned seed above the
  learned controls' range "increases", every one below it "decreases", otherwise "no change", none learned "no
  direction". The range is 394-449 at 2,000 (five learned controls) and 314-404 at 4,000 (all six learned).
- An arm with a seed whose record ends before the read point has no verdict there ("incomplete"), as
  `round19_read.py` reads every stopped seed at 4,000.

### Data copy

The six arms were copied from Della (`~/trainRNNbrain/data/trained_RNNs_rl_maze_fz/Maze_interventions/`) on
9 Nov at 12:20 Pacific. Before writing, I compared checksums of every file present on both machines:
- Every stage-1 file on the Mac matched Della's.
- The Mac already held `frm0.1_rws0.05_cont`, `frm0.3_cont`, `met0.01_cont` and `met0.1_cont`, but these were copies
  taken while the jobs ran: their records ended at updates 2,300-3,400. This tree has only ever been trained on
  Della, so none of these files came from Spock.

A plain `rsync --ignore-existing` would have kept those partial records, and the 4,000 read would then have
used them. So the four partial folders were moved, not deleted, to
`data/trained_RNNs_rl_maze_fz_partial_copies_1109/Maze_interventions/`. After that, `rsync -a --ignore-existing`
(without the Adam state files and the PNGs) added the completed folders and `met1_cont` and `met10_cont`. All
176 files read here now have the same checksums as on Della.

### Reader

`trainRNNbrain/experiments_and_analysis/round19_read_fz.py` (new; output `data/round19/fz_reruns.json`) runs
`round19_read.read_settings` unchanged on the fz tree, with the Mac's `Maze_suite` ReLU N = 1,000 networks as
controls. The per-network numbers and the 4,000 verdicts come from that function. `round19_read.py` takes its
2,000 verdicts from the JSON files of the runs before the fix, so the new script computes the 2,000 verdicts
from the same per-network numbers, by the criteria above.
- Its success condition is one-sided, as in `fig_ed_maze_interventions.rule_verdict`. `round19_read.py` tests
  the absolute difference at 4,000. Because the controls' mean there is 1.00, the two tests give the same answer.
- As a check, the script gives "works" at 2,000 for frm0.03, frm0.1 and the prune-and-duplicate rerun, with the
  per-seed numbers of entries 2026-11-08_1758 and 2026-11-09_0136.

### Results

Per seed: clean probe success and active units of 1,000 at the last probe at or before the read point; L marks
a network learned at that point, and the learned update is in brackets.

| Arm | Update 2,000: per seed | Verdict at 2,000 | Update 4,000: per seed | Verdict at 4,000 | Label (part 2 rules) |
|---|---|---|---|---|---|
| frm0.3 | L 1.00, 1,000 (970); L 0.98, 1,000 (830); 0.87, 997 | **fails**: seed 3 ends at 0.87 (mean 0.95, every count above 449) | L 1.00, 1,000; L 1.00, 1,000; L 1.00, 1,000 (2,410) | **works**: every count above 404, mean success 1.00, no seed below 0.9 | learns within 4,000; one seed after 2,000, as one of the six controls |
| frm0.1_rws0.05 | 0.39, 1,000; 0.68, 1,000; 0.34, 1,000 | **fails**: mean success 0.47; all three below 0.9 | 0.84, 1,000; L 0.99, 1,000 (2,920); 0.66, 1,000 | **fails**: mean success 0.83 against 1.00; seeds 1 and 3 at 0.84 and 0.66 | 1 of 3 seeds learn within 4,000 |
| met0.01 | L 0.99, 503 (800); L 1.00, 449 (900); 0.88, 393 | **no change** (503 and 449; 449 is not above 394-449) | L 0.98, 437; L 1.00, 367; L 1.00, 349 (1,740) | **no change** (437 above 314-404, 367 and 349 inside) | learns within 4,000; one seed after 2,000 |
| met0.1 | 0.84, 423; 0.65, 331; L 0.95, 368 (1,290) | **decreases**, on its one learned seed (368) | L 1.00, 376 (2,070); L 0.97, 311 (2,250); L 1.00, 307 | **no change** (307 and 311 below 314, 376 inside) | learns more slowly |
| met1 | 0.94, 341; stopped at 1,900; 0.87, 360 | **no direction** (no seed learned; seed 2 stopped) | L 0.96, 339 (2,000); stopped at 1,900; stopped at 3,902 | **incomplete** (seeds 2 and 3 stopped); the one learned seed, 339, is inside 314-404 | 1 of 3 seeds learn within 4,000 |
| met10 | 0.71, 190; 0.52, 217; 0.70, 219 | **no direction** (no seed learned) | 0.69, 150; 0.77, 198; 0.75, 204 | **no direction** (no seed learned) | does not learn within 4,000 |

- met1 seed 1's learned point is update 2,000 itself. Its five-probe hold ends at update 2,040, so at 2,000 it is
  not yet learned.
- frm0.3 and frm0.1_rws0.05 keep 997-1,000 of 1,000 units active at both points in every seed. The frm0.3 verdict
  therefore turns on success alone, and so does the rate + sparsity verdict.

**Stopped networks.** The rollback guard stops a run when its rollout is not finite six times in a row.
- *met1 seed 2*: stopped at update 1,900, in stage 1 (entry 2026-11-08_2346). It had 44 restore events with
  176 discarded rollouts. Every one was non-finite within a running episode, none after an episode's end.
- *met1 seed 3*: stopped at update 3,902, in stage 2 (Della 17156817_90). Its last probe, at update 3,900, read
  0.97 success with 349 active units. Its record through 3,900 has 8 restore events with 20 discarded rollouts,
  all within running episodes. The log line at update 3,902 classifies four tries, all within running episodes.
  The two further non-finite rollouts that reached the limit of six are not classified in the log.
- No other rerun network stopped.

**Guard restores in the other networks.** None in any frm0.3, frm0.1_rws0.05 or met0.01 network. met0.1 seed 2:
6 events in stage 2. met1 seed 1: 21 events in stage 2. met10 seeds 1-3: 2, 6 and 15 events over both stages. None of these
discarded rollouts was non-finite after an episode's end.

**Cross-check against the training logs.** At the last probe, the logs give these values: frm0.3 1.00 with
1,000 active in all three seeds; frm0.1_rws0.05 0.84, 0.99 and 0.66; met0.01 437, 367 and 349; met0.1 376, 311
and 307; met1 seed 1 339 (0.96); met10 150, 198 and 204 at 0.69, 0.77 and 0.75. Every reader number equals the
log value. For met1 seed 3 the log's last probe (0.97, 349) is the reader's last probe at update 3,900.

### Compared with the runs before the fix (entry 2026-11-08_0915)

| Arm | Before the fix, 2,000 / 4,000 | With freeze_after_end, 2,000 / 4,000 |
|---|---|---|
| frm0.3 | works / not continued (all learned at 2,000) | fails / works |
| frm0.1_rws0.05 | fails / works (812-891 active, success 1.00) | fails / fails |
| met0.01 | no change / no change | no change / no change |
| met0.1 | no change / no change | decreases (one learned seed) / no change |
| met1 | decreases / no change | no direction / incomplete |
| met10 | no direction / decreases (one learned seed) | no direction / no direction |

As the rerun rules say, each difference is the effect of averaging the penalties over running steps only. No
mechanism is inferred from it.

### What changes in the claims

- **The rate penalty on Maze.** At 2,000, the stated read, it works at 0.03 and 0.1 and fails at 0.3 (one seed at
  0.87). At 4,000 the 0.3 arm works: all three seeds learned, 1,000 of 1,000 units active, success 1.00. The rate
  penalty with the sparsity penalty fails at both points. At 4,000 the cause is success: two seeds end at 0.84 and
  0.66. Before the fix, this arm worked at 4,000. The Maze entry in the cross-task technique count is the rate
  penalty at 0.03 and 0.1 at update 2,000.
- **The metabolic penalty on Maze has no direction that holds at both points.** At 2,000, 0.1 decreases the number
  of active units, on one learned seed (368 against 394-449). At 4,000 it is "no change". 0.01 is "no change" at
  both points. 10 has no learned seed at either point. 1 has no learned seed at 2,000 and is incomplete at 4,000.
  The claim "the metabolic penalty 1 decreases the count on Maze" stays withdrawn, and so does "the metabolic
  penalty 10 decreases at 4,000".
