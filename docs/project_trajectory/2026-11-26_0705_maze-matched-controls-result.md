## 2026-11-26 07:05 — Matched Maze controls: the entropy term's Maze verdict holds at both sizes, and no technique's verdict changes

Read under the rules `2026-11-25_1237_rules-maze-matched-controls.md`, committed and pushed at 1fa9d54 before any
job. Submission: `2026-11-25_1312_maze-matched-controls-submitted.md`. Branch `wp/maze-matched-controls`.

**Verdict: upheld.** With the matched controls in place of the original ones, the entropy set point at λ_ent = 0.1
still passes M09's Maze rule at N = 1,000 (update 2,000) and at N = 2,000 (update 4,000). The rule allows a tolerance
of 0.05 on mean probe success and sets a floor of 0.9. At both sizes the smallest `ent0.1` count is more than 200 units
above the largest matched control's. The overturn condition was a matched control with at least 634 active units at
N = 1,000 or at least 718 at N = 2,000. The largest matched counts are 432 and 452.

### Why the two settings differed

The entropy networks (entry `2026-11-22_1808`) were compared with default Maze networks that already existed. Reusing
them saved about 80 GPU-hours. Those default networks were trained before 8 Nov, and the trainer changed after that
date in two ways, both listed in the rules `2026-11-21_0812`:
- *`freeze_after_end`.* The newer trainer holds an episode's state after the episode ends. The old default networks
  behave as if the option were false. A 3-update test had shown the two settings give equal weights, so the
  difference was judged harmless; it was never tested over 2,000 updates.
- *Continuation at N = 2,000.* The entropy networks kept their Adam (the optimiser) state at update 2,000. The old
  default networks restarted Adam and repeated the learning-rate warm-up.

Two more differences were found when the rules for this test were written. The old default networks ran from other
code commits, and on other cards. The matched controls remove all four differences.

### Verdicts side by side

| N, read point | `ent0.1` active units | `ent0.1` success (mean) | Controls | Control active units | Control success (mean) | Bar on mean success | Verdict |
|---|---|---|---|---|---|---|---|
| 1,000, update 2,000 | 634, 689, 661 | 0.94, 1.00, 1.00 (0.98) | original | 426, 449, 415 | 0.99, 0.98, 0.73 (0.90) | 0.85 | passes |
| | | | matched | 401, 432, 392 | 0.91, 0.99, 0.91 (0.94) | 0.89 | passes |
| 2,000, update 4,000 | 718, 730, 905 | 0.99, 1.00, 0.99 (0.995) | original | 400, 506, 388 | 0.98, 0.98, 1.00 (0.99) | 0.94 | passes |
| | | | matched | 376, 450, 452 | 0.90, 0.89, 1.00 (0.93) | 0.88 | passes |

"Bar on mean success" is the control mean minus 0.05; the `ent0.1` mean must reach it. Every `ent0.1` network ends at
or above 0.9, so the floor holds. All six matched controls finished, and all count. At N = 2,000, seeds 1 and 2 end at
0.898 and 0.891, below the 0.9 level at which a run counts as having learned. Under the rules, an unlearned control
still counts as it is: those two seeds enter the mean, and their counts enter the count condition.

### Descriptive only: the other techniques against the matched N = 1,000 controls

No verdict changes. The reader compares every arm with both control sets and marks any arm whose verdict differs; it
marked none. The matched controls' largest count is 432 against the original 449, and their mean success is 0.935
against 0.898. Both changes make the count condition easier to pass and the success condition harder, by 17 units and
0.037.

| Arm (N = 1,000) | Active units | Success | Against original | Against matched |
|---|---|---|---|---|
| prune-and-duplicate | 763, 749, 757 | 1.00, 1.00, 0.98 | passes | passes |
| rate penalty 0.03 | 998, 997, 952 | 1.00, 0.97, 1.00 | passes | passes |
| rate penalty 0.1 | 998, 1000, 997 | 1.00, 0.99, 1.00 | passes | passes |
| rate penalty 0.3 | 1000, 1000, 997 | 1.00, 0.98, 0.87 | fails (floor) | fails (floor) |
| rate penalty 0.1 + sparsity 0.05 (`frm0.1_rws0.05`) | 1000, 1000, 1000 | 0.39, 0.68, 0.34 | fails (success, floor) | fails (success, floor) |
| read-out dropout 0.05 | 444, 416, 498 | 1.00, 0.98, 0.97 | fails (count) | fails (count) |
| read-out dropout 0.2 | 428, 456, 436 | 0.92, 0.84, 0.93 | fails (count, floor) | fails (count, floor) |
| rescaling, 4 variants | 233-375 | 0.19-0.81 | all fail (count, success, floor) | all fail (count, success, floor) |
| `ent0.01` | 631, 667, 620 | 0.98, 0.94, 0.97 | passes | passes |
| `cor0.01` | 488, 527, 472 | 1.00, 0.98, 0.98 | passes | passes |
| `cor0.1` | 625, 608, 561 | 0.88, 0.92, 0.92 | fails (floor) | fails (floor) |

Read-out dropout at 0.05 fails on the count against both control sets: its smallest count, 416, is below 432 and 449. Two
rescaling variants have runs that did not finish (`rescale` 2 of 3, `rescale_mild` 1 of 3); the rule is applied to the
finished ones, as M09 does. At N = 2,000, `cor0.01` fails against both control sets. Against the original controls it
fails on all three conditions; against the matched controls it fails on the count and the floor only.

### Matched against original controls, per seed

| N | Seed | Active units at read point (orig. / matched) | Success at read point | Learned at update | Skipped minibatch steps | Restored rollouts |
|---|---|---|---|---|---|---|
| 1,000 | 1 | 426 / 401 | 0.99 / 0.91 | 1,050 / 1,270 | 0 / 10 | 0 / 5 |
| 1,000 | 2 | 449 / 432 | 0.98 / 0.99 | 760 / 810 | 0 / 0 | 0 / 0 |
| 1,000 | 3 | 415 / 392 | 0.73 / 0.91 | never / 1,330 | 0 / 0 | 0 / 0 |
| 2,000 | 1 | 400 / 376 | 0.98 / 0.90 | 3,000 / never | 0 / 38 | 0 / 21 |
| 2,000 | 2 | 506 / 450 | 0.98 / 0.89 | 2,480 / never | 29 / 90 | 18 / 63 |
| 2,000 | 3 | 388 / 452 | 1.00 / 1.00 | 2,840 / 2,140 | 10 / 10 | 5 / 5 |

"Learned" is the first probe from which success stays at or above 0.9 for 5 probes in a row (`common.maze_learned_index`).
A skipped minibatch step is a PPO minibatch the trainer drops; a restored rollout is one discarded with the weights
rolled back (`TrainerPPO.safe_rollout`).

At N = 2,000 and update 2,000, the end of the first stage, the counts are 464 / 449, 556 / 504 and 431 / 431 (original
/ matched), and probe success is 0.77 / 0.51, 0.84 / 0.85 and 0.84 / 0.84. The `ent0.1` networks have 757, 786 and
1,144 active units at that point.

Seed 3 at N = 2,000 gives a direct test of two of the four differences. Its matched control equals the original control
at all 201 probes of the first stage (active counts identical at every probe up to update 2,000). So for this seed,
on this card class, `freeze_after_end` and the change of code commit made no difference over 2,000 updates. The two
runs separate from update 2,001, where the matched control keeps its Adam state and the original restarted Adam. That
difference alone moves the final count from 388 (original) to 452 (matched), with success 1.00 in both. In every other
seed the two controls differ from update 1, because the cards differ.

**Against the expectation.** The rules expected counts near the original controls' and an upheld verdict. The counts
are near or below them: 392-432 against 415-449 at N = 1,000, and 376-452 against 388-506 at N = 2,000. No control
passed the warning levels set in the rules (500 at N = 1,000, 600 at N = 2,000). The matched controls are worse at the
task at N = 2,000 (mean success 0.93 against 0.99), which the rules did not anticipate.

### Runs

- Della jobs 18374320 (N = 1,000), 18374321 and 18374322 (N = 2,000, stages 1 and 2), three tasks each, all COMPLETED
  with exit code 0. No task was stopped by the rollback guard, reached its time limit or failed on its node. No network
  was rerun or dropped.
- Time per task: N = 1,000, 8.4-8.6 h on A100 40 GB cards (i14 nodes); N = 2,000 stage 1, 7.5-7.7 h, and stage 2,
  7.2-8.1 h, on 80 GB cards.
- GPU time: 25.6 + 22.7 + 22.8 = 71.1 GPU-hours, within the 120 GPU-hour cap (from `sacct`, job names `MazeIntervD`,
  started 25-26 Nov 2026).
- Read job 18376518 (`MazeMatchedRead`, CPU, 29 s) ran on Della after the training jobs and wrote
  `trained_RNNs_rl_maze_mctrl_read/{reads,verdict,oracle}_matched.*`.

### Checks

1. **Configs.** Every matched control's saved config differs from the same seed's `ent0.1` config only in
   `experiment_tag`, `paths.save_to` and `trainer.lambda_ent` (0 in the controls), plus `paths.init_from` for the three
   stage-2 runs. Pass, all nine.
2. **Same start.** The rollout of update 1 comes before any optimiser step. At N = 2,000 all three matched controls
   equal `ent0.1` bit for bit. At N = 1,000, seeds 1 and 2 equal `ent0.1` bit for bit. Seed 3 equals the stored
   `cor0.1` seed 3, which ran on the same card model (i14 nodes), and not `ent0.1` seed 3, which ran on an l02 node.
   The rules allow either of the two card-model values, so the check passes; it means the matched control and
   `ent0.1` at N = 1,000, seed 3, are matched in everything but the card model.
3. **Independent recount.** `firstmate/data/maze-matched-controls/scripts/oracle_matched.py`, written without
   `common.py`, the reader or the rule function, agrees with the reader exactly on all 63 networks (60 recounted from
   the stored participation) and on every verdict against both control sets.

### Correction to the submission entry

`2026-11-25_1312_maze-matched-controls-submitted.md` says that the continuation's 17 h limit covers a node at twice
the usual time. It does not. The rules took the usual time from the pilot, 17.7 s per update at N = 2,000, so twice
that is 19.7 h for 2,000 updates. The 17 h limit is about twice the 8.3-8.7 h of the entropy networks' own
continuations. No limit set in that entry covered twice the pilot's time: the cap forbids it, since nine such limits
sum to about 164 GPU-hours. The cap won. Every task ended at least 3.3 h before its limit, so the shortfall had no effect.

### Status board

`docs/paper_audit.md` is unchanged. The paper's claims use the original controls, and every verdict on the board is
the same against the matched controls. The entropy term is not among the paper's techniques.

### Files

- Reader: `trainRNNbrain/experiments_and_analysis/maze_matched_controls_read.py` (committed with the rules); read job
  `slurm/maze_matched_read_della.slurm`.
- Figure: `img/internal_figures/maze_matched_controls_active.{pdf,png}`: active units against PPO update for the
  original controls, the matched controls and `ent0.1`, one panel per size. Drawn with the reader's `draw` from the
  read job's JSON.
- Read output and copies of the matched trees (config, `Progress.json`, `Participation.npz`):
  `firstmate/data/maze-matched-controls/read/` and `matched_N1000/`, `matched_N2000/`. On Della:
  `~/trainRNNbrain/data/trained_RNNs_rl_maze_mctrl_N{1000,2000}/Maze_interventions/control{,_cont}/` and
  `trained_RNNs_rl_maze_mctrl_read/`.
