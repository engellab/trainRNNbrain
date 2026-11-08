## 2026-11-08 09:40 — Round 19: the 24 Maze rate-penalty and metabolic-penalty networks rerun with trainer.freeze_after_end=true (rules, before submission)

**Why.** In every Maze run before 8 Nov the RNN kept stepping on the last observation after the agent reached
the cheese, in the rollout and in the PPO recompute. The rate penalty (`lambda_frm`) and the metabolic penalty
(`lambda_met`) averaged over those post-episode steps as well. The module states that every loss is a mean
over the steps of running episodes; the policy loss is. How large the post-episode share was, from each run's
mean episode length (Progress `steps` against T = 500):
- 20-56% of the RNN steps over the last 200 updates;
- 9-42% averaged over training.

The share is per arm: frm0.03 0.45-0.55, frm0.1 0.22-0.53, frm0.3 0.50-0.56, frm0.1_rws0.05 0.20-0.44,
met0.01 0.42-0.51, met0.1 0.51-0.53, met1 0.40-0.44 and met10 0.12-0.38. This mismatch is also what froze
met0.01 seed 1. Its post-episode states grew to about 1e21, the metabolic penalty overflowed and every
optimiser step from update 1,501 on was skipped (cause entry to follow, with the scan). The fix,
`trainer.freeze_after_end` (commits d2d0b3f and e8ae647), holds an ended episode's state at its end value and
averages the two penalties over the steps before the end only. Its default stays false until the coordinator
decides.

**Verdicts withdrawn until the reruns land.** All Maze verdicts of these 8 arms, at update 2,000 and at 4,000:
- the rate penalty as a technique: frm0.03 and frm0.3 "work", frm0.1 "fails", frm0.1 + rws0.05 "fails" at
  2,000 and "works" at 4,000;
- the metabolic penalty as a setting: met0.01 and met0.1 "no change"; met1 "decreases" at 2,000 and "no change"
  at 4,000; met10 "no direction" at 2,000 and "decreases" on one seed at 4,000.
Every statement built on them (the status board's technique and settings counts across tasks) is withdrawn
for its Maze part. Arms without either penalty keep their verdicts: with no penalty, post-episode steps never
enter the loss, so the switch changes only whether non-finite values can arise there.

**Runs.** The 8 arms, seeds 1-3, with every arm key as before plus `trainer.freeze_after_end=true`:
- frm0.03, frm0.1, frm0.3, frm0.1_rws0.05: launcher indices 4-15;
- met0.01, met0.1, met1, met10: indices 82-93.

They run on Della through `slurm/rl_maze_interventions_della.slurm` with `FREEZE=true` and
`SAVE_TO=~/trainRNNbrain/data/trained_RNNs_rl_maze_fz`, in their own tree, because the readers derive the
arm from the config and would otherwise mix old and new runs. The worktree is clean and pinned to the commit of
this entry (it contains d2d0b3f and e8ae647). Stage 1 is 2,000 updates. After it, the stage protocol of
`2026-11-07_1008_round19-train-all-rules.md` applies:
- a network stops at the end of the first stage at which it is learned, cap 8,000;
- every seed of an arm shares one budget (the part 2 choice);
- continuations keep `FREEZE=true` (the launcher refuses a mismatch with the parent).

Cost: stage 1 is 24 × 5.3 = 127 GPU-h on Della (about 139 at Spock's speed). Later stages are reported with
their cost before submission.

**Read-out: the same criterion and controls as the withdrawn verdicts.**
- *At update 2,000, techniques* (frm arms): issue #5's criterion against the three uniform-series controls,
  as `read_maze_arms.py` reads it. Every seed's active count at the end exceeds every control's (449). The
  arm's mean success is within 0.05 of the controls' mean (0.898). No seed ends below 0.9.
- *At update 2,000, settings* (met arms): the direction rule against the five learned controls (394-449),
  as `round18_training_read.py settings` reads it.
- *At update 4,000*, for arms trained that far: the part 2 rules
  (`2026-11-07_1111_round19-settings-stage2-rules.md`), with ReLU N = 1,000 seeds 1-6 at 4,000 as controls.
- Also reported per network: learned or not, the learned update, active units at the learned probe and at
  the end, success at the end, and the guard restores (`restored_post_episode` is recorded from e8ae647 on).
- The controls are not rerun, for the reason above.

**What the current Maze rate-penalty verdict becomes.** The paper's Maze statement is "the rate penalty
works at 0.03 and 0.3", and the cross-task count has the rate penalty passing on CDDM (0.1, 0.2) and Maze.
Both are replaced by the rerun verdicts at update 2,000, value by value:
- both 0.03 and 0.3 pass: the statement stands as it is, now on runs whose penalty is taken over the
  running episodes only;
- one of them passes: "works at <that value>";
- neither passes: the rate penalty does not meet the criterion on Maze, and the cross-task count loses its
  Maze entry (CDDM at 0.1 and 0.2 remains, and 3BFF at 0.02 under the one-value rule, as round 18 stated);
- frm0.1 or frm0.1 + rws0.05 passes at 2,000: that value is added.
The metabolic verdicts follow the same way. A difference between the old and the new runs is reported as the
effect of excluding post-episode steps from the penalty. No mechanism is inferred from it.
