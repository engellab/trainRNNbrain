## 2026-11-08 09:25 — Round 19: does restarting Adam at a stage boundary cause the stage-2 instability? (rules, before submission)

**Why.** Every continuation in this study restarts Adam's moments (`INIT_ADAM=false`). Among the 63 settings
networks continued to 4,000 updates (`2026-11-08_0915_round19-results.md`), 24 had a rollback-guard restore
in stage 2 (updates 2,001-4,000), against 12 in stage 1. The guard restores the last finite weights when a
rollout is not finite. If the restart causes these events, the paper must say so, and the stage protocol
should restore the optimizer state wherever the state file exists. The coordinator approved this test on 8 Nov.

**What I found while preparing it.** I rebuilt rwm0.2 seed 1 from its last saved weights (update 3,000, a
few updates before the guard stopped it) and played 64 training episodes (`postep.py` in my scratch notes;
the numbers go into the results entry). Its states stay within |x| ≤ 1.27 while an episode is running. After an
episode has ended, when the network keeps stepping on the last observation and no loss term involves the
action, the states of one episode in 64 reach 6 × 10^37, close to the largest float32 number. The other
networks checked the same way had no such growth: met1 s2, met0.1 s1 and the ReLU control seed 1 at their
end weights, |x| ≤ 2.4 in both parts. So rwm0.2 seed 1 was stopped by dynamics after the end of an episode,
which do not affect its task performance (success 1.00 at update 3,000). The question for this test stays the
same: does the restart make such events more frequent?

**Which networks.**
- rwm0.2 seed 1, named by the coordinator.
- Two more, chosen by this rule before any new run. From the 63 continued networks, keep those with no guard
  restore in stage 1 and at least one in stage 2. Then remove:
  - rwm0.2, already tested;
  - the metabolic-penalty arms, whose loss includes a term over post-episode steps that can overflow by
    itself, the cause of met0.01 seed 1's freeze (entry to follow);
  - dropout and rescaling, whose internal state is lost at a restart even with `INIT_ADAM=true`;
  - networks whose stage-1 folder has no Adam state file;
  - noisy synapses, whose 28 s per update would take the test above its 18 GPU-h.
  Of what is left, take the two with the earliest first stage-2 restore, one per arm.

  Applied to the existing records: **srec0.1 seed 3** (first stage-2 restore at update 2,698) and **srad1.5
  seed 3** (3,920). Removed: met1 s2, met0.1 s1, met10 s2 and s3, met0.01 s2, met1 s1 (metabolic penalty);
  rwm0.2 s3; rescale_few s1 and s2, dropout0.2 s1 and s3 (internal state); sigmaw1.0 s1 and s2 (speed). All
  three chosen stage-1 folders hold an AdamState file on Spock.

**Runs.** Each network's stage 2 (updates 2,001-4,000) is run again from the same stage-1 weights, with
`INIT_ADAM=true`: Adam's moments are restored from stage 1, so there is no warm-up ramp. The same launcher
and pinned code as the original stage 2 (`rl_maze_interventions_spock.slurm` from `~/trainRNNbrain_r19s`,
7889823), indices 79, 63 and 69, with `SAVE_TO=~/trainRNNbrain/data/trained_RNNs_rl_maze_r19adam`. The test
runs therefore sit in their own tree and never join the main tree's chains. Spock, about 3 × 6 = 18 GPU-h.

**Outcome, set now.** Per network: the number of guard restores in updates 2,001-4,000, whether the run is
stopped, and the first non-finite update. The original stage-2 runs had:
- rwm0.2 s1: stopped at 3,001, first non-finite update 2,334;
- srec0.1 s3: restores from 2,698;
- srad1.5 s3: restores from 3,920.

**Decision rule:**
- **The restart causes the instability**, if all three `INIT_ADAM=true` runs complete updates 2,001-4,000
  with no guard restore. I will then propose: continuation stages restore the optimizer state wherever the
  file exists; the paper states that the earlier continuations restarted Adam and that the restart raised the
  rate of non-finite rollouts; the networks stopped in stage 2 with a state file are rerun that way.
- **The restart does not cause it**, if two or three of the runs have a guard restore or stop. The protocol
  keeps `INIT_ADAM=false`, and the stage-2 instability is reported as a property of the longer training.
- **No conclusion**, if exactly one run has a restore. That is reported as such.

**Limitation, stated now.** The three networks were chosen because they had a restore, and each comparison is
one run against one run. Rollouts are stochastic, so a restore-free rerun can happen by chance, and regression
toward the mean favours that. A same-seed replicate with `INIT_ADAM=false` would measure that chance and cost
another 18 GPU-h; it is offered to the coordinator, not run.

**Secondary read.** On each test run's final weights, and on the original stage-2 weights where they are finite:
the largest |state| while episodes run and after they end (the measurement above).
