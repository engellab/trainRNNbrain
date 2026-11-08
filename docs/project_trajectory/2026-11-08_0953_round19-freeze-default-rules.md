## 2026-11-08 09:53 — Round 19: freeze_after_end is the default for new Maze runs; replicates of the Adam test; rwm0.2 s1 with the fix; prune-and-duplicate rerun (rules, before submission)

**Decisions taken by the coordinator on 8 Nov**, after `2026-11-08_0943_round19-met0.01-frozen-cause.md`: option
B with option A, and a rerun of prune-and-duplicate on the Maze. This entry fixes what runs and how each
result is read. No job of this entry had started when it was written.

**The default (commit ae93e0a).**
- `configs/trainer/trainer_ppo.yaml` now has `freeze_after_end: true`, so every new Maze run holds an ended
  episode's state and averages the rate and metabolic penalties over running steps only.
- Existing chains keep false. The suite, interventions and round 19 launchers read a continuation's parent
  config and pass its setting; a config without the key ran false.
- The interventions launcher refuses a `FREEZE` that differs from the parent's unless `FREEZE_SWITCH=true`
  asks for it, and that request is used once below, as a test.
- Three Maze analysis scripts (`rl_maze_ablate.py`, `round18_maze.py`, `rl_maze_population_stats.py`) now pass
  the alive rows to the recompute. The recompute requires them when the switch is on; without it nothing changes.

Smoke tests from a clean worktree at ae93e0a, with `module`, `conda` and `srun` stubbed. Pass condition, set
before the diffs: only the listed keys differ, plus the stub's keys and keys that are switched off.

| Run | Compared with | freeze_after_end | Other differences |
|---|---|---|---|
| interventions 79 (rwm0.2 s1) continued, no FREEZE | its parent | false (the parent's) | tag `_cont`, init_from, save_to, iter_offset 0 → 2,000 |
| the same with FREEZE=true | | exits 1 | |
| the same with FREEZE=true, FREEZE_SWITCH=true | its parent | true, with a warning | as above |
| interventions 52 (duplicate s1), fresh, Della wrapper | the original duplicate s1 | true (the new default) | save_to |
| suite 29 (ReLU N = 1,000 s4) continued | its parent | false (the parent's) | tag, init_from, save_to, iter_offset, checkpoint_keep |

### A. Same-condition replicates of the Adam-restart test (about 18 GPU-h, Spock)

**Runs.** Stage 2 of rwm0.2 s1, srec0.1 s3 and srad1.5 s3 (interventions 79, 63, 69), set up exactly as the
original stage 2: `INIT_ADAM=false`, `freeze_after_end` false (the parent's), from the same stage-1 weights.
The code is ae93e0a, which classifies each guard restore as post-episode only or running-episode; with the
switch off its dynamics equal those of the original code (7889823). They go into their own tree,
`SAVE_TO=~/trainRNNbrain/data/trained_RNNs_rl_maze_r19rep`.

**Reading, with the Adam test (Spock 7402515, `INIT_ADAM=true`).** "Restore" means at least one guard restore
in updates 2,001-4,000, or a stop.
- **The restart causes the restores:** the replicates restore in at least 2 of 3 networks AND the INIT_ADAM=true
  runs restore in none.
- **The restart is not the cause:** the INIT_ADAM=true runs restore in at least 2 of 3 networks.
- **Open:** anything else. That includes the replicates restoring in fewer than 2 of 3, which would mean the
  original restores were too rare to compare. If open, the 70 GPU-h version is run: 9 more stage-2 networks
  with restores, replicated the same way. It needs no new decision.

**The post-episode share.** Over all restores in the replicates, the share classed post-episode only.
- Most post-episode: the stage-2 instability of these networks is post-episode growth, which the new default
  removes. INIT_ADAM then matters only for chains that keep the switch off.
- Most running-episode: the instability is in the task dynamics, and the INIT_ADAM reading above decides.

### rwm0.2 s1, stage 2 with freeze_after_end=true (about 6 GPU-h, Spock)

The same stage-2 continuation, with `FREEZE=true FREEZE_SWITCH=true` and `INIT_ADAM=false`, into
`~/trainRNNbrain/data/trained_RNNs_rl_maze_r19fzsw`. The original stopped at update 3,001 from post-episode
growth (6e37 after the end, at most 1.27 while running).
- **The fix removes its failure:** it completes updates 2,001-4,000 with no stop.
- A stop here would be classified by the new record. A stop within running episodes means the fix did not
  address this network's failure.

### Prune-and-duplicate on the Maze, rerun with freeze_after_end=true (about 16 GPU-h, Della)

**Why.** The headline is that prune-and-duplicate is the one technique that meets the criterion on all three
tasks. On Maze its rule decided which units to duplicate from participation statistics that included
post-episode steps.

**Runs.** Arm `duplicate`, seeds 1-3, interventions indices 52-54, fresh, stage 1 (2,000 updates). The arm
keys are as before, plus `freeze_after_end=true`, the new default, also passed explicitly with `FREEZE=true`.
Della, `SAVE_TO=~/trainRNNbrain/data/trained_RNNs_rl_maze_fz`, the penalty reruns' tree. The arms differ, so
the reader keeps them apart.

**Read-out, the same criterion and controls as the existing verdict.** At update 2,000, issue #5's criterion
against the three uniform-series controls (`read_maze_arms.py`, which gave "works" with 747, 741 and 712 active
and success 0.98, 0.95 and 0.99):
- every seed's active count is above every control's (449);
- the arm's mean success is within 0.05 of the controls' 0.898;
- no seed ends below 0.9.

**What follows.**
- **Passes:** the headline stands as written.
- **Fails:** the headline is wrong on Maze, and the text has to say so. "Prune-and-duplicate is the one technique
  that meets the criterion on all three tasks" is then false, and under the round 18 rules no technique meets it
  on all three tasks. The status board is updated either way.

**Known caveat, not rerun now.** Read-out dropout (dropout0.2_beta4, dropout0.05_beta4), rescaling (rescale,
rescale_mild, rescale_few, rescale_tgt4) and re-initialisation (redraw, zeroout) also drive their rules from
statistics that included post-episode steps: the participation average, the silent check and the graduation
statistic. All of them fail on Maze. A rerun with the switch on could change those verdicts, but none is run now.
Each is listed with this caveat wherever its Maze verdict appears.

**Cost of this entry:** about 18 + 6 + 16 = 40 GPU-h.
