## 2026-11-07 11:11 — Round 19 rules, part 2: the Maze settings networks not learned at update 2,000 trained to 4,000

**Why.** Pavel's decision of 7 Nov, relayed by the coordinator on ptolmachev/dead-relu-paper#201: continue
the Maze settings networks that are not learned at update 2,000 by one stage, to 4,000 updates, and report
each setting at both points. He chose this over the full 8,000-update protocol of
`2026-11-07_1008_round19-train-all-rules.md` for these networks. This entry fixes which networks are
continued, the controls, the read-out at 4,000 and how a setting is reported. It is committed before any of
these jobs is submitted; no network of this part had trained past update 2,000 when it was written.

**Terms** (as in the part 1 entry). *Maze*: the egocentric maze task trained by PPO, N = 1,000 here.
*Update*: one PPO batch of 256 episodes and 8 optimiser steps. *Probe*: a noise-free test of 128 episodes
every 10 updates; *success* is the share that reaches the cheese. *Learned*: `common.maze_learned_index`
finds a learned point, the first probe from which success stays at or above 0.9 for five probes, in a record
whose last probe is also at or above 0.9. *Active unit*: participation p = std(r) + q0.9(|r|) on a
noise-free pass, active if p ≥ 0.05·q95(p). *Arm*: one setting or one technique at one value, 3 seeds, the
launcher `slurm/rl_maze_interventions_spock.slurm`. *Setting* and *technique* (the launcher's "rule") are
the two verdict classes of `read_maze_arms.py`: settings are weight decay, input row norm, activity noise,
spectral radius, the sparsity penalty alone, the input-weight cap, the recurrent-weight and metabolic
penalties and the optimiser; techniques are the rate penalty, rate plus sparsity, read-out dropout, noisy
synapses and the rescaling, duplication and re-initialisation rules.

### The arms at update 2,000 (inventory of the Spock tree, read 7 Nov 11:00)

The Spock folder `Maze_interventions` holds 41 arm folders and 120 runs with records. 56 of the 120 are not
learned. The 48 that Pavel's decision names are the not-learned runs of the 36 arms of issues #5 and #6:

- **42 finished all 2,000 updates and are not learned.** adamw s1; dropout0.2_beta4 s1-s3; frm0.1 s1;
  frm0.1_rws0.05 s2, s3; inorm2 s1; inorm20 s1 (success 0.00); met0.01 s1; met0.1 s2; met1 s2;
  met10 s1-s3; rescale_few s1-s3; rescale_mild s2, s3; rwm0.2 s2; rws0.05 s2, s3; sgd0.03, sgd0.1,
  sgd0.3 and sgd1 s1-s3 (12 runs, success 0.00-0.24); sigmaw1.0 s3; srad1.5 s3; srec0.1 s1-s3;
  wd1e-4 s1, s2.
- **6 were stopped by the rollback guard before update 2,000** (the recurrent dynamics stopped being finite
  and restoring the last weights did not help): inorm5 s1-s3 (at updates 500, 600, 300), inorm20 s2, s3
  (200, 400), rescale_mild s1 (800). Each has 0 active units at its last probe.

The other 8 not-learned runs belong to the four arms of the round 18 parity package (sigmaw0.1, sigmaw0.3,
dropout0.05_beta4, rescale_tgt4), whose rules (`2026-11-07_0136_round18-parity-rules.md`) read at
update 2,000. The sigmaw0.1 and sigmaw0.3 runs are still training (Spock 7397761). The paper-value rescale arm
(three folders without records: the guard stopped all three at updates 52-60) has no network to continue.

### Which networks are continued

**The 42 finished, not-learned networks are continued.** The 6 guard-stopped networks are not. Their records
end at updates 200-800 with every unit inactive. A continuation would start from the last finite weights
under the same setting, which the guard had already restored and replayed without success. They stay
reported as "stopped by the guard". Adding them would cost about 6 × 5.8 = 35 GPU-h.

**Same budget within an arm: chosen.** In 12 of the arms with a continued network, other seeds learned by
update 2,000. Continuing those seeds too gives every seed of an arm the same 4,000-update budget. Those
21 networks are: adamw s2, s3; frm0.1 s2, s3; frm0.1_rws0.05 s1; inorm2 s2, s3; met0.01 s2, s3; met0.1 s1, s3;
met1 s1, s3; rwm0.2 s1, s3; rws0.05 s1; sigmaw1.0 s1, s2; srad1.5 s1, s2; wd1e-4 s3.

Costs at the measured speeds. On Spock the speed is 10-11 s per update, 5.8 GPU-h per 2,000 updates, 6.2 for the
rescaling arms. On Della it is 9.5 s, 5.3 GPU-h. The noisy-synapse arm sigmaw1.0 takes 28 s per update, 15.6 GPU-h.

| | Networks | GPU-h |
|---|---|---|
| Only the 42 not learned | 42 | about 246 |
| Same budget within an arm (the 42 and the 21) | 63 | about 387 |

The second option is under the coordinator's limit of about 400 GPU-h, so all 63 are continued. Arms in which
all three seeds learned by update 2,000 (13 arms, 39 networks: cap0.3, duplicate, frm0.03, frm0.3, inorm0.5,
redraw, rwm0.05, srad0.5, srec0, srec0.01, wd0, wd1e-5, zeroout) stop at 2,000, as the stage protocol of
part 1 stops a network learned at the end of a stage. They have no read at update 4,000. Adding them would
cost about 39 × 5.8 = 226 GPU-h, which takes the total above 600. That is for the coordinator.

**How the stage is trained.** It is the same continuation as part 1: `paths.init_from` the update-2,000
weights, `INIT_ADAM=false` (optimiser state afresh, the 50-update warm-up ramp again, as all 16 existing
continuations), the update clock continued, so `common.maze_load_runs` joins each network's two runs into one
record of updates 1-4,000. A continuation does not carry over the internal state of two techniques:
- the read-out dropout's participation average (dropout0.2_beta4);
- the rescaling rule's open episodes and growing units (rescale_few, rescale_mild).
Both restart at update 2,001, as every warm start does (`configs/paths/paths_DELLA.yaml`). For these arms a
4,000-update record is therefore not the same as one uninterrupted 4,000-update run; this is stated next to
their numbers.

### The controls at 4,000 updates

The ReLU N = 1,000 networks of the suite, seeds 1-6, all continued to 4,000 updates: seeds 1-3 already
(round 18 G11), seeds 4-6 in Spock 7398076 (part 1). These controls are the same networks as the update-2,000
references, read on their joined records at update 4,000.

### The read-out

**At update 2,000 nothing changes.** Every verdict stands as the existing reads computed it. For the settings,
that read is `round18_training_read.py settings` (`data/round18_training/round18_settings.json`). For the
techniques it is `read_maze_arms.py`. Continuing a network does not change its record up to update 2,000.

**At update 4,000, the same rules on the 4,000-update records:**
- *Settings (direction rule, unchanged).* The reference is the range of active counts at update 4,000 of the
  controls learned at 4,000. A setting "increases the count" when every seed learned at 4,000 has more active
  units at update 4,000 than every one of those controls. It "decreases the count" when every learned seed has
  fewer. Otherwise it is "no change". It is "no direction" when no seed is learned at 4,000.
- *Techniques ("works" criterion, unchanged).* A technique works when three conditions hold at update 4,000:
  every seed has more active units than every control (all six); the arm's mean success is within 0.05 of the
  controls' mean; and no seed ends below 0.9.
- Learned at 4,000 means `common.maze_learned_index` on the record through update 4,000. A seed learned at
  update 2,000 that ends below 0.9 at 4,000 is not learned at 4,000; it is named.
- Reported per network: success and active count at 2,000 and at 4,000, learned or not at each, the learned
  update and the active count at the learned point.

**How a setting is reported when its networks learn only by 4,000.** The controls learned 5 of 6 at update
2,000 and 6 of 6 at 4,000 so far (seed 3 learned only in its second stage; seeds 4-6 are still being read at
4,000). For each arm:
- all three seeds learned at 4,000, two or three of them not at 2,000: **"learns more slowly"**;
- all three learned at 4,000, exactly one not at 2,000: "learns within 4,000 updates; one seed after
  update 2,000, as one of the six controls". With three seeds against six this is not a difference from
  the controls;
- one or two seeds learned at 4,000: "k of 3 seeds learn within 4,000 updates";
- no seed learned at 4,000: **"does not learn within 4,000 updates"**.
The learned updates are given beside each label.

**Reader change needed before the results entry.** `fig_ed_maze_interventions.load_records` counts a run as
finished when its last update reaches `trainer.n_updates`. For a continuation that number is 2,000, so a
partly copied continuation at update 2,500 would pass as finished. The test must use
`trainer.iter_offset + trainer.n_updates`, as `fig_paper_F2_size.maze_budget` does. The update-2,000 read
must also use the stage-1 folders alone. Both are fixed in the reader before any number of this part is read,
and the fix is named in the results entry.

### What can change in the paper's claims

The update-2,000 verdicts do not change. The paper's same-size read on Maze is update 2,000 (status board,
decision #185 and #194), and that stays its stated read. The update-4,000 verdicts are reported beside them.
Where the two disagree, the text says so and restricts the claim to the read point at which it holds. The claims that rest on arms with
not-learned seeds, and so can read differently at 4,000:
- **Weight decay 1e-4 "decreases the count"** rests on 1 learned seed of 3 (378 active). It is part of "only
  weight decay lowers the number of active units on all three tasks". The control range at 4,000 is lower
  than at 2,000 (seeds 1-3: 317-404 at 4,000, against 394-449 for the learned controls at 2,000), so the bar
  moves as well.
- **The metabolic penalty 1 "decreases the count"** (2 of 3 learned).
- **SGD "never learns the maze"** becomes "does not learn within 4,000 updates" or changes. **Activity noise
  0.1, the metabolic penalty 10** and the four SGD rates are "no direction" at 2,000. Any of them can gain a
  direction at 4,000 if seeds learn.
- **Techniques that fail at 2,000 on success**: dropout 0.2, rescale_few, rescale_mild, the rate penalty 0.1
  (seed 1 at 0.63) and rate plus sparsity (seeds at 0.43 and 0.95, not learned). Each can pass at 4,000 if
  its seeds learn and keep more active units than every control at 4,000. The dropout and rescaling arms
  carry the restart of their internal state (above).
- Settings verdicts with all seeds learned at 2,000 are not re-read at 4,000.

### Launcher and local checks (commit b9ceee2)

- `slurm/rl_maze_interventions_spock.slurm` gains the suite launcher's continuation mode (`PARENT_ROOT`,
  `EXTRA_UPDATES`, `INIT_ADAM`, `STAGE`; stage 2 writes `<arm>_cont`), the same refusal to overwrite a folder
  that holds a run, and a `CLUSTER` switch (spock, the default, or della) for the module lines.
  `slurm/rl_maze_interventions_della.slurm` carries Della's #SBATCH lines and runs the same file from the
  pinned checkout with `CLUSTER=della`. `paths_local` and `paths_DELLA` differ only in `save_to`, which the
  launcher sets, so the arm table exists once.
- Smoke tests from a clean detached worktree at b9ceee2. `module`, `conda` and `srun` were stubbed, and `srun`
  appended 2 updates of 4 episodes × 6 decisions. Parent folders were copied from Spock. Pass condition, set
  before the diffs: each continuation's saved config differs from its parent's only in experiment_tag
  (`_cont`), paths.init_from, paths.save_to, trainer.iter_offset (0 or absent → 2,000), the stub's keys, and
  keys added to `trainer_ppo.yaml` after the parent ran whose values are off. Those off keys are
  `checkpoint_keep` false, `lambda_met` 0, `lambda_rwm` 0 with its `rwm_args`, `optimizer` adam,
  `momentum` 0, and the `prune_args` block with `prune_reinit` false. All pass:

| Run | Result |
|---|---|
| index 106 (sgd1 s1), 39 (wd1e-4 s3), 18 (sigmaw1.0 s3), 28 (rescale_few s1) | each config differs from its parent only in the allowed keys; the arm's own keys (optimizer sgd and lr 1; weight_decay 1e-4; sigma_w 1.0; the rescaling keys) are identical; updates 2,001-2,002 |
| index 19 (dropout0.2_beta4 s1) through the Della wrapper | the same; dropout keys identical |
| index 106 again | exits 1: refuses to overwrite `sgd1_cont/s1` |
| `STAGE=1`; `INIT_ADAM` left at true with no optimizer-state file; `CLUSTER=foo` | each exits 1 with its FATAL line |
| index 106 without `PARENT_ROOT` (a fresh run) | differs from the original sgd1 s1 config only in save_to, the stub's keys and checkpoint_keep false |

### Jobs (submitted after this entry is pushed; ids in the next entry)

| Cluster | Indices of `rl_maze_interventions_*.slurm` | Networks | GPU-h |
|---|---|---|---|
| Spock | 7-9, 13-15, 19-21, 26-30, 37-39, 43-45, 49, 61-63, 67-72, 79-90, 94-96 | 45 | about 262 |
| Della | 16-18 (sigmaw1.0, `--time=24:00:00`), 91-108 (met10, the four SGD rates) | 18 | about 127 |
| **Total** | | **63** | **about 387** |

The Della parents are copied from Spock through the Mac into the same layout under
`~/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_interventions/` on Della before submission. The code is a
clean detached worktree `~/trainRNNbrain_r19s` at the commit of this entry on both clusters. The part 1
worktree `~/trainRNNbrain_r19` is left as it is, because the coordinator's armed N = 4,000 submission uses it.
