## 2026-11-07 10:08 — Round 19 rules: every Maze network is trained until it learns (2,000-update stages, cap 8,000), and a pilot for tanh and the shifted sigmoid

**Why.** Pavel, 7 Nov: "Can we figure out a way how to train ALL maze networks? Some of them failed, which
is not great for the study." This entry fixes, before any job of the package is submitted, the training
protocol for every Maze network, what the pilot for the two activations that never learn will vary and what
counts as a pass, and what happens to a network that never learns. Tracking: ptolmachev/dead-relu-paper#201.
No job of this package had started when this entry was written. The inspection numbers below come from runs
that already existed (`round19_maze_inspect.py`, output `data/round19/maze_inspect.json`).

**Terms.** *Maze*: the egocentric maze task trained by PPO (proximal policy optimisation, a
reinforcement-learning algorithm). One *update* is one batch of 256 episodes followed by 8 optimiser steps.
*Probe*: a noise-free test of 128 episodes from fixed starts after update 1 and every 10th update; *success*
is the share of probe episodes that reach the cheese. *Learned point* (`common.maze_learned_index`): the
first probe from which success stays at or above 0.9 for five probes in a row, in a run whose last probe is
also at or above 0.9; a run without one has not learned. *Stage*: 2,000 updates of training. *Continuation*:
a run that starts from the weights of the previous stage (`paths.init_from`, `slurm/rl_maze_suite_*.slurm`
with `PARENT_ROOT`); the readers (`common.maze_load_runs`) join a network's stages into one record with
absolute update numbers. *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active if
p ≥ 0.05·q95(p). *KL*: the approximate Kullback-Leibler divergence between the policy before and after one
update (`approx_kl` in each run's Progress.json), a measure of how far one update moves the policy.

### The Maze networks today (read with `common.maze_learned_index` on the local copies)

| Cell | Seeds | Trained to update | Learned? |
|---|---|---|---|
| ReLU, N = 250 | 1-4 | 4,000 (2 stages) | all 4; seed 2 learned at 2,340, in stage 2 |
| ReLU, N = 500 | 1-3 | 4,000 | all 3, by update 1,750 |
| ReLU, N = 1,000 | 1-3 | 4,000 | all 3; seed 3's learned point is 1,750, but it was at 0.73 at update 2,000 and counts as learned only on the joined record |
| ReLU, N = 1,000 (round 18 G12) | 4-6 | 2,000 | all 3 |
| ReLU, N = 2,000 | 1-3 | 4,000 | all 3, at 2,480-3,000 |
| ReLU, N = 4,000 | 1 | 4,000 (first run ended at 1,000, continuation 3,000) | learned at 2,860 |
| ReLU, N = 4,000 | 2, 3 | 4,000 (seed 3: first run ended at 1,700, continuation 2,300) | no: last probe 0.84 and 0.80; mean success over the last four 250-update blocks 0.66, 0.75, 0.76, 0.87 (seed 2) and 0.81, 0.83, 0.85, 0.82 (seed 3) |
| ReLU, N = 4,000 | 4, 5 | 2,000; stage 2 running (Della 17044667, another agent's job, not touched) | no, at update 2,000 |
| leaky ReLU, N = 1,000 | 1-3 | 2,000 | all 3 |
| softplus, N = 1,000 | 1, 2 | 2,000 | both |
| softplus, N = 1,000 | 3 | 2,000 | no: last probe 0.91, never five probes in a row at 0.9; blocks 0.81-0.90 |
| tanh, N = 1,000 | 1-3 | 2,000 | no: success 0.00, 0.38, 0.02 at update 2,000, never above 0.56 |
| shifted sigmoid, N = 1,000 | 1-3 | 2,000 | no: success 0.21, 0.15, 0.15, never above 0.28 |

### (a) The training protocol for every Maze network

**Stages.** A network is trained in stages of 2,000 updates. Stage ends fall at updates 2,000, 4,000, 6,000
and 8,000 of the joined record (the two N = 4,000 first runs that ended early, at 1,000 and 1,700, count as
stage 1 ending at those updates; their continuations ended at 4,000). Each stage continues the previous one
with the suite launcher's continuation mode, `INIT_ADAM=false`: the optimiser (Adam) starts afresh and its
50-update learning-rate warm-up runs again. I checked this against the existing chains: every one of the 16
continuations on record (N = 250 to 4,000) has `paths.init_adam: false` in its saved config, and so does the
running N = 4,000 job 17044667. A network trained the other way at stage 3 would have one stage boundary
unlike its others, so the protocol keeps `false`.

**When a network stops.** At each stage end, `common.maze_learned_index` is applied to the joined record. A
network stops at the end of the first stage at which it is learned. A network not learned at 8,000 updates
(four stages) stops there (part (c)). The end-of-training read-out is the end of the last stage trained.

**Budgets now differ between networks.** Before this round, one cell meant one budget (2,000, or 4,000 for
the round 18 size series). From now on a network's budget depends on when it learned, in steps of 2,000.

**The learned-point read-out is unaffected.** The learned point is a property of the record up to that
point: continuing a network appends records and never changes earlier ones, so a network learned at a
stage end keeps its learned point if it is trained further, as long as its last probe stays at or above
0.9. Two cases need saying. First, the learned point can lie in an earlier stage than the one at whose end
the network was found learned: ReLU N = 1,000 seed 3 was not learned at update 2,000 (last probe 0.73) and
has its learned point at 1,750 on the 4,000-update record, because the rule's five-probe hold happened
before 2,000 and the last-probe condition was met only at 4,000. Second, a learned network trained further
and ending below 0.9 is not learned by the rule; under the protocol it then trains another stage (below).
Any read at a fixed update U (the settings comparison at update 2,000; the size exponent on equal
4,000-update budgets) is also unchanged, because the record up to U is unchanged; only the set of networks
learned by then can grow.

**What changes for the existing "end" reads.** Under the equal-budget choice below:
- ReLU N = 250, 500, 1,000 (seeds 1-3) and 2,000: nothing. Every one of these networks is already at 4,000
  updates, and no network of these sizes needs more than 4,000.
- ReLU N = 1,000 seeds 4-6, leaky ReLU and softplus at N = 1,000: the end moves from update 2,000 to
  update 4,000 (later if the N = 1,000 budget rises). This changes the end counts of the activation
  comparison at N = 1,000. The weight-decay test of round 18 (G12) and every settings verdict compare at
  update 2,000 and are unchanged.
- ReLU N = 4,000: the end moves from update 4,000 to at least 6,000. The size series' end read then has
  unequal budgets across sizes (4,000 for N ≤ 2,000, at least 6,000 for N = 4,000) and is labelled so; the
  across-size comparison stays at the learned point (the 16 Oct rule); the fit on equal 4,000-update budgets
  (`round18_training_read.py exponent`) can still be computed at update 4,000, with N = 4,000 represented
  only by the networks learned by then.

**Equal budget within a size, and its cost.** Pavel prefers the same budget for every network of a size if
the cost allows. Costs at the measured speeds (Della and Spock: 9.3-10.6 s per update at N = 1,000, 5.5 GPU-h
per stage; Della gpu80: 28 s per update at N = 4,000, 15.6 GPU-h per stage):

| | Each network ends at its own stop stage | Same budget within a size |
|---|---|---|
| Needed now | N = 4,000 seeds 2, 3 to 6,000 (31); softplus seed 3 to 4,000 (5.5): 37 GPU-h | the same, plus N = 4,000 seed 1 to 6,000 (16) and the N = 1,000 networks at 2,000 updates (ReLU seeds 4-6, leaky ReLU 1-3, softplus 1-2) to 4,000 (8 × 5.5 = 44): 97 GPU-h |
| Needed after Della 17044667 ends (about 04:00 ET, 8 Nov) | N = 4,000 seeds 4, 5 to 6,000 only if not learned at 4,000 (up to 31) | seeds 4, 5 to 6,000 whatever their state (31): the N = 4,000 budget is at least 6,000 because seeds 2 and 3 are not learned at 4,000 |
| If a network is not learned at 6,000 (N = 4,000) | that network only (15.6 each) | all five N = 4,000 networks to 8,000 (78) |
| If softplus seed 3 is not learned at 4,000 | that network only (5.5) | all 12 N = 1,000 networks to 6,000 (66) |
| If softplus seed 3 is still not learned at 6,000 | that network only (5.5) | all 12 N = 1,000 networks to 8,000 (66) |
| Worst case: every network not learned now needs the cap of 8,000 | about 140 GPU-h | about 340 GPU-h |

**Decision: the same budget within a size.** The cost now is 97 GPU-h, inside the package with the pilot
(below). The worst case comes mostly from softplus seed 3: if it needs the cap, all 12 N = 1,000
networks train two more stages; any submission that takes the package above about 200 GPU-h is reported to
the coordinator first. The groups that share a budget are the ReLU size series at each N, and at N = 1,000 every suite
network (ReLU seeds 1-6, leaky ReLU, softplus), because the activation comparison is made at N = 1,000
against the ReLU networks. The budget of a group is the largest stop stage among its networks, and never
less than what every network of the group has already been trained (4,000 at N = 250 to 2,000). A network
continued to its group's budget that ends below 0.9 there is not learned by the rule, so it trains another
stage, and its group's budget rises with it, up to 8,000. Pilot runs belong to no group: they are a screen,
read at 2,000 updates and not used in any comparison of the paper.

**Not covered by this package.**
- *tanh and shifted-sigmoid networks at the standard settings* are not continued. Two of the three tanh
  networks have every unit pinned at ±1 at their end weights (990 and 998 of 1,000 units; definition below),
  where the slope of tanh is under a tenth of its maximum, so a further stage would change them little; the
  other tanh network and the three sigmoid networks reached 0.15-0.38 success after 2,000 updates, against
  learned points of 760-1,750 for ReLU at the same size. Part (b) tests settings instead. They stay reported
  as not learned under the standard settings.
- *The settings arms* (`Maze_interventions`) are outside this package. In the local copies 48 of 108 arm
  networks are not learned at update 2,000 (all three SGD rates, dropout, two of the input-norm values, the
  rescaling arms, metabolic penalty 10 and others). Their verdicts compare each arm with the controls at
  update 2,000, and that comparison does not change if the controls are trained further. Bringing the arms
  under this protocol would change the read point of the settings rules (decisions #185 and #194) and cost
  about 48 × 5.5 = 264 GPU-h per stage; that is a decision for the coordinator, not part of this package.

### (b) The tanh and shifted-sigmoid pilot

**Did the supervised tasks use the same learning rate for tanh as for ReLU? Yes.** The saved configs of the
supervised tanh networks, `CDDM_std_g0_tanh/EqType=h_N=1000_act=tanh_iters=150000` (3 seeds, launcher
`slurm/SilentReLU_std_tanh_della.slurm`) and `NBitFlipFlop_std_g0_activations_bias0/EqType=h_N=1000_act=tanh_iters=75000`
(3 seeds, `slurm/b0_act.slurm` index 10-12), record `trainer.lr: 0.001`, `trainer.lr_scale_exp: 1/3` and
`model.spectral_rad: 1.2`, the same as their ReLU controls in the same folders. Neither launcher overrides the
rate. The shifted-sigmoid 3BFF networks record the same values. The Maze runs use the same rule
(`configs/trainer/trainer_ppo.yaml`, lr 0.001 × (100/N)^(1/3), 4.6e-4 at N = 1,000). So the Maze failure is not
a learning rate that differs from the supervised tasks; it is the same rate on a different problem (2,500 RNN
steps per episode under PPO, against a few hundred steps per supervised trial).

**What the existing runs show** (N = 1,000, `round19_maze_inspect.py`; a unit-step is *flat* where the
activation's slope is under 10% of its maximum: |r| > 0.949 for tanh, r > 0.974 or r < 0.026 for
sigmoid(7.5 (x − 0.3)); a unit is *pinned* if at least 90% of its probe steps are flat):
- **Update size.** The first update's KL is 0.000-0.002 in all 12 ReLU, leaky-ReLU and softplus networks,
  0.018, 0.030 and 0.162 in the three tanh networks (about 10 to 80 times larger), and 0.005-0.018 in the sigmoid
  networks. No ReLU, leaky-ReLU or softplus network has a single update with KL above 0.05 in 2,000 updates;
  the tanh networks have 146, 32 and 198, most of them in the first 70 updates, and the sigmoid networks 2-8.
- **tanh: saturation arrives with one very large update.** At initialisation no tanh unit is pinned (mean
  |rate| 0.31-0.34). During the first 300 updates the median participation falls from 0.51-0.76 to 0.08-0.14:
  the rates shrink toward 0. Then one update moves the policy by a KL of 678 (seed 1, update 1,319) and of
  389 (seed 3, update 1,062). In the stored participation vectors the median jumps to 1.06 between updates
  1,300 and 1,400 (seed 1) and between 1,050 and 1,100 (seed 3); at the end weights 990 and 998 of 1,000 units
  are pinned at ±1, the spectral radius of W_rec has grown from 1.2 to 5.0 and 6.6, and success stays at 0
  for the rest of training. Seed 2 has no such update: its rates end low (mean |rate| 0.04), no unit is
  pinned, and success is 0.38 at its last training probe (0.44 on the 64-episode probe of the inspection). The participation rule counts all 1,000 tanh units as
  active in every case, because q0.9(|r|) of a pinned unit is 1, so the rule cannot see saturation.
- **Shifted sigmoid: saturated from the start.** At initialisation 380-430 of 1,000 units are pinned
  (61-81 at the top, 319-349 at the bottom; 46-48% of unit-steps flat). At the end weights 933-943 are
  pinned (230-265 at the top, 678-706 at the bottom). There is no large update (largest KL 0.07-0.19); success
  stays near 0 for the first 1,000 updates and ends at 0.14-0.22.
- **A reason in the initial gain.** All models start with W_rec at spectral radius 1.2. For ReLU about half
  the units are on at the origin, so the root-mean-square gain of the linearised network is about
  1.2 × √½ = 0.85, below 1. For tanh the slope at the origin is 1, so the gain is 1.2: above 1, the regime
  in which large random tanh networks generate their own chaotic activity (Sompolinsky, Crisanti & Sommers
  1988, Phys. Rev. Lett. 61:259). The probe agrees: the untrained tanh network's mean |rate| is 0.32 at
  radius 1.2 and 0.10 at radius 0.85, against 0.06 for an untrained softplus network. For the shifted sigmoid
  the largest slope is 7.5/4 = 1.875, so the gain at the steepest point is 2.25, and many units settle at
  one of the two flat parts. Rebuilding the untrained sigmoid networks at radius 0.45 (gain 0.85) leaves 5-13
  units pinned instead of 380-430, with 1,000 of 1,000 active. These are measurements of the initial
  networks; that the initial gain is why training fails is an interpretation, which the pilot tests.

**Conditions** (`slurm/rl_maze_r19_pilot_della.slurm`): 4 conditions × 2 seeds × 2 activations at
N = 1,000, 2,000 updates each, 16 runs. Seeds 1 and 2, the seeds of the standard-setting runs, so each pilot
run has a paired standard run from the same initial draw. Only the keys named change.

| Condition | Keys | Mechanistic reason |
|---|---|---|
| lr × 0.3 | `trainer.lr=0.0003` (1.4e-4 at N = 1,000 after the size rule) | update size: tanh's first-update KL is 10-80 times ReLU's at the same rate, and its saturation followed single updates with KL of 389 and 678 |
| lr × 0.1 | `trainer.lr=0.0001` | the same, at a rate where the first-update KL should fall to ReLU's if it scales with the rate |
| gain 0.85 | `model.spectral_rad` 0.85 (tanh), 0.45 (sigmoid): radius × largest slope = 0.85, ReLU's gain at the standard radius | the initial gain: removes tanh's self-generated activity at initialisation and the sigmoid's initial pinning |
| gain 0.85 and lr × 0.3 | both | in case the two causes act together |

Not chosen: a smaller input scale, because the saturation is set by the recurrent gain (the input rows have
norm 0.1 at initialisation, and the sigmoid networks shrink them further during training, to 0.054-0.061);
a longer warm-up, because the tanh updates with KL above 0.1 continue after the 50-update ramp and the two
collapses came at updates 1,062 and 1,319, where a warm-up has no effect, while a lower rate acts at every
update.

**What counts as a pass.** A pilot run is learned if `common.maze_learned_index` finds a learned point on its
2,000-update record. A condition passes for an activation if both of its seeds are learned. For comparison,
ReLU at N = 1,000 and the standard settings learned by update 2,000 in 5 of 6 seeds. If more than one
condition passes for an activation, the setting carried forward is the first passing one in the order
lr × 0.3, gain 0.85, lr × 0.1, both: single changes before the combination, the smaller change to the rate
first. A condition with one learned seed, or with both seeds' mean success over updates 1,751-2,000 at or
above 0.7, is reported as partial; no condition is a pass on that basis. Reported per run without a
threshold: the learned update, success at the last probe, the number of updates with KL above 0.1, the first
update's KL, and the pinned and active counts at the end weights.

**What would show the interpretation wrong.** If a run at lr × 0.3 or lr × 0.1 ends with 90% or more of its
units pinned, the update-size account does not explain that run. If the gain condition does not lower the
sigmoid's pinned count at the end weights below the standard runs' 933-943, the initial gain does not set
the final saturation. Either result is reported as such.

**Reporting a setting that passes.** Any setting found is a per-activation change to the training protocol,
and it is reported as "tanh (or the shifted sigmoid) trained with <setting>", never as the standard protocol.
A comparison of such networks with the ReLU networks then differs in the activation and in the setting. If a
setting passes, I propose to the coordinator, with its cost and without submitting it: 3 seeds at
N = 1,000 with that setting under the stage protocol (3 × 5.5 GPU-h per stage, at most 4 stages), plus
3 ReLU seeds at N = 1,000 with the same setting, so that the activation comparison has a control trained the
same way (another 3 × 5.5 per stage).

### (c) A network that never learns under the cap

A network not learned at 8,000 updates is reported (cell, seed, success at the last probe, the mean success
of its last four 250-update blocks) and excluded from every statistic read at the learned point, as runs that
did not learn are excluded now. Its end-of-training numbers are listed but not drawn in a figure. A group
with an excluded network keeps its budget of 8,000 for the others.

### Jobs and cost of this package

| Job | Cluster | Indices | GPU-h |
|---|---|---|---|
| N = 4,000 seeds 1-3, stage 3 (4,001-6,000) | Della gpu80, `rl_maze_suite_della.slurm`, `STAGE=3` | 21-23 | 3 × 15.6 = 47 |
| softplus seed 3, stage 2; ReLU N = 1,000 seeds 4-6, leaky ReLU 1-3, softplus 1-2, stage 2 (2,001-4,000) | Spock, `rl_maze_suite_spock.slurm`, `STAGE=2` | 17-20, 26, 27, 29-31 | 9 × 5.5 = 50 |
| Pilot | Della, `rl_maze_r19_pilot_della.slurm` | 1-16 | 16 × 5.3 = 85 |
| **Submitted now** | | | **182** |
| Held: N = 4,000 seeds 4, 5, stage 3, after 17044667 | Della gpu80 | 24-25 | 31, which would take the package to 213; reported to the coordinator before submission |

### Code and local checks

- `slurm/rl_maze_suite_della.slurm` and `slurm/rl_maze_suite_spock.slurm` (commit 433391e): a `STAGE` switch.
  Stage k ≥ 3 continues the cell `<tag>_cont` (k = 3) or `<tag>_cont<k-1>` into `<tag>_cont<k>`; stage 2,
  the default, is the continuation as before. The launcher refuses to write into a folder that already holds
  a run. `slurm/rl_maze_r19_pilot_della.slurm`: the pilot.
- The code the runs use differs from the commit of the running N = 4,000 job (b171bf0) only in options that
  are off by default (kept snapshots, transfer to a new maze).
- Launcher smoke tests, run as sbatch would from a clean detached worktree at 433391e with `module`, `conda`
  and `srun` stubbed and `srun` appending 2 updates of 4 episodes × 6 decisions, on parent folders copied from
  the clusters. Pass condition, set before the diffs: each saved config differs from its comparison config
  only in the keys listed, plus the stub's keys and keys added to `trainer_ppo.yaml` after the parent ran
  whose values are off (`lambda_met` 0, `lambda_rwm` 0 and its `rwm_args`, `optimizer` adam, `momentum` 0,
  `prune_args.copy_*` with `prune_reinit` false, `checkpoint_keep` false). All pass.

| Run | Compared with | Keys that differ | Other checks |
|---|---|---|---|
| Della index 22, `STAGE=3` | its parent, `N=4000_act=relu_cont/s2` | experiment_tag (`_cont3`), paths.init_from, paths.save_to, trainer.iter_offset (2,000 → 4,000) | log: weights loaded from the `_cont` parent, updates 4,001-4,002; the reader joins the three stages into one record of updates 1-4,002 |
| Della index 22 again | | | exits 1: refuses to overwrite `_cont3/s2` |
| Della index 22, `STAGE=1` | | | exits 1 |
| Spock index 26 (softplus seed 3) | its parent | experiment_tag (`_cont`), paths.init_from, paths.save_to, trainer.iter_offset (0 → 2,000) | updates 2,001-2,002 |
| Spock index 29 (ReLU seed 4) | its parent | the same, and trainer.checkpoint_keep (true in the G12 parent, false here) | |
| Pilot 1 (tanh lr × 0.3, seed 1) | pilot 5 (tanh gain, seed 1) | experiment_tag, trainer.lr, model.spectral_rad | lr 1.39e-4 in the log |
| Pilot 12 (sigmoid lr × 0.1, seed 2) | pilot 15 (sigmoid both, seed 1) | experiment_tag, seed, task.seed, trainer.lr, model.spectral_rad | |
| Pilot 1 | the real standard tanh seed 3 config | experiment_tag, seed, task.seed, trainer.lr, paths.save_to | |

### Submission order

1. This entry and the inspection script are committed and pushed.
2. The commit is checked out detached in `~/trainRNNbrain_r19` on Della and on Spock.
3. Della: the three N = 4,000 stage-3 runs and the pilot. Spock: the nine N = 1,000 stage-2 runs.
4. A results entry applies these rules when the jobs land, and decides the next stages from them.
