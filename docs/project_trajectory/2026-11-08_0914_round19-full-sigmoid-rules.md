## 2026-11-08 09:14 — Round 19: full runs of the shifted sigmoid at initial spectral radius 0.45, with a same-radius ReLU control (rules, before submission)

**Why.** The pilot condition "shifted sigmoid, initial spectral radius 0.45" passed at 4,000 updates under the
rule of `2026-11-07_1723_round19-pilot-stage2-rules.md`. Both seeds are learned on the record through update 4,000:
- seed 1 learned at update 1,050 and ends at 1.00 with 237 active units;
- seed 2 learned at update 2,510 and ends at 0.97 with 216.
That pass rule was written after the pilot's 2,000-update results were seen, so the pass is labelled as such.
The coordinator approved the full runs on 8 Nov: 3 sigmoid seeds at N = 1,000 under the stage protocol, plus
3 ReLU seeds at the same spectral radius. tanh at radius 0.85 did not pass (seed 1 ends at 0.84 and seed 2 at
0.54 at update 4,000, neither learned), so tanh gets no full runs.

**Runs.** `slurm/rl_maze_r19_pilot_della.slurm`, indices 17-22 (commit 324d3b7), N = 1,000, the Maze suite's
configuration with only `model.spectral_rad=0.45` changed:
- 17-19: shifted sigmoid (`rnn_sigmoid_shifted_standard`), seeds 3, 4, 5;
- 20-22: ReLU (`rnn_relu_standard`), seeds 3, 4, 5.
The seeds are new draws. The pilot's seeds 1 and 2 are not reused, because they were carried forward on their
outcome; they are reported beside the full runs, labelled "pilot". Output is `Maze_r19full/N=1000_act=<act>_rad0.45/s<seed>`.

**Protocol.** The stage protocol of `2026-11-07_1008_round19-train-all-rules.md`:
- stages of 2,000 updates; each later stage continues with `PARENT_ROOT=<maze tree>/Maze_r19full`,
  `EXTRA_UPDATES=2000`, `INIT_ADAM=false` and `STAGE`;
- a network stops at the end of the first stage at which it is learned, with a cap of 8,000 updates;
- the same budget within each group of three, sigmoid and ReLU control separately, so each group's budget is the
  largest stop stage among its own seeds.
A network not learned at 8,000 is reported and excluded.

**Read-out.** For each network:
- the learned update;
- the active count at the learned probe and at the end of training;
- success at the end;
- for the sigmoid, the counts of units pinned at the top and at the bottom of the sigmoid, read from the end
  weights with `round19_maze_inspect.py`.

The comparison is in two parts, both reported without a verdict threshold, because none was set before:
- sigmoid at radius 0.45 against ReLU at radius 0.45: the same initial radius, so only the activation differs;
- ReLU at radius 0.45 against the standard ReLU networks (radius 1.2, seeds 1-6): this measures the effect of the
  radius itself.

The ReLU `srad0.5` arm (radius 0.5, seeds 1-3, all learned by update 2,000, 338-414 active) is listed beside
them as the nearest existing setting.

Every sigmoid result is reported as "shifted sigmoid trained with initial spectral radius 0.45", a
per-activation change to the protocol, never as the standard protocol.

**Cost.** Stage 1 is 6 × 5.3 = 32 GPU-h on Della. At most 4 stages per group: 128 GPU-h in the worst case. The
ReLU control at radius 0.5 learned by 2,000 updates, so the control will most likely need one stage. Any stage
beyond the second is reported to the coordinator with its cost before it is submitted.

**Local checks.** A smoke test from a clean worktree at 324d3b7, with `module`, `conda` and `srun` stubbed and
2 updates of 4 episodes × 6 decisions; the pass condition was set before the diffs.
- Index 17's config differs from the pilot sigmoid radius-0.45 seed 1 config only in experiment_tag, seed,
  task.seed, save_to and the stub's keys.
- Index 20's config differs from the ReLU `srad0.5` seed 3 config only in experiment_tag, model.spectral_rad
  (0.5 → 0.45), save_to, the stub's keys, and keys added since that run whose values are off.
- Index 23 exits 1 (outside the grid).
- Index 5, a pilot index, writes the same config as the real pilot run, apart from save_to and the stub.
