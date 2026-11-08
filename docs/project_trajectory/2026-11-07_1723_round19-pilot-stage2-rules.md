## 2026-11-07 17:23 — Round 19 pilot amended: two conditions continued to 4,000 updates (amends 2026-11-07_1008_round19-train-all-rules.md, part (b))

**This continuation was decided after the pilot's results were seen.** The pilot (Della 17046884, 16 runs,
N = 1,000, 2,000 updates) ended with no condition passing under the rule of the part 1 entry. On the logs'
last probes I reported the shifted sigmoid at gain 0.45 as partial and tanh at gain 0.85 as tanh's best
condition (0.58 and 0.57). The coordinator then approved continuing those two conditions by one stage. The
reason given was this: a pilot judged at 2,000 updates should not rule out an activation, when ReLU-family
networks needed up to 4,000 updates to learn (ReLU N = 1,000 seed 3, softplus seed 3, every N = 2,000 network).
So the 4,000-update result is read under rules written after the 2,000-update result was known, and it is
labelled that way wherever it is used. The part 1 rule for the pilot at 2,000 updates is not changed. Under that
rule no condition passed, and that verdict stays on record.

**The pilot at update 2,000, by the learned rule** (`common.maze_learned_index` on the copied records; part 1
gives the terms):

| Condition | tanh, seeds 1, 2: last probe (learned?) | shifted sigmoid, seeds 1, 2: last probe (learned?) |
|---|---|---|
| lr × 0.3 | 0.43, 0.54 (no, no) | 0.11, 0.39 (no, no) |
| lr × 0.1 | 0.43, 0.51 (no, no) | 0.32, 0.43 (no, no) |
| gain 0.85 (tanh radius 0.85, sigmoid 0.45) | 0.58, 0.57 (no, no) | 0.99 (learned at update 1,050), 0.79 (no; mean 0.70 over updates 1,751-2,000) |
| gain and lr × 0.3 | 0.45, 0.27 (no, no) | 0.45, 0.58 (no, no) |

No condition passes. Shifted sigmoid at gain 0.45 is partial (one of two seeds learned). Standard-setting
runs, for comparison: tanh 0.00, 0.38, 0.02; sigmoid 0.21, 0.15, 0.15.

**The end weights: are tanh units pinned at ±1?** No. Read with `round19_maze_inspect.py --root
data/trained_RNNs_rl_maze/Maze_r19pilot` (output `data/round19/maze_inspect_pilot.json`; pinned means at least
90% of probe steps where the slope is under a tenth of its maximum):
- **tanh:** 0 to 13 of 1,000 units are pinned in every pilot run. Two standard-setting runs had 990 and 998.
  No pilot run has an update with KL above 0.42; the two standard-setting collapses had KL 389 and 678. The
  part 1 entry named one result that would show the update-size account wrong: a run at lower rate ending with
  90% or more of its units pinned. That result did not occur. The tanh runs still fail, but a different way:
  - gain 0.85 runs end with mean |rate| 0.01 (initially 0.10);
  - the other conditions end with mean |rate| 0.03-0.35.
  The participation rule counts all 1,000 units active in every run, because it is relative to the 95th
  percentile.
- **Shifted sigmoid at gain 0.45:** 0 units pinned at the top, 736 and 757 pinned at the bottom
  (rate < 0.026, nearly off). The standard runs had 230-265 at the top and 678-706 at the bottom. The active
  count is 297 and 258, mean |rate| 0.03. The units this setting removes from the upper flat part are the
  units with no ReLU counterpart; the lower flat part is the sigmoid's counterpart of an inactive ReLU unit.

**What is continued.** The two pilot runs of each condition, from their update-2,000 weights, by one stage
(updates 2,001-4,000), under the stage protocol of part 1: `INIT_ADAM=false`, the warm-up ramp again, the
update clock continued, the condition's keys passed again.

| Runs | Launcher, indices | GPU-h |
|---|---|---|
| tanh gain 0.85, seeds 1, 2 | `rl_maze_r19_pilot_della.slurm` 5, 6 | 2 × 5.3 |
| shifted sigmoid gain 0.45, seeds 1, 2 | 13, 14 | 2 × 5.3 |
| **Total** | Della | **about 21** |

**What counts as a pass at 4,000.** A condition passes for its activation if both of its seeds are learned
on the joined record through update 4,000. "Learned" means `common.maze_learned_index`, including the
condition that the last probe, at update 4,000, is at or above 0.9. Sigmoid seed 1, learned at 2,000, has to
stay learned at 4,000. Anything else is no pass. A condition with one seed learned at 4,000 is reported as
partial. This package continues it no further; whether to do so is for the coordinator.

**What a pass leads to: a proposal, not a submission.** If a condition passes, I propose full runs to the
coordinator with their cost, and nothing is submitted unasked:
- **3 seeds of that activation at N = 1,000 with that initial spectral radius**, under the stage protocol
  (2,000-update stages until learned, cap 8,000): 3 × 5.3 = 16 GPU-h per stage, at most 64.
- **3 ReLU seeds at N = 1,000 as a control for the setting.** "The same gain" can mean two different
  controls, and the proposal names both:
  - At the same gain, radius × largest slope = 0.85, the ReLU control is the standard ReLU network (radius
    1.2; 1.2 × √½ = 0.85). Seeds 1-6 already exist at 4,000 updates, so this control costs nothing.
  - At the same spectral radius (0.85 for the tanh comparison, 0.45 for the sigmoid), the ReLU networks
    differ from the standard ones in their initial gain, as the activation runs do. That control is new:
    3 × 5.3 = 16 GPU-h per stage, most likely one or two stages (ReLU at radius 0.5, the `srad0.5` arm,
    learned all three seeds by update 2,000). I will recommend the same-radius control, because the radius
    is the setting that changed. A comparison with the activation then varies the activation and nothing else.
- Either way the activation's networks are reported as "tanh (or the shifted sigmoid) trained with initial
  spectral radius r", a per-activation change, never as the standard protocol.

**Launcher and local checks.** `slurm/rl_maze_r19_pilot_della.slurm` gains the continuation mode (commit
748926e). Smoke tests ran from a clean worktree at 748926e with `module`, `conda` and `srun` stubbed and
2 updates of 4 episodes × 6 decisions, on the real pilot folders copied from Della:
- Indices 5 and 13: each saved config differs from its parent's only in experiment_tag (`_cont`),
  paths.init_from, paths.save_to, trainer.iter_offset (0 → 2,000) and the stub's keys. model.spectral_rad is
  unchanged (0.85, 0.45). Updates 2,001-2,002.
- A second index 5 exits 1, refusing to overwrite `tanh_gain_cont/s1`.
The pass condition, set before the diffs, was exactly this list.
