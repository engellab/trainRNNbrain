## 2026-11-07 01:36 — Round 18 parity runs: the 3BFF technique values that pass, run on CDDM and Maze (rules before the runs)

**Why.** The G2 result of the analysis package (commit 0c2eeb9; `data/round18/g1g2.json`; status board
C6) found seven 3BFF technique values other than the ones the paper shows that meet the paper's
criterion at tolerance 0.01. The paper's rule says a technique passes a task if any one of its values
passes. On 3BFF each technique was run at 13 to 26 values, on CDDM and Maze at one, so the rule favours
3BFF. To compare like with like (Pavel: "same interventions on all of them"), the coordinator asked for
the passing 3BFF values to be run on CDDM and Maze at N = 1,000, 3 networks each. This entry is the
second rules entry of the round 18 training package; the first is `2026-11-07_0110_round18-training-rules.md`.
It is committed and pushed before any of these runs is submitted. No run of this set has started.

**Terms.** *3BFF*: the 3-bit flip-flop task. *CDDM*: context-dependent decision making. *Maze*: the
egocentric maze task trained by PPO (proximal policy optimisation). *Active unit*: participation
p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05·q95(p). *Held-out R²*: R² on a held-out
batch under the training noise, 36 noise draws (`heldout_r2.py`). *Probe success* (Maze): the share of
128 noise-free test episodes from fixed starts that reach the cheese.

### Which values, and why

From `data/round18/g1g2.json` (3BFF, N = 1,000, 75k iterations, against controls with 236-288 active
units and mean R² 0.9513):

| 3BFF cell | Active (3 networks) | Mean R² minus control mean | Run here |
|---|---|---|---|
| noisy synapses σ_w = 0.1 | 320, 362, 348 | −0.0008 | yes |
| noisy synapses σ_w = 0.3 | 373, 351, 432 | −0.0039 | yes |
| read-out dropout, rate 0.05, β 4 | 392, 452, 418 | −0.0062 | yes |
| read-out dropout, rate 0.05, β 1 and 2; rate 0.10, β 1 | 357-445 | −0.0068 to −0.0095 | no: rate 0.05 with β 4 passes, and β 4 is the paper's |
| rescaling, revive_ops2: target 4.0, protection 0.20 | 317, 388, 402 | −0.0097 | yes |

σ_w is the relative s.d. of the multiplicative noise on each synapse at every step; β is the sharpness
of the rank softmax that sets each unit's drop probability; the rescaling target is the soft-max rate,
in multiples of the rate cap, at which a revived unit stops being tilted.

The revive_ops2 rescaling rule, every key from its saved config (`NBitFlipFlop_revive_ops2/EqType=h_k=3_N=1000_op=rescale_novel=false_tgt=4.0_step=0.0005_prot=0.20`):
check_every 1, patience 1, active_rel 0.05, reinit_mode rescale, revive_op rescale, rescale_alpha 1.002,
rescale_target_frac 4.0, rescale_active_only true, rescale_protect_frac 0.2, rescale_cap 64.0,
rescale_normalize true, utility participation, maturity 0, max_replace_frac 1.0 (rescale_norm_mode and
rescale_refractory absent; their defaults, self and 0, reproduce the code of that time). The CDDM
rescaling launcher (target ladder) and the Maze rescale arm pass the same keys with only the target
changed; the syn_* keys are read only by the synaptogenesis branch and have no effect here.

### The runs (24 networks)

| Cell | Task | Launcher, indices | Output folder |
|---|---|---|---|
| σ_w = 0.1 | CDDM, 75k | `SilentReLU_synnoise_ext_della.slurm` 16-18 | `CDDM_synnoise_ext/EqType=h_N=1000_sw=0.1_iters=75000` |
| σ_w = 0.3 | CDDM, 75k | same, 19-21 | `.../sw=0.3_iters=75000` |
| dropout 0.05, β 4 | CDDM, 75k | `SilentReLU_paper_grid_della.slurm` arm `mute0.05`, `TASK=cddm SWEEP=parity_r18 ITERS_OVERRIDE=75000`, 40-42 | `CDDM_parity_r18/EqType=h_N=1000_arm=mute0.05` |
| rescaling, target 4.0 | CDDM, 75k | `SilentReLU_target_ladder_della.slurm` cell 3, `TASK=cddm`, 13-15 | `CDDM_target_ladder/EqType=h_N=1000_tgt=4.0` |
| σ_w = 0.1, 0.3 | Maze, 2,000 updates | `rl_maze_interventions_spock.slurm` arms `sigmaw0.1`, `sigmaw0.3`, 115-120 | `Maze_interventions/sigmaw0.1`, `.../sigmaw0.3` |
| dropout 0.05, β 4 | Maze | same launcher, arm `dropout0.05_beta4`, 121-123 | `Maze_interventions/dropout0.05_beta4` |
| rescaling, target 4.0 | Maze | same launcher, arm `rescale_tgt4`, 124-126 | `Maze_interventions/rescale_tgt4` |

Each CDDM technique keeps the launcher of its existing CDDM cell, so the trainer
(`trainer_ptrack_freshbatch`, `+experiment=silent_units_largeN`), the tracking options and the network are
the existing cells'. CDDM runs stop at 75k, the paper's read-out point, and keep a weight snapshot every
1,000 iterations, so R² at 75k is measured. Maze runs use seeds 1-3, as every interventions arm does.
All runs are at commit efde479 or later (the submission entry names the pin), not at the commits where
the existing cells ran (852a17f, 84b4485, 99530cf on Della; 55fab3e9 to 19ba7249 on Spock). The smoke
tests below compare each new saved config with the saved config of the existing cell.

**Cost.** CDDM: noisy synapses 6 × about 4.4 h, dropout 3 × about 5.5 h, rescaling 3 × about 3.9 h,
about 55 GPU-h. Maze: noisy synapses 6 × about 16 h, dropout 3 × about 5.7 h, rescaling 3 × about 6 h
(less if the rollback guard stops it), about 131 GPU-h. Total about 186 GPU-h, inside the 170-200 GPU-h
the coordinator gave. Together with the first part of the package (about 146 GPU-h) round 18 trains
about 332 GPU-h.

**One difference from the CDDM controls that is not a technique.** The new CDDM networks draw their
initial W_out independently of W_inp (`model.init_draws: independent`, the default since 15 Oct). Both
CDDM controls (`CDDM_paper_grid` arm control and `CDDM_std_g0_bias0_c75000`) and the existing CDDM
dropout and noisy-synapse cells predate that key and were built with the old shared draws, in which W_out
started as a rescaled copy of W_inp. The existing CDDM rescaling cell (target 10) already uses
independent draws against the same controls. Matching the old draws would mean overriding a fixed
config value in the launcher, which the config rule forbids, so the difference is stated with each CDDM
result. The Maze networks and their controls all use independent draws.

### Local smoke tests (before submission)

Each launcher ran as sbatch would, from a clean detached worktree at efde479, with `module`, `conda` and
`srun` stubbed: Maze indices 115, 118, 121 and 124 for 2 updates of 4 episodes × 6 decisions; CDDM
paper-grid task 40 (`ITERS_OVERRIDE=20`), synnoise_ext 16 and 19 and target-ladder 13 (`trainer.max_iter=20`
appended). All exited 0. Pass condition, written before the diff: each saved config differs from the
saved config of the existing cell of the same technique only in the swept key, bookkeeping (seed,
experiment_tag, paths, max_iter, iter_offset, the smoke test's own overrides) and keys that did not
exist when the existing cell ran, each at a value that is off or that reproduces the old behaviour.

| New run | Compared with | Swept key | Keys absent from the old config (value in the new run) |
|---|---|---|---|
| CDDM mute0.05 | `CDDM_paper_grid` arm mute | drop_rate 0.2 → 0.05 | init_draws (independent, see above), sigma_w 0, checkpoint_keep true / every 1000, optimizer adam, momentum 0, unused prune_args keys (prune_reinit false) |
| CDDM σ_w 0.1, 0.3 | `CDDM_synnoise_ext` σ_w 1.0 | sigma_w 1.0 → 0.1, 0.3 | init_draws (independent), checkpoint_keep true / every 1000, optimizer adam, momentum 0 |
| CDDM target 4.0 | `CDDM_target_ladder` target 10 | rescale_target_frac 10 → 4 | none |
| Maze σ_w 0.1, 0.3 | `Maze_interventions/sigmaw1.0` | sigma_w | the options added since (prune_reinit false, optimizer adam, lambda_met 0, lambda_rwm 0, momentum 0, checkpoint_keep false) |
| Maze dropout 0.05 | `Maze_interventions/dropout0.2_beta4` | drop_rate 0.2 → 0.05 | the same options, all off |
| Maze rescale target 4 | `Maze_interventions/rescale_few` (target-10 runs left no config) | target 10 → 4, protection 0.05 → 0.20 (the paper's, as the target-10 arm) | the same options, all off; copy_* keys unused in rescale mode |

The pass condition holds for every run; the one substantive difference is init_draws on CDDM.

### Read-out and criterion: the paper's, per task

The criterion is `methods/M09_remedies.tex` (lines 29-35 and 95-104 on the paper's origin/main) and
`fig_paper_F5_tasks.rule`, unchanged. A value **passes** on a task when both hold:
1. every network trained with it has more active units than every control network; and
2. the networks' mean performance is at least the controls' mean minus the tolerance (one-sided).

**CDDM.** Read at 75,000 iterations: the active count on a noise-free pass of the 75k weights, and
held-out R² (36 draws) from the 75k weights, read with the same functions as `read_uniform_cells.py`.
Controls: `CDDM_paper_grid/EqType=h_N=1000_arm=control`, the control of every CDDM technique in the
paper: 342, 334 and 333 active at 75k, R² 0.8312, 0.8350 and 0.8349 (estimated from the loss ratio,
since those 100k runs kept no snapshots), mean 0.8337. So a CDDM value passes if every network has more
than 342 active units and the mean R² is at least 0.8237 (tolerance 0.01). Also reported, not binding:
tolerances 0.005 and 0.02 (`tolerance_sensitivity`), and the second control of the read-out files,
`CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0` (5 networks, 320-334 active, measured R²
0.831-0.834). The new networks' R² is measured while the main control's is estimated; the existing
CDDM rescaling cell has the same asymmetry, and the second control, whose R² is measured, is the check.

**Maze.** Read at update 2,000: the active count and probe success at the last probe
(`read_maze_arms.py`, `active_end`, `success_end`). Criterion of decision #194 (dead-relu-paper#194: all
three controls, as first written on engellab/trainRNNbrain#5): the controls are the uniform series'
N = 1,000 ReLU networks s1, s2 and s3 (426, 449 and 415 active; success 0.992, 0.977 and 0.727; mean
0.898). A Maze value passes if every network has more than 449 active units, the mean success is at
least 0.848 (0.898 − 0.05), and no network ends with success below 0.9. A network stopped by the
rollback guard (the PPO safety stop after non-finite updates) enters no mean; a value with no finished
network fails. Also reported, not binding: the same test against the two learned controls only (more
than 449 active, mean success at least 0.935), as decision #194's sensitivity check, and against the
control set that includes the G12 networks once they finish.

**Reported with each value.** Per network: active count, R² or success, and for Maze the learned point;
per value: pass or fail and which clause fails. Whether a technique passes on a task under the one-value
rule follows by combining these values with the value the paper already shows; that combination, and
any change to the paper's verdict table, belong to the analysis package.

**Known risk, stated before the runs.** The Maze rescale arm at target 10 (same tilt rate 1.002, cap 64
and protection 0.20) was stopped by the rollback guard in all three networks at updates 52-60. Target
4.0 changes only the stopping target, so the same stop is likely; under the criterion that value then
fails on Maze.
