## 2026-11-25 12:37 — Rules: does the entropy term's Maze verdict hold against default networks matched to it?

Written and committed before any run of this test. The verdict at stake is in
`2026-11-22_1808_entropy-correlation-setpoint-maze-result.md`: at λ_ent = 0.1 the per-unit activity-entropy set
point passes M09's Maze rule at N = 1,000 (update 2,000) and at N = 2,000 (update 4,000). Its rules are
`2026-11-21_0812_rules-entropy-correlation-setpoint-maze.md`. Branch `wp/maze-matched-controls`.

**Why the default networks and the entropy networks differ in two settings.** The entropy runs were compared with
default Maze networks that already existed, which saved about 80 GPU-hours. Those default networks were trained
before 8 Nov, so two settings differ, and the rules entry `2026-11-21_0812` lists both:
- *Trainer version.* The entropy networks use `freeze_after_end: true`: an episode's state is held after the episode
  ends. The default networks predate that option and behave as false. Without any penalty the two settings gave
  bitwise equal weights over 3 updates (entry `2026-11-21_0812`). Equality over 2,000 updates was never tested.
- *Continuation at N = 2,000.* The entropy networks kept their Adam state at update 2,000. The default networks
  restarted Adam and repeated the learning-rate warm-up.

Two further differences were not listed. The existing default networks ran from other commits (stage 1 was saved by
the script that preceded `run_rl.py`), and their update-1 rollouts differ slightly from the entropy networks' (check 2
below), so the cards or the numerics differ too. This test removes all four differences.

**Terms.** *Update*: one PPO (proximal policy optimisation) update, i.e. one rollout of 256 episodes and 8 optimiser
steps. *Probe*: a noise-free test of 128 episodes from fixed starts, every 10th update. *Active unit*: participation
p = std(r) + q0.9(|r|) on the probe, active if p ≥ 0.05 × the 95th percentile of p. *Probe success*: the fraction of
the 128 episodes that reach the goal. *Original controls*: the default networks used so far. *Matched controls*: the
default networks of this test. *Rollback guard*: the trainer's automatic stop after repeated non-finite rollouts.

### Runs: six default networks, no set-point term

- N = 1,000: seeds 1-3, 2,000 updates.
- N = 2,000: seeds 1-3, 2,000 updates, then 2,000 more from the first job's weights with the Adam state restored.
- Code: the Della clone `~/trainRNNbrain_entsp_maze`, detached at e555fae and clean. The entropy networks ran from
  that clone. This branch changes no training code.
- Launcher: `slurm/rl_maze_interventions_della.slurm`, arm `control` (indices 1-3; `ent0.1` is 130-132). The launcher's
  comment says not to submit indices 1-3, because they repeat the uniform series. Here they are run on purpose, into
  trees of their own.
- Cards, as for the entropy networks: `nomig` at N = 1,000, `nomig&gpu80` at N = 2,000.

      cd ~/trainRNNbrain_entsp_maze; export REPO=$PWD; D=$HOME/trainRNNbrain/data
      SAVE_TO=$D/trained_RNNs_rl_maze_mctrl_N1000 \
        sbatch --export=ALL --array=1-3 --time=23:00:00 slurm/rl_maze_interventions_della.slurm
      export N_OVERRIDE=2000 SAVE_TO=$D/trained_RNNs_rl_maze_mctrl_N2000
      J=$(sbatch --parsable --export=ALL --array=1-3 --time=23:00:00 --constraint="nomig&gpu80" \
        slurm/rl_maze_interventions_della.slurm)
      PARENT_ROOT=$SAVE_TO/Maze_interventions EXTRA_UPDATES=2000 INIT_ADAM=true sbatch --export=ALL --array=1-3 \
        --dependency=aftercorr:$J --time=23:00:00 --constraint="nomig&gpu80" slurm/rl_maze_interventions_della.slurm

**Cost and cap.** The entropy networks ran 13.6 s per update at N = 1,000 and 17.7 s at N = 2,000 (pilot, entry
`2026-11-21_0839`), so six networks need about 25 + 53 = 78 GPU-hours. A node at twice those times needs 15.1 h and
19.7 h per job, within the 23 h limit. The cap is 120 GPU-hours. If elapsed plus projected time passes 120, I stop
and ask firstmate; no network is dropped to fit.

### The verdict at stake, and what overturns it

M09's Maze rule (`fig_paper_F5_tasks.rule`, tolerance 0.05, floor 0.9), with the matched controls in place of the
original ones and the stored `ent0.1` networks unchanged. `ent0.1` passes at a size when:
1. every `ent0.1` network has more active units than every matched control of that size;
2. the mean probe success of `ent0.1` is at least the matched controls' mean minus 0.05;
3. no `ent0.1` network ends below 0.9 (true already; it does not depend on the controls).

The stored values the matched controls are compared with: N = 1,000, 634, 689 and 661 active units, success 0.938,
1.000 and 1.000 (mean 0.979); N = 2,000, 718, 730 and 905, success 0.992, 1.000 and 0.992 (mean 0.995).

- **Overturned:** the rule fails at N = 1,000 or at N = 2,000. It fails if one matched control has 634 or more
  active units at N = 1,000, or 718 or more at N = 2,000. It cannot fail on mean success, since 0.979 and 0.995 are
  more than 0.95 (the highest bar, at a control mean of 1.0).
- **Upheld:** the rule passes at both sizes.

The result entry gives the verdict against both control sets, whichever way it goes. The bar is not changed and no
network is added after the read.

**Expectation, so that the read can surprise.** Counts near the original controls' (415-449 and 388-506) and an
upheld verdict. Seeds 1-3 start from the same initial weights as the original controls and as `ent0.1`. A matched
control above 500 at N = 1,000 or above 600 at N = 2,000 would already say that the original bar was too low.

**Every finished network counts.** A control that does not learn the task counts as it is (the original seed 3 at
N = 1,000 ends at 0.727). A control stopped by the rollback guard is not resumed; under M09 it did not finish and
enters no mean. The rule is then applied to the finished controls and reported as "on the finished controls only". A
job ended by Slurm with TIMEOUT or NODE_FAIL is continued from its last checkpoint under
`2026-11-21_1629_entropy-setpoint-maze-wall-clock-rule.md` (NODE_FAIL is added here; a preempted or failed node is
set by the cluster, not by the network). Any other failure is reported and needs a new entry before it is rerun. An
ssh failure is not a job failure.

### Read and checks

Reader: `maze_matched_controls_read.py`, committed with this entry. It uses the existing reader's
`entropy_setpoint_read.maze_point` and `common.maze_load_runs` unchanged, and `fig_paper_F5_tasks.rule`. The arm trees
are rsync copies (config, `Progress.json`, `Participation.npz`).

1. **Configs.** Each matched control's saved config must differ from the same seed's `ent0.1` config only in
   `experiment_tag`, `paths.save_to`, `trainer.lambda_ent` and (stage 2) `paths.init_from`. Done before submission on
   the composed configs (`--cfg job`, seed 1, both sizes): the only differing keys are `experiment_tag`, `lambda_ent`
   and `save_to`.
2. **Same start.** The rollout of update 1 comes before any optimiser step, so its discounted return, success and
   steps must equal exactly those of the stored same-seed set-point networks. At N = 2,000, `ent0.1` and `cor0.01`
   already agree bit for bit (both ran on 80 GB cards); the original controls differ from both in seeds 1 and 2. At
   N = 1,000 the four stored arms show two values per seed, one per card model; a matched control must equal one of
   the two. A mismatch is traced to its cause before any verdict is reported.
3. **Independent recount.** `firstmate/data/maze-matched-controls/scripts/oracle_matched.py`, written from the rules
   text without `common.py`, the reader or the rule function. It recounts active units from the stored participation,
   finds the learned update, and evaluates the rule. Pass: exact equality with the reader for every network and every
   verdict.

**Dry runs before submission.** With the original controls only, the reader reproduces every stored Maze value and
verdict (entropy entries, `ST_verdicts`), and the recount agrees on all 57 networks (54 reach their read point and
are recounted from p). With a synthetic matched tree made of the `ent0.1` networks themselves, renamed, the reader
prints OVERTURNED and the recount agrees on all 63, so the read can give a negative answer.

### Descriptive only, not pass conditions

- The Maze verdicts of the techniques trained with the corrected trainer (`trained_RNNs_rl_maze_fz`) against the
  matched N = 1,000 controls, beside the original: prune-and-duplicate; the rate penalty at 0.03, 0.1 and 0.3; rate +
  sparsity; read-out dropout at 0.05 and 0.2; the four rescaling variants. Every finished network counts. The result
  entry says plainly whether any verdict changes. The same for `ent0.01`, `cor0.01` and `cor0.1`.
- These techniques' trainer version matches the new controls'. Their code commits and cards are not matched.
- Matched against original controls, per seed: active units, success, learned update, skipped steps and restored
  rollouts; the N = 2,000 values at update 2,000.

### Out of scope

dead-relu-paper, other tasks, new techniques. Noisy synapses on Maze stay on the first trainer version.
