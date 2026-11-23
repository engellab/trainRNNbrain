## 2026-11-21 08:12 — Rules, Stage 2: the entropy and correlation set points on Maze (N = 1,000 and 2,000)

Written and committed before any Maze run of this test, the pilot included. Stage 1 result:
`2026-11-21_0755_entropy-correlation-setpoint-3bff-result.md` (both terms pass on 3BFF at both λ values). Stage 1
rules: `2026-11-20_2307_rules-entropy-correlation-setpoint.md`. Firstmate pre-authorised Stage 2 if a term passed
on 3BFF (21 Nov): Maze at N = 1,000 and 2,000, three networks per size, the M09 Maze criterion, the existing
default Maze networks of each size as controls, at most 300 GPU-h across Della and Spock. Branch
`wp/entropy-setpoint-maze`.

**Terms.**
- *Term (A)*, the activity-entropy set point: λ_ent × mean_i ((H_i − 0.75)/0.75)². Here H_i is the binary entropy,
  in bits, of Φ(μ_i/s_i), and μ_i and s_i are the mean and standard deviation of unit i's pre-activation h.
- *Term (B)*, the pairwise-correlation set point: λ_cor × mean_i (c_i/0.4)². Here c_i is the mean correlation of
  unit i's h with every other unit's h.
- *Update*: one PPO (proximal policy optimisation) update, i.e. one rollout of 256 episodes followed by 4 passes of
  2 minibatches each.
- *Probe*: a noise-free test of 128 episodes from fixed starts, run after update 1 and every 10th update. It records
  probe success (the fraction of the 128 episodes that reach the goal) and the number of active units.
- *Active unit*: p = std(r) + q0.9(|r|) on the probe, and active if p ≥ 0.05 × the 95th percentile of p.
- *Learned point*: the first probe from which probe success stays at 0.9 or above for five probes, in a run whose
  last probe is also at 0.9 or above (`common.maze_learned_index`).

### The code (already in this branch's commit, before any run)

- `TrainerPPO` gains `lambda_ent`/`ent_args` and `lambda_cor`/`cor_args` (`entropy_setpoint_penalty` and
  `correlation_setpoint_penalty`). Each is the supervised term, typed out again line for line. It is computed on the
  pre-activations of each minibatch's recompute, over the steps before each episode's end (`live_steps`), and added
  to that minibatch's loss as the metabolic penalty is. An RNN whose equation is not "h" is refused.
- `configs/trainer/trainer_ppo.yaml` sets both λ to 0, with the 3BFF arguments: target 0.75 bits for (A); target
  correlation 0 and scale 0.4 for (B).
- `slurm/rl_maze_interventions_spock.slurm` gains the arms `ent0.01`, `ent0.1`, `cor0.01` and `cor0.1` (indices
  127-138; seeds 1-3, the controls' seeds). The Della wrapper `slurm/rl_maze_interventions_della.slurm` runs the
  same launcher.
- `tests/test_setpoint_penalties.py` adds four Maze tests:
  - each PPO term equals the supervised term on the same kept steps;
  - for a unit that is off on every step of a Maze recompute, the gradient of each term with respect to the
    unit's incoming weights is nonzero, while the gradient of the rate-based metabolic penalty is exactly zero;
  - at λ = 0, `ppo_update` never calls either term and reports no key for it.
- `entropy_setpoint_read.py` gains the modes `maze-read` and `maze-verdict`.

**Check 1, off is unchanged (done).** The interventions launcher's control command was run twice on the CPU with
the same overrides (seed 7, N = 40, 3 updates of 16 episodes × 60 decisions): once from main (0c49e04, exported
with `git archive`) and once from this branch. `Params_Maze.npz`, `Participation.npz` and `Progress.json`
(apart from wall-clock fields) are identical. The saved configs differ only in the new keys at λ = 0 and in the
output path. The same command with `trainer.lambda_ent=0.1`, or with `trainer.lambda_cor=0.1`, gives different
weights, so the comparison could have failed. All 38 tests of `tests/test_rl_maze.py` pass.

### Design

**Stage 2a: N = 1,000, both λ values per term.** Arms `ent0.01`, `ent0.1`, `cor0.01` and `cor0.1`, three networks
each (12 networks), 2,000 updates. These are the two values per term that ran on 3BFF. No other value is run.

**Stage 2b: N = 2,000, one λ per term, chosen by the Stage 2a result with this rule:**
- if exactly one of the term's two λ values passes at N = 1,000, that value is run;
- if both pass, the value with the larger mean active count at update 2,000 is run (λ = 0.1 on a tie);
- if neither passes, the term fails on Maze and is not run at N = 2,000.

Each N = 2,000 network trains for 4,000 updates, as two jobs: 2,000 updates, then 2,000 more continued from the
first job's weights with its Adam state restored (`PARENT_ROOT`, `EXTRA_UPDATES=2000`, `INIT_ADAM=true`, a
per-task `aftercorr` dependency). This is the procedure of the size-series runs, entry
`2026-11-19_1749_rules-maze-duplicate-sizes.md`.

**Read points.** Each size is read at the end of its budget:
- N = 1,000 at update 2,000: M09's own Maze cell and read point;
- N = 2,000 at update 4,000: the equal budget of the Maze size series.

At update 2,000, none of the three N = 2,000 controls has learned the task (probe success 0.77, 0.84 and 0.84; their
learned updates are 3,000, 2,480 and 2,840). M09's floor of 0.9 at update 2,000 would therefore measure how fast a
network learns, not how many of its units are active. The N = 2,000 networks' update-2,000 values are reported
without a verdict.

### Controls (existing, not retrained)

- **N = 1,000:** M09's Maze controls, `trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu` s1-3, at update 2,000:
  - 426, 449 and 415 active units (largest 449);
  - probe success 0.992, 0.977 and 0.727 (mean 0.898).

  These are the values in `data/read_maze_arms.json` (`reference_techniques`, all three controls,
  dead-relu-paper#194). The reader's dry run on the Mac's copy of the tree reproduces them exactly.
- **N = 2,000:** `Maze_suite/N=2000_act=relu` s1-3 joined with `N=2000_act=relu_cont`, at update 4,000:
  - 400, 506 and 388 active units (largest 506);
  - probe success 0.984, 0.977 and 1.000 (mean 0.987).

  These equal the R2 values that `data/claim_gaps/claim_gaps_maze.json` gives these controls.

**The control tree is read from the Mac's copy** (`~/Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze`).
The folder of the same name on Della (`~/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu` and
`N=2000_act=relu`) holds different runs. Their saved configs have no `warmup_updates` key, so they predate the
warm-up ramp. Read at update 2,000, they give 455, 423 and 404 active units and probe success 0.98, 0.98 and 1.0 at
N = 1,000, which are not M09's values. Those Della runs enter nothing here.

**Known differences between the new networks and the controls**, stated before the runs:
- The new networks use the config default `freeze_after_end: true`. The controls were trained before 8 Nov and
  behave as false. For an unpenalised network over 3 updates the two settings are bitwise equal (entry
  2026-11-10_0806); with a set-point term on, `freeze_after_end: true` means the term sees only the steps before
  each episode's end.
- The N = 2,000 controls were continued at update 2,000 with a fresh Adam state and a repeated warm-up ramp; the
  new networks keep their Adam state. A cold restart adds a median of 3 active units just after its ramp, and the
  difference does not last through a stage (entry 2026-11-18_2033).
- The code is this branch's commit, pinned on Della; with both λ at 0 it trains exactly as main does (Check 1).

### Pass criterion, fixed now (M09's Maze rule)

An arm (one term at one λ) **passes at size N** when all three of its networks reach the read point and:
1. every one of its networks has more active units at the read point than the largest same-size control (449 at
   N = 1,000; 506 at N = 2,000);
2. its mean probe success at the read point is at least the controls' mean minus 0.05 (0.848 at N = 1,000; 0.937 at
   N = 2,000);
3. no one of its networks ends below a probe success of 0.9.

The verdict is `fig_paper_F5_tasks.rule(active, success, ctrl_active, ctrl_success, 0.05, floor=0.9)`.
- A term **passes at size N** if one of its λ values passes there (M09's one-value rule).
- A term **passes on Maze** if it passes at both N = 1,000 and N = 2,000.
- A network that the rollback guard stops before its read point is not resumed. Its arm then has fewer than three
  finished networks, which M09 calls "incomplete", and it does not pass at that size.

### Read-out

`entropy_setpoint_read.py maze-read` runs on the Mac. The arms are read from copies of the Della trees made with
rsync (config, `Progress.json` and `Participation.npz` only); the controls are read from the Mac tree above. Each
run is joined along its warm-start chain (`common.maze_load_runs`) and cut at its read point. The reader records:
- active units and probe success at the read point;
- the learned update and the active count there;
- summed skipped minibatch steps and restored rollouts;
- the active count at every probe;
- the arm's raw set-point penalty at update 1 and averaged over its last 50 updates.

The arm is derived from the saved config; a config that does not match its folder stops the reader.
`maze-verdict` then prints the verdicts and draws the figure (active units against updates, every network, with the
controls; one row per size, one column per term).

**Descriptive read-outs, not pass conditions.**
- The fold change of each arm over the controls' mean count, and its success difference from the controls.
- The learned update and the count there.
- The N = 2,000 values at update 2,000.
- Skipped and restored counts per network.
- The manipulation check: the raw penalty at update 1 and over the last 50 updates of each network. A term whose
  penalty did not fall did not test what it was meant to test, and is reported as such.

**Check 2, reads checked per run, by a separately written script** (adapted from the size-series oracle,
`firstmate/data/claim-gaps/proof/oracle_maze.py`, which does not use `common.py`). The script:
- joins each run's stage folders in update order and cuts at the read point;
- finds the learned point from the definition;
- recounts the active units at the read point from the stored participation p, where one is stored there.

Pass: exact equality with the reader, per run, for the read-point count, the success, the learned update and the
recount.

### Pilot and budget rule

**Pilot** (enters no result): four jobs of 40 updates each, run on Della's `gputest` partition with
`--constraint="nomig&gpu80"`: `ent0.1` and `cor0.1`, seed 1, at N = 1,000 and N = 2,000. Each runs the
launcher's command plus `trainer.n_updates=40`, into `trained_RNNs_rl_maze_entsp_pilot`. Recorded for each job:
- the running mean of seconds per update, as the log prints it at update 40 (it includes the slow first update);
- the peak GPU memory (`sacct` TRESUsageInMax, `gres/gpumem`);
- the summed skipped minibatch steps and restored rollouts.

A pilot that runs out of memory, or that has skipped or restored steps in more than 5 of its 40 updates, stops
Stage 2 for that size until the cause is found.

**Budget.** For each size, s_N is the larger of the two terms' printed means at update 40. P is the pilot's
GPU-hours. The cost per network is:
- c_1000 = s_1000 × 2,000 / 3,600 + 0.2 h;
- c_2000 = s_2000 × 4,000 / 3,600 + 0.4 h.

The design that runs is the first one that fits the cap:
- if 12 c_1000 + 6 c_2000 + P ≤ 300 GPU-h, the design above runs;
- otherwise, if 6 c_1000 + 6 c_2000 + P ≤ 300, Stage 2a runs λ = 0.1 only for both terms, and Stage 2b follows the
  same rule;
- otherwise, Stage 2a runs λ = 0.1 only and Stage 2b is not run; the result says that N = 2,000 did not fit.

The N = 2,000 jobs run with `--constraint="nomig&gpu80"`, because a default N = 2,000 run peaks at 36 GB of GPU
memory (Della job 17958687_7) and the correlation term holds several more copies of the states. The N = 1,000 jobs
use `nomig`, since a default N = 1,000 run peaks at 18 GB (job 17958689_7). All jobs run on gpu-short with
`--time` ≤ 23 h.

### Jobs (Della; code pinned at this commit in a clean clone `~/trainRNNbrain_entsp_maze`)

    # Stage 2a
    REPO=$HOME/trainRNNbrain_entsp_maze SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_entsp_N1000 \
      sbatch --array=127-138 --time=14:00:00 slurm/rl_maze_interventions_della.slurm
    # Stage 2b, for each selected arm's three indices (stage 1, then the continuation after it)
    REPO=... N_OVERRIDE=2000 SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_entsp_N2000 \
      sbatch --array=<idx> --time=23:00:00 --constraint="nomig&gpu80" slurm/rl_maze_interventions_della.slurm
    REPO=... N_OVERRIDE=2000 SAVE_TO=<same> PARENT_ROOT=<same>/Maze_interventions EXTRA_UPDATES=2000 INIT_ADAM=true \
      sbatch --array=<idx> --dependency=aftercorr:<stage-1 job> --time=23:00:00 --constraint="nomig&gpu80" \
      slurm/rl_maze_interventions_della.slurm

### What each outcome means

- **A term passes on Maze** (at both sizes): it meets the M09 criterion on two of the three tasks. CDDM is not
  tested; the brief leaves it out.
- **A term passes at N = 1,000 only:** the term meets M09's own Maze cell, and the result names the size at which
  it fails.
- **A term fails at N = 1,000 at both λ values:** the term fails on Maze. N = 2,000 is not run for it.

In every case the paper is unchanged: the set-point terms are not among its techniques.
