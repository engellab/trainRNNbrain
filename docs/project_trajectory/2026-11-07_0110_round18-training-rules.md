## 2026-11-07 01:10 — Round 18 training package: what each run is for, and the read-out rules of the two transfer tests

**Why.** The audit of 7 Nov (`2026-11-07_0047_experiments-across-tasks-audit.md`, ptolmachev/dead-relu-paper#201)
lists four priority-1 training gaps that this package closes: G11 (the Maze size series at one training
length), G12 (a third learned Maze control network at N = 1,000, with its weights kept along training),
G17 (the transfer test on Maze) and G18 (the transfer test on CDDM). Pavel approved additional jobs ("Go
ahead on additional jobs"). G19, the N = 4,000 Maze continuation (Della 17012367), belongs to another
agent and is not touched here. This entry is written and pushed before any of these jobs is submitted.
No run of this package had produced a number when it was written; the one local calculation running at
that time (the maze calibration below) had reached update 30 of 300 and no success rate above 0.03.

**Terms.** *Maze*: the egocentric maze task trained by PPO (proximal policy optimisation, a
reinforcement-learning algorithm); one PPO update is one batch of 256 episodes followed by 8 optimiser
steps. *Probe*: a noise-free test of 128 episodes from fixed starts, run after update 1 and every 10th
update; it records the share of episodes that reach the cheese (probe success) and the number of active
units. A full participation vector is stored at every 5th probe (updates 1, 50, 100, ...). *Learned
point* (`common.maze_learned_index`): the first probe from which probe success stays at or above 0.9 for
five probes, in a run whose last probe is also at or above 0.9; a run without one did not learn.
*Floor crossing* (supervised tasks): the first iteration from which the smoothed noisy training loss stays
at or below 1.05 times its fitted floor until 1.2 times that iteration (`common.noisy_floor`). *Active
unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05·q95(p). *Inactive*:
not active. *Old network*: a network trained on the first task well past its read-out point. *Fresh
network*: an untrained network.

### Cells and cost

| Cell | Runs | Where | GPU-h |
|---|---|---|---|
| G11: Maze uniform series continued from update 2,000 to 4,000 | N = 250 seeds 1-4, N = 500 seeds 1-3, N = 1,000 seeds 1-3 (10 runs) | Spock, `rl_maze_suite_spock.slurm`, indices 1-9 and 28, `PARENT_ROOT`, `EXTRA_UPDATES=2000`, `INIT_ADAM=false` | 4 × 4.4 + 3 × 5.3 + 3 × 5.7 = 51 |
| G12: Maze N = 1,000 ReLU controls, kept snapshots every 100 updates | seeds 4, 5, 6 | Spock, same launcher, indices 29-31, `KEEP_ALL=true` | 3 × 5.7 = 17 |
| G17: Maze transfer, N = 1,000 | 5 old (uniform series s1 and s2, G12 seeds 4-6) and 3 fresh (seeds 1-3) | Spock, `rl_maze_transfer_spock.slurm`; the three G12 sources chained with `--dependency=aftercorr` | 8 × 5.7 = 46 |
| G18: CDDM transfer, N = 1,000 | 3 old (`CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000`) and 3 fresh | Della, `SilentReLU_b4_transfer_della.slurm`, `TASK=cddm_hyper`, `ITERS=75000` | 6 × 5.5 = 33 |

The total is about 146 GPU-h, under the standing cap of about 200 GPU-h per package. It is about 16 GPU-h
above the audit's 130 for two reasons. G11 continues all ten uniform-series runs at the three sizes,
including the two that had not learned by update 2,000 (N = 250 seed 2, N = 1,000 seed 3), because a
size series at one budget has to treat every run alike, as the N = 2,000 and 4,000 chains did. G17 has
five old sources instead of three (the reason is under G17).

### Code added for this package (commit named in the submission entry)

- `run_rl.py`: `trainer.checkpoint_keep` (default false, `configs/trainer/trainer_ppo.yaml`). Before this
  change each Maze checkpoint overwrote the run folder, so the weights existed only at the end of a run.
  With the key on, every checkpoint and the end of training write the config and the parameter file to
  `<save_to>_checkpoints/Maze_<tag>/s<seed>/u<update>/`, outside the tree that `common.maze_load_runs`
  globs. `TrainerPPO.load_run` reads a snapshot folder as a run folder.
- `run_rl.py`: `paths.transfer_from`, the Maze counterpart of the supervised transfer path that B4 used.
- `configs/task/Maze_transfer.yaml`: the second maze. `trainRNNbrain/tasks/TaskCDDMHyper.py` and
  `configs/task/CDDMHyper.yaml`: the second CDDM task.
- Launchers: G12 indices and `KEEP_ALL` in `rl_maze_suite_spock.slurm`; the new
  `rl_maze_transfer_spock.slurm`; a `TASK` switch in `SilentReLU_b4_transfer_della.slurm` (default
  `hyper3`, which leaves the B4 command line unchanged).
- Tests: `test_kept_snapshots` and `test_transfer_to_new_maze` in `tests/test_rl_maze.py`; the new
  `tests/test_cddm_hyper.py`.

### G12: the third learned Maze control

**Runs.** N = 1,000 ReLU, seeds 4, 5 and 6, the uniform series' configuration (the current composition
`task=Maze trainer=trainer_ppo model=rnn_relu_standard` differs from the saved config of uniform-series
seed 1 only in recorded values and in options that are off), 2,000 updates, kept snapshots every 100
updates (20 per run).

**What it is read for.** Finding 7 of the audit: the Maze control range (426-449 active units at update
2,000) comes from two learned networks, and the settings verdicts of `read_maze_arms.py` compare every
learned network of a setting with that range. A setting "decreases" the number of active units when all
its learned networks lie below the smallest control count.

**Rule, applied in the results entry.** For each of seeds 4-6: learned or not (learned point exists),
the learned update, the active count at the learned probe and at update 2,000, and success at the last
probe. The learned controls are then uniform-series seeds 1 and 2 plus every new seed that learned.
- If any new learned control has 417 or fewer active units at update 2,000, weight decay 1e-5 on Maze
  (370-417) no longer lies below every control, its verdict changes from "decreases" to "no change", and
  the claim that only weight decay changes the number of active units on all three tasks loses its Maze
  direction. The status board is updated in that case.
- If every new learned control has more than 417, the weight-decay direction stands.
- If no new seed learns, the control range is unchanged, and that is reported.
The other settings' verdicts are recomputed against the new range by `read_maze_arms.py` in the analysis
package (G9); this entry fixes only the weight-decay test, because it is the one the audit names.
**Snapshot check:** each run has 20 snapshot folders `u00100` to `u02000`, and the first snapshot at or
after the learned update is at most 100 updates after it.

### G11: the Maze size series at 4,000 updates

**Runs.** The ten uniform-series runs above, each continued for 2,000 updates from its update-2,000
weights (`paths.init_from`), with Adam started afresh and the 50-update warm-up ramp repeated, because
these parents were saved before the optimizer state was (`INIT_ADAM=false`, as the N = 2,000
continuations 7384301 and 7389358 ran). The continuation records updates 2,001-4,000 and
`common.maze_load_runs` joins it with its parent into one run.

**What it is read for** (the analyses and their rules belong to the analysis package and are written
there before the numbers are computed): the Maze size series with the same training length (4,000
updates) at every size from 250 to 4,000 (G10); the flat-return test over a window comparable with the
3BFF one (G8); which inactive units are active again later, and the class split, at N = 1,000 (G15,
G16). In the results entry of this package I report, per run, without a verdict: the last probe's
success at update 4,000; learned or not under the learned-point rule on the joined record (a run that had
learned by update 2,000 and ends below 0.9 counts as not learned); the learned update; and the active
count at the learned probe, at update 2,000 and at update 4,000.

### The second maze (G17)

`python trainRNNbrain/tasks/make_maze.py --cells 7 --seed 222 --loops 4`, in `configs/task/Maze_transfer.yaml`,
which inherits every other key from `configs/task/Maze.yaml`. Seed 222 was chosen before any training, as
the seed in 0-999 (6, 15 and 20 excluded, the seeds already used) whose three geodesic statistics are
closest to the first maze's (smallest summed relative deviation):

| | First maze (seed 20) | Second maze (seed 222) |
|---|---|---|
| Longest route to the cheese, cells | 29 | 29 |
| Mean route length over free cells, cells | 15.9 | 15.9 |
| Dead ends | 6 | 6 |
| Free cells | 101 | 101 |
| Cheese | centre lattice cell | centre lattice cell |

36 of the 225 grid cells differ between the two mazes. Of the 82 free cells the mazes share, 11 have the
same optimal first move towards the cheese; in the other 71 the route leaves in another direction.

**Difficulty check (local CPU, started before this entry, unread when it was written).** N = 100, 300
updates, seeds 1-3, on each maze, the calibration that chose the first maze. The measure is probe success
averaged over the last five probes (updates 260-300) and over the three seeds. The second maze is
accepted if its measure lies between half and twice the first maze's. If the first maze's measure is
below 0.1, the check cannot tell the mazes apart and the second maze is accepted on its statistics. If the
second maze is rejected, seed 761 (the next in the ranking: 29, 15.9, 6) replaces it and is checked the
same way before any G17 job is submitted.

### G17: the Maze transfer test (counterpart of B4)

**B4's procedure, read from `SilentReLU_b4_transfer_della.slurm` and the rules in
`2026-10-15_1342_open-rules-and-commands-moved-from-agent-notes.md`.** A trained 3BFF network moves to the
3-bit hyper flip-flop: the same 3 inputs, 7 outputs (every product of the bit states, the 3 original bits
among them). W_rec, W_inp, the bias and y_init are copied; W_out is the new run's own seeded draw; Adam
starts afresh; the iteration clock starts at 0; the new task is trained for the source's budget (200k
iterations). Old sources were trained 200k iterations (floor crossings near 60k), young sources were
stopped at their floor, and fresh networks are untrained with the same config.

**The Maze version matches it as follows.** The source network's W_rec, W_inp and y_init are copied (these
networks have no trainable bias); W_out is the transfer run's own seeded draw; Adam starts afresh and its
50-update warm-up ramp runs again; the update clock starts at 0; the second maze is trained for the
first maze's budget, 2,000 updates. The inputs are the same 10 channels (8 range sensors, 2 efference-copy
inputs). How the Maze version differs from B4:
- The output channels are the same two actions (turn, speed), not a larger read-out. What is new is the
  environment: the maze layout and the routes. B4 kept the task's inputs and added outputs; here the
  input statistics change and the outputs do not. The old read-out is still discarded, to keep B4's form
  (a fresh W_out, so the old policy is not carried over directly).
- Old sources are networks at update 2,000 on the first maze, about twice the update at which they learned
  (the two learned at updates 760 and 1,050). No young sources are run.
- Five old sources instead of three: only two of the three uniform-series N = 1,000 networks learned, so
  the old set is those two plus the three G12 networks, each transferred once G12 has trained it. A
  transfer whose source did not learn the first maze is excluded from the verdict and reported.
- Transfer runs are seeded with their source's seed (1, 2, 4, 5, 6), so the run folder names its source.
  Fresh runs are seeds 1-3; a fresh seed-1 network starts from the same weights as uniform-series seed 1.

**Read-out.** Every transfer and fresh run is read at its learned point on the second maze. A run with no
learned point is excluded and counted.
- S₀: units inactive in the first stored participation vector, the probe after update 1 on the second
  maze. *New:* B4 used iteration 0; the Maze trainer has no probe before its first update, and update 1
  runs at 1/50 of the learning rate (the ramp), so the network has barely changed. The active count at
  transfer is that probe's count.
- A*: units active in the first stored participation vector at or after the learned point (vectors are
  stored every 50 updates). Revived = |S₀ ∩ A*| / |S₀|.
- Also per run: newly inactive (active at update 1, inactive in A*), the active count at the learned
  probe, |S₀ ∩ A(2,000)|, the learned update, and probe success at the learned point.

**Gate (B4's).** The test is informative only if the fresh networks' mean active count at their learned
point exceeds the old networks' mean count at transfer by more than 3 fresh-network SDs (standard
deviations).

**Verdict (B4's thresholds).**
- *Inactive units do not come back*: fewer than 10% of S₀ revive in every old run read, AND the old
  networks' mean active count at the learned point is more than 3 fresh SDs below the fresh networks'
  mean.
- *Inactive units can be recruited*: 30% or more of S₀ revive in at least two thirds of the old runs read
  (B4: 2 of 3), OR the old networks' mean at the learned point is within 1 fresh SD of the fresh mean.
- Between the two bars: numbers, no verdict.
- A verdict needs at least 2 of 3 fresh runs read and at least two thirds of the eligible old runs read
  (eligible: the source learned the first maze), and never fewer than 2. Otherwise the numbers are
  reported at update 2,000, labelled end of training, without a verdict.

**What may happen, stated before the runs.** On Maze the gap that the gate needs may not exist. The
existing N = 1,000 controls had 451 and 517 active units at their learned point and 426-449 at update
2,000, so a fresh network may end only slightly above an old network's count at transfer. If the gate
fails, B4 gives no verdict. The revival fractions are then reported per run against the 10% and 30% bars
as a per-unit result, labelled as not a B4 verdict. This descriptive report is new; it is not a way to
reach a verdict.

**Reported without a threshold.** B4's "learning the new task costs nothing in R²" has no Maze
counterpart at the learned point, where success is at least 0.9 by definition. Reported instead: the
learned update of old against fresh runs, and probe success at update 2,000. Also reported, not binding:
every quantity above at update 2,000.

### G18: the CDDM transfer test (counterpart of B4)

**The second task: `CDDMHyper`.** B4's new task keeps the flip-flop's inputs and outputs every product of
the bit states over the non-empty subsets of bits, the original bits among them. A CDDM trial has two
binary sensory signs, m = sign(motion coherence) and c = sign(colour coherence); their products over the
non-empty subsets are m, c and m·c. So `TaskCDDMHyper` keeps CDDM's six inputs, trial structure and
condition grid unchanged, and has five outputs in the decision window: the cued choice (CDDM's own two
channels, unchanged, as the hyper flip-flop's single-bit channels are the flip-flop's), the motion sign,
the colour sign, and their product (+1 when the two signs agree, -1 when they differ, 0 when either
coherence is 0), whatever the cue. The audit suggested the cued choice plus the two signs; the product
channel is added because it is the CDDM counterpart of the hyper flip-flop's pair products, which make
that task's read-out combinatorial. As in B4, only the read-out is new: the input weights carry over
unchanged, so no inactive unit is handed a new drive by new input weights.

**Runs.** Old: the three networks of `CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000` (the standard
network, 200k iterations from scratch; CDDM floor crossings at N = 1,000 fall at 24k-27k). Fresh: three
untrained networks built from the same saved config (`MODE=fresh`). Each run is built from its source's
saved config with the task swapped (`task.taskname`, `task._target_`, `task.n_outputs=5`), trained
75,000 iterations from a new random seed, with kept snapshots every 1,000 iterations. No young sources.

**Read-out and verdict: B4's, unchanged.** Every run is read at its own floor crossing on the new task's
noisy loss (`common.noisy_floor(*common.noisy_loss_series(run))`, the run alone as its chain: the source
sits under `transfer_from`, so `common.warm_start_chain` stops at the transfer run). S₀ = units inactive
at iteration 0 of the new task (the first record of the participation trace); A* = units active in the
first stored participation vector at or after the crossing; revived = |S₀ ∩ A*| / |S₀|; also newly
inactive, the active count at the crossing, held-out R² under the training noise (36 draws) on the first
kept snapshot at or after the crossing, and |S₀ ∩ A(75k)|. Gate, verdict bars and the 2-of-3 requirement
exactly as in B4 (above, with "2 of 3 old seeds" for the recruitment bar).

**New for CDDM.**
- The CDDM floor crossing is uncertain by about ±30% under any rule (`docs/AGENT_GUIDE.md`, reading
  networks), so the gate and the verdict are also computed at 1.10× and 1.20× the floor. The 1.05×
  verdict binds, as in B4; if it differs at either looser margin, the results entry says so.
- Also reported at 75k (the 16 Oct rule for comparisons at one network size), not binding; a
  disagreement with the floor verdict is stated.
- B4's known risk applies: `noisy_floor` rejects a run whose smoothed loss never leaves its starting
  level. Such a run is reported and not patched.

### Local checks before submission

Tests (Mac, CPU): `tests/test_rl_maze.py` (35 checks, including the two new ones),
`tests/test_cddm_hyper.py` (3), `tests/test_transfer_warm_start.py`, `tests/test_no_function_imports.py`
and `tests/test_scripts_start.py` pass. `test_kept_snapshots` compares the update-1 snapshot of a
2-update run with the end of an independent 1-update run of the same seed (bitwise equal);
`test_transfer_to_new_maze` checks at learning rate 0 that W_rec, W_inp and y_init equal the source's and
W_out equals an untrained run's; `test_cddm_hyper` builds the new channels from the condition list with
`np.sign`, apart from the task's own code.

Launcher smoke tests ran each launcher as sbatch would, from a clean detached worktree at 2a0b127 (and
7a91311 for the fix below), with `module`, `conda` and `srun` stubbed and `srun` appending 2 updates of
4 episodes × 6 decisions (Maze) or `ITERS=30` (CDDM), on real source folders copied from Spock and Della.
Pass conditions were written before the diffs: each saved config differs from its comparison config in
exactly the keys listed, and nothing else.

| Run | Compared with | Keys that differ | Other checks |
|---|---|---|---|
| suite index 29, `KEEP_ALL=true` | suite index 7 (N = 1,000 seed 1) | seed, task.seed, trainer.checkpoint_keep | snapshots `u00001`, `u00002`, each config + parameters |
| suite index 9, continuation of real seed-3 parent | suite index 7 | experiment_tag, paths.init_from, seed, task.seed, trainer.iter_offset | updates 2001-2002, first rate lr/50 (ramp repeated) |
| suite index 1, continuation of real N = 250 seed-1 parent | none | none | updates 2001-2002 |
| transfer from real seed-1 source | suite index 7 | experiment_tag, paths.transfer_from, task.maze | log: W_rec, W_inp, y_init loaded, W_out kept; updates 1-2 |
| transfer, fresh seed 1 | transfer from seed 1 | experiment_tag, paths.transfer_from | |
| CDDM transfer from a real 200k source | its source | experiment_tag, paths.*, seed, task (taskname, _target_, n_outputs), trainer checkpoint keys, iter_offset, max_iter | W_out (5, 1000); kept snapshots every 10 |
| CDDM fresh | CDDM transfer | experiment_tag, paths.transfer_from, seed | W_out (5, 1000) |

The transfer launcher's absent-source case exited 1 without its FATAL line (`set -e` stopped it inside
the command substitution); fixed in 7a91311 and re-run: it now prints the FATAL line and exits 1.

### Submission order

1. This entry is committed and pushed.
2. The commit is checked out detached in `~/trainRNNbrain_r18` on Spock and on Della.
3. Spock: G11, G12, G17 old sources s1 and s2, G17 fresh; G17 sources 4-6 chained on G12 with
   `aftercorr`, after the difficulty check accepts the second maze. Della: G18 old and fresh.
4. A submission entry records the job ids; a results entry applies the rules above.
