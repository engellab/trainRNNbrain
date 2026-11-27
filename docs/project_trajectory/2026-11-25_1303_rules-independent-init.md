## 2026-11-25 13:03 — Rules: do the shared initial draws change the number of active units?

Written and committed before any training run of this test. Branch `wp/independent-init`. A reviewer of the
dead-ReLU paper raised that every network in the paper except 27 starts its input, recurrent and read-out
matrices from the same random sequence, and that the paper calls the effect of varying the draws untested
(`dead-relu-paper/sections/07_discussion.tex`, `methods/M01_network.tex`, `supplement/SM_methods.tex`). Pavel
asked, via firstmate (25 Nov), for an experiment that varies them.

**Terms.**
- *3BFF*: the 3-bit flip-flop task. *CDDM*: the context-dependent decision-making task. *N*: number of units.
- *Draw mode*: how the three initial weight matrices are drawn (`model.init_draws`), `shared` or `independent`.
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass; active if p ≥ 0.05 × the 95th
  percentile of p. *q0.9*: the 90th percentile over steps and trials.
- *Update*: one PPO (proximal policy optimisation) update of the Maze trainer. *Probe*: the noise-free test of
  128 episodes run after update 1 and every 10th update. *Learned point*: the first probe from which probe success
  stays at 0.9 or above for five probes, in a network whose last probe is also at 0.9 or above
  (`common.maze_learned_index`).
- *Floor crossing*, *t\**: `common.read_run`, the iteration at which the noisy loss falls to 1.05 times its floor.

### What is shared, and what is varied

`RNN_torch.sparse` draws a matrix from a CPU generator. For the standard (non-Dale) network,
`get_connectivity_unconstrained` draws W_rec, then W_inp, then W_out from N(0, 1/N), and rescales W_rec to spectral
radius 1.2.
- **shared**: the generator is seeded again with the network's seed before each matrix, so each matrix starts with
  the same numbers. The flattened W_inp equals the flattened W_out entry for entry over their common length, and it
  equals the start of the flattened W_rec up to the rescaling factor (checked below).
- **independent**: one generator is seeded once and supplies the three matrices in order. W_rec takes the first draws, so
  W_rec is identical in both modes; W_inp and W_out differ.

**This experiment therefore varies the initial W_inp and W_out, at fixed initial W_rec.** With the same seed, the
network's other random streams (training noise, task batches) are seeded identically in both modes.

### Finding made before the rules: the Maze networks already have independent draws

The paper says that only the 27 networks of the DMTS setting sweeps have independent draws. The saved configs and
the trained weights contradict that sentence.
- **Configs.** All 431 saved Maze configs on the Mac have `init_draws: independent`. In the supervised sweeps on
  Della, 104 networks outside the 27 have it as well: 47 of CDDM and 57 of 3BFF (the sweeps trained after 15 Oct 2026,
  entry `2026-10-15_2206`), among them `CDDM_absorb_noise` (6), `CDDM_b5_optimizer` (15), `CDDM_parity_r18` (3),
  `CDDM_std_g0_penalties_bias0` (6), the N = 1,000 `CDDM_std_g0_recinit` networks (5 of 15) and most 3BFF setting sweeps
  (noise, metabolic, weight decay, recinit, activations). Which of them enter the paper is not established here.
- **Weights.** In the six default Maze networks of N = 1,000 (seeds 1-6), the correlation of the trained W_inp with the
  initial W_inp rebuilt from the seed is 0.76 to 0.79 under the independent mode and −0.01 to 0.01 under the shared mode
  (W_out: 0.39 to 0.44 and −0.02 to 0.02). W_rec is the same in both modes, so it cannot tell them apart.
- **Consequence.** An independent-draw Maze arm would vary nothing. The Maze arm of this test is therefore
  **shared** draws, against the paper's six independent-draw networks. The 3BFF and CDDM arms are independent
  draws, against the paper's shared-draw networks, as asked. Firstmate was told before submission.
- **A comparison in the paper is affected.** The 3BFF and CDDM setting-sweep networks with independent draws are compared
  with shared-draw control networks. This test measures how large that mismatch is at the default settings.

### Runs: N = 1,000, three networks per task

Launcher `slurm/independent_init_della.slurm`. Each command is that of the launcher that trained the paper's
networks, plus the seed and `model.init_draws`. Each network has the seed of one of the paper's networks, so it has
the same initial W_rec as that network.

| Task | Draws | Seeds | Length | The paper's networks |
|---|---|---|---|---|
| 3BFF | independent | 1558264694, 254798194, 254859059 (first runs of three of the four default chains) | 75,000, then 200,000 if the GPU-hour rule allows | `NBitFlipFlop_ff_revive*`, arm none |
| CDDM | independent | 3833537590, 1272763220, 457853 | 100,000 | `CDDM_paper_grid`, arm control |
| Maze | shared | 1, 2, 3 | 2,000 updates | `Maze_suite/N=1000_act=relu`, s1-s6 |

Snapshots are kept every 1,000 iterations (Maze: every 100 updates). Output: `NBitFlipFlop_independent_init`,
`CDDM_independent_init` and `Maze_independent_init` in the usual trees.

**Differences from the paper's networks that remain.** (1) 3BFF: the paper's chains are 40,000 + 35,000 + 125,000
iterations with Adam restored at each join; here 75,000 + 125,000. The paper's 3BFF task config had `seed: null`; the
current one seeds the batches with the network's seed. (2) Maze: the paper's networks ran the first trainer version
(`freeze_after_end` absent, read as false); the current default is true. Without a penalty, the setting can change
training only through the rollback guard (the trainer's restoring of weights after a non-finite rollout), and the first
version of the N = 1,000 networks used neither guard in the first 2,000 updates. Each Maze run's restores are reported.
(3) CDDM and 3BFF: the code version, checked below. (4) The participation trace and the snapshots add no computation to
training.

**Checks done before this entry.**
1. *Code version, 3BFF and CDDM.* Same overrides, N = 50, 30 iterations, CPU, seed 3833537590 (CDDM) or 1558264694
   (3BFF), penalties 0: the final W_rec, W_inp and W_out of commit 852a17f (CDDM controls' commit) or 8bfd030 (the main
   commit at the 3BFF controls' run time) equal those of the current code with `init_draws=shared` exactly (maximum
   difference 0.0), with the training noise and without. The same run with `independent` differs by up to 0.77, so the
   comparison could have failed. First attempt: a mismatch, found to be the old configs' nonzero default penalties
   (lambda_rws 0.05, lambda_frm 0.2), which the controls' saved configs have at 0.
2. *Config diff.* Each stubbed run's saved config against a control's (CDDM seed 3833537590; 3BFF seeds 1558264694 and
   4186074845; Maze seed 4): the tag, the paths, the seed, the swept keys, N and budget (smoke values), and keys added
   later at values that reproduce the old behaviour: `sigma_w` 0, `optimizer` adam, `momentum` 0, `iter_offset` 0, the
   `prune_args` and set-point keys (inert), `paths.transfer_from` null, Maze `freeze_after_end` true. 3BFF also
   `task.seed`. Repeated on the real configs at the read.
3. *Draw modes at N = 1,000*, the nine launcher seeds: W_rec identical in the two modes for all nine;
   corr(W_inp, W_out) is 1.000 (shared) and between −0.04 and 0.03 (independent); corr(W_inp, start of W_rec) is at
   least 0.995 (shared) and between −0.02 and 0.01 (independent).

### GPU-hours

Expected: 3BFF 75k 3 × 5.0 h, CDDM 3 × 4.3 h, Maze 3 × 5.3 h = 44 (measured speeds of the same networks on 80 GB
A100 cards; all jobs ask for `--constraint=gpu80`). Continuing the three 3BFF networks to 200,000 (8.2 h each, 25
h in all) brings the total to about 69. **Cap 80 GPU-hours (elapsed time × 1 GPU, finished or not).** The 3BFF
continuation is submitted only if the hours used so far plus 27 (3 × 9) are at most 80. If the hours used would pass 80,
the jobs still running are cancelled, latest submitted first, and reported. Time limits are twice the usual time: 10 h,
9 h, 11 h, and 17 h for the continuation. A network killed by its wall clock is continued from its last `resume/` state
if the cap allows.

**Every finished network counts.** A network that did not reach its length is reported and enters no verdict. None is
replaced.

### Read-out

All networks are read by `independent_init_read.py` (to be written; it reads, and refuses to give a verdict on an
incomplete set) with the project's rules:
- **3BFF, CDDM.** Active units at 75,000 iterations (`fig_paper_F2_fails.read_fixed`) is the primary read point, the
  paper's comparison of networks of one size (M05). Secondary: the floor crossing (`common.read_run`, 1.05 margin),
  for 3BFF only if trained to 200,000 iterations. Held-out R² at 75,000 (36 draws).
- **Maze.** Active units at update 2,000 is the primary read point (M09); the learned point is secondary. Probe success.

**Reference networks** (existing; not retrained). 3BFF and CDDM: every shared-draw network with the default settings and
N = 1,000 on Della, read by the same code (13 and 29; the cells are listed in `independent_init_read.py`). They
include the paper's size-series networks (4 and 3). Maze: the six suite networks (their records on the Mac).

| Task, read point | n | Range | Mean | Margin (5% of mean) | Bounds |
|---|---|---|---|---|---|
| 3BFF, 75,000 | 13 | 236-309 | 274 | 14 | 222-323 |
| 3BFF, floor crossing | 13 | 240-344 | 285 | 14 | 226-358 |
| CDDM, 75,000 | 29 | 311-354 | 333 | 17 | 294-371 |
| CDDM, floor crossing | 29 | 390-468 | 423 | 21 | 369-489 |
| Maze, update 2,000 | 6 | 394-449 | 417 | 21 | 373-470 |
| Maze, learned point | 6 | 406-517 | 440 | 22 | 384-539 |

The reference values were read (Della CPU job 18375187 and the Mac's Maze records) before this entry, and the margin was set
after seeing them. The margin is the same 5% for every task and read point.

### What would count as dependence on the draws, fixed now

For a task and read point: a new network is *outside* if its number of active units is below the range minimum minus
the margin, or above the range maximum plus the margin.
- **Depends**: at least two of the three finished networks are outside, on the same side.
- **Not detected**: all finished networks are inside (and at least two finished).
- **Unresolved**: exactly one is outside, or two on opposite sides, or fewer than two networks finished.

A task *depends on the draws* if its primary read point is "depends"; a secondary read point that disagrees
with the primary is reported and does not overturn it. The paper's result depends on the shared draws if 3BFF or
CDDM depends; Maze answers whether shared draws change a task that the paper trained with independent draws.

**Power, stated now.** The bounds accept shifts of up to about 18% (3BFF), 11% (CDDM) and 13% (Maze) of the reference
mean. If a new network were drawn from the same distribution as the reference networks, the chance that it lies beyond the
range on a given side is about 1/(n+1), so two of three outside on one side by chance is below 4%, 1% and 13% at
margin zero, and lower with the margin. A shift smaller than the bounds is therefore neither shown nor excluded; the
difference of the mean active count from the reference mean is reported with a 95% Welch interval.

**Also reported.** (1) Any network with 500 or more of its 1,000 units active would contradict the paper's statement
that most units are inactive. (2) Performance: mean held-out R² within 0.01 of the paper's default networks' mean (3BFF,
CDDM) or probe success at least 0.9 at update 2,000 (Maze). A difference in active units between networks that differ in
performance by more than that is reported as confounded.

**Prior knowledge.** Four independent-draw CDDM networks with default settings already exist: `CDDM_absorb_noise`,
noise=default (358, 329 and 329 active units at 75,000; entry `2026-11-02_0058`) and `CDDM_std_g0_recinit`,
sr=1.2, seed 1129222953 (315; floor crossing 391). All four lie inside the CDDM bounds, and they were known when the
bounds were set. They do not enter the CDDM verdict.

### Checks at the read

1. **Independent recount.** A script that shares no code with `common.py` recounts the active units of every new network
   at the read probes from the stored participation vectors (exact equality), and recomputes one network's
   participation per task from its saved weights with a numpy forward pass of equation (`h`).
2. **Draws.** For every new network, the first kept snapshot's W_inp (iteration 1,000; Maze update 100) is
   correlated with the initial W_inp rebuilt in each mode. Pass: the declared mode's correlation exceeds the
   other's by at least 0.2.
3. **Config diff** on the real configs, as Check 2 above.
