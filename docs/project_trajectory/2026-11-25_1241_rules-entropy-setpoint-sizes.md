## 2026-11-25 12:41 — Rules: the entropy set point at λ = 0.1 across network sizes, and whether active units keep falling past 75,000 iterations

Written and committed before any run of this test, the pilot included. Branch `wp/entropy-sizes`. Pavel's ask, via
firstmate (25 Nov): the paper is to include the entropy set point at its stronger setting, λ_ent = 0.1.

**Where this comes from.** At λ_ent = 0.1 the term meets the paper's technique criterion (M09) at N = 1,000 on 3BFF
(`2026-11-21_0755_...`), Maze (`2026-11-22_1808_...`, also at N = 2,000) and CDDM (`2026-11-24_1848_...`). The paper
reports prune-and-duplicate across sizes: N = 500 to 4,000 on 3BFF and CDDM (Supplementary Table 4, `tab:ED3`, and its
notes) and N = 500 to 2,000 on Maze (entry `2026-11-20_1600_...`, gap A). This entry applies the same procedure to the term.

**Terms.**
- *The term*: λ_ent × mean_i ((H_i − 0.75)/0.75)², with H_i the binary entropy, in bits, of Φ(μ_i/s_i), and μ_i and s_i the
  mean and standard deviation of unit i's pre-activation h. Unchanged: `configs/experiment/entropy_setpoint.yaml` (3BFF,
  CDDM) and `configs/trainer/trainer_ppo.yaml` (Maze). The 0.75-bit target is the N = 1,000 3BFF value; it is not
  re-derived for another size or task.
- *3BFF*: the 3-bit flip-flop task. *CDDM*: the context-dependent decision-making task.
- *Active unit*: p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05 × the 95th percentile of p.
- *Floor crossing*: the first iteration from which the smoothed noisy training loss stays at or below 1.05 times its fitted
  floor until 1.2 times that iteration (`common.read_run`). The noisy loss of a network trained with the term includes
  λ_ent times the term, so its floor is task loss plus term.
- *Held-out R²*: `heldout_r2.heldout_r2`, the training noise, 36 seeded noise draws, CPU.
- *Learned point* (Maze): the first noise-free probe from which probe success stays at 0.9 or above for five probes, in a run
  whose last probe is also at 0.9 or above (`common.maze_learned_index`).
- *M09*: `dead-relu-paper/methods/M09_remedies.tex`.

### Networks (λ_ent = 0.1, three seeds per cell)

| Task | N | Budget | Segments, Adam state restored at each join |
|---|---|---|---|
| 3BFF | 500 | 200,000 iterations | 75,000 + 125,000 |
| 3BFF | 2,000 | 200,000 iterations | 40,000 + 35,000 + 62,500 + 62,500 |
| 3BFF | 4,000 | **cut now**, see the budget | |
| CDDM | 500, 2,000 | 100,000 iterations | one run each |
| CDDM | 4,000 | 100,000 iterations, if the pilot's cost allows | 50,000 + 50,000 |
| Maze | 500 | 4,000 updates | 2,000 + 2,000 |

The budgets are those of the prune-and-duplicate networks and of the controls: 200,000 iterations on 3BFF, 100,000 on CDDM
and 4,000 updates on Maze. The N = 1,000 networks are not continued here; another task (`entropy-prune-combo`) does that.

**Launchers.** First segments: `slurm/entropy_setpoint_della.slurm` (3BFF) and `slurm/entropy_setpoint_cddm_della.slurm`
(CDDM), the launchers of the N = 1,000 networks, with `N` and `ITERS_OVERRIDE`, array tasks 4-6 (`ent0.1`). Continuations:
`slurm/SilentReLU_ff_topup200k_della.slurm`, which continued the 3BFF controls and prune-and-duplicate networks, with
`KEEP=1`. Maze: `slurm/rl_maze_interventions_della.slurm`, array tasks 130-132 (`ent0.1`, seeds 1-3), `N_OVERRIDE=500`,
then `PARENT_ROOT`, `EXTRA_UPDATES=2000`, `INIT_ADAM=true`. Weight snapshots are kept every 1,000 iterations.

**One change to the continuation launcher** (8235c71). It continues a network from the network's saved config. A 3BFF
config saved since 15 Oct holds the trial generator's seed as a number, so a continuation would draw the trials of its
parent again from the first one. The launcher now gives the generator the continuation's own new seed. Configs without a
generator seed (the controls', and CDDM's) are unaffected.

**Check 1 (done before this entry).** The two launchers and the continuation launcher, with `module`, `conda` and `srun`
stubbed, at N = 50 for 40, 30 and 20 iterations on CPU (`firstmate/data/entropy-sizes/smoke/`). Each continuation's saved
config differs from its parent's in `paths.init_from`, `paths.init_adam`, `trainer.iter_offset`, `trainer.max_iter`, the
tag, `seed`, and on 3BFF `task.seed` (equal to the new `seed`), and in nothing else.

### Controls (existing, not retrained) and how each differs in training settings

The controls are those the paper uses for prune-and-duplicate at each size. Values at the paper's read points:

| Task, N | Control cell | Active units at the floor crossing | Active units at 75,000 | R² at 75,000 (mean) |
|---|---|---|---|---|
| 3BFF, 500 | `NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=500_pen=none_arm=none`, six | 202, 211, 197, 179, 235, 206 | 198, 211, 190, 178, 219, 190 | 0.9513 |
| 3BFF, 2,000 | same root, N = 2,000, three | 385, 415, 370 | 393, 330, 364 | 0.9505 |
| 3BFF, 4,000 | `NBitFlipFlop_paper_grid_c200000/EqType=h_N=4000_arm=control`, three | 517, 598, 540 | 503, 588, 542 | 0.9512 |
| CDDM, 500 | `CDDM_paper_grid/EqType=h_N=500_arm=control`, three | 291, 302, 302 | 252, 240, 242 | 0.8333 |
| CDDM, 2,000 | same root, N = 2,000, three | 543, 522, 580 | 446, 449, 449 | 0.8351 |
| CDDM, 4,000 | same root, N = 4,000, three | 776, 769, 756 | 623, 639, 636 | 0.8307 |
| Maze, 500 | `Maze_suite/N=500_act=relu` s1-3 joined with `N=500_act=relu_cont` (Mac tree) | learned point: 281, 259, 279 | update 4,000: 226, 226, 240 | success at 4,000: 1.000, 0.992, 1.000 (mean 0.997) |

Sources: `data/remedy_reads_31oct/remedies_catalogue.json`, `data/round18/g6_reads.json` and
`data/claim_gaps/claim_gaps_maze.json`. The flattened saved config of every control was compared with the `ent0.1`
N = 1,000 config of its task, whose launcher the new networks use (`firstmate/data/entropy-sizes/`, 25 Nov). Besides N, the
term's keys, seeds, tags, paths and snapshot settings, the differences are:

- **3BFF, every size.**
  - *Segments.* The controls were trained as 40,000 + 35,000 + 125,000 iterations with the Adam state restored at each join
    (first segments of 23-29 Sep 2026, last of 12-15 Oct). The new networks have the joins listed above. Each join also
    draws a new seed and restarts the gradient-spike guard's window of 1,000 gradient norms, in both groups.
  - *Trial generator.* Unseeded in the controls (`task.seed: null`), seeded from the run's seed in the new networks. The
    distribution of trials is the same.
  - *Initial weights.* The controls' configs predate `model.init_draws` (15 Oct) and their code reused the same random draws
    for every weight matrix; the new networks set `init_draws: shared`, which reproduces that.
  - *Keys added to the code later, at the values that reproduce the old behaviour*: `model.sigma_w` 0, `trainer.optimizer`
    adam, `trainer.momentum` 0; and `prune_args` keys, inert with `prune_reinit: false`. Within the N = 500 controls, four
    inert `prune_args` keys also differ between networks.
  - *N = 500 has six controls*, N = 2,000 and 4,000 three. With six controls, the largest control count is the maximum of
    six values, so the count condition is stricter than with three.
- **CDDM, every size.** The controls are single runs of 100,000 iterations (28-29 Sep 2026), and so are the new networks of
  N = 500 and 2,000; N = 4,000 has one join at 50,000. `init_draws` and the added keys are as on 3BFF. The trial generator
  is unseeded in both groups (the batch is the full grid of conditions). `trainer.max_iter` is read only by the noise
  annealing, which is off.
- **R² at 75,000.** 3BFF: measured on the weights at 75,000 in both groups. CDDM: measured on the snapshot at 75,000 in the
  new networks, and estimated from the loss and the final weights in the controls, which kept no snapshots. This is M09's
  rule, as at N = 1,000. R² at the crossing: measured on the snapshot in the new networks where one exists, estimated in the
  controls.
- **Maze, N = 500.** The new networks use `freeze_after_end: true`; the controls behave as false. The controls were continued
  at update 2,000 with a fresh Adam state and a repeated warm-up ramp; the new networks keep their Adam state. These are the
  differences stated for gap A and for the term at N = 2,000.
- **Code version.** The controls ran at the code of 23-29 Sep (and 12-15 Oct), the new networks at the commit named in the
  submission entry.

### Question 1: the verdicts, fixed now

Two verdicts per cell, at the paper's read points.

**(S) The size test, the paper's rule for prune-and-duplicate** (G6 of `2026-11-07_0100_round18-rules.md`; gap A on Maze).
*Count condition*: every network of the cell has more active units than every same-size control. A cell **passes S** if the
count condition holds
- on 3BFF and CDDM: at the floor crossing (1.05×) and at the participation probe nearest 75,000 iterations;
- on Maze: at the learned point and at update 4,000.

A 3BFF or CDDM network without a floor crossing enters no crossing read, and the number of networks read is stated (as the
paper does: prune-and-duplicate at N = 4,000 on 3BFF rests on one of three networks). If no network of a cell has a floor
crossing, the cell fails S. On Maze a network without a learned point by update 4,000 makes its cell fail S, as in gap A.
Counts at 1.10× and 1.20×, and the difference in R² (or success) at each read point, are reported and do not enter S. The
1.05× read binds; the result entry says where a looser margin gives another answer.

**(M) The M09 criterion at the size**, at the fixed read point. A cell **passes M** if
- 3BFF, CDDM, at 75,000 iterations: the count condition holds and the mean held-out R² is at least the controls' mean minus
  0.01: 0.9413 (3BFF N = 500), 0.9405 (3BFF 2,000), 0.8233, 0.8251 and 0.8207 (CDDM 500, 2,000 and 4,000), up to rounding of
  the control means, which are recomputed at the read;
- Maze N = 500, at update 4,000: the count condition holds, the mean success is at least the controls' mean minus 0.05
  (0.947), and no network ends below 0.9.

The verdict is `fig_paper_F5_tasks.rule`. Prune-and-duplicate is judged by the same M on the same controls from its recorded
reads, for comparison. (At N = 4,000 on 3BFF its R² is 0.023 below the controls' at 75,000, so it passes S and fails M there.)

**What counts as a failure.** One network at or below the largest control at either read point fails S for its cell; a mean R²
more than 0.01 below the controls' (Maze: success more than 0.05 below, or one network below 0.9) fails M. The statement
"the term increases the number of active units at every size tested on this task" fails if one cell of the task fails S. The
statement "the term meets the criterion at every size tested" fails if one cell fails M. A failing cell is named in the
result's first paragraph.

**Counted networks.** Every network that finished its budget. A network stopped by the rollback guard did not finish and
enters no mean; it is reported, and a cell with fewer than three finished networks is reported as incomplete and passes
neither S nor M. A network stopped by its wall clock or by a node failure is continued from its last `resume/` state if the
budget allows. Skipped updates (the `gradient spikes` log line) are reported per network.

**Predictions, before any run.** S passes in every cell: at N = 1,000 the term's networks had at least 92 more active units
than the largest 3BFF control and at least 197 more than the largest CDDM control. M passes at N = 500 and 2,000. The least
certain cells are 3BFF N = 2,000 (the 0.75-bit target holds a unit's on-fraction, and the trained controls' fraction of
active units falls with N, from about 40% at N = 500 to 18% at N = 2,000) and CDDM N = 4,000 (R²; the N = 1,000 mean was
0.002 to 0.003 below the controls').

### Question 2: do active units keep falling past 75,000 iterations?

Read on the size runs only: 3BFF N = 500 and 2,000 to 200,000, and CDDM N = 500, 2,000 (and 4,000 if run) to 100,000, with
the same-size controls over the same iterations. A(t) is the median number of active units over the probes in
(t − 2,000, t]. Read points: t = 50,000, 75,000 and 100,000 (both tasks), and 125,000, 150,000, 175,000 and 200,000 (3BFF).

- **Does the fall continue?** For a cell, with t_end the end of the budget: *continues* if A(t_end) < A(75,000) in all three
  networks; *does not continue* if that holds in none; otherwise *mixed*. The same is reported for the controls.
- **Is it still falling at the end (3BFF)?** The paper's test of units lost (G7): minus the least-squares slope of the active
  count on every probe from 150,000 to 200,000, per 100,000 iterations; a network is *decreasing* if it loses at least 5
  units per 100,000.
- **Does the verdict survive?** The count condition at each read point (A(t) of every network above A(t) of every control),
  and M at the end of the budget: the count at the last probe and the held-out R² of the final weights, both groups, with
  the tolerance of 0.01. If M holds at 75,000 and not at the end, the result says that the 75,000-iteration verdict does not
  hold for longer training.

Prediction: the fall continues in every cell, in the controls as well, and the count condition holds at every read point.

### Budget: 300 GPU-hours in total, pilot included

**3BFF N = 4,000 is cut now.** The 3BFF N = 4,000 controls took 72 h for 125,000 iterations (2.07 s per iteration, job
15431475). Three networks of 200,000 iterations would take 345 GPU-hours at that speed, more than the whole budget; three of
75,000 iterations (no floor crossing at the paper's budget) would take 130. The pilot measures its time per iteration for the
record. The decision is reopened only if the pilot gives less than 0.5 s per iteration.

**Pilot.** `ent0.1`, one network each, `SWEEP=entropy_sizes_pilot`, no result: 2,000 iterations at N = 500 and 2,000 and 500
at N = 4,000, on 3BFF and CDDM. s = the log's training seconds divided by the iterations. No Maze pilot: at N = 500 the
prune-and-duplicate networks took 6.2 h per 2,000 updates, and at N = 1,000 the term's took 6.8 to 13.9 h where
prune-and-duplicate's took 6.2; 7.0 h is used.
- Cost of a cell = 3 × (iterations × s + 0.25 h per job); P = the pilot's GPU-hours; F = P plus the cells other than N = 4,000.
- If 1.05 × (F + CDDM N = 4,000 at 100,000 iterations) ≤ 300, that cell runs as planned.
- Else, if 1.05 × (F + CDDM N = 4,000 at 75,000 iterations) ≤ 300, that cell runs to 75,000 (50,000 + 25,000). It keeps both
  read points of S; its floor is fitted on a shorter record than the controls', and it has no part in question 2.
- Else CDDM N = 4,000 is cut. If F alone exceeds 300, 3BFF N = 2,000 stops at 137,500 iterations.
- While the jobs run: if the hours used plus the hours still predicted exceed 300, pending N = 4,000 segments are cancelled
  first, and that is recorded.

If a pilot runs out of memory, ends with a non-finite loss or skips more than 5% of its updates, its cell does not start and
a new entry decides.

**Cluster.** Della gpu-short (every job at most 23 h), from a clean clone `~/trainRNNbrain_entsizes` pinned to the commit
named in the submission entry; `--constraint=nomig` (`nomig&gpu80` at N = 4,000), and `--exclude` of the AMD nodes
`della-i14g[1-20]`, on which the term's earlier networks took 1.1 to 2 times as long. Each time limit is at least twice the
predicted time. Spock is not used.

### Read and checks, thresholds fixed now

- **Reader:** `entropy_sizes_read.py` (new, written after this entry, before any network is read): per network
  `read_remedies.read_one` (crossing), `fig_paper_F2_fails.read_fixed` (75,000) and the joined participation trace; Maze by
  `claim_gaps_read.maze_cells`. Run on Della CPU nodes (3BFF, CDDM) and on the Mac (Maze, on rsynced copies; controls from
  the Mac tree).
- **Gate:** the reader reproduces the controls' values of the table above before any verdict: counts exactly, R² to three
  decimals (four where the source gives four).
- **Completeness:** the verdict refuses a cell without exactly three networks and the controls' number in the table.
- **Check 2:** a separately written script (numpy only; no `common.py`, no `heldout_r2.py`) recounts, per network including
  controls, the active units from the stored participation at the 75,000 probe, at the probe nearest the reader's crossing,
  and A(t) at every read point of question 2 (exact equality); it recomputes the Maze learned point and counts (exact
  equality), and one network per cell's noise-free R² at the end of its budget by a numpy forward pass (within 10⁻⁵ of torch).
- **Check 3:** each new network's saved config, per segment, differs from its control's only as listed above.
- **Manipulation check (description):** the median over units of |H_i − 0.75| on the final weights, held-out batch, training
  noise, noise seed 20261121, per network, controls included; Maze: the raw term at update 1 and its mean over the last 50.

### Existing N = 1,000 networks, for description

The N = 1,000 `ent0.1` networks (75,000 iterations) are read by the same reader on 3BFF and CDDM and shown beside the sizes,
without a verdict at the floor crossing: their budget is shorter than the controls'. `entropy-prune-combo` gives the N = 1,000
crossing reads at the full budget.
