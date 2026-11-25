## 2026-11-22 18:31 — Rules: the entropy set point on CDDM

Written and committed before any training run with the term on CDDM. Branch `wp/entropy-cddm`. Pavel's ask, via
firstmate (22 Nov): run the per-unit activity-entropy set point on CDDM and see whether it helps there too.

**Where this comes from.** The term (A) of `2026-11-20_2307_rules-entropy-correlation-setpoint.md`
(`Penalties.entropy_setpoint_penalty`, `lambda_ent`, `ent_args` in `trainRNNbrain/trainer/Trainer.py`) met the M09
technique criterion on 3BFF at N = 1,000 at λ_ent = 0.01 and 0.1 (`2026-11-21_0755_entropy-correlation-setpoint-3bff-result.md`)
and on Maze at N = 1,000 at both values (`2026-11-22_1808_entropy-correlation-setpoint-maze-result.md`). CDDM is the
paper's third task and the only one where the term has not been run. The correlation set point (B) failed on Maze and is
not run here.

**Terms.**
- *CDDM*: the context-dependent decision-making task (`TaskCDDM`, T = 300 steps, the full grid of conditions in every batch).
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass; active if p ≥ 0.05 × the 95th percentile of p.
- *H_i*: the binary entropy, in bits, of Φ(μ_i/s_i), where μ_i and s_i are the mean and standard deviation of unit i's
  pre-activation h over the steps and trials of the training batch. The term is P_A = mean_i ((H_i − 0.75)/0.75)².
- *Held-out R²*: `heldout_r2.heldout_r2`, the training noise, 36 seeded noise draws, CPU (`docs/AGENT_GUIDE.md`).
- *M09*: `dead-relu-paper/methods/M09_remedies.tex`, the paper's methods paragraph on training techniques.

### The term, unchanged

The experiment config `configs/experiment/entropy_setpoint.yaml` as it stands at this commit: target 0.75 bits,
eps 10⁻⁶, `model.init_draws: shared`, the correlation term off. No change to the term or its config. The target is the
3BFF value (the median H_i of the active units of the three 3BFF controls); it is not re-derived for CDDM.

### Runs

- **Launcher** `slurm/entropy_setpoint_cddm_della.slurm` (new): the CDDM command of
  `slurm/SilentReLU_paper_grid_della.slurm`, which trained the controls, with `+experiment=entropy_setpoint` in place of
  `silent_units_largeN`, `trainer.lambda_ent` set per arm, 75,000 iterations, and snapshots kept every 1,000 iterations.
- **Arms** `ent0.01` (array tasks 1-3) and `ent0.1` (4-6); three networks each (`seed=random`); N = 1,000; the standard
  network and trainer (`trainer_ptrack_freshbatch`); participation probes every 10 iterations, full vectors every 100.
- **Output:** `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/CDDM_entropy_setpoint/EqType=h_N=1000_arm=<arm>/`
  (the tree that holds the controls); snapshots under `.../data/c1_snapshots/CDDM_entropy_setpoint/`.
- **Cluster:** Della gpu-short, `--time=11:00:00`, `--constraint=nomig`, from a clean clone `~/trainRNNbrain_entcddm`
  pinned to the commit named in the submission entry. Spock only if Della does not start the jobs within a few hours.

**Budget: 70 GPU-hours, by construction, with no pilot.** The six jobs have an 11-hour limit each, so the screen cannot
use more than 66 GPU-hours. The expected cost is lower: the 3BFF networks with the term took 5 h 11 min to 6 h 43 min for
75,000 iterations at N = 1,000 (PR #11), and the CDDM controls took 3.9 h for 100,000 iterations under the same
launcher family, so a CDDM network should take 3 to 7 h, and the six 18 to 40 GPU-hours. Both values of λ therefore run;
the brief's fallback to one value is not needed.

**A network killed by its wall clock** has not finished. It is continued from its last `resume/` state
(`paths.init_from` + `trainer.init_adam`, the clock from `trainer.iter_offset: auto`) to 75,000 iterations, but only if the
GPU-hours already used plus the continuation's time limit stay within 70. Otherwise it is not counted, and that is reported.

### Why the arms match the controls

The controls ran 100,000 iterations and these run 75,000. The only part of the trainer that reads `max_iter` is the noise
annealing schedule, and `anneal_noise` is false in `trainer_ptrack_freshbatch`, so the first 75,000 iterations of the
two budgets are trained the same way.

**Check 1 (done before this entry).** The launcher, with `module`, `conda` and `srun` stubbed, at N = 50 for 30 iterations
on CPU, seed fixed, array tasks 1 and 4 (`firstmate/data/entropy-cddm/smoke/`). Each saved config, flattened, against the
saved config of control seed 3833537590 (the three controls' configs differ from one another only in the seed):
- the term: `lambda_ent` (0.01 or 0.1), `lambda_cor` 0, `ent_args`, `cor_args`;
- `model.init_draws: shared`. The controls were trained on 28 Sep 2026, before 2026-10-15, when every weight matrix reused
  the same random draws; their configs predate the key. M09 requires the controls' configuration except for the
  technique's parameters;
- `checkpoint_every`, `checkpoint_keep`, `paths.*`, the seed, the tag, N and `max_iter` (smoke values);
- keys added to the code after 28 Sep, each at the value that reproduces the old behaviour: `model.sigma_w` 0,
  `trainer.optimizer` adam, `momentum` 0, `iter_offset` 0, `paths.transfer_from` null, and nine `prune_args` keys that
  are inert with `prune_reinit: false`.

The task block is identical (`seed: null` in both; CDDM's batch is the full condition grid, so the seed draws nothing).
The difference that remains is the code version: the controls ran at the code of 28 Sep, these at the submission
commit. The check is repeated on the real configs at the read (Check 3).

### Controls and read-out

- **Controls, not retrained:** M09's three default CDDM networks of N = 1,000,
  `CDDM_paper_grid/EqType=h_N=1000_arm=control` (seeds 3833537590, 1272763220, 457853): 342, 334 and 333 active units at
  75,000 iterations; held-out R² 0.8312, 0.8350 and 0.8349 (mean 0.8337). Source: `data/read_uniform_cells.json` (main
  checkout; json is gitignored), the read behind M09's CDDM values.
- **The controls' R² is an estimate.** They kept no weights at 75,000 iterations, so M09 estimates their R² as
  1 − (L(75,000)/L_end)(1 − R²_end) from the noisy training loss and the final weights (M09, "Measurement and control
  networks"; `fig_paper_F2_fails.read_fixed`, `r2_how = "estimate"`).
- **The new networks' R² is measured,** by M09's first read: the held-out R² of the weights saved at 75,000 iterations,
  36 noise draws. Those are the runs' final weights. This is the same function (`fig_paper_F2_fails.read_fixed`), which
  takes the measured branch when weights at 75,000 exist (`r2_how = "final"`). M09 uses the estimate only where no
  weights were saved. For scale: the controls' estimate at 75,000 is 0.0007 to 0.0022 below the measured R² of their
  final (100,000-iteration) weights.
- **Active count:** at the participation probe nearest 75,000 (the arms store vectors every 100 iterations to 74,900).
- **Gate on the read:** the reader must reproduce the controls' 342, 334, 333 active units exactly and their R² to four
  decimals before any arm's verdict is computed.
- **Counted networks (M09):** every network that reached 75,000 iterations. A network stopped by the rollback guard did not
  finish and enters no mean. Skipped updates (the log's `gradient spikes` line) are reported per network.

### Pass criterion, fixed now

An arm (one λ) **passes** on CDDM when both hold at 75,000 iterations:
1. every finished network of the arm has more than 342 active units, the largest control count;
2. the mean held-out R² of the arm's finished networks is at least 0.8337 − 0.01 = 0.8237.

An arm with no finished network fails. The verdict is `fig_paper_F5_tasks.rule(active, r2, ctrl_active, ctrl_r2, 0.01)`.

**Reported under both of M09's rules for a technique run at several values:**
- *one-value rule*: the term meets the criterion on CDDM if it passes at λ = 0.01 or at λ = 0.1;
- *matched-values rule*: each value run on all three tasks is judged on its own. Both values were run on 3BFF and on Maze
  at N = 1,000, so each of λ = 0.01 and 0.1 is reported as passing or failing on CDDM, beside its 3BFF and Maze verdicts.

### Descriptive read-outs, not pass conditions

- Per arm: active counts, the fold over the controls' mean count (336), the R² difference from 0.8337.
- **Figure:** active units against iterations for every network of both arms and the three controls (participation
  traces, to 75,000).
- **Still falling?** Per network, the median active count over the 2,000 iterations before 50,000 and before 75,000, as on
  3BFF. A network is "still falling" if the second median is below the first. Reported for the controls as well.
- **Manipulation check:** median over units of |H_i − 0.75| on the held-out batch with the training noise (noise seed
  20261121), at 75,000 iterations for the arms. The controls have no 75,000-iteration weights, so theirs is on their final
  (100,000-iteration) weights and labelled so.
- **Revival:** per network, the number of units inactive at some stored participation vector after iteration 1,000 and
  active at the read probe.

### Checks, thresholds fixed now

2. **Reads are checked per network by a separately written script:** the active count at the read probe recounted from the
   stored participation vector with numpy (exact equality, every network, controls included); the noise-free held-out R²
   of one network per arm recomputed with a numpy forward pass of the h equation (within 10⁻⁵ of the torch pass on the same
   weights).
3. **Config diff at the read:** each finished network's saved config against control seed 3833537590's; pass: only the
   differences listed under Check 1, with the real seed, N = 1,000 and `max_iter` 75,000.
