## 2026-11-20 16:43 — Rules: does projecting W_rec back toward its initial dynamical regime keep units active?

Written and committed before any measurement of this entry is computed and before any run is
submitted. Literature and how each rule differs from prior work: `2026-11-20_1641_criticality-projection-literature.md`.
Branch `wp/criticality-proj`. Pavel's idea, via firstmate (20 Nov): good initialisation lets gradients
propagate through time, training degrades that property, so project W_rec back toward where it
started, as a form of homeostatic plasticity, and see whether fewer units become inactive.

**Terms.**
- *3BFF*: the 3-bit flip-flop task. *CDDM*: context-dependent decision making.
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass; active if p ≥ 0.05·q95(p).
- *Floor crossing* t*: the first iteration from which the smoothed noisy training loss stays at or below
  1.05 times its fitted floor until 1.2·t* (`common.read_run`, `docs/AGENT_GUIDE.md`).
- *Step Jacobian*: J_t = (1 − α)I + α W_rec D_t, D_t = diag(1[h_t > 0]), α = 0.1, for the standard
  network's update h_{t+1} = h_t + α(−h_t + W_rec relu(h_t) + W_inp u_t). An inactive unit zeroes its
  column of W_rec D_t (in the r-equation form D_t W_rec it is the row).
- *λ1*: the largest finite-time Lyapunov exponent per step, along the noise-free trajectory of each trial
  of the seeded held-out batch (`heldout_r2.heldout_batch`), after a burn-in of 50 steps; the median over
  trials is the network's value. λ1 = 0 is the edge of chaos; with every unit inactive λ1 = log 0.9 = −0.105.
- *G100*: gradient propagation over 100 steps: the read-out's sensitivity at the last step of a trial,
  W_out[o] ⊙ 1[h_T > 0], pulled back 100 steps through J^T; log10 of its norm relative to the start;
  median over outputs and trials.
- *Init*: the network rebuilt from its saved config and seed (`criticality_read.build_rnn`), checked
  against the first snapshot (cosine of W_rec more than 20 times the cosine for seed + 1).
- Code: `trainRNNbrain/trainer/criticality.py` (measures and projections),
  `trainRNNbrain/experiments_and_analysis/criticality_read.py` (offline read),
  `tests/test_criticality.py` (known answers and numpy checks).

### Phase 1. Is the deterioration there? (gate before any remedy run)

**Networks.** The claim-1 controls with kept snapshots, N = 1,000, the standard network:
- CDDM: `CDDM_c1_snapshots/EqType=h_N=1000_arm=control`, 3 networks, 100,000 iterations, snapshots every 500.
- 3BFF: `NBitFlipFlop_c1_snapshots/EqType=h_N=1000_arm=control` (75,000 iterations), continued to 200,000 in
  `NBitFlipFlop_c1_snapshots_c200000`, 3 networks.

**Weight states.** Init; snapshots at 500, 1,000, 2,000, 5,000, 10,000, 20,000 and 40,000 iterations; the
snapshot at or just after t* (the 500-iteration grid); 75,000 (3BFF: 74,500); 3BFF 100,000, 150,000 and
199,500; and each run's final weights (CDDM 100,000; 3BFF 75,000 and 200,000).

**Measures.** λ1, G100 (also at 10, 50 and 200 steps), the median spectral radius of J_t at 12 operating
points, the fraction of units above threshold, ρ(W_rec), σ1(W_rec), the mean incoming row norm, and the
number of active units on the same pass.

**Gate, per network.** The network has *deteriorated* if, from init to its floor crossing, both hold:
1. its λ1 moves away from 0 by more than 0.01 per step: |λ1(t*)| − |λ1(0)| > 0.01;
2. its gradient propagation changes by more than a factor of e over 100 steps:
   |G100(t*) − G100(0)| > log10(e) = 0.434.

0.01 per step is a factor of e (2.7) in the size of a surviving perturbation over 100 steps (10 time
constants), so the two conditions use the same scale.

**Gate, per task.** The deterioration is there on a task if at least 2 of its 3 networks deteriorated.
- Both tasks pass: Phase 2 runs on both.
- One task passes: Phase 2 runs on that task only; the other task is reported as "no deterioration".
- Neither passes: no remedy run. The results entry reports the measurements, says the premise does not
  hold for these networks, and stops.

λ1 and G100 at 75,000 iterations and at the end of training are reported beside the gate; they do not
enter it.

### Phase 2. The four projections

**Implementation.** `Trainer.wrec_projection_`. The option is off by default (`wrec_projection: None`
in the Trainer, `none` in `configs/experiment/criticality_proj.yaml`), and a run with it off trains
exactly as before (checked before submission, below). When on, it runs every `period` iterations after
the optimiser step, and applies each change with strength η: the change is the η-th power of the
multiplicative change that reaches the target, so η = 1 lands on it.

| Arm | Rule | Target |
|---|---|---|
| `spectral` (a) | W_rec ← c·W_rec | ρ(W_rec) = model.spectral_rad = 1.2, the initial value |
| `singular` (b) | W_rec = U S V^T → U S0 V^T | S0: the singular values of W_rec at the start of the run; U, V are the learned ones |
| `rownorm` (c) | W_rec[i, :] ← c_i·W_rec[i, :] | each unit's incoming recurrent row norm at the start of the run |
| `lyapunov` (d) | W_rec ← c·W_rec, c = exp(−η·λ̂1/α), clipped to exp(±0.05) | λ1 = 0, with λ̂1 measured on a noise-free pass over the first 128 trials of the current batch, burn-in 50 |

**Settings, one per arm for this round** (in the experiment config; the launcher sweeps only
`trainer.wrec_projection`): period 50 iterations, η = 1 (a hard projection), and for (d) a largest
change of 5% per event. A hard projection is the cleanest test of the mechanism: if holding the initial
regime exactly does not keep units active, a softer pull will not either. No arm is dropped and none
added after the literature: (a) is spectral or eigenvalue normalisation with the initial target, tested
as a variant of those methods; (d) differs from gradient flossing (Engelken 2023) in being a scalar
rescale on λ1 rather than a loss term on the top k exponents.

**Runs.** `slurm/criticality_proj_della.slurm` (new; its command is that of
`slurm/SilentReLU_paper_grid_della.slurm`, the launcher of the CDDM paper-grid control and of
prune-and-duplicate on CDDM, plus `+experiment=criticality_proj` and the arm).
- Arms: `control`, `spectral`, `singular`, `rownorm`, `lyapunov`. 3 runs each (`seed=random`), N = 1,000.
- Standard network and trainer (`trainer_ptrack_freshbatch`).
- CDDM: 100,000 iterations (the paper grid's budget).
- 3BFF: 200,000 iterations in one piece (the remedy grid's budget).
- Kept snapshots every 1,000 iterations, a multiple of the period, so every snapshot holds projected
  weights.
- New trees: `$HOME/trainRNNbrain/data/trained_RNNs_crit` and
  `.../c1_snapshots/<Task>_criticality_proj`.
- 30 runs. Della gpu-short (CDDM `--time=08:00:00`, 3BFF `--time=23:00:00`), from a clean clone pinned to
  the commit named in the submission entry.
- Estimate: CDDM 3.9 h and 3BFF 14.5 h per run (the paper grid's measured speeds), so 15 × 3.9 +
  15 × 14.5 ≈ 280 GPU-h plus the projection overhead (≤ 10%): about 300 GPU-h, the budget of this round.

**Controls.**
- *Matched controls*: the `control` arm of this launcher, 3 per task, same commit, same command, only
  `trainer.wrec_projection` differs. New controls are run because the existing ones do not match the
  protocol exactly. The paper's 3BFF controls were trained before 9f10955, so flip-flop runs from that
  code cannot be regenerated exactly, and they ran 75,000 + 125,000 iterations. The CDDM paper-grid
  control used another clone and commit.
- *The paper's controls*, as the technique criterion uses them (values from the existing reads, not
  recomputed):
  - 3BFF: the four default networks `NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none`,
    with 279, 288, 242 and 236 active units at 75,000 iterations and R² 0.9514, 0.9517 and 0.9509 (the
    three with 75,000-iteration weights; mean 0.9513; `data/round18/g1g2.json`).
  - CDDM: `CDDM_paper_grid/EqType=h_N=1000_arm=control`, with 342, 334 and 333 active units at 75,000
    and R² 0.8312, 0.8350 and 0.8349 (mean 0.8337; `data/read_uniform_cells.json`).

**Read-out.**
- *Primary metric*: active units at the floor crossing (`common.read_run`, 1.05×; 1.10× and 1.20×
  reported), with held-out R² there (36 draws, on the kept snapshot within 1,000 iterations, else
  `common.r2_at_floor_estimate`).
- *The paper's technique read-out*: 75,000 iterations (`fig_paper_F2_fails.read_fixed`): active units at
  the probe nearest 75,000 and held-out R² on the `it0075000` snapshot, 36 draws.
- *Stability*: fewer than 5% of updates skipped (the training log's `gradient spikes` line) and a finite
  noisy loss throughout.
- *Criticality*: λ1 and G100 at init, 1,000, 2,000, 5,000, 10,000, 20,000, 40,000 and 75,000 iterations,
  at the snapshot at or after t*, and at the end (`criticality_read.py`).

**Hypothesis, per arm.** Holding W_rec at its initial regime keeps more units active.
- *Count condition at the floor crossing* (primary metric): every arm network has more active units at
  its t* than every matched control at its own t*.
- *The paper's technique criterion* (M09, quoted in `2026-11-07_0100_round18-rules.md`, G2): every arm
  network has more active units at 75,000 iterations than every control network, and the arm's mean
  held-out R² at 75,000 is at least the controls' mean minus 0.01 (one-sided). This is the rule by which
  prune-and-duplicate passes on 3BFF and CDDM, so a pass here means the same thing as for it.
- *Manipulation check*: the arm restored criticality if, at the floor crossing, every arm network has a
  smaller |λ1| than every matched control.

**Verdict, per arm and task.**
- **Passes**: the technique criterion holds against both control sets (matched and the paper's), every
  arm network is stable, and the count condition at the floor crossing holds. A pass against one control
  set only is reported as "control-dependent", not as a pass.
- **Falsified as a remedy**: the technique criterion fails against either control set, or any arm
  network is unstable.
- *What the failure says about the mechanism*:
  - If the manipulation check holds and the count condition at the floor crossing fails, the arm
    **falsifies the mechanism** for its route: keeping |λ1| nearer 0 than in every control did not keep
    more units active.
  - If the manipulation check fails, the arm **does not test the mechanism**: it did not restore
    criticality. It is reported as such and not counted against the idea.
- *A pass on count alone*: an arm that passes the count condition but misses the R² bar is reported with
  its R² gap; it does not pass.

**Maze.** Runs only if an arm passes on both supervised tasks and the budget has room (brief); its
rules would be a new entry written before it runs.

### Checks before and after the runs (thresholds fixed now)

1. **Off is unchanged.** Before submission: the launcher's control command at main (ad5072e) and at the
   submission commit, N = 50, 30 iterations, CPU, `module`/`conda`/`srun` stubbed. Pass: `LastParams` and
   `TrainLosses` identical array by array. Each arm's saved config is diffed against the control's; pass:
   only `trainer.wrec_projection` differs.
2. **Each projection lands on its target**, checked by a script written separately from
   `criticality.py` (numpy only), on every arm network's final weights and on its snapshots at 1,000,
   10,000, 75,000 and the end. The init references are rebuilt from config and seed.
   - (a): |ρ(W_rec) − 1.2| < 2·10⁻³ on snapshots (W_rec is stored in float16) and < 10⁻⁴ on final weights.
   - (b): max over ranks of |s/s0 − 1| < 5·10⁻³ on snapshots and < 10⁻⁴ on final weights.
   - (c): max over units of |n/n0 − 1| < 5·10⁻³ on snapshots and < 10⁻⁴ on final weights (rows with
     n0 > 0).
   - (d): no exact target. Reported: λ1 at each read point from the separate script.
   A failure stops the verdict for that arm until explained.
3. **The reads are checked per network** by a separate script, as in entry 2026-11-20_1600. It recounts
   the active units at each read probe from the saved participation (exact equality), takes skipped
   updates from each Slurm log, and confirms the loss is finite. λ1 is recomputed with an independently
   written tangent propagation on two networks per task; pass: within 10⁻³ of the reader's value.
