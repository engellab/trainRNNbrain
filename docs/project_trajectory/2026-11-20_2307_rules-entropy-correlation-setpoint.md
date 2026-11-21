## 2026-11-20 23:07 — Rules: entropy and correlation set points on 3BFF

Written and committed before any training run with either term, the pilot included. Literature and how
the two terms differ from prior work: `2026-11-20_2305_entropy-correlation-setpoint-literature.md`. Branch `wp/entropy-setpoint`. Pavel's question, via
firstmate (20 Nov): would a loss term that holds the entropy of each unit's activity at a set point, or
the correlation between units' activity at a set point, keep units from becoming inactive?

**Terms.**
- *3BFF*: the 3-bit flip-flop task. *N*: number of units.
- *Pre-activation* h_i: the state of unit i in the standard network,
  h_{t+1} = h_t + α(−h_t + W_rec relu(h_t) + W_inp u_t + noise), α = 0.1. Its rate is r_i = relu(h_i).
- *Off on every step*: r_i = 0 at every step and trial of a batch. Any term computed from rates gives
  such a unit exactly zero gradient, because relu has zero slope below 0.
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass; active if p ≥ 0.05·q95(p).
- *H_b(p)*: the binary entropy −p log2 p − (1 − p) log2(1 − p), in bits: 1 bit at p = 0.5, 0 at p = 0 or 1.
- *Φ*: the standard normal cumulative distribution function.
- *Held-out R²*: `heldout_r2.heldout_r2`, 36 noise draws, the training noise, CPU (`docs/AGENT_GUIDE.md`).
- *Technique criterion*: M09 (`dead-relu-paper/methods/M09_remedies.tex`, "Criterion"), stated for 3BFF below.

### The two terms

Both read the pre-activations h of the training pass (training noise on), over all T = 300 steps and
B = 1,024 trials of the batch, the same steps over which the rate penalty reads rates. Code:
`Penalties.entropy_setpoint_penalty` and `Penalties.correlation_setpoint_penalty` in
`trainRNNbrain/trainer/Trainer.py`, entered in the Penalties table as `entropy_setpoint` (`lambda_ent`,
`ent_args`) and `correlation_setpoint` (`lambda_cor`, `cor_args`), λ = 0 by default. Their arguments are
in `configs/experiment/entropy_setpoint.yaml`.

**(A) Entropy set point.** For each unit, μ_i and s_i are the mean and s.d. of h_i over the batch;
p_i = Φ(μ_i/s_i) is the fraction of steps that a Gaussian with these moments spends above 0; H_i = H_b(p_i).

  P_A = mean_i ((H_i − H*)/H*)²,  H* = 0.75 bits.

**(B) Correlation set point.** c_i is the mean, over the other N − 1 units, of the Pearson correlation
between h_i and h_j over the batch, computed in O(N·T·B) from the z-scored h.

  P_B = mean_i ((c_i − c*)/0.4)²,  c* = 0.

### Choices and the reasons for them

The numbers below come from the three control networks with weights at 75,000 iterations (seeds
1558264694, 254798194, 254859059 of `NBitFlipFlop_ff_revive`, continued in `NBitFlipFlop_ff_revive_c75000`),
rebuilt from their saved configs, on the held-out batch (256 trials) with the training noise. Script, kept
outside the repository: `firstmate/data/entropy-setpoint/scripts/explore_stats.py`.

| per network, median over units | init | 75,000 iterations, active units | 75,000, inactive units |
|---|---|---|---|
| active units | 957, 953, 951 | 243, 276, 288 | |
| μ_i/s_i | 0.01 to 0.08 | 0.00 to 0.29 | −2.6 to −2.2 |
| p_i (Gaussian) | 0.50 to 0.53 | 0.50 to 0.61 | 0.005 to 0.013 |
| measured fraction of steps with h > 0 | 0.50 to 0.53 | 0.42 to 0.58 | 0.017 to 0.022 |
| H_i (bits) | 0.92 to 0.94 | 0.748 to 0.757 | 0.046 to 0.101 |
| c_i | −0.0001 to 0.0002 | −0.031 to −0.011 | 0.42 to 0.47 |
| mean pairwise correlation of h, all units | 0.0000 (to 4 decimals) | 0.27 to 0.32 | |

1. **Both terms read pre-activations, not rates.** A term on rates gives a unit that is off on every step
   zero gradient: it could slow units' deaths but never revive one. The brief requires a variant of each
   term that reaches such a unit, and with one variant per term the screen tests that variant directly. No
   rate variant is implemented. `tests/test_setpoint_penalties.py` builds a 30-unit network with one unit
   off on every step and checks that the task loss and each term computed on rates give that unit's
   incoming weights exactly zero gradient, while each term on pre-activations gives a nonzero one (largest
   entry 0.0066 for (A), 0.0039 for (B)).
2. **(A) uses the entropy of the on/off state, through a Gaussian model of h.** The differential entropy
   of a rate distribution is not defined for a unit that is never on, and it depends on the units of r.
   H_b(p_i) depends only on how often the unit is above threshold, which is what makes a unit active or
   inactive; a unit that is always off and one that is always on both have H = 0. The Gaussian p_i is
   differentiable in μ_i and s_i and stays above 0 for a unit that is off on every step. In the controls it
   tracks the measured fraction for active units and underestimates it for inactive ones (table), whose h
   has heavier tails than a Gaussian. The gradient dH_i/d(μ_i/s_i) = log2((1 − p_i)/p_i)·φ(μ_i/s_i) shrinks
   with the Gaussian tail: 0.13 bits per unit of μ/s at μ/s = −2.5 (the median inactive unit), 0.011 at −3.5.
   It is computed through log Φ (`torch.special.log_ndtr`), so p log p stays finite far below threshold.
3. **Target H\* = 0.75 bits**, the median H_i of the active units of the controls (0.751, 0.748, 0.757 bits).
   A typical active unit then pays almost nothing, and the term acts mainly on the units below the target:
   the median inactive unit contributes 0.75 to 0.88. The penalty is two-sided, as the brief asks: a unit
   above 0.75 bits (one on between 21% and 79% of the time) is pulled toward 21% or 79%. The initial value,
   0.92 to 0.94 bits, was not chosen: it would push every active unit of a trained network up as well. The
   nearest prior term, the marginal-entropy term of BRE (Cao et al. 2018, literature entry), holds every unit
   at half on (1 bit); 0.75 bits asks less of the units that are already active.
4. **Target c\* = 0, scale 0.4.** "Bringing the correlation back to a set point" is read as back to its value
   at initialisation, where the median c_i is 0 to within 0.0002. By 75,000 iterations the mean pairwise correlation of h
   is 0.27 to 0.32, and the rise sits in the inactive units (median c_i 0.42 to 0.47, against −0.03 to −0.01
   for active units). The scale 0.4 makes the median inactive unit contribute about 1 (1.1 to 1.4), as a unit
   with zero rate contributes 1 to the rate penalty, so λ_ent, λ_cor and λ_frm are on comparable scales.
5. **What (B) can do, derived here and not measured.** A Pearson correlation does not change when one
   unit's h is shifted or scaled, so the gradient of P_B with respect to the time course of h_i sums to zero
   over steps: (B) does not push the mean of h_i up or down. It can lift an inactive unit only through the
   weights: since every rate is non-negative, the direction along which the population co-varies lies close
   to the direction of its mean rate, so removing a unit's shared drive along that direction should remove
   part of its mean inhibition as well. **Prediction, before any run:** (A) revives units directly; (B)
   revives fewer units than (A), or none. (B) is screened because Pavel asked about it, and because the rise
   in correlation sits in the inactive units.
6. **λ = 0.01 and 0.1 for each term**, the two values allowed. The rate penalty, on the same scale (a unit
   at zero contributes 1), was run on 3BFF at 0.01, 0.03 and 0.1: all three raised the active count 2.7- to
   3.7-fold over the controls and lowered the mean R² by 0.016 to 0.023 (`data/round18/g1g2.json`). The two
   values span that range. Adam divides each parameter's step by the running root mean square of that
   parameter's own gradient; on the incoming weights of a unit that is off on every step the task gradient is
   zero, so the term's gradient sets the step there whatever λ is, as long as it exceeds the weight-decay
   gradient (10⁻⁶ times the weight). We therefore expect λ = 0.01 to revive about as many units as λ = 0.1,
   at a smaller R² cost.

### Runs

- **Launcher** `slurm/entropy_setpoint_della.slurm` (new): the command of
  `slurm/SilentReLU_remedy_grid_della.slurm`, which trained the rate-penalty networks, with
  `+experiment=entropy_setpoint` in place of `silent_units_largeN` and one λ per arm.
- **Arms** `ent0.01`, `ent0.1`, `cor0.01`, `cor0.1`; three networks each (`seed=random`); N = 1,000; the
  standard network and trainer (`trainer_ptrack_freshbatch`); 75,000 iterations; snapshots kept every
  1,000 iterations; participation probes every 10 iterations, full vectors every 100.
- **Initial weights drawn as the controls'.** The control and rate-penalty networks were trained before
  2026-10-15, when every weight matrix reused the same random draws (W_out a rescaled copy of W_inp;
  `tests/test_init_draws.py`); the current default draws them independently. M09 requires a network trained
  with a technique to have its controls' configuration except for the technique's parameters, so the
  experiment config sets `model.init_draws: shared`.
- **Differences from the controls that remain:** the code version (the controls were trained before
  2026-10-15, these runs at the submission commit) and the flip-flop trial generator, which the controls left
  unseeded and the current task config seeds from the run seed. The second changes which trials are drawn,
  not their distribution. A full diff of a control's saved config against an arm's (N = 50 test run) shows
  no other difference in the model, task or trainer settings that the controls record.
- **Output:** `$HOME/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_entropy_setpoint/EqType=h_N=1000_arm=<arm>/`;
  snapshots under `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots/NBitFlipFlop_entropy_setpoint/`.
- **Cluster:** Della gpu-short (`--time=23:00:00`, `--constraint=nomig`), from a clean clone
  `~/trainRNNbrain_entsp` pinned to the commit named in the submission entry. Spock is allowed if it starts
  jobs sooner (firstmate, 21 Nov); the budget is the same.

**Pilot and budget: 100 GPU-hours in total, the pilot included.** Before the screen, `ent0.1` and
`cor0.1` each run for 2,000 iterations (array tasks 6 and 12 with `SWEEP=entropy_setpoint_pilot` and
`ITERS_OVERRIDE=2000`; they enter no result). From each pilot log, the time per iteration over iterations
1,000 to 2,000 gives the cost of one 75,000-iteration run; c is the larger of the two terms' costs plus
0.2 h for start-up and saving. With P the GPU-hours the pilots used:
- 12c + P ≤ 100: all 12 networks;
- else 9c + P ≤ 100: drop `cor0.1` ((B) has the weaker predicted mechanism and λ = 0.1 the larger
  predicted R² cost);
- else 6c + P ≤ 100: λ = 0.01 for both terms only;
- else no screen; the result entry reports the cost.

If a pilot runs out of memory, ends with a non-finite loss, or skips more than 5% of its updates, the
screen does not start and a new entry decides. At the guide's 0.24 to 0.30 s per iteration for N = 1,000,
c would be 5.2 to 6.5 h before the terms' own cost, and 12 networks 62 to 77 GPU-hours.

### Controls and read-out

- **Controls, not retrained:** the four default 3BFF networks of N = 1,000,
  `NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none`: 279, 288, 242 and 236 active
  units at 75,000 iterations; held-out R² 0.9514, 0.9517 and 0.9509 in the three with weights saved at
  75,000 (mean 0.9513). Source: `data/round18/g1g2.json` (main checkout; json files are gitignored).
- **Rate penalty, for description only:** `NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm=frm0.1`: 985, 995
  and 935 active units; R² 0.9329, 0.9304 and 0.9276 (mean 0.9303, 0.021 below the controls' mean) at
  75,000 iterations, same source. It meets the count condition and misses the R² condition by 0.011.
- **Read-out at 75,000 iterations,** as `round18_g1g2.read_run` reads a network: the active count at the
  participation probe nearest 75,000 (`common.load_run`, `read_remedies.nearest_probe`), and the held-out R²
  of the weights at 75,000, which are these runs' final weights.
- **Counted networks (M09):** every network that finished training is counted. A network stopped by the
  rollback guard did not finish and enters no mean. Skipped updates (the training log's `gradient spikes`
  line) are reported per network.

### Pass criterion, fixed now

An arm (one term at one λ) **passes** when both hold, at 75,000 iterations:
1. every finished network of the arm has more than 288 active units, the largest control count;
2. the mean held-out R² of the arm's finished networks is at least 0.9513 − 0.01 = 0.9413.

An arm with no finished network fails. The verdict is `fig_paper_F5_tasks.rule(active, r2, ctrl_active,
ctrl_r2, 0.01)`. A term passes on 3BFF if one of its two λ values passes (M09's one-value rule); both
values are reported.

### Descriptive read-outs, not pass conditions

- Each arm beside the rate penalty at 0.1: active counts, the fold over the controls' mean count, the R²
  difference from the controls' mean.
- The figure: active units against iterations, every network of every arm, with the four controls (their
  participation traces, joined along the warm-start chain).
- **Manipulation check:** at 75,000 iterations, the median over units of |H_i − 0.75| for (A) and of |c_i|
  for (B), in each arm network and in the three controls with 75,000-iteration weights, on the held-out batch
  with the training noise and a fixed noise seed. A term that left its own statistic no nearer its target
  than in every control did not get tested by its run, and is reported as such.
- **Revival:** per network, the number of units that were inactive at some stored participation vector
  after iteration 1,000 and are active at 75,000.

### Checks, thresholds fixed now

1. **Off is unchanged** (done before this entry). The control command of the remedy-grid launcher
   (λ_frm = 0) at main (0c49e04) and at a45b342, N = 50, 30 iterations, CPU, with `module`, `conda` and
   `srun` stubbed and the seed fixed: `LastParams` and `TrainLosses` are identical array by array. Each of
   the four arms' saved configs differs from that control's only in `lambda_ent`, `lambda_cor`, `ent_args`,
   `cor_args`, the tag, the paths and `checkpoint_every` (set to 10 in the check's control run only).
   `tests/test_setpoint_penalties.py` checks the same at the Trainer
   level. The arm diff is repeated at the submission commit; pass: the same differences plus
   `model.init_draws: shared`.
2. **Reads are checked per network** by a separately written script: the active count at the read probe
   recounted from the stored participation vector with numpy (exact equality, every network); the
   noise-free held-out R² of one network per arm recomputed with a numpy forward pass of the h equation
   (within 10⁻⁵ of the torch pass on the same weights).

### Stage 2

If a term passes, a new rules entry for Maze (N = 1,000 and 2,000, three networks per size, the M09 Maze
criterion, the existing default Maze networks of each size as controls, at most 300 GPU-hours) is written
and committed before any Maze run (firstmate, 21 Nov). The terms are not yet in `TrainerPPO`. If neither
term passes, the work stops after the 3BFF result.
