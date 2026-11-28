## 2026-11-25 13:11 — Rules: training noise × weight decay, and a penalty switched on in a trained network (CDDM)

Written, committed and pushed before any run. Branch `wp/mechanism-tests`. Two reviewers of the paper proposed the same
mechanism (firstmate `data/review-directions/report.md`, points N1, N2, C1, C3; `data/review-referee/report.md`, point 2.5).
Their numbers are one reviewer's estimates and are treated here as hypotheses.

**The account under test.** Every default network is trained with Adam and weight decay 1e-6 added to the gradient (coupled
decay). Adam divides each weight's step by the running size of that weight's own gradient. A weight whose loss gradient is
near zero is therefore shrunk fast by the decay alone; the reviewer estimates e-fold (to 37%) in 345 to 496 iterations for a
weight from an active unit onto an inactive CDDM unit. Decoupled decay of the same size would shrink a weight by 0.0035% in
75,000 iterations. So if decay of 1e-6 changes the number of active units at all, the change comes through Adam's rescaling.

**Terms.**
- *CDDM*: the context-dependent decision-making task (450 training conditions, all in every batch). *3BFF*: the 3-bit flip-flop.
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass over the 450 training conditions; active if
  p ≥ 0.05 × the 95th percentile of p. *Silent*: p < 1e-6. *Low-active*: neither.
- *Noise*: the training noise σ_rec = σ_inp = 0.05, σ_out = 0.03 (default), or all three 0.
- *M*: the number of active units of 1,000, on the final weights of a run.
- *Held-out R²*: `heldout_r2.heldout_r2`, 392 new conditions, 36 noise draws, CPU. *M09*: the paper's criterion for a training
  technique; its R² tolerance is 0.01.
- *Set point*: `Penalties.entropy_setpoint_penalty` (reads the net input). *Rate penalty*: `Penalties.fr_magnitude_penalty`
  (reads the rate, so a unit that is below zero on every step gives it no gradient).

### What was already known when these rules were written

- 3BFF networks without coupled decay end with about 200 active units of 1,000 (203-226 with AdamW, 208-221 with decay 0, against
  169-203 with it, 200,000 iterations; `2026-10-14_1301`). CDDM networks without decay: 442-469 at 75,000 against 325-345 with it.
- CDDM networks trained without noise had fewer active units (146, 157, 159) than noisy ones (329, 329, 358) at 75,000
  (`2026-11-02_0058`). The reviewer worked out the sign of the account's prediction after reading that result.
- Under the training noise, silent CDDM units are above zero on about 1.6% of steps (`2026-11-01_0716`).
- From scratch on CDDM, N = 1,000: set point 0.1 gives 539-566 active units (`2026-11-24_1848`), rate penalty 0.1 gives 1,000.

### Where I change the reviewers' designs, and why

1. **C1 reused six old networks and trained six.** Here all twelve are trained with the same code and settings, with three new
   seeds (1153179695, 2873231781, 2425511753), the same three in every cell. The noise deficit at decay 1e-6 is then replicated
   on networks nobody has looked at, and it becomes a gate: if it does not replicate, there is nothing to explain.
2. **C1's bars are kept (10% and 25%) and a per-seed condition is added**, because three seeds can agree in the mean and
   disagree in pairs.
3. **C1 supported would not show that decay is why most units are inactive.** It shows only that decay is the route by which
   removing the noise lowers M. Without decay the older CDDM networks still have more than half of their units inactive. So a second
   bar, S, asks how much of the inactivity decay accounts for, and the two answers are reported separately.
4. **A fixed iteration compares the arms at different stages:** the noise-free loss falls to about 1e-6 and has no floor
   read-out (`2026-11-02_0058`), so no matched-loss read exists. 75,000 is the primary read; the same ratios at 25,000 and 50,000
   are reported without a verdict.
5. **C3's 50% bar for the set point is above what the term can do.** Networks trained with it from the start have 548 active units
   on average against 336, so full recovery would be 208 of about 664 inactive units, 31%. A 50% bar could only fail. The bar
   here is 20% for both penalties, above the arm without a penalty; the reviewer's bars are reported beside it.
6. **C3's premise leaves out the training noise.** The penalty is computed on the noisy batch, where silent units are above
   zero on some steps, so the rate penalty does have a gradient on them. I expect the rate penalty to make most inactive units
   active, against the reviewer's prediction (under 10%). The gradients are measured on the weights at the switch (no training).
7. **C3 does not say what happens to Adam's state.** It is restored in every arm, so the arm without a penalty is the same run
   trained on. A fresh state would give every weight a full-size first step in every arm.

### Part A. Noise × decay

- Four cells, three networks each, N = 1,000, 75,000 iterations: noise {default, zero} × weight decay {1e-6, 0}, Adam.
  Config `configs/experiment/mech_noise_decay.yaml`; launcher `slurm/mech_noise_decay_della.slurm` (tasks 1-12), which sets the
  seed, the three σ and the decay and nothing else. Snapshots every 500 iterations. Independent initial draws (the current default).
- **Read:** M on the final weights.
- **Gate:** R1 = mean M(zero, 1e-6) / mean M(default, 1e-6) ≤ 0.75. Otherwise no verdict.
- **R0 = mean M(zero, 0) / mean M(default, 0).**
  - *Supports the account as the explanation of the noise deficit:* R0 ≥ 0.90 and every same-seed ratio above 0.75.
  - *Refutes it:* R0 ≤ 0.75 and every same-seed ratio below 0.90.
  - Anything else is inconclusive. R0 above 1.10 counts as support for this narrow claim only, and a noise effect of the
    opposite sign is then reported as unexplained. Also reported: 1 − ln R0 / ln R1, the share of the deficit that goes away.
- **S = [mean M(default, 0) − mean M(default, 1e-6)] / [1,000 − mean M(default, 1e-6)]**, the share of the default network's
  inactive units that decay accounts for. S ≥ 0.50: decay is the main cause of inactivity on CDDM. S < 0.25: a minor cause, and
  the account as the explanation of why most units are inactive is refuted on CDDM. Between: a partial cause. (The older networks give
  about 0.18, so I expect "minor".)
- **Manipulation check, reported:** the ratio of the median incoming row norm of inactive and active units per cell. The
  account predicts below 0.3 with decay and above 0.5 without.
- Every finished network counts. A network stopped by the rollback guard or by its wall clock did not finish; it is reported.

**If 3BFF and CDDM disagree.** On 3BFF the account as the explanation of inactivity is already refuted by the known numbers:
S is about 0.08 at 75,000 iterations and 0.04 at 200,000. If CDDM is supported (R0) and 3BFF is not, the conclusion is: on
CDDM decay inside Adam is the route by which removing noise lowers M, and it accounts for the share S of the inactive units; on 3BFF
it accounts for less than a tenth; the paper cannot name coupled decay as the mechanism of inactivity in general, and no run here
says why the tasks differ. If CDDM is refuted, the account is refuted on both tasks as the explanation of inactivity. In no
outcome does one task's result overrule the other's.

### Part B. Resurrect or prevent

- Parents: the three default networks of part A at 75,000 iterations. Each is continued for 25,000 iterations with its weights
  and Adam state, three times: `none`; `ent0.1` (λ_ent = 0.1, target 0.75 bits, the term unchanged); `rate0.1` (λ_frm = 0.1, the
  config's `frm_args`). Config `configs/experiment/mech_switch.yaml`; launcher `slurm/mech_switch_della.slurm` (tasks 1-9).
- **Followed units:** those inactive on the parent's final weights. *Resurrected*: inactive there and active on the final weights at
  100,000. Reported beside it: how many of these were active at half or more of the stored vectors of the last 2,000 iterations.
- **Share** f = resurrected / inactive at the switch. **Net share** = f(arm) − f(none), per network.
  - *Resurrects:* net share ≥ 0.20 in all three networks. *Does not resurrect:* net share ≤ 0.10 in all three. Else *partial*.
  - The same shares for the silent and the low-active units at the switch are reported; the zero-gradient account is about
    the silent ones. **It is supported** if the set point resurrects and the rate penalty's net share of the silent units is ≤ 0.10
    in all three networks. **It is refuted** if the rate penalty's net share of the silent units is ≥ 0.30 in all three
    (the reviewer's bar), or if the set point's net share of the silent units is ≤ 0.10 in all three.
  - The threshold moves with the network's 95th percentile, so f is also given with the bar fixed at its value at the switch.
- **Performance:** mean held-out R² of the arm minus that of `none`; within tolerance if ≥ −0.01.
- **Removal** (`mech_removal.py`): each resurrected unit's outgoing weights are set to zero, alone, nothing retrained; the fall in
  held-out R² over 4 noise draws, the same draws for the intact and the cut network. A unit's removal **costs performance** if
  the mean fall exceeds 1e-4. Reported beside it: the same for units active throughout (positive reference) and for 100 inactive
  units (null). The bar comes from one older control (seed 457853): removing each of 100 inactive units lowered R² by 2.8e-5 at
  most, median 9e-8, so the bar is 3.5 times the largest null value. The active units of that network had not been measured when this
  was written.
  - *The resurrected units contribute:* at least half of them cost performance singly in all three networks, **and** removing all of
    them at once lowers R² by more than 0.01 in all three. *They do not:* fewer than 10% cost performance and the group fall is
    below 0.01, in all three. Else *mixed*. Group falls of as many randomly chosen kept units and of as many inactive units are
    reported (3 draws each).
- **Gradients at the switch** (`mech_read.py grad`, no training): each penalty's gradient on the incoming weights of silent,
  low-active and active units, on a noise-free and on a noisy batch.

### Part C. Optimizer evidence, no GPU (`mech_optimizer.py`, written without the reviewers' code)

Data that existed before this task: Adam states of the CDDM controls (N = 1,000, 2,000, 4,000), CDDM AdamW, 3BFF Adam and AdamW,
CDDM noise-free and default (`CDDM_absorb_noise`); snapshots every 500 iterations of `CDDM_absorb_noise`,
`CDDM_c1_snapshots` (control) and `NBitFlipFlop_c1_snapshots` (control). Blocks are named post-pre: IA is a weight onto an
inactive unit from an active one. After part A the same is run on its twelve networks.

1. **Replication of the reviewer's estimates** (CDDM controls, N = 1,000): gradient-size ratio AA/IA 28-52; step ratio IA/AA
   0.15-0.34 (over all nine controls); predicted decay time (√v + ε)/(lr·1e-6) AA 13,000-18,500, IA 345-496, II 31-54; row
   ratio 0.17-0.20. Replicated if each of my three values lies within the stated range widened by 10%.
2. **Is the predicted decay time the measured one?** Measured: −Δ / ln b, b the slope of w(t+Δ) on w(t) through the origin over
   IA weights whose two units keep their class, Δ = 500 and 1,000, t = 25,000, 50,000, 69,000, default networks. *Supported*: the
   median over the three networks and three times is within a factor 3 of 345-496 (115 to 1,500 iterations). *Refuted*: above
   5,000. Else *right direction, wrong size*.
3. **Noise-free weights decay faster.** Predicted decay time of IA weights in the noise-free networks below half of that in
   the default ones, and the measured one too. Refuted if either ratio is ≥ 1.
4. **N2, weight change.** Median norm of the change of a unit's incoming weights over 500 and 5,500 iterations from the Figure 1d
   snapshots, active against inactive, beside the same ratio of the loss-gradient norm. N2 says the steps differ far less than the
   gradients. Supported if the weight-change ratio is ≤ 10 and at most a third of the gradient ratio, in all three CDDM networks;
   refuted if the weight-change ratio is ≥ half the gradient ratio in all three. 3BFF is reported without a bar.

### Runs, budget, chain

- **Cluster:** Della gpu-short, `--constraint=nomig`, clean clone `~/trainRNNbrain_mech` pinned to the commit named in the
  submission entry. The connection to Della may drop for days, so the chain is submitted at once with dependencies: part A tasks 1-6;
  tasks 7-12 after them; part B after those; then CPU read jobs that write their JSON under `trained_RNNs/CDDM_mech_reads/`. Every choice of
  a later step is fixed above.
- **Cap 80 GPU-hours, enforced by the jobs** (`slurm/mech_budget.sh`): a task starts only if the hours used, plus the time limits of
  running tasks, plus its own limit (8.5 h in part A, 3 h in part B) stay within 79 h. Expected use: 12 × 3.3 h + 9 × 1.2 h ≈ 51.
  Part A's limit is twice the slowest earlier run (4 h 04 min). A task refused by the cap is not resubmitted.
- **Smoke test, done:** both launchers with `module`, `conda` and `srun` stubbed at N = 50 (`firstmate/data/mechanism-tests/smoke`).
  Part A's saved configs differ from the default cell's only in the tag, the three σ and `weight_decay`. Part B's `none` differs from its parent
  only in the tag, `init_from`, `init_adam`, `iter_offset`, `lambda_ent: 0` and `ent_args`; the other two differ from `none` only
  in the tag and their one λ. (`max_iter` is the smoke value in both.) The budget guard refuses a task above the cap.

### Independent recount, thresholds fixed now

`mech_recount.py`, numpy only, no import from this repository (own CDDM conditions, own forward pass, own file reading): (a) M of
every finished network on its final weights, within 2 units of the reader's; (b) the active count of the last stored
participation vector, recounted exactly; (c) resurrected units per part B network, within 2; (d) noise-free R² on the 392
held-out conditions (own generator) within 1e-4 of the reader's; (e) the gate, R0, S and the
part B shares recomputed from its own counts give the same verdicts. A disagreement is traced to its cause before a verdict is reported.

### Out of scope

The paper's text, changes to the entropy term, other tasks or sizes.
