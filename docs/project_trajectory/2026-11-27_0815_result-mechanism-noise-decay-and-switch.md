## 2026-11-27 08:15 — Result: coupled weight decay does not explain why CDDM units are inactive; the penalty switch diverged in three of six networks

Rules: `2026-11-25_1311_rules-mechanism-noise-decay-and-switch.md`, amended by `2026-11-25_1317_amendment-mechanism-removal-bar.md`.
Submission: `2026-11-25_1321_mechanism-tests-submitted.md`. Part C on the older networks: `2026-11-25_1330_mechanism-optimizer-evidence-older-networks.md`.
Training: Della jobs 18376294, 18376295 (part A) and 18376296 (part B), clone at d1b5355. Reads: Della jobs 18376948-18376950
(`mech_read.py`, `mech_removal.py`, `mech_optimizer.py`, `mech_recount.py`, clone at ba5cd89), output in Della
`trained_RNNs/CDDM_mech_reads/`, copied to `firstmate/data/mechanism-tests/results/`. Figure: `img/internal_figures/mech_noise_decay_switch.png`.

### Short answer

The reviewers' account: Adam with weight decay 1e-6 added to the gradient (coupled decay) shrinks the weights of units that
have no loss gradient, and that shrinkage is why units become inactive. **On CDDM the account is refuted** as the explanation of
both things it was offered for.

1. **The noise deficit does not need decay.** Without decay, networks trained without noise have 163 active units against 456 for
   networks trained with the default noise (ratio 0.36). With decay the numbers are 151 against 343 (ratio 0.44). Removing decay
   makes the deficit larger, not smaller.
2. **Decay accounts for 17% of the default network's inactive units** (S = 0.17; the rules call S < 0.25 a minor cause).

The decay does shrink the weights as the account says. The incoming weights of inactive units are 0.15-0.16 times the size of
active units' with decay and 0.60-0.66 times without it (default noise). In noise-free networks with decay they are zero. But the
units are inactive with or without those weights: in noise-free networks without decay, 711-732 of 1,000 units are silent, and
their incoming weights are 0.70-0.73 times the size of active units'. By the rule fixed in advance for the two tasks, the account
is then refuted on both CDDM and 3BFF as the explanation of inactivity (3BFF: S about 0.04-0.08 from earlier runs).

Part B is weakened by a failure I did not foresee. Switching a penalty on in a trained network made 4 of the 6 penalty runs
diverge within their first six updates; 3 of them never recovered. In the networks that stayed stable, the rate penalty made
every inactive unit active again (2 of 2 networks), and the entropy set point did not (1 of 1 network).

### What ran

| part | networks | GPU time | notes |
|---|---|---|---|
| A, noise × decay, 75,000 iterations | 12 of 12 finished | 42.2 GPU-h (3.2-4.0 h each) | no skipped updates, no rollbacks |
| B, penalty switched on, 25,000 iterations | 9 of 9 ran to the end | 9.9 GPU-h (1.1 h each) | 3 diverged (below) |
| total | 21 | **52.2 of the 80 GPU-h cap** | budget guard never refused a task |

Every network counts in the pre-registered verdicts. The rules exclude only a network "stopped by the rollback guard or by its
wall clock"; the three diverged networks were not stopped, they ran all 25,000 iterations.

### Part A: training noise × weight decay (CDDM, N = 1,000)

Seeds in the order 1153179695, 2873231781, 2425511753. M is the number of active units of 1,000 on the final weights. Held-out
R² is under the network's own training noise, so it is the noise-free R² in the noise-free cells.

| noise, decay | M per seed | mean M | silent units | held-out R² | incoming-weight ratio, inactive / active |
|---|---|---|---|---|---|
| default, 1e-6 | 339, 349, 340 | 343 | 278, 424, 480 | 0.834, 0.835, 0.830 | 0.155, 0.151, 0.163 |
| default, 0 | 450, 467, 452 | 456 | 132, 109, 140 | 0.835, 0.834, 0.832 | 0.599, 0.635, 0.662 |
| zero, 1e-6 | 138, 168, 148 | 151 | 815, 794, 794 | 0.993, 0.997, 0.994 | 0.0, 0.0, 0.0 |
| zero, 0 | 166, 168, 155 | 163 | 711, 732, 714 | 0.995, 0.997, 0.991 | 0.731, 0.727, 0.702 |

Verdicts, with the bars of the rules:
- **Gate passes:** R1 = M(zero, 1e-6) / M(default, 1e-6) = 151 / 343 = 0.44, at or below 0.75. The noise deficit of the older
  networks (146-159 against 329-358) replicates on three new seeds.
- **R0 = M(zero, 0) / M(default, 0) = 163 / 456 = 0.36; the same-seed ratios are 0.37, 0.36 and 0.34. Refuted** (bar: R0 ≤ 0.75
  and every same-seed ratio below 0.90). The share of the deficit that goes away without decay, 1 − ln R0 / ln R1, is −0.26.
- **S = (456 − 343) / (1,000 − 343) = 0.17: a minor cause.** The rules: "S < 0.25: a minor cause, and the account as the
  explanation of why most units are inactive is refuted on CDDM." The older networks gave about 0.18.
- **Manipulation check met:** the incoming-weight ratio is below 0.3 in every network with decay and above 0.5 in every
  network without it.
- Decay does not change performance in the noisy networks (R² 0.830-0.835 with and without it).
- Reported without a verdict (rules, design change 4), from the participation vectors stored during training: at 25,000
  iterations R1 0.70, R0 0.57, S 0.08; at 50,000, R1 0.53, R0 0.43, S 0.15; at the last vector (74,990), R1 0.45, R0 0.36, S 0.17.
  The order R0 < R1 holds at every read.

### Part B: a penalty switched on at 75,000 iterations

Parents: the three default networks of part A (339, 349 and 340 active, so 661, 651 and 660 inactive at the switch). Each was
continued for 25,000 iterations with its weights and Adam state: without a penalty (`none`), with the entropy set point at
λ_ent = 0.1 (`ent0.1`), or with the rate penalty at λ_frm = 0.1 (`rate0.1`). *Resurrected*: inactive at the switch, active at
100,000. *Net share*: resurrected / inactive at the switch, minus the same share in the `none` network of the same seed.

| arm | seed | active at 100k | resurrected (lasting) | net share, inactive | net share, silent | R² minus `none` | training after the switch |
|---|---|---|---|---|---|---|---|
| none | 1153179695 | 312 | 4 (4) | | | | no skipped updates |
| none | 2873231781 | 317 | 10 (10) | | | | no skipped updates |
| none | 2425511753 | 324 | 14 (14) | | | | no skipped updates |
| ent0.1 | 1153179695 | 336 | 16 (13) | 0.018 | 0.004 | −0.0004 | diverged at update 4, recovered after 2 rollbacks |
| ent0.1 | 2873231781 | 355 | 9 (8) | −0.002 | −0.002 | −5e19 | **diverged at update 4**, 99% of updates skipped |
| ent0.1 | 2425511753 | 358 | 18 (17) | 0.006 | −0.002 | −3e20 | **diverged at update 5**, 99% of updates skipped |
| rate0.1 | 1153179695 | 1,000 | 661 (661) | 0.994 | 1.000 | −0.006 | 11% skipped, 5 rollbacks, stable at the end |
| rate0.1 | 2873231781 | 267 | 6 (6) | −0.006 | −0.002 | −5e27 | **diverged at update 6**, 98% of updates skipped |
| rate0.1 | 2425511753 | 1,000 | 660 (660) | 0.979 | 0.998 | −0.008 | no skipped updates |

**Pre-registered verdicts, all nine networks counted:**
- ent0.1: **does not resurrect** (net share ≤ 0.10 in all three).
- rate0.1: **partial** (≥ 0.20 in two networks; the third diverged).
- Zero-gradient account (silent units stay inactive because their loss gradient is zero): **refuted**, through the clause "the
  set point's net share of the silent units is ≤ 0.10 in all three". The rate-penalty clause (≥ 0.30 in all three) fails only
  because of the diverged network.
- Performance: **neither arm is within the 0.01 tolerance**, because the arm means include the diverged networks.
- Removal of the resurrected units (amended bar): **mixed** in both arms.

Two of these verdicts rest mostly on diverged networks: "does not resurrect" and the set-point clause of the zero-gradient
verdict each have one informative network out of three. The performance failures come only from the diverged networks.

**Why three networks diverged.**
- The loss rose to 1e22-1e26, or to infinity, at update 4, 5 or 6 after the switch in all three ent0.1 networks and in rate0.1 seed
  2873231781. No `none` network skipped a single update, so the warm start itself is sound.
- I reproduced the divergence on CPU from parent seed 2873231781 with the set point switched on. With Adam's state restored, as in
  the design, the loss reached 6.8e25 at update 4. With a fresh Adam state it reached 1.2e18 at update 9. Restoring the state
  brings the divergence forward; the switch diverges without it. (`firstmate/data/mechanism-tests/diverge/`.)
- At the switch the set-point penalty is 0.24. For about 90% of the units the ratio of mean to standard deviation of the net
  input is below the value that gives 0.75 bits (−0.79; median over units −1.47), so the penalty's gradient raises the net input of
  most units at once. Adam scales each weight's step by the running size of that weight's own gradient. The incoming weights of
  inactive units had almost no gradient before the switch, so they all move by about a learning rate or more per update, in the
  same direction. The likely end point, not measured here, is a network whose activity grows without bound under the training
  noise.
- The trainer's guard could not undo it. The spike guard flags a gradient only after 100 updates of history, which a warm start does
  not carry; the explosive updates came from ordinary-sized gradients anyway. After 500 skipped updates in a row the guard restores
  its "last good" weights, but its health test (the latest loss at or below the median of recent losses) passes on a diverged
  state. In the three networks that never recovered, the guard rolled back 46-49 times and skipped 98-99% of the 25,000 updates.
  Their final weights are a diverged state that trained very little after the switch. Their noise-free R² is 0.91, 0.93 and 0.75,
  but under the training noise their output grows without bound.
- In ent0.1 seed 1153179695 the first rollback, after about 500 skipped updates, restored the weights of update 1 and cleared Adam's state. That
  network then trained stably, but from that point its Adam state was fresh, which is not the registered design.

**Post hoc reading, decided after seeing the results: the six stable networks.**
- Rate penalty (seeds 1153179695 and 2425511753): every unit inactive at the switch became active, 661 of 661 and 660 of 660,
  within about 5,000 updates (figure, panel b). All 1,000 units are active. Held-out R² is 0.006 and 0.008 lower than in the
  `none` network of the same seed. Removing all resurrected units at once lowers R² by 0.88 and 0.78. Removed one at a time, 20% and
  17% of them lower R² by more than 1e-4, against 55% and 41% of the units active throughout. The resurrected share is 0.36 and 0.42 times the share
  of those units, below the amended bar of 0.5. So the networks need the resurrected units as a group, while each one alone matters
  less than an original active unit.
- Set point (seed 1153179695): 16 of 661 inactive units are active at 100,000, 1 of them from the 278 silent units. The share of
  the inactive units that was active rose to 0.20 within 2,000 updates and stayed near 0.17 from 5,000 to 8,000, then fell to 0.02
  by 12,000 (panel b). Removing the 16 units raises R² by 0.001.
- No penalty: the number of active units fell from 339-349 to 312-324 in 25,000 more updates; 4-14 inactive units became active.
- Both stable penalty arms point against the zero-gradient account. The rate penalty, which reads the rate and so has a zero
  gradient on silent units on the noise-free batch, made every silent unit active (net share 1.000 and 0.998). The set point,
  which reads the net input and does have a gradient on them, made 1 of 278 active.

**Gradients at the switch** (`mech_read.py grad`, the three parents, no training). On the noise-free batch the gradient of the rate
penalty and of the task loss on the incoming weights of silent units is exactly zero for 98-99% of them. On the noisy batch none
is zero: under the training noise silent units are above zero on 2.2% of steps, and the rate penalty's gradient on their incoming
weights has a median of 2.5e-5 to 1.05e-4, against 1.2-1.4 for active units. The set point's gradient has about the same size on
silent (1.0-1.3e-3), low-active (0.7-0.9e-3) and active units (1.0-1.7e-3). The rate penalty's gradient on silent units is 10,000
or more times smaller than on active units, and still made all of them active; Adam's per-weight scaling makes a small, consistent
gradient give full-size steps.

### Part C on the twelve part A networks (no bars; the rules ask for the same read as on the older networks)

- Adam-state estimates in the new default networks match the older controls: gradient-size ratio, active to inactive
  postsynaptic unit, 32-46; predicted decay time of weights from active onto inactive units (IA) 416-495 iterations; active onto
  active 15,400-18,200; inactive onto inactive 36-46.
- Measured IA decay time (slope of w(t + D) on w(t), t = 25,000, 50,000 and 69,000, D = 500 and 1,000): median 5,400 iterations in
  the default networks with decay, 5,700 in the noise-free networks with decay. The prediction is 416-495 for the default
  networks and 22 for the noise-free ones, where Adam's running gradient size on these weights is zero and the prediction is set by
  ε alone. The measurement covers only the IA weights that are not yet zero.
- Without decay the IA weights do not shrink: 2 of 18 slopes are below 1 in the default networks, 0 of 18 in the noise-free ones.
- So the part C result on the older networks replicates: the decay time measured from the weights is 11 to 260 times longer than
  the one predicted from Adam's state. The post hoc per-weight bins of entry `2026-11-25_1330` were not rerun on these networks.

### Independent recount

`mech_recount.py` (numpy only, its own CDDM conditions, forward pass and file reading) made 72 checks with 0 disagreements: M of
all 21 networks (equal in every network, threshold ±2), the active count of the last stored participation vector (exact),
resurrected units (±2), noise-free held-out R² (±1e-4), and the part A and part B verdicts recomputed from its own counts. I
also reran the verdict step on the Mac from the copied read files; its text is identical to the Della read.

### Corrections and notes

- The rules entry says the removal bar of 1e-4 is "3.5 times the largest null value", from one network. Across the three older
  controls of the amendment's calibration the largest fall from removing an inactive unit is 2.8e-5 to 6.3e-5, and across the six
  stable part B networks 3.4e-5 to 5.6e-5. So the bar is 1.6 to 3.6 times the largest null value, and 0 of 100 inactive units
  exceed it in any stable network.
- Panel b of the figure draws the diverged runs dashed. That change to `mech_read.draw` was made after the results were read and
  changes no number.
- On the Mac, Python started outside the worktree imports an older editable checkout (`~/Documents/GitHub/trainRNNbrain`) that
  has no entropy set point, and `filter_kwargs` then drops `lambda_ent` without a warning. My first CPU replay ran without the
  penalty for that reason; the replay above sets `PYTHONPATH` to the worktree. The Della runs are not affected: their loss at
  update 1 is 0.050 with the set point against 0.026 without it.

### Not run (possible follow-ups)

1. A stable switch for the set point. A ramp of λ alone would not help, because Adam's step on a weight with no earlier gradient is
   about one learning rate whatever λ is. A learning-rate warm-up after the switch might, but it needs a pilot and its own rules.
   About 3.3 GPU-h for three networks. Today the ent0.1 result rests on one network.
2. The trainer's guard (for its owner; not changed here): carry the gradient-norm history across a warm start, and give the
   snapshot health test an absolute bound, so that a rollback cannot restore a diverged state.
