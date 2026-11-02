## 2026-11-01 21:19 — Absorbing-boundary hypothesis: plan, predictions and pass/fail rules (set before any number is computed)

Work package absorb (paper push), trainRNNbrain branch `wp/absorb` from main a8bdb68. Nothing below
has been computed yet; the inventory was done by listing Della directories on 1 Nov, not from an old
cell list.

### The hypothesis being tested

Pavel's proposal, in his words: "the number of silent units grows as number of iteration progresses.
Perhaps there is a way to show that this remind a process of multiple units independently in some
force field, but there is a zero-gradient-boundary, and stochasticity drives some of these units over
that boundary, and the units never recover from beyond this boundary."

It has three separable parts, tested separately:
1. **Absorption.** A unit that has crossed the boundary does not come back.
2. **Stochastic crossing.** Units reach the boundary through fluctuations (diffusion), not through a
   smooth deterministic approach (drift), and they cross independently and throughout training.
3. **Noise sets the rate.** More gradient noise means faster crossing, once loss progress is
   accounted for.

The work package tests the hypothesis; it does not argue for it. Each part passes, fails or is
inconclusive on its own rule below.

### The per-unit distance to the boundary

For unit i at a weight snapshot, **d_i = the largest value of its state x_i (the net input current
of the h-equation, the quantity whose sign defines "silent") over every step t ≥ 1 of every trial of
the noise-free reference batch.** Step 0 is excluded because x = 0 there in every trial (decision
dead-relu-paper#59). d_i ≤ 0 is exactly the paper's definition of a silent unit: its rate is zero on
every step, so on that batch the loss gradient with respect to its incoming and outgoing weights is
zero (C4). d_i > 0 means the unit is above zero on at least one step. The boundary is d = 0. d is in
the units of x.

Reference batch: CDDM, the task's fixed 450 trials (`get_batch` is deterministic); DMTS and 3BFF
(3-bit flip-flop), the batch of task seed 12345, as in C4.

**What the boundary means on 3BFF.** C4 found no silent 3BFF unit: the inactive 3BFF units
(participation below 5% of the network's q95) have d > 0, and their incoming weights get a nonzero
gradient. So on 3BFF the zero-gradient boundary is not where units go inactive. The 3BFF networks
are measured anyway, for two things: whether any 3BFF unit crosses d = 0 at any snapshot, and
whether inactivity (participation below the 5% bar, a boundary without zero gradient) is also
absorbing. If it is, absorption does not need a zero-gradient boundary.

### Networks (inventory of Della `data/c1_snapshots/`, 1 Nov)

Only networks with kept snapshots at many iterations can give d over training. Standard network
(ReLU, equation h, bias fixed at 0, wd 1e-6, Adam, N = 1,000) in every case:

| Set | Networks | Snapshots |
|---|---|---|
| CDDM control (`CDDM_c1_snapshots/arm=control`) | 3 (s1423687489, s2657899762, s2670295995) | every 500 iterations, 500-99,500 |
| CDDM input-weight scale 0.5, 2, 5, 20 (`arm=winp*`) | 12 | every 500, 500-49,500 |
| CDDM revival screen, no-revival arm (`CDDM_screen3_c75000/pen=none_arm=none`) | 3 | every 500, 30,500-74,500 (continuations of 30k runs) |
| DMTS control (`DMTS_c1_snapshots`) | 3 | every 500, 500-149,500 |
| 3BFF control, three lineages joined along `init_from` | 3 (s2670297180→s1258717025; s1542248578→s2461724787→s3194339211; s2851654641→s4113307577→s3204992658) | every 500, 500-199,500 (two lineages to 299,500) |
| New: CDDM noise-free and default-noise pairs (below) | 6 | every 500, 500-75,000 |

The other snapshot trees on Della are per-unit-rule networks (rate penalty, noisy synapses,
rescaling, duplication, dropout, transfer) and are not standard; they are left out. Only the
CDDM control and revival-screen networks carry the default noise and input scale; the input-scale
networks are standard in every other respect and are used for the absorption and crossing tests
(P1, P2), not for the noise test.

Every network is read at its own noisy-loss floor crossing t* (`common.read_run`, 1.05×) where a
time point is needed. A network without a read-out is reported and the fallback stated in each
prediction is used.

### New training (P3b): removing all training noise

The only sources of randomness in CDDM training are the three noise terms: σ_rec = σ_inp = 0.05 on
the state and inputs, σ_out = 0.03 on the rates before the read-out. The trial batch is the same 450
trials every iteration and dropout is off. Setting all three to 0 makes the gradient at each
iteration a deterministic function of the weights: Adam then follows a deterministic path.

- Arms: `noise=zero` (σ_rec = σ_inp = σ_out = 0) and `noise=default` (config values), each with the
  three CDDM control seeds (1423687489, 2657899762, 2670295995), so each pair starts from the same
  initial weights. 75,000 iterations, snapshot every 500, N = 1,000, everything else the CDDM
  control's launcher settings.
- Cost: the CDDM control ran 3.9 h per 100,000 iterations; 6 runs × 75,000 ≈ 18 GPU-hours.
- Launcher `slurm/absorb_noise_della.slurm`, smoke-tested locally first at N = 50 for 30 iterations
  with `srun` stubbed; pass condition of the smoke test: the two arms' saved configs differ in
  exactly the three σ keys plus seed-independent tags.

### Predictions, read-outs and pass/fail rules

**P1. Absorption (CDDM and DMTS; any network with at least 50 silent units at its crossing).**
For each snapshot s, A(s) = (units with d ≤ 0 at s and at every later snapshot) / (units with d ≤ 0
at s).
- *Pass:* in every such network, A(s*) ≥ 0.95 at the snapshot s* nearest t*, and A(s) ≥ 0.90 at
  every snapshot s ≥ 5,000 iterations.
- *Falsified:* A(s*) < 0.80 in any CDDM network: more than a fifth of the units silent at the
  crossing have d > 0 again at some later snapshot.
- Otherwise inconclusive.
- Also reported, without a rule: the share of units silent at s* that are later *active*
  (participation ≥ 5% of q95); for 3BFF, the number of units with d ≤ 0 at any snapshot, and the
  absorption fraction of inactive units under the 5% participation bar (same formula, with
  "inactive" for "d ≤ 0").
- *What would show the hypothesis wrong here:* silent units returning above zero often, for
  example A(s*) of 0.6, which would mean the boundary is a revolving door.

**P2. Stochastic crossing rather than deterministic drift** (CDDM and DMTS networks; units that
cross d = 0 for the first time at snapshot τ_i ≥ 10,500 iterations, so 20 snapshot intervals
precede the crossing).
- (a) *Straightness* S_i = |d(τ) − d(τ − 20 intervals)| / Σ|Δd| over those 20 intervals. A pure
  random walk gives about 1/√20 ≈ 0.22; a steady approach gives 1. *Diffusive* if the median S over
  a network's crossing units is below 0.4; *drift-like* if above 0.6; between, inconclusive.
- (b) *Lag-1 autocorrelation* ρ1 of the increments Δd after subtracting each unit's own mean
  increment, pooled over the network's crossing units. Uncorrelated if |ρ1| < 0.2; smooth
  (drift-like) if ρ1 > 0.4; between, inconclusive.
- (c) *Crossings spread over training:* at least 10% of the units silent at the last snapshot first
  crossed after t*.
- (d) *Independence:* the number of first crossings per 500-iteration interval after t* has a
  variance-to-mean ratio below 2 (Poisson-like). Above 5 means crossings come in bursts; between,
  inconclusive.
- *P2 passes* if (a) and (b) are diffusive/uncorrelated and (c) holds in at least two thirds of the
  networks. *Falsified* if (a) is drift-like in at least two thirds of the networks. (d) is reported
  with its own verdict.
- *Noise-free check:* if the noise-free networks of P3b show the same (a) and (b) as their paired
  default-noise networks (median S within 0.1), the irregular increments are not caused by the
  training noise, and the "diffusive" reading of (a) and (b) does not support stochastic crossing.

**P3a. Noise sets the rate: the existing recurrent-noise ladder** (`CDDM_77fbd9_g0_noise_ptrack`,
σ_rec = 0, 0.01, 0.05, 0.1 with σ_inp = 0.05 and σ_out = 0.03 fixed; 5 networks per value trained to
200,000 iterations, and 5 more to 75,000). No snapshots; the participation trace stores every unit's
participation every 100 iterations on a noise-free pass of the training batch, which on CDDM is the
reference batch, so a unit with participation below 1e-6 there is silent.
- *Read-out:* the slope of the silent-unit count against iteration, in units per 10,000 iterations,
  fitted by least squares over [t*, t* + 50,000] (200k runs) or [t*, 75,000] (75k runs). *Loss
  progress is accounted for* by starting each window at the network's own floor crossing, where its
  noisy loss is within 5% of its fitted floor; the smoothed loss change across the window is
  reported beside each slope. The same slope for the active count is reported.
- *Pass:* across the 20 networks of the 200k set, Spearman ρ between σ_rec and the slope ≥ 0.4
  (one-sided p < 0.05), and the mean slope at σ_rec = 0.1 above the mean at σ_rec = 0.
- *Falsified:* ρ ≤ 0, or the mean slope at σ_rec = 0 at least as large as at 0.1.
- Otherwise inconclusive. The 75k set is read the same way as a replicate.
- Caveat stated in advance: σ_rec also changes the task solution (noise robustness) and lets silent
  units go above zero on the noisy training pass; P3b removes those confounds by removing all noise.

**P3b. Noise sets the rate: the noise-free pairs.**
- *Read-out:* per pair, the slope of the silent count (d ≤ 0 at the snapshots) over [t*, 75,000],
  and the number of first crossings after t*. Fallback if either arm has no floor read-out: the
  fixed window [30,000, 75,000] for both arms.
- *Pass:* the noise-free slope is at most 0.5 × its paired default-noise slope in all three pairs.
- *Falsified:* the noise-free slope is at least 0.75 × the paired slope in at least two of the three
  pairs (silencing proceeds without any noise).
- Otherwise inconclusive.

**P4 (optional). Shape of the curve.** For each CDDM network, the survival curve F(t) = share of the
units with d > 0 at t0 = 5,000 iterations that have not yet reached d ≤ 0 by snapshot t. Three
models, fitted by least squares to F at every snapshot after t0:
- DD: first passage of Brownian motion with drift, started at each unit's own d_i(t0), two global
  parameters (drift μ, diffusion σ); F is the mean over units of the survival probability
  1 − Φ((−a + μτ)/(σ√τ)) − exp(2μa/σ²) Φ((−a − μτ)/(σ√τ)), with a = d_i(t0), τ = t − t0 and μ > 0
  toward the boundary.
- EXP: F∞ + (1 − F∞) exp(−τ/T), two parameters.
- STR: F∞ + (1 − F∞) exp(−(τ/T)^β), three parameters.
- *Criterion:* BIC with Gaussian residuals; a model wins in a network if its BIC is lower than both
  others by at least 10. *P4 supports DD* if DD wins in at least two thirds of the CDDM networks;
  *does not support* if EXP or STR wins in at least two thirds; otherwise inconclusive. Snapshot
  residuals are correlated in time, so BIC overstates the evidence; P4 carries less weight than
  P1-P3.

### Overall verdict rule

- *Supported:* P1 passes, P2 passes, and P3b passes (P3a not falsified).
- *Partly supported:* P1 passes and at least one of P2 or P3 is not falsified; the verdict names the
  parts that pass.
- *Not supported:* P1 is falsified, or both P2 and P3b are falsified.
- The draft Discussion paragraph is written only for a supported or partly supported verdict, and
  only for the parts that pass, labelled as interpretation.

### Oracle checks before any prediction is read

The script computing d is written independently of `RNN_torch`, `c4_gradient_mechanism.py` and the
figure scripts: its own numpy Euler pass of the h-equation from the snapshot's parameter file. The
task class builds the input batch.
- (o1) On the Fig. 1d CDDM weights (it25000, it26000, it24000): silent counts 310, 288, 253 and
  active counts 430, 416, 438 (entry 2026-11-01_1021), exactly. Any mismatch stops the analysis
  until it is explained.
- (o2) On the 3BFF Fig. 1d weights (it67000 of s2670297180, it50000 of s1542248578, it58000 of
  s2851654641): active counts 269, 342, 287 within ±5, and 0 silent.
- (o3) On the CDDM control networks, at every snapshot, the units with d ≤ 0 against the units with
  participation below 1e-6 in the production participation trace at the nearest stored iteration
  (a torch forward on the same 450 trials, written by different code): agreement in at least 99% of
  units at every snapshot, the residual attributed to the one-iteration offset between the trace
  probe and the snapshot.
