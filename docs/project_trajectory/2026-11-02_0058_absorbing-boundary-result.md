## 2026-11-02 00:58 — Absorbing-boundary hypothesis: not supported (P1 and P3 falsified, P2 inconclusive, P4 against the drift-diffusion model)

Follows `2026-11-01_2119_absorbing-boundary-plan.md`, which fixed every prediction, read-out and
threshold before anything was computed. Work package absorb, trainRNNbrain branch `wp/absorb`.

### What was run

- **Script:** `trainRNNbrain/experiments_and_analysis/absorb_boundary.py` (8cf62f3 for every Della
  stage; the report stage on the Mac at the branch head). Its forward pass is its own numpy Euler
  integration from each snapshot's parameter file; it shares no code with `RNN_torch`, the c4
  scripts or the figure scripts.
- **Oracles (all before any prediction was read).** o1: on the Fig. 1d CDDM weights, silent 310, 288,
  253 and active 430, 416, 438, exactly as entry 2026-11-01_1021. o2: on the 3BFF Fig. 1d weights,
  0 silent and active 269, 342, 287, exactly. o3 (d ≤ 0 against participation < 1e-6 in the
  production trace, a torch pass written by other code): agreement at the worst snapshot is
  94-99.7% in the 21 CDDM networks trained with noise and 85-89% in the 3 noise-free networks; at
  the median snapshot it is 96-100%. **o3 misses its 99% bar in 14 of the 24 CDDM networks** (the
  3 controls, 2 of 3 at input scale 2, all 6 at input scales 5 and 20, the 3 noise-free networks).
  Root cause: every one of the disagreeing unit-snapshots, in every network, is a unit with
  0 < d < 4 × 10^−4 that the trace calls silent (participation < 1e-6) and the d ≤ 0 rule does not.
  In the noise-free networks the median disagreeing unit has d ≈ 1e-30, a unit whose incoming
  weights have shrunk to nothing. The verdicts below were re-checked with the boundary at d ≤ 1e-4,
  1e-3 and 1e-2 (post hoc), and with the trace count for
  P3b; no verdict changes.
- **Della jobs.** Training (P3b): array 16628052 (6 × N = 1,000 CDDM, 75,000 iterations, launcher
  `slurm/absorb_noise_della.slurm` at ef5050c, pinned worktree `~/trainRNNbrain_absorb`; command
  `sbatch --chdir=$HOME/trainRNNbrain_absorb --time=08:00:00 --array=1-6 slurm/absorb_noise_della.slurm`),
  3.2 h per run, **19.4 GPU-hours**. Smoke test before submission: the launcher with `module`,
  `conda` and `srun` stubbed, N = 50, 30 iterations, both arms; the two saved configs differ only in
  the tag and the three σ keys (pass condition of the plan). Analysis: arrays 16628199 (24
  networks) and 16630023 (the 6 new networks), job 16628200 (noise ladder), from
  `~/trainRNNbrain_absorbana` at 8cf62f3; **87 CPU-hours**. Outputs on Della `~/absorb/dist/*.npz`,
  `~/absorb/ladder.pkl`.
- **New networks:** `CDDM_absorb_noise/EqType=h_N=1000_noise={zero,default}`, seeds 1423687489,
  2657899762, 2670295995; snapshots every 500 iterations in Della `data/c1_snapshots/CDDM_absorb_noise/`.
  These runs use the current initialisation (`init_draws: independent`), so they do not reproduce
  the `CDDM_c1_snapshots` controls of the same seeds, which used the older shared draw; the
  comparison is within each new pair.
- Figure: [absorb_boundary](../../img/internal_figures/absorb_boundary.svg).

### What d looks like

On CDDM a unit that goes silent drops from d ≈ 0.1 to about d = −0.002 within a few thousand
iterations and stays near that value (Fig. a). At the floor crossing the median silent unit of the
three controls has d = −0.0022 to −0.0025, against a median d of 0.30-0.32 for the active units.
On DMTS the silent units sit even closer to zero (median −0.0004 to −0.0009; active units 2.0-3.6).
On 3BFF no unit has d ≤ 0 at any of 1,592 snapshots (smallest d over all units and snapshots,
2-3 × 10^−5); the inactive units sit at median d = 0.007-0.012 and the active units at 2.3-2.7.

### P1. Absorption — **falsified**

A* is the share of units silent at the floor crossing that stay silent (d ≤ 0) at every later
snapshot. Rule: pass if A* ≥ 0.95 and A(s) ≥ 0.90 from 5,000 iterations; falsified if A* < 0.80 in
any CDDM network.

| Networks | Silent at the crossing | A* | Verdict |
|---|---|---|---|
| CDDM control s1423687489, s2657899762, s2670295995 | 310, 288, 253 | 0.013, 0.000, 0.004 | falsified ×3 |
| CDDM revival-screen no-revival arm s1468600595, s1504400307, s341166437 | 349, 351, 362 | 0.17, 0.12, 0.22 | falsified ×3 |
| CDDM input scale 0.5 s1905926612, s673966819 (s3783738710 no read-out) | 105, 87 | 0.30, 0.28 | falsified ×2 |
| CDDM input scale 2 s1882398748, s3194334622, s95527116 | 87, 79, 71 | 0.00, 0.03, 0.06 | falsified ×3 |
| CDDM input scale 5 s2187712155, s592895343, s632557033 | 88, 110, 80 | 0.00, 0.00, 0.00 | falsified ×3 |
| CDDM input scale 20 s184348896, s2510451149, s3104947826 | 173, 154, 167 | 0.00, 0.00, 0.00 | falsified ×3 |
| CDDM new default noise s1423687489, s2657899762, s2670295995 | 241, 267, 278 | 0.18, 0.19, 0.09 | falsified ×3 |
| CDDM new noise-free (no floor read-out; loss falls to 1e-6) | — | A from 5,000 iterations: min 0.01-0.06 | no read-out |
| DMTS s198893703, s3753404946 | 100, 65 | 0.00, 0.00 | falsified ×2 |
| DMTS s3322309204 | 40 | 0.00 | not tested (< 50 silent) |
| 3BFF ×3 | 0 | — | not tested (no silent unit) |

Post hoc, with the boundary moved to d ≤ 1e-4, 1e-3 or 1e-2, the largest A* in any CDDM or DMTS
network is 0.69 (input scale 20 at 1e-2); every network still fails the 0.80 bar.

**What the return is (post hoc, CDDM controls).** Of the units silent at the crossing, 302, 281
and 243 are still silent 5,500 iterations later (of 310, 288, 253), and at most 1 per network is
active, as C4 found. Over the next 74,000 iterations the norms of their input-weight rows fall to 45-53% of their value
at the crossing and those of their recurrent-weight rows to 66-70% (active units' recurrent rows:
72-76%); their median d rises from about −0.0025 to between 0 and +0.0004, and by +74,000 iterations 156, 165 and 164 of them have d > 0 (Fig. b). They come
back to the boundary, not to activity: at any single later snapshot 0-4 of them are active. On
DMTS the silent units leave silence within about 25,000 iterations (5, 16 and 0 of 100, 40 and 65
still silent at +25,000), and the silent count falls from 40-100 at the crossing to 2-18 at 150,000
iterations while the active count also falls. The zero-gradient region is therefore not absorbing
on either task; what persists is inactivity.

**3BFF with the participation bar as the boundary (reported, no rule).** Of the units inactive at
the crossing, 83%, 83% and 93% are inactive at every later snapshot (to 200,000 or 300,000
iterations); 1, 4 and 5 are active 5,500 iterations later.

### P2. Stochastic crossing — **inconclusive**

Units that first cross at ≥ 10,500 iterations, 20 snapshot intervals before the crossing.
S = straightness (random walk ≈ 0.22, steady approach 1); ρ1 = lag-1 autocorrelation of the
increments; late = share of the units silent at the last snapshot that first crossed after t*;
VMR = variance-to-mean ratio of first crossings per 500 iterations after t*.

| Networks | Crossing units | Median S | ρ1 | Late | VMR | (a) / (b) / (c) / (d) |
|---|---|---|---|---|---|---|
| CDDM control ×3 | 504-516 | 0.24-0.26 | −0.37 to −0.41 | 0.51-0.62 | 1.5-2.0 | diffusive / inconclusive / holds / independent |
| CDDM revival-screen arm ×3 | 133-153 | 0.23-0.26 | −0.39 to −0.42 | 0.37-0.39 | 1.6-2.2 | diffusive / inconclusive / holds / independent, independent, inconclusive |
| CDDM input scale 0.5 ×3 | 284-321 | 0.41 | −0.36 to −0.38 | 0.69-0.72 (2 read) | 0.8-0.9 | inconclusive / inconclusive / holds / independent |
| CDDM input scale 2 ×3 | 478-509 | 0.57-0.60 | −0.35 to −0.43 | 0.82-0.84 | 1.6-1.8 | inconclusive / inconclusive / holds / independent |
| CDDM input scale 5 ×3 | 559-601 | 0.72-0.76 | −0.31 to −0.37 | 0.82-0.87 | 1.9-2.3 | drift-like / inconclusive / holds / independent or inconclusive |
| CDDM input scale 20 ×3 | 541-544 | 0.77-0.81 | −0.37 to −0.41 | 0.66-0.69 | 0.8-1.1 | drift-like / inconclusive / holds / independent |
| CDDM new default noise ×3 | 439-470 | 0.23-0.25 | −0.39 to −0.40 | 0.46-0.53 | 1.2-1.3 | diffusive / inconclusive / holds / independent |
| CDDM new noise-free ×3 | 596-625 | 0.30-0.35 | −0.06 to −0.13 | no read-out | no read-out | diffusive / uncorrelated / — / — |
| DMTS ×3 | 422-516 | 0.21 | −0.09 to +0.07 | 0.11-1.00 (of 2-18 units) | 29-40 | diffusive / uncorrelated / holds / bursts |

Rule: pass if (a) diffusive, (b) uncorrelated and (c) holds in at least two thirds of the
networks; falsified if (a) is drift-like in at least two thirds. Here (a), (b) and (c) all hold in 3
of 27 networks (DMTS) and (a) is drift-like in 6 of 27, so P2 is inconclusive.

The noise-free check decides how to read (a). The noise-free networks, whose training has no
random input at all, also score as diffusive (median S 0.30-0.35, 0.05-0.11 above their noisy
partners) and as uncorrelated. The straightness test therefore classifies deterministic gradient
descent as diffusive, and the "diffusive" scores do not support stochastic crossing. The training
noise does leave one mark: it makes the increments anti-correlated (ρ1 ≈ −0.4 with noise, −0.1
without), which is jitter of d around a trend, not a random walk.

### P3a. Noise sets the rate: recurrent-noise ladder — **falsified**

Slope of the silent count (participation < 1e-6 in the trace) after each run's floor crossing, in
units per 10,000 iterations; 5 networks per σ_rec; the noisy loss falls by 3-5% across every
window, at every σ_rec.

| σ_rec (σ_inp 0.05, σ_out 0.03) | 200k runs: slope over [t*, t* + 50k] | 75k runs: slope over [t*, 75k] |
|---|---|---|
| 0 | 88 to 95 (mean 92) | 74 to 79 (mean 76) |
| 0.01 | 65 to 71 (mean 69) | 60 to 67 (mean 64) |
| 0.05 | −36 to 23 (mean 6) | −20 to 52 (mean 25) |
| 0.1 | −25 to 33 (mean −5) | −18 to 13 (mean −5) |

Spearman ρ between σ_rec and the slope: −0.89 (200k set, 20 of 20 networks read) and −0.92 (75k
set). The rule needed ρ ≥ 0.4; ρ ≤ 0 falsifies. More recurrent noise means slower silencing, and at
σ_rec = 0.1 the silent count falls after the floor crossing in 6 of 10 networks. The active count
falls under every σ_rec, more slowly with more noise (200k set: 33-42 per 10,000 iterations at
σ_rec = 0, 5-7 at 0.1).

### P3b. Noise sets the rate: noise-free pairs — **falsified**

The noise-free networks have no floor read-out (their loss falls to about 1e-6 and the fitted floor
is pinned at its lower bound), so the pre-registered fallback window, 30,000-75,000 iterations, is
used for both arms of every pair.

| Seed | Silent units gained per 10,000 iterations, noise-free / default | Ratio | Silent at 75k, noise-free / default | Active at 75k, noise-free / default |
|---|---|---|---|---|
| 1423687489 | 54.5 / 45.3 | 1.21 | 774 / 519 | 157 / 358 |
| 2657899762 | 52.2 / 46.2 | 1.13 | 788 / 506 | 146 / 329 |
| 2670295995 | 40.0 / 11.4 | 3.50 | 759 / 376 | 159 / 329 |

Rule: pass if every ratio ≤ 0.5; falsified if at least two of three are ≥ 0.75. All three are above
1. With the trace count (participation < 1e-6) instead of d ≤ 0 the ratios are 1.02, 1.10 and 3.45.
Removing all training noise roughly halves the number of active units at 75,000 iterations and
leaves 1.5 to 2 times as many units silent.

### P4. Shape of the curve — **does not support the drift-diffusion model**

BIC of three models of the share of units not yet silent since 5,000 iterations, 24 CDDM networks:
the stretched exponential wins (BIC lower than both others by ≥ 10) in 19, the drift-diffusion
first-passage model in 1 (input scale 0.5, s1905926612), none in 4. The criterion for "does not
support" (another model wins in at least two thirds) is met. In the control networks the
drift-diffusion model misses the first 20,000 iterations (Fig. f).

### Overall: **not supported**

By the plan's rule, P1 is falsified, so the hypothesis is not supported; P3a and P3b are also
falsified, P2 is inconclusive and P4 favours a different curve. What the data show instead:
1. Silent units do not stay beyond the zero-gradient boundary. They sit within about 0.002 of it,
   and as their incoming weights shrink they return to it; on DMTS nearly all of them leave silence
   within 25,000 iterations. What persists is inactivity: at any snapshot 0-4 of the 250-310 units
   silent at the CDDM floor crossing are active, and on 3BFF, where no unit is ever silent, 83-93%
   of the inactive units stay inactive at every later snapshot.
2. Training noise slows the loss of units. With no noise at all, CDDM networks lose units at least as
   fast and end with half as many active units.
3. Before crossing, the trajectories of d are as irregular without noise as with it, so their
   irregularity is not evidence for stochastic crossing.

### Why there is no Discussion paragraph

The plan allowed a Discussion paragraph only for a supported or partly supported verdict. None of
the three parts of the hypothesis passed. The measured facts in points 1 and 2 above could go into
Results as facts, not as interpretation; whether they belong in the paper is a separate decision
(issue to be opened by the integrator if wanted). One interpretation, not tested here: with Adam,
weight decay on a weight whose loss gradient is zero moves it by a step of order the learning rate,
whatever the decay coefficient (Loshchilov & Hutter, *Decoupled weight decay regularization*,
ICLR 2019), which would explain why the incoming weights of silent units shrink by half over
74,000 iterations and bring them back to the boundary.
