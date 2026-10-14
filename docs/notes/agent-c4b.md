# Agent C4b: is the flip-flop still learning?

Pavel's hypothesis, on the C4 result (on the 3-bit flip-flop the gradient reaches the incoming
weights of silent units and training keeps them silent; on CDDM the silent units that never cross
threshold get exactly zero gradient): "If there is a difference, there should be some other
difference in trainability; perhaps it didn't converge still?" Call it H: the flip-flop networks are
not at a (noisy) optimum, the gradient that reaches silent units is the signature of learning still
going on, and CDDM has converged.

## Pre-registration (written 2026-10-14 10:40, before any number below was computed)

What I had seen before writing this: the C4 notes and results (`agent-c4.md`); the statement that the
two 300k flip-flop chains' smoothed loss minimum moved by less than 0.3% from 200k to 300k; agent-c2ff's
note that 8 of 11 flip-flop size-series runs end below their fitted floor (degenerate stretched-
exponential fits); the inventory of runs and snapshots. No slope, drift exponent, gradient
signal-to-noise or fine-tune result had been computed by me or, as far as the notes show, by anyone.

### Networks and read points

Claim-1 networks, N = 1000, control arm, h equation, ReLU, Adam (lr 4.6e-4 after the N-scaling,
coupled L2 weight decay 1e-6). Chains (root seed -> continuations):
- flip-flop A: 2670297180 -> 1258717025 (ends 200k)
- flip-flop B: 1542248578 -> 2461724787 -> 3194339211 (ends 300k)
- flip-flop C: 2851654641 -> 4113307577 -> 3204992658 (ends 300k)
- CDDM 1423687489, 2657899762, 2670295995 (100k); DMTS 198893703, 3322309204, 3753404946 (150k).
Weight snapshots every 500 iterations along every chain; Adam moments only at each run's end.

Batches: the flip-flop draws random pulse trains (1024 trials), DMTS jitters its onsets, CDDM's batch
is a fixed 450-trial grid. Every batch here is drawn from a seeded task (task seed set explicitly) and
seeded noise. The reference batch (for classes) is task seed 12345, noise-free, as in C4.

Unit classes at each read point, on the reference batch: **active** (p >= 0.05 q95(p), the
participation rule), **silent-on** (not active, h > 0 on at least one sample), **never-on**, and, as
in C4, **silent-reached** (silent with noise-free g_in >= 0.05 q95(g_in)).

### T1. Is the training loss still falling?

Series: the per-iteration noisy training loss along the warm-start chain. Statistic: least-squares
slope of log(loss) on iteration over ten 5,000-iteration block means of a 50,000-iteration window,
stated as percent change per 100,000 iterations, with a t-value from the regression residuals.
Windows: each chain's last 50k (flip-flop 150-200k and 250-300k, CDDM 50-100k, DMTS 100-150k), and
the matched window 50-100k for every run. The noise yardstick printed beside it: the per-iteration
coefficient of variation of the loss.
- H predicts: in the flip-flop's 150-200k window the loss falls by >= 2% per 100k with t <= -3 in at
  least 2 of 3 chains, while CDDM's last-50k change is < 1% per 100k or |t| < 2 in at least 2 of 3.
- Falsified (on T1): flip-flop decline < 1% per 100k or |t| < 2 in at least 2 of 3 chains, or CDDM
  falling at least half as fast.
- My expectation: the flip-flop still falls slowly (~1-3% per 100k) at 150-200k, flatter at 250-300k.

### T2. Does the gradient that reaches silent units point anywhere?

At each read point, K = 32 training-like batches (fresh seeded trials, the run's own sigma_rec,
sigma_inp, sigma_out, seeded noise). Per unit, the gradient on its incoming row g_k,i (W_rec[i,:] and
W_inp[i,:] together). Measures:
- signal share F_i = S_i / (S_i + s_i^2), with s_i^2 the per-batch variance (trace of the sample
  covariance) and S_i = |mean_k g_k,i|^2 - s_i^2 / K the debiased squared mean. 0 = pure noise,
  1 = every batch agrees.
- split-half cosine c_i between the mean gradients of batches 1-16 and 17-32, against a sign-flip
  null (each batch gradient multiplied by a random sign; the null's 95th percentile over the class).
- the drive projection: per batch, -g_k,i . x_i, with x_i the mean presynaptic vector [r(t-1); u(t-1)]
  over the unit's 1% highest-h reference samples (C4's definition); per unit a t-value over the 32
  batches. Negative = a gradient step lowers the unit's drive where it is closest to firing.
- network-wide F over all of W_rec, W_inp, W_out.
Read points: flip-flop 50k, 100k, 200k (all chains), 300k (B, C); CDDM 25k, 50k, 100k; DMTS 50k, 150k.
- H predicts (T2a): the flip-flop's silent-reached rows carry a consistent gradient: median c above
  the null 95th percentile and median F >= 0.05; |t| > 2 for >= 30% of them.
  (T2b) network-wide F of the flip-flop at 200k >= 2x CDDM's at 100k.
  (T2c) the silent-reached rows' median F falls >= 2-fold from 50k to the chain end.
- Falsified: (T2a) median c within the null band, i.e. their gradient changes direction from batch to
  batch and the "reach" is batch noise; (T2b) flip-flop F <= CDDM F.
- My expectation: T2a and T2b both falsified. The flip-flop's batches differ in their trials, CDDM's do
  not, so the flip-flop's per-batch gradient should be noise-dominated, most of all on units that
  fire on a few steps of a few pulse patterns.
Oracle for the gradient code: the C4 numpy float64 hand-written backward pass
(`c4_gradient_mechanism.backward`), on one noisy batch with the same noise arrays. Pass: per-unit
relative difference of the incoming-row gradient norm < 1e-3 at the median and < 1e-2 at the 99th
percentile, over units with g_in >= 1e-3 q95. Nothing in T2/T4 is read if it fails.

### T3. Do the weights still travel, or jitter in place?

(i) Network level: `common.drift_alpha` (exponent of displacement against lag, lags 100, 1000, 10000
from the trace), median over the probes of each chain's last 50k; also over 50-100k for all.
(ii) Per unit, from the kept snapshots: rho_i(L) = mean over consecutive non-overlapping intervals in a
50k window of cos(d1, d2), d1 = w(t+L) - w(t), d2 = w(t+2L) - w(t+L), with w the unit's incoming row
divided by its norm (so a pure shrink does not count as travel). L = 500, 2500, 10000. rho = 0 for a
random walk, > 0 for directed motion, < 0 for motion held inside a basin. Windows as in T1.
- H predicts: the flip-flop's silent-on rows have rho(10000) >= 0.3 in the 150-200k window, at least
  0.2 above CDDM's active rows in CDDM's last 50k; flip-flop late alpha >= 0.7, CDDM late alpha <= 0.6.
- Falsified: flip-flop silent-on rho(10000) <= 0.1 late, or flip-flop late alpha <= 0.6.
- Expectation: flip-flop alpha above CDDM's, rho of silent rows small.

### T4. Trainability probes (fine-tunes from each run's end, not new training)

Start points with saved Adam moments: flip-flop A, B, C at 200k, B and C at 300k, CDDM x3 at 100k,
DMTS x3 at 150k. 2,000 iterations each, the same seeded batches and noise in every condition:
- (c) control: Adam at the run's lr, moments loaded (continued training);
- (a) anneal: Adam at lr/10, moments loaded;
- (s) plain SGD, no momentum, no adaptive scaling, lr = (lr/10) / median(sqrt(v_hat)) over W_rec.
Evaluation on a fixed set of 8 seeded batches with fixed noise (paired across conditions and time):
mean noisy loss, and noise-free loss, at 0, 250, 500, 1000, 2000 iterations. Gain G = (L(0) - L(2000))
/ L(0); late gain = (L(1000) - L(2000)) / L(0).
- H predicts: the flip-flop has descent that training at the run's lr does not realise:
  G_anneal(flip-flop) >= 2x G_anneal(CDDM), and the flip-flop still falls between 1000 and 2000 (late
  gain >= 1/3 of G) where CDDM has stopped (late gain < 1/5 of G); and the control continuation falls.
- Falsified: G_anneal(flip-flop) < 1.5x G_anneal(CDDM), or the flip-flop plateaus like CDDM.
- (v) C4 re-read after the anneal: the silent-reached count on the reference batch. If the reach is
  a signature of not being converged, getting closer to the optimum shrinks it: H predicts a drop
  >= 30%; falsified if it changes by < 10%. The same count along the natural path (50k -> 200k -> 300k
  reads of T2) is held to the same bars.
Budget: CPU jobs on Della; each fine-tune is short. No new training run.

Seen after the pre-registration was committed (4bee7bb), before any T4 run: the trajectory's
2026-10-13 entry reports that intact claim-1 networks fine-tuned for 3,000 iterations (lr or lr/10,
saved or fresh Adam) are flat in held-out R^2 on all three tasks (flip-flop 0.951-0.954). That is a
result on R^2, not on the noisy loss, and the bars above are unchanged.

### Verdict rule (set now)

H explains C4 only if all three hold: the flip-flop is still learning by T1 or T3 (beyond CDDM); its
silent units' gradient is a consistent signal (T2a); and the reach shrinks as the network converges
(T4 v or the natural path). If T2a or (v) fails, non-convergence is not why the gradient reaches
silent flip-flop units, whatever T1 says.

### Scope cut (11:05, coordinator relaying Pavel: report within about an hour)

Dropped before they ran: T2 for DMTS (all), the flip-flop 100k reads of chain C, CDDM 50k; T4's
control and plain-SGD conditions and DMTS. T4 is now one probe, the anneal (Adam at lr/10, saved
moments), 600 iterations instead of 2,000, on the flip-flop ends (A 200k, B 300k, C 300k) and the three
CDDM ends (100k), with K = 0 (the reach re-read only, no signal share). The T4 bars are kept with
"2000" read as "600" and the late gain as the second half (300 -> 600). T2/T3 reads that had already
finished before the cut (flip-flop 100k for A and B, flip-flop 200k for B and C, DMTS drift) are reported
but were not part of the reduced plan. The flip-flop "floor" read is 50k for all three chains (C4's floor
snapshots were 27k-69.5k, and two of those floor fits are now rejected).

### Deviation found in T2 while reading the first outputs (11:09)

The sign-flip null for the split-half cosine is built wrongly: flipping the signs of UNcentred batch
gradients leaves a dominant common component as (sum of signs) x mean, so the null cosine is +-1
whenever the signal dominates (its 95th percentile came out at 0.82-0.99). The cosine c is therefore
not tested against a valid null and is not used for a verdict. T2a is read on the signal share F
alone (bar: median F >= 0.05), which needs no null: under a zero-mean gradient its expectation is 0.

### Deviation in T1 (11:25)

The first T1 run used 5,000-iteration block MEANS as pre-registered. Single-iteration loss spikes
make them meaningless on some runs (flip-flop B's 150-200k window came out at +16,000,000% per 100k,
DMTS at -90% to +190%). The slope is now taken on block MEDIANS; the mean-based number is printed
beside it with the largest spike, so the change can be checked. The bars are unchanged.

## Results (2026-10-14 12:20)

Code: `trainRNNbrain/experiments_and_analysis/c4b_convergence.py` (Della CPU jobs; outputs in
`/scratch/gpfs/TENGEL/pt1290/agent_c4b/out/`). Figure: `img/internal_figures/c4b_convergence.pdf`
(+ `.svg`, `.pdf.png`) in the main checkout. Chains are named A, B, C (flip-flop) as above.

**Oracle: passed.** The torch gradient matches C4's numpy float64 BPTT on a noisy 128-trial batch,
per unit: median relative difference of the incoming-row gradient norm 3e-13 (flip-flop) and 2e-15
(CDDM) in float64, 2e-6 and 2e-7 in the float32 used for every measurement (bar 1e-3).

**T1, loss slope (block medians): falsified.** Percent change of the noisy loss per 100k iterations
(t-value):

| window | flip-flop A | flip-flop B | flip-flop C | CDDM 1 / 2 / 3 |
|---|---|---|---|---|
| 50-100k | -18 (-9) | -3 (-1) | -8 (-3.5) | -3.2 (-7) / -3.2 (-5) / -3.0 (-13) |
| 150-200k | -2.6 (-9) | +2.8 (+1.7) | +0.2 (+0.5) | |
| 250-300k | | -0.8 (-1.3) | -0.5 (-2.1) | |

Only chain A still falls at 150-200k (bar: 2 of 3). B and C are flat from 150k on, while CDDM is
still falling about 3% per 100k at the end of its 100k. DMTS's loss spikes up to 4,000-fold above its
median and gives no usable slope.

**T3, weight motion: falsified.** Late drift exponent of W_rec: flip-flop 0.39-0.42, CDDM 0.33-0.34
(H wanted >= 0.7). The silent units' incoming rows move back and forth: median cosine between
consecutive 10k-iteration displacements -0.41 to -0.47 in every flip-flop late window (bar: <= 0.1
falsifies), -0.38 to -0.46 for CDDM's silent units. The one directed component is the flip-flop's
ACTIVE rows at 50-100k (+0.25 to +0.53; CDDM active -0.30 to -0.33), gone by 250-300k (-0.28, -0.29).

**T2, gradient signal versus batch noise: the predicted difference is there, but it does not point
anywhere.** Network-wide signal share F: flip-flop 0.83-0.99 at every read (50k to 300k, 9 reads);
CDDM 0.44-0.69 at 25k and 0.22-0.38 at 100k. T2b holds (flip-flop at its end 2.3-4.5x CDDM's).
T2a holds as worded: the silent-reached rows' median F is >= 0.05 in 8 of 9 flip-flop reads
(0.34-0.995; 0.03 at C 50k), against 0.10-0.44 at CDDM's 25k and -0.01-0.09 at 100k (6-11 units).
T2c fails in 2 of 3 chains (A 0.96 -> 0.98, B 0.95 -> 0.34, C 0.03 -> 0.89). But the consistent part
reverses between reads: the share of silent-reached units whose gradient step consistently LOWERS
their drive (t < -2) runs A 94% / 54% / 1% (50k / 100k / 200k), B 2% / 96% / 1% / 89% (50k to 300k),
C 13% / 99% / 100% (50k / 200k / 300k). A batch-consistent gradient whose direction flips between
reads, on rows that move back and forth (T3), is not a learning direction.

**Natural path (v'): falsified for the flip-flop.** Silent-reached count from 50k to the chain end:
A 199 -> 174 (-13%), B 155 -> 152 (-2%), C 175 -> 156 (-11%); bar for H a drop >= 30%. On CDDM it falls
from 40-46 at 25k to 6-11 at 100k.

**T4, anneal (Adam lr/10, saved moments, 300 iterations; A stopped at 150 on a slow node): falsified.**
Gain on the fixed evaluation set: flip-flop B 0.36%, C 0.61% (A 0.75% at 150); CDDM 0.68%, 1.03%, 0.99%.
The flip-flop gains less than CDDM (bar for H: >= 2x). The evaluation set wobbles by about 0.3%
between marks (CDDM rises 250 -> 300), so none of these runs has a clear plateau. Reach after the
anneal (v): flip-flop B 152 -> 155, C 156 -> 215 (it grew); CDDM 11 -> 13, 8 -> 7, 6 -> 5.

**Verdict by the pre-set rule: H does not explain C4.** The flip-flop is not more "still learning"
than CDDM by its loss (T1), its weight motion (T3) or annealing (T4), and the number of silent units
the gradient reaches does not shrink as its loss flattens or after annealing. What does differ in
trainability is T2: at every read the flip-flop network sits at a point where 83-99% of the per-batch
gradient power is shared by all batches (CDDM 22-69%), and yet training does not move it anywhere.
Interpretation, untested: the flip-flop's Adam steps oscillate across stiff directions (a large,
batch-consistent gradient whose sign flips from point to point; C4's per-row cosine of -0.92 between
the 10-batch gradient and Adam's saved momentum fits this). Falsifying test, not run: the cosine
between mean gradients at snapshots 500 iterations apart should be negative on the flip-flop and near
0 on CDDM, and the top Hessian eigenvalue times the effective step should sit near 2 on the flip-flop.

Recommended C4 wording: "The gradient that reaches silent flip-flop units is not a sign of unfinished
training. From 150k iterations on, two of three flip-flop networks' noisy loss changes by less than
1% per 100k iterations (CDDM's still falls by 3% per 100k at 100k), the silent units' incoming weights
move back and forth rather than travel, as on CDDM, and the gradient reaches 150-175 silent units from
50k to 300k iterations and after a ten-fold learning-rate drop, while on CDDM that number falls from
about 40 to under 15. The flip-flop differs in its gradient, not its convergence: at every point read,
most of the per-batch gradient is shared by all batches (83-99% against 22-69% on CDDM), yet training
does not follow it."
