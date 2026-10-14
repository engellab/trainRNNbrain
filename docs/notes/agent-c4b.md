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

### Verdict rule (set now)

H explains C4 only if all three hold: the flip-flop is still learning by T1 or T3 (beyond CDDM); its
silent units' gradient is a consistent signal (T2a); and the reach shrinks as the network converges
(T4 v or the natural path). If T2a or (v) fails, non-convergence is not why the gradient reaches
silent flip-flop units, whatever T1 says.
