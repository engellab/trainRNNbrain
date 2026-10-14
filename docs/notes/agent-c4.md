# Agent C4: why the incoming-weight gradient reaches more flip-flop units than are active

Claim C4 says a silent unit is a unit the task loss no longer reaches. `gradient_participation.py`
counts, per unit, the gradient on its incoming weights (g_in = ||dL/dW_rec[i,:]||, ||dL/dW_inp[i,:]||
together) and on its outgoing weights (g_out), a unit "reached" if g >= 5% of the 95th percentile.
On CDDM the reached count equals the active count. On the 3-bit flip-flop g_in reaches 1.5 to 4.5
times as many units as are active. This note asks why.

Networks: claim-1 flip-flop, N = 1000, control arm, read at the noisy-loss floor snapshot (3 seeds);
contrast: claim-1 CDDM, N = 1000, control arm, at its floor (3 seeds).

## Pre-registration (written 2026-10-13 23:30, before any new number was computed)

What I had already seen: the three flip-flop floor JSONs in `gradreach_ff/`. In them, the units the
participation rule calls silent have NO exactly-zero g_out (so every one of them fires at least once
somewhere), and the g_in count moves from 796 to 342 as the bar goes from 1% to 20% of q95 (seed
2461724787). So candidate (c) is not blind; its prediction below is stated anyway.

Definitions. "Active" = participation p = std(r) + q90(|r|) >= 0.05 q95(p), noise-free.
"On" for a sample (t, trial) = h > 0 strictly. "Reached" = g_in >= 0.05 q95(g_in).
Silent-reached = silent by participation but reached by g_in. All tests are per unit.

Oracle: a separate numpy float64 forward and hand-written backward pass through time (no torch
autograd, no `unit_gradients`), derived from the update equation
h[t] = h[t-1] + alpha(-h[t-1] + W_rec f(h[t-1]) + W_inp u[t-1] + b) and the masked MSE.
Pass bar for the oracle against the production `unit_gradients` (torch float32, same batch):
per-unit relative difference of g_in and g_out below 1e-3 at the median and 1e-2 at the max, over
units with g >= 1e-4 q95. If it fails, nothing below is read until the disagreement is explained.

(a) Silent units are briefly on.
  True if: >= 90% of silent-reached units are on in at least one sample, AND their on-fraction is
  below 10% of samples (which would make q90(|r|) = 0, so the participation rule cannot see them),
  AND among silent units Spearman rho(g_in, on-count) > 0.5.
  False if: > 10% of silent-reached units are never on, or rho < 0.3.

(b) Exact zeros leak through the clamp's gradient (f'(0) = 1 in torch.clamp).
  Mechanism I expect: during the pre-input window every rate and input is exactly 0, so every term
  of dL/dW that the window contributes is multiplied by 0. Prediction: false.
  True if: g_in with f'(0) = 0 differs from g_in with f'(0) = 1 by > 1% for any silent unit, or the
  reached count changes by > 5%.  False if: max per-unit relative difference < 1e-6.

(c) The 5%-of-q95 bar on a heavy-tailed g_in.
  True if: log g_in has no gap between active and silent units (the two distributions overlap),
  and the reached count changes by more than 2x between bars of 1% and 20% of q95 on the flip-flop,
  while on CDDM it changes by less than 1.5x.
  False if: the flip-flop count is flat (within 20%) over that range, or CDDM is as bar-dependent
  as the flip-flop.

(d) A large loss sensitivity on units that are almost always off.
  Measure: the RMS of dL/dh_i over the unit's ON samples (the per-sample gradient when the gate is
  open). True if: its median over silent-reached units is >= 2x the median over active units.
  False if: <= 1x (then their g_in comes from being on at all, not from being unusually potent).

(e) The ReLU gate is binary: a unit barely above 0 passes the whole gradient to its inputs, while
  its outputs learn in proportion to its tiny rate. True if: silent-reached units' peak rate is
  below 10% of the active units' median peak rate, AND g_out/g_in is lower for silent-reached
  units than for active units by >= 3x. False if: peak rates are comparable (> 30%).

(f) Training is noisy (sigma_rec = sigma_inp = 0.05), the read-out is not. True if: with the
  training noise, g_in reaches >= 1.2x as many units as noise-free. False if: < 1.05x.

Does the gradient change a silent unit? One Adam step, estimated: M = mean of K = 10 noisy
training-batch gradients, V = mean of their squares, step = -lr M / (sqrt(V) + 1e-8), lr = 1e-3
(the run's own). For each silent-reached unit, the change of its own drive
dd_i(t) = dW_rec[i,:] r(t-1) + dW_inp[i,:] u(t-1), averaged over the unit's 1% highest-h samples
(where it is closest to threshold).
  "Gradient revives": > 60% of silent-reached units have mean dd > 0.
  "Gradient holds them down": > 60% have mean dd < 0.
  Otherwise: undirected.
Also reported: gradient signal-to-noise per row, |M|/sqrt(V), against the value for pure noise with
K = 10 (about 0.25), and the step against the unit's distance below threshold.


### Added pre-registration (23:49, after the floor results for (a)-(f), before this test ran)

(d') Is the 4-5x larger per-on-sample dL/dh of the flip-flop's silent-reached units a property of
the units, or of WHEN they are on (93-99% of their on-samples fall within 20 steps of an input
pulse, where the output error is largest)? Restrict every unit's on-samples to those within 20 steps
(and, separately, 10 steps) of a pulse and recompute the per-unit RMS dL/dh.
  Timing explains it if: the silent-reached / active ratio of medians drops below 1.5.
  The units are more potent per sample if: it stays >= 3.

### Added pre-registration (23:51, after the one-step results, before this test ran)

The one-step Adam test turned out to be uninterpretable (see Results: the K-batch estimate points
against the saved optimizer's own step, and its sign is shared by active and silent units alike).
So read what training actually did, from the kept weight snapshots (every 500 iterations):
compare each unit's incoming weights at the floor snapshot t* with t* + 500 and t* + 5000.
  (g1) The gradient changes silent-reached units: their median relative incoming-row change
       ||dW[i,:]|| / ||W[i,:]|| (W_rec and W_inp rows together) is >= 0.5x the active units'.
       False if < 0.2x.
  (g2) It revives them: > 20% of silent-reached units are active at t* + 5000, and the real
       5000-iteration weight change raises their drive at their 1% highest-h samples in > 60%.
       It does not: < 10% become active, and that drive change is positive in 40-60% or fewer.
  Baselines reported beside them: the same numbers for silent-unreached and for active units.

## Results (2026-10-14 00:05)

Six networks, N = 1000, read at the floor snapshot: flip-flop seeds 1258717025 / 2461724787 /
4113307577 (floor at iteration 69,500 / 27,000 / 39,500) and CDDM seeds 1423687489 / 2657899762 /
2670295995 (26,000 / 25,000 / 25,500). Reference batch: task seed 12345, 1024 flip-flop trials or the
450-trial CDDM grid, noise-free. "q95" is the 95th percentile over units. Numbers are listed per seed
in that order.

| | flip-flop | CDDM |
|---|---|---|
| active units | 265 / 355 / 333 | 415 / 426 / 434 |
| reached by g_in | 407 / 624 / 525 | 440 / 428 / 443 |
| silent but reached | 145 / 273 / 195 | 50 / 46 / 46 |
| active but not reached | 3 / 4 / 3 | 25 / 44 / 37 |
| units never above 0 (g_in exactly 0) | 0 / 0 / 0 | 322 / 275 / 271 |

**Oracle: passed.** The numpy float64 hand-written backward pass agrees with the production
`unit_gradients` (torch float32 autograd) per unit: median relative difference 2e-7 to 9e-7, worst
unit 1e-4, on all six networks for g_in and g_out; reached counts identical. Before that, the backward
pass matched autograd to 1e-15 on a toy network with noise.

**(b) exact zeros through the clamp: falsified, as predicted.** g_in with the ReLU derivative set to 0
at h = 0 equals g_in with it set to 1, bit for bit, on every unit of every network. Every unit sits at
h = 0 exactly at step 0 (and on the flip-flop until the first pulse), but every weight-gradient term
from those steps is multiplied by a rate or input that is also exactly 0.

**(a) silent units are briefly on: survived.** Every silent-but-reached flip-flop unit is above
threshold somewhere: the median one for 2 to 4% of the time steps, but in 874 to 959 of the 1024
trials. Of their above-threshold samples, 93 to 99% fall within 20 steps after an input pulse, against
63% of all samples. Across silent units, g_in follows the number of above-threshold samples (Spearman
rank correlation 0.88 to 0.95). The participation rule cannot see such a unit: a unit on for under
10% of samples has q90(|r|) = 0, so its p is its standard deviation alone. 96 to 97% of these units are
on for under 10% of samples. Spot check, seed 2461724787: unit 843 is on in all 1024 trials, for 2% of
samples, peak rate 0.04, every on-sample within 20 steps of a pulse, g_in 8% of q95.
On the flip-flop no unit is gradient-starved: all 1000 units cross threshold at some point. On CDDM
271 to 322 units never cross it and get exactly zero gradient, as a ReLU must.

**(c) the bar: the pre-registered prediction held, but it applies to both counts.** From a bar at 1%
of q95 to one at 20%, the flip-flop g_in count falls 2.3- to 2.4-fold (CDDM: 1.4- to 1.5-fold). The
participation count falls 2.1- to 2.2-fold over the same range on both tasks (flip-flop seed
2461724787: 479 to 254 active; CDDM seed 2657899762: 528 to 243). So the "1.5 to 1.8 times as many"
figure depends on the bar, and so does the CDDM agreement: at a 20% bar CDDM would show 1.4 times as
many reached units as active ones. What does not depend on any bar is the shape of panel B: on CDDM
27 to 32% of the units have g_in exactly 0, while on the flip-flop the silent units' g_in
distribution runs continuously into the active units' (silent median 10 to 40 times below the active
median, upper tail overlapping it).

**(d) larger loss sensitivity when a silent unit fires: survived on the flip-flop, not on CDDM.**
Per above-threshold sample, the RMS of dL/dh is 4.1 / 4.9 / 4.9 times larger for silent-but-reached
flip-flop units than for active ones (CDDM: 0.8 / 0.9 / 1.1). This sensitivity is what lifts them over
the bar: they are on 13 to 23 times less often than active units, yet their median g_in is
0.35 to 0.43 of the active median (rough arithmetic, not a fit).
(d') timing does not explain it: restricted to above-threshold samples within 10 or 20 steps of a
pulse, the ratio stays 3.3 to 4.2 (pre-registered bar for "timing explains it": below 1.5). Their
outgoing weights are smaller, not larger (median ||W_rec[:,i]|| 0.66 to 0.94 against 1.42 to 1.51 for
active units). Why the loss is more sensitive to them per sample is untested.

**(e) binary gate: undecided by the pre-set bars.** Silent-but-reached flip-flop units peak at 11 / 17 /
32% of the active units' median peak rate (bars: under 10% true, over 30% false). Their g_out/g_in
ratio is 2.9 / 6.8 / 8.5 times lower than active units' (CDDM 1.8 to 2.8 times). The gate passes the
full error to a unit's inputs however weakly it fires, while its outputs learn in proportion to its
rate; that asymmetry is why g_out tracks the active count (1.1 to 1.3 times) and g_in does not.

**(f) training noise: falsified.** With the run's own noise, g_in reaches 0.79 to 0.84 times as many
flip-flop units as noise-free (CDDM 0.95 to 1.07). Under noise the training gradient on the flip-flop
is also 4 to 8 times smaller (ratio of q95 values) than the noise-free one the read-out uses.

**Does the gradient change a silent unit?**
Size: a silent-but-reached unit's g_in is 0.35 to 0.43 of the active median (CDDM 0.26 to 0.40).
One estimated Adam step: uninterpretable. At the end of each run the saved optimizer state gives the
true next step; the per-row cosine between it and the 10-batch estimate has medians from -0.92 to
+0.28, and in all six networks the estimated step moves the drive of active and silent units in the
same direction, with a sign that flips between seeds. The pre-registered "revive / hold down" test therefore reads a
network-wide mode, not anything specific to silent units. (Interpretation, untested: an Adam
oscillation along a stiff direction, whose phase differs between seeds.)
Real training, read from the snapshots (test g; `c4_snapshot_drift.py`, panel D), over the 5000
iterations after the floor:

| | flip-flop silent-reached | flip-flop active | CDDM silent-reached | CDDM active |
|---|---|---|---|---|
| active 5000 iterations later | 4 / 4 / 2 % | 97 / 99 / 93 % | 24 / 28 / 28 % | 92 / 89 / 89 % |
| drive at its 1% highest-h samples fell | 72 / 68 / 80 % | 53 / 54 / 47 % | 58 / 78 / 59 % | 58 / 58 / 63 % |
| incoming row shrank | 72 / 82 / 78 % | 59 / 52 / 46 % | 100 / 100 / 100 % | 96 / 96 / 95 % |
| relative row change, x active | 1.15 / 1.27 / 1.68 | 1 | 1.30 / 1.16 / 1.20 | 1 |

(g1) true on both tasks: the incoming rows of silent-but-reached units change by 1.2 to 1.7 times the
active units' relative change. (g2) on the flip-flop: no revival (2 to 4% become active, below the 10%
bar) and the drive at the moments they fire falls for about three quarters of them, against half of the
active units. On CDDM the result sits between the bars: about a quarter become active, but the drive
change is no more often upward than for active units. CDDM's silent-but-reached units are on for 6.5
to 7.5% of samples, close to the participation rule's 10% edge, and 8 to 11% of active units drop out
over the same window, so I read the CDDM movement as turnover at the bar rather than revival
(interpretation).

Figure: `img/internal_figures/c4_gradient_mechanism.pdf` (+ `.svg`, PNG preview `.pdf.png`), in the
main checkout. Row 1 flip-flop, row 2 CDDM. A: g_in against the share of samples a unit is above
threshold; B: g_in distributions; C: reached / active as the bar moves; D: the 5000-iteration outcome.

Code (branch agent-c4): `c4_gradient_mechanism.py` (Della CPU jobs, outputs in
`/scratch/gpfs/TENGEL/pt1290/agent_c4/` and `.../agent_c4/end/`), `c4_snapshot_drift.py` (local, on
snapshots it*, it*+500, it*+5000 copied from `data/c1_snapshots/<sweep>/<cell>/s<seed>/`; the
flip-flop floors live under the parent runs s1542248578, s2851654641, s2670297180), and
`c4_mechanism_report.py` (prints every test, draws the figure).

Not tested: DMTS, sizes other than N = 1000, other training lengths.

## Recommended wording for claim C4

CDDM: "About half of the silent units (271 to 322 of 1000) never cross threshold on the task and
receive exactly zero gradient on their incoming weights; the other silent units fire briefly and are
reached." Drop "reached = active" as a count identity: both counts depend on the 5% bar, and the
reached and active sets differ by 25 to 50 units each way.

Flip-flop: "Silence is not loss of gradient. Every unit crosses threshold; the units the participation
rule calls silent fire for a few steps after nearly every input pulse, and their incoming weights
receive gradient and keep changing as much as active units' do. Training keeps them silent anyway:
over the next 5000 iterations 2 to 4% become active, and for about three quarters of them the drive
at the moments they fire falls." Also state, for claim 1, that "silent" on the flip-flop means "above
threshold for under 10% of the time", not "never above threshold".
