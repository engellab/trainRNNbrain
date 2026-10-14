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

## Results

(filled in below after the runs)

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
