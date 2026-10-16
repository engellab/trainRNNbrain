## 2026-10-15 14:00 — Read-out rule rebuilt: Hampel curve, sustained crossing, several margins, bursts counted

**Why.** Pavel asked for a fairer read-out across sizes, hyperparameters and training methods. The
old rule took the first iteration at which a 2,001-iteration running median of the noisy loss
touched 1.05x the fitted floor. A first touch is the first noise dip under the bar, so a noisier
network was read earlier. That is a bias between arms, not scatter. A fixed 2,001-iteration window
also smooths a fast run far more, relative to its training, than a slow one. Bursts (the loss briefly
climbing several-fold and recovering) were smoothed over and never reported.

**What changed** (`common.noisy_floor`, `trainRNNbrain/experiments_and_analysis/common.py`):
- The curve is a Hampel mean over [t/1.1, 1.1t]: within each window, points more than 3 MADs (median
  absolute deviations) from the median are dropped and the rest averaged. It is computed on a t-grid
  with points 1% apart and interpolated back to every iteration.
- The floor is fitted to that same curve instead of to log-binned medians, so the floor and the crossing
  use one statistic.
- The crossing is sustained: the first t after which the curve stays at or below the bar through
  1.2t. It is read at 1.05x, 1.10x and 1.20x (`t_stars`); `t_star` is still the 1.05x read-out.
  Pavel: 1.05x is the tightest margin every network can be held to.
- Bursts are stretches where a 51-iteration running median of the raw loss sits above 2x a wide
  [t/1.5, 1.5t] Hampel reference. They are returned in `bursts` and masked out of the read-out curve,
  each padded by max(51, half its length) on both sides. A run with more than half of its training
  after the fit start inside bursts gets no read-out.
- New: `burst_stats.py` writes one row per cell to `data/burst_stats.csv`: runs, runs with bursts,
  bursts per 100k iterations, percent of iterations in bursts, median length and peak ratio. It
  counts each run's own iterations only, so a warm-started run does not inherit its parent's bursts.
- `docs/AGENT_GUIDE.md` "Reading networks" now describes this rule.

**Validation against a known answer.** Synthetic traces: (0.01 + 0.5 exp(-sqrt(t/1000))) times
mean-1 gamma noise, 150k iterations, three seeds. Some traces add 1% of iterations x10 (real
per-iteration loss is heavy-tailed) and five planted bursts of 300-3,000 iterations at 3-20x. The
truth is the noise-free curve times the Hampel-trimmed mean of the noise distribution, computed
analytically with `scipy.stats`, so it shares no code with `hampel_curve`. Thresholds were set
before running: floor within 1%, t* within 10% at every margin, 5 of 5 bursts found with edges
within max(100, 20% of width), no false bursts.
- Real noise first: the bulk per-iteration CV (SD over mean) is small. The robust CV, scaled MAD
  over median near the end of a run, is 0.01-0.03 on the flip-flop and DMTS and 0.2 on Yang. The
  plain CV is 0.05-11, because rare huge outliers dominate it.
- Bulk CV 0.03 and 0.2 with outliers, and CV 0.3 without, with and without bursts: all pass. The
  floor is within 0.4%, the 1.05x crossing within 2%, and every burst is found.
- On the same traces the old rule read 2-19% early.
- Stress case, gamma CV 1, about 5x the noisiest real task: fails. The floor is off by up to 1.3%,
  t* by up to 10.6%, and one clean trace gives two false bursts of 6-8 iterations. The same failures
  appear without bursts, so this is the method's precision limit at that noise level.

**Two fixes the comparison on real runs forced.**
1. A robust soft-L1 fit loss, in the first version, converged to a degenerate fit on 231 CDDM and
   flip-flop runs: floor about 0, beta on its lower bound, then "loss ends 3,000x above its floor". On
   the same data the plain least-squares fit finds the floor the old rule found (0.0222 on the run checked).
   Soft-L1 was dropped; the synthetic results moved by under 0.3 percentage points.
2. Without masking, DMTS bursts (hundreds to thousands of iterations, recurring every few thousand)
   cover a quarter of the [t/1.1, 1.1t] window at t of 20-30k. The Hampel cut keeps them, and the
   floor came out up to 1.6x the between-burst level. The mask fixed that; the synthetic burst traces
   now match their clean twins to within 0.15 percentage points of floor error.

**Old vs new on all 1,775 local runs** (`data/trained_RNNs`):
- 1,081 runs are read by both rules; 173 lose their read-out and 36 gain one.
- Among runs read by both, the median t* new/old is 1.00-1.06 on every task. On the flip-flop, 10%
  of runs are read at least 32% later; DMTS spans 0.82-2.09 between its 10th and 90th percentiles.
- Of the 173 lost:
  - 134 never stay under the bar: 81 never reach it, 36 reach it too close to the end for the 1.2t
    hold, and 17 reach it and climb back. Most are 40k-budget flip-flop sweeps that end 5-18% above
    their fitted floor. The old rule read them near the end of training on a noise dip (old t* of
    28k of 40k, 73.5k of 100k, 149.6k of 150k).
  - 28 end more than 1.5x above the floor.
  - 9 have a fit pinned at the lower bound.
  - 2 have more than half of training in bursts.
- Runs read at each margin: 1,117 at 1.05x, 1,210 at 1.10x, 1,296 at 1.20x.

Figure: `img/internal_figures/readout_hampel_vs_median_extremes.png`. It shows the six runs where
the rules disagree most, with the raw loss, both curves, both bars, both read-outs and the bursts.
- Two DMTS runs and one flip-flop run: the old fit put the floor 1.3-2.4x too high and read the
  network while the loss was still falling (4.5k, 5.3k, and the 2k fit start).
- One DMTS N=500 run: the trough level keeps falling to 150k while bursts under 2x continue. The new
  rule reads it at 34k (the average stops improving); the old one read the last iteration.
- One flip-flop run with a firing-rate penalty (`std_penlong`, k=2, N=2000): the loss dips to the bar
  at 19k, then steps up to about 1.1x the floor for the rest of training, just after the hold. The
  new rule reads the dip, the old rule read a noise dip at 110k; neither describes this run well.

**Bursts by cell** (`burst_stats.py`, 564 cells):
- CDDM paper grid: no bursts.
- DMTS paper-grid controls: bursts in every run at N=500 and N=1000, with 9-15% of iterations inside
  bursts.
- Flip-flop paper grid: bursts in every run of its duplication cells (N=500-4000) and its N=4000
  control. The rate rises with N in the duplication arm, from 9 per
  100k iterations at N=500 to 108 at N=4000, with 17% of iterations inside bursts at N=4000.
- Revive, rescale and norm-mode interventions: 50-320 bursts per 100k iterations, some with the loss
  blowing up by more than 1e10.

**Known gaps.**
- `smoothed` is computed on the finite samples only, so it is shorter than `it` when the loss has
  NaNs. The old code did the same. Callers that index it with their own `it`
  (`fig_knobs_at_floor.py`, `fig_supp_task_demand.py`) are misaligned on such runs.
- Docstrings of `ablate_silent_units.py`, `fig_paper_F1.py` and `flipflop_ksweep.py` still describe
  the 2,001-iteration median. Their cached results were made under it, and any figure cache must be
  rebuilt before those numbers follow the new rule.

**Reproduce.** `python trainRNNbrain/experiments_and_analysis/common.py` (self-check, includes a
synthetic floor/crossing/burst test); `python trainRNNbrain/experiments_and_analysis/burst_stats.py
--workers 10` (37 s on the Mac for all runs).
