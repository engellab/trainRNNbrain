## 2026-11-01 10:21 — C4 on the Figure 1d snapshots passes its rule; Figure 4 and ED Figs. 5-6 redrawn (dead-relu-paper #91, #83, #88, #63)

Follows `2026-11-01_0948_c4-on-the-fig1d-snapshots-rule-set-before-the-run.md`, which set the rule
and the oracle checks before the run. Supersedes the per-network table of
`2026-10-31_2207_c4-at-the-current-floor-passes-its-rule-figure-4-built.md` and the silent-unit
table of `2026-11-01_0716_silent-units-get-small-gradient-under-training-noise.md` for what the
paper prints; those entries stay as the record of the 500-iterations-later read.

**How it was run.** trainRNNbrain `wp/fig4-r3` (27514fa for the measurement code), on the Mac:
`python trainRNNbrain/experiments_and_analysis/fig_paper_F4_noreturn.py fig1d` (26 min, CPU, no
GPU), then `... draw`. Inputs: `data/pop_stats_nets/` (the Figure 1d weights) and the Della read's
copies `data/fig4_noreturn/snap/<seed>_d0.npz`, `_d5000.npz`. Outputs: `data/fig4_noreturn/c4_fig1d/`,
`traces_fig1d/`, `activity_fig1d/`. Della refused the login, so the later snapshots are the ones on
the Mac: 500 and 5,500 iterations after the Figure 1d weights.

**Oracle checks (all pass, per network).**
- (a) Active counts on the Figure 1d weights: CDDM 430, 416, 438, equal to the removal file; 3BFF
  269, 342, 287 against 267, 342, 287 (within 5; different trial batch).
- (b) CDDM silent counts 310, 288, 253: equal to the committee's independent numpy count on #91.
- (c) The full-batch participation pass reproduces the C4 classes in every unit of all six networks.
- (d) Participation on the `d0` copies reproduces job 16566988's classes in every unit (largest
  relative difference in p 7e-15), so those copies are the weights 500 iterations later.
- Autograd against the float64 backward: per-unit median relative difference of g_in 1.9e-7 to
  5.8e-6 (largest 3.8e-4); reached counts identical in all six networks.

**Per network** (noise-free reference batch, task seed 12345):

| Network | Weights (crossing) | Active | Silent | Low-active | Reached low-active | Reached low-active active at +500 / +5,500 |
|---|---|---|---|---|---|---|
| CDDM s1423687489 | 25,000 (25,020) | 430 | 310 | 260 | 40 | 23% / 23% |
| CDDM s2657899762 | 26,000 (26,028) | 416 | 288 | 296 | 44 | 34% / 39% |
| CDDM s2670295995 | 24,000 (24,219) | 438 | 253 | 309 | 60 | 22% / 32% |
| 3BFF s1258717025 | 67,000 (67,191) | 269 | 0 | 731 | 136 | 0% / 0.7% |
| 3BFF s2461724787 | 50,000 (50,073) | 342 | 0 | 658 | 155 | 1.3% / 2.6% |
| 3BFF s4113307577 | 58,000 (58,035) | 287 | 0 | 713 | 161 | 1.2% / 4.3% |

- Silent CDDM units: largest noise-free g_in exactly 0 in the numpy backward and in autograd. At
  +500 and +5,500 at most 1 silent unit per network is active.
- Under the training noise (10 batches): silent units have x > 0 on a median 1.54-1.61% of steps;
  no per-batch gradient is exactly 0; the median silent unit's g_in is 94-155 times below the reach
  threshold (was 60-291 on the later weights); 6 of 851 silent units cross the threshold in at least
  one batch (1, 4, 1).
- 3BFF reached low-active units: 19-24% of the low-active units; median share of steps with x > 0
  2.4-3.7%; 95-99% of their on-steps within 2 tau after an input.
- Active units still active at +5,500: CDDM 89%, 3BFF 96-99%.
- Reached / active at the 5% bar: 3BFF 1.45-1.56, CDDM 0.98-1.04.

**Rule: all four clauses pass.** 3BFF has 0 silent units; every CDDM network has at least 200 silent
units, largest g_in exactly 0; every 3BFF network has at least 100 reached low-active units (136, 155,
161); fewer than 10% of them are active 5,500 iterations later (0.7%, 2.6%, 4.3%).

**What changes in the paper.** Figure 4a-c and Extended Data Fig. 5a,b now draw these weights, the
same as Fig. 1d; Methods (Gradient reach) names the iterations; panel c and the text say 5,500
iterations. Silent count 253-310 (was 260-313); the 3BFF bound "at most 6%" becomes "at most 5%".

**Figure changes in the same redraw** (trainRNNbrain wp/style-r3 39e5a1e merged): every marker is
filled (all values are read at or after a floor crossing; #83), so silent units in Fig. 4b are
diamonds instead of hollow circles, young networks in Fig. 4e,f triangles instead of open circles,
and the learning-rate squares of ED Fig. 5c filled; every axis label sits 1.36 mm from its tick
labels (#63); the ED Fig. 6b categories read "gradient on incoming weights" and "sensitivity to
incoming weights" on three lines, 3.2 mm apart at the closest (#88). `tests/test_paperstyle.py`
passes on all three PDFs.

Figures: [Figure 4](../../img/internal_figures/fig_paper_F4_noreturn.svg),
[ED Fig. 5](../../img/internal_figures/fig_ed_mechanism.svg),
[ED Fig. 6](../../img/internal_figures/fig_ed_tanh.svg).
