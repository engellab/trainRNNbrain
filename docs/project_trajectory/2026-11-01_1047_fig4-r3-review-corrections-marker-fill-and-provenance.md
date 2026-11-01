## 2026-11-01 10:47 — fig4-r3, review of PR #101: marker fill per the #83 decision; provenance of the C4 run (corrects 2026-11-01_1021)

Corrects `2026-11-01_1021_c4-on-the-fig1d-snapshots-passes-figure-4-redrawn.md` in two places,
both found by the independent review of dead-relu-paper PR #101.

**Provenance.** Entry 1021 says the run used "wp/fig4-r3 (27514fa for the measurement code)". The
`fig1d` run started from the working tree about three minutes before 27514fa was committed (first
`c4_fig1d` output 09:53:45, commit 09:56:33). Between the start of the run and that commit only
drawing code changed in `fig_paper_F4_noreturn.py` (marker styles, the panel-d printout, the ED
tanh labels, the module docstring); `fig1d()` and `c4_gradient_mechanism.run_one` were not edited
after the run started, so 27514fa holds the code that produced `c4_fig1d/`, `traces_fig1d/` and
`activity_fig1d/`. No number changes.

**Marker fill.** Entry 1021 filled every marker "read at or after a floor crossing". The decision on
#83 (merged in paperstyle, trainRNNbrain main 36c18d7) is: filled = read at the floor crossing;
open = read at a fixed iteration, which the caption names; fill means nothing else. Redrawn on
`wp/fig4-r3` after merging main:
- filled (floor crossing): Fig. 4b (the Fig. 1d weights, the kept snapshot nearest the crossing),
  Fig. 4e,f (the new task's floor crossing);
- open (`ps.mark_fixed`): Fig. 4c (500 and 5,500 iterations after the Fig. 1d weights), Extended
  Data Fig. 5c,d (CDDM 25,000 and 100,000; 3BFF 50,000 to 300,000; the learning-rate squares 300
  iterations after the last), Extended Data Fig. 6b,c (end of the 150,000-iteration budget, and
  the untrained networks).
Silent units (Fig. 4b) stay diamonds and young networks (Fig. 4e,f) triangles: shape, not fill.

**Threshold names.** The threshold set on the untrained network (5% of the untrained network's
95th percentile; tanh comparison) is now called the "untrained threshold" in the Fig. 4d legend and
the ED Fig. 6b axis, so "reach threshold" means only 5% of the network's own 95th percentile.

`tests/test_paperstyle.py` passes on all three PDFs; `check_layout` reports no overlap and no text on
a line. Figures: [Figure 4](../../img/internal_figures/fig_paper_F4_noreturn.svg),
[ED Fig. 5](../../img/internal_figures/fig_ed_mechanism.svg),
[ED Fig. 6](../../img/internal_figures/fig_ed_tanh.svg).
