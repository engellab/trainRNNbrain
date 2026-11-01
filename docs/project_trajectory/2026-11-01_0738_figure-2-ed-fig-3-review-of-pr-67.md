# Figure 2 and Extended Data Fig. 3: changes after the review of PR #67

Follows and corrects entry `2026-11-01_0716`. Review: dead-relu-paper PR #67, comment 6560050115
(approved, findings 1-7). No number changes. Every value comes from the same collect record
(Della job 16558986) and `data/fig_ed_readout_batch_noise.json`.

- **The batch-noise ranges are a description, not a test.** The CDDM disagreement between t* and the
  snapshot estimate (−33% to +64%) was in the notebook from entry `2026-10-15_2054`, before the
  range analysis was chosen. Rule 1 of entry `0118` is only a reproduction check. So the ranges
  describe how far batch noise alone moves the snapshot estimate on n = 3 networks. Entry `0716`
  says the t* − snapshot differences of the three networks are "within what batch noise alone moves
  the snapshot estimate by". That sentence is an interpretation of three networks and does not go
  into the paper as a tested result. The 11 CDDM networks with scaled input weights have no range.
  The reviewer varied the seed, the resample count and the percentile: the ranges moved by at most
  one snapshot spacing (1,000 iterations).
- **Extended Data Fig. 3.** The batch-noise bars moved from panel c, where they overlapped into one
  stroke, to panel d. Each of the three CDDM default networks now carries a vertical bar from
  t*/hi − 1 to t*/lo − 1: −4% to +56% (seed 1423687489), −7% to +74% (2657899762), and −10% to +51%
  (2670295995). The CDDM default networks sit left of the networks with scaled input weights, and a
  CDDM-coloured legend swatch marks the bars. `BURST_SHADE`, which was never used, is removed.
- **Figure 2e** draws the losing-units threshold (5 active units per 100,000 iterations,
  `c3_flat_loss_units.DIE_BAR`) as a dashed line labelled "losing units". Per size: 4 of 5 flat
  networks lose units at N = 500, 5 of 6 at N = 1,000, 1 of 1 at N = 2,000 (56.6 per 100,000) and
  3 of 3 at N = 4,000. That makes 13 of 15 at all four sizes.
- Both PDFs pass `tests/test_paperstyle.py`, with 0 overlapping text pairs.
