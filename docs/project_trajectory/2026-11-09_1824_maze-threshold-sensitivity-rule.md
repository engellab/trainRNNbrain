## 2026-11-09 18:24 — Activity-threshold sensitivity of the Maze size exponent: rule, written before the run

Work package r21-text (ptolmachev/dead-relu-paper#218; committee #217, Reviewer R item R2). trainRNNbrain
branch `wp/r21-text`. Methods M04 gives the size exponent b for the active-unit factor f from 0.025 to 0.10 for
3BFF, CDDM and DMTS (entries `2026-10-31_2209` rule, `2026-10-31_2211` result), not for Maze, the third task
of Fig. 2a since 4 Nov. This entry adds Maze with the same design.

**Networks.** The Maze networks of the Fig. 2a fit at the learned point: every ReLU run without a goal cue of
`data/trained_RNNs_rl_maze/Maze_suite` that `fig_paper_F2_size.maze_series` marks learned, at N ≥ 500. Expected
16 networks: 3, 6, 3 and 4 at N = 500, 1,000, 2,000 and 4,000 (M12; `data/round19/size_exponent.json`,
`learned`, b = 0.343, 0.291-0.395).

**Measurement.** The Maze trainer stores the participation vector every 50 updates (`probes['p']`,
`probes['p_update']`) and the active count at every probe (every 10 updates). For each network: the stored
participation vector nearest its learned point t\*, and the count with p ≥ f·q95(p) (`common.active_mask`) at
f = 0.025, 0.05 and 0.10. Exponent per factor: `fig_paper_F2_size.power_fit` over N ≥ 500 with the bootstrap of
`boot_exponent` (2,000 resamples within size, `BOOT_SEED`), as for Fig. 2a.

**Checks, set before running.**
- Same quantity: at every stored vector whose update is also a probe update, in all 16 networks, the count at
  f = 0.05 must equal the probe's stored `active`. Any mismatch: the result does not enter the paper.
- The vector nearest t\* is up to 25 updates from t\*, so the count at f = 0.05 may differ from the count of
  Fig. 2a. If the exponent at f = 0.05 from these vectors differs from 0.343 by more than 0.03, the paper reports
  only the count changes, not the exponents.

**What the paper gets.** The range of b over the three factors, beside the other tasks in M04, and the range of
the count change at 0.025 and at 0.10 relative to 0.05, in Supplementary Methods.

Script: `trainRNNbrain/experiments_and_analysis/maze_threshold_sensitivity.py`. Result in a separate entry.
