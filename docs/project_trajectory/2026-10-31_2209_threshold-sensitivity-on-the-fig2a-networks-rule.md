## 2026-10-31 22:09 — Activity-threshold sensitivity on the networks Fig. 2a fits: rule, written before the run

Work package `methods-r2` (ptolmachev/dead-relu-paper#51; decision on #21). trainRNNbrain branch
`wp/methods-r2`. Replaces the network set of entry `2026-10-31_1657` (35 networks, DMTS N = 500-2000)
with the set Figure 2a fits, so that Methods M04 states one exponent per task at the factor 0.05.

**Networks.** Every size-series row of the Figure 2 collect record (`data/fig_paper_F2_size_collect.pkl`,
Della job 16558986, `~/trainRNNbrain_fig2size` at 8bb2939; entry `2026-10-31_1957`) that Figure 2a
fits: N ≥ 500, a crossing at 1.05× (`reason` None), participation probes spanning the crossing, and
not excluded as never learned. The record's one other exclusion string ("replaced by a fourth seed")
is cleared at draw time by `fig_paper_F2_size.apply_decisions` (`EXCLUDED_RUNS` is empty), so that
network counts. Expected: 3BFF 16, CDDM 12, DMTS 9 (37).

**Measurement.** For each network, the stored participation vector at the probe nearest its 1.05×
crossing t\* (the record's `t_star`); the active count with the factor f in p ≥ f·q95(p) at
f = 0.025, 0.05, 0.10; q50/q95 at that probe. Exponent per task and factor: least-squares slope of
log(mean count per size) against log N over N = 500-4000. The script reads the participation traces
directly and does not call `common.read_run` or `fig_paper_F2_size`, so the count at 0.05 is an
independent check on the record.

**Pass/fail, set before running.**
- The recompute is wrong (and nothing from it enters the paper) if, at f = 0.05, any network's count
  differs from the record's `active_at[0.05]`, or any task's exponent differs from Fig. 2a's printed
  value (3BFF 0.478, CDDM 0.443, DMTS 0.551) by more than 0.001.
- The sensitivity reported in M04 is then the range over networks of the count change at 0.025 and
  at 0.10 relative to 0.05, and the exponent at each factor.

Script: `trainRNNbrain/experiments_and_analysis/threshold_sensitivity.py`. Run on Della (the 3BFF
traces at N ≥ 500 are not on the Mac). Result in a separate entry.
