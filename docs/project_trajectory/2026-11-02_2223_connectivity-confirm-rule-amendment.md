## 2026-11-02 22:23 — Amendment to 2026-11-02_2218 (confirmatory connectivity rule): a second twin check for CDDM first runs without stored row norms; fixed before any statistic is computed

Amends `2026-11-02_2218_connectivity-confirm-rule.md` (commit 44cd160). No connectivity statistic and
no #172 slope has been computed on any of the 48 networks. The weights were copied from Della (CPU job
16712346, `scratchpad/r12-weights/fetch_confirm.py` with `joblist.json`, into
`/scratch/gpfs/TENGEL/pt1290/r12w_confirm`). Only the trace quantities of the checks were read.

**Why.** Check T compares the rebuilt initial weights with the `w_rec_row` vector of the first run's
trace. Copying showed that the first runs of 41 of the 48 networks predate that field. Only networks
#4, #37-#39 and #46-#48 store it. Under the rule as written, all six weight-decay-0 CDDM networks would
leave the #172 test for lack of data, and the test could not be performed. The consequence follows
from the trace format, not from any value of the test.

**Change.** For a CDDM network whose first-run trace has no `w_rec_row`, the twin is also valid if
check **T'** passes. T' simulates the rebuilt twin without noise on the `population_stats.batch`
batch (for CDDM, the fixed grid of the task's trained conditions). It then compares the twin's
per-unit participation p = std(r) + q0.9(|r|) with the vector stored at iteration 0 of the first
run's trace. That probe is noise-free on the training batch and precedes the first update. T' passes
if the largest difference, relative to the mean stored participation, is below 1e-3.

Calibration, on the three seen CDDM networks of Fig. 1e (their own traces) before this entry
(`scratchpad/r12-weights/calib_partic.py`):
- the true twin gives 1.1e-5, 4.5e-5 and 3.1e-5 (correlation 1.0000);
- a twin with seed + 1 gives 3.9 to 5.7 (correlation −0.05 to 0.06).

If the old task class builds a different batch than current code, T' fails for the true twin as
well. The network is then treated as the rule says for a failed twin, which is the conservative
outcome.

T' applies wherever the rule uses twin validity: the choice of θ in the main test, and inclusion in
the #172 test. Everything else in entry 2218 is unchanged. The 3BFF and DMTS networks without row
norms keep the rule's fallback θ = 2 × 1.2/√N. Their batches are random draws, so T' cannot
reproduce them.
