## 2026-11-01 07:45 — Which held-out R² values in Figure 5 come from weights inside a loss burst: rule set before the run

Package fig5-r2 (dead-relu-paper#55), decision on dead-relu-paper#18 (editor, round 2): the
floor-crossing code is not changed; every held-out R² drawn in Figure 5 or listed in Supplementary
Table 2 (paper `floats/EDT3_remedies.tex`) is checked against the loss bursts and the non-finite
stretches of its own network, and each affected value is marked in the figure and the table. This
entry fixes the rule before any value is computed.

**Which values.** Every R² those two items show, plus the two neighbours the text quotes:
- the per-unit-rule catalogue (entry 2026-10-31_1732, Della `~/rm_read/runs_*.json`, 267 runs): R² at
  the floor crossing (measured or estimated) and R² at 75,000 iterations;
- the removal analysis (same entry, `~/a3_read/a3_N*.json`): intact R² and R² after removal, all on
  the weights at 75,000 iterations;
- the combination test (entry 2026-10-31_1750, `~/b3b4_read/read_b3.json`): R² at the floor crossing
  and at 75,000 iterations;
- the CDDM rate-penalty networks of Fig. 3a (`CDDM_std_g0_bias0_c75000`, λ_frm = 0.2 and 0, five
  networks each; R² of the final weights at 75,000 iterations). Their bursts were never counted
  (board, PR #49 review); the same run counts them.

**Which weights each value was computed from (iteration t_w).**
- R² at the floor crossing, measured: the kept snapshot's iteration (`it<iter>` in its path).
- R² at the floor crossing, estimated (1 − (L(t\*)/L_end)(1 − R²_end)): the final weights, t_w = the
  last iteration of the loss series. L(t\*) is read on the burst-masked smoothed curve, so only the
  final weights can carry a burst into the value.
- R² at 75,000 iterations, and every removal value: t_w = 75,000.

**The rule.** A value is *affected* if t_w lies inside a loss burst or a non-finite stretch of its
network's noisy training loss (warm-start chain joined, `common.noisy_loss_series`):
- loss bursts are `common.noisy_floor(...)["bursts"]` (first and last iteration), each widened on both
  sides by max(51, half its length), the padding `noisy_floor` uses when it masks bursts out of the
  floor fit;
- a non-finite stretch is a maximal run of consecutive iterations whose loss is NaN or infinite,
  widened the same way.

**Falsification check, fixed now.** The rule is meant to catch weights that do not do the task. If
any intact R² (not a removal value) below 0.5 is *not* marked affected, the rule misses a failure
mode: that value is reported and investigated before anything is marked. Values marked affected with
R² above 0.9 are reported as such (marked, and counted separately), not dropped from the mark.

**What the paper does with it.** Figure 5 draws an affected value with its own marker and the
caption says so; Supplementary Table 2 footnotes every mean that includes one; the PR body reports
the count. If an affected value sits outside the combination test and the rate-penalty networks, the
#18 decision is reopened on that issue before any claim changes.

**Also collected in the same job** (no rule needed, a drawing input): the active count at every
participation probe of every N = 1,000 network of the five per-unit rules' size series and their
control networks, for Figure 5a.

Script: `trainRNNbrain/experiments_and_analysis/fig5_burst_check.py` (branch wp/fig5-r2). Job id,
commit and result: next entry.
