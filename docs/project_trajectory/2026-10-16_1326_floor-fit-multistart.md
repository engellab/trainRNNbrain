## 2026-10-16 13:26 — Floor fit: refit from a grid of starts when the floor lands at its bound

**Found by Pavel** while browsing the per-network trajectory figures
(`2026-10-16_1151_loss-trajectory-figures.md`). CDDM_77fbd9_g0_noise_ptrack, EqType=s, N=1000,
sigrec=0.05, seed 4007701949, trained fine, yet was marked unreadable with "loss ends 21,000x above
its floor".

**Cause.** `noisy_floor` fits its stretched exponential from one fixed start (floor = 0.9x the
curve's minimum). On some slowly flattening losses the optimiser slides from that start to floor
about 0 with beta near its lower bound, where the stretched exponential bends into a power law.
That is a local minimum. On this run a grid of starts finds floor 0.0226 with a fitting error 30x
smaller, and the run ends at 1.03x that floor.
- The existing guard ("fit pinned at its lower bound") fires only at or below 1.01x the bound,
  1.01e-6. This fit sat at 1.1e-6 and slipped past it.

**How common it is.** Every one of the 1,098 runs with a fit, in the figure sweeps of
`2026-10-16_1122_topup-census.md`, was refitted from 80 starts. On 1,053 the single start already
finds the best fit. On 45 a real floor fits better, and in all 45 the current floor is about 0. Most
are CDDM: all 20 unreadable runs in CDDM_std_g0, 13 in CDDM_77fbd9_g0_noise_ptrack, plus
CDDM_std_g0_bias0_c75000, CDDM_std_g0_recinit and 3 std_switch runs.

**Fix** (`common.py`, `noisy_floor`). When the first fit's floor is within 10x its lower bound, the
fit is repeated from `GRID_STARTS`: floor at 0.5, 0.8, 0.9 or 0.97x the curve's minimum, tau 3e2 to
3e5, beta 0.2 to 2, 64 starts. The fit with the lowest error is kept. Healthy fits never reach the
retry, so they cost nothing extra.

**Checks**, with pass conditions set before running:
- Every non-degenerate run must be unchanged: 0 of 1,106 changed (floor, t* at all margins, status).
- `common.py` self-check passes.
- `tests/test_scripts_start.py` passes: 32 scripts start.
- The synthetic validation is unchanged: it fails only at CV 1, as before.

**Effect on the 1,149 figure-sweep runs.**
- 43 fits were degenerate; 41 now find a real floor. Of those, 17 go from unreadable to done and 23
  to top-up. 3 stay unreadable: 2 are still degenerate, and 1 is excluded for another reason.
- Totals: 889 done (was 872), 175 top-up (was 152), 85 unreadable (was 125).
- **Figure 2, recurrent-noise ladder** (`CDDM_77fbd9_g0_noise_ptrack`, read through
  `heldout_r2.read_rung`): 13 runs previously dropped now have a read-out, with floors 0.0226-0.0231
  and t* 27-40k.
  - sigrec=0.1, equation h: 5 runs, all done.
  - sigrec=0.05, equation s: 5 runs, 4 done and 1 needing a top-up (s4007701949, trained to 1.9x t*).
  - sigrec=0.1, equation s: 3 runs, all needing a top-up.
  - Those rungs gain these runs whenever Figure 2 is rebuilt.

**Figures** redrawn in `img/internal_figures/loss_trajectories/` with the fixed fits. The 1.05x
(blue, solid) and 1.10x (violet, dashed) bars and their read-outs are now drawn strong and labelled
on the axis; 1.20x is faint grey. Warm-start joins are brown.
