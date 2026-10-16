## 2026-10-16 16:10 — Floor fit: end-weighted on the flip-flop, a floor above the final loss is rejected, DMTS read at its late level

Follows `2026-10-16_1326_floor-fit-multistart.md`.

**Why.** Pavel, browsing the per-network figures, found flip-flop networks whose loss ends below
their fitted floor, for example NBitFlipFlop_paper_grid_c200000, N=1000 duplication, s4177404780:
floor 0.0394, loss at the end 0.97x of it, read at 44.5k while still improving. He asked for:
- a check on whether the final loss lies below the floor;
- a look at the fit residuals;
- a check of whether different tasks need different curves.

**What was tried, in order.** All on the 1,149 figure-sweep runs of `2026-10-16_1122_topup-census.md`.

1. **Three curve models** (stretched exponential SE, power law with offset, generalised power law
   GPL), each fitted from a grid of starts and scored per task:
   - end-of-run loss predicted from a fit to the first half;
   - share of runs whose loss ends below 0.98x the fitted floor;
   - AICc;
   - residuals of the fit to the read-out curve.

   GPL won on prediction and on wrong floors (flip-flop: 31% of runs below the SE floor, 11% below
   GPL). Residual size turned out to flag bumpy curves rather than wrong floors (figures
   `img/internal_figures/floor_model_residuals.png`, `floor_model_outliers.png`). No smooth model
   fits DMTS: every model's floor lies above the final loss in about three quarters of DMTS runs.
2. **GPL rejected on truncation** (production code, rule from be17ac9 vs GPL, runs cut to 75% and 50%
   of their length). Its read-out moved about 2x more for the worst tenth of runs: flip-flop 90th
   percentile 10.7% -> 20.3% at the 75% cut, CDDM 7.9% -> 10.7%. The gap persisted for runs trained 3x
   past their read-out. A slow power-law tail pins the asymptote weakly. The pass condition, set
   before the run, was "no worse than the previous rule".
3. **Pavel's suggestion: start the fit later, or weight later points more.** Variants on the stored
   read-out curves: SE on the last decade only; SE with residual weights (t/t_end)^0.5 and t/t_end;
   GPL with and without weights.
   - SE weighted by t/t_end gave the best balance on the flip-flop and the hyper flip-flop.
   - On CDDM it made the floor less stable and, against the snapshot oracle, read several runs
     57-77% late.

**The rule now** (`common.noisy_floor`, new argument `task`):
- **Flip-flop and hyper flip-flop** (`FIT_WEIGHT_EXP`): SE with log residuals weighted by t/t_end
  (t from the escape). **CDDM and any other task:** SE unweighted.
- **Every fit** is made from 24 starts (`SE_STARTS`), keeping the lowest-error fit. This replaces
  the retry for a floor at its bound and the smoothed-minimum floor for a stretch exponent on its
  upper bound.
- **A loss that ends below 0.98x its floor gets no read-out** (`END_BELOW`, reason "loss ends below
  its floor"). Under the top-up rule such a run is still falling and gets a top-up.
- **DMTS** (`LATE_LEVEL_TASKS`, Pavel's choice): floor = median of the read-out curve over the last
  25% of training after the escape (`floor_kind = "late_level"`). A late level not below half the
  starting level is the stall, not a floor: no read-out ("late level is back at its starting level").
  That guard was found when DMTS_paper_grid_d5 N=2000 s597040509 was marked done at its 0.07 stall
  level; 3 of 65 late-level runs are caught.
- **`load_run` records carry `task`** (`task_of`); `floor_crossing`, `at_own_floor` and `read_run`
  pass it on. Direct callers pass it too: fig_paper_F2_fails, c3_flat_loss_units,
  ablate_silent_units, fig_supp_task_demand, burst_stats, run_trajectories. `floor_curve(t, fit)`
  evaluates a fit; c2_ff_floor uses it.

**Validation**, with pass conditions set before each run:
- **Self-check passes**: an SE truth with planted bursts; a DMTS-shaped loss read at its late level;
  a DMTS relapse rejected. `tests/test_scripts_start.py`: 32 scripts start.
- **Synthetic** (SE truth, bulk CV 0.03-0.3, with and without bursts): floor within 0.22%, t*
  within 2.8%. The pass condition was 1% and 10%; the rule from be17ac9 managed 0.4% and 5.4%.
- **Truncation** (production code, Mac-side figure sweeps, median / 90th-pct |t* shift|, previous ->
  now):

  | Task | 75% cut | 50% cut |
  |---|---|---|
  | Flip-flop | 2.4/10.7% -> 1.1/8.5% | 6.1/22.7% -> 2.9/21.8% |
  | Hyper flip-flop | 4.5/13.0% -> 3.7/13.0% | 12.5/29.0% -> 4.8/21.2% |
  | CDDM | 2.7/7.9% -> 2.2/7.1% | 7.7/16.3% -> 5.7/16.0% |

  All pass "no worse than the previous rule".
- **Snapshot oracle** (`2026-10-15_2054`):
  - Flip-flop: unchanged, median miss 1.2%, 6 of 6 within 10%.
  - CDDM: 6 within 10%, as before; reads 14 of 15 runs (was 13); median miss 15.6% (was 14.0%).
  - DMTS late level: worse than the previous fit (median miss 18% vs 10%), with one run read at 20k
    against 81k. Kept, because Pavel chose it and the fitted floor sits above the final loss in about
    three quarters of DMTS runs.

**Effect on the 1,149 figure-sweep runs** (previous be17ac9 -> now). Figures redrawn in
`img/internal_figures/loss_trajectories/`.

| Task | Runs | Done | Top-up | Unreadable |
|---|---|---|---|---|
| CDDM | 374 | 324 -> 273 | 36 -> 92 | 14 -> 9 |
| DMTS | 100 | 18 -> 23 | 31 -> 38 | 51 -> 39 |
| Flip-flop | 588 | 482 -> 471 | 87 -> 90 | 19 -> 27 |
| Hyper flip-flop | 87 | 65 -> 66 | 21 -> 21 | 1 -> 0 |
| **Total** | **1,149** | **889 -> 833** | **175 -> 241** | **85 -> 75** |

- CDDM read-outs read by both rules do not move (ratio 1.00), but 56 more CDDM runs need a top-up:
  their loss ends 2-5% below the floor (still creeping down), so the guard refuses them.
- Flip-flop read-outs read by both rules: median ratio 1.01, 10th-90th percentile 0.92-1.48.

**Paper figures.**
- **Figure 2's CDDM recurrent-noise ladder** (`CDDM_77fbd9_g0_noise_ptrack`): no read-out changes.
- **Figure 1's size-series cells:**
  - `NBitFlipFlop_ff_revive_c200000`: every run is done under both rules, and t* moves by -8% to +6%
    per run. Two N=500 runs (s2484557599, s2545439171) gain a read-out (57.6k, 65.6k).
  - `NBitFlipFlop_paper_grid_c200000` N=4000: 3 runs, t* 71.2/71.2/62.1k -> 66.2/69.4/59.3k, all
    done.
  - `DMTS_paper_grid_d5`, now read at its late level, controls by N:
    - N=500: s1084904964 9.5k -> 62.1k (now done); s2099578504 gains 109k but needs a top-up (1.4x);
      s2106510642 16.0k -> 18.2k.
    - N=1000: s86371460 9.8k -> 18.6k; s1167683015 81.4k -> 75.7k (top-up); s2946835052 loses its
      read-out (loss ends 0.93x the late level).
    - N=2000: s2585017553 52.4k -> 51.6k; s4124951173 60.3k -> 73.4k (now done); s597040509 is
      rejected by the stall guard.
    - Duplication N=500: s2923428690 loses its 13.7k read-out (still falling).
- Per-run tables: `run_trajectories.py extract` output in the session scratchpad, not committed.
