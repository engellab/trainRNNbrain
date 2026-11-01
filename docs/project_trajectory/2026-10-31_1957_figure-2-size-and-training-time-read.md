## 2026-10-31 19:57 — Figure 2 read: C2 holds on all three tasks at every margin; C3 holds at N = 500, 1000 and 4000

Work package fig2-size-time (ptolmachev/dead-relu-paper#6), trainRNNbrain branch `wp/fig2-size-time`.
Rules: entry `2026-10-31_1900` (written while the job ran, before any output existed).

**Job.** Della CPU job 16558986 (`fig_paper_F2_size.py --collect`, checkout `~/trainRNNbrain_fig2size`
at 8bb2939) finished COMPLETED in 1 h 15 min; 61 new held-out R² scores. Record copied to the Mac as
`data/fig_paper_F2_size_collect.pkl` (+ `_r2_cache.json`); the drawing stage printed every number below.
Figures: `img/internal_figures/fig_paper_F2_size.{pdf,svg}`, `fig_ed_readout.{pdf,svg}`; table body
`img/internal_figures/edt2_runs_rows.tex`.

**Decision followed.** The 3BFF N = 1000 network seed 3374117298, once excluded as a straggler, reads
out under the current rule (277 active, R² 0.949 estimated) and is counted, as dead-relu-paper#20
recommends (option 2) until Pavel answers. `EXCLUDED_RUNS` is now empty; `apply_decisions` re-applies
the list at draw time, so the Della record did not have to be rebuilt. With it, N = 1000 is
277.5 ± 26.6 (n = 4) instead of 277.7 ± 32.6 (n = 3); the 3BFF exponent moves by less than 0.01.

**C2: active units at 1.05× each run's floor** (mean ± SD, n read, % of N)

| N | 3BFF | CDDM | DMTS |
|---|---|---|---|
| 100 | 81.7 ± 2.5 (3, 82%) | 89.7 ± 1.5 (3, 90%) | 66.0 ± 8.5 (2, 66%) |
| 500 | 205.0 ± 18.4 (6, 41%) | 298.3 ± 6.4 (3, 60%) | 191.0 ± 17.3 (3, 38%) |
| 1000 | 277.5 ± 26.6 (4, 28%) | 431.0 ± 9.5 (3, 43%) | 320.0 ± 144.2 (2, 32%) |
| 2000 | 390.0 ± 22.9 (3, 20%) | 548.3 ± 29.4 (3, 27%) | 425.7 ± 39.5 (3, 21%) |
| 4000 | 551.7 ± 41.7 (3, 14%) | 767.0 ± 10.1 (3, 19%) | 620 (1, 16%) |

Exponent (fit from N ≥ 500 on the size means; bootstrap 95% interval, runs resampled within size,
2,000 resamples):

| Margin | 3BFF | CDDM | DMTS |
|---|---|---|---|
| 1.05× | 0.478 [0.436, 0.521] | 0.443 [0.432, 0.457] | 0.551 [0.483, 0.631] |
| 1.10× | 0.509 [0.474, 0.543] | 0.517 [0.506, 0.528] | 0.533 [0.463, 0.624] |
| 1.20× | 0.525 [0.485, 0.563] | 0.567 [0.555, 0.578] | 0.608 [0.491, 0.786] |

- **Verdict: C2 holds on every task.** Every upper bound is below 1 at all three margins (largest
  0.786, DMTS at 1.20×); no interval contains 1 at 1.05×.
- 1,000 active units at N ≈ 14,200 (3BFF), 7,300 (CDDM), 9,200 (DMTS) on the 1.05× fits.
- DMTS now has a fourth size (N = 4000, relayed from Spock) but only 1 of its 3 networks reads out:
  s3271883334 never learned (R² 0.40), s2048016264's loss ends below its fitted floor. Its interval
  therefore resamples N = 4000 as a single value; the exponent's interval understates the spread
  there. Also excluded: DMTS N = 1000 s2946835052 (loss ends below floor) and N = 2000 s597040509
  (never learned, R² 0.42). The DMTS exponent at 1.05× is 0.55 (board had 0.58 on three sizes).
- Mean 1.05× crossing (ED Fig. 3b): 3BFF 59-65k for N = 500-4000 (119k at N = 100); CDDM 26.7-32.5k
  (21.5k at 100); DMTS 47-79k (111k at 100).

**Panel d (N = 1000, more units inactive between t\* and the end).** 3BFF +49, +96, +119, +126 by
200k (four networks); CDDM (std_g0_drift_bias0, 200k) +150, +150, +162; DMTS +267 (s86371460, read at
18.6k) and +31 (s1167683015, read at 75.7k) by 150k; s2946835052 has no read-out.

**C3 (panel e), the flat-loss test of entry 2026-10-15_0955.** Flat = |loss change| < 3% per 100k from
150k (5k-block medians); units lost per 100k = minus the least-squares slope of the active count.

| N | flat runs | lose ≥ 5 per 100k | units lost per 100k (flat runs) |
|---|---|---|---|
| 500 | 5 of 6 | 4 | 37.4, 13.6, 7.6, 66.6; 1.7 |
| 1000 | 6 of 7 | 5 | 22.5, 37.1 (both 300k), 26.6, 83.1, 19.9; −5.4 |
| 2000 | 1 of 3 | 1 | 56.6 (others still falling 3.6-4.3% per 100k) |
| 4000 | 3 of 3 | 3 | 120.9, 30.6, 101.1 |

- The rule was written for three runs per size ("2 of 3 flat runs"); with 5-6 flat runs it is read as
  a majority of the flat runs. **Units keep becoming inactive at N = 500, 1000 and 4000; N = 2000 is
  untestable (one flat run).** 13 of the 15 flat runs lose 7.6-121 units per 100k.
- The non-flat N = 1000 run is the straggler s3374117298 (loss +5.6% per 100k, 121 units per 100k).
- Differences from entry 0955 (5 of 6 flat at N = 1000, 22-83 lost): the claim-1 run s1258717025 is
  now flat (−2.8% per 100k, was just above 3%), and N = 4000 is new.
- Not recomputed here: the last-100k rate of the two 300k runs (11-22 per 100k in entry 0955).

**ED Fig. 3c (rule against the snapshot oracle, 26 runs with kept weights).** 22 have both crossings
(3 have no rule read-out: two 75k-only flip-flop runs and one CDDM winp0.5; one DMTS run has no oracle
crossing). 3BFF 6 of 6 within 2%; DMTS −4% and −19%; CDDM 14 runs from −33% to +64% (controls +47%,
+63%, +1%), as in entry 2026-10-15_2054.

**Paper.** Figure 2, Results 2, Methods "Read-out at the loss floor and at a fixed iteration",
Extended Data Fig. 3 and Extended Data Table 2 (61 rows: every network of Figures 1 and 2) are in
dead-relu-paper PR for #6.
