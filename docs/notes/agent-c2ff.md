# agent-c2ff: claim C2, 3-bit flip-flop size series at the loss floor (2026-10-13)

Read-out: common.noisy_floor on the joined warm-start chain (40k + 35k + 125k = 200k), which gives
the first iteration where the running median (2,001 iterations wide) of the NOISY loss reaches 1.05x
the run's own fitted floor. Active rule: participation >= 5% of q95, from the trace. Script:
`trainRNNbrain/experiments_and_analysis/c2_ff_floor.py`. Della outputs are in
`/scratch/gpfs/TENGEL/pt1290/agent_c2ff/` (c2_ff_floor.txt, c2_diag.txt, c2_diag_cells.txt).

## Size series (ff_revive_c200000)

| N | read / dropped | crossings | active at crossing, mean ± SD |
|---|---|---|---|
| 500 | 6 / 0 | 37,744–111,401 | 201 ± 19 |
| 1000 | 2 / 1 (never reaches the bar) | 63,679–65,703 | 291 ± 5 |
| 2000 | 3 / 0 | 54,334–72,831 | 387 ± 24 |
| 4000 | still training (FFtop4000), no cell on disk | | |

PROVISIONAL exponent, 3 sizes, fit on cell means: active ∝ N^0.47 (CDDM N^0.46, DMTS N^0.51).
The crossings and counts agree with the earlier numbers (N = 500: 6 of 6 at 38–111k; N = 1000: 2 of 3
at 64–66k). Probes span the whole chain (0–199,900, absolute iterations), so a count taken at an
absolute crossing is read at the right time.

## Fitted floor vs the smoothed minimum

In 8 of 11 runs the smoothed loss at 200k is BELOW the fitted floor, by up to 8%. So the fitted
floor is not an asymptote, and where it is high the crossing fires early. As a re-read I took
1.05x the minimum of the smoothed loss. This re-read is backward-looking, so it is a robustness
check and not a replacement rule.

| N | seed | fitted floor | end / floor | fitted-floor crossing (active) | 1.05x-min crossing (active) |
|---|---|---|---|---|---|
| 500 | 2489235792 | 0.0346 | 0.993 | 56,138 (187) | 62,307 (186) |
| 500 | 893469585 | 0.0370 | 0.922 | 37,744 (238) | 57,118 (237) — fit's beta at its upper bound, 3.0 |
| 500 | 822903475 | 0.0340 | 1.005 | 56,234 (199) | 57,177 (212) |
| 500 | 822939552 | 0.0342 | 0.995 | 60,739 (197) | 69,617 (202) |
| 500 | 2484557599 | 0.0326 | 1.047 | 111,401 (197) | 57,121 (218) |
| 500 | 2545439171 | 0.0333 | 1.021 | 75,796 (189) | 62,168 (196) — fit's tau at its lower bound, 100 |
| 1000 | 3374117298 | 0.0322 | 1.092 | dropped | 59,976 (277) |
| 1000 | 1914910909 | 0.0337 | 1.016 | 65,703 (287) | 66,307 (288) |
| 1000 | 3866906883 | 0.0339 | 1.004 | 63,679 (294) | 76,939 (243) |
| 2000 | 4145304148 | 0.0344 | 0.986 | 63,657 (387) | 70,361 (393) |
| 2000 | 2422919889 | 0.0342 | 0.995 | 54,334 (410) | 58,399 (409) |
| 2000 | 620846234 | 0.0338 | 1.008 | 72,831 (363) | 71,080 (360) |

Pre-registered test: the exponent depends on the floor definition if the re-read moves it by more
than 0.1. Re-read means: 208 ± 18, 269 ± 24, 387 ± 25, giving N^0.45 against N^0.47. The shift is
0.02, so the test PASSES for the size series.

## Claim-1 runs (c1_snapshots_c200000, N = 1000 control; used by Figure 1 panels d/e)

| seed | fitted floor | end / floor | fitted-floor crossing (active) | 1.05x-min crossing (active) |
|---|---|---|---|---|
| 2461724787 | 0.0415 | 0.835 | 26,949 (352) | 52,159 (341) — beta at its upper bound, 3.0 |
| 4113307577 | 0.0371 | 0.914 | 39,142 (337) | 59,951 (292) — beta at its upper bound, 3.0 |
| 1258717025 | 0.0338 | 1.022 | 69,050 (266) | 67,348 (266) |

In two of the three runs the stretched-exponential fit has its stretch exponent beta pinned at the
upper bound (3), with the floor 9–18% above where the loss ends. Their fitted-floor crossings
(27k, 39k) fire 21–25k iterations before the 1.05x-minimum crossings. For seed 4113307577 the
active count differs by 45 units (337 vs 292). noisy_floor does not catch a fit pinned on the beta
or tau bound; it only checks the floor's own lower bound. The PI decides whether the rule changes;
common.noisy_floor is untouched.

## N = 1000 straggler (seed 3374117298)

Smoothed loss over its fitted floor: 1.078 at 75k, 1.058 at 125k, 1.055 at 175k, 1.092 at 199k. It
sits just above the 1.05 bar for ~100k iterations, then rises at the end. The earlier "~7%" figure
is `late_rise` (median of the last 10% over the floor), 1.074. Continuing it to 300k is 100k more
iterations × 0.24 s ≈ 6.7 GPU-hours (one job, about 8 h with overhead). Its loss has been flat at
~1.055x for 100k iterations, and the floor is refitted on the longer series, so a crossing is not
assured. Not submitted.

## fig_paper_F1.py (commit 458faf0)

SCALING's flip-flop entries now point at the c200000 cells. panel_c requires a floor read-out at
every LISTED size: before, a size with no runs on disk dropped out and the series was still drawn
at the floor. Until N = 4000 lands, the flip-flop draws at end of training, labelled. Figure 1 was
not rebuilt.
