# agent-c3: B5 with SGD at weight decay 0, and the flip-flop's unit count while its loss is flat

Branch `agent-c3` (from `restore-oct5` 02ddcc4). Started 2026-10-15 09:07.

## Tests written before any number was computed (2026-10-15 09:30)

### Item 1: B5, SGD without weight decay

The claim under test (C3, 14 Oct): "SGD keeps more units only where it converges to a worse
solution." Read at each run's 1.05x noisy-loss floor (R² estimated from the loss ratio) and at 200k.

- **Falsified if** any SGD weight-decay-0 arm has, at the floor or at 200k, R² within 0.01 of the
  standard Adam arm (≥ 0.94 at the floor, where Adam wd 1e-6 reads 0.950-0.951; ≥ 0.943 at 200k,
  where it reads 0.952-0.953) AND more than 1.5x Adam's active count (> 400 active, against Adam's
  237-271 at the floor and 169-203 at 200k), in at least 2 of its runs.
- **Holds if** every SGD weight-decay-0 arm with more than 400 active units has R² at least 0.02
  below Adam's, and every arm within 0.02 of Adam's R² has fewer than 400 active units.
- Anything else (for example a run in between) is reported as it is, with no verdict forced.

### Item 2: units lost while the flip-flop's loss is flat

Runs: the standard 3-bit flip-flop networks that reach 200k or 300k (claim-1 N = 1000 lineages
`NBitFlipFlop_c1_snapshots` → `_c200000` → `_c300000`; `NBitFlipFlop_ff_revive` → `_c75000` →
`_c200000` at N = 500, 1000, 2000), each read along its whole warm-start chain.

Per run: the noisy loss over 150k → end as 5k-block medians (change from the first block to the
last, in % per 100k); the active count at the floor, 150k, 200k and 300k; units lost per 100k after
150k (least-squares slope of the active count on the participation probes at ≥ 150k).

- A run's loss counts as **flat** when its 5k-block median changes by less than 3% per 100k over
  150k → end (agent-c4b's three claim-1 runs read -2.6%, +2.8%, +0.2%).
- **Units stop dying when the loss is flat** if, in at least two thirds of the flat-loss runs at
  each size, the run loses fewer than 5 units per 100k.
- **Units keep dying while the loss is flat** if, in at least two thirds of the flat-loss runs at
  each size, the run loses 5 or more units per 100k.
- A size with neither outcome is reported as mixed. 5 units per 100k is 1% of N at N = 500 and
  0.25% at N = 2000; the absolute bar is kept because it is the one written down first.

## Item 2 result: units keep dying while the flip-flop's loss is flat (read 2026-10-15 ~09:45)

Script `trainRNNbrain/experiments_and_analysis/c3_flat_loss_units.py`, run on a local copy of the
traces, losses and configs of all 44 folders in the 15 lineages (Della
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/`; each run's parents linked in as
`parent_*` so `common.warm_start_chain` joins the loss). The final folder's participation trace
already holds the whole history (the 300k trace equals its 200k parent's on the first 200k, exactly).
Rule health: q50/q95 after 40k is at most 0.003 at N ≥ 1000 and 0.044 at N = 500, under the 0.05 at
which the 5%-of-q95 rule breaks.

| size | run (seed) | end | floor (k) | active: floor / 150k / 200k / 300k | loss %/100k, 150k→end | units lost /100k after 150k (± SE) |
|---|---|---|---|---|---|---|
| 1000 claim-1 | 3204992658 | 300k | 62* | 291 / 231 / 194 / 183 | -0.3 | 22.5 ± 0.5 |
| 1000 claim-1 | 3194339211 | 300k | 52* | 341 / 221 / 191 / 169 | -0.1 | 37.1 ± 0.4 |
| 1000 claim-1 | 1258717025 | 200k | 69 | 266 / 202 / 183 / - | -2.8 | 26.6 ± 0.9 |
| 1000 ff_revive | 3374117298 | 200k | none (never reaches the bar) | - / 235 / 181 / - | +5.6 (not flat) | 120.7 ± 3.0 |
| 1000 ff_revive | 1914910909 | 200k | 66 | 287 / 206 / 170 / - | +2.1 | 83.1 ± 2.8 |
| 1000 ff_revive | 3866906883 | 200k | 64 | 294 / 189 / 178 / - | -2.5 | -5.4 ± 1.9 (gains) |
| 500 | 2489235792 | 200k | 56 | 187 / 150 / 130 / - | +1.4 | 37.4 ± 2.0 |
| 500 | 893469585 | 200k | 57* | 237 / 135 / 142 / - | -1.1 | 13.6 ± 0.8 |
| 500 | 822903475 | 200k | 56 | 199 / 151 / 135 / - | -2.0 | 7.6 ± 2.4 |
| 500 | 822939552 | 200k | 61 | 197 / 149 / 154 / - | -0.7 | 1.7 ± 0.6 |
| 500 | 2484557599 | 200k | 111 | 197 / 192 / 153 / - | +0.2 | 66.6 ± 4.3 |
| 500 | 2545439171 | 200k | 76 | 189 / 177 / 156 / - | -3.5 (not flat) | -1.5 ± 2.0 |
| 2000 | 4145304148 | 200k | 64 | 387 / 270 / 255 / - | -0.6 | 56.6 ± 2.1 |
| 2000 | 2422919889 | 200k | 54 | 410 / 264 / 250 / - | -4.3 (not flat) | 21.8 ± 2.0 |
| 2000 | 620846234 | 200k | 73 | 363 / 270 / 253 / - | -3.6 (not flat) | 33.8 ± 1.4 |

(* floor read at 1.05x the smoothed minimum: the fit's beta sits on its bound. Single-probe counts
at 150k/200k/300k jitter by ±2-6 units probe to probe; the rate is fitted over every probe.)

Verdict of the written test (flat = |loss change| < 3% per 100k; dying = ≥ 5 units per 100k):
- N = 1000: 5 of 6 runs flat; 4 of the 5 lose ≥ 5 units per 100k (22-83). **Keep dying.**
- N = 500: 5 of 6 flat; 4 of the 5 lose ≥ 5 per 100k (7.6-67). **Keep dying**, by one run: two of
  the five lose under 8 per 100k, and an independent estimate (mean of 150-160k minus mean of the
  last 10k, written separately on the raw traces) reads 822903475 at 0.0, so 3 of 5 and "mixed".
- N = 2000: only 1 of 3 runs is flat by the bar (it loses 57 per 100k); the other two still fall
  3.6-4.3% per 100k and lose 22-34 units per 100k. **Keep dying**, on one flat run.
- The loss of units slows: on the two 300k claim-1 lineages, 150k→200k loses ~63 per 100k (window
  means on their 200k parents) and 200k→300k ~11-22 per 100k (probe counts), at a loss that does not
  change (-0.1%, -0.3% per 100k). It has not stopped by 300k.

## Item 1 result: SGD without weight decay does not change the B5 conclusion (read 2026-10-15 ~10:30)

Script `trainRNNbrain/experiments_and_analysis/b5_read.py` on a local copy of
`spock:~/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_b5_optimizer` (synced 09:10, lr 10 cells
excluded). Finished SGD wd 0 runs: lr 0.03, 0.1, 0.3 (3 each), lr 1 (1 of 3; two were still
running on Spock, jobs 6928465_38-39), lr 3 none (pending). The arms read on 14 Oct reproduce to the
third decimal. Floor R² estimated from the loss ratio (`fig_knobs_at_floor.floor_r2`, 36 seeded
draws); 200k R² measured on the final weights (36 draws).

| arm | n | floor crossing | active at floor | R² at floor (est.) | active at 200k | R² at 200k |
|---|---|---|---|---|---|---|
| Adam, wd 1e-6 (standard) | 3 | 69-77k | 237-271 | 0.950-0.951 | 169-203 | 0.952-0.953 |
| Adam, wd 0 | 3 (2 at floor) | 125-166k | 225, 254 | 0.960-0.961 | 208-221 | 0.962 |
| AdamW, wd 1e-6 | 3 (2 at floor) | 117-118k* | 282, 292 | 0.961 | 203-226 | 0.962 |
| SGD lr 0.03, wd 1e-6 | 3 | 105-112k | 718-736 | 0.882-0.884 | 683-695 | 0.893-0.896 |
| SGD lr 0.03, wd 0 | 3 | 106-107k | 703-727 | 0.883-0.884 | 670-692 | 0.895 |
| SGD lr 0.1, wd 1e-6 | 3 | 97-120k | 483-585 | 0.901-0.905 | 451-523 | 0.909-0.914 |
| SGD lr 0.1, wd 0 | 3 | 89-105k | 524-555 | 0.888-0.902 | 470-513 | 0.901-0.913 |
| SGD lr 0.3, wd 1e-6 | 3 | 51-94k | 254-331 | 0.903-0.911 | 229-250 | 0.923-0.925 |
| SGD lr 0.3, wd 0 | 3 | 50-81k | 230-295 | 0.904-0.911 | 218-228 | 0.918-0.926 |
| SGD lr 1, wd 1e-6 | 3 | 20-87k* | 249-271 | 0.894-0.908 | 212-247 | 0.908-0.917 |
| SGD lr 1, wd 0 | 1 | 109k | 272 | 0.911 | 300 | 0.920 |
| SGD lr 3, wd 1e-6 | 3 (1 at floor) | - | - | fails | 198-260 | 0.30-0.44 |

(* includes a smoothed-minimum read.) Against the written test: no SGD arm, with or without weight
decay, comes within 0.02 of Adam's R² (best SGD 0.911 at the floor and 0.926 at 200k, against Adam's
0.950 and 0.952). The SGD arms with more than 400 active units (lr 0.03 and 0.1, both weight
decays) sit 0.045-0.07 below Adam at the floor. **The test holds; removing weight decay from SGD does
not change the conclusion.** At each SGD learning rate the wd 0 arm matches the wd 1e-6 arm in count
and R² to within the seed spread. The one finished lr 1, wd 0 run gains units after its floor
(272 → 300), the only B5 run that does; one run, so not read further.

Figure: `img/internal_figures/diag_b5_optimizer.{png,pdf}` (left: both at the floor; right: both at
200k; open triangles = SGD wd 0; the legend is built from every arm, with the number of runs that
have a floor read).
