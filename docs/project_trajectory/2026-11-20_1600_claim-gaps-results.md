## 2026-11-20 16:00 — Claim gaps A, B and C: results by the pre-set rules

Rules, committed before any run (07c0a83): `2026-11-19_1749_rules-maze-duplicate-sizes.md` (gap A),
`2026-11-19_1749_rules-cddm-rate-penalty-sizes.md` (gap B), `2026-11-19_1749_rules-maze-rate-penalty-sizes.md`
(gap C). Ranking, and the gaps not run: `2026-11-19_1749_claim-gaps-ranking.md`. Jobs:
`2026-11-19_1756_claim-gaps-submitted.md`; correction on the Della cap: `2026-11-19_1841_...`.

**Verdicts.**
- Gap A, prune-and-duplicate on Maze at N = 500, 1,000 and 2,000: **increases the number of active units at every
  size** (count condition holds in 6 of 6 size-and-read cells).
- Gap B, the rate-penalty size test (B2) on CDDM at N = 500, 1,000 and 2,000: **supports** (stable training across
  sizes) at the binding margin, 1.05× the floor. Also supports at 1.10×; at 1.20× "neither" (see below).
- Gap C, B2 on Maze at N = 500, 1,000 and 2,000: **falsifies**, at N = 2,000 only: 1 of 3 runs stable, and mean
  probe success at update 4,000 is 0.148 below the controls'.

### Runs

All 42 Della tasks (jobs 17958685-17958691) ended COMPLETED between about 03:00 and 18:17 ET on 20 Nov. Every Maze run reached
update 4,000 (each log ends `done: 2000 updates`); none stopped, so no run was resumed. Stage 2 of each Maze run
started on its own stage-1 task with `aftercorr` (the per-task form of the rules' `afterok`). GPU time: 252 GPU-h
(Maze 222, CDDM 30), all on Della gpu-short; Spock was not used (saturated, ranking entry).

### Read

- Maze (gaps A, C): `claim_gaps_read.py maze` on the Mac (2dd8a6d), on copies of the three Della trees
  `trained_RNNs_rl_maze_cg_N{500,1000,2000}` (Adam states left out). The N = 1,000 parents in the copy are
  byte-identical to Della's `trained_RNNs_rl_maze_fz` (`rsync -c` dry run: no difference). Output:
  `data/claim_gaps/claim_gaps_maze.json`.
- CDDM (gap B): `claim_gaps_read.py cddm` as Della CPU job 18022656 from the read clone `~/trainRNNbrain_cgread`
  (2dd8a6d), set up as in the correction entry so that R² at the crossing is measured on a kept snapshot where one
  exists. Output: `data/claim_gaps/claim_gaps_cddm.json`.

**Independent check.** The scripts and their outputs are kept outside the repository, in
`firstmate/data/claim-gaps/proof/` (`oracle_maze.py`, `oracle_cddm.py`, `*_output.txt`). Both scripts were written
from the rules' text, without `common.py` or `heldout_r2.py`. The pass condition was exact equality for each run.
- *Maze.* The check joins each run's stage folders in update order, cuts at update 4,000 and finds the learned
  point from the definition. It gives the same learned update, R1, R2, success at 4,000, flagged share and last
  update in all 33 runs (18 new, 12 controls, 3 σ_w); 0 mismatches. In every new run the stored active count also
  equals a recount from the saved participation p (p ≥ 0.05·q95(p)) at updates 2,000 and 4,000. The verdicts
  recomputed from these values are the same as the reader's.
- *CDDM.* At the reader's crossing iteration, the check recounts the active units from `*ParticipationTrace.pkl`.
  It also confirms that the saved training loss is finite and takes the skipped updates from each Slurm log. The
  first version took the first probe after the crossing, and 6 of 9 control counts differed by 6-17 units. The
  reader takes the nearest probe (`common.read_run`); with the nearest probe all 18 runs match. Under the
  first-after choice the control means move by at most 3 units, and no verdict changes. Every run's loss is finite,
  and every new log ends with "Executed training". Skipped updates: 499 of 75,000 in N = 2,000 seed 3230350039,
  0 in the other eight networks. The verdicts recomputed from the per-run rows are the same at all three margins.
  The held-out R² values themselves were not recomputed independently.

### Gap A and gap C, Maze (per network)

R1 = active units at the learned point (the first probe from which probe success stays ≥ 0.9 for five probes, in a
run whose last probe is also ≥ 0.9); R2 = active units at update 4,000. Σ skipped / Σ restored: summed over 4,000
updates.

| Network | N | Seed | Learned update | R1 | R2 | Probe success at 4,000 | Σ skipped / Σ restored |
|---|---|---|---|---|---|---|---|
| control | 500 | s1 | 1,350 | 281 | 226 | 1.000 | 21 / 7 |
| control | 500 | s2 | 1,280 | 259 | 226 | 0.992 | 0 / 0 |
| control | 500 | s3 | 1,750 | 279 | 240 | 1.000 | 7 / 8 |
| prune-and-duplicate | 500 | s1 | 1,710 | 416 | 440 | 1.000 | 0 / 0 |
| prune-and-duplicate | 500 | s2 | 1,430 | 401 | 406 | 1.000 | 0 / 0 |
| prune-and-duplicate | 500 | s3 | 1,460 | 409 | 436 | 1.000 | 0 / 0 |
| rate penalty 0.1 | 500 | s1 | 1,900 | 500 | 500 | 0.984 | 0 / 0 |
| rate penalty 0.1 | 500 | s2 | 1,190 | 500 | 500 | 1.000 | 0 / 0 |
| rate penalty 0.1 | 500 | s3 | 2,710 | 500 | 500 | 0.969 | 0 / 0 |
| control | 1,000 | s1 | 1,050 | 451 | 360 | 1.000 | 12 / 8 |
| control | 1,000 | s2 | 760 | 517 | 404 | 1.000 | 0 / 0 |
| control | 1,000 | s3 | 1,750 | 428 | 317 | 1.000 | 0 / 0 |
| prune-and-duplicate | 1,000 | s1 | 760 | 699 | 750 | 1.000 | 0 / 0 |
| prune-and-duplicate | 1,000 | s2 | 950 | 683 | 800 | 1.000 | 0 / 0 |
| prune-and-duplicate | 1,000 | s3 | 710 | 648 | 765 | 1.000 | 0 / 0 |
| rate penalty 0.1 | 1,000 | s1 | 560 | 1,000 | 1,000 | 1.000 | 0 / 0 |
| rate penalty 0.1 | 1,000 | s2 | 760 | 1,000 | 954 | 1.000 | 0 / 0 |
| rate penalty 0.1 | 1,000 | s3 | 1,390 | 994 | 997 | 1.000 | 0 / 0 |
| control | 2,000 | s1 | 3,000 | 393 | 400 | 0.984 | 0 / 0 |
| control | 2,000 | s2 | 2,480 | 557 | 506 | 0.977 | 29 / 18 |
| control | 2,000 | s3 | 2,840 | 414 | 388 | 1.000 | 10 / 5 |
| prune-and-duplicate | 2,000 | s1 | 2,550 | 1,285 | 1,380 | 1.000 | 19 / 5 |
| prune-and-duplicate | 2,000 | s2 | 1,480 | 1,279 | 1,495 | 0.992 | 7 / 0 |
| prune-and-duplicate | 2,000 | s3 | 2,490 | 1,311 | 1,357 | 0.922 | 22 / 10 |
| rate penalty 0.1 | 2,000 | s1 | none | – | 2,000 | 0.898 | 0 / 0 |
| rate penalty 0.1 | 2,000 | s2 | none | – | 1,686 | 0.617 | 0 / 0 |
| rate penalty 0.1 | 2,000 | s3 | 2,430 | 1,999 | 1,332 | 1.000 | 0 / 0 |
| control, not binding | 1,000 | s4 | 1,530 | 427 | 373 | 1.000 | 16 / 12 |
| control, not binding | 1,000 | s5 | 1,180 | 412 | 336 | 1.000 | 17 / 12 |
| control, not binding | 1,000 | s6 | 1,330 | 406 | 314 | 1.000 | 3 / 1 |
| synaptic noise σ_w 1.0 | 1,000 | s1 | 1,950 | 456 | 409 | 1.000 | 13 / 5 |
| synaptic noise σ_w 1.0 | 1,000 | s2 | 1,230 | 458 | 455 | 0.969 | 5 / 2 |
| synaptic noise σ_w 1.0 | 1,000 | s3 | 1,950 | 448 | 425 | 0.961 | 5 / 3 |

**Gap A.** The count condition (every prune-and-duplicate network has more active units than every same-size
control) holds in all six cells. The fewest active units in any prune-and-duplicate network, compared with the
most in any control:

| N | R1: fewest in an arm network / most in a control | Fold change at R1 | R2: fewest / most | Fold change at R2 |
|---|---|---|---|---|
| 500 | 401 / 281 | 1.50 | 406 / 240 | 1.85 |
| 1,000 | 648 / 517 | 1.45 | 750 / 404 | 2.14 |
| 2,000 | 1,279 / 557 | 2.84 | 1,357 / 506 | 3.27 |

Against G12's three further N = 1,000 controls (s4-6, not binding) the condition also holds (fold change 1.63 at R1,
2.26 at R2). Success at update 4,000 (reported, not part of the direction): the prune-and-duplicate mean is within
0.016 of the controls' mean at every size (+0.003, 0.000, −0.016), and no network ends below 0.9 (lowest: N = 2,000
s3, 0.922). Verdict: **prune-and-duplicate increases the number of active units at every size on Maze**
(N = 500-2,000).

**Gap C.** Per size, by B2's Maze form:

| N | Stable runs | Multiplier (mean R1 / control mean R1) | Success at 4,000, arm mean minus control mean | Per-size result |
|---|---|---|---|---|
| 500 | 3 of 3 | 1.83 | −0.013 | passes |
| 1,000 | 3 of 3 | 2.14 | 0.000 | passes; synaptic noise's multiplier 0.98, below the arm's |
| 2,000 | 1 of 3 | 4.40 (seed 3 alone) | −0.148 | falsifies (stable runs and success) |

Verdict: **falsifies**. The rate penalty fails B2 on Maze at N = 2,000.

Why the N = 2,000 runs are not stable: none of the three skipped or restored an update, and none stopped. Two had
no learned point by update 4,000:
- Seed 2 never reached 0.9 probe success (highest 0.72; 0.62 at update 4,000).
- Seed 1 first reached 0.9 at update 3,480 and held it for at most 8 probes in a row. Its last probe is 0.898
  (115 of 128 episodes), so it has no learned point by the definition.

The controls at N = 2,000 learned at updates 2,480-3,000. On Maze the failure is a failure to learn within the
4,000-update budget at N = 2,000. On 3BFF, B2 failed differently: no run at N ≤ 1,000 reached a floor read.

The verdict does not depend on seed 1's borderline last probe. With seed 1 counted as learned, 2 of 3 runs are
stable, but the success clause still falsifies (mean 0.838 compared with 0.987). The success clause does depend on
seed 2: without it the difference would be −0.038.

Two further readings, reported without a threshold:
- Seed 3, the one learned run, had 2,000 of 2,000 units active at update 2,000 and 1,332 at update 4,000.
- At N = 500 every unit is active at both reads (500 of 500); at N = 1,000, 954-1,000 of 1,000.

### Gap B, CDDM (per network, at the binding margin 1.05×)

Crossing: the first iteration from which the smoothed noisy loss stays at or below 1.05 times its fitted floor until
1.2 times that iteration. For the penalised runs the floor includes the penalty term. Active units: at the probe
nearest the crossing.

| Network | N | Seed | Crossing (iteration) | Active units | Held-out R² | R² source | Skipped updates |
|---|---|---|---|---|---|---|---|
| control | 500 | 3291291930 | 26,935 | 291 | 0.827 | estimated | – |
| control | 500 | 2251851164 | 27,463 | 302 | 0.828 | estimated | – |
| control | 500 | 3565681970 | 25,772 | 302 | 0.828 | estimated | – |
| rate penalty 0.1 | 500 | 2784942107 | 15,664 | 500 | 0.824 | snapshot | 0 |
| rate penalty 0.1 | 500 | 3907653330 | 17,372 | 500 | 0.818 | snapshot | 0 |
| rate penalty 0.1 | 500 | 1885387082 | 17,856 | 500 | 0.820 | snapshot | 0 |
| control | 1,000 | 3833537590 | 27,259 | 436 | 0.827 | estimated | – |
| control | 1,000 | 1272763220 | 26,535 | 420 | 0.829 | estimated | – |
| control | 1,000 | 457853 | 23,929 | 437 | 0.828 | estimated | – |
| rate penalty 0.1 | 1,000 | 3986057758 | 19,075 | 1,000 | 0.822 | snapshot | 0 |
| rate penalty 0.1 | 1,000 | 3681082541 | 19,717 | 1,000 | 0.825 | snapshot | 0 |
| rate penalty 0.1 | 1,000 | 1954623883 | 18,747 | 1,000 | 0.818 | snapshot | 0 |
| control | 2,000 | 3640167262 | 26,933 | 543 | 0.832 | estimated | – |
| control | 2,000 | 575781518 | 31,147 | 522 | 0.831 | estimated | – |
| control | 2,000 | 4259132547 | 26,511 | 580 | 0.826 | estimated | – |
| rate penalty 0.1 | 2,000 | 1091721975 | 20,640 | 2,000 | 0.826 | snapshot | 0 |
| rate penalty 0.1 | 2,000 | 1503474172 | 19,831 | 1,997 | 0.809 | snapshot | 0 |
| rate penalty 0.1 | 2,000 | 3230350039 | 32,521 | 2,000 | 0.808 | snapshot | 499 |
| synaptic noise σ_w 1.0 | 1,000 | 3139128930 | 28,026 | 478 | 0.814 | estimated | – |
| synaptic noise σ_w 1.0 | 1,000 | 145705583 | 30,337 | 471 | 0.824 | estimated | – |
| synaptic noise σ_w 1.0 | 1,000 | 2925113920 | 27,453 | 480 | 0.802 | estimated | – |

Per size and margin (multiplier = arm mean active / control mean active; ΔR² = arm mean minus control mean):

| Margin | N = 500: stable, multiplier, ΔR² | N = 1,000 | N = 2,000 | σ_w multiplier at 1,000 | Verdict |
|---|---|---|---|---|---|
| **1.05× (binds)** | 3/3, 1.68, −0.007 | 3/3, 2.32, −0.006 | 3/3, 3.65, −0.016 | 1.11 | **supports** |
| 1.10× | 3/3, 1.58, −0.006 | 3/3, 2.07, −0.002 | 3/3, 2.98, −0.003 | 0.98 | supports |
| 1.20× | 3/3, 1.31, −0.005 | 3/3, 1.66, −0.006 | 3/3, 2.30, +0.005 | 0.87 | neither |

No network has a non-finite loss (0 `nan` folders in every cell). At 1.05× every run is stable, every multiplier is
at least 1.5 and every ΔR² is within 0.02. The closest value to a bar is ΔR² at N = 2,000, −0.016 compared with the
limit of −0.02. Verdict: **supports**. The rate penalty passes B2 on CDDM at N = 500-2,000.

The rules require this entry to say so when a looser margin gives another verdict: at 1.20× the verdict is
"neither". At N = 500 the multiplier is 1.31, below 1.5, because at 1.20× the controls cross earlier (iterations
3,780-4,234), while 377-390 units are still active. Nothing falsifies at any margin.

Known limitation, fixed in the rules: the penalised networks' R² is measured on a kept snapshot (every 1,000
iterations). The controls and the σ_w networks kept no snapshots, so their R² is estimated as
1 − (L(t*)/L_end)(1 − R²_end), as in B2 and in the existing N = 1,000 rate-penalty verdict.

### What the results do to the claims

- **Prune-and-duplicate increases the number of active units across sizes** (Results 5): holds on all three main
  tasks: 3BFF and CDDM over N = 500-4,000 (G6 and the earlier size series), Maze over N = 500-2,000 (gap A). Maze at
  N = 4,000 is untested.
- **The rate penalty alone fails the stable-training test across sizes** (B2): holds on 3BFF and on Maze, and not
  on CDDM.
  - On CDDM the same test passes at N = 500-2,000: all 9 networks are stable, have 1.7-3.6 times as many active
    units as the controls, and have R² within 0.016 of them.
  - On Maze it fails at N = 2,000. Two of three networks had not learned the task by update 4,000, and mean success
    is 0.15 below the controls'. No update was skipped. On 3BFF, by contrast, B2 failed through runs without a floor
    read at N ≤ 1,000 and through an R² cost of 0.024 at N = 2,000.
  - The claim stays task-scoped: "on 3BFF and Maze, not on CDDM", with a different failure on each of the two tasks.
    N = 4,000 is untested on CDDM and Maze.
- **Few inactive units come back after transfer** and **DMTS continued decrease**: not run (ranking entry). The
  transfer claim stays 3BFF-only, and the DMTS result stays as it is.
