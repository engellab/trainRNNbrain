## 2026-11-27 22:39 — Result: the entropy set point at λ = 0.1 across network sizes passes every size test run; active units keep falling past 75,000 iterations

Branch `wp/entropy-sizes`. Rules: `2026-11-25_1241_rules-entropy-setpoint-sizes.md` (verdicts S and M, question 2) and
`2026-11-25_1259_rules-entropy-sizes-rate-penalty-test.md` (test B). Submission: `2026-11-25_1329_entropy-setpoint-sizes-pilot-and-submission.md`.

**No cell fails.** The entropy set point at λ_ent = 0.1 passes the size test (S) and the paper's technique criterion (M) in
every cell run: 3BFF (3-bit flip-flop) and CDDM (context-dependent decision making) at N = 500 and 2,000 units, and Maze at
N = 500. The statement "the term increases the number of active units at every size tested" holds on all three tasks, and
so does "the term meets the criterion at every size tested". The sizes tested are N = 500 to 2,000 on 3BFF and CDDM, and on
Maze N = 500 here, with N = 1,000 and 2,000 from entry `2026-11-22_1808`. N = 4,000 was cut on both supervised tasks for
budget (below), so nothing here speaks to N = 4,000. On the rate-penalty size test (B), the pass conditions hold on 3BFF
and on Maze at the sizes judged. On CDDM the outcome is inconclusive: the fold change at the floor crossing is below 1.5.
After 75,000 iterations the number of active units keeps falling in every network, in the controls as well. Every entropy
network stays above every control at every read point up to the end of training.

Terms as in the rules: a unit is *active* if its participation, std(r) + q0.9(|r|) on a noise-free pass, is at least 5% of
the 95th percentile over the network's units. The *floor crossing* is the first iteration from which the smoothed noisy
training loss stays at or below 1.05 times its fitted floor. *Fold change* is the mean number of active units in the entropy
networks divided by the controls' mean at the same read point. The *count condition* holds when every entropy network has
more active units than every same-size control.

### What ran

- Della, gpu-short, from the clone `~/trainRNNbrain_entsizes` pinned at c27403c. All training jobs completed within their
  time limits (Della jobs 18377530-18377543); none was rerun, continued by hand or cancelled. The Della connection was
  down for about a day while the chain ran. The jobs did not depend on it.
- Reads 18377532-18377546 and the verdict 18377549 ran as dependent jobs on Della. The verdict was run again on the Mac
  from rsynced reads, and its text output is identical (`firstmate/data/entropy-sizes/reads_final/verdict_*.txt`).
- **GPU-hours: 216 used of the 300 cap** (the rules' prediction was 228). 3BFF N = 500: 28; 3BFF N = 2,000: 109; CDDM
  N = 500: 12; CDDM N = 2,000: 28; Maze N = 500: 38; pilots: 1.2. The start-of-job GPU-hour check never refused a job.
- **Cuts, as the budget rule set them before any run.** 3BFF N = 4,000 was cut in the rules: it needs about 345 GPU-hours
  at the controls' speed, and the pilot measured 2.21 s per iteration, above the 0.5 s that would have reopened the cut.
  CDDM N = 4,000 was cut after the pilot: with it, the total with a 5% margin would have been 338 GPU-hours at 100,000
  iterations, or 314 at 75,000.

### Checks

- **Gate** (the reader reproduces the controls' recorded values before any verdict): passes in every cell. Counts are
  exact; R² agrees to within 3.3 × 10⁻⁹; Maze counts and success are exact.
- **Check 2** (separately written, numpy only, no `common.py`; Della jobs 18377547 and 18377548). It recounts, network by
  network (controls included), the active units at the crossing, at 75,000 and at the end, plus A(t) at every read point of
  question 2. It also recomputes the Maze learned point and counts. Every value is equal to the reader's, in all 40
  supervised networks (the N = 1,000 cells included) and all 6 Maze networks. In one entropy network per cell, the noise-free R² from a numpy forward
  pass agrees with torch to within 8.2 × 10⁻⁹ (the threshold was 10⁻⁵). Output: `all checks pass`, twice.
- **Check 3** (each segment's saved config compared with its control's): the only differences are in keys listed in the
  rules. These are the term's keys, `task.seed`, `iter_offset` and `max_iter` at the joins, `init_draws: shared`,
  `sigma_w` 0, `optimizer` adam, `momentum` 0, snapshot keys, and `prune_args` keys, which have no effect with
  `prune_reinit: false`. The rules list how each control differs in training settings (segments and joins, the trial
  generator's seed, the code version, six controls at 3BFF N = 500, and CDDM controls' R² at 75,000 estimated from the
  loss). Nothing else differs.
- **Manipulation check** (median over units of |H_i − 0.75| bits, final weights): 0.004 to 0.013 in the entropy networks,
  against 0.25 to 0.72 in the controls.

### Question 1: verdicts per cell

| Cell | Active units at the crossing, entropy / controls | Fold | Active units at 75,000, entropy / controls | Fold | ΔR² at 75,000 | S | M |
|---|---|---|---|---|---|---|---|
| 3BFF N = 500 | 303, 298, 354 / 179-235 (six) | 1.55 | 255, 264, 302 / 178-219 | 1.38 | −0.0004 | passes | works |
| 3BFF N = 2,000 | 661, 733, 712 / 370, 385, 415 | 1.80 | 629, 635, 623 / 330, 364, 393 | 1.74 | +0.0008 | passes | works |
| CDDM N = 500 | 405, 401, 420 / 291, 302, 302 | 1.37 | 382, 385, 383 / 240, 242, 252 | 1.57 | −0.0007 | passes | works |
| CDDM N = 2,000 | 651, 670, 686 / 522, 543, 580 | 1.22 | 566, 570, 554 / 446, 449, 449 | 1.26 | +0.0028 | passes | works |

ΔR² is the entropy networks' mean held-out R² minus the controls' mean; M allows −0.01. Every network has a floor
crossing. At 1.10 and 1.20 times the floor the count condition also holds in every cell, so no looser margin changes S. The
smallest margin is at CDDM N = 500 at 1.20 times the floor: 412 active units in the lowest entropy network, against 390
in the highest control.

**Maze N = 500** (read at the learned point and at update 4,000; performance is probe success over 128 episodes):
- At the learned point, the entropy networks have 462, 482 and 438 active units, against 281, 259 and 279 in the controls
  (fold 1.69).
- At update 4,000 they have 407, 422 and 417, against 226, 226 and 240 (fold 1.80).
- Mean success at update 4,000 is 0.003 above the controls'; every network ends at 1.00.
- The entropy networks learned at updates 1,430, 1,050 and 2,100, the controls at 1,350, 1,280 and 1,750.
- S passes and M works.

**Prune-and-duplicate, judged by the same M on the same controls** (from its recorded reads), works in every cell. Its
fold change at 75,000 iterations is larger than the term's in every cell: 2.22, 3.35, 1.66 and 2.44 for 3BFF N = 500 and
2,000 and CDDM N = 500 and 2,000, and 1.85 on Maze at update 4,000. Its ΔR² is −0.0039, −0.0087, −0.0028 and +0.0006,
and +0.003 in success on Maze. The term's ΔR² is closer to zero than prune-and-duplicate's on three of the four
supervised cells; on CDDM N = 2,000 prune-and-duplicate's is closer (+0.0006 against +0.0028).

**Predictions.** The rules predicted S in every cell and M at N = 500 and 2,000. Both predictions held. The cell named as
least certain, 3BFF N = 2,000, has the largest fold change of the four.

### Test B, the rate-penalty size test, with λ_ent = 0.1 in place of the rate penalty

A network is *stable* if it has a floor crossing, fewer than 5% of its updates were skipped, and no saved loss value is
non-finite.

| Task, N | Stable | Fold at the crossing | ΔR² at the crossing | Outcome at this N |
|---|---|---|---|---|
| 3BFF 500 | 3 of 3 | 1.55 | +0.0000 | pass conditions hold |
| 3BFF 2,000 | 3 of 3 | 1.80 (noisy synapses: 1.54) | +0.0003 | pass conditions hold |
| CDDM 500 | 3 of 3 | 1.37 | +0.0041 | not a pass (fold below 1.5) |
| CDDM 1,000 (existing, 75,000 iterations) | 3 of 3 | 1.40 (noisy synapses: 1.11) | +0.0007 | not a pass, no fail condition |
| CDDM 2,000 | 3 of 3 | 1.22 | +0.0043 | not a pass, no fail condition |
| Maze 500 | 3 of 3 | 1.69 (at the learned point) | success +0.003 | pass conditions hold |
| Maze 2,000 (existing, entry `2026-11-22_1808`) | 3 of 3 | 1.92 (871 against 455) | success +0.008 | pass conditions hold |

- **3BFF: the pass conditions hold at N = 500 and 2,000, and no fail condition holds at N = 2,000.** Skipped updates:
  998 of 200,000 (0.5%) in one N = 500 network and 499 (0.25%) in one N = 2,000 network; none in the other four. No
  saved loss value is non-finite in any network. The statement over every N needs N = 1,000 and N = 4,000. N = 4,000 is
  cut. At N = 1,000 the existing 75,000-iteration networks are reported, not judged, as the rules set. Only 1 of those 3
  networks has a floor crossing (fold 1.60 at that one crossing, below noisy synapses' 2.03). Applied to them, the test
  would fail by its stability condition. The rules exempt them because they were trained for 75,000 of the controls' 200,000 iterations.
  `entropy-prune-combo` trained N = 1,000 entropy networks to 175,000 iterations, but its result
  (`2026-11-27_1048`) does not read floor crossings. So the 3BFF N = 1,000 test at the full budget has not been read.
- **CDDM: inconclusive.** All nine networks are stable, with no skipped update. R² at the crossing is above the controls'
  at every N. The fold change at the crossing is 1.22 to 1.40, below the 1.5 a pass needs, and above noisy synapses' 1.11 at
  N = 1,000, so no fail condition holds. The rules expected this outcome.
- **Maze: the pass conditions hold at N = 500 and 2,000.** No network skipped a minibatch step or restored a rollout. Maze
  N = 1,000 was trained for 2,000 updates only. Its 3 networks learned by update 2,000 (at 1,610, 1,470 and 1,000), and at
  update 2,000 they have 1.54 times the controls' active units. Their counts at the learned point were not read, and
  success at update 4,000 does not exist.
- **The lenient reading of a finite loss** (the run's folder is not named `nan`) gives the same outcome in every cell.

**Prune-and-duplicate under test B** (from published numbers, entry `2026-11-25_1259`, unchanged). On 3BFF it **fails**
at N = 4,000: only 1 of 3 networks has a floor crossing (Supplementary Table 4), and the test needs 2. On CDDM it is
inconclusive: the fold at N = 500 is 1.37. On Maze it is inconclusive: the folds at the learned point are 1.50 at N = 500
and 1.45 at N = 1,000. This answers the referee's point that prune-and-duplicate was never put through the test: by the
paper's own table it fails on 3BFF.

### Question 2: do active units keep falling past 75,000 iterations?

A(t) is the median number of active units over the probes in (t − 2,000, t]. **The fall continues in every cell, in the
entropy networks and in the controls:** A(end) < A(75,000) in 3 of 3 entropy networks in all four cells, and in every
control.

| Cell | Entropy networks, mean A(75,000) → A(end) | Change | Controls, mean A(75,000) → A(end) | Change |
|---|---|---|---|---|
| 3BFF N = 500 (end 200,000) | 277 → 202 | −27% | 201 → 147 | −27% |
| 3BFF N = 2,000 (end 200,000) | 638 → 453 | −29% | 361 → 252 | −30% |
| CDDM N = 500 (end 100,000) | 383 → 373 | −3% | 248 → 228 | −8% |
| CDDM N = 2,000 (end 100,000) | 568 → 553 | −3% | 445 → 411 | −8% |

- On 3BFF, entropy networks and controls lose the same share of their active units from 75,000 to 200,000 iterations.
  On CDDM, entropy networks lose about 3% from 75,000 to 100,000, and controls about 8%.
- **Is it still falling at the end (3BFF, units lost per 100,000 iterations over probes from 150,000)?** At N = 500,
  entropy networks lose 23.6, 0.2 and 34.9 (decreasing, at least 5, in 2 of 3); controls lose −1.5 to 66.6 (4 of 6). At
  N = 2,000, entropy networks lose 109.2, −16.4 and 47.2 (2 of 3); controls lose 21.8 to 56.6 (3 of 3).
- **The verdict survives longer training.** The count condition on A(t) holds at every read point: 50,000 to 200,000 on
  3BFF, and 50,000 to 100,000 on CDDM. M at the end of the budget works in every cell, with ΔR² between −0.0006 and
  +0.0008. Folds at the end: 1.40 (3BFF N = 500), 1.79 (3BFF N = 2,000), 1.67 (CDDM N = 500) and 1.33 (CDDM N = 2,000).
- The rules predicted that the fall continues in every cell, the controls included, and that the count condition holds at
  every read point. Both predictions held.

### Figures

`img/internal_figures/entropy_sizes_traces.{pdf,png}`: the number of active units over training per network, entropy
networks (blue) and controls (grey), one panel per supervised cell. Dots mark the floor crossings; a vertical line marks
75,000 iterations.

`img/internal_figures/entropy_sizes_folds.{pdf,png}`: the fold change against N at the crossing and at 75,000 iterations,
entropy set point (blue) and prune-and-duplicate (orange). The N = 1,000 entropy networks were seen before the rules. They
are drawn as open circles, for description only, and are left out of the mean line. At the 3BFF crossing only one of them has a
crossing. One change was made to the reader's figure code after the reads: the N = 1,000 points were at first left off the
crossing panels and shown filled at 75,000. No verdict depends on the figure.

### Reproduce

```
python trainRNNbrain/experiments_and_analysis/entropy_sizes_read.py verdict <reads dir> \
    --recorded <repo>/data --fig img/internal_figures/entropy_sizes
```

Reads: `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/entropy_sizes_reads/` (Della), copied to
`firstmate/data/entropy-sizes/reads_final/`. Networks: `trained_RNNs/NBitFlipFlop_entropy_setpoint{,_c200000}/`,
`CDDM_entropy_setpoint/` and `Maze_interventions` (N = 500, `ent0.1`, seeds 1-3) under the Della data tree.
