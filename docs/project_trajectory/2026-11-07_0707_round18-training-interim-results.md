## 2026-11-07 07:07 — Round 18 training package, interim results by the committed rules: G11, G12, G18 complete; CDDM parity partly in; G17 not run

Rules applied: `2026-11-07_0110_round18-training-rules.md` (G11, G12, G17, G18), `2026-11-07_0136_round18-parity-rules.md`
(parity) and `2026-11-07_0259_round18-second-maze-calibration.md` (G17's maze). Read with
`trainRNNbrain/experiments_and_analysis/round18_training_read.py` at 6bee469: the `maze` part on the Mac on
the rsynced trees, the `cddm` part on Della (CPU job 17039418; the first attempt, 17036322, failed in 36 s
because the read worktree had no `data/c1_snapshots` link, so the kept snapshots were not found). Output:
`data/round18_training/round18_maze.json` and `round18_cddm.json`. Run folders copied to the Mac without the
optimizer-state files. Still running: the Maze parity arms (Spock 7397761, noisy synapses, about 12 h left;
7397762, dropout and rescaling, about 1.5 h), CDDM dropout 0.05 (Della 17031166) and the third networks of
CDDM σ_w 0.1 and 0.3 (17031165_18, _20). A final entry follows when they land.

**Terms.** *Probe*: a noise-free test of 128 Maze episodes from fixed starts; *success* is the share that
reaches the cheese. *Learned point*: the first probe from which success stays at or above 0.9 for five
probes, in a run whose last probe is also at or above 0.9. *Active unit*: participation
p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05·q95(p). *Floor crossing*: the first
iteration from which the smoothed noisy loss stays at or below 1.05 times its fitted floor until 1.2 times
that iteration.

### Jobs

| Cell | Cluster, job | State |
|---|---|---|
| G11 | Spock 7397701 (10 tasks) | all finished 2,000 further updates (4.0-5.3 h each) |
| G12 | Spock 7397702 (3 tasks) | all finished |
| G18 | Della 17030263, 17030264 | 6 of 6 COMPLETED (2.8-3.5 h) |
| CDDM parity | Della 17031165, 17031166, 17031167 | rescaling 3 of 3 and σ_w 4 of 6 finished; dropout running |
| Maze parity | Spock 7397761, 7397762 | running; rescale target 4 seed 2 stopped by the rollback guard at update 800 |
| Calibration | Spock 7397777 (39 CPU tasks) | finished; no maze passed (entry 2026-11-07_0421) |

### G12: the third learned Maze control, and the weight-decay test

| Seed | Success at update 2,000 | Learned at update | Active at the learned point | Active at update 2,000 | Kept snapshots |
|---|---|---|---|---|---|
| 4 | 0.953 | 1,530 | 427 | 394 | 20 (u00100-u02000); first after the learned point 70 updates later |
| 5 | 0.938 | 1,180 | 412 | 398 | 20; 20 updates later |
| 6 | 1.000 | 1,330 | 406 | 417 | 20; 70 updates later |

All three new networks learned. Each has 417 or fewer active units at update 2,000 (394, 398 and 417), so
by the rule **weight decay 1e-5 on Maze (370-417 active) no longer lies below every learned control**. Its
verdict changes from "decreases the number of active units" to "no change", and the claim that only weight
decay changes the number of active units on all three tasks loses its Maze direction. The learned
controls now number five (uniform series seeds 1 and 2, G12 seeds 4-6) and span 394-449 active units at
update 2,000. The snapshot check passes: 20 snapshots per run, the first at or after the learned point at
most 70 updates after it. The other settings' verdicts against the new range are for the analysis package
(G9, `read_maze_arms.py`); the status board gets this entry's weight-decay result.

### G11: the Maze size series at 4,000 updates (descriptive, no verdict)

| N | Seed | Success at 4,000 | Learned at update | Active at the learned point | At 2,000 | At 4,000 |
|---|---|---|---|---|---|---|
| 250 | 1 | 1.000 | 1,190 | 154 | 147 | 140 |
| 250 | 2 | 0.992 | 2,340 | 136 | 128 | 125 |
| 250 | 3 | 1.000 | 830 | 173 | 158 | 146 |
| 250 | 4 | 1.000 | 940 | 138 | 132 | 132 |
| 500 | 1 | 1.000 | 1,350 | 281 | 265 | 226 |
| 500 | 2 | 0.992 | 1,280 | 259 | 241 | 226 |
| 500 | 3 | 1.000 | 1,750 | 279 | 276 | 240 |
| 1,000 | 1 | 1.000 | 1,050 | 451 | 426 | 360 |
| 1,000 | 2 | 1.000 | 760 | 517 | 449 | 404 |
| 1,000 | 3 | 1.000 | 1,750 | 428 | 415 | 317 |

All ten runs end at success 0.99-1.00 and count as learned on the joined record. Two of them had not
learned by update 2,000. N = 250 seed 2 (success 0.79 at 2,000) learns at update 2,340. N = 1,000
seed 3 held 0.91-0.98 near update 1,750 and was at 0.73 at update 2,000; on the joined record its last
probe is 1.00, so the rule places its learned point at 1,750. In every one of the ten runs the number of
active units at update 4,000 is at or below the number at update 2,000, and below the number at the
learned point. The largest drops between updates 2,000 and 4,000 are at N = 1,000: 426 to 360, 449 to
404 and 415 to 317. The size exponent at one budget, the flat-return window and the later participation
are for the analysis package (G8, G10, G15, G16).

### G18: the CDDM transfer test (B4's rules)

Old networks: the three `CDDM_std_g0_drift_bias0` N = 1,000 networks trained 200k iterations on CDDM,
moved to `CDDMHyper` (CDDM's six inputs; five outputs: the cued choice, the motion sign, the colour sign
and their product), 75k iterations. Fresh networks: three untrained networks on `CDDMHyper`. R² is
held-out R² under the training noise, 36 draws, measured on the kept snapshot at or after the crossing for
all six runs.

| Source | Floor crossing | Active at transfer | Inactive at transfer (S₀) | Revived | Newly inactive | Active at the crossing | Active at 75k | S₀ active at 75k | R² at the crossing |
|---|---|---|---|---|---|---|---|---|---|
| old | 2,566 | 254 | 746 | 188 (25%) | 39 | 403 | 256 | 45 (6.0%) | 0.719 |
| old | 4,890 | 270 | 730 | 86 (12%) | 47 | 309 | 257 | 38 (5.2%) | 0.727 |
| old | 2,000 | 270 | 730 | 88 (12%) | 29 | 329 | 251 | 35 (4.8%) | 0.723 |
| fresh | 24,906 | 766 | 234 | 110 | 468 | 408 | 355 | 103 | 0.719 |
| fresh | 27,974 | 793 | 207 | 87 | 456 | 424 | 362 | 79 | 0.722 |
| fresh | 28,578 | 785 | 215 | 104 | 469 | 420 | 349 | 83 | 0.726 |

Each row adds up: active at transfer − newly inactive + revived = active at the crossing.

**Gate passes** at 1.05× the floor. The fresh networks have 417 ± 8 active units (mean ± SD) at their
crossing, 153 above the old networks' 265 at transfer; the bar is 3 SD = 25.

**Verdict at 1.05× (binding): between the bars, no verdict.** The "do not come back" bar needs fewer than
10% of S₀ revived in every old network. The old networks revive 12%, 12% and 25%. Their count at the
crossing (347 mean) is 70 below the fresh networks' 417, which is more than 3 SD (25). The "can be
recruited" bar needs 30% or more revived in 2 of 3 old networks, or the old count within 1 SD of the
fresh count; neither holds.

**At 1.10× and 1.20× the floor**, the CDDM check of the rules: the verdict is the same, between the bars.
The old networks revive 12-25% at both margins. The fresh networks cross earlier at the looser margins,
when more of their units are still active (520 and 704 mean).

**At 75k (reported, not binding): "do not come back".** The old networks keep 4.8-6.0% of S₀ active.
They end with 251-257 active units against 349-362 for the fresh networks, 101 below on the means; the bar
is 3 SD = 20. The floor verdict and the 75k verdict disagree.

R² does not differ between old and fresh networks: 0.719-0.727 against 0.719-0.726.

**What sets the disagreement, from the table.** The old networks reach their floor on the new task at
2,000-4,890 iterations, the fresh networks at 24,906-28,578. 2,000 is the first iteration the floor fit
uses, so one old network's crossing sits at the earliest possible point. B4's known risk was a transfer
run rejected because its loss falls before the fit starts; here the old runs are read, but at about a
tenth of the fresh runs' training time. Between their crossing and 75k the old networks lose most of
the units they had revived: S₀ active falls from 86-188 to 35-45. This comparison of read points is an
observation from these six runs. It has not been tested as a cause, and the rule's verdict stands as
written: no verdict at the floor.

### CDDM parity cells so far (the paper's criterion at 75k)

The control is `CDDM_paper_grid` arm control: 342, 334 and 333 active units, R² 0.8312, 0.8350 and 0.8349
(estimated from the loss ratio), mean 0.8337. To pass, every network needs more than 342 active units and
the mean R² must be at least 0.8237. The new networks' R² is measured on their 75k weights.

| Value | Networks read | Active | R² | Result |
|---|---|---|---|---|
| rescaling, target 4.0 | 3 of 3 | 231, 230, 224 | 0.827, 0.827, 0.827 | **fails** at every tolerance: fewer active units than every control; the second control (320-334) gives the same |
| σ_w = 0.1 | 2 of 3 | 316, 333 | 0.833, 0.832 | **fails**: two networks at or below 342, whatever the third shows |
| σ_w = 0.3 | 2 of 3 | 360, 380 | 0.834, 0.835 | both clauses hold for the two networks so far (R² 0.0008 above the control mean); waits for the third |
| dropout 0.05, β 4 | 0 of 3 | | | running |

All new CDDM networks start from independent W_out draws, while the controls used the old shared draws
(parity rules entry).

### G17

Not run: none of 15 candidate second mazes passed the difficulty check (entry 2026-11-07_0421). The options
are with the coordinator on dead-relu-paper#201.
