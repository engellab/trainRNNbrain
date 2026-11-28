## 2026-11-27 08:04 — Result: the shared initial draws do not change the number of active units on 3BFF or CDDM; Maze unresolved

Read by the rules in `2026-11-25_1303_rules-independent-init.md`, as corrected by
`2026-11-25_1307_correction-independent-init-maze-guards.md` and amended by
`2026-11-25_1312_rules-independent-init-unattended-chain.md`. Branch `wp/independent-init`. Terms as in the
rules entry: *3BFF*, the 3-bit flip-flop task; *CDDM*, the context-dependent decision-making task; *draw mode*,
`model.init_draws` (`shared`: the generator is seeded again before each of W_rec, W_inp and W_out, so the three
matrices start with the same numbers; `independent`: one generator supplies all three in order). Both modes give
the same initial W_rec, so the test varies only the initial W_inp and W_out.

### Verdicts

**3BFF: dependence not detected. CDDM: dependence not detected.** Three networks per task, N = 1,000 units,
default settings, independent draws, compared with every shared-draw default network of N = 1,000 on Della
(13 for 3BFF, 29 for CDDM). Every new network lies inside the bounds at both read points.

| Task, read point | New networks (active units of 1,000) | Reference: n, range, mean | Bounds | Mean difference, 95% Welch interval | Verdict |
|---|---|---|---|---|---|
| 3BFF, 75,000 iterations (primary) | 248, 273, 302 | 13, 236-309, 274 | 222-323 | 0 (−56 to +56) | not detected |
| 3BFF, floor crossing | 253, 264, 322 | 13, 240-344, 285 | 226-358 | −5 (−86 to +76) | not detected |
| CDDM, 75,000 iterations (primary) | 338, 349, 328 | 29, 311-354, 333 | 294-371 | +5 (−18 to +28) | not detected |
| CDDM, floor crossing | 457, 449, 413 | 29, 390-468, 423 | 369-489 | +17 (−38 to +71) | not detected |
| Maze, update 2,000 (primary) | 418, 492, 447 | 6, 394-449, 417 | 373-470 | +36 (−44 to +116) | **unresolved** |
| Maze, learned point | (none), 505, 488 | 6, 406-517, 440 | 384-539 | +56 (+11 to +102) | not detected |

**What the 3BFF and CDDM verdicts cover.** By the rules, the bounds accept a shift of the mean number of active
units of up to about 18% on 3BFF and 11% on CDDM. A smaller shift is neither shown nor excluded. The Welch interval
of the mean difference is narrower than the bounds on CDDM: compared with the shared-draw networks, independent
draws change the mean number of active units at 75,000 iterations by between −18 and +28 units (about −5% to +8%
of the reference mean of 333). On 3BFF the interval is ±56 units (±20% of 274), as wide as the bounds. Per the
rules, the paper's counts for 3BFF and CDDM do not depend on the shared draws.

**Maze: unresolved.** The Maze arm trained **shared** draws, because the paper's Maze networks already have
independent draws (rules entry, "Finding made before the rules"). It asks whether shared draws change the count
of a task that the paper trained with independent draws. One of the three networks, seed 2 with 492 active units,
lies above the upper bound of 470. Seeds 1 and 3 (418, 447) lie inside. One network outside is "unresolved" by
the rules. At the learned point, a secondary read point, the two networks that learned lie inside the bounds
(505, 488). Their mean is 56 units above the reference mean of 440, with a Welch interval of +11 to +102 that
excludes zero. That interval rests on two networks, and the rules do not use it for a verdict. A secondary read
point does not overturn the primary. A shift of the Maze count by shared draws, of the order of 10%, is therefore
neither shown nor excluded. No more networks were trained, because the rules did not fix a follow-up for an
unresolved verdict.

### Also reported (rules, "Also reported")

1. **No network has 500 or more of its 1,000 units active at a primary read point.** The largest count is 492
   (Maze seed 2, update 2,000). At the Maze learned point, a secondary read point, seed 2 has 505 active units,
   the only count of 500 or more in this test.
2. **Performance.**
   - 3BFF: held-out R² at 75,000 iterations 0.9514, 0.9517, 0.9513 (mean 0.9515), compared with the paper's
     default networks' mean of 0.9512. Within 0.01.
   - CDDM: 0.8336, 0.8338, 0.8345 (mean 0.8340), compared with the paper's mean of 0.8337. Within 0.01. (The
     run-folder names carry the final R² at 100,000 iterations, 0.868 to 0.885; the read uses 75,000.)
   - Maze: probe success at update 2,000 is 0.85, 0.98, 0.95. Seed 1 is below 0.9 and did not reach the learned
     point within 2,000 updates. In the six reference networks, success at update 2,000 ranges from 0.73 to 1.00
     (one below 0.9). The network outside the bounds (seed 2) has success 0.98, so its count is not confounded
     with performance. Seed 1's count is confounded by the rules' definition but lies inside the bounds.
3. **Restores and skipped updates.** Maze: seed 1 had 1 restore (rollback after a non-finite rollout), seed 2 had
   1, seed 3 had none; no skipped minibatch. In the reference networks, seeds 1-3 had none and seeds 4-6 had 6, 4
   and 1 restores (correction entry `2026-11-25_1307`). CDDM: no skipped updates. 3BFF: no skipped update in the
   first 75,000 iterations. In the continuation from 75,000 to 200,000 iterations, two networks skipped 998 and
   499 updates (seeds 3227418851 and 439145860). Every 3BFF read lies before 75,000 iterations (the floor crossings
   are at 48,800, 58,000 and 64,700), so the skipped updates enter only the fit of the noisy-loss floor. I did not
   look into their cause.

### Checks at the read (all pass)

1. **Independent recount** (`recount_indinit.py`, `recount_maze.py`, which share no code with `common.py` or the
   reader). Exact equality at every read probe for all nine networks: 18 supervised counts and 6 Maze counts. The
   numpy forward pass of equation (h) from the saved weights at 75,000 iterations gives 248 active units for
   3BFF seed 1117702193 and 338 for CDDM seed 3833537590, the same as the stored participation vectors. The two
   participation vectors correlate at 0.99996 and 1.000000 (3BFF uses a fresh batch, CDDM the full condition grid).
2. **Draws.** The W_inp of the first kept snapshot was correlated with the initial W_inp rebuilt in each mode.
   - 3BFF, iteration 1,000: 0.645 to 0.662 with independent, 0.002 to 0.023 with shared.
   - CDDM, iteration 1,000: 0.928 to 0.935 with independent, −0.010 to 0.005 with shared.
   - Maze, update 100: 0.969 to 0.975 with shared, −0.013 to 0.011 with independent.

   Every network trained in its declared mode.
3. **Config diff** of the real configs against a reference network of the same task. Supervised: the tag, paths,
   seeds, `task.seed` (3BFF), checkpoint settings and keys added later at values that reproduce the old behaviour
   (`init_draws`, `sigma_w` 0, `optimizer` adam, `momentum` 0, `iter_offset` 0, the inert `prune_args` keys,
   `transfer_from` null), as the rules listed. Maze, against the suite's continuation configs: the tag, paths,
   `init_draws` (shared against independent), `checkpoint_keep`, `iter_offset` (0 against 2,000, the continuation's
   offset), the entropy and correlation set-point keys at λ = 0, and `freeze_after_end` true against absent. The
   rules entry treats `freeze_after_end`: it changes only the network state after an episode has ended. In
   `TrainerPPO` the losses, advantages and the participation probe use only the steps before the end
   (`probe`: `rates[:, alive]`), so without a penalty the setting can change training only through the rollback
   guard. The restores are listed above.

The Maze arm's code version was not compared bit for bit with the suite's (the code-drift check of the rules
covered CDDM and 3BFF). The Maze comparison therefore differs from the reference in the draw mode and in the
trainer's code version since the suite was trained, and an unresolved verdict cannot separate the two.

### Two script fixes after the read job, neither changing a number

The read job (Della 18376522) wrote the reads, the 3BFF and CDDM verdicts and the Maze recount, then stopped at two
points:
- `independent_init_read.py verdict` refused the Maze reference because their mean is 416.5 and the rules record
  417 (416.5 rounded half up; the reader rounded half to even). The check now accepts a mean within 0.5 of the
  recorded value. The range (394-449) matched, and the margin (21 units, 5% of 416.5 or of 417) and the bounds are
  the same either way.
- `recount_indinit.py` looked for a 3BFF snapshot at iteration 75,000. The 3BFF run of 75,000 iterations keeps
  snapshots up to 74,000 and saves the weights at 75,000 as its final parameters, so the script now takes those.
  It had finished the first network's exact recounts (all equal) before stopping.

Both steps were rerun as Della CPU job 18491136 (1 min) on the existing reads, with the same commands as the chain
launcher's read mode: `independent_init_read.py verdict reads.json maze_reads.json --fig independent_init_active`,
then `recount_indinit.py reads.json`. The training, the reads and the Maze recount were not rerun.

### Compute

71.9 GPU-hours of the 80-hour cap: 47.1 for the nine training runs (3BFF 4.9 h each, CDDM 4.2 to 4.5 h, Maze 6.3
to 6.5 h on A100 80 GB nodes) and 24.8 for the three 3BFF continuations (8.2 to 8.3 h each; the in-job rule allowed
10.9 h each). No network was cut by a time limit, and none failed.

### Where things are

- Networks (Della): `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/{NBitFlipFlop_independent_init,
  NBitFlipFlop_independent_init_c200000,CDDM_independent_init}/EqType=h_N=1000_draws=independent/`,
  `.../trained_RNNs_rl_maze/Maze_independent_init/N=1000_act=relu_draws=shared/s{1,2,3}`; snapshots under
  `.../data/c1_snapshots/` and `.../trained_RNNs_rl_maze_checkpoints/Maze_independent_init/`.
- Reads (Della): `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/independent_init_reads/`
  (`reads.json`, `maze_reads.json`, `verdict.txt`, `recount_supervised.txt`, `recount_maze.txt`, the figure).
- Recount scripts: `~/indinit_out/scripts/` on Della (copies on the Mac in
  `~/firstmate/data/independent-init/scripts/`); they are kept out of the package so that they share no code with it.
- Jobs: training 18375654 (3BFF), 18375655 (CDDM), 18375656 (Maze); continuation 18376521; read 18376522; rerun
  of the verdict and recount 18491136. Training code e4c2ae3 (`~/trainRNNbrain_indinit`), analysis code at this
  entry's commit.
- Figure (internal): `img/internal_figures/independent_init_active.{pdf,png}`: active units of 1,000 along training
  for the nine networks, with the reference range and bounds at the primary read point.

### For the paper (not edited here)

- The Discussion's limitation (`07_discussion.tex:153-157`) can state the test: on 3BFF and CDDM, networks with
  independent draws have the same number of active units at 75,000 iterations as networks with shared draws,
  within the bounds above.
- The sentence that only 27 networks have independent draws (`methods/M01_network.tex:63-67`, and the
  Discussion) is wrong. All 431 saved Maze configs, the paper's Maze networks among them, have independent draws
  (rules entry, "Finding made before the rules"), and so do 104 supervised networks trained after 15 Oct 2026. Which of those 104 enter the paper's figures
  was not established.
