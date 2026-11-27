## 2026-11-27 08:40 — Optimizer swap: no candidate meets the M09 criterion on all three tasks

Rules: `2026-11-25_1318_rules-optimizer-remedy.md`, committed at cdb57fa before any run. Submission:
`2026-11-25_1324_optimizer-remedy-submitted.md`. Branch `wp/optimizer-remedy`.

**Verdict.** Changing the optimizer is not a remedy among the five candidates tried. The rules call a candidate a remedy if it
passes the M09 criterion on 3BFF (the 3-bit flip-flop task), CDDM (context-dependent decision making) and Maze at update 2,000.
No candidate passes 3BFF, so none can be a remedy, whatever its result on the other two tasks. All three candidates run on CDDM
pass there, and one of the two run on Maze passes there; a pass on two tasks is not called a remedy (rules).

| Candidate (networks of N = 1,000 units) | 3BFF | CDDM | Maze, update 2,000 | Maze, update 4,000 |
|---|---|---|---|---|
| Adam, no decay (identical to AdamW at decay 0 and at 1e-6) | fails (count) | passes | passes, by 2 units | fails (count) |
| AdamW, decay 0.01 | fails (count; one network diverged) | passes | fails (count) | fails (count) |
| AMSGrad (Adam, decay 1e-6 in the gradient) | fails (count and R²) | passes | not run | not run |
| SGD, momentum 0.9, rate 0.03 | fails (R²) | not run | not run | not run |
| SGD, momentum 0.9, rate 0.1 | fails (count and R²) | not run | not run | not run |

"Count" means at least one candidate network has no more active units than the control with the most. "R²" means the
candidate's mean held-out R² is more than 0.01 below the controls' mean. The cells marked "not run" follow the rules' order:
CDDM for the SGD candidates and Maze for any other candidate run only after a pass on 3BFF (and CDDM). No such run could change
the verdict, so the verdict is not undetermined.

*Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05 × the 95th percentile of p.
*M09 criterion* (dead-relu-paper `methods/M09_remedies.tex`): every candidate network has more active units than every control,
the candidates' mean performance is at least the controls' mean minus 0.01 in held-out R² (3BFF, CDDM) or 0.05 in probe success
(Maze), and on Maze no network ends below 0.9.

### AdamW at decay 1e-6 is Adam without decay

AdamW multiplies each weight by 1 − lr × decay at every step. At N = 1,000 the learning rate is 1e-3 × (100/1,000)^(1/3) =
4.64e-4, so the multiplier at decay 1e-6 is 1 − 4.6e-10. In float32 that number rounds to exactly 1.0, so AdamW at 1e-6 never
changes a weight through its decay term. On CPU (`tests/test_optimizer_switch.py`) and on an A100 80 GB card (Della job 18377213),
AdamW at 0 and at 1e-6 give final weights equal to Adam without decay, maximum difference 0.0; AdamW at 0.01 and Adam with
coupled decay 1e-6 differ from it. So the brief's four Adam and AdamW arms are two distinct candidates: no decay, and AdamW at
0.01. That left room for AMSGrad and SGD with momentum at two rates (five candidates, under the brief's limit of six).

### The numbers

Controls: on 3BFF and CDDM, the default networks with independent initial draws (`*_independent_init`, the same three seeds,
Della 18375654 and 18375655); on Maze, the matched default networks of `2026-11-26_0705_maze-matched-controls-result.md`
(Della 18374320), which I continued from update 2,000 to 4,000 with the optimizer state kept (18377212). 3BFF and CDDM are read at
75,000 iterations, Maze at updates 2,000 and 4,000.

**3BFF.** Controls 294, 248 and 262 active units (largest 294), held-out R² 0.951-0.952, mean 0.9515.

| Candidate | Active units | Held-out R² | Mean R² minus control mean |
|---|---|---|---|
| Adam, no decay | 290, 270, 334 | 0.957-0.958 | +0.006 |
| AdamW, decay 0.01 | 330, 236, 293 | −10,527, 0.954, 0.955 | (diverged network) |
| AMSGrad | 289, 322, 310 | 0.929-0.932 | −0.021 |
| SGD, momentum, rate 0.03 | 339, 312, 412 | 0.909-0.916 | −0.038 |
| SGD, momentum, rate 0.1 | 176, 162, 175 | 0.906-0.928 | −0.031 |

Only SGD with momentum at rate 0.03 has more active units in every network than in every control, and its R² is 0.038 below the
controls'. Adam without decay fails by two networks (290 and 270 against 294). The AdamW network of seed 254859059 has
held-out R² of about −10,500 from its weights at 75,000 iterations (the training score in its folder name is −10,358); its other
two networks fail the count anyway (236 and 293).

**CDDM.** Controls 338, 349 and 328 active units (largest 349), held-out R² 0.834-0.835, mean 0.8340.

| Candidate | Active units | Held-out R² | Mean R² minus control mean | Mean active units / control mean |
|---|---|---|---|---|
| Adam, no decay | 415, 468, 476 | 0.833-0.835 | +0.0001 | 1.34 |
| AdamW, decay 0.01 | 435, 478, 492 | 0.832-0.834 | −0.0008 | 1.38 |
| AMSGrad | 657, 718, 721 | 0.832-0.833 | −0.0011 | 2.07 |

**Maze.** Probe success is the fraction of 128 noise-free episodes solved at the probe nearest the read point.

| Group | Update 2,000: active units | success | Update 4,000: active units | success |
|---|---|---|---|---|
| Matched controls | 401, 432, 392 | 0.91, 0.99, 0.91 | 346, 353, 378 | 1.00, 1.00, 1.00 |
| Adam, no decay | 434, 452, 449 | 0.91, 1.00, 0.99 | 376, 353, 425 | 1.00, 1.00, 0.98 |
| AdamW, decay 0.01 | 468, 476, 398 | 0.96, 0.98, 1.00 | 396, 386, 331 | 1.00, 1.00, 0.98 |

Adam without decay passes at update 2,000 by 2 units: its smallest network has 434 active units and the largest control 432.
Seed 1 of that candidate had not yet learned at update 2,000 (it learned at update 2,200). By update 4,000 every network has lost
units, and the candidate fails: seed 2 has 353, the same as the control of seed 2.

**Against M09's own controls**, which have shared initial draws (3BFF 236-288, CDDM 333-342, Maze 415-449): AMSGrad and SGD at
rate 0.03 are above 288 on 3BFF in every network; every CDDM candidate is above 342; no Maze candidate is above 449 in every
network. This comparison is secondary (rules) and changes no verdict.

**The size test's numbers, as a description only** (`ST_verdicts`): every network trained without a non-finite loss and skipped
fewer than 5% of its updates (the most is 498 of 75,000, in the AMSGrad network of seed 254798194). On CDDM, AMSGrad has twice
the controls' mean number of active units (fold 2.07, at least 1.5) with R² within 0.02; Adam without decay and AdamW at 0.01 have
1.34 and 1.38 times. On 3BFF and Maze no candidate reaches 1.5 times.

**Choosing the best of five.** Had a candidate passed all three tasks, the chance of that by luck (arm and control counts
exchangeable) would have been at most 5 × (1/20)³ = 0.0006. No candidate did, so nothing needs correcting. The single-task passes
are weak evidence by themselves: with five candidates, a pass on one task happens by chance with probability up to 0.25.

### Late decline (all networks, description only)

The median number of active units over the 2,000 iterations before 50,000 and before 75,000:
- *CDDM controls* fall in all three networks: 378 → 344, 376 → 350, 362 → 330.
- *CDDM, Adam without decay*: 438 → 419, 482 → 468, 489 → 483. *AdamW at 0.01*: 456 → 440, 482 → 476, 500 → 490. Both fall less.
- *CDDM, AMSGrad*: 642 → 657, 707 → 723, 723 → 716; two of the three rise.
- *3BFF*: controls 321 → 309, 294 → 247, 263 → 272. No candidate shows a consistent difference from the controls; AMSGrad is flat
  (282 → 288, 322 → 322, 313 → 312).
- *Maze*: every network of every group has fewer active units at update 4,000 than at 2,000 (controls 392-432 → 346-378,
  Adam without decay 434-452 → 353-425, AdamW 398-476 → 331-396).

The rules' removal test, near-copy analysis and longer runs apply only to a candidate that passes both 3BFF and CDDM. None did,
so they were not run.

### Against the expectation

The rules expected Adam without decay to pass 3BFF and CDDM and to fail Maze. It passed CDDM, failed 3BFF and passed Maze at
update 2,000 by 2 units. The 3BFF result differs from `ST_verdicts` (decay-free 252-367 against controls 233-270) because these
controls have independent initial draws and one of them has 294 active units; the decay-free networks themselves (270-334) are in
the earlier range.

### Checks (rules section "Checks")

1. **Code.** Passed on the smoke runs before the rules were committed (maximum difference 0.0).
2. **`max_iter`.** Passed on the smoke runs.
3. **Configs.** Every candidate's saved config differs from its same-seed control's only in the candidate's optimizer keys, the
   experiment tag and the paths (Maze continuations also in `init_from`, `init_adam` and the update offset). No unexpected key in
   any of the 30 candidate networks.
4. **Same start.** On 3BFF and CDDM the first stored count equals the same-seed control's in every network. On Maze, success and
   steps of the update-1 rollout (before any optimizer step) equal the control's; the return differs in the fifth decimal
   (−2.32075 against −2.32073 in seed 1) because the controls ran on 40 GB cards (i14 nodes) and the candidates on 80 GB cards. In
   every seed both Maze candidates equal, bit for bit, the stored same-seed `ent0.01` network of N = 1,000, which holds the
   other card model's value, as the rules allow.
5. **Independent recount.** `firstmate/data/optimizer-remedy/scripts/oracle_optrem.py` (numpy only; shares no code with
   `common.py`, the reader or the rule function) recounted 48 networks at the read points from the stored participation vectors:
   0 mismatches, and the same verdict in every cell (Della 18491819).
6. **Read gate.** 3 of 3 controls and 3 of 3 networks in every run cell, every network at its last iteration or update.

### Runs, compute and corrections

Della, gpu-short only: 3BFF 18377208 (15 networks), CDDM 18377209 (9), Maze 18377210 and 18377211 (two stages, 6 networks),
Maze control continuation 18377212 (3), GPU equality check 18377213. All 40 tasks COMPLETED with exit code 0, none ended by its
wall clock. **201 GPU-hours** of the 250 allowed (sacct, elapsed × 1 GPU). Reads on CPU nodes: 18491128, 18491415, 18491815.

- The reader's first run (18491128) stopped on a missing folder: the analysis clone on Della had no links to the scratch
  `trained_RNNs` and `c1_snapshots` trees. Its Maze read had finished and is the one used.
- The second read (18491415) gave the same skipped-update count to every candidate of a seed. `entropy_setpoint_read.skipped`
  matches a Slurm log on the seed alone, and the five candidates share their seeds, so all five got the count from one log.
  Fixed in 2c515de (`optimizer_remedy_read.skipped_of` matches the experiment tag and the seed); the third read (18491815) is the
  one reported. The count enters only the size test's stability description, and no count was near its 5% limit before or after.

Outputs: `firstmate/data/optimizer-remedy/reads/` (reads.json, maze_reads.json, verdict.txt, the recount log).
Figure: `img/internal_figures/optimizer_remedy_active.{pdf,png}`: active units against iterations (3BFF, CDDM) and PPO updates
(Maze) for every network of every candidate and control.

`docs/paper_audit.md` is not changed: the board records no claim that an optimizer swap is or is not a remedy, and the optimizer
comparison it does record (B5, the late loss of units) is not affected.
