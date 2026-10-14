# A5: the task-demand supplementary panel (agent-a5)

## Pre-registration, written 2026-10-13 23:29, before any floor read-out was computed

**Question.** A reviewer's first objection: "the task just doesn't need more units." Does a harder
task fill the network, or does the network still leave most units silent?

**What I already knew before writing this** (trajectory, 2026-09-12 to 09-16, all at the END of
150k iterations, scale-free criterion): hyper flip-flop `none` reaches 977 of 1000 active units at
k=8 and 1616 of 2000; plain flip-flop stays near 240-620. So the end-of-training data already leans
against the paper for the combinatorial task. The floor read-out is new; silencing continues after
the floor in this project, so I expect floor counts to be HIGHER than the 150k counts, which would
push further against the paper.

**Read-out.** Each run where its noisy training loss first reaches 1.05x its own fitted floor
(`common.noisy_floor` on `common.noisy_loss_series`, warm-start chains joined); active count from
the stored noise-free participation trace at the probe nearest the crossing, criterion
p >= 0.05 q95(p). End of training as a labelled second read-out. Health check q50/q95 < 0.05 at
every read-out (above it, the relative rule is broken).

**Tests and what would show the paper wrong.**
1. *Fill test (flip-flop family, unpenalised, fixed N).* The paper loses this point if, at N = 2000,
   any cell whose task is solved (r^2 >= 0.90 at the read-out, i.e. within 0.05 of the plain task's
   ~0.95; threshold set now, knowing the 150k r^2 values) has active >= 0.5 N. It also loses it in
   a weaker form if the most demanding cell at any N has active >= 0.9 N at all, solved or not:
   then "silence" is only what an under-demanding task leaves, and the panel must say so.
2. *Size test at fixed demand.* For each demand level with all three sizes read, fit
   active = A N^b. If b >= 0.9 at the highest demand, a hard task recruits a bigger network in
   proportion and the silent share does not grow with N there. The paper's size claim survives only
   if b stays well below 1 (< 0.7) at every demand level.
3. *Number of tasks (Yang 20-task vs the CDDM pair, N = 1000, whole training batch).* "More tasks
   recruit more units" in a substantive sense := multi20 active >= 1.5x ctxdm active at the floor in
   every seed. "Fills the network" := multi20 active >= 0.9 N.

**Excluded by design.** Penalised (`both`) arms: the question is about plain task training.

## Inventory (local mirror; Della's copies of the hyper-long and Walsh sweeps were deleted in the 2026-10-13 cleanup, Spock holds the hyper grid and Yang 30k)

Every network below is **pre-standard**: ReLU, h equation, self-connections on, no Dale, gamma 0,
but the bias is TRAINED in [-1, 1] (the saved weights reach the bound). None is a standard network.
All have per-iteration `TrainLosses.json` and a noise-free participation trace every 100 iterations.

| sweep | task | demand (target functions / rules) | N | seeds | budget |
|---|---|---|---|---|---|
| `NBitFlipFlop_std_ksweep` | plain flip-flop, k = 1..8 | k | 500, 1000, 2000 | 3-5 | 150k (k=1 N=500), 400k (N=2000), 500k |
| `NBitFlipFlopHyper_std_hyper` | parity flip-flop, k = 2, 4, 6, 8 | 2^k - 1 = 3, 15, 63, 255 | 500, 1000, 2000 | 3 | 150k (`both` arm also on disk, not used) |
| `NBitFlipFlopHyper_std_hyper_long` | same, five cells | 3, 15, 63 | 500, 1000 | 3 | 500k |
| `NBitFlipFlopWalsh_std_walsh` | one output, all products up to degree | 63 (k=6), 255 (k=8) | 1000 | 1 | 30k + 70k warm start (k=6), 150k (k=8) |
| `Yang_std_30k` | CDDM pair vs all 20 Yang rules, 33 outputs both | 2, 20 rules | 1000 | 3 | 30k (one seed per set warm-started from the 10k pilot) |

`Yang_std_multi` and `MultiRule_std_multi` are empty folders on both the Mac and Spock.

## Results (2026-10-13 23:35)

Script `trainRNNbrain/experiments_and_analysis/fig_supp_task_demand.py`, cache
`data/task_demand_cache.json`, figure `img/internal_figures/fig_supp_task_demand.{pdf,svg,png}`.

Active units, mean over seeds, at the floor read-out (end of training in brackets):

| N | plain k=1..8 | parity 3 | parity 15 | parity 63 | parity 255 | Walsh 255 (1 seed) |
|---|---|---|---|---|---|---|
| 500 | 247-280 (129-177) | 284 (216) | 363 (342) | 471 (464) | 496, 1 seed (494) | — |
| 1000 | 350-444 (161-226) | 488 (287) | 562 (491) | 732 (715) | 968 (977) | 924 (916) |
| 2000 | 446-505 (229-330) | 680 (389) | 823 (672) | 1174 (1016) | 1610, 1 seed (1616) | — |
| r^2 at the floor (estimated) | 0.95-0.96 | 0.94 | 0.88 | 0.79-0.80 | 0.63-0.70 | 0.86 |

Yang, N = 1000, whole training batch: CDDM pair 378 at the floor (370 ± 8 at the end), 20 rules
440 (428 ± 18). Only one seed per set reaches its floor by the rule; the other two end 1.6-2.0x
above their fitted floor at 30k.

Per-unit check against an independent oracle (my own Euler forward pass on the final weights,
fresh trials, participation recomputed): hyper k=8 N=1000 975 active vs trace 978, plain k=3
N=1000 196 vs 195; the two agree on 99.7% and 99.9% of units. Rank correlation over the trace's
active units 0.97 and 0.999. Rank correlation over ALL units was 0.80 for the plain network, below
the 0.95 I set; the disagreement sits among the silent units, whose participation is near 2.5e-4 and
is reordered by the trial draw.

The relative rule's health ratio q50/q95 exceeds 0.05 in the parity cells at k >= 4 and in the
N = 500 plain cells. There more than half the units are active, and the flip-flop's absolute
threshold (p >= 4e-2) gives counts within 8% of the relative rule in every flip-flop cell, so the
high ratio reflects real activity, not a broken rule. On the Yang networks the absolute threshold
reads 454 and 628 (it was calibrated on the flip-flop and does not transfer).

**Against the pre-registered tests.**
1. Fill test, solved cells at N = 2000 (r^2 >= 0.90): plain flip-flop at most 505 of 2000 active,
   parity k=2 680 of 2000. The paper survives this test. Weaker form: the most demanding cell fills
   the network at N <= 1000: parity k=8 968 of 1000 and 496 of 500, Walsh k=8 924 of 1000. **This
   form of the test goes against the paper**, but only where the network fails the task: those
   cells score r^2 0.63-0.86 against the plain task's 0.95.
2. Size exponent at fixed demand: plain flip-flop b = 0.35-0.51 at every k; parity 0.59-0.66 for
   3-63 targets and **0.85 at 255 targets**, above the 0.7 bar I set. At 255 targets the networks
   with N = 500 and 1000 are full, so the exponent there measures saturation, not proportional
   recruitment, but the pre-registered bar is failed.
3. Number of rules: 20 rules give 440 active units against 378 for the CDDM pair, 1.16x, below the
   1.5x bar; 44% of the network. More rules do not fill the network.

**Answer to the objection.** Demand moves the count: at N = 2000 the parity flip-flop goes from 680
active units (3 targets) to 1610 (255 targets), while the plain flip-flop stays at 450-505 across
k = 1..8. So "the task doesn't need more units" is half right: a combinatorial task recruits more.
But every network that solves its task to r^2 >= 0.88 leaves most of a 2000-unit network silent
(at most 823 active at the floor, fewer by the end), and a network fills only when the task exceeds
what it can learn at that size. Growing N at fixed demand adds active units at roughly N^0.4-0.65,
so the silent share still grows with size unless the network is saturated.

**Caveats.** All networks pre-standard (trained bias). Parity k=8 at N = 500 and 2000: 2 of 3 seeds
never reach the floor bar (still descending at 150k); the floor value there is one seed. Walsh: one
seed per cell; the k=6 cell does not reach the floor (end only). Yang: one seed per set at the
floor. R^2 at the floor is estimated from the loss ratio, not measured (no weights saved there). The
parity task's r^2 pools channels of unequal variance (trajectory 2026-09-11), and its noise-free
r^2 is higher (0.79 vs the noisy 0.68 for one k=8 N=1000 network).
