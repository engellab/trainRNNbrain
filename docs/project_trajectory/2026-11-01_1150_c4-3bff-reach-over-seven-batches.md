## 2026-11-01 11:50 — 3BFF gradient reach over seven trial batches; Results 4 wording (dead-relu-paper #112, option b)

Package results-r4 (round 4). Follows the committee's finding on dead-relu-paper#112 that the 3BFF
reach numbers of Results 4 (entry `2026-11-01_1021`) come from one seeded batch of 1,024 trials
(task seed 12345) and move with the batch.

**What was run.** The committee's independent code (own forward model from Methods M01, own 3BFF
batches from Methods M02, PyTorch autograd for the backward pass, no trainRNNbrain import), copied
unchanged to `trainRNNbrain/experiments_and_analysis/c4_reach_batches_indep.py` (md5
849884e4215a48222f114577018f9ce3). Weights: `data/pop_stats_nets/NBitFlipFlop_*` (the Fig. 1d
weights) and `data/fig4_noreturn/snap/<seed>_d5000.npz` (5,500 iterations later). The committee ran
batch seeds 7, 11, 12 and 13 (logs on #112); I ran seeds 21 and 22 on the Mac (CPU, about 1 min
each): `python3 c4_reach_batches_indep.py 3BFF 21` and `... 3BFF 22`.

**Rule, fixed before reading the seed 21 and 22 logs.** The wording "about a sixth to a third of the
low-active units are reached" fails if any batch gives a reached share below 15% or above 34% in any
network; "at most 6% of these are active later" fails if any batch gives more than 6.0%.

**Result (per network, per batch).** Reached = reached low-active units; later = share of them active
5,500 iterations later.

| Batch | s1258717025 reached / low, later | s2461724787 | s4113307577 |
|---|---|---|---|
| paper, seed 12345 (entry 1021) | 136 / 731, 0.7% | 155 / 658, 2.6% | 161 / 713, 4.3% |
| seed 7 | 138 / 734, 2.2% | 110 / 658, 3.6% | 161 / 713, 5.6% |
| seed 11 | 137 / 733, 2.9% | 150 / 658, 3.3% | 167 / 714, 6.0% |
| seed 12 | 222 / 731, 0.9% | 118 / 658, 2.5% | 167 / 715, 5.4% |
| seed 13 | 138 / 734, 2.9% | 108 / 660, 3.7% | 157 / 714, 5.1% |
| seed 21 | 146 / 732, 0.7% | 133 / 659, 3.0% | 165 / 714, 5.5% |
| seed 22 | 144 / 733, 2.1% | 131 / 658, 2.3% | 160 / 713, 5.0% |

Reached share 16% (108 / 660) to 30% (222 / 731); later share 0.7% to 6.0%. Both wordings hold in
all seven batches. Every batch stays far below the C4 rule's 10% bound (`2026-11-01_0948`), so the
C4 verdict does not change. Seed 21 and 22 active counts (267-268, 341-342, 286-287) match the
paper's within 2 units.

**What changes in the paper.** Results 4 says "In each of seven batches of trials, about a sixth to a
third of the low-active units are reached and at most 6% of these are active later"; the closing
sentence no longer repeats the share. Fig. 4c still draws the paper's batch. Methods (Gradient
reach) should state the seven batches of 1,024 trials (methods-r4, noted on #112). Board row C4
should read "a sixth to a third reached, at most 6% active 5,500 iterations later, over seven
batches".
