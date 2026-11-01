## 2026-11-01 12:00 — 3BFF gradient reach across six trial batches: result (dead-relu-paper #112)

Follows `2026-11-01_1143_c4-reach-across-3bff-batches-rule-set-before-the-run.md`, which fixed the
measurement and the rule. Run on the Mac, CPU only, trainRNNbrain `wp/methods-r4`:
`python trainRNNbrain/experiments_and_analysis/c4_reach_batches.py` (task seeds 12345, 101-105; about
45 s per network and batch). Output `data/fig4_noreturn/c4_reach_batches.json` (gitignored data).

**Rule clause 1, reproduction: pass.** Task seed 12345 gives exactly the numbers of entry
2026-11-01_1021: low-active 731, 658, 713; reached 136, 155, 161; share active at +5,500 0.7%,
2.6%, 4.3%.

**Per network and batch** (1,024 trials each; no 3BFF unit is silent in any batch):

| Network | Task seed | Low-active | Reached | Reached share | Reached active at +500 | at +5,500 |
|---|---|---|---|---|---|---|
| s1258717025 | 12345 | 731 | 136 | 18.6% | 0 | 1 (0.7%) |
| | 101 | 732 | 132 | 18.0% | 0 | 3 (2.3%) |
| | 102 | 735 | 142 | 19.3% | 3 | 5 (3.5%) |
| | 103 | 732 | 140 | 19.1% | 2 | 3 (2.1%) |
| | 104 | 732 | 129 | 17.6% | 3 | 3 (2.3%) |
| | 105 | 734 | 139 | 18.9% | 2 | 5 (3.6%) |
| s2461724787 | 12345 | 658 | 155 | 23.6% | 2 | 4 (2.6%) |
| | 101 | 659 | 134 | 20.3% | 3 | 6 (4.5%) |
| | 102 | 658 | 155 | 23.6% | 1 | 4 (2.6%) |
| | 103 | 658 | 149 | 22.6% | 1 | 4 (2.7%) |
| | 104 | 660 | 130 | 19.7% | 4 | 6 (4.6%) |
| | 105 | 659 | 131 | 19.9% | 2 | 5 (3.8%) |
| s4113307577 | 12345 | 713 | 161 | 22.6% | 2 | 7 (4.3%) |
| | 101 | 713 | 161 | 22.6% | 2 | 9 (5.6%) |
| | 102 | 714 | 160 | 22.4% | 3 | 10 (6.2%) |
| | 103 | 713 | 168 | 23.6% | 2 | 7 (4.2%) |
| | 104 | 714 | 162 | 22.7% | 2 | 7 (4.3%) |
| | 105 | 713 | 159 | 22.3% | 3 | 10 (6.3%) |

**Rule clause 3: the bound "fewer than a tenth" holds** in every network and batch (largest 6.3%,
10 of 159 units).

**Rule clause 4, sanity: pass.** Reached shares 17.6-23.6% and shares at +5,500 0.7-6.3% lie inside
or next to the committee's 16-30% and 0.9-6.0%.

**What this changes.** results-r4 (entry 2026-11-01_1150 on `wp/results-r4`) wrote Results 4 as
"at most 6%" from seven batches (the paper's and six from the committee's independent code, maximum
6.0%). Two of these five production-code batches exceed it in s4113307577 (6.2% and 6.3%), so "at most
6%" fails on twelve batches. Over all twelve batches the reached share is 16-30% of the low-active
units and the share active 5,500 iterations later is 0.7-6.3%. Posted on #112 for results-r4. Methods
M07 (paper `wp/methods-r4`) names the twelve batches.
