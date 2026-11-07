## 2026-11-06 22:38 — Maze technique verdicts return to the criterion as written (corrects 2026-11-05_0812); the diverged optimizer network reported; provenance table checked byte for byte

Second review of dead-relu-paper PR #193 (comment 6703391936) and decision dead-relu-paper#194.
**This entry corrects `2026-11-05_0812_fig-ed-maze-interventions.md`,** which moved the Maze control
reference to the two learned controls, citing "the pre-registered exclusion of issue #3". Issue #3
covers size statistics and says nothing of the controls of techniques. No network was trained.

### The criterion as written (engellab/trainRNNbrain#5, 5 Nov, 05:48 UTC, before any run)

"Controls: the three N = 1,000 ReLU runs of the uniform series." A technique works when the arm's
count exceeds every control's in all three seeds, at a success within 0.05 of the controls' mean. "An
arm that does not learn (success below 0.9 at the end) is reported as such and counts against the
technique at that value." The interim read-out (15:54 UTC, 15 of 18 runs) used 0.90. Entry
2026-11-05_0812 (16:14 UTC) moved it to 0.984. Noisy synapses, the two gentler rescaling variants and
prune-and-duplicate had not yet been read at that moment.

### What the code does now

`fig_ed_maze_interventions.technique_reference` gives all three controls; `rule_verdict` and
`fig_paper_F5_tasks.rule` have the 0.9 condition; `read_maze_arms` writes `reference_techniques`
beside `reference` and uses it for the rule class. The settings class is unchanged. `fig_paper_F3_tasks`
printout is identical line for line on the new JSON (figure restored, not recommitted).

### Per technique (fig_paper_F5_tasks printout, `learned_control_sensitivity`, and an independent recount from the JSON's runs)

Controls: success 127, 125 and 93 of 128 (0.992, 0.977, 0.727), active 426, 449, 415. All three: 0.898, 430. Two learned: 0.984, 437.5.

| arm | success per seed | all three: difference, fold, verdict | without the 0.9 condition | two learned: difference, verdict |
|---|---|---|---|---|
| prune-and-duplicate | 0.984, 0.953, 0.992 | +0.078, 1.71, works | works | -0.008, works |
| rate 0.03 | 1.000, 0.984, 0.969 | +0.086, 1.96, works | works | 0.000, works |
| rate 0.1 | 0.633, 0.969, 0.992 | -0.034, 2.07, fails | **works** | -0.120, fails |
| rate 0.3 | 0.961, 1.000, 1.000 | +0.089, 2.10, works | works | +0.003, works |
| read-out dropout | 0.836, 0.789, 0.758 | -0.104, 1.02, fails | fails | -0.190, fails |
| rate 0.1 + sparsity | 1.000, 0.430, 0.945 | -0.107, 2.13, fails | fails | -0.193, fails |
| slow tilt (2 of 3 finished) | 0.750, 0.633 | -0.207, 0.74, fails | fails | -0.293, fails |
| 5% grow | 0.656, 0.750, 0.812 | -0.159, 0.56, fails | fails | -0.245, fails |
| noisy synapses | 0.906, 0.961, 0.898 | +0.023, 1.07, fails | **works** | -0.062, fails |
| redraw, zero-out (JSON only) | | +0.086, +0.055, fail on the count | fail | 0.000, -0.031, fail |

The sensitivity claim holds: no verdict differs between the two references. Two verdicts under all
three controls rest on the 0.9 condition alone. Noisy synapses are marginal: seed 3 ends at 115 of 128
episodes, 0.8984, and 116 would give 0.906.

**Numbers that differ from the decision.** The fold of noisy synapses over all three controls is 1.07
(461 / 430). The 1.05 in dead-relu-paper#194 is the fold over the two learned controls. The paper says
1.07.

Other numbers that moved: Maze removal of the k added units, k now from 430 and interpolated on each
ablation's existing k grid (no new ablation): 0.55-0.78 -> 0.53-0.76 (0.530-0.757), random 0.08-0.34
-> 0.08-0.33 (0.075-0.328). Rate penalty 0.03 fold in the text 1.9 -> 2.0.

### The settings rule stays

Learned networks against learned controls, 426-449. With 415 in the range, recounted over all 25
setting arms, one direction changes: weight decay 1e-5 (402, 417, 370) from a decrease to no change.
Weight decay 1e-4 (378, one network) holds with either range, so "only stronger weight decay ... on
all three tasks" stays true through 1e-4. Results 3 and M08 carry the sensitivity. The abstract, the
Introduction, the Fig. 3 title and the Discussion say "stronger weight decay" and were not changed.

### The diverged optimizer network

s2560134940 (SGD lr 3, weight decay 0): last probe entirely non-finite (149 of 2,000 probes are, the
first at iteration 1,600). The other 127 networks all have a finite last probe; that is the stated
rule, since networks kept in the check also have probes with some non-finite unit. Crossing at 30,441,
128 active. Median over the last 2% of probes (40): 268 with its four non-finite probes counted as 0,
as the review's recount does; 275 over the 36 finite ones. Either way above 128, so by the median it
would be a third network that does not fall. This corrects the one-sided "127 of 128" in
`2026-11-06_2219_r17-claims-pr193-review.md`. SGD lr 10: no network is reported (folder empty, entry
2026-10-14_1301).
