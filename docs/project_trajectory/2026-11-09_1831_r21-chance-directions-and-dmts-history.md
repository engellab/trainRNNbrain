## 2026-11-09 18:31 — Chance level of the setting rule over every value run; the DMTS history for Supplementary Table 7

Work package r21-text (ptolmachev/dead-relu-paper#218; committee #217, Reviewer R items R1 and R3).

### Expected chance directions (R3)

Script `trainRNNbrain/experiments_and_analysis/chance_directions.py` (branch `wp/r21-text`). Under
exchangeability, all n networks of a value lie above all m control networks with probability n! m!/(n + m)!,
and the value gets an increase or a decrease with twice that. n per value and task is the number of networks
that learned the task in the paper's Supplementary Table 2 (`floats/ST_verdicts.tex`, paper main ad52b64; the
script types them in), m = 3 (3BFF, CDDM) and 5 (Maze, the five default networks learned by update 2,000).
The CDDM five-network sweeps (activity noise, sparsity penalty) are scored with three controls, the larger
probability. Output:

```
3BFF: 32 values run, expected chance increases or decreases 3.30
CDDM: 29 values run, expected chance increases or decreases 2.44
Maze: 29 values run, expected chance increases or decreases 1.89
28 values run on all three tasks: expected chance same-direction conjunctions 0.0038 (1 in 260)
weight decay 1e-4: same direction on all three tasks by chance, 1 in 1200
```

Observed increases or decreases in Supplementary Table 2 (arrows counted by a parser over the table rows):
3BFF 19 of 32, CDDM 25 of 29, Maze 4 of 29. So on 3BFF and CDDM the observed number is six to ten times the
chance expectation; on Maze it is about twice (4 against about 2). The weight-decay 1e-4 conjunction, with its
Maze leg of one network against five controls, is about 1 in 1,200 (Reviewer R's figure; the paper's earlier
"1 in 4,000" assumed three networks on Maze). Over the 28 values run on all three tasks, the expected number of
values with the same direction on all three by chance is about 0.004, so the one observed (weight decay 1e-4)
is not explained by the number of values tested.

### DMTS history (R1), for the Supplementary Table 7 row

- Until 4 Nov DMTS was the main text's third task (status board, 12 Oct: "DMTS (short delay) appears only as a
  third series on the scaling curve"; entry `2026-11-01_2124`).
- 2 Nov 18:22: the DMTS setting sweeps fail their decrease rule, set before training (entry
  `2026-11-02_1822`): 8 of 9 networks that learned and have a floor crossing decrease, 1 (σ_rec = 0.1) rises
  288 → 308; three settings untested.
- 2 Nov (PAVEL_FEEDBACK item 56): replace "wayward DMTS" with another task if the paper benefits.
- 4 Nov: Maze replaces DMTS in Figs. 1 and 2 (entries `2026-11-04_1957`, `2026-11-05_0435`).
- What DMTS shows: size exponent 0.55 at the 1.05 margin (entry `2026-10-31_2211`; 0.54-0.60 over the
  factors 0.025-0.10); fewer than half the units active for N ≥ 1,000 at the floor crossing (Supplementary
  Table 1: 422 and 218 of 1,000; 426, 386 and 465 of 2,000; 620 of 4,000).
- The 15 DMTS technique networks of the size series (`DMTS_paper_grid_d5`): only the three prune-and-duplicate
  networks of N = 500 learned the task (entry `2026-11-09_1825`, last section).
