## 2026-11-06 20:09 — The direction of a setting comes from the networks that learned the task: the maze SGD arms have no direction; Figures 3 and 5 print "1" beside a one-network marker; the maze techniques figure names the rescaling variants as Figure 5 does (review of dead-relu-paper PR #188)

**Issue:** dead-relu-paper#183 (PR #188, review items 1, 8, 9a). **Scripts changed:**
`trainRNNbrain/experiments_and_analysis/read_maze_arms.py`, `fig_paper_F3_tasks.py`,
`fig_paper_F5_tasks.py`, `fig_ed_maze_interventions.py`. **Data:** `data/read_maze_arms.json`
rebuilt from the same run folders (no new run; the technique verdicts and every count are
unchanged; only the settings' verdict strings changed). **Figures (re-rendered, `git add -f`):**
`img/internal_figures/fig_paper_F3_tasks.{pdf,svg}`, `fig_paper_F5_tasks.{pdf,svg}`,
`fig_ed_maze_interventions.{pdf,svg}`.

### The rule's scope (review item 1; the coordinator's decision)

The pre-registered direction rule (every seed outside the control range on one side) counted the
maze SGD arms, none of whose networks learned the task (success 0.00 to 0.24), as "increases the
count", so that four values moved the count the same way on all three tasks (weight decay 1e-5
and 1e-4, SGD 0.03 and 0.1) while the paper's headline said weight decay alone. The decision: a
direction comes only from the networks that reached the task's criterion, as the maze reference
already came from the learned controls. `read_maze_arms.setting_verdict` and
`fig_paper_F3_tasks.direction` now take the learned seeds only (the maze: a learned point,
`common.maze_learned_index`; 3BFF and CDDM: held-out R^2 at or above the read-out file's bar,
0.8); an arm with no learned seed reads "no direction" (the JSON) / "none learned" (the table). The
two scripts' verdicts agree in every finished cell, as before.

What changed on the maze: SGD 0.03, 0.1, 0.3, 1, activity noise 0.1, the metabolic penalty 10,
the shifted sigmoid and tanh have no direction (no network learned); weight decay 1e-4 decreases
the count on the one network that learned (378), the sparsity penalty on one (399), the metabolic
penalty 0.01 on two (413, 379; it read "no change" with the not-learned 537) and 1 on two (390,
359); input norm 2 (443, 357), spectral radius 1.5 (350, 471), the metabolic penalty 0.1 (431,
387), the recurrent-weight penalty 0.2 (517, 405), AdamW (452, 429) and softplus (437, 420) read
"no change" on their two learned networks. On CDDM the metabolic penalty 10 (R^2 0.787 to 0.789,
below the bar) has no direction; on 3BFF SGD at rate 1 reads "no change" on its two learned
networks (265, 220). Every other cell is as in entry 2026-11-06_0934.

**The tallies the paper quotes**, printed by `fig_paper_F3_tasks.py` (new lines at the end of
its output): 27 setting values have a direction on both supervised tasks, 11 of them the same
(weight decay 1e-5 and 1e-4, cap 0.3, spectral radius 2.0, metabolic 0.1 and 1, recurrent-weight
penalty 0.05 and 0.2, AdamW, SGD 0.03 and 0.1); 18 have a direction on all three tasks, 2 of them
the same (weight decay 1e-5 and 1e-4, a decrease). Before the scoping: 28 and 11; 25 and 4.

### The figures (review items 8, 9a)

- Figures 3 and 5 print the number of networks beside a marker whenever it is not 3, "1"
  included (`n_group >= 1`; it was `> 1`): the maze weight-decay 1e-4 and sparsity slots of
  Figure 3 and the maze rate + sparsity slot of Figure 5 now read "1".
- The maze techniques figure's key says "rescaling, slow tilt" and "rescaling, 5% grow" (was "5%
  growing") in Figure 5's order (slow tilt before 5% grow; panel c's columns follow).

### Checks

`tests/test_paperstyle.py` on the three PDFs: ok. `tests/scan_open_markers.py fig_paper_F5_tasks
fig_paper_F3_tasks`: 0 rule failures. `pdfimages -list`: no raster. The scripts' overlap checks:
no overlapping text (117, 93 and 64 artists); the maze techniques figure's two noisy-synapse
markers 1.1 pt apart in b remain (reported in the PR). `tests/test_no_function_imports.py`: 212
files. The F5 printout is unchanged apart from the label count.
