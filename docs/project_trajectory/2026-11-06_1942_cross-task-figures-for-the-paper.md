## 2026-11-06 19:42 — Cross-task Figures 3 and 5 as the paper's figures: reader fixes, the 3BFF techniques Extended Data figure, the same names in Figure 5 and the maze figure (dead-relu-paper#183)

**Issue:** dead-relu-paper#183 (WP cross-task, round 15). **Scripts changed:**
`trainRNNbrain/experiments_and_analysis/fig_paper_F5_tasks.py`, `fig_paper_F3_tasks.py`,
`fig_paper_F5_drive.py`. **Figures (re-rendered, `git add -f`):**
`img/internal_figures/fig_paper_F5_tasks.{pdf,svg}` (165.1 x 132 mm, was 136),
`fig_paper_F3_tasks.{pdf,svg}` (165.1 x 133.5 mm, was 132) and the new
`fig_ed_techniques_3bff.{pdf,svg}` (165.1 x 96 mm). **Data:** none recomputed; every number of
the three printouts is the one entry 2026-11-06_0934 reported (the F5 and F3 printouts diff only
in the two rescaling labels; the drive script's (a), (d), (e) lines are unchanged).

### What a reader of the reference would have stumbled on, and the fix

Figure 5 (`fig_paper_F5_tasks.py`):
- Panel d's note "CDDM: not measured" printed over the maze markers at (0.33, 0.78) and (0.26,
  0.74); the text-overlap check sees text only. It now sits in the empty lower right of the maze
  axes (`D_LO + 0.02`, bottom-aligned).
- Panel c's legend ran through the rate-penalty line (its N = 2,000 point, 4.81, sat under
  "read-out dropout"). The fold axis now runs to 8.6 (ticks 1, 3, 5 unchanged), so the legend's
  three rows sit above every line (the rate penalty ends at 5.64).
- The maze rescaling variants were labelled "mild" and "few" under the column, where the maze
  Extended Data figure calls the same arms "slow tilt" and "5% grow"; one name per thing, so Figure
  5 now uses "slow tilt" and "5% grow" (`MAZE_ARMS`, `MAZE_RESCALE_NOTE`).
- Panel d's y label said "recruited units" where the paper, Supplementary Table 4 and the old
  Figure 5e say "added units"; the label and the docstring now say "added units" (the read-out
  files keep "recruited").
- 136 -> 132 mm so that the float with a 300-word caption fits the page (the first build was 114
  pt over at 136 mm with a 400-word caption; both were cut). The lower row moved up 1 mm and
  panel d's x label 1 mm closer to its axes so that nothing leaves the canvas (paperstyle's clip
  warning at 132 mm, 1.5 mm then 0.5 mm, now none).

Figure 3 (`fig_paper_F3_tasks.py`):
- The column titles are staggered on two rows and a reader could not tell "input-weight cap"
  (upper row) from its neighbours' columns. A thin rule (`TITLE_RULE_MM`, 0.9 mm under each title,
  `ps.MUTED`, `THIN_LW`) now spans each title's slots. `slot_positions` returns each column's
  edges for it; the title rows moved up 1 mm (2.0 and 7.5 mm above block a) and the blocks down
  1.5 mm, so the figure is 133.5 mm.
- The value labels stay at the bottom only (a reader of block a follows the column down, as in
  any shared-x figure): labels under every block would add about 20 mm, which the page cannot
  take with the caption.

Extended Data figure of the 3BFF techniques over training (`fig_paper_F5_drive.ed_techniques_3bff`,
written as `fig_ed_techniques_3bff`): the cross-task Figure 5 replaces the 3BFF Figure 5, whose
panels b (R^2 against count at 75k) and c (sizes) it carries, but not a (the count over training
at N = 1,000 under each technique), d (the rate penalty's loss and count over training) and e
(the removal at N = 1,000 and 4,000). Those three are now one Extended Data figure, drawn by the
same panel functions (`panel_traces`, `panel_rate_alone`, `panel_removal`) from the same loaders
in a 96 mm layout with Figure 5's key row, lettered a, b, c; `build()` draws it after the main
figure, so running the script makes all three figures. `fig_paper_F5_drive.pdf` itself is
unchanged.

### Checks

`tests/test_paperstyle.py` on `fig_paper_F5_tasks.pdf`, `fig_paper_F3_tasks.pdf`,
`fig_ed_techniques_3bff.pdf` (and the two maze figures as rendered on main): ok (Helvetica only,
letters 8.5 pt, y labels 1.36 mm from their ticks). `tests/scan_open_markers.py fig_paper_F5_tasks
fig_paper_F3_tasks`: 0 rule failures. `pdfimages -list`: no raster in any of the five. Each
script's own overlap check: no overlapping text (91 and 113 artists). `tests/test_no_function_imports.py`:
212 files; `tests/test_scripts_start.py`: 58 scripts start. The maze interventions figure's
printout flags two noisy-synapse markers in its panel b 1.1 pt apart (seeds 1 and 3, 451 and 454
active at success 0.91 and 0.90); that figure is drawn as on main and the flag is reported in the
paper PR.

### In the paper (dead-relu-paper wp/cross-task)

Figure 3 is `fig_paper_F3_tasks.pdf` and Figure 5 is `fig_paper_F5_tasks.pdf`; the previous Figure
3 (`fig_paper_F3_settings.pdf`) is retired, its panels b, c, e being Extended Data Fig. 4a, i, j, k
over training and its panel d the optimiser column of the new Figure 3; the previous Figure 5's
a, d, e are the new Extended Data figure; `fig_ed_maze_settings.pdf` and
`fig_ed_maze_interventions.pdf` are Extended Data figures. Results 3 and 5, the captions, the
Methods, the abstract, introduction and discussion follow the printouts of 6 Nov (this entry's
figures) and the verdicts of decision dead-relu-paper#185.
