## 2026-11-02 22:58 — r12-figs: review of dead-relu-paper PR #177 applied (Fig. 2f bottom row and grey non-flat networks, error-bar caps, Fig. 5 rings); correction to 2026-11-02_2226

Branch `wp/r12-figs` (trainRNNbrain main 35a64c6 merged in as dd0512f). Review:
dead-relu-paper PR #177, comment 6597296648 (CHANGES REQUESTED, rows 1-4 required, 5-8 optional).
No number of the paper changes.

**Correction to `2026-11-02_2226_r12-figs-marker-edges-and-fig2f.md`.** That entry says a drawing
of the raw active count would contradict the test for 2 of 19 networks, seeds 822903475 (count
ending 3 units above its start) and 3866906883. Read as 5,000-iteration block medians (last block
minus the first, the blocks running to the end of training), only seed 822903475 contradicts the
test: its count ends 2 units above its start, while its least-squares decrease is 7.6 units per
100,000 iterations. Seed 3866906883 ends at -1 against a dashed line at -2.4 and agrees with the
test. The 2226 entry's block medians stopped at iteration 195,000, one block short. Recomputed with
`fig_paper_F2_size.flat_quantities` on `data/fig_paper_F2_size_flat.pkl`.

**Row 1, Fig. 2f bottom row.** The thin lines were block medians minus each network's least-squares
fit at 150,000, so 8 of 19 started more than 5 units from zero. They are now the block medians
minus the network's own first-block median: every thin line starts at zero, as "change in the number
of active units since iteration 150,000" says. The thick least-squares lines are drawn from zero at
150,000, as before; their slopes, and so the test, are unchanged.

**Row 2, the 4 networks whose loss is not flat** (seeds 2545439171, 3374117298, 2422919889,
620846234) are drawn in grey (`paperstyle.FAINT`) in both rows, under the coloured lines, with a key
"loss not flat". The coloured thick lines are the test's 15 networks; 13 of them end below the
dashed line. The caption states the counts.

**Row 5 (optional), thresholds at print size.** The dashed line is 0.9 pt (was 0.5), with shorter
gaps, and is drawn above every data line. The bottom row's y-range is -76 to +14 units, set from
the data (block medians -72 to +10; the old -70 cut one N = 4,000 line). The counts in the caption
carry the verdicts that a line width cannot resolve (seeds 822903475 and 822939552 end within about
1 pt of the dashed line).

**Row 3, error-bar caps.** `setup()`'s `lines.markeredgewidth` = 0.35 also set the width of
error-bar caps, which matplotlib draws as markers. Every `errorbar` call in the figure scripts (three:
`fig_paper_F2_size` panels a-b and c, `fig_paper_F1`) now passes `capthick=ps.DATA_LW`, 1.0 pt, the
width on main. `tests/scan_marker_edges.py` gained rule 4: a stroke glyph (x, +, |, _, caps included)
whose width equals MARKER_EW took it from the rcParam and fails. It found one more: the 'x' key of
Fig. 4e's legend, now `mew=0.7` like the data (row 8).

**Row 4, Fig. 5 rings.** The ink rim of an open marker closed the white gap between the marker and
the ring that marks an R² from weights inside a loss burst. Each ring is now sized to the marker it
circles (`fig_paper_F5_drive.ring_size`): the farthest vertex of the marker's path (a corner of a
square, diamond or triangle) plus its stroke (rim included for an open marker), plus a gap of 0.3 pt,
plus half the ring's 0.5 pt stroke; never smaller than the old ring. Group labels are placed from the
same radius (`ring_radius`). Check (`scratchpad/r12-figs/ring_gap.py`, pass rule: every ring at least
0.3 pt from the marker it circles): Fig. 5, 16 of 17 rings pass; the 17th circles a rate-penalty
circle (its own gap passes) that Fig. 5e draws over a read-out-dropout triangle at the same values on
purpose (`see_through`, half-white faces), and the triangle crosses the ring, as on main. The
combination figure's 5 rings pass (smallest gap 0.6 pt). On main, by the same measure, 9 of
Fig. 5's 17 rings were under 0.3 pt.

**Row 7 (optional).** `fig_paper_F3_settings.py` docstrings of `panel_a` and `traces` describe the
grey circles and squares instead of the ink edge.

**Fig. 2 height.** With the counts added, the Fig. 2 float (figure and caption) was 11-23 pt taller
than the page. `FIG2_H_MM` is now 117 mm (was 125); the GridSpec margins are fractions, so every
panel shrinks by 6%. The overlap check, the panel-f line test and `test_paperstyle.py` pass at the
new height, and the paper builds with no float too large.

**Checks.** `tests/scan_marker_edges.py` on the 14 figures: PASS (rules 1-4).
`tests/scan_open_markers.py` passes on Figs. 2, 3 and 5. Extended Data Fig. 4 keeps only its failures
from main (#176), 11 now. Panel f: no drawn line crosses any text or key (line-segment test on
the rendered panel); `check_overlaps` finds no overlapping text. `test_no_function_imports.py`,
`test_scripts_start.py`, `test_paperstyle.py` pass with plain python3; no PDF holds a raster. The
drawing stage's printout is unchanged: 19 networks, 15 with a flat loss, 13 of those decreasing.

Figures: [Fig. 2](../../img/internal_figures/fig_paper_F2_size.pdf),
[Fig. 5](../../img/internal_figures/fig_paper_F5_drive.pdf),
[Fig. 1](../../img/internal_figures/fig_paper_F1.pdf),
[Fig. 4](../../img/internal_figures/fig_paper_F4_noreturn.pdf),
[combination](../../img/internal_figures/fig_ed_combination.pdf); the other nine figures of the
2226 entry were redrawn from the same commit.
