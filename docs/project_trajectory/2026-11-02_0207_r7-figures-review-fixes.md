## 2026-11-02 02:07 — Round 7 figures, review fixes: merge-label counts corrected; Fig. 1e relabelled

Corrects `2026-11-02_0139_r7-figures-fig1e-and-moved-panels.md` (review of dead-relu-paper PR #147,
rows 10 and 15).

### Correction: the merge labels that changed

Entry 0139 said that eight labels "3" and one "1/3" became five labels "2" and one "1/2". That
count was wrong: I had compared word multisets over whole figures, so changes in different panels
cancelled. I recounted from the word positions in the PDFs, by region:

- **Combination test (old Fig. 5f, now `fig_ed_combination`).** Old: ten "3", seven "2" and one
  "1/3". New: two "3", fourteen "2" and one "1/2". Eight 3-network groups split into a pair and a
  single network. The "1/3" group split into "1/2" and a single network. One pair (noise row,
  active units) split into two single networks. The columns are about twice as wide as before, so
  fewer open markers overlap.
- **Fig. 5e (N = 4,000), a change entry 0139 did not report.** Old: four "2" and two "3". New: one
  "2" and two "3". Three pairs split into single networks (prune-and-duplicate, rate penalty,
  read-out dropout), because the panel is wider.
- Fig. 5a-d: unchanged.

The values do not change: the per-network printout of `fig_paper_F5_drive.py` for panel e and for
the combination test is identical before and after. The PR #147 reviewer checked all 48
row-column-measurement cells of both combination figures independently, with 0 mismatches.

### Fig. 1e relabelled (review row 10)

"all" and "active" are now tick labels under the two positions of every task. The task names sit
on a second line below them. Before, only the 3BFF pair carried the words, and "all units" sat on
the axis. Markers are drawn at `ps.MARKER_PT` (3 pt), as `docs/figure_style.md` specifies; before,
they were 3.6 pt. The axes are 1.5 mm shorter, so the second label line fits on the 140 mm canvas.
The values are unchanged; the printout is identical to entry 0139. Checks: `check_overlaps` finds
no overlap among 90 text artists, and the PyMuPDF probe finds 0 overlapping words and no text
outside the canvas. `tests/test_paperstyle.py` and `tests/test_no_function_imports.py` pass, and
the PDF has no raster.

- [Fig. 1](../../img/internal_figures/fig_paper_F1.svg)
