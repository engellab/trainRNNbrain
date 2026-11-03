## 2026-11-02 21:56 — r11-xref: Supplementary Table 1 rows written with the paper's wording for the two DMTS exclusions (dead-relu-paper #168)

Branch `wp/r11-xref` (from main 627dc1e), package r11-xref of round 11 (Pavel's item 42,
clickable cross-references; brief `~/Documents/Projects/paper_push/ROUND11_XREF_BRIEF.md`).

**Problem (issue #168).** `fig_paper_F2_size.py` writes the body rows of Supplementary Table 1 to
`img/internal_figures/edt2_runs_rows.tex`; the paper keeps the same rows inline in
`floats/EDT2_runs.tex`. Two rows differed in the Note column: DMTS N = 2,000 seed 597040509 and
DMTS N = 4,000 seed 3271883334. The script wrote "never learned the task" (the exclusion reason set
at collect time from `MIN_R2`); the paper says "final $R^2$ below 0.8", edited by hand. A later
regeneration would have reverted the paper's wording.

**Change.** In `tex_row`, a row whose exclusion starts with "never learned" gets the note
`final $R^2$ below {MIN_R2[task]}`, which prints "final $R^2$ below 0.8". The collect-time string
in the pickle is unchanged, because `apply_decisions` and `threshold_sensitivity.py` match its
prefix "never learned". No Della read was needed: the table was rewritten from the existing
`data/fig_paper_F2_size_collect.pkl` by calling `write_table(apply_decisions(record))`, so the
Fig. 2 drawing was not regenerated.

**Check (pass rule fixed in the brief before the work: the rows file equals the rows of
`floats/EDT2_runs.tex` byte for byte).** Before the change, 2 of 61 rows differed, the two DMTS rows
above. After it, 0 of 61 differ. A second comparison, a plain `diff` of the rows file (header
comment removed) against the lines between `% ROWS BEGIN` and `% ROWS END` of the paper's float,
reports no difference. The cross-reference edits of the paper branch touch only the caption and
notes of that float, not its rows. `tests/test_no_function_imports.py`,
`tests/test_scripts_start.py` and `tests/test_paperstyle.py` pass (plain `python3`).

**Paper side (dead-relu-paper branch `wp/r11-xref`).** Every figure and table reference is now one
link covering name, number and panel letters (Pavel's `\Figref`, `\ExtFigref`, `\ExtTableref`,
plus `\SuppFigref`, `\SuppTableref`, `\Figsref`, `\Figrefextra` on his pattern). Every pointer to
Methods, Supplementary Methods or a Supplementary Note is a link reading "Methods: <subsection
name>". Checks, scripts in `~/Documents/Projects/paper_push/scratchpad/r11-xref/`:
`textdiff.py` (word diff of `pdftotext -layout` against 99de503: every differing word is a pointer,
apart from page-break moves and two text-extraction artefacts listed in the PR) and `linkcheck.py`
(357 of 357 cross-reference links jump to the page of their target, found from the PDF's own
captions and headings; heading links also land within 20 pt above the heading).
