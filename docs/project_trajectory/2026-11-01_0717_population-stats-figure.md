## 2026-11-01 07:17 — Extended Data Fig. 2 (population statistics) drawn; paper text written

Follows `2026-10-31_2220_population-stats-results.md`. Figure
`img/internal_figures/fig_ed_population.pdf` (and `.svg`), drawn by
`trainRNNbrain/experiments_and_analysis/fig_ed_population.py` from `data/population_stats.json`, with
the shared style `paperstyle.py` as merged to main in e28b86c (dead-relu-paper#62). Six panels, every
network drawn, value over active units (x) against over all units (y): (a) participation ratio of
rates, (b) of normalised rates, (c) selective (filled) and mixed-selective (open) counts, (d) the same
as shares, (e) k* (tasks offset in x, networks in y where values coincide), (f) matched class shares.
Checks: `tests/test_paperstyle.py` passes (Helvetica and Helvetica-Bold only, scale 1.0, median text
6.0 pt, panel letters 8.5 pt, no run under the minimum); no rasters; no text overlap or overflow
(`check_overlaps`).

Paper (dead-relu-paper branch wp/pop-stats): `sections/R1_population.tex`,
`methods/M06b_population.tex`, `floats/ED_population.tex`, `figures/fig_ed_population.{pdf,svg}`.
No new references (gao2017theory, rigotti2013importance and yang2019task are in refs.bib).
The status board row C8 is updated.
