## Figure 4a: the 3BFF "no silent unit" panel text scoped to the Fig. 1d networks (dead-relu-paper #127, PR #140 N3)

Work package final2-c-fix, trainRNNbrain branch `wp/final2-c-fix` (from main cd2e0db).

**Question.** Panel a of Figure 4 printed "no silent unit in any 3BFF network". The caption and
Results 4 limit the statement to the 3BFF networks of Fig. 1d. Is the broad version established?

**Answer: no.** Every check of silent units in 3BFF networks is a C4 measurement on the three
3BFF networks of Fig. 1d: entries 2026-10-31_2207 and 2026-11-01_1021 ("3BFF has 0 silent units"),
and 2026-11-01_1200 (six trial batches of 1,024 trials each, on the same three networks: "no 3BFF unit is silent in any
batch"). No entry counts silent units in the 3BFF size series, the setting sweeps or the per-unit-rule
networks. (Entry 2026-10-16_1115 says "On 3BFF no unit is silent" about the C4 networks of that
date, not about every 3BFF network.)

**Change.** `fig_paper_F4_noreturn.py draw`, panel text now "no silent unit in the 3BFF networks of
Fig. 1d". Drawn from the existing caches. Checks: the script's layout check reports 0 overlapping
text pairs and 0 texts on a line; the printout is identical to the run before the edit; a pdftotext
word diff against the figure in the paper shows only the changed words; at 400 dpi the pixel
difference is confined to the box of that text. Figure: `img/internal_figures/fig_paper_F4_noreturn.pdf`.

The status board (`docs/paper_audit.md`, C4 paragraph) also says "On 3BFF no unit is silent" in
the context of these networks. I did not change the board.
