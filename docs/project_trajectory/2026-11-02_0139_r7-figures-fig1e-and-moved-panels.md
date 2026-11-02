## 2026-11-02 01:39 — Round 7 figures: Fig. 1e added; tanh and combination-test panels moved to Extended Data

Work package r7-figures (dead-relu-paper#127, round 7 brief and selection plan, risks R1-R3).
Branch `wp/r7-figures` from trainRNNbrain main 3be97d3. No new measurement: every panel is drawn
from the caches the figures already used, on the Mac, with
`/opt/miniconda3/envs/rnns/bin/python <script>` from the repository root.

### What changed

- **Fig. 1e (new).** `fig_paper_F1.py`, `panel_selective`. The panel shows the fraction of units
  selective for a task variable in the nine Fig. 1d networks, three per task. Each fraction is
  computed two ways: over all 1,000 units, and over the active units only. A unit is selective if
  its partial R² is at least 0.10 for one or more task variables (rule entry 2026-10-31_2212). The
  panel reads `data/population_stats.json`, the file Extended Data Fig. 2 draws (results entry
  2026-10-31_2220); nothing is recomputed. The script asserts that the nine seeds in that file are
  the nine seeds of panel d. Markers are filled, because the weights are the floor-crossing
  snapshots of panel d. The canvas grew from 136 to 140 mm (the cap is 141 mm). Panel e sits under
  panel c, and panel d spans the height of both.
- **Fig. 4.** The tanh gradient histograms (old panel d) moved to the tanh Extended Data figure
  (`fig_ed_tanh`, paper label `fig:ED7`) as its panel d. The transfer panels e and f became d and
  e. Panel c spans the full width. The canvas shrank from 140 to 131 mm. In the tanh figure the
  trained histograms now take the activation's colour from panels a-c (ReLU grey, tanh teal), and
  the untrained histograms are dotted ink. Before the move, grey meant "untrained" in the panel and
  "ReLU" in the figure it joined.
- **Fig. 5.** The combination test (old panel f) is a new Extended Data figure,
  `fig_ed_combination` (paper label `fig:EDcomb`). Its columns are lettered a-c (active units,
  held-out R², dimensionality), because tests/test_paperstyle.py requires a panel letter. Panels
  a-e keep their letters: e moved under c, and d spans the height of c and e. The canvas shrank
  from 141 to 133 mm.

### Numbers checked before drawing (pass rule: equal to the results entry after rounding)

Fraction selective, over all units / over active units (printed by the script):
3BFF 0.179, 0.245, 0.197 / 0.663, 0.718, 0.688; CDDM 0.543, 0.548, 0.580 / 0.986, 0.998, 0.982;
DMTS 0.343, 0.470, 0.229 / 0.929, 0.900, 0.824. The ranges match entry 2026-10-31_2220: 3BFF
0.18-0.24 / 0.66-0.72, CDDM 0.54-0.58 / 0.98-1.00, DMTS 0.23-0.47 / 0.82-0.93. The upper 3BFF
value is 245 of 1,000 = 0.245, which the entry rounds to 0.24. The shares over active units were
also recomputed from the counts (for example 179 / 270 = 0.663).

### Checks

- Layout: `check_layout` of `fig_paper_F4_noreturn.py` reports 0 overlapping text pairs and 0
  texts on a line for Fig. 4 and both Extended Data figures. `check_overlaps` of
  `fig_paper_F1.py`: no overlapping text among 86 artists. `paperstyle.save` warned that the
  first Fig. 5 draw extended 0.7 mm below the canvas; rows d and e were raised 1.5 mm, and the
  warning is gone.
- An independent probe reads the word boxes of each PDF with PyMuPDF. It found 0 overlapping words
  in Fig. 1, Fig. 4, the tanh figure and the combination figure. In Fig. 5 it found one pair: the
  two lines of the single label "equal R²". The old Fig. 5 has the same pair.
- Text of old against new PDFs (pdftotext, word multisets). Every word removed from Fig. 4 is
  added to the tanh figure; the only other change is the letter (f out, d in). The combination
  figure has the words of old Fig. 5f plus its marker key. The exception is the merge labels. Its
  columns are about twice as wide as before, so fewer open markers overlap: eight labels "3" and
  one "1/3" became five labels "2" and one "1/2". The per-network values the script prints are
  identical before and after.
- `pdfimages -list`: no raster in any of the five PDFs. `python3 tests/test_paperstyle.py` passes
  on all five. `python3 tests/test_no_function_imports.py` passes (182 files).

### Figures

- [Fig. 1](../../img/internal_figures/fig_paper_F1.svg)
- [Fig. 4](../../img/internal_figures/fig_paper_F4_noreturn.svg)
- [Extended Data, tanh](../../img/internal_figures/fig_ed_tanh.svg)
- [Fig. 5](../../img/internal_figures/fig_paper_F5_drive.svg)
- [Extended Data, combination test](../../img/internal_figures/fig_ed_combination.svg)

The status board (`docs/paper_audit.md`, rows C4, C7 and C8) now cites the new panel letters.
