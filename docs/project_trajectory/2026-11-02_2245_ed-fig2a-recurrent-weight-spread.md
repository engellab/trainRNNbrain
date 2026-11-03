## 2026-11-02 22:45 — Extended Data Fig. 2a: s.d. of the recurrent weights on the 39 confirmatory networks; paper text

Work package r12-weights (dead-relu-paper #127, Pavel's item 52). The manager accepted the
confirmatory result (`2026-11-02_2232_connectivity-confirm-result.md`) and asked for W1 only in the
paper; W3 and W4 stay in the notebook.

**Figure.** `fig_ed_population.py` panel a now draws W1 from `data/connectivity_confirm.json`:
over the active units (x) against over all units (y), log axes from 0.003 to 0.3, one marker per
network in its task's colour and shape. A grey line at the network's value over all units spans the
range over the 200 random sets of |A| units. Fill follows the paper's rule: filled for the two DMTS
networks read at kept snapshots within 1,000 iterations after the floor crossing (#38, #39), open
for the 37 read at saved weights of a fixed iteration. Every marker in the figure now has the ink
edge of Pavel's item 54 (MARKER_EW 0.35 pt): the script uses paperstyle's `MARKER_EW`,
`mark_floor` and `mark_fixed` when paperstyle defines `MARKER_EW` (r12-figs, dead-relu-paper PR
#177), and sets the same keywords itself until then (filled: ink edge; open: white inside, coloured
ring, ink rim by path effect). The crosses of panels c, d are strokes and stay without an edge.
The panel replaces the participation ratio of the rates; panel b (normalised rates) is unchanged.
Checks: `check_overlaps` 0; `pdfimages -list` shows no raster; one font family.
[PDF](../../img/internal_figures/fig_ed_population.pdf)

**Paper** (dead-relu-paper branch wp/r12-weights): ED Fig. 2 caption panel a; two Results 1
sentences (W1 over the 39 networks, and in CDDM the weight-decay cause from the #172 test);
the Introduction list names the spread of the recurrent weights with `schuessler2020interplay`
(Frobenius norm and singular values of the full trained W and of W − W0; full text checked);
Methods "Population statistics" gains a paragraph on the comparison, its read point and its rule;
Supplementary Methods gains "Spread of the recurrent weights" (inventory table, read points, the
weight-decay comparison, the notebook entries). The definition of dimensionality stays in Methods,
because ED Fig. 2b, the combination figure, ED Table 5 and the Supplementary Notes use it;
`gao2017theory` moved there from the old panel a caption.

The Results sentence on the cause is scoped to CDDM. In 3BFF networks, the s.d. over all units
(0.040-0.042) is close to the initial s.d. (about 0.037), and the gap to the active units comes
from the large weights among active units (0.098-0.200), so shrinkage of the inactive weights is
not the whole cause there.
