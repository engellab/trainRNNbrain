# Correction to entry 2026-11-06_1829: Figure 2 is 103 mm tall, and the maze rows of Supplementary Table 1 carry shorter notes and corrected panel marks (dead-relu-paper#182)

**Date:** 2026-11-06 1844
**Issue:** ptolmachev/dead-relu-paper#182
**Script:** fig_paper_F2_size.py (drawing stage), three commits after 90db2f7; every number printed is
unchanged (the printout equals the one of entry 2026-11-06_1829).

Three things changed while the paper's Fig. 2 float and Supplementary Table 1 were built from the
files of entry 2026-11-06_1829.

1. **Figure 2 is drawn 103 mm tall (FIG2_H_MM), not 117.** With the caption's maze read-out and the
   networks per size, the float was 108 pt too tall for the page at 117 mm; the caption was cut to the
   old caption's length and the figure shortened by 12 mm (one uniform scale of the GridSpec; the
   letters stay 8.5 pt, no overlapping text, tests/test_paperstyle.py ok, smallest y-label gap 1.36 mm).
   At 105 mm the float was still 2.3 pt too tall.
2. **The Note column of the maze rows is under 25 characters**: "end: 147 active, 1.00" for a learned
   run (the count and the success rate at the end of training) and "not learned; 128 at end" for a
   finished run that did not learn (its end success stays in the R^2 column, in parentheses). The
   longer notes of the first version made the longtable 85 pt wider than the line; with these the
   overfull is the 17 pt it had before the maze rows (the supervised rows' own width).
3. **Panel marks.** "1d" now marks only the learned N = 1,000 runs whose ablation file exists (seeds 1
   and 2), the networks Figure 1d draws; the first version marked every run with an ablation file
   (N = 250 seed 1, 500 seed 1, 1,000 seed 3, 2,000 seed 2), which Figure 1d does not draw. Every
   finished maze run is marked "EDM", the Extended Data maze figure (a-c; d draws the four ablated
   learned runs of N = 250 to 1,000; e the N = 1,000 runs); the paper's table defines the mark.
