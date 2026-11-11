## 2026-11-11 04:24 — r23-figures: figure accuracy fixes from the round-22 committee (dead-relu-paper #232)

**Why.** The round-22 committee (dead-relu-paper #231, paper main b49b481) rated every axis `equals` or better and
listed figure defects to fix before the paper goes to the author. This package fixes six of them and nothing else.
No new training or analysis; every number is from a figure script's printout or from a data file named here.

### Code changes (branch wp/r23-figures, from main ae2f432)

- `fig_paper_F5_tasks.py`: panel c's two columns are 30 mm wide (were 31). Panel d's rotated y label ran into the
  frame of the CDDM column of panel c (Reviewer S, figures 1). Only panel c's drawing moves; panel d and panels a, b
  render pixel-identical to main.
- `fig_paper_F4_noreturn.py`, panel e: the number printed beside the one old N = 4,000 network near the pass
  threshold (613) is gone; the old networks' mean is printed right of its bar at both sizes (244 at N = 1,000, 508 at
  N = 4,000; printout "old mean 244.0" and "507.7"). The pass criterion applies to that mean (Reviewer S, figures 2).
- `fig_ed_maze.py`, panel e: a network without a learned point at its read is a cross in the task colour at both of
  its counts and enters no mean tick, as Fig. 3c marks a network that did not learn. These are the three tanh and the
  three shifted-sigmoid networks at radius 1.2, all at update 2,000 (printout: "0 of 3 learned by the read" for both;
  every other column "n of n"). They were open circles and squares (Reviewer S, figures 3). Legend: "did not learn".
- `fig_paper_F3_tasks.py`: a cell measured only at another iteration (CDDM tanh, 150,000) draws no control band; its
  band in the R² row stood beside no marker (Reviewer S, figures 4). The column is now empty and the caption says why.
- `tests/scan_text_on_axes.py` (new): every text of a figure against the frame of every other axes and the canvas
  edge. The scripts' own check (`fig_ed_maze.check_overlaps`) tests text against text only and missed the Fig. 5d
  label. Pass rules were set before the first scan. On main's Fig. 5 the scan fails (the 5d label overlaps a CDDM
  frame of panel c by 1.6 mm, text box including its padding); after the change it passes.

### Checks

- Render comparison against the paper's main figures at 300 dpi (any channel differing by more than 6 of 255): Fig. 3
  differs only in the CDDM tanh slot (the two bands); Fig. 4 only at the three labels of panel e; Fig. 5 only inside
  panel c (x < 81.5 mm, y > 80 mm); Extended Data Fig. 2 only in panel e (its legend and the tanh and shifted-sigmoid
  groups). Extended Data Figs. 7 and 8 (drawn by the same script as Fig. 4) are pixel-identical.
- `tests/scan_marker_edges.py`: Figs. 3, 4, 5 and Extended Data Fig. 2 pass. `tests/scan_open_markers.py`: Figs. 3, 5,
  Extended Data Fig. 2 and Fig. 4 pass. Extended Data Figs. 7 and 8 have 5 dark open-marker centres, unchanged from
  main (outside this package; reported on a new issue).
- `tests/scan_text_on_axes.py`: Figs. 3, 4, 5 and Extended Data Figs. 2 and 7 pass. Extended Data Fig. 8 fails on its
  panel d x label, whose box reaches past the bottom canvas edge; unchanged from main (outside this package; reported
  on the same issue).
- Fig. 1d caption: the 3BFF R² values come from `data/fig_paper_F1_collect_v2.pkl` (ablation, 3-bit flip-flop,
  `quiet_first` at k = number of inactive units): 0.260, 0.347 and 0.517 with every inactive unit removed (733, 658 and
  713 units); from there until k = 950, 0.157-0.272, 0.148-0.438 and 0.171-0.307, so 0.15 to 0.45 holds.

### Not done

- Optional item (Extended Data Fig. 9a-c rings for the 3BFF rate-penalty networks with non-finite losses): panels a-c
  draw one marker per network, and the two networks of one size lie within 1 mm of each other at N = 2,000 in panel a
  (fold change 4.8 and 4.75), so their rings would fail rule 3 of `tests/scan_open_markers.py` (rings at least 1.0 mm
  apart). Left as it is.
