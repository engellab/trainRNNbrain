## 2026-11-02 21:09 — Extended Data Fig. 3b points at fixed sizes; Fig. 2f threshold label moved (r11-fig2, follow-up)

Follow-up to `2026-11-02_2102_r11-fig2-jitter-and-floor-schematic.md`, same branch `wp/r11-fig2`.
The coordinator asked the package to close dead-relu-paper #167 and #169 in the same PR. Both
figures were drawn again from the existing records; no job was run.

**#167, Extended Data Fig. 3b.** This panel plots the floor crossing t* against N, one small point
per network of Fig. 2a. It used `network_offsets`, the rows Fig. 2a,b had before round 11. It now uses
`network_xs` from `fig_paper_F2_size.py`, the same placement as Fig. 2a,b: one column per size,
centred on N, placed in page millimetres. The axis limits are set before the points are placed.
No other script used `network_offsets`, so the function and its two constants (NET_GAP_DEX,
NET_STEP_DEX) were deleted. The caption now says the points are spread sideways at their size so
that none hides another.

**#169, Fig. 2f.** The label "decrease of 5 active units" sat on one line across the flat-loss band.
Its first letter touched the open circle of an N = 500 network at a loss change of about -1.3%. It
is now set on two lines ("decrease of" / "5 active units"), right-aligned at the right end of the
axis, above the dashed line, in the area right of the band where no network lies.

### Checks (pass rules set before each run)

1. **Extended Data Fig. 3b points, read from the PDF with PyMuPDF** (scratchpad
   `r11-fig2/check_dots_ed3.py`). The rules were the same as for Fig. 2a,b: one point per network,
   45 expected; widest offset under a quarter of the smallest gap between size ticks; no two centres
   closer than half a point's width (0.36 mm); no point centre within 0.85 mm of a mean marker's
   centre.

   | | points | widest offset | smallest gap between sizes | ratio | closest pair | closest to a mean |
   |---|---|---|---|---|---|---|
   | before | 45 | 3.76 mm | 6.87 mm | 0.55 (fails) | 0.57 mm | 0.92 mm |
   | after | 45 | 1.50 mm | 6.87 mm | 0.22 | 0.50 mm | 0.88 mm |

2. **Fig. 2f label, read from the PDF** (scratchpad `r11-fig2/check_flat_label.py`). The rule: the
   label's text box keeps at least 0.3 mm from every marker and error bar in the panel, from the
   right edge of the flat-loss band, and from the dashed line. Before: 0.00 mm from a marker (they
   touch), and the label crosses the band. After: 7.05 mm from the nearest marker, 0.41 mm right of
   the band (a first try at x = 6.9 left 0.18 mm and failed; the label now ends at x = 6.98),
   0.62 mm above the dashed line.

3. **No number changed.** The printouts of `fig_ed_readout.py` and `fig_paper_F2_size.py` are
   identical to those before this follow-up. The `pdftotext` word lists of both figures are
   unchanged (for Extended Data Fig. 3 compared with the copy on main, for Figure 2 with commit
   `541025d`).

4. **Style.** `tests/test_paperstyle.py` passes on both figures (Helvetica faces only).
   `tests/test_no_function_imports.py` passes. Renders at 300 and 900 dpi were read.
