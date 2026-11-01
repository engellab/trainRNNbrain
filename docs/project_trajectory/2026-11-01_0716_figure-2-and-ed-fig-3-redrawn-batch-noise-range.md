# Figure 2 and Extended Data Fig. 3 redrawn in the house style; the snapshot estimate's batch-noise range

Work package fig2-r2 (ptolmachev/dead-relu-paper#53), trainRNNbrain branch `wp/fig2-r2` (from
`wp/figure-style`, 3b39d86). Follows entry `2026-11-01_0118`, which set the rule below before the
computation ran. No new Della read: everything comes from the collect record of Della job 16558986
(`data/fig_paper_F2_size_collect.pkl`) and, for the ranges, the CDDM snapshot losses of Della job
15635966 copied to the Mac.

## Batch-noise range of the snapshot estimate (CDDM default networks, N = 1,000)

`fig_ed_readout.py --batch-noise` (1,000 resamples of each snapshot's 32 batch losses; 5th-95th
percentile; 99 snapshots per network; no loss bursts in these networks):

| seed | snapshot estimate | record `t_oracle` | 90% range from batch noise | floor crossing t* |
|---|---|---|---|---|
| 1423687489 | 17,000 | 17,000 | 16,000-26,000 | 25,020 |
| 2657899762 | 16,000 | 16,000 | 15,000-28,000 | 26,028 |
| 2670295995 | 24,000 | 24,000 | 16,000-27,000 | 24,219 |

- Rule 1 holds: the re-implementation reproduces the recorded estimate on all three networks, so the
  ranges are drawn (Extended Data Fig. 3c, horizontal bars). No resample lacked a crossing.
- Each of the three floor crossings lies inside its network's range. On these three networks, the
  difference between t* and the snapshot estimate (+47%, +63%, +1%) is within what batch noise alone
  moves the snapshot estimate by.
- Against entry `2026-10-15_2054` (15,000-25,000 and 17,000-26,000 on two unnamed controls, an
  unkept script and an earlier floor), the new ranges are 1,000-3,000 iterations wider at the top.

## Redraw

Both figures are drawn at 165.1 mm with `paperstyle` (Figure 2 125 mm tall, Extended Data Fig. 3
102 mm, under the 0.62 and 0.45 `\textheight` caps of their floats, so both print at 100%). Task
colours from `ps.TASK_COLORS`; Fig. 2e is all 3BFF, so marker shape codes N; Extended Data Fig. 3a
marks loss bursts as grey ticks along the top instead of a tan band. Ticks use `count_ticks`
("1,000", "100,000"); no "k". Label changes: "read-out margin" -> "margin over the loss floor";
"crossing of the rule" -> "floor crossing t*"; "snapshot oracle" -> "snapshot estimate"; the ∝
glyph is gone from the figure (it falls back to STIX). Extended Data Fig. 3 gains panel d, the
difference between t* and the snapshot estimate in percent of the snapshot estimate, with the CDDM
extremes (−33% and +64%) printed on it, because the Discussion quotes them (#43).

The drawing of Extended Data Fig. 3 moved from `fig_paper_F2_size.py` (`draw_ed_readout`) to the new
`fig_ed_readout.py`; the collect stage stays in `fig_paper_F2_size.py`. Supplementary Table 1 rows
(formerly Extended Data Table 2) now write "200,000" and "59,000" instead of "200k" and "59k", and
mark the two networks trained on to 300,000 iterations with a superscript c.

Audit (`python tests/test_paperstyle.py`): both PDFs pass, fonts Helvetica, Helvetica-Bold and
Helvetica-Oblique only, median text 6.0 pt, panel letters 8.5 pt; text-overlap check 0 pairs in
each. Every number the drawing stages print is unchanged from entry `2026-10-31_1957`.

## Removed from Results 2

The projected N at which the fitted laws reach 1,000 active units (7,300 on CDDM, 9,200 on DMTS,
14,200 on 3BFF; entry `2026-10-31_1957`, board C2) is no longer in the paper: it is an extrapolation
beyond the measured sizes, it is not drawn, and no claim depends on it.
