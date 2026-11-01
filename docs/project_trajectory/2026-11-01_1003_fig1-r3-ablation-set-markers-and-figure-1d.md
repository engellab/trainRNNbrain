## 2026-11-01 10:03 — fig1-r3: the network set of Extended Data Fig. 1, the marker convention in Figs. 1 and ED 1-2, Figure 1d redrawn

Work package fig1-r3 (round 3; dead-relu-paper#89, #85, #83, #78, #81 items 1-2). trainRNNbrain
branch `wp/fig1-r3`; paper branch `wp/fig1-r3`.

### Which networks Extended Data Fig. 1 tests (#89)

The figure reads the 36 files `data/c1_ablation_seeded/*__end.json` (Della `c1_ablation_seeded`,
written 14 Oct 07:46-09:19 by the file times), one DMTS network with intact R² 0.418 (s597040509)
left out. `fig_supp_ablation.py` now prints the set by task, size and seed:

| task | N | networks | measured at |
|---|---|---|---|
| 3BFF | 500 / 1,000 / 2,000 / 4,000 | 6 / 3 / 3 / 3 | 75,000 iterations (end of the second of three runs) |
| CDDM | 500 / 1,000 / 2,000 / 4,000 | 3 each | 100,000 (end of training) |
| DMTS | 500 / 1,000 / 2,000 | 3 / 3 (incl. s2946835052) / 2 | 150,000 (end of training) |

Compared with Supplementary Table 1 (the size series): 34 of the 35 are among the 37 networks Fig. 2a
fits at N ≥ 500. The 35th is DMTS s2946835052 (learned, R² 1.000, no floor crossing: its loss ends
below its fitted floor). Three networks of Fig. 2a are not tested: **3BFF s4186074845**, DMTS
s632794669 (N = 2,000) and DMTS s2627187151 (N = 4,000); no N = 100 network is tested. All four of
those groups were trained after 14 Oct (entry 2026-10-15_2213; FFseed4 = s4186074845, entry
2026-10-31_1532).

**Correction to the committee's account in #89.** The issue names 3BFF s3374117298 as the 3BFF
network without an ablation file. It is s4186074845. The 3BFF ablation files carry the seeds of the
second runs (`NBitFlipFlop_*_c75000`), not the size-series folder seeds, so the mapping has to be
made. Evidence:
1. Folder names in `data/fig_paper_F2_size_collect.pkl`: s3374117298, s1914910909 and s3866906883
   have `MI=125000` (the third run of a 40k + 35k + 125k chain, Methods M03); s4186074845 has
   `MI=200000`, a single run. Only the three chains at N = 1,000 have 75,000-iteration second runs,
   and there are three N = 1,000 ablation files.
2. Per network, the ablation file's active count at 74,990 iterations against each size-series
   network's participation trace averaged over 74,000-76,000 (difference = trace minus file):

   | ablation file (N = 1,000) | s3374117298 | s1914910909 | s3866906883 | s4186074845 |
   |---|---|---|---|---|
   | s3095970260 (242 active) | +33 | +43 | **0** | −7 |
   | s3502253762 (278) | **−3** | +7 | −36 | −43 |
   | s3039548724 (288) | −13 | **−3** | −46 | −53 |

   The counts alone do not separate s3866906883 from s4186074845 for the 242-unit file (0 against
   −7); the folder names (point 1) do. At N = 2,000 and 4,000 every file matches one network
   within 3 units and the next best by 24 or more. At N = 500 the six counts lie within 41 units of
   each other and matching is not unique; the folder names put all six chains at N = 500 in the
   same cell, so the set there is the whole cell either way.
3. Dates: the ablation files are from 14 Oct; s4186074845 was submitted on 16 Oct (job 15657728).

The paper now describes the set exactly (Methods M06, the ED Fig. 1 caption, Results 1 "35 default
networks of 500 to 4,000 units (Methods)"), and no sentence calls it every network of the size
series. Supplementary Table 1's Fig. column does not yet mark which rows ED Fig. 1 draws; that file
belongs to the package that owns Supplementary Table 1 (issue opened on dead-relu-paper). The
alternative #89 offers, ablating the three missing networks, was not taken: the claim ("removing the
inactive units costs less R² than removing as many at random") holds in all 35 by a margin of at
least 0.137 R² (`fig_supp_ablation.py` printout, unchanged), and Della asked for a password today.

The board's C1 paragraph now says the same (docs/paper_audit.md).

### Markers (#83: filled = floor crossing, open = 75,000 iterations, fill means nothing else)

- Fig. 1a: inactive glyphs are light grey dots, no longer hollow; the classes differ by colour.
- Fig. 1c: filled (floor crossing), unchanged.
- Fig. 1d: the retrained networks are filled stars (every inactive unit removed) and filled
  triangles (as many removed at random); the weights are the floor-crossing snapshots.
- ED Fig. 1: every marker open (3BFF at 75,000 iterations; CDDM and DMTS at the end of training,
  also a fixed iteration and not the floor crossing); every inactive unit removed is a circle, as
  many at random a triangle (shape, as in Fig. 1d).
- ED Fig. 2: every network at its floor-crossing snapshot, so markers are filled; mixed-selective
  units are crosses in the task colour instead of open task markers.

Decision recorded here: for a value at the end of training (neither measurement point), the figure
uses open markers and the caption names the iteration, so that a filled marker always means the
floor crossing.

### Figure 1d (#85)

- The left spine is 4 pt (1.4 mm) outward, so the busiest-first curve, which reaches R² = 0 within
  the first 1-2% of units, runs clear of it.
- The inactive units are a solid grey bar at the top of each sub-panel (R² 1.055-1.09), not an
  alpha band behind the curves; the legend swatch is the same solid colour.
- The retrained values sit in a strip right of each sub-panel ("retrained"), stars and triangles in
  separate columns, so the 3BFF difference (0.945-0.948 against 0.920-0.930) is visible.

`tests/test_paperstyle.py` passes on all three PDFs; `check_overlaps` reports no overlapping text.
Numbers in the figures are unchanged (same collect record and ablation files).

### Text (#78, #81)

- Fig. 1 title names ReLU.
- Results 1: "busiest first (by decreasing participation)"; the closing sentence no longer says
  removal "costs no R² on CDDM"; held-out R² is defined in one clause (the trial detail stays in
  Methods); Results 1 is 388 words, 5.9 numerals per 100. With the population paragraph at #81's
  limit of 200 words, Results 1 totals 588 words against the blueprint's 500 (18% over; the 15%
  margin would be 575).
- Population paragraph rewritten statistic by statistic, each with every task in turn, 200 words
  (#81 acceptance limit), 5.5 numerals per 100. New derived values, from entries 2026-10-31_2220 and
  2026-11-01_0741: normalised-rate dimensionality over all units / over active units 1.84-2.02 on
  3BFF ("about twice"), 1.65-1.71 on CDDM ("1.7 times"), 0.52-0.79 on DMTS; with the divisor
  softened by 5% of the active units' median sd, 1.24-1.33, 1.22-1.24 and 1.09-1.13 ("1.1-1.3 times").
