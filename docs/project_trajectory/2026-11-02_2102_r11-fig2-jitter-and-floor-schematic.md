## 2026-11-02 21:02 — Figure 2: network points at fixed sizes, and a floor-crossing schematic (r11-fig2)

Paper push round 11, package r11-fig2 (dead-relu-paper #127), Pavel's feedback of 1 Nov, items 46
and 47. Branch `wp/r11-fig2`. Script: `trainRNNbrain/experiments_and_analysis/fig_paper_F2_size.py`,
drawing stage only, from the existing collect record `data/fig_paper_F2_size_collect.pkl`
(31 Oct). No network was read again and no job was run.

### What changed in the figure

**Item 46, the single-network points of panels a and b.** Pavel: the points were spread so widely
that a reader could not tell whether the networks have fixed sizes (N = 100, 500, 1,000, 2,000,
4,000) or random sizes. The old rows put each task's points in a line beside the mean; the outermost
point sat more than half of the way to the next size (measured below). Now the points of one size
form a narrow column centred on N. Placement is done in millimetres on the printed page
(`network_xs`): the points are taken in order of value, and each goes to the slot nearest N
(0, then ±0.5 mm, ±1.0 mm, ...) at which it is at least 0.5 mm from every point already placed
and at least 0.88 mm from every mean marker. The y values are unchanged, and N is never moved on
the data axis (decision #83): the offset is a drawing offset only. The caption says the points are
spread sideways at their size so that none hides another.

**Item 47, the schematic.** New panel d draws the floor crossing on one real network: CDDM,
N = 1,000, seed 3833537590, the first CDDM read-out example in the collect record (no loss bursts,
one entry into the 1.05× band). The panel shows the smoothed noisy training loss divided by the
fitted loss floor, the band from the floor to 1.05 times the floor, t* = 27,259 iterations as a
filled marker (smoothed loss / floor = 1.0500 there), and the hold to 1.2 t* = 32,711 as a double
arrow. The fitted floor curve and the noisy loss are left out: at print size the fitted curve lies on
the smoothed loss, and the 10th to 90th percentile of the noisy loss (0.93 to 1.18 times the floor
near t*) would cover the panel.

**Layout and letters.** Stacking the schematic under the old panel e left that panel about 27 mm
tall (it is 50 mm), and a third panel in the bottom row needed about 173 mm of the 151 mm available.
The schematic therefore went into the top row beside panel c, which plots the exponent at the 1.05,
1.10 and 1.20 margins. Letter map: new d = schematic, old d → e, old e → f; a, b and c are
unchanged. The canvas stays 165.1 × 125 mm. Panels a and b now span N = 80 to 4,800 (was 75 to
5,500) so their tick labels fit the narrower panels.

The Supplementary Table 1 rows (`img/internal_figures/edt2_runs_rows.tex`) change only in the Fig.
column (2d → 2e, 2e → 2f): checked by applying that substitution to the committed file and
comparing with the new one (identical).

Extended Data Fig. 3b (`fig_ed_readout.py`) imports the old `network_offsets`, so that function and
its two constants stay in the script unchanged; that panel still uses the old wide rows.

### Checks (pass rules set before each run)

1. **Placement of the network points, measured independently.** A separate script (scratchpad
   `r11-fig2/check_dots.py`) reads the PDF's vector paths with PyMuPDF, not the script's
   coordinates. Rules: (i) each of panels a and b holds exactly as many small points as networks
   read; (ii) every point's centre is within a quarter of the smallest gap between neighbouring size
   ticks; (iii) no two centres are closer than half a point's diameter.

   | | points (networks read) | widest offset | smallest gap between sizes | ratio | closest pair (point 0.71 mm wide) |
   |---|---|---|---|---|---|
   | before, panel a | 45 (45) | 3.97 mm | 7.26 mm | 0.55 (fails ii) | 0.60 mm |
   | after, panel a | 45 (45) | 1.51 mm | 6.16 mm | 0.24 | 0.51 mm |
   | after, panel b | 45 (45) | 1.50 mm | 6.16 mm | 0.24 | 0.51 mm |

   The new figure passes with a narrow margin on rule (ii): 0.24 against 0.25. The same script
   gives 0.90 mm as the least distance from a point to a mean marker's centre, so no point is hidden
   under a mean (a mean marker's radius plus a point's radius is 0.88 mm). Two earlier designs
   failed and were dropped. A value-ordered zigzag put two CDDM points at N = 100 0.28 mm apart.
   A one-sided beeswarm reached 2.1 to 2.6 mm (ratio 0.34 to 0.42), because three networks of one
   task often have nearly equal counts.

2. **The panel-d crossing, recomputed independently.** From the stored smoothed curve (400
   log-spaced points), my own loop found the first iteration after 2,000 from which the curve stays at
   or below 1.05 times the floor through 1.2 times that iteration: 27,295 against the rule's 27,259
   (0.13% apart). The rule was "within 2%". For the other five 3BFF and CDDM examples the
   recomputed value lies 1.0% to 2.9% above the stored t*, never below it. Two of these exceed
   2%. The cause is the coarse stored grid, not a fault in the rule: neighbouring points of the
   stored curve are about 3% apart, not the 1.5% I assumed when I set the threshold. Only the
   network that passes is drawn.

3. **No number changed.** The script's printout before and after, with the old panel headings d and
   e renamed e and f, is identical line for line. The new printout adds one line for panel d.
   `pdftotext` of the old and new PDFs: the new text is panel d's words and ticks; the only text
   missing is fragments of the rotated "every unit active" label, which pdftotext splits differently
   now that its angle follows the diagonal.

4. **Style.** `tests/test_paperstyle.py`: ok (Helvetica, Helvetica-Bold and Helvetica-Oblique only;
   letters 8.5 pt; least y-label gap 1.36 mm). A first version wrote t* with mathtext's `\ast`,
   which pulled in STIXGeneral, and the test failed; the asterisk is now Helvetica's own.
   `tests/test_no_function_imports.py`: ok. The script's text-overlap check: no overlap among
   107 text artists. 300 and 900 dpi renders were read for overlaps.

The figure on page 6 of the paper keeps its caption on the same page (paper branch `wp/r11-fig2`).
