## 2026-11-03 08:51 — Fig. 2d becomes a schematic of the floor crossing (package r14-floor, dead-relu-paper #127)

Corrects the panel d of `2026-11-02_2102_r11-fig2-jitter-and-floor-schematic.md`.

**Why.** Pavel, 3 Nov: "Loss floor figure is not properly shown, it is not clear that the training
loss approaches it assymptorically. If difficult to show with real data, You may use schematic
instead". Fig. 2d drew the smoothed loss of CDDM N = 1,000 seed 3833537590 from 0 to 45,000
iterations. The curve ended at 1.03 times the floor, so the panel never showed the approach.
The manager chose a schematic; Extended Data Fig. 3a already shows real losses with their fitted
curves.

**What panel d now draws.** The curve of Eq. (floor) in Methods, L∞ + A exp[−((t − t0)/s)^β],
divided by L∞, so the floor is at 1. Parameters: A/L∞ = 2.55, s = 9,250, β = 0.70, t0 = 145. The
iteration axis runs to 2.5 t* and has no tick values. The panel title is "schematic". The floor,
the dashed 1.05 × floor line, the grey band between them, the t* marker and the arrow from t* to
1.2 t* are drawn as before. The "1.05× floor" label moved from the left end to the right end,
above the band: at the left end the new curve ran through it. The curve is smooth and decreasing;
small fluctuations were left out because they would add clutter without showing the definition
any better.

Printout of `fig_paper_F2_size.py`: t* 65,543, hold to 78,651, axis end 163,857, curve value at
the axis end 1.00145 times the floor.

**Where the parameters come from.** The collect record (`data/fig_paper_F2_size_collect.pkl`)
stores each read-out example's fitted curve but not its parameters. I recovered them by fitting
log(curve/L∞ − 1) = log(A/L∞) − ((t − escape)/s)^β to the stored curve with
`scipy.optimize.curve_fit` (largest relative error 1e-14, so the recovery is exact). Scratch
script: `~/Documents/Projects/paper_push/scratchpad/r14-floor/refit_all.py`.

| example (N = 1,000) | A/L∞ | s | β | t* | end of training | loss / floor at the end |
|---|---|---|---|---|---|---|
| 3BFF s1914910909 | 2.55 | 9,253 | 0.70 | 59,917 | 200,000 | 1.001 |
| 3BFF s3866906883 | 2.35 | 10,543 | 0.74 | 62,703 | 200,000 | 0.997 |
| 3BFF s4186074845 | 2.55 | 8,976 | 0.67 | 67,871 | 200,000 | 0.999 |
| CDDM s3833537590 | 2.36 | 100 | 0.24 | 27,259 | 100,000 | 1.014 |
| CDDM s1272763220 | 1.91 | 245 | 0.28 | 26,535 | 100,000 | 1.010 |
| CDDM s457853 | 1.52 | 420 | 0.30 | 23,929 | 100,000 | 1.008 |

The DMTS examples have a late-level floor and no fitted curve. The schematic takes the shape of
the 3BFF s1914910909 fit, not the CDDM fit the brief suggested. With β = 0.24 the CDDM curve
comes within 0.15% of its floor only after 15 t* (417,000 iterations, beyond its 100,000);
the other two CDDM fits need 10.6 and 11.3 t*. An axis long enough to show that approach would
make the hold window from t* to 1.2 t* narrower than 1 mm on the page.

**Supplementary Table 1.** The row of CDDM seed 3833537590 no longer lists "2d", because panel d
draws no network. `write_table` now asserts that no row lists 2d. No other row changed
(`edt2_runs_rows.tex` diff: one line).

**Checks.**
- Figure size unchanged: the PDF page is 468 × 331.65 pt (165.1 × 117.0 mm); no raster images
  (`pdfimages -list` empty); Helvetica only.
- Other panels unchanged: a pixel diff of the 2,000-px renders before and after changes pixels
  only at x 1,634-1,955 and y 37-629, which is panel d's axes and title. Panels a-c, e and f and
  panel d's y axis have no changed pixel. The printout differs only in panel d's line and in the
  number of text artists checked (108 against 111: the three iteration tick labels are gone).
- Text: `check_overlaps` finds no overlapping text. A scratch check of panel d's text boxes against
  densely sampled points of the curve, the guide lines, the marker and the arrow
  (`scratchpad/r14-floor/check_panel_d.py`) finds no text on the curve (least gap 2.9 mm, the
  title). The vertical guide lines at t* and 1.2 t* touch the "t* " and " 1.2 t*" boxes only on
  the padding space at the anchored edge, as in the previous panel.
- `tests/test_no_function_imports.py`, `test_scripts_start.py`, `test_paperstyle.py`: pass with
  `python3`.

**Figure.** [fig_paper_F2_size.pdf](../../img/internal_figures/fig_paper_F2_size.pdf)
(also `.svg`). Paper: branch wp/r14-floor, caption of panel d in `floats/F2.tex`.
