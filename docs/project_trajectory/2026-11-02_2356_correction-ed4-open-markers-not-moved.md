## 2026-11-02 23:56 — Correction to 2026-11-02_2341: Extended Data Fig. 4 open markers are no longer moved

Corrects section 2 of `2026-11-02_2341_r13-figfix-seeded-fig1ab-and-ed4-open-markers.md` (work
package r13-figfix, dead-relu-paper #176, PR #179). Section 1 of that entry (Fig. 1a,b, #165) stands.

**What was wrong.** Entry 2341 fixed the stacked open markers by moving a marker along the iteration
axis or the R² axis, by up to 0.009 in R². The coordinator rejected the fix on PR #179. Both axes
carry measured values, so a moved marker shows a value that was not measured, and saying so in the
caption does not change that. `draw_open_groups(free_axis=...)` and `report_moves` are removed, and
so are the caption sentences about the moves.

**New drawing: rings on top, nothing moved.** Every open marker is drawn as on main: a white inside,
the coloured ring and the 0.35 pt ink rim. After all markers of a panel are drawn, each marker's
coloured ring is drawn again, with no face and no rim, above every white inside
(`fig_paper_F2_fails.ring_overlay`, gid `ring-overlay`). A later marker's white inside then no longer
hides an earlier marker's ring. Lines and filled floor-crossing markers still stay hidden inside
every open marker. The open markers of the trace panels are no longer stepped right along the
iteration axis either (main stepped them by up to two marker widths), so no marker in Extended
Data Fig. 4 is off its measured value. Figure 3 keeps its own settings. Its PDF renders
pixel-identical before and after the change (300 dpi, 1950 × 1512 pixels, 0 differ).

**The transparent face was tried first and dropped.** The coordinator's first recommendation was a
transparent face with the ring and rim kept. On the scan, that drawing failed 60 times, even with
lines left out of the centre test. Three causes:
- filled floor-crossing markers under an open marker showed through it;
- the rim's inner half closed up the open triangles;
- every stacked ring still crossed its neighbours' centres.

`tests/scan_marker_edges.py` rule 2 also fails a transparent face by design, because the rim shows
inside the marker.

**Rule changes in `tests/scan_open_markers.py`, and why they are honest.** Two open markers whose
centres are closer than one marker diameter (1.17 mm) must cross if both stay at their measured
values. Each ring then runs over or near the other marker's centre. On main the scan failed 11 times
for that reason, and no drawing that moves nothing can pass the old rules on those pairs. The rules
change only for markers that have a ring on top:
- **1' (centre).** A centre that reads dark is printed as "ring crossing" and not counted if
  another such marker's outline passes within CROSS_MM = 0.42 mm of it. The 0.42 mm is half the
  ring plus the ink rim, half the 3 × 3 pixel block at 300 dpi, and 0.05 mm of anti-aliasing.
  Lines and filled markers lie under the white insides, so only another ring can darken such a
  centre. 0.42 mm is close to the marker's radius (0.46 mm). In practice, rule 1' therefore accepts
  any centre that lies inside another overlapping marker with a ring on top. Each case is printed by
  name.
- **2' (overlap).** A pair closer than half a diameter is printed as "rings on top" and not counted
  if the two differ in colour or in marker shape. A pair of the same colour and shape still fails,
  because it prints as one marker.
- **Edge scan.** `tests/scan_marker_edges.py` skips the overlay artist: it is the ring of a marker
  that is already scanned, rim included.

Markers without a ring on top are scanned exactly as before, and the rule changes were tested for
that. On main's Extended Data Fig. 4, the new scan still reports the same 11 failures. On the scan's
default figures (`fig_paper_F2_size`, `fig_paper_F5_drive`), its printout is identical to the old
scan's.

**Result.** `scan_open_markers.py fig_paper_F2_fails`: PASS, 0 rule failures, with 14 pairs printed
as "rings on top" and 27 centres printed as "ring crossing". Every pair and centre is in the
printout. `scan_marker_edges.py fig_paper_F2_fails`: 476 markers pass, 0 fail. The layout check
reports 185 texts, 11 legends and 0 problems. `pdfimages` lists no rasters.

**What the reader sees.** Where values lie within 1 mm of each other, their rings now overlap where they
were measured. In the trace panels of b to k, two to five rings form a short chain at 75,000
iterations. Overlapping pairs also appear in the R² panels of b, c, d and g to k, and in l.
Panel a has no overlapping open markers. Every ring is
visible, but the reader has to tell the values apart by colour and shape. Panel c's trace panel has
five values within 1 mm.

Figure: [img/internal_figures/fig_paper_F2_fails.pdf](../../img/internal_figures/fig_paper_F2_fails.pdf)
(+ .svg).

**Paper.** The Extended Data Fig. 4 caption is back to its text on main. Its source comment names
this entry.
