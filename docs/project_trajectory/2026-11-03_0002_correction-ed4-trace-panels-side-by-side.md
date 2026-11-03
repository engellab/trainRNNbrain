## 2026-11-03 00:02 — Correction to 2026-11-02_2356: Extended Data Fig. 4 trace panels set stacked open markers side by side again

Corrects the trace-panel part of `2026-11-02_2356_correction-ed4-open-markers-not-moved.md` (work
package r13-figfix, dead-relu-paper #176, PR #179). That entry's handling of the R² panels and
panel l stands, as do its rules 1' and 2'.

**What was wrong.** Entry 2356 stopped stepping the open markers of the trace panels (the right
panel of each pair). As a result, two to five near-equal groups formed chains of overlapping rings
at 75,000 iterations, and the coordinator could not count them (review of PR #179). Every open
marker in a trace panel sits at one read point: 75,000 iterations, or the budget where that is
shorter. A small sideways step there separates markers that share one x value, as the jitter at
fixed N does in Fig. 2. Provided the caption says so, the step misstates no measurement.

**Change.** Trace panels: `draw_open_groups(..., max_shift=2, both_sides=True)`. A marker that would
overlap one already placed takes the first free position of 0, +1, −1, +2, −2 steps of 1.15 marker
diameters, the step size main used. Main stepped only to the right, at most two steps, and in
panel c five groups ran out of steps and stacked. The largest step used is two (25 markers at 0
steps, 16 at one step, 2 at two steps). These markers have no ring overlay, so the scan's
exemptions do not apply to them. R² panels and panel l are unchanged from entry 2356: nothing moves,
and the rings are redrawn on top. Figure 3 keeps its own one-sided step. Its PDF renders
pixel-identical (300 dpi, 0 of 1950 × 1512 pixels differ).

**Checks.**
- `scan_open_markers.py fig_paper_F2_fails`: PASS, 0 rule failures. The trace panels pass without
  any exemption. The exemptions all fall in the R² panels and l: 3 pairs print as "rings on top" and
  10 centres as "ring crossing" (were 14 and 27).
- `scan_marker_edges.py`: 476 markers pass.
- The layout check reports 0 problems, and `pdfimages` lists no rasters.
- The scan's printout on its default figures is unchanged.

**Caption.** One sentence, on the continued page because the first caption page is full: "Open
markers that would overlap in a right panel are drawn side by side at their read point." The paper
builds to 62 pages, as on main.

Figure: [img/internal_figures/fig_paper_F2_fails.pdf](../../img/internal_figures/fig_paper_F2_fails.pdf)
(+ .svg).
