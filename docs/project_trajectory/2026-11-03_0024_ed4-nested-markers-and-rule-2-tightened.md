## 2026-11-03 00:24 — Extended Data Fig. 4: near-coincident open markers nested; scan rule 2' tightened

Work package r13-figfix, dead-relu-paper #176. This entry answers the review of PR #179 (comment
6598736858, CHANGES REQUESTED). It adds to entries 2026-11-02_2356 (R² panels and l) and
2026-11-03_0002 (trace panels). It corrects one point of entry 2356: rule 2' no longer excuses a
pair that differs in shape alone.

**What the review found.** In panel l at 200,000 iterations, two means sit 0.05 mm apart in one
colour (#0e5c64):
- the AdamW networks, a square, 218 active units at R² 0.962;
- the networks trained with Adam without weight decay, a circle, 216 active units at R² 0.962.

The circle is inscribed in the square, so the pair printed as one almost solid square. Rule 2'
excused the pair because the two shapes differ.

**Change, nothing moved.** `fig_paper_F2_fails.nested_sizes`: in a panel with rings on top, two
open markers whose centres are closer than 0.25 of a diameter are drawn concentric at two sizes.
The circle is drawn inside, at 0.9 times its usual width. The other marker is drawn outside, at 2.1
times its usual width. The ring overlay uses the same sizes. Two pairs qualify:
- panel l, the pair above;
- panel c (R² panel), the default networks (grey square) and λ_met = 0.01 (light-teal circle),
  0.02 mm apart. The review flagged this pair as optional, because the square's corners could read
  as a filled grey square.

Why the outer marker grows. The coloured ring is 0.7 pt wide and the ink rim 0.35 pt. A square of
the usual size has about 0.96 pt of white from its centre to its ring, so no circle with its own
ring and rim fits inside it. The outer marker therefore grows, and the inner one stays close to its
usual size. With the inner marker at 0.75 times its width, its own white inside was too small, and
the square's centre (0.05 mm away) read dark (darkest channel 96). At 0.9 and 2.1, both centres
read as paper, and about 0.45 pt of paper lies between the two rings.

**Scan changes (`tests/scan_open_markers.py`).**
- **Rule 2'** now excuses an overlapping pair with rings on top only if the two differ in colour.
- **New rule 2''.** A pair is printed as "nested" and not counted if the smaller marker's outline
  lies wholly inside the larger one's, at least NEST_GAP_MM from it everywhere. NEST_GAP_MM is the
  two half rings, the inner marker's rim and 0.1 mm of paper. Rule 1 still applies to both centres,
  and both pass without an exemption.
- **Falsification.** With nesting switched off (NEST_FRAC = 0), the tightened scan fails the
  panel l pair (1 rule failure, "overlap ... 0.05 mm apart"). With nesting on, it passes.

**Result.** `scan_open_markers.py fig_paper_F2_fails`: PASS, 0 rule failures.
- Exemptions: 2 pairs "nested", 2 pairs "rings on top" and 11 centres "ring crossing" (were 0, 3
  and 10).
- The new "rings on top" pair and the new crossing both come from the larger grey square in
  panel c. It now overlaps the neighbouring teal circle (0.90 mm apart; darkest channel 183 at that
  circle's centre).

**Other checks.**
- `scan_marker_edges.py`: 476 markers pass.
- Layout check: 0 problems. `pdfimages` lists no rasters.
- Figure 3: pixel-identical (0 of 1950 × 1512 pixels differ).
- The scan's printout on its default figures is unchanged.
- The three tests pass.

**Caption.** Two edits, sized so that the paper stays at 62 pages.
- First page. "Open markers show 75,000 iterations (50,000 in h), side by side in a right panel
  where they would overlap." This replaces "Open markers show 75,000 iterations, or 50,000
  iterations in h." It defines the read point, and the sentence of entry 0002 on the continued page
  is removed.
- Panel l text. "The AdamW and Adam-without-weight-decay means (218 and 216 active units, R² 0.962)
  nearly coincide and are drawn one inside the other, as are two in c."

The step size (at most 2.3 marker widths) is not in the caption. Every longer wording tried either
overflowed the first caption page ("Float too large") or pushed the next Extended Data figure off
the continued page (63 pages). The step size is in the source comment and in entry 0002. The
panel l numbers are from `fig_paper_F2_fails.py`'s printout (optimizer comparison, end: AdamW 203,
225, 226; Adam without weight decay 208, 219, 221).

Figure: [img/internal_figures/fig_paper_F2_fails.pdf](../../img/internal_figures/fig_paper_F2_fails.pdf)
(+ .svg).
