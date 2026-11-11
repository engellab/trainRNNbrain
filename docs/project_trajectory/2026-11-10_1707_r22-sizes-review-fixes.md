## 2026-11-10 17:07 — Round-22 sizes package: changes after the independent review of dead-relu-paper PR #227

Follows `2026-11-10_0826_r22-sizes-figures-and-paper.md`. The review (PR #227, comment 6783099412) recomputed
every new number with independent code and found them correct. It required five text changes and named two
problems that were already on main. No number changed; no network was read again.

**Fig. 2a,b.** The leftmost CDDM N = 100 marker touched the y axis (also on main d3b7912). `SIZE_XLIM` changed from
(80, 4800) to (70, 4800), and the "every unit active" diagonal now starts at the axis limit. Check (written for this
change, scratchpad `r22-sizes/tools/spine_gap.py`): the smallest gap between a marker's edge and the left spine is
1.17 mm in panel a and 1.67 mm in panel b. The script's printout is identical to that of 6da8ad8.
`tests/scan_marker_edges.py` and `tests/scan_open_markers.py`: PASS. No overlapping text, nothing past the canvas
edge, no rasters.

**Paper text (dead-relu-paper `wp/r22-sizes`).**
1. Results 1, Maze paragraph: "After the first PPO update, 576 to 632 of their units are active", with a pointer
   to Methods: Participation (the six N = 1,000 networks; first-probe counts as in the 0826 entry). For the word
   limit, the Results 2 sentence "In the largest 3BFF and CDDM networks, of 4,000 units, fewer than a fifth of the
   units are active" moved to the notes of Supplementary Table 1 (3BFF 517-598, CDDM 756-776 of 4,000).
2. Methods: Statistics defines "bends at small N": the exponent over every size lies outside the bootstrap 95%
   interval of the exponent over N ≥ 500.
   - It bends for CDDM (0.571 against 0.431-0.456) and for Maze at the learned point (0.584 against 0.291-0.395).
   - It does not bend for 3BFF (0.516 against 0.438-0.521), although the 3BFF slope between N = 250 and 500 is 0.617.
   - The Maze departure at N = 100 is not caused by the ceiling of N units: 61-75 of 100 units are active at the
     learned point and 56-65 after the first update.
3. Supplementary Table 7, fit-range row: for Maze, the range N ≥ 500 was applied after the N = 250-4,000 series had
   been measured and fitted over every size (b = 0.47; issue #201, G10, round 18).
4. Methods: Participation: from the first probe to the learned point, the N = 100 Maze count rises by 7 to 19
   units in three networks (64 → 71, 56 → 64, 56 → 75) and falls by 4 in one (65 → 61).
5. The Methods: Statistics sentence on the new networks is split in three.

Also from main: the caption of Extended Data Fig. 4b said "every network of Fig. 2a", but the panel also draws
DMTS. It now names the 3BFF and CDDM networks of Fig. 2a and the DMTS size-series networks that learned the task
and have a floor crossing.
