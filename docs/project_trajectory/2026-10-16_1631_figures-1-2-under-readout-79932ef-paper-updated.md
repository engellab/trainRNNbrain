## 2026-10-16 16:31 — Figures 1 and 2 under read-out 79932ef; paper numbers and Methods updated

Rebuilt after 9d's read-out change (79932ef: end-weighted fits on the flip-flop, DMTS at its late
level, 24-start fits, no read-out for a loss ending below 0.98x its floor). Della job F12new
(15703377) built Figure 1 and collected Figure 2, which was drawn on the Mac. Both are committed as
13f3fa7 and synced to the paper (25f6ff9).

**Figure 1, panel c at 1.05×** (mean ± SD):
- flip-flop: N = 500 205.0 ± 18.4 (6 of 6 runs; 2 regained a read-out), 1000 290.0 ± 11.3,
  2000 390.0 ± 22.9, 4000 551.7 ± 41.7. Exponent N^0.47; 1000 active units at N ≈ 14,300.
- CDDM: unchanged (N^0.44).
- DMTS: N = 500 191.0 ± 17.3, 1000 320.0 ± 144.2 (2 of 3 runs), 2000 425.5 ± 55.9. Exponent
  N^0.58 (was 0.43), read later: mean crossings 47-63k.

Every exponent stays between 0.44 and 0.60 at margins of 1.05×, 1.10× and 1.20×.

**Panel e:** 90, 154 and 31 more units become inactive after the floor. The DMTS count was 194; its
crossing now falls at 81k instead of 36k.

**Figure 2 changes** (CDDM rungs otherwise unchanged):
- Flip-flop input-scale floors are now 268-297, 370-411, 441-526, 560-663 and 615-920.
- Flip-flop revival rungs: untreated 277-297, treated 210-276.
- CDDM input scale 0.5 gains a second read-out.
- The CDDM sparsity and rate + sparsity rungs lose all floor read-outs (0 of 5 each); their
  end-of-training counts are unchanged.
- Figure 2 is relabelled "inactive", and its flip-flop R² ticks now sit inside the axis range; an
  edge tick had run into the neighbouring "2k" label.

**Paper:**
- Captions, intro, Results (02, including the N = 1000 table) and the knob section now carry the
  new numbers.
- The Methods read-out paragraph is rewritten to the current rule, and "first reaches" is replaced
  throughout.
