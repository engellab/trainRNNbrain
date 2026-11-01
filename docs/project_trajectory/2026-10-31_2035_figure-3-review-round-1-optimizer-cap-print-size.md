## 2026-10-31 20:35 — Figure 3 after review round 1: the optimizer sentence, the input-weight cap, print size

Work package fig3-settings (dead-relu-paper#7, PR #26), branch `wp/fig3-settings`. Answers the
round-1 review on PR #26. No new runs: every number below is from the caches of entry
2026-10-31_2010 (`data/fig_paper_F2_fails_cache.pkl`, `data/b5_read_cache.pkl`), redrawn on the Mac
with `fig_paper_F3_settings.py` and `fig_paper_F2_fails.py` (draw mode).

**The optimizer does not raise the count only at a lower R².** The 2010 entry and Results 3 said it
did; that holds for SGD only. At 75,000 iterations on 3BFF (3 runs per arm, R² estimated by the loss
ratio), from the B5 printout of `fig_paper_F3_settings.py`, which now prints each arm's mean:

| arm | active units at 75k | mean | R² | active at 200k |
|---|---|---|---|---|
| Adam, weight decay 1e-6 (default) | 233-270 | 246 | 0.951 | 169-203 |
| Adam, weight decay 0 | 252-367 | 305 | 0.956-0.959 | 208-221 |
| AdamW, decoupled decay 1e-6 | 306-358 | 325 | 0.957-0.958 | 203-226 |
| SGD lr 0.03 (both decays) | 721-759 | 736-749 | 0.873-0.875 | 670-695 |

Adam without weight decay and AdamW have 59 and 79 more active units than the default Adam (mean
246), at an R² higher by 0.005-0.008. Removing weight decay on CDDM moves the count the same way
(+116). All three Adam arms still lose units from 75k to 200k. Results 3 now says "SGD has more
active units than Adam only at a lower R²" and gives the Adam/AdamW means.

**The input-weight cap adds about 60 units.** At 75k on CDDM the cap (every |W_inp| clipped at 0.3)
has 390-400 active (mean 395) at R² 0.827-0.832, against 332-340 (mean 335) at 0.833-0.835
untreated. Re-initialising inactive units gives 330-346 (mean 338). The cap acts on every input
weight, not only on inactive units, so the group "acting on inactive units" is renamed everywhere
(Figure 3 panels a and e, Extended Data Fig. 4 panels j-l, Results 3, Methods, captions) to
"interventions designed to raise the active count", which covers both (the 22 Sep screen was
designed that way). Results 3 states the cap's numbers.

**Print size.** At a 6.5 in text width the figures printed below 5 pt (Figure 3 at a scale of 0.80,
Extended Data Fig. 4 at 0.72). Changes:
- Figure 3 drawn 165 mm tall (was 180); row labels 4.9 → 5.3 pt, legends 5.2 → 5.5 pt; placed at
  `height=0.695\textheight`, which prints at a scale of 0.966.
- Extended Data Fig. 4 drawn 180 mm tall (was 200) with row gap 0.50 (was 0.72), so the axes keep
  their height; legends 5.0 → 5.5 pt with tighter handle spacing; rate and sparsity legend labels
  shortened to "rate", "sparsity" (the λ values are in Methods); panel l letter lowered. It prints at
  full text width, a scale of 0.926. Caption cut from 295 to 206 words (the per-panel value lists
  are in Methods M08).
- Extended Data Fig. 5 legends 5.2 → 5.5 pt (scale 0.93).
Smallest printed text: 5.1 pt in each of the three figures. The only smaller glyphs are math
superscripts (the 2 of R², exponents of 10), which matplotlib sets at 70% of their label.
Layout checks: 0 problems in all three.

**SGD colours.** The five SGD learning rates now differ in marker shape as well as shade
(0.03 ^, 0.1 >, 0.3 diamond, 1 plus, 3 cross) and the shades span CIELAB lightness 81 to 18
(was 78 to 27). Runs drawn on the bottom edge (R² below 0.86) keep the downward triangle.

**Names.** Extended Data Fig. 4 titles and panel l rows say 3BFF instead of flip-flop / FF, matching
Figure 3 and Results 3. The rest of the paper (Results 1 and 4, Methods) still says "flip-flop";
filed as a separate text issue.

The C5 wording follows option 1 of dead-relu-paper#24, relabelled from needs-pavel to decision
(PROTOCOL.md, 31 Oct).
