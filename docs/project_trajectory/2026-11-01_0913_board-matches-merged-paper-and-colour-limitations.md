## 2026-11-01 09:13 — Status board brought in line with the merged round-2 paper; colour-pair limitations stated in the style sheet

Package board-style-r3 (dead-relu-paper#48 and #63), branch `wp/board-style-r3`. Checked against
dead-relu-paper `main` 46622b2 (every round-2 package merged, frame-discussion PR #73 included).

**Board (`docs/paper_audit.md`).** Every claim row, its status and what remains, read against
Results 1-5, the five main captions, the Extended Data and Supplementary captions and the Discussion.
- Figures paragraph: Fig. 3e is the two revival methods; Fig. 4 is listed panel by panel (a-c
  gradient reach, d tanh, e-f transfer test); Fig. 5a is active units over training at N = 1,000.
- "What remains" names no merged package. Two items stay open: C4, Results 4's tanh sentence and
  its panel citation (dead-relu-paper#70, owner #58); C7, the combination test's floor R² of
  0.889 rests on one network and the paper does not say so (#27).
- C1: the removal figure at every size has 35 networks (3BFF 15 at 75,000 iterations, CDDM 12 and
  DMTS 8 at the end of training; DMTS s597040509 never learned and is left out), from entry
  2026-11-01_0716. The CDDM random-removal retrain is "within 0.01" of the inactive-removed one, as
  in Results 1, not "costs nothing".
- C2, C3: mean crossings stated as 59.0k-65.0k and 26.7k-32.5k (Results 2's source comment); the
  N = 2,000 flat-loss result is one network, which Results 2's "at each of the four sizes" rests on;
  the optimizer comparison is now a panel of the Extended Data figure of every standard-setting sweep.
- C4: Extended Data figures cited by content; the paper's wording of the silent criterion (decision
  #59) recorded.
- C5: the CDDM rate-penalty R² separated: 0.828-0.830 alone against 0.831-0.834 without
  (−0.004, the abstract's and Results 5's number), −0.008 with the sparsity penalty (entries
  2026-11-01_0735, _0813). The board had pooled the two as 0.822-0.830.
- C6: rescaling and read-out dropout at 75,000 iterations are 1.7-2.0× at N ≥ 2,000 and
  1.55-2.05× across N = 500-4,000 (Results 5's "1.5- to 2.1-fold"); burst-check counts added.
- C7, C8, catalogue, open work and open questions: round-2 outcomes recorded; the C4-wording
  question closed; the A5 task-demand panel is not in the paper.

`grep -n "36 networks\|several\|stays excluded\|waits on Pavel\|ED Figs"` on the board returns
only lines that quote a decision as history.

**Style sheet (`paperstyle.py` docstring; paper `docs/figure_style.md`).** No colour constant
changed, so no figure needs a redraw. Correction to `2026-10-31_2208_shared-figure-style-sheet.md`:
3BFF blue and sparsity violet (2.9) collapse under *protanopia*, not deuteranopia (11.1 under
deuteranopia); DMTS-rescale 2.7 is deuteranopia, as written.

Every failing pair now carries a stated limitation. Values from the dataviz validator
(`validate_palette.js --pairs all`, white surface) and, independently, from a short script
(Machado 2009 severity-1 matrices in linear sRGB, OKLab distance × 100) that shares no code with it;
the two agree to 0.1 on every pair.

| pair | colour-blind ΔE | normal ΔE | limitation |
|---|---|---|---|
| SETTING (ramp step 2) - rescale | 3.7 (protan) | 26.5 | teals never share a panel with rescale, duplicate or synaptic noise |
| ramp step 1 - rescale | 6.4 | 26.6 | same |
| ramp step 3 - duplicate | 5.2 | 18.5 | same |
| SETTING - synaptic noise | 11.6 | 13.7 | same |
| BASELINE - INACTIVE | 8.6 | 8.6 | never in one panel |
| ACTIVE - ramp step 3 | 11.2 | 12.4 | never in one panel |
| BASELINE - ramp step 0 | 4.4 | 8.3 | the default's markers get an ink edge and a legend entry |

The last three rows and the ramp-step 1 and 3 pairs were not in the PR #62 review; they came from
running every ramp step against every rule and grey. BASELINE has a contrast of 2.35:1 on white, so
a baseline series is always named directly; SYNAPTIC_NOISE (chroma 0.089) and SETTING (0.07) are
below the chroma floor of 0.10, so neither is the only cue to identity. Checked against the figures
on paper main by reading which colours each figure script draws in each panel: no panel holds a
teal with rescale, duplicate or synaptic noise, or BASELINE with INACTIVE (Fig. 4 uses INACTIVE in
a-c and BASELINE in d and f; Fig. 5 uses BASELINE in a and d and INACTIVE only as the hollow outline
of an absent rule in f's dot matrix). On a render of the Extended Data figure of every
standard-setting sweep, the 5-value sweeps (metabolic penalty, input-weight scale) put BASELINE
beside ramp step 0, with ink-edged default markers and a legend.

Minor points of #63: the mathtext fallback is named as STIX (`pdffonts` lists STIXGeneral-Regular,
checked on a rendered $\propto$); `tests/test_paperstyle.py` prints `FAIL <path>: no such file`
for a missing PDF instead of a traceback; `demo_figure` draws the three tasks on three power laws
(exponents 0.48, 0.44, 0.55) instead of one line.

Tests: `tests/test_paperstyle.py` ok (Helvetica only, letters 8.5 pt);
`tests/test_no_function_imports.py` ok (176 files); `tests/test_scripts_start.py` ok (41 scripts).
