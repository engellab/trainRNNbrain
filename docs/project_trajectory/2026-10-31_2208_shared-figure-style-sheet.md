## 2026-10-31 22:08 — Shared figure style sheet in paperstyle.py (dead-relu-paper #40, #15)

**Why.** The committee (dead-relu-paper #40, #15) found the manuscript figures used two typefaces
(Helvetica labels, DejaVu Sans maths), regular-weight panel letters, three letter sizes, task hues
reused for rules and unit classes, mixed label case and mixed number formats. Round 2 redraws every
figure; this entry records the style they redraw to. Branch `wp/figure-style`.

**What changed in `trainRNNbrain/experiments_and_analysis/paperstyle.py`.**
- Bold panel letters: matplotlib reads only face 0 of macOS `Helvetica.ttc`, so a bold request
  returned the regular face. `register_fonts()` extracts faces 0-3 (regular, bold, oblique,
  bold-oblique) with fontTools into `matplotlib.get_cachedir()/paperstyle_fonts/` and registers them.
  Mathtext uses a custom font set of the same family, so DejaVu is no longer embedded.
- Print width: `main.tex` is article 11 pt, letter, 1 in margins; `\the\textwidth` = 469.75 pt =
  165.1 mm, `\textheight` = 228.6 mm. Figures drawn at `W_FULL_MM` = 165.1 mm print at 100%; the old
  180 mm canvas prints at 92%. `save()` writes the exact canvas (no tight crop) for a print-width
  figure and warns if anything overflows it; other widths keep the old tight crop.
- Sizes: labels 6.5 pt, ticks 6 pt, panel letters 8.5 pt bold, base text at least 5 pt; mathtext
  scripts are 70% of their base (4.2 pt at tick size), the floor `MIN_SCRIPT_PT`.
- Palettes by meaning: tasks 3BFF `#2a78d6`, CDDM `#b35806`, DMTS `#12916a`, reserved; rules in
  three hue families (penalties violet, noise ochre, prune-and-reinitialise magenta); settings teal
  with an ordered ramp; `BASELINE` grey, `ACTIVE` near-black, `INACTIVE` light grey; `BAD` = ink.
  Legacy names (`SLOTS`, `COND_COL`, `BASE`, `W1`, `W2`) still exist; `COND_COL` now maps to the new
  rule colours.
- Numbers: counts and iterations as "1,000", "75,000" (`fmt_count`, `count_ticks`), matching the
  prose ("75{,}000 iterations"); no "1k" ticks. Labels lower case (`LABELS`).

**Palette checks** (dataviz `validate_palette.js --pairs all`, white surface; colour-blind ΔE is
OKLab ΔE × 100 under simulated protan/deutan vision, target 8; normal-vision floor 15):
tasks 9.7 / 20.2; Figure 5 rule set (baseline, active, rate, rate + sparsity, dropout, synaptic
noise, duplicate, rescale) 8.0 / 15.5; Figure 3 set (baseline, active, three penalties, setting ramp
steps 1-2) 8.9 / 15.9. Across tasks and rules two pairs collapse under deuteranopia (DMTS-rescale
2.7, 3BFF-sparsity 2.9) while 19-29 apart in normal vision; the rule is that tasks and rules never
share colour coding in one panel. Reserving three task hues leaves too little of the colour wheel
for eight colour-blind-separable rule colours in one set; the family-tint scheme is the trade-off.

**Test.** `tests/test_paperstyle.py` renders `paperstyle.demo_figure()` and reads the PDF back with
`pdffonts` and PyMuPDF. Pass rule, fixed before the run: Helvetica only, a Helvetica-Bold face,
panel letters bold 8.5 ± 0.05 pt, labels 6.5 pt, ticks 6.0 pt, page 165.1 ± 0.1 mm. Result: pass
(`pdffonts`: Helvetica, Helvetica-Bold, Helvetica-Oblique). With PDF arguments it audits figures at
their printed size. On the round-1 figures in the paper repo (F1, F2_size, F5_drive, supp_ablation)
it fails as expected: DejaVu present, no bold face (supp_ablation: DejaVu only).

**Existing scripts re-run with the new module** (local caches, worktree output, not committed): all
manuscript figure scripts run; `fig_paper_F2.py` stops on its own label-collision assert, identically
on unchanged `main` (not used by `main.tex`). Every re-rendered figure now embeds Helvetica with a
bold face and no DejaVu. Still failing the audit until the round-2 redraw: panel letters print at
7.8 to 10.8 pt because no canvas is yet drawn at 165.1 mm (LaTeX scales them by 0.92 to 1.27 at
\textwidth), and STIX glyphs for →/∼ remain in fig_paper_F2_size, fig_paper_F3 and fig_supp_tasks.

Paper-side guide: dead-relu-paper `docs/figure_style.md`.
