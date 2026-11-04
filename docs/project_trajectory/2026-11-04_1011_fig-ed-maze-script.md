## 2026-11-04 10:11 — Extended Data figure script for the maze task, fig_ed_maze.py (issue #3)

**What it is.** `trainRNNbrain/experiments_and_analysis/fig_ed_maze.py` draws the maze's Extended
Data figure (paper package ptolmachev/dead-relu-paper#182): Figure 1 has no room for a fourth
ablation sub-panel, so the maze gets a figure of its own. Five panels, 165.1 x 107 mm, bold letters:
(a) the clean probe's success rate against PPO updates, one thin line per ReLU run, a filled mark at
the run's "learned" probe; (b) active units against PPO updates, one sub-panel per size with y from 0
to that N; (c) active units at "learned" (filled) and at the end of training (open) against N, the
seed mean as a bar, a power law through the size means of the end counts with its exponent b in the
legend, as Results 2 reports the supervised series; (d) the ablation of `rl_maze_ablate.py`,
success rate under the training noise against the share of units removed, quietest first, busiest
first and random order (band: range), a line at each network's inactive count with "count of N"
above it; (e) the activations at N = 1,000, the participation-rule count (circles) and the
modulation count (squares) at the end of training, the mean success rate under each name.

**Shared code moved to `common.py`.** `learned_index` and its two constants left
`rl_maze_readout.py` for `common.maze_learned_index`, `MAZE_LEARNED_SUCCESS` (0.9) and
`MAZE_LEARNED_HOLD` (5 probes), with a self-check line. `load_runs` moved too, as
`common.maze_load_runs`: the figure script and the read-out script read the same run-folder layout,
and a second copy in the figure script would be the divergent loader the README's rule 1 warns
against; rule 2 (keep your own loader) covers a layout only one script reads. `rl_maze_readout.py`
imports both from `common` and prints the same table as before on `Maze_second`.

**Choices, and why.** N is a lightness ramp of the maze hue (`N_SHADE`: #d29ebd, #bf74a1, #a94681,
#941862 for 250, 500, 1,000, 2,000), as Figure 2f shades the 3BFF sizes; `plotstyle.col_n` was not
used because its tab10 blue and green are the 3BFF and DMTS hues and it has no entry for N = 250.
Marker fill follows the style sheet (dead-relu-paper#83), not the brief's assignment: open = read at
the end of the fixed budget, filled = the run's own convergence point ("learned", the maze
counterpart of the floor crossing); the two counting rules of panel e differ by shape, both open.
Panel b has one sub-panel per size because on a shared 0 to 2,000 axis the N = 250 curve would sit
in the bottom eighth, where a change of 20 units is 0.3 mm. A run whose end success is below 0.9 is
drawn as a cross in panel c and enters no mean and no fit (the pre-registration excludes a size that
does not learn); every size with a learned run is fitted, since the suite has no N = 100 cell. The
legend of panel d sits in a strip above it, off the data, because a network with more than half its
units inactive runs its quietest-first curve at success 1 across the top of the axes.

**Degrades cell by cell.** `--root` defaults to `data/trained_RNNs_rl_maze/Maze_suite` (empty on
the Mac until the Della jobs finish), `--ablation` to `data/rl_maze_ablation/`. Every panel draws
what exists; the script prints the missing suite cells, one row per run (N, activation, seed,
learned update, active at learned, active at end, modulation count at end, success at end), the fit
with its bootstrap 95% interval, the ablation numbers and the text-overlap check. Runs with
`env.goal_cue` true are dropped unless `--keep-cue`.

**The stand-in render** (`Maze_second`, the four walls-only ReLU runs; one ablation JSON of the cue
network `N=500_cue=true__s2` from the scratch folder):
`img/internal_figures/fig_ed_maze.{pdf,svg}`, preview `img/internal_figures/fig_ed_maze_standin_preview.png`.
N = 1,000 learned at updates 460 and 440 with 620 and 601 units active there, 620 and 613 of 1,000
at the end (update 1,000), success 1.00; the two N = 500 runs never learned (success 0.22 and 0.76
at the end, 200 and 386 active) and are crosses in panel c, so there is no fit yet. The ablated
network has 93 of 500 units inactive; with those removed the success rate is 1.00 quietest first,
0.84 in random order and 0.33 busiest first (intact 1.00). Panel e shows the ReLU pair only; the
four other activations read "no runs". Checks: `tests/test_paperstyle.py` on the PDF (Helvetica
only, letters 8.5 pt, no text within 0.3 mm of the page edge), `pdfimages -list` empty, the
script's own overlap pass clean (90 text artists), `test_scripts_start.py` 48 scripts,
`test_no_function_imports.py` 204 files. The panel a legend (lower right) crosses the stand-in's
unlearned N = 500 curve at success 0.2; a learned run sits at 1, above it.
