# Figure 2, the Extended Data maze figure and Supplementary Table 1 redrawn with N = 4,000 and the N = 2,000 chains (dead-relu-paper#182)

**Date:** 2026-11-06 1829
**Issue:** ptolmachev/dead-relu-paper#182 (the maze package), engellab/trainRNNbrain#3 (the size series)
**Scripts:** fig_paper_F2_size.py (drawing stage, Mac), fig_ed_maze.py, fig_supp_tasks.py, all run on
the suite as it is on the Mac tonight (data/trained_RNNs_rl_maze/Maze_suite; the N = 4,000 chains of
entry 2026-11-06_1723). Checks: tests/test_paperstyle.py on the three PDFs (ok: Helvetica only,
letters 8.5 pt), the scripts' own overlap checks (none), `pdfimages -list` (no raster image),
tests/test_no_function_imports.py (211 files clean), tests/test_scripts_start.py (58 scripts start).

## Figure 2 (fig_paper_F2_size.py; printout equals the numbers of entry 2026-11-06_1723)

The maze series of panels a, b and e now has N = 250 (four seeds), 500, 1,000, 2,000 (the 2,000 +
2,000 chains) and 4,000. Panels a and b draw each learned run at its learned point (filled) and each
finished run that did not learn as a cross at its end-of-training count (N = 250 seed 2, 128; N =
1,000 seed 3, 415; N = 4,000 seeds 2 and 3, 516 and 693). Learned-point means: 155.0 +- 17.5 (N =
250, n = 3, 62.0%), 273.0 +- 12.2 (500, n = 3, 54.6%), 484.0 +- 46.7 (1,000, n = 2, 48.4%), 454.7 +-
89.2 (2,000, n = 3, 22.7%), 619 (4,000, n = 1, 15.5%). The fit drawn in panel a, over the learned-point
means of every size, is b = 0.473 (bootstrap 95% 0.436 to 0.512); the legend reads "Maze, b = 0.47".
Over the end-of-training counts of the same runs (printed, not drawn): b = 0.516 (0.486 to 0.548),
means 145.7 (58.3%), 260.7 (52.1%), 437.5 (43.8%), 431.3 (21.6%), 676 (16.9%). 3BFF and CDDM
unchanged (0.478 and 0.443; every mean as in entry 2026-10-31_1957). Panel e's maze column is the
three N = 1,000 runs as before (inactive at the learned point -> end: 549 -> 574, 483 -> 551; seed 3
not learned, 585 at the end).

Learned point against the end over the 12 learned ReLU runs (N, seed: learned -> end): 250, 1: 154 ->
147; 250, 3: 173 -> 158; 250, 4: 138 -> 132; 500, 1: 281 -> 265; 500, 2: 259 -> 241; 500, 3: 279 ->
276; 1,000, 1: 451 -> 426; 1,000, 2: 517 -> 449; 2,000, 1: 393 -> 400; 2,000, 2: 557 -> 506; 2,000,
3: 414 -> 388; 4,000, 1: 619 -> 676. Ten of twelve end below their learned count, by 3 to 68 units
(1.1% to 13.2%); N = 2,000 seed 1 ends 7 above (1.8%) and N = 4,000 seed 1 ends 57 above (9.2%).

## Supplementary Table 1 rows (img/internal_figures/edt2_runs_rows.tex, 77 rows, 16 maze runs)

`tex_row_maze` now writes the end of training into the Note column: a learned run reads "PPO; at the
end 147 active, success 1.00", a finished run that did not learn "PPO; not learned; at the end 128
active" (its end success stays in the R^2 column, in parentheses). Trained is the chain's last update
(2,000, or 4,000 for N = 2,000 and 4,000), t* the learned point, Active the count there, R^2 the
success rate there. The N = 4,000 rows: seed 1 learned at 2,860 (619; 0.95), 676 active at the end
(0.96); seeds 2 and 3 not learned, 516 (0.84) and 693 (0.80) at the end. The DMTS rows carry only their
Extended Data marks (the script never writes the ED2 mark of the population figure; the paper's table
adds it by hand to the three DMTS population networks).

## Extended Data maze figure (fig_ed_maze.py)

Three changes so that the figure shows what its fit uses. (1) SIZES gains 4,000: panel b has a
fifth sub-panel (14.5 mm wide, every 21.6 mm; the first at 57 mm) and panel c's axis runs to 5,600,
so the N = 4,000 markers (619 filled, 676 open, crosses at 516 and 693) are drawn; before, the axis
ended at 2,800 and the fit's N = 4,000 mean was invisible. (2) Panel a's key moved to its own axes
below the panel (8, 49.5, 46 x 6 mm, three columns): inside the axes it sat on the N = 2,000 and
4,000 traces. (3) The "every unit active" label is anchored at the lower-left end of the diagonal
(at the middle its box met the legend's; near the largest size it ran into panel d's y label).
ACT_SEEDS is (1, 2, 3) for the expected-cells check. Panel c: b = 0.516, bootstrap 95% 0.484 to
0.547 (seed 20261104), means 146 (n = 3), 261 (3), 438 (2), 431 (3), 676 (1). Panel e, third seeds:
leaky ReLU 610 (success 1.00), softplus 393 (0.91 at the end, but no learned point: its success never
stayed at or above 0.9 for five probes), tanh 999 (0.02), shifted sigmoid 295 (0.15); per-function
mean success 0.90, 0.99, 0.97, 0.13, 0.17.

## Supplementary Fig. 1 row d (fig_supp_tasks.py)

Redrawn from the uniform series' N = 1,000 seed 1 (MAZE_RUN = Maze_suite/N=1000_act=relu, s1; the
committed row of fce0216 was the first suite's seed 1): one clean episode of median length among
the probe's episodes that reach the cheese, 165 decisions (82 tau), 10 inputs and 2 outputs. The
caption of floats/ED1_tasks.tex describes the row as drawn.

**Files:** trainRNNbrain/experiments_and_analysis/fig_ed_maze.py, fig_paper_F2_size.py;
img/internal_figures/fig_paper_F2_size.{pdf,svg}, fig_ed_maze.{pdf,svg}, fig_supp_tasks.{pdf,svg},
edt2_runs_rows.tex.
