# Round 20 figures package (dead-relu-paper #214; committee #212)

Branch `wp/r20-figures` (code and paper). Every figure below is drawn from read-out files that already
existed; nothing was retrained or re-measured. The scripts at main 3122a4e reproduced the paper's figures
pixel for pixel before any change (render at 100 dpi, 0.000% of pixels differ, nine figures).

## What changed in each figure

- **Fig. 1** (`fig_paper_F1.py`): panel d's legend names the three removal orders as Methods and Results 1
  do ("least to most participating", "most to least participating", "random orders") and the grey bar
  "share of inactive units"; panels c and e widened from 40 to 45 mm and made taller (the left column's
  empty space); the Maze mean label (465) moved off the half-of-the-units line.
- **Fig. 1d, the 3BFF rise** (Reviewer R): checked on `data/fig_paper_F1_collect_v2.pkl`. Beyond each
  network's inactive count the removal grid is every 50 units from k = 800. Held-out R^2 of the three
  networks: 0.19-0.22 at k = 750, 0.16-0.29 at 800, 0.22-0.44 at 850, 0.27-0.31 at 900, 0.18-0.22 at 950, 0 at
  1,000. The rise is present in every network, inside a band of low R^2 (0.15-0.45); the lines between grid
  points are interpolations. The caption now says so.
- **Fig. 3** (`fig_paper_F3_tasks.py`): column titles by the Methods names (input-weight norm, initial
  spectral radius, magnitude penalty, activation function, shifted sigmoid); "no read at 75k" replaced by
  "measured at 150,000 only"; Maze row label "success rate"; canvas 122 -> 125 mm.
- **Fig. 4** (`fig_paper_F4_noreturn.py`): task colours (CDDM orange, 3BFF blue) for the traces of a, the
  active units of b, the rings of c and the 3BFF-trained networks of d and e; inactive units and fresh
  networks stay grey. Extended Data Figs. 7 and 8 unchanged.
- **Fig. 5** (`fig_paper_F5_tasks.py`): printed at 100% (was 0.89, text 5.3 pt); 6 mm gap between a and b;
  b's Maze row draws each value's success rate with the 0.9 line (every network below 0.9 has no learned
  point, so it is a cross under the line; checked: in every Maze cell the crosses are exactly the networks
  below 0.9); count labels moved off the tolerance line; panel c 31 x 36 mm per sub-column (was 26 x 29)
  with task colours and the technique shapes of a, b and d, markers spread by a factor 1.07 along N; value
  labels `10` (m = 10), `smaller factor`, `fewer units` (the names of M09 and Supplementary Table 2 after
  dead-relu-paper PR #215); d's axis "k least participating active units removed". Panel d keeps 30 mm:
  at 32 mm and more a see-through pair of 3BFF markers fails rule 1 of `tests/scan_open_markers.py`.
- **Extended Data Fig. 2** (`fig_ed_maze.py`): the 0.9 level named by its tick label; axis "noise-free
  probe"; removal orders named as in Fig. 1d.
- **Extended Data Fig. 3** (`fig_ed_population.py`): the Maze networks added (below); drawn at print size,
  165.1 x 126 mm (it was 112 mm tall and shrunk to 0.78 by the float, text 4.7 pt); panel a's overlapping
  open markers grouped as in Figs. 3 and 5 (the scan failed there before: 116 rule failures at main).
- **Extended Data Fig. 5** (`fig_paper_F2_fails.py`): panel titles "activity noise", "magnitude penalty",
  "input-weight norm".
- **Extended Data Fig. 6** (`fig_ed_maze_settings.py`): 0.9 named by tick labels in b, d, f, i; g and i draw
  each network's success rate with the 0.9 line (as Fig. 5b); rescaling variants named as in Fig. 5;
  canvas 216 -> 202 mm so that figure and caption fit one page.
- **Extended Data Fig. 10** (`fig_paper_F5_drive.py`, ed_techniques_3bff): axis "k least participating
  active units removed".

## Maze population statistics (Extended Data Fig. 3)

`data/population_stats_maze.json` (rl_maze_population_stats.py, 5 Nov) holds seeds 1-3 of the default
N = 1,000 Maze networks at update 2,000; seeds 1 and 2 learned the task, seed 3 had not (the networks of
Fig. 1e). It is current for Fig. 1e and was not re-run; the figure draws the two learned networks, as Fig. 1e
does. Panel a uses `data/round18/g13_maze_weight_spread.json` (entry 2026-11-07_0235, G13) for the same two.

| Statistic | Maze seed 1 | Maze seed 2 | 3BFF (3 networks) | CDDM (3 networks) |
|---|---|---|---|---|
| active units | 421 | 451 | 270-340 | 416-438 |
| dimensionality of normalised rates, all / active | 46.2 / 20.0 | 42.7 / 19.5 | 23.0-27.2 / 12.5-13.5 | 18.4-19.5 / 11.1-11.4 |
| selective units, all / active (inactive selective) | 154 / 154 (0) | 200 / 199 (1) | 179-245 / 179-244 (0-1) | 543-580 / 415-430 (119-150) |
| fraction selective, all / active | 0.15 / 0.37 | 0.20 / 0.44 | 0.18-0.25 / 0.66-0.72 | 0.54-0.58 / 0.98-1.00 |
| k-means classes, all / active | 3 / 3 | 2 / 2 | 4 / 3 | 4 / 3 |
| s.d. of recurrent weights, all / active | 0.041 / 0.055 | 0.042 / 0.055 | (other networks) | (other networks) |

Reading against the text: the Maze networks behave like the 3BFF networks. The fraction of selective units
is lower over all units (0.15-0.20 against 0.37-0.44), but almost no inactive Maze unit passes the
selectivity criterion (0 and 1), so the inflated count of selective units that the Discussion's closing
recommendation cites (119 to 150 inactive selective units per network) is a CDDM result (and DMTS, 14-99), not
a three-task one. The recommendation to compute the fraction of selective units over the active units still
holds on all three tasks, because of the denominator. Normalised-rate dimensionality over all units is 2.2-2.3
times its value over the active units in Maze networks (3BFF about 2, CDDM 1.7). The number of classes is the
same over both sets in Maze networks, as in DMTS.

## Checks

- `tests/scan_open_markers.py`: PASS (0 rule failures) for fig_paper_F1, fig_paper_F3_tasks, fig_paper_F5_tasks,
  fig_ed_maze, fig_ed_population, fig_paper_F2_fails, fig_ed_maze_settings, fig_paper_F5_drive.
  `tests/scan_marker_edges.py`: PASS for the same. `tests/test_paperstyle.py`: ok for all nine PDFs (Helvetica,
  scale 1.0, median 6.0 pt). `tests/test_no_function_imports.py`: ok.
- An independent text-overlap check (round-20 scratch tool: text-text, text over marker centres, text crossed
  by guide lines, text over another axes, text outside the canvas, on the tight bboxes of the finished
  figure): 0 problems in Figs. 1, 3, 4, 5 and Extended Data Fig. 3. Remaining flags, all present at main
  and checked on renders: guide lines that run under opaque legend frames (Extended Data Fig. 5) or under the
  masked word "stopped" (Extended Data Fig. 6), count labels on row grid lines (Extended Data Fig. 10), the
  identity-line label "every unit active" and neighbouring y tick labels of the panel-b sub-axes (Extended
  Data Fig. 2).
