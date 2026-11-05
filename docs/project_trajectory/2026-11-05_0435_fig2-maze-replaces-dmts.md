## 2026-11-05 04:35 — Figure 2: the maze replaces DMTS in panels a, b and e (dead-relu-paper#182)

Pavel's decision of 4 Nov (ptolmachev/dead-relu-paper#182): the maze-running task (an agent trained
by PPO to reach the cheese in a fixed maze; engellab/trainRNNbrain#3) is the main text's third task
beside the 3-bit flip-flop (3BFF) and CDDM; DMTS moves to Extended Data. Figure 1 was changed on
4 Nov (entry `2026-11-04_1957_fig1-maze-replaces-dmts.md`, commit ef5ab33). This entry records the
same change in Figure 2 (`trainRNNbrain/experiments_and_analysis/fig_paper_F2_size.py`, round 14 in
the module docstring). The paper's text and caption are not touched here.

**What changed in the script.** `MAIN_TASKS = [3BFF, CDDM, Maze]` is the task list panels a, b and
e draw; `TASKS` keeps DMTS because `collect()` (Della) reads it and `fig_ed_readout.py` reads the
collect record. The maze is read when the figure is drawn, from this repo and not from the collect
record: `maze_series()` reads every ReLU run without a goal cue of the uniform series under
`data/trained_RNNs_rl_maze/Maze_suite/` with `common.maze_load_runs` (a run and its `_cont`
continuations are one record with absolute update indices) and reads each run at its "learned"
probe, the first probe from which the clean probe's success rate stays at or above 0.9 for 5 probes
(`common.maze_learned_index`), the maze counterpart of the floor crossing; the end of training is
the chain's last probe. A finished run with no learned probe is a cross at its end-of-training
count in a and b and enters no mean and no fit (as `fig_ed_maze.py` draws it). A run whose records
end before its budget (`maze_budget`: `trainer.n_updates` plus `trainer.iter_offset` of the chain's
last config) is unfinished and is listed, not drawn. No maze size is hard-coded: the x ticks of a
and b are the sizes any drawn series has (250 joined them), the fit range is the fitted sizes, and
panel e's maze axis runs to the longest run's end rounded up to 500 updates. A check with no maze
rows, with one maze size, and with a synthetic N = 4,000 cell plus a 4,000-update N = 2,000 chain
drew without error and without overlapping text (scratch run, not kept).

**Panel a (active units against N) and b (% of N).** 3BFF and CDDM are unchanged (3BFF b = 0.478,
1,000 active at N = 14,177; CDDM b = 0.443; every mean, SD and n as in entry 2026-10-31_1957). DMTS
is no longer drawn. The maze, one filled dot per learned run at its learned probe (update, active
units, success rate at that probe):
- N = 250: seed 1 at 1,190, 154 active (0.94); seed 3 at 830, 173 (0.93); seed 2 did not learn
  (success at the end 0.79; a cross at its end count, 128 of 250). Mean 163.5 ± 13.4 (n = 2, 65.4%).
- N = 500: seed 1 at 1,350, 281 (0.91); seed 2 at 1,280, 259 (0.93); seed 3 at 1,750, 279 (0.96).
  Mean 273.0 ± 12.2 (n = 3, 54.6%).
- N = 1,000: seed 1 at 1,050, 451 (0.95); seed 2 at 760, 517 (0.91); seed 3 at 1,750, 428 (0.94).
  Mean 465.3 ± 46.2 (n = 3, 46.5%).
- N = 2,000: seeds 1 and 2 finished 2,000 updates without learning (success at the end 0.77 and
  0.84; crosses at 464 and 556 of 2,000, 23.2% and 27.8%); seed 3 is unfinished (records end at
  update 300 of 2,000, stopped by the guard; being rerun as Spock 7383948), not drawn. The
  continuations of seeds 1 and 2 to 4,000 updates (Spock 7383947, `N=2000_act=relu_cont/`) are not
  on the Mac yet; when they land, `maze_load_runs` joins them and the script redraws N = 2,000.
- Fit over the learned-probe counts of every size with a learned run (N = 250, 500, 1,000; the maze
  suite has no N = 100 cell, so no lower cut as the supervised fit's N ≥ 500): b = 0.754, bootstrap
  95% [0.666, 0.842] (2,000 resamples of runs within each size, seed 20261031). The legend reads
  "Maze, b = 0.75" and "Maze, not learned" (the cross). Over the same runs' end-of-training counts
  (printed, not drawn): b = 0.748 [0.703, 0.793]; means 152.5 of 250 (61.0%), 260.7 of 500 (52.1%),
  430.0 of 1,000 (43.0%).
- Of the finished maze runs, the largest size with a learned run is N = 1,000 (46.5% active at the
  learned probe). The paper's sentence "fewer than a fifth of the units are active at N = 4,000"
  holds for 3BFF and CDDM (13.8%, 19.2%); the maze has no finished N = 4,000 run yet.

**Panel c (exponent at the three margins).** 3BFF and CDDM only, centred on each margin's tick; the
margins are margins over the supervised loss floor, and the maze has none. 3BFF 0.478 [0.436, 0.521],
0.509 [0.474, 0.543], 0.525 [0.485, 0.563]; CDDM 0.443 [0.432, 0.457], 0.517 [0.506, 0.528], 0.567
[0.555, 0.578] (unchanged; the leave-one-out upper ends unchanged). DMTS (0.551, 0.533, 0.608) is no
longer drawn. The maze's exponent and interval above are printed by the script for the text.

**Panel d (schematic).** Unchanged.

**Panel e (N = 1,000 over training).** The 3BFF and CDDM columns are unchanged. The maze column
draws the suite's three N = 1,000 ReLU runs on a linear axis of PPO updates (0 to 2,000): top, the
clean probe's success rate (ink lines, a dashed guide at 0.9, ticks and the label "success rate" on
the right); bottom, inactive units (N minus the active count) in the maze hue; dashed vertical lines
at the learned probes. Inactive units at the learned probe and at the end: seed 1 (learned at 1,050)
549 → 574 (+25); seed 2 (760) 483 → 551 (+68); seed 3 (1,750) 572 → 585 (+13). Every maze run
has more inactive units at the end than at its learned probe, as every supervised network with a
crossing has. The gap between panels e and f widened from 0.27 to 0.34 of the mean panel width
(`BOTTOM_WSPACE`, about 13 mm) so the maze column's right-hand label clears panel f's y label.

**Panel f.** Unchanged: 19 3BFF networks, 15 with a flat loss, 13 of those decreasing.

**Supplementary Table 1 rows** (`img/internal_figures/edt2_runs_rows.tex`, 73 rows, 12 of them maze
runs). The DMTS rows keep only their Extended Data marks (ED1, ED3a) in the Fig. column; their
"1c", "1d", "2a--c" and "2e" marks are gone, and the DMTS Extended Data figure adds its own mark
once it exists. The maze rows (`tex_row_maze`) carry the chain's last PPO update under Trained, the
learned probe's update under t*, the active count there, the probe success rate at that probe in
the R² column (in brackets at the end for a finished run that did not learn), and a note naming the
units; N = 1,000 runs are marked 1c and 2e (1d when an ablation JSON of the run exists under
`data/rl_maze_ablation/`; none did at this run). The table's header and notes live in the paper
repo and are not changed here.

**Two things to carry.** (1) N = 1,000 seed 3 meets the learned rule at update 1,750 (six probes at
0.91 to 0.98) and then falls back: its success at the end is 0.73. By the pre-registered rule
(`maze_learned_index`, the rule Figure 1 uses) it is learned and drawn at 428 active; `fig_ed_maze.py`
panel c keys on the success at the end (≥ 0.9) and draws it as a cross. The two figures disagree on
that one run until the rule is settled; the supervised read-out has a late-rise exclusion the maze
rule lacks. (2) Figure 1's docstring and entry 2026-11-04_1957 give the maze N = 1,000 counts of the
first suite (511, 434, 410, now `Maze_suite_v1_nowarmup/`); `MAZE_ROOT` of both figures now holds
the uniform series (451, 517, 428 at the learned probe), so a redraw of Figure 1 changes its panel c
and its caption numbers. Not done here.

**Checks.** `tests/test_paperstyle.py` on the PDF: ok (Helvetica, Helvetica-Bold and -Oblique only,
letters 8.5 pt, smallest y-label gap 1.36 mm). `tests/test_scripts_start.py`: 48 scripts start.
`tests/test_no_function_imports.py`: 202 files clean. The script's own overlap check: 113 text
artists, none overlapping. `pdfimages -list`: no raster. Every 3BFF and CDDM number printed equals
the printout before the change.

**Files.** `trainRNNbrain/experiments_and_analysis/fig_paper_F2_size.py`;
`img/internal_figures/fig_paper_F2_size.{pdf,svg}` and `edt2_runs_rows.tex` (committed); the PNG
preview `img/internal_figures/fig_paper_F2_size.png` (not tracked). Read:
`data/fig_paper_F2_size_collect.pkl` (Della job 16558986), `data/fig_paper_F2_size_flat.pkl` and
`data/trained_RNNs_rl_maze/Maze_suite/N={250,500,1000,2000}_act=relu/s{1,2,3}/` (not tracked).
Draw with `PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/fig_paper_F2_size.py` from the
checkout root; no Della access is needed.
