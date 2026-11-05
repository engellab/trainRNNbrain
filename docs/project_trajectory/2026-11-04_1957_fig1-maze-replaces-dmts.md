## 2026-11-04 19:57 — Figure 1: the maze replaces DMTS in panels c, d and e (dead-relu-paper#182)

Pavel's decision of 4 Nov (recorded on ptolmachev/dead-relu-paper#182): the maze-running task (an
agent trained by PPO to reach the cheese in a fixed maze; engellab/trainRNNbrain#3) is the main
text's third task beside the 3-bit flip-flop (3BFF) and CDDM; DMTS moves to Extended Data. This
entry records the figure change; the DMTS Extended Data figure is not part of it.

**What changed in the script** (`trainRNNbrain/experiments_and_analysis/fig_paper_F1.py`, round 14
in the module docstring). `MAIN_TASKS = [3BFF, CDDM, Maze]` is the task list panels c, d and e draw;
`PANEL_TASKS` and `ABLATION` keep DMTS because `collect()` (Della) and `fig_cosyne2027.py` read them,
and `panel_d` keeps its default `tasks=ABLATION`, so the COSYNE figure draws as before (checked: the
default call returns the three supervised tasks with one R² label). The maze is read when the
figure is drawn, from this repo and not from the collect record: `maze_rows()` reads the N = 1,000
ReLU runs with `common.maze_load_runs` and `common.maze_learned_index`; `maze_ablation()` reads the
JSONs of `rl_maze_ablate.py` and returns them in the shape `panel_d` reads (success rates under the
`r2_*` keys; the docstring says so). `SCORE` names each task's score for the tick label, the y axis
and the printed numbers. The sub-panels of d are 18.0 mm wide instead of 20.5 mm, and `D_YLABEL_MM`
(7.5 mm) sits before the maze sub-panel for its own tick labels and y label "success rate". The
"half of the units" label of panel c moved from the right end of the dashed line to the left end,
because the maze's seed-1 network (511 active) now sits under the right end.

**Panel c: active units of 1,000, every run at its own read-out point.**
- 3BFF: 277, 296, 297, 240 (mean 278; held-out R² 0.950; n = 4), unchanged.
- CDDM: 436, 420, 437 (mean 431; held-out R² 0.828; n = 3), unchanged.
- Maze: 511, 434, 410 (mean 452; n = 3) at the "learned" probe of seeds 1, 2 and 3, the first probe
  from which the clean probe's success rate stays at or above 0.9 for 5 probes (updates 1,010, 1,000
  and 1,130), the maze counterpart of the floor crossing. The probe success rate at that probe is
  0.914, 0.938 and 0.945, drawn in the tick label as "success = 0.93" (two decimals: one probe of
  128 episodes has a standard error near 0.02). The counts at the end of training (update 2,000),
  455, 423 and 404, are not drawn.

**Panel d: the score as units are removed.** The 3BFF and CDDM sub-panels are unchanged (3BFF: 733,
658, 713 inactive; held-out R² 0.950 intact, 0.375 with every inactive unit removed, 0.037 with as
many removed at random, 0.947 after pruning the inactive units and retraining. CDDM: 570, 584, 562
inactive; 0.830, 0.829, -0.033, 0.827). The maze sub-panel draws the success rate under the training
noise (`rl_maze_ablate.py`) against units removed, the same three orders, on a y axis of its own
labelled "success rate", from `data/rl_maze_ablation/N=1000_act=relu__s1.json` and `__s3.json`:
- seed 1: 542 inactive of 1,000; success 0.908 intact, 0.923 with the 542 inactive units removed,
  0.394 with 542 removed at random (mean of 5 orders), 0.034 with the 458 busiest removed (busiest
  first, first grid point past zero at 136 removed).
- seed 3: 593 inactive; 0.922 intact, 0.946 with the inactive units removed, 0.195 with as many at
  random, 0.015 at the first grid point of the busiest-first order.
- means over the two: 568 inactive (the grey bar at the top spans 56.8%); 0.915 intact, 0.935 with
  the inactive units removed, 0.294 random, 0.009 with the active units removed (busiest-first
  curve interpolated at the active count).
- The ablated weights are the final ones (update 2,000), because no earlier snapshot of a maze run
  is saved; the supervised ablations are at the floor crossing. The caption has to say so.
- Seed 2's JSON is being computed and lands in the same folder; `maze_ablation()` reads every
  `N=1000_act=relu__s*.json`, so re-running the script adds it. No prune-and-retrain has been run
  on the maze: its retrain strip reads "no data".

**Panel e: fraction of units selective.** 3BFF and CDDM unchanged (3BFF over all units 0.179, 0.245,
0.197, over active units 0.663, 0.718, 0.688; CDDM 0.543, 0.548, 0.580 and 0.986, 0.998, 0.982).
The maze column reads "no data": `data/population_stats.json` holds no maze network. The column is
kept, with its "all" and "active" ticks, so the columns of c and e stay aligned; the script draws
the maze pairs as soon as the file holds them.

**Checks.** `tests/test_paperstyle.py` on the PDF: ok (Helvetica only, letters 8.5 pt, smallest
y-label gap 1.36 mm). `tests/test_scripts_start.py`: 48 scripts start. `tests/test_no_function_imports.py`:
205 files clean. The script's own overlap check: 96 text artists, none overlapping. `pdfimages -list`:
no raster.

**Files.** `trainRNNbrain/experiments_and_analysis/fig_paper_F1.py`;
`img/internal_figures/fig_paper_F1.{pdf,svg}` (committed); the PNG preview
`img/internal_figures/fig_paper_F1.png` (not tracked; the stale five-panel preview that sat there was
replaced). Read: `data/trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu/s{1,2,3}/` (not tracked) and
`data/rl_maze_ablation/N=1000_act=relu__s{1,3}.json` (`__s3.json` was untracked when this was
written; the ablation agent owns it). Draw with
`PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/fig_paper_F1.py` from the checkout root,
with `data/fig_paper_F1_collect_v2.pkl` present; no Della access is needed.
