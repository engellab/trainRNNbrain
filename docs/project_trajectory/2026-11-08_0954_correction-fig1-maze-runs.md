## 2026-11-08 09:54 — Correction to 2026-11-08_0943_round19-met0.01-frozen-cause.md: Figure 1 draws the Spock runs; only its docstring numbers are Della's

The cause entry said that Figure 1's Maze panel uses the Della runs of the first uniform series. That is wrong.
- The code: `fig_paper_F1.py` reads whatever the Mac holds in `data/trained_RNNs_rl_maze/Maze_suite`.
  `maze_rows` takes every N = 1,000 ReLU run there through `common.maze_load_runs`, which joins
  continuations, at its learned probe.
- The inputs of panels (d) and (e): `data/rl_maze_ablation/N=1000_act=relu__s*.json` and
  `data/population_stats_maze.json`, computed on 5 Nov. They name `Maze_suite/N=1000_act=relu/s1-s3`, with
  learned updates 1,050, 760 and none, and 421, 451 and 420 active. Those are the Spock runs; the Della seed 3
  learned at 1,130.
- The coordinator's printout of 6 Nov (451 and 517 at the learned probe, two runs) is the Spock runs too.
- What is Della's: the `maze_rows` docstring's "511, 434 and 410 of 1,000 at the learned probe; 455, 423 and
  404 at update 2,000". Those are exactly the Della values: the Mac copy held Della's files when the docstring
  was written, before the Spock files replaced them. So the docstring is stale, and the figure's data are not.

**What changes if Figure 1 is re-run today.** `maze_rows` now returns six learned runs on the joined
4,000-update records:
- seeds 1-3 at 451, 517 and 428 (seed 3 now counts as learned, with learned point 1,750);
- the G12 seeds 4-6 at 427, 412 and 406.
Nothing in the figure was changed. Which set panel (c) draws is for the text package.
