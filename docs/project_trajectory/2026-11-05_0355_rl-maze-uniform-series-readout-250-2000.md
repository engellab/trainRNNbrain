## 2026-11-05 03:55 — Maze uniform series (warm-up, one protocol) read out for N = 250 to 2,000: the share of active units falls as N^-0.24; N = 2,000 needs a longer budget (issue #3)

Follows `2026-11-04_2028_rl-maze-warmup-uniform-series-spock-della.md`. Spock arrays 7383494 and
7383495 (commit 9dce02e): 19 of 20 runs completed; N = 2,000 seed 3 was stopped by the rollback
guard at update 341 (three non-finite rollouts in a row), so the full-rate instability at
N = 2,000 is rarer with the warm-up, not gone. The runs are on the Mac under
`data/trained_RNNs_rl_maze/Maze_suite/`; the first suite moved to `Maze_suite_v1_nowarmup/` and
its ablations to `data/rl_maze_ablation_v1/`. Read-out `rl_maze_readout.py --name suite`,
figures `img/internal_figures/rl_maze_suite*.png` and `fig_ed_maze.pdf` (commits fcd9852,
0ea1ac7). Read-out points as pre-registered: "learned" = probe success at or above 0.9 for 5
probes; "end" = update 2,000; counts by the participation rule (modulation count in brackets).

| N | act | seed | probe success, end | learned at | active, learned | active, end |
|---|---|---|---|---|---|---|
| 250 | ReLU | 1, 2, 3 | 1.00, 0.79, 1.00 | 1,190, never, 830 | 154, -, 173 | 147, 128, 158 (157, 139, 172) |
| 500 | ReLU | 1, 2, 3 | 1.00, 0.98, 0.95 | 1,350, 1,280, 1,750 | 281, 259, 279 | 265, 241, 276 (297, 276, 309) |
| 1,000 | ReLU | 1, 2, 3 | 0.99, 0.98, 0.73 | 1,050, 760, never | 451, 517, - | 426, 449, 415 (529, 554, 497) |
| 2,000 | ReLU | 1, 2, 3 | 0.77, 0.84, stopped | never | - | 464, 556, (605 at 300) |
| 1,000 | leaky ReLU | 1, 2 | 0.98, 1.00 | 730, 1,120 | 631, 639 | 603, 600 (676, 693) |
| 1,000 | softplus | 1, 2 | 0.98, 1.00 | 1,020, 630 | 452, 510 | 437, 420 (491, 468) |
| 1,000 | tanh | 1, 2 | 0.00, 0.38 | never | | 1,000, 1,000 |
| 1,000 | shifted sigmoid | 1, 2 | 0.21, 0.15 | never | | 314, 322 |

**Against the pre-registered claims.** (1) Inactive units on a solved task: every learned run
ends with 92 to 103 of 250, 224 to 259 of 500 and 551 to 574 of 1,000 units inactive. (2) The
share falls with size: seed means at the end over the learned runs 152 of 250 (61%), 261 of 500
(52%), 438 of 1,000 (44%); the two finished N = 2,000 networks, not learned by the rule, have 464
and 556 of 2,000 (23% and 28%); fit over the learned sizes, count ~ N^0.76 (bootstrap 95% 0.72 to
0.81), so the share falls as N^-0.24; steeper than the first suite's 0.69. (3) Worse with
training: the end count is below the learned count in 7 of 7 learned ReLU runs (by 3 to 68
units) and in all four learned leaky-ReLU and softplus runs. (4) Activations: tanh and the
shifted sigmoid do not learn within 2,000 updates (both move this time: 0.38 and 0.21 at best);
leaky ReLU keeps 600 to 603 of 1,000, ReLU 426 to 449, softplus 420 to 437.

**What the warm-up changed.** Learning is slower at every size (learned at 760 to 1,750 updates
against 580 to 1,130 in the first suite), and N = 2,000 does not reach the criterion within 2,000
updates (0.77 and 0.84, rising at the end). The size series needs a longer budget above
N = 1,000; the trainer has no resume path, so one is being added (issue #4), after which the
N = 2,000 and 4,000 runs continue from their end states to 4,000 updates. N = 4,000 (Della
16883344) is at about update 400.

**Ablations** of the uniform networks run on the Mac into `data/rl_maze_ablation/`.
