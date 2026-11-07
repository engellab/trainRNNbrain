## 2026-11-07 — Fig. 5d shows two different removals (dead-relu-paper PR #197 second review, #199)

No network was trained and no number changed. Checked against the code and the read-out files, after the
second review of PR #197 reported that the Maze sub-panel of Fig. 5d does not do what its label said.

**3BFF** (`a3_remedy_ablation.py`, `ablate_net`; `data/remedy_reads_31oct/a3_N1000.json`). The vertical axis
is held-out R² after the k added units are removed: `act_order[:k]`, the k least participating active units.
The inactive units stay in the network. The horizontal axis is k active units removed at random
(`rng.choice(act_idx, size=k)`, median of 5). k runs from 94 to 726 by network.

**Maze** (`fig_paper_F5_tasks.maze_removal`; `data/rl_maze_ablation_interventions`). The vertical axis is
the success rate after `k_total = n_inactive + k_added` units are removed from the least participating up,
so the inactive units go first and then the k added units. With the control mean of 430, k_total is 570 in
all five networks (recomputed from the JSONs). The horizontal axis is 570 units removed at random from all
1,000, median over five orders. It is not k active units.

**So they are not one test on two tasks.** The caption, Results 5, M09 and the axis labels now say which
units each sub-panel removes.

**Does the Maze value still bear on the added units?** Yes. Removing the inactive units alone (18 to 147 units)
leaves the success rate within 0.016 of intact in all five networks: 0.936 to 0.932, 0.889 to 0.902, 0.943 to
0.942, 0.842 to 0.827, 0.956 to 0.968. The fall to 0.53-0.76 comes with the k added units.

**Would 3BFF read the same under the Maze removal?** The file holds that value as `r2_keep_ctrl`, every unit
but the n_ctrl most participating removed. For the rate penalty, rate + sparsity and read-out dropout it is
within 0.05 of the k-only value, and for two of three prune-and-duplicate networks within 0.003 (the third has
a k-only R² of -4,902 and 0.273 here). For noisy synapses it is 0.39-0.44 against 0.57-0.60, and for rescaling
0.29-0.43 against 0.31-0.70. The conclusion that performance depends on the added units does not change: these
values are still below the intact 0.93-0.95. Fig. 5d keeps the k-only values for 3BFF.

**Figure.** `fig_paper_F5_tasks.py`: one label per axis, the Maze random label taking 570 from the data. Panel d
moved 4 mm left and its gap widened to 8 mm, so each sub-panel keeps 22 mm; at 20 mm three open markers failed
`tests/scan_open_markers.py`. Checks: `tests/scan_open_markers.py fig_paper_F5_tasks` PASS, `tests/test_paperstyle.py`
ok, no overlapping text, and the printout is unchanged apart from the panel d header.
