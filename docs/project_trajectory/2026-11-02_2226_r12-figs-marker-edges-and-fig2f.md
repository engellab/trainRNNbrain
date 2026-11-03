## 2026-11-02 22:26 — r12-figs: a thin ink edge on every marker, and Fig. 2f drawn from the raw quantities of the flat-loss test (Pavel's items 54 and 55)

Branch `wp/r12-figs` (from main 627dc1e; main 2eb3c2f merged in as 0b8d460), package r12-figs of
round 12 (brief `~/Documents/Projects/paper_push/ROUND12_BRIEF.md`). Code commit b12e992; the
figures and this entry are in the commit that adds this file. No number in the paper changes.

### Item 54: every marker has a thin black edge

Pavel asked for a thin black edge round every scatter marker, and the round-7 pattern adds that no
other meaning may be carried by a marker edge.

**Encoding.** `paperstyle.MARKER_EW` = 0.35 pt, in ink (`INK`).
- A filled marker (value at the floor crossing) is its series colour inside the ink edge.
  `setup()` makes the ink edge the default of `ax.plot`, `ax.errorbar` and `ax.scatter` markers
  (rcParams `lines.markeredgecolor`, `lines.markeredgewidth`, `scatter.edgecolors`); scatter's edge
  width is not an rcParam, so scripts pass `linewidths=ps.MARKER_EW` or use `mark_floor(col,
  scatter=True)`.
- An open marker (value at a fixed iteration) is a white inside, a ring of its series colour
  (`OPEN_MARKER_EW` = 0.7 pt) and an ink rim of 0.35 pt round the ring. The rim is a path-effect
  stroke (`paperstyle.open_rim`, used by `mark_fixed`): an ink stroke 1.4 pt wide is drawn first
  and the marker over it, so 0.35 pt of ink shows outside the ring and the white face covers the
  inside. The marker still reads as open: its inside is paper.
- `paperstyle.open_key` makes the legend key of a line through an open marker as a (line, marker)
  pair, because the rim on a Line2D with a line would outline the line too.
- Markers that are strokes by shape ('x', '+', '|', '_') have no inside and get no edge. Fig. 5's
  ring (an ink circle drawn round a marker whose R² comes from weights inside a loss burst) is an
  annotation, not a marker edge, and is unchanged.

**Edge encodings replaced.**
- Fig. 2f: "black edges = networks trained to 300,000 iterations". Replaced by the new panel's
  layout: those two networks have a column of their own (below).
- Extended Data Fig. 4 and Fig. 3b-e: the default setting's markers carried an ink edge, the cue
  that told grey (`BASELINE`) from the lightest teal (`SETTING_RAMP[0]`, 4.4 apart under
  deuteranopia; paperstyle's palette notes). Now, in a sweep of ordered values, where every other
  value is a circle, the default is a grey square (`fig_paper_F2_fails.rung_style`); where the
  values already have shapes of their own (activation function, penalties, revival methods,
  optimizers) the default stays a grey circle. The legend key of an ordered sweep now shows the
  shapes.
- `fig_paper_F2_fails.draw_open_groups` draws a circle, square or diamond 2 × 0.35 pt narrower, so
  that its outer edge with the rim is where it was before; a triangle keeps its size, because its
  white inside would close up.

**Check 1, every marker artist (`tests/scan_marker_edges.py`, new).** Pass rule, fixed before the
first scan: every filled marker has an ink edge (every RGB channel ≤ 0.2) between 0.3 and 0.4 pt;
every open marker has a white face, a 0.7 pt coloured ring and the ink rim; no line carries the rim.
The first scans found failing markers in Figs. 1, 3 and 5 and in the setting-sweep, mechanism,
combination and techniques-across-sizes Extended Data figures (series-coloured or absent edges,
white edges in Fig. 1, the rim round a line); Fig. 4 stopped before the scan, on a `mew=0.6` that
collided with `mark_floor`'s edge width. Final scan: 0 fail, in all 14 figures
(`fig_paper_F1`, `fig_paper_F2_size`, `fig_paper_F2_fails`, `fig_paper_F3_settings`,
`fig_paper_F4_noreturn`, `fig_ed_mechanism`, `fig_ed_tanh`, `fig_paper_F5_drive`,
`fig_ed_combination`, `fig_ed_rate_alone`, `fig_ed_remedies_size`, `fig_ed_readout`,
`fig_supp_ablation`, `fig_supp_tasks`).

**Check 2, open markers on the printed page (`tests/scan_open_markers.py`).** Fig. 2, Fig. 3, Fig. 5
and the combination Extended Data figure pass. Extended Data Fig. 4 fails on main 2eb3c2f already
(6 open-marker centres read dark, 4 overlapping pairs, all where groups of networks with near-equal
values stack on a data axis). With the rim at full size the scan found 3 more there and 1 in Fig. 3d
(the rim enlarges each marker by 0.35 pt, and a stacked marker then covers its neighbour's
centre); with the size compensation above the failures are the same 10 as on main, at the same
places. Those 10 predate this package and are reported on the board.

**Other checks.** `tests/test_no_function_imports.py`, `tests/test_scripts_start.py` and
`tests/test_paperstyle.py` (also on the six main figures' PDFs) pass with plain `python3`;
`pdfimages -list` shows no raster in any of the 14 PDFs; no figure's content leaves its canvas.
Extended Data Fig. 2 (`fig_ed_population.py`) was not regenerated: package r12-weights owns it,
and it takes the new defaults when next drawn.

### Item 55: Fig. 2f shows the quantities the flat-loss test measures

The old panel plotted each network's least-squares slope of the active count against its loss
change per 100,000 iterations, with the two thresholds as a band and a line. Pavel could not read
it. The networks (19 3BFF networks trained to 200,000 or 300,000 iterations), the thresholds
(3% per 100,000 iterations for the loss, 5 units per 100,000 iterations for the decrease) and the
test (M12, "Statistics and pre-registration") are unchanged.

**New panel.** Two columns: the 17 networks trained to 200,000 iterations and the 2 trained to
300,000, each column with its own iteration axis from 150,000. Lines are coloured by N in four
shades of the 3BFF blue.
- Top: the noisy training loss as medians over 5,000-iteration blocks, as a % change from the first
  block after iteration 150,000. The grey wedge opens from that block's centre at ±3% per 100,000
  iterations. The test compares the last block with the first per 100,000 iterations between their
  centres, so a line ends inside the wedge exactly when the test calls the loss flat.
- Bottom: each network's least-squares line of the number of active units against iteration, over
  every participation probe from 150,000 (thick), drawn from its fitted value at 150,000, over the
  5,000-iteration block medians of the count minus that value (thin). The dashed line is a decrease
  of 5 units per 100,000 iterations. A thick line ends below the dashed line exactly when the test
  calls the number of active units decreasing.

**Why least-squares lines and not the raw count alone.** The brief proposed the change in the
active count since 150,000. Read that way, the panel would contradict the test for 2 of 19
networks: N = 500 seed 822903475 decreases by 7.6 units per 100,000 iterations by the test, but its
block-median count ends 3 units above its start; N = 1,000 seed 3866906883 gains 5.4 units per
100,000 by the test, but its count ends on the dashed line. The test thresholds the slope, so the
panel draws the slope's line, with the count behind it.

**Data.** `fig_paper_F2_size.py --collect-flat` (new) walks the same lineages as `collect()`
(`flat_lineages`, shared) and writes the loss from iteration 145,001 and every participation probe
from 145,000 to `data/fig_paper_F2_size_flat.pkl` (19 rows). Run on Della from a copy of this
branch's code (`~/r12figs_ana`, `data/trained_RNNs` linked to
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs`):

```
sbatch --nodes=1 --ntasks=1 --cpus-per-task=4 --mem=32G --time=01:00:00 --job-name=r12flat \
  --wrap "module load anaconda3/2024.10; source \$(conda info --base)/etc/profile.d/conda.sh; \
  conda activate torch-env; export PYTHONPATH=$HOME/r12figs_ana; cd $HOME/r12figs_ana && \
  python trainRNNbrain/experiments_and_analysis/fig_paper_F2_size.py --collect-flat"
```

Job 16711291, CPU only, 38 s. The pickle was copied to the Mac's `data/`.

**Check (pass rule set before the numbers were seen: the same 19 seeds as the collect record's
panel-(f) rows, and both test quantities equal to within 1e-6).** The drawing stage recomputes the
loss change and the decrease from the raw traces with code written apart from
`c3_flat_loss_units` (its own block medians and `np.polyfit`) and asserts both against the collect
record, network by network. All 19 agree. The panel's printout: 19 networks, 15 with a flat loss,
13 of those with a decreasing number of active units, as Results 2 states. A separate measurement
of the rendered panel finds no drawn line segment crossing any text or the legend.

**Layout.** Bottom row widths of Fig. 2 changed from 2.5 : 1 to 2.15 : 1.35 (panel e narrower,
panel f wider), right margin 0.975. `check_overlaps` finds no overlapping text among 111 artists.

### Figures (img/internal_figures, PDF + SVG)

- [Fig. 2](../../img/internal_figures/fig_paper_F2_size.pdf), new panel f and edges in a-d
- [Fig. 1](../../img/internal_figures/fig_paper_F1.pdf),
  [Fig. 3](../../img/internal_figures/fig_paper_F3_settings.pdf),
  [Fig. 4](../../img/internal_figures/fig_paper_F4_noreturn.pdf),
  [Fig. 5](../../img/internal_figures/fig_paper_F5_drive.pdf)
- Extended Data: [setting sweeps](../../img/internal_figures/fig_paper_F2_fails.pdf),
  [removal of inactive units](../../img/internal_figures/fig_supp_ablation.pdf),
  [read-out](../../img/internal_figures/fig_ed_readout.pdf),
  [mechanism](../../img/internal_figures/fig_ed_mechanism.pdf),
  [tanh](../../img/internal_figures/fig_ed_tanh.pdf),
  [rate penalty alone](../../img/internal_figures/fig_ed_rate_alone.pdf),
  [techniques across sizes](../../img/internal_figures/fig_ed_remedies_size.pdf),
  [combination](../../img/internal_figures/fig_ed_combination.pdf),
  [tasks](../../img/internal_figures/fig_supp_tasks.pdf)
