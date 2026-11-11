## 2026-11-10 08:26 — Round-22 sizes package: the N = 250 3BFF/CDDM and N = 100 Maze networks in Fig. 2, Extended Data Figs. 2 and 4, Supplementary Table 1 and the every-size fits (dead-relu-paper #226)

Rules: `2026-11-09_1925_n250-n100maze-rules.md` (written before training). Results:
`2026-11-10_0806_n250-n100maze-results.md`. This entry records the figure changes and the paper text built on
them; no network was trained or re-read for it. Branch `wp/r22-sizes` (from main 8db1eeb).

*Floor crossing*: the first iteration after which the smoothed noisy training loss stays at or below 1.05 times its
fitted floor through 1.2 times that iteration. *Learned point* (Maze): the first probe from which the noise-free
probe success stays at or above 0.9 for five probes, in a record whose last probe is also at or above 0.9.
*Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05·q95(p).

### Code

- `fig_paper_F2_size.py`: `add_size250` appends the six N = 250 rows of `data/size250/supervised.json` (written by
  `size250_n100maze_read.py` with `read_one`) to the collect record's size series when the figure is drawn;
  `cells_of` and `panel_size` take the sizes from the rows, so N = 250 is drawn in Fig. 2a,b and tabulated, and
  every fit with `FIT_MIN_N` (500) is unchanged. The N = 100 Maze networks enter through `maze_series`, which
  finds them under `Maze_suite`. `maze_series` now also returns the first probe's update and active count
  (`update_first`, `active_first`), printed per network. `check_overlaps` floors the label pads as `save()` does
  and reports text past the canvas edge (as `fig_ed_maze.check_overlaps` does since round 21). The Fig. 2f label
  position of PR #225 (`axes[1][1].text(298_000, -1.0, ...)`) is untouched.
- `fig_ed_readout.py` (Extended Data Fig. 4): panel b draws the N = 250 networks too (`add_size250`).
- `fig_ed_maze.py` (Extended Data Fig. 2): N = 100 in `SIZES`/`SIZE_SEEDS`; a sixth shade (N = 100 takes the old
  N = 250 tint, 250 and 500 re-spaced towards the N = 1,000 tint); panel b has six sub-panels (11.5 mm wide, every
  19.4 mm, the middle x tick unlabelled); panel a is 31.0 mm wide; panel c's y axis starts at 45 so the N = 100
  counts (60-75) are drawn, and its "every unit active" label moved to x = 125 (at 280 it met the legend); in
  panel b the end markers are spread round the stopped network's cross (in the narrower N = 4,000 sub-panel one
  landed on it and failed the open-marker scan, darkest centre channel 11).

### Checks

- Panel c of Fig. 2 (exponents at 1.05x, 1.10x, 1.20x with their intervals) is identical to the printout of the
  main-branch script on the same data (diff of the two printouts empty).
- The script's table rows differ from those of the main-branch script by exactly the six N = 250 rows; the four
  N = 100 Maze rows appear in both (the main-branch script already found them). Ten rows added to the paper's
  Supplementary Table 1; its hand-edited rows are kept.
- N = 250 estimated R² at the crossing: loss ratio L(t*)/L_end 1.04-1.05, so each estimate lies within 0.008 of
  the final R², inside the table note's 0.87-1.10 and 0.01.
- `tests/scan_marker_edges.py`: PASS (Fig. 2: 180 markers; ED Fig. 2: 181; ED Fig. 4: 127). `tests/scan_open_markers.py`:
  PASS on all three after the cross fix. `check_overlaps`: no overlapping text and nothing past the canvas edge
  (Fig. 2: 112 texts; ED Fig. 2: 108; ED Fig. 4: 67). `pdfimages -list`: no rasters. All three rendered at 300 dpi
  and inspected. `tests/test_no_function_imports.py`: ok.

### Numbers the paper takes from the printouts

- Fig. 2a,b means at the 1.05x crossing, N = 250: 3BFF 133.7 ± 9.0 (n = 3, 53.5%), CDDM 183.0 ± 7.8 (n = 3, 73.2%);
  Maze at the learned point, N = 100: 67.8 ± 6.4 (n = 4). Networks per size for Maze: 4, 4, 3, 6, 3, 4.
- Maze, learned point to end of training (4,000 updates), N = 100: 61 → 60, 71 → 71, 64 → 60, 75 → 68. Over the 24
  learned networks of the size series, 20 have fewer active units at the end (by 1 to 113 units, 1.6 to 25.9%), one
  (N = 100 seed 2) the same number, and three more (N = 2,000 seed 1; N = 4,000 seeds 2 and 3). With the seven
  3BFF and CDDM networks of Fig. 2e, the end count is lower in 27 of 31 networks (was 24 of 27).
- Every-size fits (from `data/size250/*.json`): 3BFF 0.516 (0.498-0.533), CDDM 0.571 (0.563-0.579) at 1.05x; 3BFF
  0.541 and 0.540, CDDM 0.610 and 0.693 at 1.10x and 1.20x; Maze 0.584 (0.551-0.618) at the learned point, 0.619
  (0.596-0.642) at update 4,000. Slopes of log mean count on log N between neighbouring sizes at 1.05x: CDDM 0.779
  (100-250), 0.705 (250-500), 0.531, 0.347, 0.484; 3BFF 0.538, 0.617, 0.437, 0.491, 0.500. The Maze fit over
  N = 250-4,000 at the learned point (b = 0.471) gives 116.6 at N = 100; measured 67.8.
- Active units at the first probe of every Maze size-series network (after PPO update 1): N = 100: 65, 64, 56, 56;
  250: 169, 157, 159, 155; 500: 290, 308, 295; 1,000: 632, 616, 583, 576, 594, 621; 2,000: 1,255, 1,182, 1,202;
  4,000: 2,408, 2,397, 2,471, 2,434 (seed 4, later stopped), 2,452. That is 56% to 68% of N at every size. At
  N = 100 the learned-point count (61-75) stays near the first-probe count; at N = 1,000 it is lower (406-517).

### Decisions (recorded on dead-relu-paper #226)

1. The N = 100 Maze networks enter the end-of-training count of Results 2 and Extended Data Fig. 2b,c, because
   Extended Data Fig. 2 now draws them. The count becomes 27 of 31; the N = 100 network with an unchanged count is
   one of the four exceptions, its value (71 at both points) is in the per-network table.
2. No main-text claim says that inactive units arise only through training. The Introduction and Results 5 speak
   of "the tendency of units to become inactive"; at N = 1,000, where the techniques were tested, Maze networks
   go from 576-632 active units at the first probe to 406-517 at the learned point, so the phrase holds there. The
   first-probe counts go into Methods: Participation, with the 3BFF count before training for comparison.
3. Methods: Statistics keeps the stated reason for the range N ≥ 500 (a ceiling at N = 100) as the expectation it
   was ("we expected that ceiling to bend the curve"), and states that CDDM and Maze means bend at small N, so the
   every-size intervals, which reflect only the spread between networks, understate the uncertainty.

Figures: `img/internal_figures/fig_paper_F2_size.pdf`, `fig_ed_maze.pdf`, `fig_ed_readout.pdf` (+ .svg).
