## 2026-11-06 09:34 — Cross-task figures: the training techniques (Figure 5) and the standard settings (Figure 3) on 3BFF, CDDM and the maze at N = 1,000 (dead-relu-paper#183, #184)

**Issue:** dead-relu-paper#183 (Pavel, 5 Nov: report the techniques that work on every task first,
then the task-specific ones; the same settings and techniques on the 3-bit flip-flop (3BFF),
context-dependent decision making (CDDM) and the maze). **Scripts (new):**
`trainRNNbrain/experiments_and_analysis/fig_paper_F5_tasks.py` and `fig_paper_F3_tasks.py`. The
current paper's `fig_paper_F5_drive.py` and `fig_paper_F3_settings.py` are not changed.
**Figures:** `img/internal_figures/fig_paper_F5_tasks.{pdf,svg}` and `fig_paper_F3_tasks.{pdf,svg}`
(165.1 mm wide, 136 and 132 mm tall), PNG previews beside them (untracked). **Data:** read-out files
only, nothing recomputed: the remedies catalogue `data/remedy_reads_31oct/remedies_catalogue.json`
and the removal analysis `a3_N1000.json` (3BFF techniques), `data/read_uniform_cells.json` and
`data/read_uniform_cells_2.json` (CDDM techniques; 3BFF and CDDM settings), the Figure 3 cache
`data/fig_paper_F2_fails_cache.pkl` (the CDDM ladders, the 3BFF input-norm ladder, the CDDM rate
penalty 0.2 cells), `data/read_maze_arms.json` (every maze arm), the maze ablation JSONs under
`data/rl_maze_ablation_interventions/` and, for the maze activations, the uniform series' run
folders (`data/trained_RNNs_rl_maze/Maze_suite`, read with `common.maze_load_runs` and
`fig_ed_maze.read_points`, the loaders `read_maze_arms.py` uses).

### The read-out and the rules

Every cell is read at the same-size read-out point: 75,000 iterations on 3BFF and CDDM (the CDDM
input-weight norm ladder at its 50,000-iteration budget), update 2,000 on the maze (the end of its
budget). Every marker is therefore open (the paper-wide marker rule, dead-relu-paper#83). The
rules are the pre-registered ones of `docs/interventions_matrix.md`:

- A technique WORKS on a task when every seed has more active units than the largest control and
  the arm's mean performance is within the tolerance of the controls' mean (held-out R² within
  0.01 on the supervised tasks; the clean probe's success within 0.05 on the maze); otherwise it
  FAILS, with the condition that failed. A technique run at several values on one task works on
  that task if it works at at least one finished value.
- A setting value INCREASES or DECREASES the number of active units when every seed lies outside
  its control's range of seeds on the same side; otherwise NO CHANGE. A cell with fewer finished
  seeds than expected gets a provisional direction.

Each script applies the rule itself and prints the verdict the read-out file stores beside it; the
two agree in every finished cell. The 3BFF techniques are read against the catalogue's control
(Figure 1's four N = 1,000 networks: 279, 288, 242, 236 active at 75,000; R² 0.951, 0.952, 0.951 in
the three with 75,000-iteration weights). The rate penalty's 75,000-iteration weights are not in
the catalogue's chain, so its R² is the removal analysis's intact score (9 noise draws: 0.932,
0.930, 0.927) against that analysis's controls (0.9515, 0.9517, 0.9509); every other R² has 36
draws. CDDM techniques are read against the paper-grid control (342, 334, 333; R² 0.831, 0.835,
0.835) except the rate penalty 0.2 cells of the Figure 3 cache, which are read against that
ladder's control (329, 320, 321, 320, 334; R² mean 0.833). The maze reference is the two learned
controls (426 and 449 active, mean 437.5; success 0.992 and 0.977, mean 0.984).

### Figure 5 (cross-task): layout

165.1 x 136 mm. A key row on top: the six techniques as marker shapes in ink (prune-and-duplicate
up triangle, rate penalty circle, read-out dropout down triangle, rate + sparsity left triangle,
rescaling diamond, noisy synapses square) and the three tasks as colours (3BFF blue, CDDM burnt
orange, maze raspberry), with the open-marker meaning. Panels a and b share one x layout: one
column per technique, inside it one group per task and, where a technique was run at several
values on a task, one group per value (the value written under the column: the rate penalty's
λ, the rescaling variants). Rate + sparsity takes a left triangle because Figure 5's filled plus
has no white inside left once it is drawn open (the scan's rule 1, below).

- (a) the number of active units as a fold change over the task's own control mean, 0 to 4.3; a
  grey band per group is the control's own range of seeds; the line at 1.
- (b) the performance difference from the controls' mean: top, held-out R² for 3BFF and CDDM (the
  tolerance 0.01 dashed); bottom, the maze's success (the tolerance 0.05 dashed).
- (c) 3BFF across sizes, the current Figure 5c reused through `fig_paper_F5_drive.size_series` and
  `per_run`: prune-and-duplicate, the one technique that works on every task, with one filled
  marker per network at its floor crossing and a line through the means (fold change, top; R²
  minus control, bottom); the other techniques as faint lines through their means (rescaling and
  read-out dropout at 75,000 iterations, dashed, as Figure 5c draws them).
- (d) removing the recruited units: performance after removing a network's k recruited units (its
  number of active units minus the controls' mean; the k least participating active units, after
  the inactive ones) against performance after removing as many active units at random (the
  median over 5 draws). Two sub-axes: 3BFF (held-out R², the removal analysis A3) and the maze
  (success under the training noise, the five rate-penalty networks ablated on 5 Nov; the random
  order there draws from all units, and its value at k is the median over the five orders,
  interpolated on the k grid). CDDM has no ablated technique network, and the panel says so.

The columns are ordered by the number of tasks on which the technique works (ties: the number of
tasks on which the count condition passes, then the name): prune-and-duplicate, the rate penalty,
rate + sparsity, read-out dropout, noisy synapses, rescaling.

**Open markers are grouped.** The brief asked for one marker per seed. At any spacing that fits
24 groups in 146 mm (or 34 slots in 142 mm in Figure 3) an open marker's ink rim runs through its
neighbour's white inside, and `tests/scan_open_markers.py` (dead-relu-paper#104, rule 1) fails:
the first draft had 55 such failures. Both figures therefore use the paper's convention for open
markers that would overlap (Figure 3a, the reviews of PR #96 and #118): one open marker at the
seeds' mean with a thin bar over their range, the number of seeds beside it when it is not 3. A
maze run that did not learn (`common.maze_learned_index`), a supervised run with R² below 0.8 and
a run the rollback guard stopped are crosses, set beside the group marker. The merged
`fig_ed_maze_settings.py` draws one open marker per seed 0.9 mm apart and fails the same scan
(13 rule failures on today's tree); that figure is outside this package and is not changed here.

### Figure 5: the verdict table (technique x task, the drawn read-out)

| technique | 3BFF | CDDM | maze | works on |
|---|---|---|---|---|
| prune-and-duplicate | works (3.07x; R² −0.005) | works (2.11x; −0.003) | works (1.68x; success −0.008) | 3 tasks |
| rate penalty | fails at λ 0.1 (3.72x; R² −0.021) | works at 0.1 (2.97x; −0.007) and 0.2 (3.08x; −0.004) | works at 0.03 (1.93x; +0.000) and 0.3 (2.06x; +0.003); fails at 0.1 (2.04x; −0.120: seed 1 ends at 0.63) | 2 tasks |
| rate + sparsity 0.05 | fails at 0.1 (3.57x; −0.028) | works at 0.2 (3.08x; −0.008); 0.1 provisional, 1 of 3 (2.97x; −0.011) | fails at 0.1 (2.10x; −0.193: two seeds did not learn) | 1 task |
| read-out dropout | fails (2.05x; −0.017) | works (1.54x; −0.008) | fails (1.00x, two seeds inside the controls' range; −0.190, no seed learned) | 1 task |
| noisy synapses | fails (2.19x; −0.022) | fails (1.47x; −0.015) | fails (1.05x; −0.062) | 0 tasks |
| rescaling | fails (1.65x; −0.013) | fails (0.54x; −0.015) | the paper's values stopped (3 runs, no checkpoint); mild (α 1.0005, cap 8) 2 of 3 at 0.73x, 1 stopped; few (protect 0.05 N) fails (0.55x; −0.245) | 0 tasks |

**Where this differs from the brief's verdicts (the status board of 6 Nov).** The board says
read-out dropout works on 3BFF and rescaling works on 3BFF. Both rest on the floor-crossing read
of the catalogue, where 1 of 3 dropout runs (502 active, R² 0.940 against the controls' 0.950,
0.0099 below) and 1 of 4 rescaling runs (418, R² 0.943, 0.007 below) have a read. At 75,000
iterations, the same-size read-out point the pre-registration names, every run has a read and
both fail the 0.01 tolerance: dropout 496, 569, 539 active at R² 0.932, 0.933, 0.939 (0.017 below
the controls' mean of 0.951); rescaling 476, 364, 422, 460 at R² 0.938, 0.938, 0.938, 0.940
(0.013 below). The brief's "rate + sparsity: 3BFF and CDDM at 0.2 + 0.05" has no 3BFF cell behind
it: the standard-network 3BFF pair is 0.1 + 0.05 (`NBitFlipFlop_ff_both40k_c200000`), which fails
at 75,000 (R² 0.028 below) and at the floor (0.022 below); the 0.2 + 0.05 cells of the
`std_penlong` sweep have a trainable bias and are not comparable. The status board's C6 paragraph
now carries these three corrections.

### Figure 5: every number drawn

Panels a and b, one row per group (fold = active units over the control's mean; Δ = performance
minus the controls' mean; the control's seeds in the last column):

| technique | task | value | active units (fold) | performance (Δ) | control |
|---|---|---|---|---|---|
| prune-and-duplicate | 3BFF | | 793, 781, 830 (3.04, 2.99, 3.18) | R² 0.946, 0.947, 0.947 (−0.005, −0.004, −0.004) | 279, 288, 242, 236; 0.951, 0.952, 0.951 |
| prune-and-duplicate | CDDM | | 701, 699, 726 (2.08, 2.08, 2.16) | 0.832, 0.827, 0.832 (−0.001, −0.007, −0.002) | 342, 334, 333; 0.831, 0.835, 0.835 |
| prune-and-duplicate | maze | | 747, 741, 712 (1.71, 1.69, 1.63) | success 0.984, 0.953, 0.992 (0.000, −0.031, +0.008) | 426, 449; 0.992, 0.977 |
| rate penalty | 3BFF | 0.1 | 985, 995, 935 (3.77, 3.81, 3.58) | 0.932, 0.930, 0.927 (−0.019, −0.021, −0.024; 9 draws) | as above; 0.9515, 0.9517, 0.9509 |
| rate penalty | CDDM | 0.1 | 1,000 x 3 (2.97) | 0.825, 0.829, 0.827 (−0.009, −0.005, −0.007) | paper grid |
| rate penalty | CDDM | 0.2 | 1,000 x 5 (3.08) | 0.830, 0.828, 0.829, 0.829, 0.828 (−0.003 to −0.005) | 329, 320, 321, 320, 334; 0.833, 0.831, 0.834, 0.834, 0.834 |
| rate penalty | maze | 0.03 | 878, 862, 792 (2.01, 1.97, 1.81) | 1.000, 0.984, 0.969 (+0.016, 0.000, −0.016) | learned controls |
| rate penalty | maze | 0.1 | 763, 927, 982 (1.74, 2.12, 2.24) | 0.633 (not learned), 0.969, 0.992 (−0.352, −0.016, +0.008) | |
| rate penalty | maze | 0.3 | 962, 930, 812 (2.20, 2.13, 1.86) | 0.961, 1.000, 1.000 (−0.023, +0.016, +0.016) | |
| rate + sparsity | 3BFF | 0.1 | 894, 946, 958 (3.42, 3.62, 3.67) | 0.916, 0.926, 0.927 (−0.035, −0.025, −0.025) | catalogue |
| rate + sparsity | CDDM | 0.2 | 1,000 x 5 (3.08) | 0.822, 0.824, 0.825, 0.829, 0.828 (−0.012 to −0.004) | Figure 3a control |
| rate + sparsity | CDDM | 0.1 | 1,000 (2.97), 1 of 3, 2 training | 0.823 (−0.011) | paper grid |
| rate + sparsity | maze | 0.1 | 901, 907, 946 (2.06, 2.07, 2.16) | 1.000, 0.430 (not learned), 0.945 (not learned) (+0.016, −0.555, −0.039) | |
| read-out dropout | 3BFF | | 496, 569, 539 (1.90, 2.18, 2.06) | 0.932, 0.933, 0.939 (−0.019, −0.019, −0.013) | catalogue |
| read-out dropout | CDDM | | 524, 505, 523 (1.56, 1.50, 1.56) | 0.824, 0.825, 0.830 (−0.010, −0.009, −0.004) | paper grid |
| read-out dropout | maze | | 413, 461, 436 (0.94, 1.05, 1.00) | 0.836, 0.789, 0.758, none learned (−0.148, −0.195, −0.227) | |
| noisy synapses | 3BFF | | 572, 552, 589 (2.19, 2.11, 2.25) | 0.929, 0.926, 0.933 (−0.022, −0.025, −0.019) | catalogue |
| noisy synapses | CDDM | | 486, 487, 511 (1.44, 1.45, 1.52) | 0.820, 0.828, 0.808 (−0.014, −0.006, −0.026) | paper grid |
| noisy synapses | maze | | 451, 478, 454 (1.03, 1.09, 1.04) | 0.906, 0.961, 0.898 (not learned) (−0.078, −0.023, −0.086) | |
| rescaling | 3BFF | 10x | 476, 364, 422, 460 (1.82, 1.39, 1.62, 1.76) | 0.938, 0.938, 0.938, 0.940 (−0.013, −0.013, −0.013, −0.011) | catalogue |
| rescaling | CDDM | 10x | 196, 183, 169 (0.58, 0.54, 0.50) | 0.819, 0.813, 0.824 (−0.015, −0.021, −0.009) | paper grid |
| rescaling | maze | 10x | 3 runs stopped at updates 52-60, no checkpoint (crosses at 0, "stopped") | | |
| rescaling | maze | mild | 314, 321 (0.72, 0.73), neither learned; 1 stopped at update 800 | 0.750, 0.633 (−0.234, −0.352) | |
| rescaling | maze | few | 255, 231, 234 (0.58, 0.53, 0.53), none learned | 0.656, 0.750, 0.812 (−0.328, −0.234, −0.172) | |

3BFF at the floor crossing (printed, not drawn; the catalogue's 1.05x read): prune-and-duplicate
745, 788, 772 at R² 0.944, 0.945, 0.946 (works); the rate penalty 0 of 3 read (two never stay
under the bar, one ends below its floor); read-out dropout 1 of 3 read, 502 at 0.940 (works);
rate + sparsity 976, 986, 961 at 0.932, 0.924, 0.929 (fails, 0.022 below); rescaling 1 of 4 read,
418 at 0.943 (works); noisy synapses 561, 544, 583 at 0.926, 0.926, 0.930 (fails, 0.023 below);
the controls 277, 296, 297, 240 at 0.949, 0.950, 0.950, 0.950.

Panel c (3BFF, fold change and the median R² difference per size; floor crossing except rescaling
and read-out dropout at 75,000): prune-and-duplicate 2.06, 2.77, 3.24, 3.59 (n = 3, 3, 3, 1) at
−0.003, −0.005, −0.009, −0.018; the rate penalty 4.81 and 5.64 at N = 2,000 and 4,000 (n = 3, 2;
no floor read below); noisy synapses 1.81, 2.03, 1.54, 1.60 at −0.044, −0.024, −0.027, −0.019;
rescaling 1.55, 1.65, 1.73, 1.95 at −0.011, −0.013, −0.014, −0.017; read-out dropout 2.05, 2.05,
2.00, 2.01 at −0.020, −0.019, −0.013, −0.013.

Panel d (x = after removing k random active units, median of 5; y = after removing the k recruited
units). 3BFF, held-out R²: read-out dropout (k 228, 300, 270) 0.228/0.193, 0.127/0.176,
0.181/0.272; rate + sparsity (624, 677, 689) 0.159/0.489, 0.151/0.373, 0.140/0.355;
prune-and-duplicate (530, 513, 560) 0.057/−4,902 (at the bottom edge with an arrow), 0.104/0.233,
0.054/0.227; the rate penalty (716, 726, 666) 0.106/0.218, 0.128/0.184 (weights inside a loss
burst), 0.146/0.279; noisy synapses (303, 284, 321) 0.155/0.578, 0.175/0.604, 0.152/0.571;
rescaling (210, 94, 155, 194) 0.040/0.309, 0.211/0.703, −41,974/0.430 (at the left edge; burst),
0.073/0.558 (burst). Maze, success: rate 0.03 s1 (k 482) 0.330/0.775, s2 (416) 0.263/0.739; rate
0.3 s1 (524) 0.336/0.680, s2 (492) 0.284/0.617; rate 0.1 s3 (544) 0.077/0.550. The means over the
random orders, which entry 2026-11-05_1230 reports, are 0.345, 0.273, 0.320, 0.294 and 0.073.

### Figure 3 (cross-task): layout

165.1 x 132 mm. One block per task (a 3BFF, b CDDM, c maze), each of two axes sharing the columns:
the upper one the number of active units (0 to 1,000), the lower one the performance (held-out R²
on 3BFF, axis 0.80 to 0.975, and CDDM, 0.775 to 0.85; the maze's success, 0 to 1, the "learned"
level 0.9 dashed). One column per setting, one slot per value in the ladder's order (so a ladder
that spans decades is evenly spaced in the log of its value), 34 slots of 3.6 mm: control; weight
decay 0, 1e-5, 1e-4; input-weight row norm 0.5, 2, 5, 20; input-weight cap 0.1, 0.3, 0.5, 2;
activity noise 0, 0.01, 0.1; initial spectral radius 0.5, 1.5, 2; the sparsity penalty 0.05; the
metabolic penalty 0.01, 0.1, 1, 10; the recurrent-weight penalty 0.05, 0.2; AdamW and SGD at 0.03,
0.1, 0.3, 1; leaky ReLU, softplus, shifted sigmoid, tanh. The column titles are staggered on two
rows; the value labels run along the bottom. Inside a slot the seeds that learned are one open
marker at their mean with a bar over their range (the number beside it when not 3); the band
behind each slot is that cell's OWN control (each ladder is read against the control its read-out
file names, so the bands differ between columns: on 3BFF the B5 Adam control 233-270 for most
columns and the input-norm ladder's default 245-296 for the norm and the caps 0.1, 0.5, 2; on
CDDM the drift default 325-345, the noise ladder's 311-340, the recinit control 315-347, the
Figure 3a control 320-334, the screen-3 control 332-340, the paper-grid control 333-342 and the
50,000-iteration control 355-376 of the norm ladder). A slot reads "unfinished" when no run has
finished, "k/3" when some have, "not run" where the programme has no such cell (the maze's upper
spectral-radius rung is 1.5 where the supervised tasks' is 2; the cap was run at 0.1, 0.5 and 2 on
3BFF only), "stopped" for the maze input norm 5, and "no read at 75k" for CDDM's tanh cell (read
at the end of 150,000 iterations only). Adam without weight decay is the weight-decay column's 0
and is not repeated in the optimiser column.

**Why the performance is a second row.** The current Figure 3a writes R² minus the default beside
nine rows of counts, and `fig_ed_maze_settings` draws a success row under its count row. A caption
cannot carry 34 cells times three tasks of performance, and a marker's fill is reserved for the
read-out point, so a second row per task is the one encoding that keeps every number on the page.

### Figure 3: the direction per value (f finished, i incomplete, u unfinished, s stopped, n not run or no read)

| setting | value | 3BFF | CDDM | maze |
|---|---|---|---|---|
| weight decay | 0 | no change f (252, 296, 367) | increase f (469, 448, 442) | no change f (404, 452, 432) |
| weight decay | 1e-5 | decrease f (225, 229, 203) | decrease f (261, 260, 263) | decrease f (402, 417, 370) |
| weight decay | 1e-4 | decrease f (157, 144, 163) | decrease f (154, 166, 159) | decrease f (378; 335, 341 not learned) |
| input norm | 0.5 | no change f (314, 284, 343) | decrease f (334, 334, 336; at 50k) | increase f (577, 593, 549) |
| input norm | 2 | increase f (424, 418, 387) | decrease f (247, 268, 257) | no change f (443, 357; 290 not learned) |
| input norm | 5 | increase f (412, 443, 349) | decrease f (210, 216, 201) | stopped s (3 runs) |
| input norm | 20 | increase f (477, 500, 455) | decrease f (166, 185, 178) | decrease i (331, success 0.00; 2 stopped) |
| cap | 0.1 | increase f (303, 429, 402; R² 0.82-0.87) | n | n |
| cap | 0.3 | increase i (365, 1 of 3; R² 0.904) | increase f (396, 390, 400) | no change f (434, 463, 380) |
| cap | 0.5 | increase f (362, 425, 321; R² 0.91-0.92) | n | n |
| cap | 2 | no change f (280, 343, 352) | n | n |
| activity noise | 0 | increase f (284, 289, 281) | decrease f (246, 249, 258, 258, 245) | no change f (440, 425, 400) |
| activity noise | 0.01 | u | decrease f (272, 249, 279, 268, 275) | no change f (456, 380, 436) |
| activity noise | 0.1 | u | increase f (412, 378, 391, 394, 408) | decrease f (346, 317, 311; none learned) |
| spectral radius | 0.5 | no change i (295, 270; 2 of 3) | increase f (355, 368, 367) | decrease f (338, 393, 414) |
| spectral radius | 1.5 | n | n | no change f (350, 471; 319 not learned) |
| spectral radius | 2 | u | no change i (308, 317; 2 of 3) | n |
| sparsity penalty | 0.05 | no change i (233; 1 of 3) | decrease f (267, 270, 257, 281, 248) | decrease f (399; 378, 386 not learned) |
| metabolic penalty | 0.01 | increase i (351; 1 of 3) | no change f (343, 344, 324) | decrease i (379; 1 of 3) |
| metabolic penalty | 0.1 | u | increase f (347, 356, 350; R² −0.005) | no change i (431; 1 of 3) |
| metabolic penalty | 1 | u | increase f (360, 377, 387; R² −0.013) | u |
| metabolic penalty | 10 | u | no change f (342, 321, 328; R² 0.787-0.789, below the learned bar) | u |
| recurrent-weight penalty | 0.05 | u | increase i (441, 434; 2 of 3) | no change i (481, 394; 2 of 3) |
| recurrent-weight penalty | 0.2 | u | increase i (445, 450; 2 of 3) | increase i (517; 1 of 3) |
| optimiser | AdamW | increase f (358, 306, 311; R² +0.007) | increase f (453, 431, 448) | u |
| optimiser | SGD 0.03 | increase f (731, 759, 757; R² −0.077) | increase f (842, 813, 834; −0.014) | u |
| optimiser | SGD 0.1 | increase f (620, 578, 586; −0.058) | increase f (857, 845, 859; −0.007) | u |
| optimiser | SGD 0.3 | no change f (259, 260, 272; −0.051) | increase f (895, 882, 881; −0.007) | u |
| optimiser | SGD 1 | no change f (265, 220; 212 at R² 0.578 not learned) | increase f (846, 821, 767; −0.013) | u |
| activation | leaky ReLU | u | increase f (385, 378, 348) | increase f (603, 600, 610) |
| activation | softplus | no change i (244; 1 of 3) | increase f (372, 376, 376) | no change f (437, 420; 393 not learned) |
| activation | sigmoid | u | decrease f (218, 234, 198) | decrease f (314, 322, 295; none learned) |
| activation | tanh | u | n (no read at 75k) | increase f (1,000, 1,000, 999; none learned, success 0.00-0.38) |

The controls drawn in the first slot: 3BFF 270, 234, 233 at R² 0.951 (B5 Adam); CDDM 342, 334,
333 at 0.831, 0.835, 0.835 (paper grid); maze 426, 449 learned (success 0.992, 0.977) and 415
not learned (0.727, a cross). The SGD arms on 3BFF are the B5 cells with weight decay 1e-6; the
wd 0 variants exist and are not drawn. Across the three tasks today: weight decay 1e-5 and 1e-4
lower the number of active units on all three; the input-weight norm raises it on 3BFF, lowers it
on CDDM and, at 0.5, raises it on the maze (5 and 20 diverge there); the sparsity penalty alone
lowers it on CDDM and the maze; AdamW raises it on both supervised tasks at the same R²; SGD
raises it on both at a lower R² (0.05-0.08 below on 3BFF, 0.007-0.014 on CDDM). Every other row
waits for the runs below.

### Unfinished, and how to rerun

Still training at 09:30: on Della (arrays 16957556-16957562), 3BFF activity noise 0.01 and 0.1,
spectral radius 2.0 (and the third seed of 0.5), λ_rwm 0.05 and 0.2, λ_met 0.1, 1 and 10 (and two
seeds of 0.01), two seeds of the sparsity penalty, leaky ReLU, sigmoid, tanh and two seeds of
softplus, two seeds of the cap 0.3; CDDM spectral radius 2.0 and λ_rwm 0.05 and 0.2 third seeds,
rate 0.1 + sparsity 0.05 seeds 2 and 3; on Spock (array 7389398), the maze λ_rwm, λ_met, AdamW,
SGD, redraw and zero-out arms (redraw and zero-out are not in Figure 5, which holds the paper's
six techniques). Both scripts read only the JSONs and caches, so the rerun is: rsync the finished
folders, rebuild `data/read_uniform_cells_2.json` (`read_uniform_cells_2.py`, after
`data/della_batch1_status.tsv` is rebuilt from `sacct`) and `data/read_maze_arms.json`
(`read_maze_arms.py`), then

```
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/fig_paper_F5_tasks.py
CUDA_VISIBLE_DEVICES= PYTHONPATH=. python trainRNNbrain/experiments_and_analysis/fig_paper_F3_tasks.py
```

Every status, number and verdict follows the files; a cell copied while training shows as
"unfinished" or "k/3" until its JSON is rebuilt. The column order of Figure 5 is recomputed on
every run.

### Checks

`tests/test_paperstyle.py` on both PDFs: ok (Helvetica, Helvetica-Bold and Helvetica-Oblique
only; panel letters 8.5 pt; every y label 1.36 mm from its ticks). `pdfimages -list`: no raster in
either. `tests/scan_open_markers.py fig_paper_F5_tasks fig_paper_F3_tasks`: 0 rule failures (the
first draft, one marker per seed, had 55). `fig_ed_maze.check_overlaps`: no overlapping text (92
and 133 text artists). `tests/test_scripts_start.py`: 57 scripts start.
`tests/test_no_function_imports.py`: 209 files, none.
