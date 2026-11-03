## 2026-11-02 23:41 — r13-figfix: Fig. 1a,b on a seeded batch (#165); Extended Data Fig. 4 open markers (#176)

Work package r13-figfix (dead-relu-paper #127; issues #165 and #176). Code branch `wp/r13-figfix`
(local, from main 757d90b).

### 1. Fig. 1a,b drawn on a seeded 24-trial batch (#165)

**Cause** (entry 2026-11-02_2048, section "Anomaly found on the first run"). `TaskNBitFlipFlop`
draws its trials from its own generator `self.rng`, seeded from `cfg.task.seed`, which every
training config leaves null. `fig_paper_F1.example_network_pair` seeded only `np.random`, so the
24 trials of Fig. 1a,b differed on every call. The figure showed one unseeded draw, cached by Della
job 16556248 (945 active units before training, 183 after).

**Change.** `example_network_pair` sets `cfg.task.seed = TRACE_SEED` (11) before the task is
instantiated, as `active_before_after.py` does. The simulation cache moves to
`data/fig_paper_F1_cache_v3.npz`, so the unseeded v2 cache is never read again. A new option,
`--collect-example SOURCE`, re-simulates panels a and b and copies every other panel from the
record SOURCE. It writes `data/fig_paper_F1_collect_v2.pkl` (the new `COLLECT`). The v1 record
`data/fig_paper_F1_collect.pkl` is left as it was, so main's drawing code still reproduces main's
figure.

**Rule, written before the run.**
1. Two runs of `--collect-example data/fig_paper_F1_collect.pkl` give identical participation
   vectors before and after training (largest absolute difference 0) and so identical counts.
   Otherwise the batch is still not fixed.
2. The counts equal measurement B of `active_before_after.py` for network 3374117298 on task seed
   11: 940 active before training and 184 after (entry 2026-11-02_2048, table). Both paths simulate
   the same weights on the same seeded batch, so any other count means the two paths differ.
3. The `tasks`, `ablation` and `floor_now` entries of the v2 record equal those of v1 exactly, so
   panels c to e do not move.
4. The ten-network numbers of Results 1 (measurement A, the logged participation trace) are
   independent of the batch. Re-run on the r11-results1 copies, they must stay 936.9 ± 14.4 before
   training and 187.7 ± 16.0 after (entry 2026-11-02_2122).

**Run.** Della job 16717218 (CPU, 4 cores, 57 s), on a copy of commit b095a96 at
`~/r13-figfix-b095a96` (PYTHONPATH set there, data linked to the shared Della data). The job ran
`python trainRNNbrain/experiments_and_analysis/fig_paper_F1.py --collect-example
data/fig_paper_F1_collect.pkl` twice and kept each output (`fig_paper_F1_collect_v2_run1.pkl`,
`_run2.pkl`). The three files were copied to the Mac's `data/`. Measurement A was re-run on the Mac
with `active_before_after.py` on the r11-results1 copies of the ten networks.

**Verdict against the rule.**
1. The two runs wrote byte-identical records (md5 633487b8...). Largest absolute difference of the
   participation vectors and of the rates between the runs: 0. Passes.
2. 940 active units before training and 184 after, equal to measurement B for 3374117298. Passes.
3. `tasks`, `ablation` and `floor_now` of v2 equal v1 exactly; the excitatory fraction of panel a
   is unchanged (0.328). The drawing stage's printout differs from v1 only in the panel a and b
   lines. Passes.
4. Measurement A over the ten networks: 936.9 ± 14.4 before and 187.7 ± 16.0 after (n = 10, s.d.
   with n − 1); measurement B: 934.0 ± 13.5 and 188.0 ± 17.0. Unchanged. Passes.

**Old and new counts of Fig. 1a,b** (network 3374117298, N = 1,000, 24 trials):

| | v1 (unseeded batch, job 16556248) | v2 (task seed 11, job 16717218) |
|---|---|---|
| active before training (panel b, top) | 945 | 940 |
| active after 200,000 iterations (panels a and b) | 183 | 184 |
| of the 8 units drawn at random in panel a, active | 2 | 2 |

Figure: [img/internal_figures/fig_paper_F1.pdf](../../img/internal_figures/fig_paper_F1.pdf)
(+ .svg; no rasters; no overlapping text among 90 text artists; `scan_marker_edges.py` and
`scan_open_markers.py` pass on it).

### 2. Extended Data Fig. 4: open markers no longer cover each other's centres (#176)

`tests/scan_open_markers.py fig_paper_F2_fails` failed 11 times on main 757d90b: 7 open markers
whose centre reads dark (darkest channel 11 to 128 of 255) and 4 overlapping pairs (0.02 to 0.42 mm
apart). Every case is a group mean of networks with near-equal values. On the data axes the later
marker was drawn whole on top (`draw_open_groups`, `max_shift` 0). In the panel c traces, five
groups of the metabolic ladder lie within 1 mm of each other at 75,000 iterations, and two steps
to the right were not enough.

**Why not drawing order or a smaller marker.** An open marker is 1.17 mm across, and the scan reads
a 0.25 mm block at its centre. When two centres are closer than about 0.8 mm, one marker's ring
crosses the other's centre whatever the order, and shrinking one marker cannot clear both centres
while keeping the pair from overlapping by more than half a width (scan rule 2). So a marker has to
move.

**Change.** `draw_open_groups(..., free_axis="x" | "y")`: a marker that would overlap one already
drawn is moved, with its range bars, by the smallest distance that leaves the two at least one
marker diameter apart (touching, not overlapping), to either side. The marker moves along the
iteration axis in a trace panel and along the R² axis in an R² panel and in panel l. The number of
active units is never moved. Figure 3 keeps its own call (`max_shift=2`; `b5_scatter` stacks unless
asked), and its PDF renders pixel-identical before and after (300 dpi, 0 pixels differ).

Largest move along R² per panel (script printout): b 0.0006, c 0.0092, d 0.0011, g 0.0009,
h 0.0038, i 0.0004, j 0.0010, k 0.0001, l (200,000 iterations) 0.0062; none in a, e and f. Each
move is at most one marker width in print.

**Checks.** `tests/scan_open_markers.py fig_paper_F2_fails`: PASS, 0 rule failures (was 11);
268 of 268 filled-marker centres read dark (the mapping check). `tests/scan_marker_edges.py
fig_paper_F2_fails`: 476 markers pass, 0 fail (the 0.35 pt ink edge is kept). The layout check
reports 185 texts, 11 legends, 0 problems.

Figure: [img/internal_figures/fig_paper_F2_fails.pdf](../../img/internal_figures/fig_paper_F2_fails.pdf)
(+ .svg; no rasters).

### Paper

Fig. 1 caption: 183 → 184 (panel a); the caption now names the task seed of the 24 trials. Extended
Data Fig. 4 caption: two sentences on the open markers (group mean with range bars; the moves along
iteration or R²). Results 1 quotes the ten-network numbers, which do not change. Paper PR on
dead-relu-paper branch `wp/r13-figfix`.
