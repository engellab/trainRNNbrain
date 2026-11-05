## 2026-11-05 08:12 — Extended Data figure script for the maze interventions ladder, fig_ed_maze_interventions.py (issue #5)

**What it is.** `trainRNNbrain/experiments_and_analysis/fig_ed_maze_interventions.py` draws the maze
counterpart of Figure 5 (the training techniques) from the N = 1,000 ladder of issue #5
(`data/trained_RNNs_rl_maze/Maze_interventions/`, Spock arrays 7383696, 7383697, 7383755) and the
uniform series' N = 1,000 ReLU runs as controls (`Maze_suite/N=1000_act=relu/s{1,2,3}`). Four
panels, 165.1 x 106 mm, bold letters, a key row above the panels as in Figure 5: (a) active units
against PPO updates, one thin line per finished run, a filled marker at each run's "learned" probe;
(b) probe success at the end against active units at the end, one marker per run; (c) top, the fold
change in active units at the end over the learned controls' mean, one marker per seed and a bar at
the arm's mean, bottom, each run's end success minus the learned controls' mean success, with the
pre-registered tolerance 0.05 as a dashed line; (d) the ablation of `rl_maze_ablate.py` for every
technique network with a JSON under `data/rl_maze_ablation_interventions/`, overlaid with one learned
control's (`data/rl_maze_ablation/`, the lowest seed). Output `img/internal_figures/fig_ed_maze_interventions.{pdf,svg}`
(vector, `paperstyle.save`); preview `img/internal_figures/fig_ed_maze_interventions_preview.png`
(not tracked).

**Choices, and why.** The arm of every run comes from its saved config (`arm_of`: `trainer.lambda_frm`,
`trainer.lambda_rws`, `model.sigma_w`, `trainer.dropout` with its rate and beta, `trainer.prune_reinit`),
not from the folder name; the folder name is printed beside it and a mismatch is reported (none
today). The controls are BASELINE grey (Figure 5's convention); the three rate-penalty strengths are
Figure 5d's three shades of the rate violet ordered light to dark by lambda (#c3b0e8, #9d7fd3,
RATE_PENALTY); rate + sparsity, read-out dropout, noisy synapses and rescaling take their
`paperstyle` colours and Figure 5's marker shapes. Every marker of (b) and (c) is read at the end of
the fixed budget (update 2,000), so every marker is open (dead-relu-paper#83: filled means a run's
own convergence point, which only (a)'s "learned" marks are); a run that did not learn is a cross in
its arm's colour, as in `fig_ed_maze`. The controls' reference is the mean over the LEARNED controls
(the pre-registered exclusion of issue #3 applied to the controls as to the arms); the printout gives
the all-controls means beside it, because the rate 0.1 verdict depends on the choice (below). A run
is finished when its last logged update reaches the config's `n_updates`; an unfinished run is
listed and drawn nowhere; a folder with no run files is listed as stopped and drawn nowhere.
Panel (c) keeps one column per arm of the ladder, so the layout does not change as runs finish:
an empty column reads "unfinished" or "stopped".

**The numbers (printout of the script, 15 finished technique runs and 3 controls).** Read-out
points as pre-registered: "learned" = the first probe from which the clean probe's success stays
at or above 0.9 for 5 probes, with the last probe at or above 0.9 (`common.maze_learned_index`);
"end" = update 2,000.

| arm (from config) | seed | success, end | learned at | active, learned | active, end | fold |
|---|---|---|---|---|---|---|
| control | 1, 2, 3 | 0.99, 0.98, 0.73 | 1,050, 760, never | 451, 517, - | 426, 449, 415 | 0.97, 1.03, 0.95 |
| rate 0.03 | 1, 2, 3 | 1.00, 0.98, 0.97 | 1,040, 1,030, 1,450 | 936, 877, 783 | 878, 862, 792 | 2.01, 1.97, 1.81 |
| rate 0.1 | 1, 2, 3 | 0.63, 0.97, 0.99 | never, 570, 1,450 | -, 953, 941 | 763, 927, 982 | 1.74, 2.12, 2.24 |
| rate 0.3 | 1, 2, 3 | 0.96, 1.00, 1.00 | 1,100, 850, 720 | 958, 973, 985 | 962, 930, 812 | 2.20, 2.13, 1.86 |
| rate 0.1 + sparsity 0.05 | 1, 2, 3 | 1.00, 0.43, 0.95 | 1,610, never, never | 950, -, - | 901, 907, 946 | 2.06, 2.07, 2.16 |
| read-out dropout (mute 0.2, beta 4) | 1, 2, 3 | 0.84, 0.79, 0.76 | never | - | 413, 461, 436 | 0.94, 1.05, 1.00 |
| noisy synapses (sigma_w 1) | 1, 2, 3 | unfinished at update 1,200 of 2,000 | | | | |
| rescaling | 1, 2, 3 | stopped: no run files on disk (also none on Spock) | | | | |

The rate 0.1 + sparsity seed 3 ends at 0.95 but never held 5 probes at or above 0.9, so it is
"not learned" by the rule. Controls' reference: 2 learned controls, 437.5 active at the end
(largest 449), end success 0.984; over all 3 controls, 430.0 and 0.898.

**Against the pre-registered criterion** (issue #5: at the end, the arm's count exceeds every
learned control's in all three seeds AND the arm's mean end success is within 0.05 of the controls'
mean; a run that does not learn counts against the technique at that value):

| arm | mean fold | success cost | verdict |
|---|---|---|---|
| rate 0.03 | 1.93 | +0.000 | works |
| rate 0.1 | 2.04 | -0.120 | fails: success cost beyond 0.05 (seed 1 did not learn) |
| rate 0.3 | 2.06 | +0.003 | works |
| rate 0.1 + sparsity 0.05 | 2.10 | -0.193 | fails: success cost beyond 0.05 (seeds 2 and 3 did not learn) |
| read-out dropout | 1.00 | -0.190 | fails: count not above 449 in seeds 1 and 3; success cost beyond 0.05 |
| noisy synapses | - | - | no verdict (unfinished) |
| rescaling | - | - | no verdict (stopped) |

Under the all-controls success reference (0.898), the rate 0.1 arm's cost would be -0.035 and it
would pass; the other verdicts do not change. The rate penalty at 0.03 and 0.3 doubles the number
of active units (792 to 962 of 1,000 against 426 to 449 in the learned controls) at the controls'
success; read-out dropout at 0.2 leaves the count where the controls' is. Unlike 3BFF (Figure 5d),
where the count moved between the control level and N, the maze's rate-penalty runs do not fall
back: every penalised run reaches 950 or more active units within 30 updates, peaks at 984 to 1,000,
and then drifts down with swings of up to 142 units between probes (rate 0.03 seed 2); the lowest
count after update 100 is 737 (rate 0.1 seed 1), against 426 to 449 in the learned controls (panel a).

**Panel d.** No technique network had an ablation JSON when the figure was drawn; the panel shows
the learned control s1 (579 of 1,000 inactive; success under the training noise 0.90 intact, 0.89
with the inactive units removed quietest first, 0.31 in random order, 0.03 busiest first) and
says so. The ablation of the rate 0.1 seed 3 network (982 active) was started on the Mac into
`data/rl_maze_ablation_interventions/` (CPU, `rl_maze_ablate.py`, default arguments); the panel
and its printout take it up on the next run of the script.

**Checks.** `tests/test_paperstyle.py` on the PDF (Helvetica only, letters 8.5 pt, no glyph under
the minimum, every y label at least 1.36 mm from its ticks, no text within 0.3 mm of the page
edge: the first render had the lower (c) label 0.13 mm from the edge, so (c) moved 3 mm right);
`pdfimages -list` empty; the script's own text-overlap pass clean (63 artists) and no two markers
of (b) closer than one marker diameter (18 markers); `tests/test_scripts_start.py` 49 scripts;
`tests/test_no_function_imports.py` 202 files. Panel d's technique path was exercised with a
stand-in JSON (a copy of the control's pointed at the rate 0.1 seed 3 folder) in the scratchpad.
