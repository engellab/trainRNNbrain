## 2026-11-05 08:48 — Maze interventions figure, panel d filled: the ablation of the rate 0.1 seed 3 network (issue #5)

Follows `2026-11-05_0812_fig-ed-maze-interventions.md`, whose panel d had no technique network.
The ablation started there finished at 08:48 (48 minutes on the Mac CPU): `rl_maze_ablate.py
data/trained_RNNs_rl_maze/Maze_interventions/frm0.1/s3 --out data/rl_maze_ablation_interventions`,
default arguments (256 scoring drops, 4 noise draws, 128 probe drops, 5 random orders), output
`data/rl_maze_ablation_interventions/frm0.1__s3.json`. `fig_ed_maze_interventions.py` re-run
unchanged; `img/internal_figures/fig_ed_maze_interventions.{pdf,svg}` re-drawn, panel d now shows
this network (rate violet) over the learned control s1 (grey).

**Numbers (success rate under the training noise; the control's for comparison).** Rate 0.1
seed 3: 982 of 1,000 active (18 inactive); intact 0.94 (clean probe 0.99). Units removed quietest
first: 0.94 with 18 (the inactive units), 0.94 with 116, 0.94 with 264, 0.70 with 509, 0.02 with
754. Random order (mean over 5 orders): 0.94 with 18, 0.78 with 116, 0.34 with 264, 0.09 with 509.
Busiest first: 0.87 with 18, 0.25 with 116, 0.02 with 264. Its "added" units number 544 (982
minus the learned controls' mean 437.5); the grid point nearest 18 + 544 = 562 removed is 509,
where quietest-first success is 0.70. Control s1: 421 of 1,000 active (579 inactive); intact 0.90;
with the 579 inactive units removed quietest first 0.89, in random order 0.31, busiest first 0.03.

**Reading.** The penalised network's output depends on far more of its units than the control's:
removing the 509 quietest units (all but 18 of them active) lowers its success from 0.94 to 0.70,
and in random order its success falls faster than the control's (0.34 against 0.54 at 264 units
removed, 0.09 against 0.35 at 509; the control's values interpolated between its grid points 145,
290, 434 and 579). The control keeps 0.89 with its 579 inactive units removed and loses
performance only once active units go (0.81 at 684, 0.47 at 790). So the units the rate penalty adds
on the maze carry task performance, as the added units do on 3BFF (Figure 5e). One network, one
arm; the other finished technique runs have no ablation yet.

**Checks.** `tests/test_paperstyle.py` on the PDF ok (one family, letters 8.5 pt, y-label gap
1.36 mm, no text near the page edge); `pdfimages -list` empty; the script's text-overlap pass
clean (62 artists); no two panel-b markers closer than one marker diameter. Preview
`img/internal_figures/fig_ed_maze_interventions_preview.png` (not tracked).
