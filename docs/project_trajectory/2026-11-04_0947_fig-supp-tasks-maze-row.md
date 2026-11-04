## 2026-11-04 09:47 — Supplementary Fig. 1 gains row d, the maze (issue #3)

**What changed.** `fig_supp_tasks.py` draws four rows: 3BFF, CDDM, DMTS and the maze. Row d's
pictogram is the maze of the run's own config (walls as filled cells, the cheese as a star), one
agent at its drop with its heading tick and its 8 range rays drawn to the walls, and the path of one
clean episode in the task colour; beside it a key names the four glyphs. The channel stack shows that
episode's inputs (8 range sensors in ink, the two efference-copy inputs in grey) and the two executed
outputs, turn / turn_max and speed / v_max, in the task colour; one decision is 0.5 tau (5 RNN steps
at alpha 0.1); a dotted line marks the episode's end ("cheese reached").

**Which episode.** The first seed under `MAZE_RUN` (`data/trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu`,
the final design) is rebuilt with `trainRNNbrain.rl.saved.load_run` and played noise-free with the
policy mean on the training probe's drops (`ppo.probe_episodes` from `ppo.probe_seed`); of the
episodes that reach the cheese, the one of median length is drawn. No final run is on the Mac yet, so
the committed figure shows the pending fallback: `configs/rl/maze_ppo.yaml` composed, an untrained
network played for 60 decisions, the row labelled "config not yet trained". The drawing was developed
and checked on the older run `Maze_second/N=1000_cue=false/s2` (11 x 11 maze, 8 inputs, 135
decisions to the cheese): `img/internal_figures/fig_supp_tasks_oldrun_preview.png`. Once the suite's
run folder is copied in, rerunning the script draws the final design unchanged.

**The hue.** `paperstyle.TASK_MAZE = #941862` (raspberry; OKLCH L 0.45, C 0.17, hue 350), chosen by
sweeping 479 OKLCH candidates through the dataviz validator against the three task colours: every
candidate that passes lies between hue 280 and 10, the purple-to-magenta range, so no hue clears
every check against the rules as well. Against the tasks: worst colour-blind dE 13.1 (DMTS), worst
normal-vision 19.7 (CDDM), contrast 7.97:1 on the paper surface. Against the rules it fails only the
magentas in normal vision (duplicate 10.5, rescale 11.0) and the dark setting teals under simulation
(SETTING_RAMP[2] 3.3, [3] 5.5); tasks never share a panel with rules or with the setting teals, so
the limitations list in paperstyle.py records these numbers.

**The height.** 165.1 x 176 mm, 35 mm over MAX_H_MM (141 mm). Four rows cannot fit 141 mm: row d's
12 channels need about 2.5 mm each for their tick labels and the three supervised pictograms lose
their text clearances below 26 mm per row. The paper's main.tex has no height-cap macro; at
width=\textwidth the figure plus its caption (about 46 mm) fit one float page. Text overlaps checked
with the script's own window-extent pass (none) and the PDF audit (tests/test_paperstyle.py: one
family, no glyph under 5 pt, no text within 0.3 mm of the page edge). The DMTS boxes and the CDDM
cue boxes moved by 0.01-0.05 of their panel so that no line touches at the new row height.
