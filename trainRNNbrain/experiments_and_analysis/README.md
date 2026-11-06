# Analysis scripts: conventions

Every script here answers one question about trained networks and writes its figures to
`img/internal_figures/` (manuscript figures through `paperstyle.save`, PDF + SVG). How networks are
read (the noisy-loss floor, active units, held-out R²) is in `docs/AGENT_GUIDE.md`; this file is
about how the code is written.

## The live pipeline

| module | role |
|---|---|
| [`common.py`](common.py) | shared primitives: participation and the active-unit rule (`active_mask`, `active_count`), the read-out (`noisy_loss_series`, `noisy_floor`, `warm_start_chain`; one run at its floor: `read_run`; a cell: `runs_with_loss` + `at_own_floor`; `snapshot_at`, `r2_at_floor_estimate`), fitting, sweep loading. No matplotlib, no side effects at import. `python common.py` runs its self-check |
| [`heldout_r2.py`](heldout_r2.py) | the performance measure (seeded held-out R²; run on CPU); R² at the floor (`floor_r2`) and a cell read at the floor with it (`read_rung`) |
| [`ablate_silent_units.py`](ablate_silent_units.py) | claim 1: ablation and retrain at the floor |
| [`gradient_participation.py`](gradient_participation.py) | claim 4: gradient reach; the untrained twin of a run |
| [`fig_paper_F1.py`](fig_paper_F1.py) | Figure 1 |
| [`fig_paper_F2_fails.py`](fig_paper_F2_fails.py), [`fig_knobs_at_floor.py`](fig_knobs_at_floor.py) | Figure 2 (what fails): knob ladders and revival screens at the floor |
| [`fig_paper_F2.py`](fig_paper_F2.py), [`f2_remedies_cache.py`](f2_remedies_cache.py) | the remedies figure and its cache (becomes Figure 3) |
| [`read_uniform_cells.py`](read_uniform_cells.py) | issue #6: the three cell groups no paper script had read (3BFF input-weight cap 0.1, 0.5, 2.0; CDDM noisy synapses sigma_w 1.0; CDDM read-out dropout and prune-and-duplicate), each read at 75,000 iterations against its N = 1,000 control with `fig_paper_F2_fails.read_fixed` and judged by the matrix file's criterion; writes `data/read_uniform_cells.json` |
| `fig_paper_F3.py` … `fig_paper_F5.py`, `fig_supp_*.py` | figures the paper's `main` still includes |
| [`rl_maze_readout.py`](rl_maze_readout.py), [`rl_maze_ablate.py`](rl_maze_ablate.py), [`rl_maze_animate.py`](rl_maze_animate.py), [`fig_ed_maze.py`](fig_ed_maze.py) | the maze (reinforcement learning, issue #3): read-out of the run folders (`common.maze_load_runs`, `common.maze_learned_index`), ablation under the training noise, a GIF of a trained agent, the Extended Data figure. A trained agent is rebuilt with `trainRNNbrain.trainer.TrainerPPO.load_run` (task, RNN, trainer), which reads the layout `training/run_rl.py` writes and the earlier one of the deleted `trainRNNbrain/rl/` |
| [`fig_ed_maze_interventions.py`](fig_ed_maze_interventions.py), [`read_maze_arms.py`](read_maze_arms.py), [`fig_ed_maze_settings.py`](fig_ed_maze_settings.py) | the maze arms of issues #5 and #6 (`Maze_interventions/<arm>/s<seed>`): every arm derived from its saved config (`arm_of`), the rules figure (Figure 5's per-unit rules on the maze), the read-out of all 37 arms with the pre-registered verdict of each class (rule: works/fails; setting: increases/decreases/no change in the count), written to `data/read_maze_arms.json`, and the settings figure (Figure 3a on the maze) |
| [`fig_paper_F5_tasks.py`](fig_paper_F5_tasks.py), [`fig_paper_F3_tasks.py`](fig_paper_F3_tasks.py) | the cross-task rebuild of Figures 5 and 3 (dead-relu-paper#183): the training techniques and the standard settings on 3BFF, CDDM and the maze at N = 1,000, drawn from the read-out files only (the remedies catalogue, `data/read_uniform_cells*.json`, `data/fig_paper_F2_fails_cache.pkl`, `data/read_maze_arms.json`, the maze ablation JSONs); one open marker per group of seeds at their mean with a bar over their range; the verdict (technique x task) and the direction (setting value x task) computed by the pre-registered rules and checked against the files' own; a cell still training reads "unfinished" and is picked up on a rerun |
| [`paperstyle.py`](paperstyle.py) | how manuscript figures look; [`plotstyle.py`](plotstyle.py) is the deck's equivalent |
| `fig_slides.py`, `deck_pngs.py`, `deck_html.py`, `check_presentation.py` | the talk deck, `docs/presentation.md` |

`common.noisy_floor` is the one read-out rule, and `common.read_run` the one way to read a run at
it (active count at the probe nearest t*, the kept snapshot there, the inputs of the R² estimate).
`common.active_mask` is the one active-unit rule; do not retype `p >= 0.05 * q95(p)`. Some deck and older paper-figure scripts still carry
their own read-outs (`pr_matrix.fit_floor_at`, fixed iterations); do not use those for new numbers.

## The rules

1. **Import shared primitives from `common`, never from a sibling script.** Divergent copies of an
   analysis primitive are how a project reports two numbers for one quantity.
2. **Keep your own domain loader** when the folder layout is specific to your sweep. Only shared
   things belong in `common`.
3. **Guard execution** behind `if __name__ == "__main__": main()`. No work at import time.
4. **Module docstring states the question, the method, and the output path.** Where a choice could
   have gone another way (fit range, read-out point, criterion), say why in the docstring.
5. **Every function gets a docstring** with inputs (types, shapes, units) and return value.
6. **Seed every random draw** with `np.random.default_rng(seed)` or a seeded torch generator, never
   the global state.
7. **One variable, one visual channel, across every figure.** N is colour from a fixed map (N = 1000
   is the same colour everywhere), k is viridis (dark = low k), seeds are error bands rather than a
   channel of their own.

## Standing analysis rules

Each exists because ignoring it produced a wrong result in this project.

- **Never pool a cell across iteration budgets.** Use `common.keep_longest_budget`.
- **A penalised run's noisy loss includes λ·penalty**, so its floor is task plus penalty. Compare
  performance across penalty conditions with held-out R², never with the training loss.
- **Never average a quantity over "whatever sizes this cell happens to have".** Use one size and
  say which, or restrict to complete cells.
- `stretched`'s `A` and `tau` are not individually identified once `tau` falls below the fit's start
  iteration. Compare fits through the floor, never `A` or `tau`.
- `stable_crossing` takes `window` and `base` explicitly; pass both.
