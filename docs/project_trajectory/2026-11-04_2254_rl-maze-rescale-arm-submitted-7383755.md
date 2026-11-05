## 2026-11-04 22:54 — Maze interventions, stage 2: the rescaling rule merged (1033e5c) and its arm submitted, Spock array 7383755 (issue #5)

Follows `2026-11-04_2233_maze-rescale-once-per-gradient-step.md` (the port, by an independent
agent) and the ladder entry of 22:45.

**What was merged.** `trainer.prune_reinit` + `prune_args` in `TrainerPPO`: the supervised
three-state row-tilt rule (`Trainer.prune_and_reinit_`, mode rescale) run once per optimizer
step on the recompute's states of that minibatch (8 checks per update, 16,000 per run; a growth
episode reaches the cap of 64 after 2,083 steps), with the paper's parameters (alpha 1.002, target
10 x the rate cap, cap 64, protect 0.2 N, active sources only, normalised rows, check every step,
patience 1). With the option off the runs are bit-identical to the series' code. Reviewed by an
independent agent on issue #5: a bitwise oracle drove the supervised rule and the port side by
side through five parameter sets, every weight, moment and state tensor identical at every call;
approve. `tests/test_rl_maze.py` 23 of 23.

**Pilot** (20 updates, N = 1,000, Spock job 7383750): 11 s per update, the same as the control;
200 units (the 0.2 N ceiling) opened at the first check, growing throughout; no error.

**Submitted.** `REPO=$HOME/trainRNNbrain_rlmaze3 sbatch --array=22-24 --time=12:00:00
slurm/rl_maze_interventions_spock.slurm` from `~/trainRNNbrain_rlmaze3` at 1033e5c (clean):
array 7383755, the `rescale` arm, N = 1,000, seeds 1-3, about 20 GPU-h; outputs
`~/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_interventions/rescale/s<seed>/` on Spock. Same
pre-registration and criterion as the ladder of 22:45. Prune-and-duplicate is not built.
