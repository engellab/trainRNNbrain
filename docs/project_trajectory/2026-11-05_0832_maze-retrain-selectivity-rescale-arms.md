# Maze: prune-and-retrain and selectivity for Figure 1, and two gentler rescaling arms, submitted

**Date:** 2026-11-05 08:32
**Issues:** #3 (Figure 1's maze panels), #5 (interventions)
**Commit:** 4c65946

Figure 1 reads "no data" for the maze in the retrain strip of panel d and in the selectivity
column of panel e. Both analyses now exist for the maze and the figure reads their files when they
land. Written before any result was read.

## Prune-and-retrain (Figure 1d strip): `rl_maze_retrain.py`, Spock array 7384815 (9 jobs)

The maze counterpart of the supervised retrain (ablate_silent_units.py): for each of the three
N = 1,000 ReLU agents of the uniform series, three arms trained 200 further PPO updates with the
run's saved trainer settings (a fresh Adam at the size-rule rate, the 50-update warm-up ramp):
intact (nothing removed), silent_removed (every inactive unit's outgoing columns of W_rec and W_out
zeroed and held at zero by a gradient hook; the script checks after training that they are still
exactly zero), random_removed (as many units at random, the draw seeded from the run's label). The
score is the success rate under the training noise with the mean action over 256 fixed drops and
4 seeded draws (the ablation's score), at update 0 and every 25 updates; the clean probe's success
and active count are recorded beside it. 200 updates is a tenth of the 2,000-update run; the
supervised 3,000 iterations are 5 to 15% of a run's training to its floor crossing.

Local smoke test (seed 2, CPU, 16 drops): 555 of 1,000 units removed, score 0.75 straight after
the removal, 0.875 after two updates; random removal of 555 units, score 0.19, the hook check
passed.

**Pre-registered pass rule**, the maze form of claim C1's: in every learned network (seeds 1 and
2; seed 3 ended at 0.73 and is reported, not scored), the agent retrained without its inactive
units ends within 0.05 of the intact agent retrained the same way (the sampling error of a success
rate over 1,024 episodes is about 0.015). The random-removal arm is reported beside it with no
rule. A failure, both learned networks ending more than 0.05 below their intact controls, would
mean the inactive units of the maze agents do work the ablation does not see, and the maze would
not support C1.

## Selectivity (Figure 1e): `rl_maze_population_stats.py`, running on the Mac

The S1 to S4 rule of population_stats.py on the three N = 1,000 agents at their final weights, the
responses from the training's clean probe (128 fixed drops, noise off, mean action). The samples
of S3 and S4 are the decisions before the cheese; the five task variables, fixed before any value
was computed, are the turn and speed commands executed at that decision, the geodesic distance
from the agent's cell to the cheese in cells, and the heading's cosine and sine: the maze's
counterpart of the flip-flop's bits, CDDM's context, coherences and choice, and DMTS's sample,
test and match (the task's outputs and the agent's position). The file is
data/population_stats_maze.json, read beside data/population_stats.json by fig_paper_F1 only;
fig_ed_population.py keeps its three supervised tasks. Fig. 1e draws the maze with open markers:
its networks are read at the end of a fixed budget, the supervised ones at their floor-crossing
snapshots (dead-relu-paper#83). The retrain strip's maze markers are open for the same reason.

Descriptive, as C8 is: the fraction selective over all units and over the active units, and the
number of inactive units counted selective (119 to 150 of 1,000 on CDDM, 0 to 1 on 3BFF).

## Two gentler rescaling arms: interventions launcher indices 25-30, Spock array 7384816 (6 jobs)

The paper's rescaling arm blew up on the maze at the end of the warm-up ramp in all three seeds
(updates 52-60; issue #5). Two variants, 3 seeds each, N = 1,000, every other key the paper's:
- `rescale_mild`: the supervised defaults for the tilt and the cap, alpha 1.0005 per optimizer
  step and cap 8 (an episode ends after 4,159 steps, 520 updates, instead of 2,083 steps at alpha
  1.002 and cap 64).
- `rescale_few`: rescale_protect_frac 0.05 instead of 0.20, at most 50 units growing at once.
Local smoke test of the overrides at N = 100: three updates, 20 units growing, the saved config
records alpha 1.0005 and cap 8. Read-out as issue #5's: "works" = more active units than every
learned control (449 is the largest) in all three seeds at an end success within 0.05 of the
learned controls' mean (0.984).

## Also

- Della job 16907630 (N = 4,000 seed 1 continued from update 1,000) is running on the
  `~/trainRNNbrain_rlmaze4` worktree (2afc209); the Spock jobs above run on `~/trainRNNbrain_rlmaze5`
  (4c65946).
- Pavel (5 Nov): report the interventions that work on every task first, then the task-specific
  ones, so the same settings and rules (weight decay, input-weight norm, noisy synapses, dropout,
  and the rest) are to be tested on all three tasks. An inventory of the (task x setting/rule)
  cells that exist and are missing is being written to docs/interventions_matrix.md and a new
  issue.
