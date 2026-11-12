# Rules, written before the data: (1) Maze networks trained to 12,000 updates; (2) the removal-and-retraining result read at 250 iterations

Pavel, 11 Nov (late): show on the Maze too that the number of active units keeps decreasing as training
progresses; and the 3,000 further iterations after the removal of the inactive units are long, about 300
would make the point better.

## 1. Maze: continue the default networks to 12,000 updates

**Why new training.** `2026-11-11_2313_maze-flat-return-full-records-result.md` (the entry of this evening):
at 4,000 updates no Maze network has a flat return, so the flat-return test cannot be decided on the
existing records.

**Networks.** The six default ReLU networks of N = 1,000 units of the Maze size series (seeds 1-6), each
continued from its weights at update 4,000 for four more stages of 2,000 updates (to 12,000), by the stage
protocol of `2026-11-07_1008_round19-train-all-rules.md`: the same code version, trainer version, learning
rate (constant, with the 50-update warm-up at each stage start) and configuration as their earlier stages.
The learning rate is not decayed: a decayed rate would flatten the return and stop the weights together,
and the question is whether units become inactive while the weights are still trained at the rate used
throughout. If the cost allows (cap about 200 GPU-h for the package; N = 1,000 is 5.5 GPU-h per stage, so
132 GPU-h), the three N = 500 networks (seeds 1-3) are continued the same way.

**Reading A, the decrease as training progresses (needs no flat return).** Per network, the least-squares
slope of the noise-free probe's number of active units over updates 4,000 to 12,000. A network
"decreases" if it loses at least 5 active units per 1,000 updates over that range AND has fewer active
units at update 12,000 than at update 4,000 (medians over the last 10 probes before each). Verdict:
"keeps decreasing" if at least two thirds of the continued networks decrease; "does not" if at least two
thirds do not; otherwise mixed. A network whose last probe is below 0.9 is reported and left out.

**Reading B, the decrease under a flat return (the G8 test, window fixed now).** Window: updates 9,000 to
12,000, the last quarter. Return change D and rise R as in `maze_flat_return_full.py`; flat if
|D| < 0.03 R per 1,000 updates; decreasing if at least 5 active units are lost per 1,000 updates in the
window. Verdict over the flat networks by the two-thirds rule; fewer than 4 flat networks (of up to 9):
"return still not flat at 12,000 updates", no verdict.

**What each outcome does to the paper.** A "keeps decreasing": the paper says that on all three tasks the
number of active units decreases as training progresses, with the Maze counts at 4,000 and 12,000. B
"decreasing while flat": the after-convergence sentence covers three tasks. B "not decreasing": the paper
states that in Maze networks the decrease stops once the return is flat. B without a verdict: the
after-convergence sentence keeps its two tasks.

**What would prove the claim wrong.** Reading A: slopes within 5 units per 1,000 updates of zero, or
rising, in more than a third of the networks: the Maze counts level off. This outcome is credible: at
N = 1,000 one of six networks already gained units over updates 3,000 to 4,000.

## 2. Removal and retraining read at 250 iterations (no new training)

The retraining of Fig. 1d (`ablate_silent_units.py retrain`, Della `c1_retrain_79932ef`) recorded held-out
R^2 every 250 iterations of the 3,000. Only the values at 0 and 3,000 have been read. 250 iterations is
0.4 to 1% of the 24,000 to 67,000 iterations before the removal. The Maze retraining
(`rl_maze_retrain.py`, `data/rl_maze_retrain`) recorded the success rate every 25 updates of the 200; 25
updates is about 1% of the 2,000 before the removal.

- **Dispensable (the claim).** At 250 iterations the R^2 of the network without its inactive units is
  within 0.01 of the intact network trained the same 250 iterations, in each of the six 3BFF and CDDM
  networks (the bar of rule C1, `2026-10-31_1702`). Maze, at 25 updates: success rate within 0.05 of the
  intact network's, in both networks (the Maze tolerance of the technique criterion). If a network fails,
  the paper keeps 3,000 iterations (200 updates) and says how fast the recovery is.
- **Specific to inactive units.** At the same point the network with as many random units removed is
  below the network without its inactive units by more than 0.01 in R^2 (0.05 in success rate). Passes if
  this holds in at least 5 of the 6 supervised networks. If it passes, the paper says that random units
  are not dispensable in that time; if not, it says the two removals are separated only before further
  training, as now.
- The DMTS networks are read too and reported in the notebook only.
