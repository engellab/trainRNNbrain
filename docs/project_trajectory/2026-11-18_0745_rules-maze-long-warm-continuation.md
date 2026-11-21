# Rules, written before any run: do Maze networks lose active units while the return is flat, with no optimizer restarts and the supervised decrease bar? (Phase B)

Follows `2026-11-17_2249_rules-maze-warm-start-test.md` (Phase A, the warm-start test, submitted 18 Nov 07:27
Pacific, entry `2026-11-18_0727_maze-warm-start-test-submitted.md`). This entry is written while Phase A is still
queued: no record of Phase A, and no record after update 20,000 of any Maze network, has been seen. Phase B is
submitted only if Phase A's R1 (the warm start is not disruptive) and R4 (no loss of performance after a warm
start) pass, as Phase A's rules require; if either fails, nothing below runs, and the cause of the disruption is
looked for first.

## What is new here, and what was chosen after seeing data

- **The decrease bar is the supervised one, in the supervised unit: 5 active units lost per 100,000 Adam steps.**
  One PPO update is 8 Adam steps (4 epochs x 2 minibatches), so the bar is 0.4 units per 1,000 updates. The
  earlier Maze bar was 5 units per 1,000 updates (62.5 per 100,000 Adam steps, 12.5 times stricter per step).
  **This matched bar was chosen after seeing the earlier Maze data** (entry
  `2026-11-17_2226_maze-flat-return-threshold-units-diagnostic.md`: at the matched bar, 5 of the 7 flat-return
  networks of the update-20,000 reading would count as decreasing). The runs below are fresh observations: the
  window (updates 24,001-34,000) has not been trained yet. They are the same nine networks as before, so the
  observations are new in time, not new networks.
- **The flat criterion is matched as well: the return changes by less than 3% of its rise per 100,000 Adam steps.**
  The supervised flat-loss test calls a loss flat if it changes by less than 3% per 100,000 iterations. The earlier
  Maze criterion (3% of the rise per 1,000 updates) is 12.5 times looser per Adam step and, over this 10,000-update
  window, would call a return flat that rose by 30% of its rise. The slope is a least-squares line through every
  update's return rather than the supervised test's two block medians. Reason (a design check, made on the
  returns of updates 15,000-20,000, already seen, and on no active-unit count): around a linear trend, medians of
  625-update blocks (5,000 Adam steps, the supervised block) scatter by 1.5-7% of the rise between blocks, so two
  block medians cannot resolve 3% per 100,000 steps; a line through 10,000 updates has an expected standard error
  of about 1.6-7.6% of the rise per 100,000 Adam steps, smallest in the N = 500 networks. A network with a truly
  flat return can therefore still miss the bar in the noisiest networks (N = 1,000 seed 1 most of all); the
  earlier criterion is kept as a secondary reading (P2).
- **No optimizer restarts.** Every stage boundary is a warm start (`paths.init_adam=true`): the parent's Adam
  moments and step count are loaded and the 50-update warm-up is not repeated. The last restart these networks
  had is at update 18,000 (stage 10, cold), 6,000 updates before the window starts.

## Runs

The nine networks of the warm-start test (N = 500 seeds 1-3; N = 1,000 seeds 1-6), continued from the end of
their warm arm, warm stage 12 (update 24,000, in `~/trainRNNbrain/data/trained_RNNs_rl_maze_warmtest`):

- stage 13: updates 24,001-29,000, warm from warm stage 12;
- stage 14: updates 29,001-34,000, warm from stage 13.

Launcher `slurm/rl_maze_warmlong_della.slurm` (`slurm/rl_maze_warmtest_della.slurm` with the changes its header
lists), trainer pinned to `wp/maze-finite` 8c6b36e (checkout `~/trainRNNbrain_mazefinite`, the trainer of the
warm-start test), new output tree `~/trainRNNbrain/data/trained_RNNs_rl_maze_warmlong`, cells
`N=<N>_act=relu_warm_cont<stage>`. Nothing existing is written. 18 stage tasks of 5,000 updates, `--time=23:00:00`;
at the 5.0-7.1 h per 2,000 updates of the earlier stages, a stage takes 12.5-17.7 h, so about 225-320 GPU-h,
and with Phase A (about 135-190 GPU-h) 360-510 GPU-h in all, within the 600 GPU-h budget.

**Recovery.** A stage that stops early (wall clock, node failure) is continued warm from its own last checkpoint
(the run folder is rewritten every 100 updates, Adam state included) to the stage's end, into the cell
`..._warm_cont<stage>r` (`RECOVER=1`); the next stage then continues the recovery cell. The partial folder is kept.
`common.maze_load_runs` joins the chain into one record. Each recovery gets its own notebook entry. A stage that
stops on non-finite rollouts is handled inside the trainer by its fallback restores; if it stops anyway, it is
recovered the same way and the restores are listed.

## Readings (thresholds fixed here)

Window W: updates 24,001-34,000 (10,000 updates, 80,000 Adam steps; the supervised flat-loss windows run from
iteration 150,000 to the end of 200,000- or 300,000-iteration runs, 50,000-150,000 Adam steps). Probes every 10
updates, plus the first update of each stage.

A network is read if its joined record reaches update 34,000 and its last probe success is at least 0.9; a
network that is not read is counted and named. n = 9 if all are read.

Per network:
- **Decrease** U: minus the least-squares slope of the noise-free probe's active count against update, over the
  probes in W, in units per 100,000 Adam steps (slope per update x 100,000 / 8). **Decreasing** if U >= 5.
- **Return slope** F: the least-squares slope of the per-update training return (`discounted_return` of
  Progress.json) against update over the updates in W, per 100,000 Adam steps, divided by the network's rise. The
  rise is as in `maze_flat_return_full.measure`: the median return over the 100 updates up to the learned point
  minus the median over the first 100 updates of training. **Flat** if |F| < 3%.

Verdicts (the two-thirds rule of the earlier flat-loss and flat-return tests, `round18_maze.three_way`):
- **P1 (primary): decrease while the return is flat, matched bars.** Needs at least 5 flat networks, else "no
  verdict". "Decreasing while the return is flat" if at least two thirds of the flat networks are decreasing; "not
  decreasing" if at least two thirds are not; "mixed" otherwise.
- **P2 (secondary): the earlier Maze flat criterion.** Flat by `maze_flat_return_full.measure` over W (window start
  24,001; medians of the first and last 100 updates; |D| < 3% of the rise per 1,000 updates); decreasing as in P1;
  same rule, at least 5 flat networks.
- **P3 (secondary): the earlier Maze decrease bar.** P1's flat networks, decreasing if at least 5 units are lost
  per 1,000 updates (62.5 per 100,000 Adam steps); same rule.
- **P4 (secondary): stage 14 alone** (updates 29,001-34,000, 11,000-16,000 updates after the last restart): P1's
  quantities and rule over that half window, to check that a slow recovery from the restart at update 18,000 does
  not carry the decrease.
- **P5 (descriptive): the warm boundaries.** The jumps J(24,000) and J(29,000) (Phase A's definition: median active
  count of the first 5 probes after the boundary minus that of the last 5 up to it) against each network's noise
  from Phase A (the largest |J| over updates 18,250, 18,500, ..., 19,750). If |J| exceeds the noise at either
  boundary in more than 2 of the 9 networks, the result entry says the window contains a disruption, next to P1.

Every per-network value is reported: active counts at the start and end of W, U, F, both flat calls, the
probe success at the end.

**Timing.** The readings are taken when every stage-14 task has ended, or at 22:00 Eastern on Friday 20 Nov,
whichever comes first. If at that time fewer than 7 of the 9 networks reach update 34,000, every reading uses
W' = updates 24,001-29,000 instead (stage 13 alone; 5,000 updates, 40,000 Adam steps, shorter than any supervised
window), with every bar unchanged and P4 dropped, and the result entry says so.

**Consequence for the paper's claim** (that the number of active units keeps decreasing while the loss or return
is flat), decided here: P1 "decreasing while the return is flat" means the claim holds on Maze at the supervised
bar, with the bar marked as chosen after the earlier Maze data; P1 "not decreasing" means Maze contradicts the
claim even without optimizer restarts; "mixed" or "no verdict" means Maze neither supports nor contradicts it,
and the claim stays scoped to the supervised tasks.

The reading script (`maze_warm_long_read.py`) is committed before any record of Phase B is pulled.
