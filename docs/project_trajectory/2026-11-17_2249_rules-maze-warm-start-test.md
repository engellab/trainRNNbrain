# Rules, written before any run: does a warm start (Adam state reloaded) remove the disruption at Maze stage boundaries?

Pavel (17 Nov): every Maze continuation so far restarted Adam (`INIT_ADAM=false`: fresh moments, 50-update warm-up),
and units reactivate at each restart (entry `2026-11-17_2226_maze-flat-return-threshold-units-diagnostic.md` and the
20,000 result entry). Test warm starts so that they are not disruptive; afterwards the Maze experiments will probably
be repeated with warm starts. Nothing below has been run.

## Runs

The nine networks at update 20,000 (N = 1,000 seeds 1-6, N = 500 seeds 1-3; parents `..._cont10` in the long tree for
the four never stopped, `..._resume_cont10` in the resumed tree for the five resumed). Trainer: `wp/maze-finite`
8c6b36e (identical to 1a80f5e without a non-finite rollout), checkout `~/trainRNNbrain_mazefinite`. New output tree;
nothing existing is written.

- **warm**: `INIT_ADAM=true` (the parent's Adam state, including its step count, is loaded; no warm-up). Two stages:
  updates 20,001-22,000 from the parent, and 22,001-24,000 warm-started from the warm stage before it.
- **cold**: `INIT_ADAM=false`, the protocol used so far. One stage, 20,001-22,000, from the same parent.

About 27 stages x 5-7 GPU-h, about 160 GPU-h.

## Readings (thresholds fixed here)

Probes are every 10 updates. For a boundary at update U, the **jump** J(U) is the median active count of the first 5
probes after U minus the median of the last 5 probes up to U. The **noise** of a network is the largest |J| over the
non-boundary points 18,250, 18,500, ..., 19,750 of its parent stage (same definition, seven points).

- **R1, the warm start is not disruptive** if |J(20,000)| of the warm arm is at most the network's noise in at least
  7 of 9 networks, AND the same holds for J(22,000) of the warm arm in at least 7 of 9.
- **R2, the cold restart is disruptive (replication)** if J(20,000) of the cold arm exceeds the network's noise in at
  least 5 of 9 networks.
- **R3, paired**: J(20,000) warm < J(20,000) cold in at least 7 of 9 networks.
- **R4, performance**: the warm arm keeps probe success at or above 0.9 at every probe of updates 20,001-20,500 in
  every network (a warm start must not disrupt behaviour either).
- **R5, the decrease without restarts (descriptive, no verdict on the paper from this alone)**: in the warm arm over
  updates 20,001-24,000 (one boundary, warm), units lost per 100,000 Adam steps (one update = 8 Adam steps) from a
  least-squares line, against the supervised bar of 5 per 100,000 Adam steps; and the flat-return test of
  maze_flat_return_full.measure over the same window with the 3% bar. Reported per network.

Warm starts are adopted for every later Maze continuation if R1 and R4 pass. If R1 fails, the warm start is itself
disruptive and the cause is found before anything is repeated.

## Before submission

A local CPU check at small N that a run split into two halves with `INIT_ADAM=true` continues the optimizer state
exactly: the Adam state loaded at the start of the second half equals the state saved at the end of the first (every
tensor and the step count), the learning rate after the split is the constant rate (no ramp), and the update clock
continues. Differences in the episodes drawn after the split (random number streams) are expected and are listed.
