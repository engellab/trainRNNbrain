# Uniform interventions, packages 2 and 3: Pavel's go-ahead and the maze submissions

**Date:** 2026-11-05 22:47
**Issues:** #6 (the matrix), dead-relu-paper#183
**Commit:** da1f467 (the maze launchers)

Pavel, 5 Nov evening, on the question of the remaining cells: "Go ahead on additional jobs". The
whole programme of docs/interventions_matrix.md is therefore authorised: package 2 (the remaining
standard settings on 3BFF and the maze, CDDM's optimiser comparison, about 550 GPU-h) and package 3
(maze code for the input cap, the recurrent-weight and metabolic penalties, the optimiser switch and
the redraw modes, about 250 GPU-h).

## Submitted now (Spock, worktree ~/trainRNNbrain_rlmaze7 at da1f467)

- Interventions launcher arms 55-72 (array 7389373, 18 jobs, N = 1,000, 3 seeds each): activity
  noise sigma_rec 0, 0.01, 0.1; spectral radius 0.5 and 1.5; the sparsity penalty alone 0.05.
  CDDM's ladder has spectral radius 2.0; on the maze an untrained network at 2.0 is not finite in
  its first rollout (local smoke test, N = 100: "the rollout was not finite" at update 1), so the
  upper rung is 1.5, which runs with a first-update KL of 1e14 at N = 100 and is left to the
  rollback guard at N = 1,000. Local smoke tests of sigma_rec 0, spectral radius 0.5 and 1.5 and
  lambda_rws 0.05 (N = 100, two updates): the saved configs record the values.
- Suite launcher indices 24-28 (array 7389374, 5 jobs): the third seed of tanh, shifted sigmoid,
  softplus and leaky ReLU at N = 1,000, and an N = 250 seed 4 (seed 2 did not learn).

## Delegated, running in parallel

- 3BFF package (Della, about 56 runs): activity noise, spectral radius and lambda_rwm (with the
  CDDM third seeds of both), the metabolic penalty, sparsity alone, the four activations on the
  bias-0 network, the input cap 0.3; launchers get a TASK switch as b0_wd did.
- CDDM package (Della, 21 to 24 runs): AdamW and SGD at four rates, rate penalty 0.1 alone and
  with sparsity 0.05, and the threshold lift only if it can act on a fixed-zero bias.
- Maze code (package 3): the cap, lambda_rwm, lambda_met, the optimiser switch, the redraw modes,
  each mirrored from Trainer.py with tests, then 14 launcher arms (42 runs) to submit.

Read-outs are pre-registered in each package's own entry. Already in the queue from package 1:
the maze weight-decay, input-norm and prune-and-duplicate arms (7389335), 3BFF weight decay and
CDDM rescaling (Della 16955456, 16955457).
