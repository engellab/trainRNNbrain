# Maze resumed networks submitted (fixed trainer, 23 stage tasks), and a fourth network stopped

Follows `2026-11-14_1605_maze-nonfinite-recovery.md` (diagnosis, fix on `wp/maze-finite` 8c6b36e, tests).

## A fourth network stopped: N = 500 seed 3 in stage 7

Job 17546732_6 (updates 12,001-14,000) exited 1 after 1 h 6 min with `the rollout was not finite 6 times in a
row`, at stage update 400 (update 12,400), after 90 "not finite" lines with restores from the checkpoint at
updates 398-400: the pattern of the two networks whose Adam moments became NaN. Its later stages (17546733_6,
17546734_6, 17546735_6) were cancelled. Per the rule entry (2026-11-14_1107), it is resumed with the fixed
trainer from the end of its last complete stage (`_cont6`, update 12,000), stages 7-10, and counted as resumed.
Five of the six MazeLong networks continue (stage 9 running).

## The Adam-moment check (prediction written in the recovery entry before this check)

Share of NaN entries in Adam's two moment estimates, for W_inp, W_rec, W_out, in the optimizer files saved at the
stops:

| Stopped stage | Predicted | Found |
|---|---|---|
| N = 1,000 seed 6, `_cont5` (update 8,500) | all NaN | 1.0 in all six |
| N = 500 seed 2, `_cont6` (update 10,900) | all NaN | 1.0 in all six |
| N = 1,000 seed 2, `_cont3` (update 5,900; unstable snapshot, not NaN moments) | none NaN | 0.0 in all six |
| N = 500 seed 3, `_cont7` (update 12,400; predicted here from its log pattern) | all NaN | 1.0 in all six |

So the two failure modes of the recovery entry are confirmed on the cluster files: three stops came from NaN
moments, one from an unstable snapshot.

## Submission

Script `slurm/rl_maze_resume_submit_della.sh` (commits a8b86b7: index 6 added to it and to
`slurm/rl_maze_resume_della.slurm`; 425351d: the first run stopped at `conda activate` because of `set -u`,
before any submission, and left only the new worktree at the pin, which the script now reuses only at the pin).
Checkout `~/trainRNNbrain_mazefinite` at 8c6b36e, clean. Login-node launcher checks (index 8 stage 3 and index 6
stage 7, two updates each, output under /tmp): both load the right parent (update clocks 4,001-4,002 and
12,001-12,002) and finish. Outputs: `~/trainRNNbrain/data/trained_RNNs_rl_maze_long_resumed/Maze_suite/`.
Job name MazeResume, one task per stage, `aftercorr` between stages.

| Network | Index | From | Stages | Job ids |
|---|---|---|---|---|
| N = 1,000 seed 2 | 8 | `_cont`, update 4,000 | 3-10 | 17581405-17581412 |
| N = 1,000 seed 6 | 28 | `_cont4`, update 8,000 | 5-10 | 17581413-17581418 |
| N = 500 seed 2 | 5 | `_cont5`, update 10,000 | 6-10 | 17581419-17581423 |
| N = 500 seed 3 | 6 | `_cont6`, update 12,000 | 7-10 | 17581424-17581427 |

About 23 x 5 = 115 GPU-h. The longest chain (8 stages) takes about 42 h once it starts.
