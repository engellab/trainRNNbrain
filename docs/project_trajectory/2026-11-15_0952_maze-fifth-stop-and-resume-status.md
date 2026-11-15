# Maze long training: a fifth network stopped (N = 500 seed 1, update 18,300) and is resumed; the fixed trainer recovers in the resumed runs

Follows `2026-11-15_*_maze-resume-submitted.md`.

## The original continuation to update 20,000 (MazeLong, arrays 17546732-17546735)

Four of six networks finished all four stages (N = 1,000 seeds 1, 3, 4, 5; exit 0, last stage ended 10:16-10:41
Eastern, 15 Nov). N = 500 seed 1 stopped in stage 10 (job 17546735_4) after 47 min, at stage update 300 (update
18,300), with 35 "not finite" lines and restores from the checkpoint before the stop, the pattern of the NaN-moment
stops. Its stage 9 (`_cont9`, update 18,000) is complete. Per the rule entry (2026-11-14_1107) it is resumed with the
fixed trainer: launcher index 4 added (commit 4ef9e53), stage 10 only, job **17622458** (MazeResume). Login-node check
before submission: the launcher loads `_cont9` (update clock 18,001-18,002) and finishes.

So five of the nine networks in this package stopped at least once with the 1a80f5e trainer: the two N = 1,000 networks
and N = 500 seed 2 before update 12,000, and N = 500 seeds 1 and 3 after it. Four never stopped (N = 1,000 seeds 1, 3, 4, 5).

## The resumed networks (MazeResume): the fix works on real runs

Up to 12:50 Eastern 15 Nov, eight resumed stages have completed and four are running; none failed. Recovery events in the
logs (each event: three non-finite rollouts, then a verified snapshot with its optimizer state restored, one snapshot
tried each time):

| Stage log | Non-finite rollouts logged | Snapshot restores |
|---|---|---|
| 17581405_8 (N = 1,000 seed 2, stage 3) | 1 | 1 |
| 17581407_8 (stage 5, running) | 1 | 1 |
| 17581413_28 (N = 1,000 seed 6, stage 5) | 3 | 3 |
| 17581414_28 (stage 6, running) | 5 | 1 |
| 17581419_5 (N = 500 seed 2, stage 6) | 2 | 1 |
| 17581424_6 (N = 500 seed 3, stage 7) | 3 | 1 |

Each of these would have been a stop or a NaN cascade with the old trainer when the restores failed; with the new
trainer every run continued with probe success 1.00. The exact counts per update are in each run's progress record
(`restored`, `snapshot_restores`, `skipped_grad`) and will be reported with the result.
