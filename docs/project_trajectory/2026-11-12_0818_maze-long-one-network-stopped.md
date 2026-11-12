# Maze long training: one of nine networks stopped at update 5,900; its later stages cancelled

Checked on Pavel's report of a Della job with reason `DependencyNeverSatisfied`.

- **What failed.** Task 8 of stage one (job 17379992_8: N = 1,000, seed 2, updates 4,001 to 6,000) exited
  with code 1 after 4 h 48 min. The log's last line is update 1,900 of 2,000 (update 5,900 on the joined
  clock) with a KL divergence of 0.19, against about 0.008 in the lines before it. The trainer then
  stopped with `RuntimeError: the rollout was not finite 6 times in a row; the run is stopped`
  (`TrainerPPO.safe_rollout`). Up to that line the network had probe success 1.00 and 357 active units.
  Its folder `N=1000_act=relu_cont3/s2` holds the files written at the stop (progress, participation,
  parameters); nothing was deleted.
- **What was cancelled.** The three later stages of this network, which could never start
  (`scancel 17379993_8 17379994_8 17379995_8`).
- **The other eight networks** finished stage one (exit 0, about 5 h each) and are in stage two; their
  stages three and four wait on dependencies as intended. Three of them logged non-finite rollouts that
  the trainer undid by restoring weights (17379993_7: 4 events; 17379993_26: 3; 17379993_27: 1; and
  17379992_26: 2 in stage one) and continued.
- **Consequence for the reading.** The rule entry (2026-11-11_2322) does not say what to do with a
  network that stops. `maze_long_read.py` counts only networks whose record reaches update 12,000, so
  this network is not counted, and the result is reported as "eight of nine continued networks; one
  stopped with non-finite rollouts at update 5,900". The network is not rerun: repeating a stopped run
  until one finishes would select the networks that are counted. With six or more of the eight counted
  networks the two-thirds rule can still give a verdict; reading B still needs four flat networks.
- **Correction of a file name.** Entry 2026-11-11_2322 cites the Maze flat-return result entry as
  "2313"; that entry's file is `2026-11-11_2308_maze-flat-return-full-records-result.md`.
