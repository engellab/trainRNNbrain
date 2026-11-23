## 2026-11-21 16:29 — Set points on Maze: a network killed by its wall clock is finished from its last checkpoint

Adds one rule to `2026-11-21_0812_rules-entropy-correlation-setpoint-maze.md`. Written before any Stage 2a network
was read and before the network that prompted it ended. Branch `wp/entropy-setpoint-maze`.

**Why a rule is needed.** At 00:28 UTC on 22 Nov, seven of the twelve Stage 2a tasks (Della job 18071039) had
finished, in 6.8 to 7.1 h each. Task 130 (`ent0.1`, seed 1) runs on della-i14g16, an A100 40 GB PCIe card. It runs
at 24.8 s per update, while the other four running tasks run at 15.9 to 16.9 s. At update 1,100 it had run 7.7 h. At
24.8 s per update it reaches update 2,000 after about 50,200 s, against a limit of 50,400 s (`--time=14:00:00`).
Slurm may therefore kill it before update 2,000, which is the Stage 2a read point. The rules say what happens to a
network that the rollback guard stops: it is not resumed, and its arm is incomplete. They do not say what happens
to a network that Slurm kills at its time limit.

**Rule.** A Stage 2a or Stage 2b network that Slurm ends with state TIMEOUT before its read point is continued
from its last checkpoint to the read point. It is read as one run.
- The run folder is rewritten every 100 updates (`trainer.checkpoint_every: 100`). Each write includes the weights,
  Adam's moment estimates, `Progress.json` and `Participation.npz`.
- The continuation uses the launcher's continuation mode: `PARENT_ROOT=<SAVE_TO>/Maze_interventions`,
  `INIT_ADAM=true`, and `EXTRA_UPDATES` = the read point minus the last update in the parent's `Progress.json`. It
  runs from the same pinned clone (`~/trainRNNbrain_entsp_maze`, e555fae), writes to `<arm>_cont/s<seed>`, and
  uses the same `--time` and `--constraint` as the network's own job.
- `maze-read` and the independent check script join `<arm>/s<seed>` and `<arm>_cont/s<seed>` already, so the read
  changes in no other way.
- A Stage 2b network whose first job times out is continued with `STAGE=2` and `EXTRA_UPDATES` = 4,000 minus the
  last recorded update. Its pending second job, whose `aftercorr` dependency can then never be met, is cancelled.

**Why continuing does not change the network.** The kill is set by the node's speed, not by anything in the
network's training. A continuation with `INIT_ADAM=true` restores the weights and Adam's moments. It also skips the
learning-rate warm-up, because `TrainerPPO` ramps the rate only when no Adam state was restored. The run script
leaves two parts of the trainer's state unsaved: the state of prune-and-reinitialise and the dropout participation
average. Both techniques are off in these runs (`prune_reinit: false`, `dropout: false` in the saved config of
task 130). The continuation therefore repeats the uninterrupted run, except for two things. The updates after the
last checkpoint are run again, with new random draws. The Slurm job, node and GPU-hours of those last updates
differ.

**What stays as it was.** A network that the rollback guard stops is still not resumed, and its arm is still
incomplete. Any other failure (FAILED, OUT_OF_MEMORY, NODE_FAIL, CANCELLED) is reported. Such a network is not
rerun or continued without a new entry written before it is read. The GPU-hours of continuations count toward the
300 GPU-h cap.
