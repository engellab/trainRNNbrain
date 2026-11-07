## 2026-11-07 07:15 — Round 18, G17: the second maze is the first maze mirrored left to right (amended rule, before the runs)

Amends `2026-11-07_0110_round18-training-rules.md` (sections "The second maze (G17)" and "G17") and closes
`2026-11-07_0259_round18-second-maze-calibration.md`. No G17 job had been submitted when this was written.

**The choice and when it was made.** The difficulty check found no `make_maze.py` draw in range. 15 mazes
with the first maze's size, loop count and geodesic statistics scored 0.000-0.132 in N = 100 probe
success after 300 PPO updates, against 0.311 on the first maze, and the acceptance range was 0.156-0.622
(entry `2026-11-07_0421_round18-second-maze-calibration-result.md`). After that result, the coordinator
chose the first maze mirrored left to right (dead-relu-paper#201, 7 Nov). This choice was made after the
calibration had been read; it is not the procedure of the original rules entry.

**The coordinator's reason.** The mirror has exactly the first maze's difficulty: the same geodesic
structure, dead ends and distances. With egocentric sensors and steering outputs it still demands a
different policy, because every left turn on a route becomes a right turn. So it tests transfer to a
related task of matched difficulty, which is closer to B4's design (same inputs, a new read-out demand)
than a maze outside the difficulty range. Seed 1086, the closest candidate at 0.132, would confound
transfer with difficulty.

**The second maze.** `python trainRNNbrain/tasks/make_maze.py --cells 7 --seed 20 --loops 4 --mirror`
(the new `make_maze.mirror`; every row of `configs/task/Maze.yaml` reversed), in
`configs/task/Maze_transfer.yaml`. Longest route 29 cells, mean route 15.9 cells, 6 dead ends, 101 free
cells, cheese at the centre lattice cell, all identical to the first maze's.
`test_second_maze_is_the_first_mirrored` in `tests/test_rl_maze.py` checks the config against string
reversal of the first maze, which is computed apart from `make_maze.mirror`, and checks that the two
mazes have identical statistics and the same distance histogram. The seed-222 maze that the first rules
entry named is not used.

**What is new relative to B4, in addition to the first entry's list.** The second maze shares the first
maze's corridors: a network that has learned the first maze has learned this layout's geometry in mirror
image. The transfer runs therefore test recruitment when the needed policy is the mirror of the learned
one, not when the environment is unrelated.

**Runs.** These are unchanged from the first entry except the sources, now all available:
- Old: N = 1,000 controls at update 2,000 that learned the first maze, which are uniform-series seeds 1
  and 2 (`Maze_suite/N=1000_act=relu/s1`, `s2`, their update-2,000 folders, not the continuations) and
  G12 seeds 4, 5 and 6 (all three learned). That is 5 old runs, each seeded with its source's seed.
- Fresh: seeds 1-3, untrained.
- Every run is 2,000 updates on the mirrored maze. Launcher `slurm/rl_maze_transfer_spock.slurm`.

**Read-out, gate and verdict.** These are those of the first entry, unchanged: S₀ from the probe after
update 1, A* from the first stored vector at or after the learned point, B4's gate (3 fresh SDs, where SD
is standard deviation) and bars (10% and 30% revival, 3 SDs and 1 SD). "Every" and "two thirds" are taken
over the old runs read. A verdict needs at least 4 of the 5 old runs and 2 of the 3 fresh runs read.

**A difficulty check that the runs give for free, reported without a threshold.** Fresh seeds 1-3 start
from the same initial weights as uniform-series seeds 1-3 on the first maze (same seed, same N, same
configuration). The learned update of each fresh run on the mirrored maze is reported beside that seed's
learned update on the first maze: 1,050 for seed 1, 760 for seed 2, and not learned by update 2,000 for
seed 3. These are the same networks on mirror-image tasks.
