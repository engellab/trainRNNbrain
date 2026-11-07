## 2026-11-07 02:59 — Round 18, G17: the second maze fails its difficulty check; a wider calibration, rule written first

Amends `2026-11-07_0110_round18-training-rules.md` (section "The second maze (G17)"). No G17 job has been
submitted.

**The check, as written in that entry.** N = 100 networks, 300 PPO updates (one update is a batch of 256
episodes followed by 8 optimiser steps), network seeds 1-3, on each maze; the measure is probe success
(the share of 128 noise-free test episodes from fixed starts that reach the cheese) averaged over the last
five probes (updates 260-300) and over the three seeds. A candidate is accepted if its measure lies
between half and twice the first maze's.

**Result (Mac CPU, current code at 2a0b127 plus the calibration task configs).**

| Maze (`make_maze.py` seed) | Seed 1 | Seed 2 | Seed 3 | Mean |
|---|---|---|---|---|
| First maze (20) | 0.375 | 0.366 | 0.061 | 0.267 |
| 222 (the chosen second maze) | 0.008 | 0.008 | 0.008 | 0.008 |
| 761 (next in the ranking) | 0.000 | 0.000 | 0.000 | 0.000 |
| 830 (next after 761, run at the same time) | 0.000 | 0.005 | 0.000 | 0.002 |

The acceptance range was 0.134-0.534, so all three candidates are rejected. In all three, 0 or 1 of the
128 probe episodes reached the cheese after 300 updates; on the first maze 47-48 of 128 did for two
seeds and 8 for the third. 0.008 is one episode in 128.

**Why matched statistics did not give matched difficulty, as far as the record shows.** The first maze was
itself chosen by this calibration: of the three seeds tried on 4 Nov (20, 15, 6), seed 20 was the one an
N = 100 network learned (`configs/task/Maze.yaml`, the comment on the maze). Seeds 15 and 6 scored 0.00
and 0.02, close to the three candidates here, but they also had longer routes (32 and 36 cells from the
far end). The three candidates here match the first maze on all three geodesic statistics (longest
route, mean route length, dead ends) and still score near 0, so those statistics do not predict how
easy a maze is for this agent. This is an interpretation from four rejected mazes and no test of a cause.

**Amended procedure, fixed before any of these runs.**
1. Candidates: the next twelve seeds in the statistics ranking, now over seeds 0-1999 (6, 15, 20 and the
   three rejected seeds excluded), in ranking order: 1446, 1610, 958, 1531, 1485, 1086, 793, 1926, 1644,
   861, 781, 1372. All have 6 dead ends, a longest route of 28-30 cells and a mean route length of
   15.2-16.3 cells.
2. The same check, run as CPU jobs on Spock (`slurm/rl_maze_calibrate_spock.slurm`, commit 87d05ae or
   later, network seeds 1-3, 300 updates at N = 100), with the first maze (seed 20) run in the same array.
   The reference is the first maze's mean from this Spock batch, so both sides are measured on the same
   hardware; the Mac value above (0.267) is reported beside it.
3. The second maze is the highest-ranked candidate whose mean lies between half and twice the reference.
   Its rows replace the maze in `configs/task/Maze_transfer.yaml`, with its statistics and calibration in
   the file's comment, and G17 is submitted from that commit. The G17 read-out and verdict rules are
   unchanged.
4. If no candidate is accepted, G17 is not submitted. The record of the rejected mazes and the choice
   between a maze from a different generator setting, or reporting the transfer test on 3BFF and CDDM
   only, go to the coordinator on dead-relu-paper#201.

**Cost.** 39 CPU jobs of about 30-45 min with 4 threads each, no GPU.

**Selection effect, stated.** Choosing the second maze by this check matches how the first maze was chosen.
Both mazes are then the easier draws from `make_maze.py` at these settings, not typical ones.
