## 2026-11-04 09:20 — Maze suite: the RNN is the whole agent, maze seed 20, pilot 16842100 at 6d65d7b (issue #3)

Updates `2026-11-04_0818_rl-maze-suite-preregistration-and-pilot.md`: the design and the maze
changed before the suite ran; the read-out points and the falsifiers did not.

**Pavel, 4 Nov, during the day.** (1) "There is ONLY the RNN": no value network, no learned
exploration parameters. (2) Feed the executed action back as input. (3) A small cost for bumping
into walls. (4) Leaving: work autonomously; make the maze a task in the paper on par with CDDM and
the 3-bit flip-flop.

**The agent now** (commit 6d65d7b). The standard RNN with its parameters W_inp, W_rec, W_out and
nothing else (tests/test_rl_maze.py::test_only_the_rnn_is_trained). Inputs: 8 egocentric range
sensors and the previous executed turn and speed (an efference copy); nothing about the cheese.
Read-out rows: turn and speed; a third row, the value, exists only with ppo.critic. The
exploration std is fixed at 0.5 raw units. Default algorithm: PPO without a critic, advantages =
discounted Monte-Carlo returns minus their per-step batch mean. A move into a wall slides along it;
every bump costs 0.02, the step 0.01, a cell of geodesic progress pays 0.2, the cheese 5.

**Algorithm comparison** (old 11 x 11 maze, N = 100, 64 episodes, 300 updates; probe success at
update 300, seeds 5 and 6): PPO with the value row 0.69, 0.41; PPO without a critic 0.52, 0.77;
A2C-like and REINFORCE (one epoch, no clip) diverged to NaN; PPO without the efference copy 0.56,
0.03. Rule set before: mean over seeds, ties to the simpler one. Chosen: no critic (mean 0.65
against 0.55); the efference copy stays (0.55 against 0.30). Off-policy methods were out of scope
(they need Q-networks outside the RNN).

**Maze calibration with the final design** (same scale, seeds 5 and 6, probe success at 300):
seed 15 (32 cells from the far end, 17.5 on average) 0.00, 0.00; seed 20 (29, 15.9, 6 dead ends)
0.69, 0.25. Rule (at least 0.2, hardest wins): **seed 20**. Earlier calibration with the previous
design: seed 6 0.02, seed 13 0.00, seed 9 0.11, seed 15 0.31, seed 20 0.45; the old maze 0.55.
Picture: img/internal_figures/rl_maze_candidates.png.

**The cancelled pilot** (array 16838504, previous design, seed-6 maze): N = 500 reached probe
success 0.70 by update 370 and N = 1,000 0.28 by 350, so the hard maze is learnable at 256
episodes per update though no local run learned it; cancelled as obsolete, about 3 GPU-h.

**Reviews.** Review 1 (suite config and launcher) and review 2 (the agent changes) by independent
agents on issue #3; every required change applied before 6d65d7b.

**What runs.** `cd ~/trainRNNbrain_rlmaze && sbatch --array=4,7 --time=12:00:00
slurm/rl_maze_suite_della.slurm` at 6d65d7b (clean): array 16842100, tasks 4 (N = 500, seed 1)
and 7 (N = 1,000, seed 1), 2,000 updates, maze 20, final design, about 11 GPU-h. Gate: probe
success above 0.5 by update 1,000 at N = 1,000, then the other 18 runs from the same commit.
Outputs `/home/pt1290/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite/N=<N>_act=<act>/s<seed>/`;
logs `~/trainRNNbrain_logs/MazeSuite.16842100_<i>.out`.
