# Correction to 2026-11-07_0032_fig5d-two-removals: the bound is 0.015, not 0.016

Entry 2026-11-07_0032 says removing the inactive units alone leaves the Maze success rate "within
0.016" of intact. The largest measured change over the five networks is 0.0146, which is 15 of 1,024
episodes. The changes are -0.0039, +0.0137, -0.0010, -0.0146 and +0.0117, recomputed by the reviewer
of dead-relu-paper PR #197 from the five JSON files in data/rl_maze_ablation_interventions. The old
bound was true but matched no measured value. The paper now says 0.015.
