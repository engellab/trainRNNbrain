## 2026-11-21 08:39 — Set points on Maze: pilot cost, and Stage 2a (N = 1,000) submitted

Follows the rules `2026-11-21_0812_rules-entropy-correlation-setpoint-maze.md`, committed at e555fae before any Maze
run. Branch `wp/entropy-setpoint-maze`.

**Pinned code.** `~/trainRNNbrain_entsp_maze` on Della is a clean clone, detached at e555fae. The pilot and all
Stage 2 jobs run from it. The pilot script (the launcher's command plus `trainer.n_updates=40`) is kept outside the
repository at `firstmate/data/entropy-setpoint/scripts/maze_pilot.sh`.

**Pilot** (Della jobs 18069887-18069890, seed 1, 40 updates each, on 80 GB A100 cards; Della routed the jobs to
`gputest` because of their short time limit). Output is in `trained_RNNs_rl_maze_entsp_pilot_N{1000,2000}` and enters
no result.

| Job | N | Arm | Mean s per update at update 40 (printed running mean) | Peak GPU memory | Updates with skipped or restored steps | Wall time |
|---|---|---|---|---|---|---|
| 18069887 | 1,000 | `ent0.1` | 13.63 | 15.4 GB | 0 of 40 | 9 min 56 s |
| 18069888 | 1,000 | `cor0.1` | 14.17 | 19.0 GB | 0 of 40 | 9 min 51 s |
| 18069889 | 2,000 | `ent0.1` | 17.74 | 31.6 GB | 0 of 40 | 12 min 23 s |
| 18069890 | 2,000 | `cor0.1` | 18.54 | 38.9 GB | 0 of 40 | 12 min 32 s |

No pilot ran out of memory, and none skipped or restored a step, so Stage 2 starts at both sizes. At N = 2,000,
the correlation run peaked at 38.9 GB, so the 80 GB constraint for the N = 2,000 jobs is needed.

**Budget rule applied.**
- s_1000 = 14.17 s, so c_1000 = 14.17 × 2,000 / 3,600 + 0.2 = 8.1 GPU-h per network.
- s_2000 = 18.54 s, so c_2000 = 18.54 × 4,000 / 3,600 + 0.4 = 21.0 GPU-h per network.
- P = 0.75 GPU-h.

12 c_1000 + 6 c_2000 + P = 97 + 126 + 1 = 224 GPU-h, within the 300 allowed, so the full design runs: both λ values of
both terms at N = 1,000, then one λ per term at N = 2,000.

**The correlation penalty starts near zero on Maze.** This was recorded before any Stage 2a network was read. At
update 1, the raw correlation penalty mean_i (c_i/0.4)² is 3.9 × 10⁻⁵ at N = 1,000 and 9.5 × 10⁻⁶ at N = 2,000,
which corresponds to a root-mean-square c_i of about 0.0025 and 0.0012. By update 40 it has risen to 2.6 × 10⁻⁴ and
4.4 × 10⁻⁴. The untrained Maze network's units are almost uncorrelated, so term (B) acts only as correlations
develop during training. The rules' manipulation check asks whether the penalty fell from update 1 to the last 50
updates. For (B) that check cannot show that the term acted, because the penalty starts near its minimum. The result
will report the check as written, together with the penalty's course over training. The entropy penalty starts at
0.29 (N = 1,000) and 0.21 (N = 2,000) and falls to 0.014 and 0.017 by update 40.

**Stage 2a submitted.** Della job 18071039, array 127-138, on gpu-short (partition `gpu`, QOS `gpu-short`). The
tasks are `ent0.01` 127-129, `ent0.1` 130-132, `cor0.01` 133-135 and `cor0.1` 136-138, seeds 1-3:

    ssh della 'cd ~/trainRNNbrain_entsp_maze && REPO=$HOME/trainRNNbrain_entsp_maze \
      SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_entsp_N1000 \
      sbatch --export=ALL --array=127-138 --time=14:00:00 slurm/rl_maze_interventions_della.slurm'

Output: `$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_entsp_N1000/Maze_interventions/<arm>/s<seed>`; logs
`~/trainRNNbrain_logs/MazeIntervD.18071039_<task>.{out,err}`. Stage 2b is submitted once these are read, under the
λ-selection rule.
