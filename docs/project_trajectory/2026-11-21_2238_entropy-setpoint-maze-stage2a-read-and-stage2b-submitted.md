## 2026-11-21 22:38 — Set points on Maze: Stage 2a (N = 1,000) read, λ chosen, Stage 2b (N = 2,000) submitted

Follows the rules `2026-11-21_0812_rules-entropy-correlation-setpoint-maze.md` and the wall-clock rule
`2026-11-21_1629_entropy-setpoint-maze-wall-clock-rule.md`. Branch `wp/entropy-setpoint-maze`.

**Stage 2a finished cleanly.** All 12 tasks of Della job 18071039 ended COMPLETED with exit code 0, and each ran
2,000 updates. No network was stopped by the rollback guard, and none hit its wall clock. Task 130 (`ent0.1`,
seed 1), on the slow della-i14g16 node, ended 7 minutes before its 14-hour limit, so the wall-clock rule was not
used. The networks took 6.8 to 13.9 h each, 98.8 GPU-h in all.

**Read.** The arm trees were copied from Della with rsync (config, `Progress.json` and `Participation.npz` only) to
`firstmate/data/entropy-setpoint/maze_N1000/`. `entropy_setpoint_read.py maze-read` read them with the controls from
the Mac tree, and its JSON is kept at `firstmate/data/entropy-setpoint/maze_reads_2a.json`. The independent check
script (`firstmate/data/entropy-setpoint/scripts/oracle_maze_setpoint.py`, Check 2) agrees with the reader exactly
on all 18 networks: the 12 new networks and the 6 controls. For each network it compares the number of active
units at the read point, the probe success there and the learned update. It also recounts the active units from
the stored participation.

**Stage 2a result, N = 1,000, read at update 2,000.** For an arm to pass, every network needs more than 449 active
units, the mean probe success must be at least 0.848, and no network may end below a probe success of 0.9.

| Arm | Active units, seeds 1–3 | Probe success, seeds 1–3 | Mean success | Learned update | Verdict |
|---|---|---|---|---|---|
| controls | 426, 449, 415 | 0.992, 0.977, 0.727 | 0.898 | 1,050, 760, not learned | — |
| `ent0.01` | 631, 667, 620 | 0.977, 0.938, 0.969 | 0.961 | 1,590, 1,910, 1,510 | passes |
| `ent0.1` | 634, 689, 661 | 0.938, 1.000, 1.000 | 0.979 | 1,610, 1,470, 1,000 | passes |
| `cor0.01` | 488, 527, 472 | 1.000, 0.977, 0.977 | 0.984 | 1,450, 1,320, 1,210 | passes |
| `cor0.1` | 625, 608, 561 | 0.875, 0.922, 0.922 | 0.906 | not learned, 1,730, 1,650 | fails: seed 1 ends at 0.875 |

- The mean number of active units is 1.49 times the controls' mean (430) for `ent0.01`, 1.54 times for `ent0.1`,
  1.15 times for `cor0.01` and 1.39 times for `cor0.1`.
- The set-point networks learned later than the controls. Their learned updates run from 1,000 to 1,910; the two
  controls that learned did so at updates 760 and 1,050.
- Four networks skipped minibatch steps or restored rollouts after non-finite values. In `ent0.01` seed 1, 5 steps
  were skipped and 4 rollouts restored. The other three networks were `cor0.01` seed 1 (7 and 10), `cor0.1` seed 1
  (3 and 4) and `cor0.1` seed 3 (23 and 13). The N = 1,000 controls had none.

**Manipulation check, as the rules set it** (the raw penalty at update 1 and its mean over the last 50 updates).
- Entropy term: the penalty falls in every network, from 0.28–0.36 to 0.033–0.038 at λ = 0.01 and to 0.012–0.016
  at λ = 0.1. The term acted.
- Correlation term: the penalty rises in every network, from 1.5 × 10⁻⁵ – 9.5 × 10⁻⁵ to 0.0077–0.0084 at
  λ = 0.01 and to 0.0019–0.0027 at λ = 0.1. By the rules, a term whose penalty did not fall did not test what it
  was meant to test, so the correlation term at N = 1,000 is reported as untested by this check. The pilot entry
  (08:39) predicted this before any network was read: an untrained Maze network's units are almost uncorrelated,
  so the penalty starts near its minimum.
- Not a pre-set read-out: over the last 50 updates the correlation penalty is 3.5 times lower at λ = 0.1 than at
  λ = 0.01 (means 0.0023 and 0.0082). The size of λ therefore changes the units' correlations. A comparison with
  the controls' correlations would show whether either λ holds them below their untreated level. That
  comparison is not made here.

**λ chosen for N = 2,000, by the rule set before any run.**
- Entropy: both λ values pass, and `ent0.1` has the larger mean number of active units (661, against 639 for
  `ent0.01`), so `ent0.1` runs.
- Correlation: only λ = 0.01 passes, so `cor0.01` runs.

**Stage 2b submitted.** It runs from the same pinned clone (`~/trainRNNbrain_entsp_maze`, e555fae), on gpu-short,
with `--constraint="nomig&gpu80"`. The tasks are `ent0.1` 130-132 and `cor0.01` 133-135, seeds 1-3. Firstmate asked
for a time limit that covers a node running at twice the usual time per update: twice the pilot's 18.5 s is
37 s, or about 20.6 h for 2,000 updates, within the rules' 23 h.

    ssh della 'cd ~/trainRNNbrain_entsp_maze && export REPO=$HOME/trainRNNbrain_entsp_maze N_OVERRIDE=2000 \
      SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_entsp_N2000 && \
      J1=$(sbatch --parsable --export=ALL --array=130-135 --time=23:00:00 --constraint="nomig&gpu80" \
        slurm/rl_maze_interventions_della.slurm) && \
      PARENT_ROOT=$SAVE_TO/Maze_interventions EXTRA_UPDATES=2000 INIT_ADAM=true sbatch --parsable --export=ALL \
        --array=130-135 --dependency=aftercorr:$J1 --time=23:00:00 --constraint="nomig&gpu80" \
        slurm/rl_maze_interventions_della.slurm'

The first 2,000 updates are Della job 18109618. The continuation to update 4,000 is Della job 18109619, whose
tasks each start when the matching task of 18109618 succeeds. Output goes to
`$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_entsp_N2000/Maze_interventions/<arm>{,_cont}/s<seed>`.

**Budget.** 0.75 GPU-h for the pilot plus 98.8 for Stage 2a leaves 200.4 of the 300 GPU-h cap. At the pilot's
time per update the six N = 2,000 networks need 126 GPU-h, for a total of about 226. The cap would be exceeded only
if the six networks averaged more than 1.6 times the pilot's time per update. The N = 1,000 networks averaged
8.2 GPU-h each, against the 8.1 h the budget rule estimated.
