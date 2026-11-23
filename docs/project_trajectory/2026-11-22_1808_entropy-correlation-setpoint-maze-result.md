## 2026-11-22 18:08 — Set points on Maze: the entropy term passes at N = 1,000 and N = 2,000; the correlation term fails at N = 2,000

Stage 2 of the set-point test, read under the rules `2026-11-21_0812_rules-entropy-correlation-setpoint-maze.md`.
Those rules were committed at e555fae before any Maze run. The wall-clock rule `2026-11-21_1629_...` was added
later, before any network was read, and was never used. Branch `wp/entropy-setpoint-maze`. Stage 1 (3BFF) is
`2026-11-21_0755_entropy-correlation-setpoint-3bff-result.md`.

**Verdicts under M09's Maze rule.** The rule allows a tolerance of 0.05 on probe success and sets a floor of 0.9.
- **Entropy term (A): passes on Maze.** At λ = 0.1 it passes at both N = 1,000 and N = 2,000; at λ = 0.01 it
  passes at N = 1,000, the only size where λ = 0.01 was run.
- **Correlation term (B): fails on Maze.** At λ = 0.01 it passes at N = 1,000 and fails at N = 2,000 on all three
  conditions. At λ = 0.1 it fails at N = 1,000, because one network ends below the floor.

By the rules' outcome table, the entropy term meets the M09 criterion on two of the three tasks, 3BFF and Maze;
CDDM was not tested. The correlation term meets M09's own Maze cell, which is at N = 1,000 only, and fails at
N = 2,000. The paper is unchanged, because the set-point terms are not among its techniques.

### Runs

Every network ran to its read point. No network was stopped by the rollback guard (the trainer's automatic stop
after repeated non-finite rollouts), none hit its wall clock, and every Slurm task ended COMPLETED with exit code 0.
- Stage 2a, N = 1,000: Della job 18071039, 12 networks, 2,000 updates each. Read in entry `2026-11-21_2238_...`,
  which also records how λ was chosen for N = 2,000.
- Stage 2b, N = 2,000: `ent0.1` and `cor0.01`, three networks each. Della job 18109618 ran updates 1–2,000 in
  8.9–9.4 h per network. Job 18109619 continued each network from update 2,000 to 4,000, keeping its Adam state,
  in 8.3–8.7 h. Both jobs ran on 80 GB A100 cards.
- GPU time: 0.75 GPU-h for the pilot, 98.8 for Stage 2a and 105.0 for Stage 2b, a total of 204.5 GPU-h within the
  300 GPU-h cap.

### Read and check

The arm trees were copied from Della with rsync (config, `Progress.json` and `Participation.npz` only) to
`firstmate/data/entropy-setpoint/maze_N1000/` and `maze_N2000/`. The controls are read from the Mac tree
(`Documents/GitHub/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_suite`) through the symlink folder
`firstmate/data/entropy-setpoint/mac_suite_view/`. `entropy_setpoint_read.py maze-read` wrote
`firstmate/data/entropy-setpoint/maze_reads.json`, and `maze-verdict` printed `maze_verdict.txt` and drew the figure.

The independent check script (`firstmate/data/entropy-setpoint/scripts/oracle_maze_setpoint.py`, Check 2 of the
rules) agrees with the reader exactly on all 24 networks: 6 controls and 18 set-point networks. It was written
from the rules text, without `common.py`. For each network it compares three values: the number of active units
at the read point, the probe success there, and the learned update. It also recounts the active units from the
stored participation. Its output is `firstmate/data/entropy-setpoint/maze_oracle.txt`.

### Result at N = 2,000, read at update 4,000

For an arm to pass, every network needs more than 506 active units (the largest control), the mean probe success
must be at least 0.937 (the controls' mean, 0.987, minus 0.05), and no network may end below 0.9.

| Arm | Active units, seeds 1–3 | Probe success, seeds 1–3 | Mean success | Learned update | Verdict |
|---|---|---|---|---|---|
| controls | 400, 506, 388 | 0.984, 0.977, 1.000 | 0.987 | 3,000, 2,480, 2,840 | — |
| `ent0.1` | 718, 730, 905 | 0.992, 1.000, 0.992 | 0.995 | 2,540, 2,640, 2,360 | passes |
| `cor0.01` | 419, 754, 492 | 0.945, 0.984, 0.813 | 0.914 | 3,330, 2,010, not learned | fails all three conditions |

- With `ent0.1`, the mean number of active units is 784, 1.82 times the controls' mean of 431. The entropy
  networks learned at about the same update as the controls.
- With `cor0.01`, two networks end with fewer active units than the largest control (419 and 492, against 506).
  Mean success is 0.073 below the controls', beyond the 0.05 tolerance. Seed 3 ends at 0.813 and never reached the
  learned criterion.

The N = 1,000 result (entry `2026-11-21_2238_...`) is summarised here: every network of `ent0.01`, `ent0.1` and
`cor0.01` ended with more than 449 active units (472 to 689) and a probe success of at least 0.94. With `cor0.1`,
seed 1 ended at 0.875, below the 0.9 floor.

### Descriptive read-outs (not pass conditions)

**N = 2,000 at update 2,000.** No network had learned by update 2,000. This includes the controls, which is why
the rules read N = 2,000 at update 4,000.
- Controls: 464, 556 and 431 active units, at probe successes of 0.77, 0.84 and 0.84.
- `ent0.1`: 757, 786 and 1,144 active units, at 0.69, 0.69 and 0.74.
- `cor0.01`: 399, 638 and 644 active units, at 0.62, 0.92 and 0.85.

**Skipped steps and restored rollouts.** After a non-finite rollout, the trainer skips the minibatch step or
restores the weights and replays the rollout.
- At N = 2,000, the `ent0.1` networks had none. The three `cor0.01` networks skipped 37, 21 and 47 minibatch
  steps and restored 22, 18 and 34 rollouts. For comparison, two of the three N = 2,000 controls skipped 29 and 10
  steps and restored 18 and 5 rollouts.
- At N = 1,000, the counts are in entry `2026-11-21_2238_...`. Four networks had some (3 to 23 skipped steps);
  the controls had none.

**One probe with zero active units.** In `cor0.01` seed 2 at N = 2,000, the probe at update 3,910 records 0
active units, at a probe success of 0.95. Update 3,910 had a non-finite rollout. Update 3,911 had another, after
which the weights were restored from the update-3,900 checkpoint. By update 3,920 the probe again recorded 727
active units. The zero is an artefact of that non-finite rollout and shows as a one-probe drop in the figure. It
does not touch the read at update 4,000 (754 active units, probe success 0.98).

### Manipulation check, as the rules set it

The check compares the raw penalty at update 1 with its mean over the last 50 updates of each network.
- **Entropy term:** the penalty falls in every network at both sizes. At N = 1,000 it falls from 0.28–0.36 to
  0.033–0.038 at λ = 0.01 and to 0.012–0.016 at λ = 0.1. At N = 2,000 (λ = 0.1) it falls from 0.20–0.23 to
  0.009. The term acted.
- **Correlation term:** the penalty rises in every network at both sizes. At N = 1,000 it rises from
  1.5 × 10⁻⁵ – 9.5 × 10⁻⁵ to 0.0077–0.0084 at λ = 0.01 and to 0.0019–0.0027 at λ = 0.1. At N = 2,000 (λ = 0.01)
  it rises from 3.1 × 10⁻⁶ – 9.5 × 10⁻⁶ to 0.013–0.014. By the rules, a term whose penalty did not fall did not
  test what it was meant to test. The correlation term on Maze is therefore reported as untested by this check.
  The pilot entry (`2026-11-21_0839_...`) predicted this before any network was read: an untrained Maze network's
  units are almost uncorrelated, so the penalty starts near its minimum.
- **Beyond the pre-set check:** at N = 1,000 the correlation penalty over the last 50 updates is 3.5 times lower at
  λ = 0.1 than at λ = 0.01 (means 0.0023 and 0.0082). So λ changes the units' correlations. Whether either λ holds
  them below an untreated network's level was not measured. The correlation term's failure at N = 2,000 is
  therefore a failure of a term whose effect on the correlations was not established against the controls.

### Known differences from the controls, stated in the rules before the runs

- The new networks use `freeze_after_end: true`. The controls behave as false. For an unpenalised network the two
  settings are bitwise equal over 3 updates; with a set-point term on, the term sees only the steps before each
  episode's end.
- The N = 2,000 controls were continued at update 2,000 with a fresh Adam state and a repeated warm-up ramp. The new
  networks kept their Adam state. A cold restart adds a median of 3 active units just after its ramp (entry
  2026-11-18_2033). That is small beside the gap at update 4,000: every `ent0.1` network has at least 212 more
  active units than the largest control.

### The 3BFF rate penalty at λ_frm = 0.1: the paper's R² and Stage 1's

Firstmate asked why the Stage 1 entry reads the rate penalty at 0.1 as failing on R² when the paper was said to
count it as meeting the criterion. The paper does not count it as meeting the criterion.
- The paper's verdict table (`floats/ST_verdicts.tex`, dead-relu-paper at a6c3f46) gives this cell as "fails (p):
  935–995; 3.72×; −0.021", where (p) means the mean performance is beyond the tolerance. The row has read "fails
  (p)" since the table was added (dead-relu-paper f928023).
- The figure script's printout (`data/round19/fig_paper_F5_tasks_printout_r22rerun.txt`) lists the rate penalty as
  "fails | works | works" on 3BFF, CDDM and Maze.
- Results 5 states that prune-and-duplicate is the only technique that meets the criterion on all three tasks.

The two reads measure the same three networks in different ways:
- **The paper** takes the intact score of the removal analysis: held-out R² with 9 noise draws (M09, lines
  115-117). It reads 0.932, 0.930 and 0.927, against controls of 0.952, 0.952 and 0.951, a difference of −0.0213.
- **Stage 1** takes `round18_g1g2.read_run`: held-out R² with 36 noise draws on the weights at iteration 75,000.
  It reads 0.9329, 0.9304 and 0.9276, against controls of 0.9514, 0.9517 and 0.9509, a difference of −0.0211.

The two differences agree within 0.0002, inside the 0.0005 that M09 states. Each misses the 0.01 tolerance by about
0.011. Neither read is wrong; the 36-draw read is the more precise. Under the matched-values rule, the rate penalty
at 0.1 would meet the criterion on all three tasks only with a tolerance of 0.021. The paper's "Dependence on the
tolerance" paragraph (Supplementary Methods) says the same. The paper was not edited.

### Figure and files

`img/internal_figures/entropy_setpoint_maze_active.{pdf,png}`: the number of active units against PPO update for
every network. Rows are the two sizes and columns are the two terms; each panel shows the controls and a dotted
line at the largest control count.
- The entropy networks peak in the first few hundred updates. At N = 1,000 they reach 848–873 active units at
  λ = 0.01 and 968–971 at λ = 0.1; at N = 2,000 (λ = 0.1) they reach 1,636–1,879. They then fall. From update 200
  on, no entropy network has a probe with as few active units as the largest control at that probe.
- The correlation networks stay closer to the controls. At N = 2,000, `cor0.01` seed 1 is at or below the largest
  control at every probe from update 200 on.

**Commit hashes.** The earlier entries name e555fae as the code pinned on Della. Rebasing this branch onto
origin/main (623e561) for its pull request gives that commit, and the later ones, new hashes with unchanged
content. The Della clone `~/trainRNNbrain_entsp_maze` stays at e555fae.
