## 2026-11-19 17:49 — Rules, gap A: does prune-and-duplicate raise the number of active units at every size on Maze (N = 500, 1,000, 2,000)?

Written and committed before any run of this gap is submitted. Ranking and cluster choice:
`2026-11-19_1749_claim-gaps-ranking.md`.

**Claim at stake.** Results 5 states that prune-and-duplicate increases the number of active units across
N = 500-4,000 on 3BFF and CDDM; Maze was tested at N = 1,000 only. This test asks the same question on Maze.

**Terms.** *Prune-and-duplicate*: every 100 optimiser steps, up to 2.5% of the units that have been silent for
3 checks are overwritten with a jittered copy of a working unit (the interventions launcher's arm `duplicate`:
copy noise 1.0 with the row norm kept, maturity 1,000 optimiser steps, utility "contribution"). *Learned point*,
*active unit*: as in the ranking entry. *Probe*: a noise-free test of 128 episodes from fixed starts after update
1 and every 10th update; it records probe success and the active count.

### Networks

| N | Prune-and-duplicate networks | Controls (existing default ReLU networks, Maze uniform series) |
|---|---|---|
| 500 | new: seeds 1-3, 2,000 updates, then 2,000 more with the Adam state restored (`INIT_ADAM=true`) | `Maze_suite/N=500_act=relu` s1-3 joined with `N=500_act=relu_cont` (updates 2,001-4,000) |
| 1,000 | the three runs of `trained_RNNs_rl_maze_fz/Maze_interventions/duplicate/s1-3` (Della 17108178, `freeze_after_end` true, 2,000 updates), continued for 2,000 updates with their Adam state restored | `Maze_suite/N=1000_act=relu` s1-3 joined with `_cont`: the controls of the paper's Maze criterion (decision dead-relu-paper#194) |
| 2,000 | new: as at N = 500 | `Maze_suite/N=2000_act=relu` s1-3 joined with `_cont` |

New runs use the arm's values unchanged, the config default `freeze_after_end: true`, and the commit named in
the submission entry; only N differs from the N = 1,000 runs. Continuations restore the Adam state, which is
exact (entry 2026-11-18_2033). Each run ends at update 4,000, the equal budget of the Maze size series (G11).

Known differences from the controls, stated before the runs: the controls were trained before 8 Nov
(`freeze_after_end` false, which is bitwise equal to true for an unpenalised network over 3 updates, entry
2026-11-10_0806) and continued at update 2,000 with a fresh Adam state and a repeated warm-up ramp; the new runs
keep their Adam state. A cold restart adds a median of 3 active units just after its ramp and does not last
through a stage (entry 2026-11-18_2033). The paper's N = 1,000 criterion compares the same kinds of runs.

### Read-out (G6's two points, in their Maze form)

Every run is read on its joined record (`common.maze_load_runs`), cut at update 4,000:
- **R1, the learned point** (`common.maze_learned_index`): the active count at the learned probe. The Maze
  counterpart of the floor crossing, used by the Maze size series.
- **R2, update 4,000**: the active count at the last probe. The counterpart of G6's fixed 75,000-iteration read.
- Also per network, reported: the learned update, probe success at update 4,000, and, at N = 1,000, the
  count against G12's three further controls (s4-6, not binding).

### Rule (G6's, unchanged apart from the Maze read points)

Per size and read point:
- *fold change* = mean count of the prune-and-duplicate networks / mean count of the same-size controls;
- *count condition* = every prune-and-duplicate network has more active units than every same-size control.

**Prune-and-duplicate increases the number of active units at every size on Maze** if the count condition
holds at N = 500, 1,000 and 2,000, at R1 and at R2 (6 of 6 size-and-read cells). The verdict is **fails at a
size** if the count condition fails there at R1 or at R2; the results entry names each failing cell.

- A network with no learned point by update 4,000 has no R1 count, and the count condition at R1 fails at its
  size ("every network" cannot hold). A network that stops before update 4,000 is resumed once through the
  continuation path (`PARENT_ROOT`, the stopped folder as parent); if it stops again, the count condition fails
  at its size at every read it lacks. Each stop and resume is recorded in a notebook entry.
- Success at update 4,000 is reported against the paper's Maze tolerance (mean within 0.05 of the same-size
  controls' mean, no network below 0.9). As in G6 it does not enter the direction.

### What each outcome does to the paper

- **Holds at every size (6 of 6):** "prune-and-duplicate increases the number of active units across sizes"
  holds on all three tasks, over N = 500-2,000 on Maze and N = 500-4,000 on 3BFF and CDDM.
- **Fails at a size:** the size claim stays scoped to 3BFF and CDDM; the Maze cell that fails is stated.

N = 4,000 is not run (a Maze N = 4,000 run needs about 32 h per 2,000 updates, past the 23 h jobs that start
before the deadline), so no outcome speaks for Maze at N = 4,000.

### Jobs (Della, `slurm/rl_maze_interventions_della.slurm`, arm `duplicate` = indices 52-54)

- N = 500 and N = 2,000, stage 1: `N_OVERRIDE=<N>`, `SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_cg_N<N>`.
- Stage 2 of each, `--dependency=afterok` on its stage-1 task: `PARENT_ROOT=<that SAVE_TO>/Maze_interventions`,
  `EXTRA_UPDATES=2000`, `INIT_ADAM=true`.
- N = 1,000: `PARENT_ROOT=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_fz/Maze_interventions`,
  `EXTRA_UPDATES=2000`, `INIT_ADAM=true`, `SAVE_TO=$HOME/trainRNNbrain/data/trained_RNNs_rl_maze_cg_N1000`.
  Only the new tree is written; the parent tree is read.
