## 2026-11-19 17:56 — Claim gaps A, B and C submitted on Della

Rules (committed and pushed at 07c0a83 before this submission): `2026-11-19_1749_rules-maze-duplicate-sizes.md`
(gap A), `2026-11-19_1749_rules-cddm-rate-penalty-sizes.md` (gap B), `2026-11-19_1749_rules-maze-rate-penalty-sizes.md`
(gap C); ranking and the reason for Della in `2026-11-19_1749_claim-gaps-ranking.md`.

**Code.** A standalone clone on Della, `~/trainRNNbrain_cg`, detached at 07c0a83 (a clone with its own `.git`, not a
worktree of `~/trainRNNbrain`). Submitted by `slurm/claim_gaps_submit_della.sh` (`ssh della 'bash -s' < ...`),
which refuses a dirty clone, a missing parent Adam state, or an output tree that already exists.

**Checks before submission (the Mac, a scratch clone at 07c0a83, `module`, `conda` and `srun` stubbed).** Pass
condition, set before the diffs: each saved config differs from the config of its N = 1,000 counterpart only in N,
seed, output path and the smoke-size keys.
- Maze launcher, `N_OVERRIDE=50`, 2 updates of 4 episodes × 6 decisions: `duplicate` s1 against the Della config of
  `trained_RNNs_rl_maze_fz/.../duplicate/s1` differs in `model.N`, `paths.save_to`, `trainer.T`, `trainer.n_episodes`,
  `trainer.n_updates`; `frm0.1` s2 against `frm0.1/s1` in those and `seed`, `task.seed`. Pass.
- Maze stage 2 (`PARENT_ROOT`, `INIT_ADAM=true`) against its stage 1: `experiment_tag`, `paths.init_adam`,
  `paths.init_from`, `trainer.iter_offset`; the log says the Adam state was restored. Pass.
- The `N_OVERRIDE` guard: with the default `SAVE_TO` the launcher prints its FATAL line and exits 1. Pass.
- CDDM penalties launcher, index 5, `N_OVERRIDE=50 ITERS_OVERRIDE=30`, against the config of the N = 1,000 frm0.1 cell
  (seed 3986057758): `experiment_tag`, `model.N`, `paths.save_to`, `seed`, `trainer.max_iter`. Pass.

**Jobs (submitted 20:56 ET).** Logs: Della `~/trainRNNbrain_logs/<name>.<job>_<index>.out`.

| Gap | Cells | Job (name) | Array | Time limit | Output tree (new) |
|---|---|---|---|---|---|
| A, C | Maze N = 500, stage 1 | 17958685 (CGmaze1_N500) | 7-9 (frm0.1), 52-54 (duplicate) | 12 h | `~/trainRNNbrain/data/trained_RNNs_rl_maze_cg_N500` |
| A, C | Maze N = 500, stage 2, `aftercorr:17958685` | 17958686 (CGmaze2_N500) | same | 12 h | same |
| A, C | Maze N = 2,000, stage 1 | 17958687 (CGmaze1_N2000) | same | 20 h | `..._cg_N2000` |
| A, C | Maze N = 2,000, stage 2, `aftercorr:17958687` | 17958688 (CGmaze2_N2000) | same | 20 h | same |
| A, C | Maze N = 1,000, continuation of the fz runs, 2,000 → 4,000 | 17958689 (CGmaze2_N1000) | same | 12 h | `..._cg_N1000` |
| B | CDDM rate penalty 0.1, N = 500, 75,000 iterations | 17958690 (CGcddm_N500) | 4-6 | 8 h | `~/trainRNNbrain/data/trained_RNNs_cg` |
| B | CDDM rate penalty 0.1, N = 2,000 | 17958691 (CGcddm_N2000) | 4-6 | 23 h | same |

36 Maze tasks (24 run at once at most) and 6 CDDM tasks. The `MazeWarmLong` jobs (17897470) and their trees are
not touched.
