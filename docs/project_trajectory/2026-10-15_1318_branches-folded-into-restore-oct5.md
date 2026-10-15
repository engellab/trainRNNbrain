## 2026-10-15 13:18 — All branches folded into restore-oct5; merged branches deleted

Follows `2026-10-15_1257_repository-cleanup-15-oct-pavel-we-dont-need-multiple-versio.md`.
Pavel: "Only me is working on the repo, so you can delete branches if they have been merged. Yes,
fold everything into a single branch."

**What was merged into `restore-oct5`:**
- `c66bb1f`: `b5-optimizer` (`trainer.optimizer` adam/adamw/sgd and the B5 launchers).
- `d84b181`: `ffact-leaky-softplus` (flip-flop activation arms). The model configs keep restore-oct5's
  version, with the bias fixed at 0.
- `9c88d81`: `dimensionality-analysis`, the pre-repair commits (per-unit gain tracking, metabolic
  withdrawal, slide stacking).

**Conflicts:**
- Code: restore-oct5 won wherever it was newer.
- Slide scripts: took the one-column layout from the incoming branch.
- `docs/presentation.md`: blocks 1, 2, 3 and 5 from restore-oct5, block 4 combined, block 6 from
  the incoming branch.

**Check, pass rule set before running:**
- All 20 test scripts must pass.
- A short CDDM run and a short flip-flop run must give bitwise-identical losses and weights before
  and after the merge.

Both checks passed.

**Branches:** every merged branch was deleted locally and on GitHub. `main` and `restore-oct5`
remain; `main` has not been fast-forwarded yet. There is one checkout, `~/Documents/GitHub/trainRNNbrain`.
`stash@{0}` keeps the pre-repair working tree as a safety net.

**Notebook rule:** CLAUDE.md now also says how to read the notebook, that
`docs/project_trajectory.md` must not be recreated, and that a result which changes a claim also
updates `docs/paper_audit.md`. The hourly heartbeat prompt (cron, now `28800cdd`) was rewritten to
file each result as a new entry instead of appending to the old file.
