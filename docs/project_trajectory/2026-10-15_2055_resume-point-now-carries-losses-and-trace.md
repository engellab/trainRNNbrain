## 2026-10-15 20:55 — The resume point now carries the losses and the trace

Corrects the resume advice in `2026-10-15_1342_open-rules-and-commands-moved-from-agent-notes.md`
("a run that times out continues from `.../s<seed>/resume/` with `paths.init_from`").

**The defect** (code audit, entry 1500): `write_checkpoint` saved weights, Adam state and config,
but not the losses or the participation trace. A run continued from `resume/` therefore had a
chain whose losses stopped at the kill, and `common.noisy_loss_series` returned nothing for the
whole chain, so the run had no floor read-out.

**The fix.** The resume copy now also holds `ckpt_s<seed>_TrainLosses.json` (every loss so far) and
`ckpt_s<seed>_ParticipationTrace.pkl` (the trace so far, joined to the parent's), written in a
finished run's formats. `trace_to_save` builds the trace for the final save and for the checkpoint
alike. Kept `it*/` snapshots are unchanged.

**Test, pass/fail set before running.** End to end on the Mac:
- Run A: 30 iterations at N = 20, checkpointing every 10.
- Run B: continued from A's `resume/` for 10 iterations.
- Required: `resume/` holds 20 losses; `common.noisy_loss_series(B)` returns 30 losses whose first
  20 equal A's; the trace joins across the resume.

All held: 20 losses, a 30-loss series, the first 20 identical, and trace iterations 0-20. Contract 10
in `tests/test_periodic_checkpoint.py` now covers the file contents. All 22 test scripts pass.

**Runs already on Della are not covered.** They run from worktrees pinned before this commit
(`~/trainRNNbrain_redo` 709c17d, `~/trainRNNbrain_c1top` 7739a60, `~/trainRNNbrain_b2della`
2816d14), so their `resume/` folders still lack losses. If one of them times out, its per-iteration
training loss survives only in its Slurm log (`iteration i/M, train: <loss>` lines in
`~/trainRNNbrain_logs/<job>.<id>_<task>.out`); rebuild `TrainLosses.json` from that log before
continuing, or the continuation cannot be read at its floor.
