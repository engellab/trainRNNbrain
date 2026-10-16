## 2026-10-16 09:50 — One read-out path through common

Pavel approved audit item 4b on 2026-10-15. Merged as f74721d (branch commit 4c1bbd3, made by an
agent in a separate worktree).

**What is now in `common.py`** (one copy each; the README lists them):
- `read_run(run_dir)` reads one run at its floor. It returns t*, the active count at the probe
  nearest t*, whether the trace covers t*, the kept snapshot there, and the inputs of the R² estimate.
- `runs_with_loss`, `floor_crossing` and `at_own_floor`, moved unchanged from `fig_paper_F1`.
- `snapshot_at`, moved from `heldout_r2`, and `r2_at_floor_estimate`.
- `active_mask(p)`, the one copy of p ≥ 0.05·q95(p). It replaces the rule retyped in 24 scripts and
  the duplicated `SILENT_REL` / `REL` constants.

`heldout_r2.floor_r2` and `read_rung` (moved from `fig_knobs_at_floor`) give R² at the floor.
b5_read, c3, fig_paper_F2_fails, fig_knobs_at_floor and a4 no longer import a figure script for the
read-out.

**Checks, pass condition: every per-run output identical.**
- The agent's harness compared about 400 local runs before and after: t*, reason, active counts,
  R² at the floor and printed tables. There were 0 differences. Some uncached R² values came from a
  deterministic stub; the scorer itself is unchanged. The flip-flop cells are not on the Mac, so
  they were not compared.
- My own script, run through each tree's public functions, read all 94 runs of the Figure 1
  size-series cells and the Figure 2 knob ladders. All 94 were identical: 61 read at the floor, 33
  dropped for the same reasons.
- All 23 test scripts and the `common.py` self-check pass on the merged tree.

**Left open:**
1. `ablate_silent_units.floor_snapshot` still takes the nearest weight state with no distance
   bound. Switching it to the bounded `snapshot_at` could change which weights Figure 1 panel d
   ablates, and the snapshot layout can only be checked on Della. Decide after checking there.
2. `fig_supp_task_demand` still uses the R² written in the folder name, not held-out R², for its
   floor estimate. Changing that changes the A5 panel's numbers, so it is a separate change.
3. `fig_slides.py` (the deck) is untouched; its calls to `F1.runs_with_loss` / `F1.at_own_floor`
   now resolve to the common functions.
4. Two edge cases, absent from all data checked:
   - A diverged network with NaN participation now counts every unit as silent in the ablation's
     retrain set; before, it counted none.
   - `read_run` indexes the smoothed curve on the unfiltered loss series. That misaligns if a loss
     series ever contains NaN.
