### Seeded scoring and the floor fallback, everything recomputed (agent-fix, merged af0c605; notes `docs/notes/agent-fix.md`)

- **Seeding (Pavel: project-wide).** `heldout_r2.heldout_batch` seeds the task's generator
  (HELDOUT_SEED); `score_net` seeds noise draw d with NOISE_SEED + d (the RNN's own generator was
  unseeded too, because `task.seed` is None); the ablation's ordering batch has its own seed; the R²
  cache key ends in `|seeded-v1`. Same weights twice: bit-identical on all three tasks; heldout_r2
  and the ablation scorer now agree exactly (the 0.847-vs-0.836 outlier reads 0.836 in both). New
  minus old over 61 networks: mean -0.0003, SD 0.0043 (so the old ±0.01 disagreements were scoring
  noise).
- **Floor fallback (Pavel).** Over 152 runs exactly the 6 beta-pinned runs change, all to
  `floor_kind = smoothed_min`; every other read is identical. The two claim-1 flip-flop runs cross
  at 52,159 (341 active) and 59,951 (292).
- **Claim 1 on all nine networks at the floor holds** (new folders `c1_ablation_seeded`,
  `c1_retrain_seeded`; CDDM reproduces to 4 decimals). Flip-flop: all silent removed 0.25 / 0.22 /
  0.53, retrained 0.941-0.948 against intact 0.948-0.951, as many at random 0.01-0.08. One check sits
  on its bar: 1258717025's pruned retrain ends 0.0103 below its intact control (bar 0.01), still
  climbing at 3,000 iterations; fine-tune batches were still unseeded - being rerun seeded.
- Figures rebuilt (F1, supplementary ablation, trade-off scatter, Figure 2). **Figure 1 is held back
  from the paper**: panel c now points at the flip-flop's 200k cells and draws the series at the end
  of training (N^0.40, 3 sizes) until N = 4000 lands (~15 Oct 16:00).
- Paper numbers that change are listed in the notes; edits being drafted on a paper branch.

