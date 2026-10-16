## 2026-10-16 01:07 — Correction: the noise top-up's trainer mismatch does not touch Figure 2

Corrects `2026-10-16_0058_penalties-off-by-default-standard-network-is-the-config-default.md`, which
said the mismatch "affects how the recurrent-noise ladder is read".

`slurm/b0_noise_topup.slurm` continued the `b0_noise` runs (trained under `trainer_ptrack_plain`)
under `trainer`. The two configs differ in exactly three keys:
- `monitor`: diagnostics only.
- `task_safe_gradients`: projects penalty gradients against the task gradient. With every lambda
  at 0 there is no penalty gradient to project.
- `track_participation`: False under `trainer`, so the top-up segments of `CDDM_*_g0_noise_std_c75000`
  recorded no participation trace.

Figure 2's recurrent-noise ladder (`fig_knobs_at_floor.LADDERS`, `slide_k_recnoise`) reads
`CDDM_77fbd9_g0_noise_ptrack/..._iters=200000`, not those top-ups, so Figure 2 is unaffected. Any
future read of the `g0_noise_std_c75000` cells has no active-unit trace past each parent's end.
