## 2026-10-15 22:13 — size-series gaps submitted: N = 100, DMTS N = 4000, and a third network where cells had two

Pavel (COSYNE abstract review): say the effect is absent at N = 100; report 3 networks per setting;
extend DMTS to N = 4000. Submitted:

| what | where | job | code | output |
|---|---|---|---|---|
| N = 100 controls, 3BFF (200k), CDDM (100k), DMTS 5τ (150k), 3 seeds each | Spock | 6965913 (9 tasks) | `n100-spock` 63c6c97 = c35d71b + launcher `SilentReLU_n100_spock.slurm` | `<Task>_n100/EqType=h_N=100_arm=control` (Spock; to sync to Della) |
| DMTS 5τ N = 2000 control, a fourth seed (one of three never learned) | Della | 15658099 | `~/trainRNNbrain_papergrid` c35d71b, Della-only launcher `SilentReLU_paper_grid_dmts5_della.slurm` task 7 | `DMTS_paper_grid_d5/EqType=h_N=2000_arm=control` |
| DMTS 5τ N = 4000 controls, 3 seeds (~45 h each, gpu80, 4-day limit) | Della | 15658100 | same, tasks 10-12 | `DMTS_paper_grid_d5/EqType=h_N=4000_arm=control` |
| 3BFF N = 1000 straggler (seed 3374117298, never reaches 1.05x its floor by 200k) to 300k | Della | 15658101 | `~/trainRNNbrain_c1top` 7739a60, launcher from `~/trainRNNbrain_redo` (d158716, TARGET-named root) | `NBitFlipFlop_ff_revive_c300000/EqType=h_k=3_N=1000_pen=none_arm=none` |

- The N = 100 launcher runs the paper grid's training command unchanged. A local smoke run's DMTS
  config differs from a `DMTS_paper_grid_d5` N = 500 control only in N, seed, tag, paths and budget.
  The 3BFF size series was trained on the 7739a60 lineage, not c35d71b; both are the standard network.
- The "94-99% active at N = 100" in the abstract comes from the 2026-08-18 CDDM sweep (trainable bias);
  these runs replace it with the standard network.
- **A DMTS size series to N = 4000 already exists at the 7τ delay:** `DMTS_d7_pen/EqType=h_N={500,1000,2000,4000}_pen=none`,
  5 / 3 / 5 / 3 networks, 210k iterations, standard network. At N = 4000, 2 of 3 learned (R² 0.999;
  the third 0.43). Not read at the floor yet. It cannot extend the 5τ series in Fig. 1c, but could stand
  in for it.
- `papergrid-dmts5` was deleted from GitHub in the cleanup; c35d71b survives in `n100-spock`'s history.
  The 5τ launcher exists only as an ignored file in Della's `~/trainRNNbrain_papergrid`.
