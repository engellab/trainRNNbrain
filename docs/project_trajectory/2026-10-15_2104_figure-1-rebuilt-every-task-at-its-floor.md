## 2026-10-15 21:04 — Figure 1 rebuilt: every task's size series at its floor

Built on Della (job 15655514) from `~/trainRNNbrain_c1ana` at c2af704, after the cosyne2027-fig
merge; RETRAIN_DIRS = (`c1_retrain_seeded2/rep1`, `c1_retrain_seeded`). Figure
`img/internal_figures/fig_paper_F1.pdf` (+ .svg); no rasters.

**Panel c, all read at each run's 1.05× floor.** Coverage audit: no size missed on any task.
- 3-bit flip-flop: M = 10.8·N^0.47. 1,000 active units would need N ≈ 14,500. One N = 1000 run
  has no read-out (straggler s3374117298).
- CDDM: M = 16.7·N^0.46; 1,000 active units would need N ≈ 6,800.
- DMTS 5τ: M = 10.3·N^0.52, three sizes only. One N = 2000 seed is below r² 0.8 and dropped.

**Panel d, at the floor.** Retraining after removing the silent units:
- flip-flop: 0.948, against 0.949 intact;
- CDDM: 0.827, against 0.832 intact;
- DMTS: 0.999, against 0.999 intact.

**Panel e.** After each network's own floor crossing, training silences more units by the end:
116 on the flip-flop, 152 on CDDM, 277 on DMTS (N = 1000).

**Not yet synced to the paper.**
- The read-out rewrite awaiting Pavel (entry 1400) may move panel c and panel e.
- Two layouts break the no-overlap rule. In panel c, the "every unit active" line crosses the
  "1,000 active units" label, and CDDM's extrapolation crosses "10% active". In panel d (DMTS), the
  legend sits over the thin traces.
- The caption still says the flip-flop is drawn at the end of training.

Fix the layout and caption, then sync, after the read-out rule lands.
