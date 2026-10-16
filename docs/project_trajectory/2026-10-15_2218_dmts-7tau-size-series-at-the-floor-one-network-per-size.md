## 2026-10-15 22:18 — the 7τ DMTS size series at the floor: one readable network per size above N = 500

Read with Figure 1c's own functions (`fig_paper_F1.runs_with_loss` / `at_own_floor`, solve gate
R² ≥ 0.8, 1.05x noisy floor, late-rise guard), Della CPU job 15658308 from `~/trainRNNbrain_cosyne`
(f8d2a3a); script kept at `/scratch/gpfs/TENGEL/pt1290/cosyne_d7/dmts7_floor.py`.

| DMTS 7τ (`DMTS_d7_pen`, pen=none) | N = 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|
| on disk / solved | 5 / 4 | 3 / 3 | 5 / 3 | 3 / 2 |
| read at the floor | 4 | 1 | 1 | 1 |
| active at the floor | 114, 183, 326, 390 (mean 253, 51%) | 338 (34%) | 293 (15%) | 602 (15%) |

Fit N^0.35 (means). Five solved runs have no floor read-out, all by the late-rise guard: the loss
ends 1.6x to 81x above its fitted floor (DMTS's bursts back toward chance after the escape).
For comparison, the 5τ series read the same way: 226 (45%) / 444 (44%) / 462 (23%) at N = 500 /
1000 / 2000, every solved run read, N^0.51.

Reading: the 7τ series shows the same fall in active fraction (about half at N = 500 to 15% at
N = 4000) but rests on one network per size above N = 500, so it cannot replace the 5τ series in
Fig. 1c; at most it is a one-line supporting statement. A read-out rule that tolerates DMTS bursts
(see the 2026-10-15 readout entries on sustained crossings) may recover the dropped runs.
