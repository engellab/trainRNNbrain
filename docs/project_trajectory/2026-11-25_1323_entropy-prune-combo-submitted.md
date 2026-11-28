## 2026-11-25 13:23 — Entropy x prune-and-duplicate: pilots, reader dry runs and the submitted chain

Follows `2026-11-25_1245_rules-entropy-prune-combo.md` and its amendment `2026-11-25_*_entropy-prune-combo-length-and-chain.md`.

**Pilots** (Della jobs 18374857, 18374858, 18374859; enter no result). All eight ended COMPLETED; 1.1 GPU-hours. No EP
pilot skipped an update, and the pilot check (job 18376384) exited 0.

| Task | Cell | Seconds per iteration (3BFF, CDDM) or wall time | Card | Skipped |
|---|---|---|---|---|
| 3BFF | D, P, EP | 0.264, 0.262, 0.276 | 40 GB | 0 |
| CDDM | D, P, EP | 0.189 (40 GB), 5 min 15 s and 6 min 29 s wall (P on 80 GB, EP on 40 GB) | mixed | 0 |
| Maze | P, EP (40 updates) | 10 min 52 s and 9 min 39 s wall | 40 GB, 80 GB | 0 |

**Reader dry runs on E's existing networks only** (jobs 18376429, 18376430; no D, P or EP network was read). The reader
reproduces every recorded count and value: 3BFF 384, 380, 454 and R² 0.9503, 0.9515, 0.9516; CDDM 540, 566, 539 and
0.8340, 0.8268, 0.8344; Maze 634, 689, 661 and 0.938, 1.000, 1.000. On Maze the replayed probe also gives the stored
success and the stored count.

**Submitted** (`bash slurm/entropy_prune_combo_submit_della.sh 18374857 18374858 18374859`, at aa7157a for the reads and
0558a88 for the training):

| Step | Jobs |
|---|---|
| Pilot check | 18376384 |
| 3BFF stage 1, stage 2, E stage 2 | 18376385, 18376387, 18376389 |
| CDDM stage 1, stage 2, E stage 2 | 18376386, 18376388, 18376390 |
| Maze stage 1, stage 2 (P, EP), stage 2 (E), stage 2 (D) | 18376391, 18376392, 18376393, 18376394 |
| Read 3BFF and CDDM, read Maze, verdict and recount | 18376395, 18376396, 18376397 |

Output of the reads: `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/entropy_prune_combo_reads/`.
