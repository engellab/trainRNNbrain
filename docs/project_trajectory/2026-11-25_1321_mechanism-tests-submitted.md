## 2026-11-25 13:21 — Mechanism tests submitted: the whole chain, with dependencies

Rules: `2026-11-25_1311_rules-mechanism-noise-decay-and-switch.md` (pushed at d1b5355 before any submission) and its amendment
`2026-11-25_1317_amendment-mechanism-removal-bar.md` (ba5cd89). Branch `wp/mechanism-tests`. Nothing below depends on the connection
to Della staying up.

| Step | Della job | What | Waits for |
|---|---|---|---|
| part A, tasks 1-6 | 18376294 | default noise, decay 1e-6 and 0, three seeds each | — |
| part A, tasks 7-12 | 18376295 | no noise, decay 1e-6 and 0 | 18376294 |
| part B, tasks 1-9 | 18376296 | none, set point 0.1, rate penalty 0.1 from the three default networks | 18376295 |
| reads of older networks | 18376947 (30-42) | part C on networks that existed before (CPU) | — |
| part C on part A's networks | 18376948 (10-21) | Adam states and snapshots (CPU) | 18376294, 18376295 |
| removal | 18376949 (1-9) | `mech_removal.py` on the nine part B networks (CPU) | 18376296 |
| read, recount, verdict | 18376950 (0) | `mech_read.py`, `mech_recount.py` (CPU) | 18376949 |

GPU jobs run from `~/trainRNNbrain_mech` at d1b5355, CPU jobs from `~/trainRNNbrain_mechana` at ba5cd89. Commands:
`PIN=d1b5355 bash -s < slurm/mech_submit_della.sh` and `A1=18376294 A2=18376295 B=18376296 bash -s < slurm/mech_submit_reads_della.sh`
over ssh. Every dependency is `afterany`, so one failed task does not block the rest; the part B launcher refuses a network
whose parent has not finished, and the readers report a cell with fewer than three networks.

**The cap.** `slurm/mech_budget.sh` keeps one file per task in `~/mech_budget`. If no task is slower than its limit, the worst case is
79 GPU-hours. Tested locally: a task is refused when 260,008 s are already used and its own limit would pass 284,400 s.

The 3BFF Adam states of part C are on the Mac only; they are read there with `mech_optimizer.py adam`.
