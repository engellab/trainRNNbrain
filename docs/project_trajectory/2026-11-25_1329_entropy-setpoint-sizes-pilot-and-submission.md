## 2026-11-25 13:29 — Entropy set point across sizes: pilot cost, CDDM N = 4,000 cut, chain submitted

Follows the rules `2026-11-25_1241_...` (8d31c36) and their addition `2026-11-25_1259_...` (ea004e9). Branch `wp/entropy-sizes`.

**Pilot** (Della jobs 18374551-18374556, pinned at 8d31c36, `SWEEP=entropy_sizes_pilot`; enters no result). Every job ended
COMPLETED with a finite R², and no log has a `gradient spikes` line, so no update was skipped. No pilot condition stops a cell.

| Task | N | Iterations | Training seconds | s per iteration |
|---|---|---|---|---|
| 3BFF | 500 | 2,000 | 335 | 0.168 |
| 3BFF | 2,000 | 2,000 | 1,314 | 0.657 |
| 3BFF | 4,000 | 500 | 1,104 | 2.21 |
| CDDM | 500 | 2,000 | 272 | 0.136 |
| CDDM | 2,000 | 2,000 | 683 | 0.342 |
| CDDM | 4,000 | 500 | 553 | 1.106 |

**Budget rule applied.** Cost of a cell = 3 × (iterations × s + 0.25 h per job), in GPU-hours: 3BFF N = 500 29.4, 3BFF N = 2,000
112.5, CDDM N = 500 12.1, CDDM N = 2,000 29.2, Maze N = 500 43.5 (7.0 h per 2,000 updates); pilot 1.2. F = 228.
- 3BFF N = 4,000 stays cut: 2.21 s per iteration, 368 GPU-hours for three networks of 200,000 iterations.
- CDDM N = 4,000 at 100,000 iterations: 93.7; 1.05 × (228 + 93.7) = 338, above 300.
- CDDM N = 4,000 at 75,000 iterations: 70.7; 1.05 × (228 + 70.7) = 314, above 300.
- **CDDM N = 4,000 is therefore cut as well.** F is below 300, so 3BFF N = 2,000 runs to 200,000.

So the test covers N = 500 and 2,000 on 3BFF and CDDM, and N = 500 on Maze. No statement about the term at N = 4,000 can be made.

**Unattended chain** (firstmate, 25 Nov: the Della connection may drop for long stretches). All jobs are submitted at once with
Slurm dependencies. Each training job runs `slurm/entropy_sizes_job.slurm`, which sums, at its start, the elapsed hours of
every ended EntSize training and pilot job and the full time limits of the running ones, and exits before training if the sum
exceeds 300 GPU-hours. A refused job fails; its continuations are cancelled (`--kill-on-invalid-dep=yes`). Time limits are
at least twice the predicted times. CPU jobs then read each cell into
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/entropy_sizes_reads/`, and after all reads the independent check
(`firstmate/data/entropy-sizes/scripts/oracle_sizes.py`, copied to `~/entsizes_scripts/`) and the verdict run. The verdict
job names every incomplete cell and gives it no verdict. The figures of the entry will be drawn on the Mac.

**Dry runs before submission.** The reader on every control cell reproduces the recorded values: counts exactly, R² within
2 × 10⁻⁹. The independent script agrees with the reader on every network read so far (controls of all sizes, the N = 1,000
`ent0.1` networks, and on Maze the N = 500 controls and the N = 2,000 `ent0.1` networks).

**Submission** (from the Mac; job ids in the next commit of this file):

    ssh della 'bash -s run <PIN> ff500 ff2000 cddm500 cddm2000 maze500 ff1000 cddm1000' < slurm/entropy_sizes_submit_della.sh
    ssh della 'bash -s finish <PIN>' < slurm/entropy_sizes_submit_della.sh
