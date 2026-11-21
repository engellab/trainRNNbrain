## 2026-11-19 18:41 — Correction to the claim-gaps ranking: Della's gpu-short cap is 24 GPUs per user; dry runs of the CDDM reader

**Corrects** `2026-11-19_1749_claim-gaps-ranking.md`, which says "gpu-short allows 44 GPUs per user" (copied from
`docs/AGENT_GUIDE.md`). `sacctmgr show qos gpu-short` gives 24 GPUs and 24 jobs per user (gpu-medium 20 GPUs, 24
jobs). The guide is corrected in the same branch.

**Effect on the claim-gap jobs (Della 17958685-17958691).** At 21:50 ET the `MazeWarmLong` jobs held 9 slots and
this package 15, so 9 of this package's tasks (CDDM N = 500 and 2,000, two N = 1,000 Maze continuations, Maze N = 2,000
prune-and-duplicate seed 3) wait with `QOSMaxJobsPerUserLimit`, and the stage-2 Maze tasks wait on their stage 1. A
gpu-medium job (24.5 h or more) had an estimated start of 21 Nov 22:41, and a Spock 8 h job 26 Nov 05:55, so the
tasks stay where they are. The running `MazeWarmLong` jobs are unaffected; a new `MazeWarm` submission would wait
behind this package's queued tasks until slots free (the four running N = 1,000 continuations end near 04:00 ET
Fri). Firstmate was told.

The expected end of the longest chain, Maze N = 2,000 prune-and-duplicate seed 3: starts when a slot frees (about
04:00 Fri), two stages of about 10-11 h each (17-20 s per update early in training), so about 02:00 Sat. If it has
not finished by 05:00 Sat, the results entry reports its seed as incomplete and gap A's count condition fails at
N = 2,000 at the reads it lacks, as the rules say for a network without a count.

**CDDM reader dry runs (Della CPU jobs 17959197, 17960161, 17960865, 17961080; read clone `~/trainRNNbrain_cgread`,
existing cells only, before any new run finished).** Four problems were found and fixed in
`claim_gaps_read.py` before any new network was read:
1. the verdict dictionary was built with integer keys through `dict(**...)` (crash);
2. `run_experiment.py` prints the "gradient spikes" line only when an update was skipped, so a finished log without
   it now records 0 skipped updates (before the fix every run read as unstable);
3. the CDDM σ_w = 1.0 cell (`CDDM_synnoise_ext/EqType=h_N=1000_sw=1.0_iters=100000`) exists only on the Mac; a copy
   without Adam state is at Della `~/claim_gaps_read/inputs/CDDM_synnoise_ext/` and the reader takes it by path;
4. `common.SNAP_DIR` is `data/trained_RNNs/../c1_snapshots`; in the read clone `data/trained_RNNs` is an empty
   folder and `data/c1_snapshots/<sweep>/<cell>` links point at `trained_RNNs_checkpoints` (the N = 1,000 rate-penalty
   cell, kept every 1,000 iterations) and `trained_RNNs_cg_checkpoints` (the new cells), so R² at the crossing is
   measured on a kept snapshot where one exists, as B2's read does.
