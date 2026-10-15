## 2026-07-26 — first submission of the three sweeps: 55/120 nets, and two failure modes worth recording

### Outcome

| Array | Sweep | Usable nets | Lost | SLURM state |
|---|---|---|---|---|
| `11609846` | `CDDM_ptrack_g0` (Dale) | **18** | 22 | 18 COMPLETED, 22 TIMEOUT |
| `11610813` | `CDDM_ptrack_g0_nodale` | **18** | 22 | 18 FAILED, 22 TIMEOUT |
| `11610886` | `CDDM_ptrack_g0_nodale_trainablebias` | **19** | 21 | 19 FAILED, 21 TIMEOUT |

### Failure 1 — "FAILED" is misleading: the non-Dale nets are complete

`cluster_neurons` splits units by `dale_mask` into an E and an I group. The `dale=false` fallback added
that morning put **every** unit in the "positive" group, so the recursion into the empty group died on
`cannot reshape array of size 0`. The crash sits in post-processing **after** the config, params,
`LossBreakdown`, `GradsRaw/Scaled`, `ParticipationTrace.pkl`, `participation.png` and
`participation_trace.png` are all written — verified on disk: 18/18 and 19/19 of those nets have every
essential file. Only `avg_responses.png` and `intercluster_connectivity_matrices.png` are missing.
**No retraining was needed for those 37 nets.** Fixed by passing `dale_mask=None` when `model.dale` is
false, which `cluster_neurons` already handles as a single group.

### Failure 2 — the timeouts were not random: they were exactly the `frm` cells

Every `LmbdFR=0.2` condition came back empty in all three sweeps (20 jobs each), plus 5 stragglers.
Measured from the logs at N=1000 / 30000 iterations:

| Job type | s/iter | wall time for 30000 |
|---|---|---|
| no `frm` | 0.203 | 1 h 42 m ✓ |
| `frm`, unconstrained | 0.343 | ~2 h 51 m ✗ |
| `frm`, Dale, **on a MIG slice** | 0.632 | ~5 h 16 m ✗ |

Two compounding causes. (i) `frm` adds a backward pass per iteration — its gradient-norm monitoring
plus the task-safe projection — so `frm` cells are ~1.7× slower than `none`/`rws` cells. The 2:30
request was calibrated on a **non-`frm`** test run (0.154 s/iter on a Spock L40S) and never covered
them. (ii) Della's `gpu` partition mixes full A100s with `3g.40gb` **MIG slices** (`della-l01g3-12`, 10
of 89 nodes); the slowest job had landed on `della-l01g8`, a slice, and ran at half speed.

**Fixes:** `--time=7:30:00` (~2.6× the expected `frm` runtime) and `--constraint=nomig` to exclude the
slices — 79 of 89 nodes remain available, so queue access is barely affected. Both are now in all three
launchers, together with the measured numbers, so the calibration is not lost again.

### Resubmission (commit `5360791`)

Only the missing cells, 65 jobs, one array per sweep — the `frm` tasks map exactly onto array indices
`6-10, 16-20, 26-30, 36-40`, plus per-sweep top-ups for the non-`frm` stragglers (`seed="random"`, so
re-running an index simply adds another net to that condition):

```bash
cd ~/trainRNNbrain            && sbatch --array=1,6-11,16-20,26-30,36-40 slurm/SilentReLU_ptrack_gamma0_N1000_della.slurm         # 11635039, 22 jobs
cd ~/trainRNNbrain_nodale     && sbatch --array=1,6-10,16-21,26-30,36-40 slurm/SilentReLU_ptrack_nodale_gamma0_N1000_della.slurm   # 11635040, 22 jobs
cd ~/trainRNNbrain_nodalebias && sbatch --array=6-11,16-20,26-30,36-40   slurm/SilentReLU_ptrack_nodalebias_gamma0_N1000_della.slurm # 11635041, 21 jobs
```

All three worktrees were advanced to `5360791` first — safe here because the queue was empty, unlike
the mid-flight edit recorded in the 16:00 entry.

### Lesson

Calibrate wall time on the **slowest** cell of a grid, not a representative one, and check whether the
partition is heterogeneous before trusting a single timing measurement. A penalty that adds a backward
pass is a wall-time change, not just a science change.

