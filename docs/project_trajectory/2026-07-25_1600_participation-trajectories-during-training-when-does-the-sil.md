## 2026-07-25 — participation trajectories during training: *when* does the silent mode appear? (submitted)

### Why

Every result above is an **endpoint**. We know the two ends of the story — at initialisation all units sit in one
narrow participation band (~0.05–0.08, 0% silent; 2026-07-01) and after 30k iterations the population is bimodal
with ~45–55% in a near-zero mode under `none`/`rws` and a single active mode under `frm` (2026-06-26). The path
between them was never observed, so four questions stayed open:

1. **When** does the bifurcation happen — early (while task loss is still dropping) or as a slow drift after the
   task is already solved?
2. **Is it reversible?** Does a unit that falls below the silent line ever climb back under `none`?
3. **What does `frm` do in time** — hold the population up from iteration 0 (prevention), or let it start
   splitting and then pull it back (resurrection)?
4. **Why does `rws` fail where `frm` succeeds?** Does it accelerate the descent or leave the `none` trajectory
   unchanged? (Answering this needs the `rws`-only *and* `frm`-only cells, which is why the grid is 4 conditions,
   not 2.)

This is the "logged training" item — the last HIGH entry in [`TODO.md`](../TODO.md).

### Implementation (commits `5b74972`, `d29ed5a`)

Per-unit participation is logged **during** training, on a **noise-free** forward pass, every `track_every`
iterations:

- `Trainer.participation_from_states_` — `std(fr) + 0.9-quantile(|fr|)` pooled over (time, trials), with
  `fr = activation(states)` for `equation_type="h"` and `states` for `"s"`. Deliberately identical to
  `PerformanceAnalyzer.plot_participation`, so the trace is comparable to every participation figure in this
  document. (Not to be confused with the pre-existing `Trainer.get_participation_`, which serves the dropout
  controller and uses `|x|` in both terms without the activation.)
- `Trainer.track_participation_` — one extra `RNN(input, w_noise=False)` forward, taken in `run_training`
  **before** the train step, so snapshot 0 is the untrained network. Overhead ≈ 3% wall clock at
  `track_every=10`. The noise-free pass is the point: the training pass runs at σ_rec = 0.05 and would not be
  comparable to the offline figures.
- Config: `trainer.track_participation` (default `False`) and `trainer.track_every` (10) in
  `configs/trainer/trainer.yaml`; this experiment selects `configs/trainer/trainer_ptrack.yaml`, which is
  `trainer.yaml` with tracking on.
- Output per net: `{score}_ParticipationTrace.pkl` = `{"iters": [0, 10, ...], "participation": [array(N,) float32,
  ...]}` (~12 MB at 3000 snapshots × 1000 units; the same data is ~100 MB as indented JSON, hence pickle), plus
  `participation_trace.png` via the new `PerformanceAnalyzer.plot_participation_trace` (log₁₀ heatmap, units
  sorted by final participation). The pre-existing `participation.png` is unchanged.

**Validation (thresholds set before running).** Smoke test at N=50 / 500 iters produced exactly 50 snapshots, and
the last online (torch) snapshot matched the offline (`RNN_numpy` + `PerformanceAnalyzer`) readout **per unit** at
**r = 0.991** (threshold r > 0.99). These are two independently-written integrators, so this validates the metric,
not just the plumbing; the residual 0.009 is the 10 training steps between the last snapshot and the saved weights.

### Cluster note — the sweep moved to Della

Spock's `scotty-l40s` **lost its SLURM client** in the 2026-07-24 Rocky Linux 9.8 update: `rpm -qa | grep slurm`
returns only `slurm-example-configs-24.05.4-1.sdl9.2`, and no `sbatch`/`squeue`/`sinfo` exists anywhere on the
filesystem, while `/etc/slurm/slurm.conf` (ClusterName `Spockmk2`) and a running `munge` show the host is still
meant to be a submit client. Needs a sysadmin package reinstall. **This sweep therefore runs on Della**, which is
also why the launcher differs from every previous one in this document (`--gpus-per-node`, `--mem-per-gpu`, and
`--time=2:30:00` — under 2 h silently lands on the `gputest` QOS with a 3-concurrent-job cap).

### Cluster test run (Spock, direct on scotty's GPU — no scheduler)

One net before committing to the sweep: eq `h`, N=1000, γ=0, no penalties, **3000** iterations. 7 min 43 s
(**0.154 s/iter** on an L40S), r² = 0.79, trace exactly 300 × 1000 float32 = 1.2 MB. Output in
`data/trained_RNNs/CDDM_ptrack_TEST/`. It already shows three things:

1. **The bifurcation is early and then frozen.** The split completes by iteration **~400–600** and the silent
   fraction is flat at ~55% from there to 3000 — consistent with the 44–47% known for h/`none` at 30000, i.e.
   essentially nothing changes over the remaining 27000 iterations. Supports the "fate decided early" prediction.
2. **A global collapse comes first.** Every quantile crashes to ~3e-4 within the first ~20 iterations — the
   *whole* network goes quiet — and then the eventual-active subset climbs back out by iteration ~300 while the
   rest stays down. This is **not** the predicted picture of half the units drifting downward; it is
   collapse-then-partial-recovery, and it was not visible in any endpoint measurement.
3. **Silencing deepens in discrete steps.** A second sharp event at iteration ~2150 drops q5/q25/q50 by another
   half-decade (a clean vertical boundary in the heatmap).

**Threshold caveat for the analysis:** the init band straddles the 0.05 inter-mode dip (init median 0.0504, range
0.044–0.060), so "fraction < 0.05" reads a meaningless 43% at iteration 0. Use `< 0.01` (final: 53%) or a per-net
dip when scoring the trace.

### The sweep (submitted — Della array `11609846`, commit `d29ed5a`)

40 jobs = **2 equations {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds**, N=1000.

| Axis | Config key | Values |
|---|---|---|
| Equation type | `model.equation_type` | `h`, `s` |
| Sparsity penalty | `trainer.lambda_rws` | `0`, `0.05` |
| FR-magnitude penalty | `trainer.lambda_frm` | `0`, `0.2` |
| Seeds | `seed="random"`, 5 array reps | 5 |

Fixed: `configs/model/rnn_relu_Dale.yaml` (**γ=0**, ReLU slope 1.0, `dt=1`, `tau=10`, sticky Dale boundary,
`bias_range=[0,0]`, `spectral_rad=1.2`, σ_rec=σ_inp=0.05), `+experiment=silent_units_N1000` (N=1000,
`max_iter=30000`, empty trainer tag), `trainer=trainer_ptrack` (`track_every=10` → 3000 snapshots/net). All
resolved values were verified on Della with `--cfg job` before submitting. These are **exactly** the settings of
the `CDDM_4a031e_g0` sweep, whose endpoints are known (h/`none` 44%, s/`none` 55%, `fr`/`both` 0%) — so the final
snapshot of every net must reproduce that distribution, which is the built-in correctness check on the run.

Launcher: [`slurm/SilentReLU_ptrack_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_gamma0_N1000_della.slurm)
(`--array=1-40`, `--gpus-per-node=1`, `--mem-per-gpu=32G`, `--time=2:30:00`; task 1 copies
[`docs/experiments/participation_trace.md`](experiments/participation_trace.md) into the sweep folder as
`EXPERIMENT.md`, stamped with the array ID and commit). Output →
`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/CDDM_ptrack_g0/EqType=<eq>_N=1000_LmbdRWS=<rws>_LmbdFR=<frm>/`.
Submitted **2026-07-25 15:53 EDT**; task 1 started 2 minutes later on `della-l03g14`. ~80 min/job expected.

### Predictions (recorded before the results)

- **`none`:** most of the eventual silent set crosses the dip within the first ~2–4k of 30k iterations,
  coincident with the steepest part of the task-loss curve, and does not return (<5% of units that stay below the
  dip for ≥500 iterations climb back). *The 3000-iteration test already supports the timing, with the collapse
  even earlier than predicted, and adds the unpredicted global-collapse-first phase.*
- **`frm` / `both`:** participation rises from the init band toward the cap and no unit descends below the dip.
- **`rws` only:** nearly indistinguishable from `none`, slightly faster/deeper descent.

**Falsifiers.** A descent spread gradually over all 30k iterations kills "fate is decided early". Units churning
in and out of the silent set kills the "fate" framing altogether — silence would be a dynamic state, not an
outcome. `frm` nets that dip and then recover would contradict the prevention result of 2026-07-01.

### Read-out (planned)

Analysis script `trainRNNbrain/experiments_and_analysis/plot_participation_trace.py` (to be written):
(1) trace heatmap per condition; (2) 5/25/50/75/95th percentile bands vs iteration; (3) silent fraction vs
iteration with the task loss from `*_LossBreakdown.json` on a twin axis; (4) per-unit crossing statistics — first
iteration below the dip, total time below, number of upward re-crossings (the reversibility answer);
(5) endpoint validation against the offline participation, per unit, r > 0.99.

