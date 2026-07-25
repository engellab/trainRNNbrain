# Experiment: participation trajectories during training ("when does the silent mode appear?")

*This file is copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

## What this data is

Every net in this sweep carries a `*_ParticipationTrace.pkl` alongside the usual outputs. It is a
pickled dict:

```python
{"iters":         [0, 10, 20, ...],                  # training iteration of each snapshot
 "participation": [array(N,) float32, ...]}          # per-unit participation at that iteration
```

Participation of unit *i* is `std(fr_i) + 0.9-quantile(|fr_i|)` pooled over (time, trials) — the same
quantity as the `participation.png` figure and every participation number in
`docs/project_trajectory.md`. It is measured on a **noise-free** forward pass (`w_noise=False`) of the
fixed training batch (`same_batch=True`), so the trace is directly comparable to the offline figures.
Unit indices are stable across training, so row *i* is the same neuron at every snapshot.

## Question

All previous results in this project are **endpoints**: at initialisation the population sits in one
narrow band (participation ~0.05–0.08, 0% silent); after 30k iterations it is bimodal with ~50% of
units in a near-zero mode under `none`/`rws`, and one tight active mode under `frm`. The trajectory in
between was never observed. This sweep records it, to answer:

1. **When** does the bifurcation happen — early (while task loss is still dropping) or as a slow drift
   after the task is already solved?
2. **Is it reversible?** Does a unit that falls below the silent line ever climb back under `none`, or
   is the descent one-way?
3. **What does `frm` do in time** — hold the whole population up from iteration 0 (prevention), or let
   it start splitting and then pull it back (resurrection)?
4. **Why does `rws` fail where `frm` succeeds?** Does `rws` accelerate the descent, or leave the
   `none` trajectory essentially unchanged?

This is the "logged training" item, the last HIGH entry in `TODO.md`.

## Prediction (recorded before the run)

- **`none`:** silencing is early and fast — most of the eventual silent set crosses the inter-mode dip
  (~0.05) within the first ~2–4k of 30k iterations, coincident with the steepest part of the task-loss
  curve; and it is irreversible (<5% of units that stay below the dip for ≥500 iterations return).
- **`frm` / `both`:** participation rises monotonically from the init band toward the cap; no unit
  ever descends below the dip.
- **`rws` only:** nearly indistinguishable from `none`, slightly faster/deeper descent.

**Falsifiers.** A gradual descent spread over all 30k iterations kills "fate is decided early". Units
churning in and out of the silent set kills the "fate" framing altogether — silence would be a dynamic
state, not an outcome. `frm` nets that dip and recover would contradict the prevention result of
2026-07-01.

## Grid — 40 jobs

`2 equation types {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds`, N=1000.

| Axis | Config key | Values |
|---|---|---|
| Equation type | `model.equation_type` | `h`, `s` |
| Sparsity penalty | `trainer.lambda_rws` | `0`, `0.05` |
| FR-magnitude penalty | `trainer.lambda_frm` | `0`, `0.2` |
| Seeds | `seed="random"`, 5 array reps | 5 |

Fixed: `configs/model/rnn_relu_Dale.yaml` with `model.gamma=0`, N=1000, `dt=1`, no bias, sticky Dale
boundary, `trainer=trainer_ptrack` (= `trainer.yaml` + `track_participation: True`, `track_every: 10`),
`max_iter=30000`. These are **exactly** the settings of the `CDDM_4a031e_g0` sweep, whose endpoints are
known (h/none 44% silent, s/none 55%, fr-only and both 0%) — so the final snapshot of this sweep must
reproduce that distribution. That is the built-in correctness check on the whole run.

`track_every=10` over 30000 iterations gives 3000 snapshots × 1000 units ≈ 12 MB per net as a pickle.

## Readouts

Analysis script: `trainRNNbrain/experiments_and_analysis/plot_participation_trace.py`.

1. **Trace heatmap** — units (sorted by final participation) × iteration, log colour, one panel per
   condition. The bifurcation is visible directly.
2. **Quantile bands** — 5/25/50/75/95th percentile vs iteration, conditions overlaid, log y.
3. **Silent fraction vs iteration** — fraction below the dip (0.05), with the task loss from
   `*_LossBreakdown.json` on a twin axis, to test the "coincident with task learning" prediction.
4. **Crossing statistics** — per unit: first iteration below the dip, total time below, number of
   upward re-crossings. This is the reversibility answer.
5. **Endpoint validation** — final tracked vector vs the offline `PerformanceAnalyzer` participation,
   per unit; pass threshold r > 0.99 (set before running; the smoke test gave r = 0.991).

## Implementation

- `Trainer.participation_from_states_` / `Trainer.track_participation_`
  (`trainRNNbrain/trainer/Trainer.py`) — the snapshot is an extra noise-free forward pass taken in
  `run_training` *before* the train step, so the first snapshot (iter 0) is the untrained network.
  Overhead ≈ 3% wall clock at `track_every=10`.
- Config: `trainer.track_participation` (default `False`) and `trainer.track_every` in
  `configs/trainer/trainer.yaml`; this experiment selects `configs/trainer/trainer_ptrack.yaml`.
- Saved by `run_experiment.py` as `{score}_ParticipationTrace.pkl` (pickle, not JSON: the same array
  would be ~100 MB per net as indented JSON).
