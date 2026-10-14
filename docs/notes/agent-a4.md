# Agent A4: the unread CDDM rate and sparsity top-ups

Item A4: read `CDDM_std_g0_bias0` and its continuation `CDDM_std_g0_bias0_c75000` (Della), so the
rate penalty has a standard-network CDDM cell beside the standard control.

## Predictions written before any read-out (23:29, 2026-10-13)

What would show the read-out is wrong, decided before running:

1. **The in-sweep control disagrees with the reference control.** The sweep's no-penalty cell
   (`LmbdRWS=0_LmbdFR=0`) is the same standard network as `CDDM_std_g0_drift_bias0` N = 1000, which
   reads about 418 active units at a floor crossing of 24-29k iterations. If the in-sweep control
   reads outside 330-500 active, or crosses outside 15-40k, either the read-out or the sweep is not
   what it claims to be.
2. **The trace's counts are not the network's counts.** An independent numpy forward pass of the
   final weights (own integrator, own participation code, noise-free, on the training grid) must
   agree with the trace's last stored count within 20 units, and classify at least 97 of every 100
   units the same way. The trace was probed 100 iterations before the final weights, so a small
   disagreement is expected; more than that means the trace or the threshold is wrong.
3. **The floor R² estimate is unsafe.** No run in this sweep kept weight snapshots, so R² at the
   floor is estimated from the end (`fig_knobs_at_floor.floor_r2`), and on penalty arms the loss in
   that estimate includes the penalty. Each chain has measured weights at 50k (the parent's
   LastParams) and at 75k. If held-out R² at 50k and at 75k differ by more than 0.01 in a cell, the
   end-based estimate cannot be trusted there.
4. **Re-reading the drift control through my script does not give about 418.** Then my call
   differs from `fig_knobs_at_floor`'s.

## Results (Della jobs 15525676 and 15527664, read 2026-10-14)

Script: `trainRNNbrain/experiments_and_analysis/a4_cddm_std_penalties.py`; raw output
`/scratch/gpfs/TENGEL/pt1290/agent_a4/a4_cddm_std_penalties.json`, log `~/f2_logs/a4read_15525676.log`.
Held-out R² cache written to `/scratch/gpfs/TENGEL/pt1290/agent_a4/heldout_r2_cache_a4.json`, not the
shared cache.

### Inventory

- `CDDM_std_g0_bias0`: 8 cells, 5 runs each, 50k iterations. Penalties: lambda_rws (recurrent-weight
  sparsity, target in-degree 20) in {0, 0.05} times lambda_frm (firing-rate magnitude, two-sided pull
  toward a rate cap, so it also lifts silent units) in {0, 0.2}, for equation types h and s.
- `CDDM_std_g0_bias0_c75000`: the same 8 cells, 5 runs each, each continuing a distinct 50k run by 25k
  (budget 75k, warm start with Adam state). Participation traces are joined, 0 to 74.9k.
- Spock holds no CDDM sweeps at all (14 folders, none CDDM).
- The metabolic top-up `CDDM_std_g0_metabolic_bias0_c75000` is already a ladder in
  `fig_knobs_at_floor.py`; its `--probe` rows read (job 15527664), 10.0 rung has one run dropped
  (loss ends 7.8x above its floor). Not redone.

### Standardness

All four h cells match the standard network on every listed key. Against the reference control's
whole config they differ only in `light_outputs` (None vs True) and `trainer.track_drift` (False vs
True), both bookkeeping. The four s cells are NOT standard (`equation_type: s`) and were not read.

### Read-outs (5 runs per cell, none excluded; R² at 36 draws)

| cell (75k chain) | active at floor | crossing (k it) | R² at floor (estimated) | R² at 50k (measured) | active at 75k | R² at 75k |
|---|---|---|---|---|---|---|
| no penalty (in-sweep control) | 415 (381-455) | 21-32 | 0.827 | 0.833 | 325 (320-334) | 0.833 |
| rate, lambda_frm 0.2 | 1000 (all 5) | 17-21 | 0.820 | 0.828 | 1000 | 0.827 |
| sparsity, lambda_rws 0.05 | 669 (655-692) | 12.2-12.8 | 0.820 | 0.831 | 265 (248-281) | 0.835 |
| both | 1000 (all 5) | 12.7-13.2 | 0.812 | 0.827 | 1000 | 0.828 |
| reference: drift control, 200k (3 runs) | 418 (416-422) | 26.7-29.0 | 0.831 | - | 337 at 75k | 0.839 at 200k |

Seed-to-seed SD of R² is 0.002-0.004 in every cell. Differences from the in-sweep control (SE of the
difference in brackets): rate arm -0.005 (0.0016) at 50k and -0.007 (0.0022) at 75k; both arms
-0.006 at 50k and 75k; sparsity arm -0.003 at 50k, +0.002 at 75k.

### The predictions

1. In-sweep control 415 active, crossing 21-32k: inside 330-500 and 15-40k. Passed.
2. Oracle (own numpy forward pass, final weights) against the trace's last count: within 6 units in
   every run, per-unit agreement 97.2-100%. Passed, with the 97.2% run (control seed 5) close to the
   97% bar; the trace is 100 iterations older than the weights.
3. R² at 50k against 75k: largest gap 0.0097 (sparsity arm, R² still rising). Passed, narrowly in the
   sparsity cell.
4. Drift control re-read: 417, 422, 416 at 26.7-29.0k, as in the trajectory. Passed.

### What the cells say

- Rate penalty on the standard network: every one of the 1000 units is active at the floor and at
  75k, against 415 and 325 in the control. Its held-out R² is lower by 0.005-0.007, about three
  standard errors, the same sign on both sets of measured weights. That gap is 25-35 times the
  0.0002 gap measured on the pre-standard networks, so "no cost" does not carry over as stated; it
  is still below the ~0.01 by which two different scorers disagree on identical CDDM weights, but
  both cells here went through one scorer at 36 draws.
- The sparsity arm's floor count (669) is a timing artefact, not a remedy. Its noisy loss includes
  the sparsity term, ends 4% BELOW its fitted floor in every run, and crosses at ~12.5k against the
  control's 21-32k. At matched iterations it has 659 vs 482 active at 12.5k, 402 vs 424 at 25k, 297
  vs 359 at 50k, and 265 vs 325 at 75k: by the end it silences more units than no penalty does.
- The stored loss of the penalty arms includes the penalty, so their floor read-outs (and the
  end-based R² estimate at the floor) are of task + penalty. The R² claims above rest on the
  measured 50k and 75k weights, not on the floor estimate.

### Figure

`fig_knobs_at_floor.py` gains a ninth ladder, `slide_k_frm_rws` (categorical, reference = the
in-sweep control). The trade-off grid was a fixed 2x4, so `zip(axes.flat, LADDERS)` would have dropped
the ninth ladder silently; it now has ceil(len/4) rows at 50 mm a row. Its margins were set for two
rows and need checking on the 3-row build. `--tradeoff` was not run.
