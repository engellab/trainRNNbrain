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
