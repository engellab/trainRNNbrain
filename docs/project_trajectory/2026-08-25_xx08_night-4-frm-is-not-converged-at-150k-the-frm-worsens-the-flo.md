## 2026-08-25 (night, 4) — ⚠️ frm is NOT CONVERGED at 150k; the "frm worsens the floor" flag is RETRACTED

Checked before starting the penalty characterisation, because the decisive criterion needs the
networks to have settled.

**Two independent measures agree that `frm` has not converged at the 150k penalty budget:**

| condition | W_inp reaches α<0.6 within budget | loss fall over last third |
|---|---|---|
| none (400–500k budget) | 64/64, median 63k | −0.3% … +0.1% |
| **rws** (150k) | **43/43, median 54k, range 40–97k** | 0.0% … +0.4% |
| **frm** (150k) | **0/4 — still BALLISTIC** | **+5.4% … +8.4%** |

frm's α(W_inp) starts at ~1.0 and *ends* at 0.87–1.005, never sustaining a drop below 0.6; its
minimum over the whole run is 0.58–0.74. The loss is still descending ~8% over the final tenth.

### Retraction

The 2026-08-25 flag that **"frm moves the floor the WRONG way (+8.3% at k=1, +9.8% at k=2)"** is
**RETRACTED**. Those were stretched-exponential floors fitted to a curve that is still descending, so
they are not floors at all. The apparent excess (~8–10%) is almost exactly the amount the loss is
still falling over the last third of the run (8.4%). Nothing can be said about frm's floor, in either
direction, from a 150k run.

### Consequences for the penalty programme

- **`rws` is ready to characterise now.** Converged by both measures, the W_inp diffusion criterion is
  available, and N=500 is complete for k=1..8 with N=1000 nearly so.
- **`frm` at 150k is unusable** for anything that assumes convergence — floors, matched-performance
  read-outs, and the diffusion criterion all require it. Only fixed-compute statements are safe, and
  those are exactly the ones the headline analysis showed to be confounded by convergence depth.
- **`both` contains frm and has NOT STARTED** (tasks 145–216, still queued behind rws/frm). It will
  almost certainly hit the same wall. There is still time to raise its budget before it runs.

The 150k budget was justified in the launcher header from CDDM penalty measurements: "the clean task
loss reaches within 2% of its final value at 4.6k–14.6k (rws, frm)". **That transfers to rws on the
flip-flop and does NOT transfer to frm** — a fourth instance of a CDDM-calibrated constant failing on
this task, after the 1e-6 silence threshold and the drift criterion.

---

