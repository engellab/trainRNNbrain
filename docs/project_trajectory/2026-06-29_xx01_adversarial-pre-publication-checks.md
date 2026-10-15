## 2026-06-29 — adversarial pre-publication checks

Goal: exclude trivial mechanisms before claiming the result. Checks run on existing data
(`tmp/check_now.py`, deliberately not committed — one-off); controls submitted as new sweeps;
remaining items in [`TODO.md`](../TODO.md).

**Now-checks (h, N=1000):**
1. **Not a code bug.** An independently-written from-scratch Euler integrator (no `RNN_numpy`/`RNN_torch`)
   reproduces the production per-unit peak rates **exactly** (max abs diff 0.0e+00, silent-set Jaccard 1.000).
2. **Not an init artifact.** Fresh untrained nets are **0% silent** at every `spectral_rad` ∈ {0.6,1,1.2,1.6}
   — the silence **emerges during training**, it is not present at initialisation.
3. **Refinement — it's a low-activity *mode*, not "dead ReLUs".** The peak-rate distribution is bimodal:
   a tight active mode (~0.3–0.5) and a broad low-activity mode (~1e-3, spanning 1e-5–1e-1). Only **~2.5%**
   are *truly* dead (peak≈0); the silent fraction is therefore threshold-dependent (13% at <1e-4, 44% at
   <1e-2, 49% at <5e-2). The principled cut is the inter-mode dip (~0.05) → ~49% in the low-activity mode.
   Penalised nets have a single tight active mode (every unit peaks >0.1). **Report the distribution, not a
   single thresholded number.**

**Controls submitted (gamma=0, N=1000, same grid):**
- **Activation:** softplus(β=25) `CDDM_f8be3e_g0_softplus25` (`5100301`), leaky-ReLU `CDDM_f8be3e_g0_leakyrelu`
  (`5100302`) — both have nonzero gradient everywhere (no dead-gradient trap). The decisive test: if they
  still show the low-activity mode, the phenomenon is "trained RNNs concentrate computation", not a hard-ReLU
  pathology; if not, the near-zero tail is ReLU-specific.
- **Noise:** sigma_rec ∈ {0,0.01,0.05,0.1}, none penalty `CDDM_f8be3e_g0_noise` (`5100303`) — is the silence
  noise-driven?

Deferred (see TODO): trainable-bias control, prevention-vs-resurrection mechanism (logged training),
second task, task-dimensionality probe.

