# TODO — silent-ReLU / dead-unit project

Pre-publication checks to exclude trivial mechanisms. See `docs/project_trajectory.md` for the full
record. Status as of the gamma=0 / boundary / activation / noise controls.

## Done / running
- gamma=0 control (cubic term not a confound) — `CDDM_4a031e_g0`.
- weight_boundary control (sticky vs reflective; silence not a projection artifact) — `CDDM_2bc3c1_g0_reflective`.
- **Activation controls (running):** softplus(beta=25) `5100301`, leaky-ReLU `5100302` — the key test:
  does a non-dead-gradient activation still show the low-activity mode? (CDDM_f8be3e_g0_softplus25 / _leakyrelu)
- **Noise control (running):** sigma_rec in {0,0.01,0.05,0.1} `5100303` — is the silence noise-driven? (CDDM_f8be3e_g0_noise)
- Now-checks (tmp/check_now.py, not committed): independent oracle = NOT a bug; 0% silent at init across
  spectral_rad = silence emerges in training; only ~2.5% truly dead, rest is a broad low-activity mode
  (threshold-dependent; principled cut at the dip ~0.05 -> ~49%).

## TODO (prioritized)

### HIGH — mechanism: prevention vs resurrection
Log per-unit activity every ~500 iters during training for `none` vs `frm` (h and s, N=1000, a few seeds,
fixed/known seeds so init is reconstructible). Questions:
- Does `frm` keep units alive from the start (prevention) or revive already-low units (resurrection)?
  Hunch: mostly prevention; possibly some resurrection for the h equation.
- Identity tracking: are the units that are low/dead at init the same ones low/dead at convergence?
  (Couldn't do from saved data — seed not stored. This logged run also answers it.)
- This single experiment explains *why* `frm` rescues but `rws` does not.

### MED — trainable bias (the obvious reviewer question)
Rerun the gamma=0 grid with a trainable bias (`bias_range=[-b,b]`, `bias_trainable=True`). With no bias a
ReLU unit with always-negative input cannot be lifted into the active range. If silence collapses ->
claim narrows to "*bias-free* ReLU-Dale nets"; if it persists -> much stronger. (Lower priority only because
many other controls are in flight.)

### MED — second task (generality)
Repeat on DMTS / GoNoGo / MemoryAngle (configs in repo). Is the low-activity mode CDDM-specific?

### LOW / needs design — task dimensionality (clarified)
A 1000-unit net on a low-dimensional task trivially has spare units, so "unused capacity" is not itself
surprising. The probe: vary task difficulty / dimensionality (more contexts, compositional variants) and
ask whether the low-activity fraction *shrinks as the task could use more units* (-> just spare capacity)
or *persists regardless* (-> a genuine pathology). Also quantify the solution's effective dimensionality
(participation ratio) vs the active-unit count and the task's intrinsic dimension.

### LOW — ablation (likely redundant)
Zero the silent units, check R^2 unchanged; ablate matched random active units, check R^2 drops. Probably
redundant since "silent" is already per-unit peak rate ~0 — only informative if a weird bug let silent
units influence others. Do only if a reviewer insists.

### Deprioritized / not planned
- Trained `spectral_rad` sweep — the init check already shows 0% silent at init for all spectral_rad, so
  the silence is not an init-scale artifact; a trained sweep is low value.
- More seeds (n>=10), lr / optimizer / training-duration robustness — not planned (low expected payoff).
