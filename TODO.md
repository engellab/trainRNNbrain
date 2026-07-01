# TODO — silent-ReLU / dead-unit project

Pre-publication checks to exclude trivial mechanisms. See `docs/project_trajectory.md` for the full
record. Status as of the gamma=0 / boundary / activation / noise controls.

## Done
- gamma=0 control (cubic term not a confound) — `CDDM_4a031e_g0`.
- weight_boundary control (sticky vs reflective; silence not a projection artifact) — `CDDM_2bc3c1_g0_reflective`.
- **Activation controls (DONE, `fb2792` re-run):** softplus(beta=25) `5100397` (38/40), leaky-ReLU `5100398` (40/40).
  The key test — do non-dead-gradient activations still show the low-activity mode? **Yes:** all three activations
  concentrate ~40–64% of units into a low-activity mode under none/rws (scale-free criterion); softplus-h is 0%
  hard-dead yet 41–64% relatively silent. → NOT a hard-ReLU dead-gradient artifact. fr penalty collapses all to
  one active mode in every activation. See `docs/project_trajectory.md` (2026-07-01).
- **Noise control (DONE, `fb2792` re-run):** sigma_rec in {0,0.01,0.05,0.1} `5100399` (40/40). **Silence is NOT
  noise-driven:** adding noise never increases it; for h, sigma=0 is a distinct ~80%-silent regime and any noise
  drops it to ~46% (mild regularisation); for s it is ~55–60% and flat. Persists across all realistic noise.
- **Init vs trained identity tracking (DONE):** each net trained as its own job (`n_nets=1`), so init is a pure
  function of the saved config seed (per-net rnn seed = seed + 3508); reconstructed exactly and scored on the same
  batch, per-unit paired. **0% silent at init (both metrics)** across nets → confirms silence emerges in training;
  frm works by **prevention** (init-active units stay active) not resurrection (there are ~no init-silent units to
  resurrect). See `plot_init_vs_trained.py` + `docs/project_trajectory.md` (2026-07-01).
- Now-checks (tmp/check_now.py, not committed): independent oracle = NOT a bug; 0% silent at init across
  spectral_rad = silence emerges in training; only ~2.5% truly dead, rest is a broad low-activity mode
  (threshold-dependent; principled cut at the dip ~0.05 -> ~49%).

## TODO (prioritized)

### HIGH — mechanism: prevention vs resurrection
**Partly resolved by the init-vs-trained tracking (see Done).** Because ~0% of units are silent at init and the
silent population is entirely training-created, `frm` cannot be *resurrecting* dead units — there are none at
init. What the per-unit init→trained scatter shows is that under `none`/`rws` many *init-active* units are driven
silent, and `frm` **prevents** that. So on the *natural* init the mechanism is prevention.
Remaining, to make the prevention claim airtight against the harder case:
- **Force-init-silent test (Pavel's original design) — SUBMITTED (job `5103664`, commit `d9e0ec7`).** Targeted
  inhibition: over-inhibit a fixed random 25% (set S, reconstructable from seed) at init via
  `model.inhibitory_boost=2.0` (`configs/model/rnn_relu_Dale_silentinit.yaml`,
  `slurm/SilentReLU_silentinit_gamma0_N1000.slurm`); 20 jobs = 2 eq × {none, frm} × 5 seeds. Train with `frm` and
  check whether S stays silent (prevention) or climbs into the active mode (resurrection). Global inhibitory boost
  was rejected — it dims the homogeneous init collectively instead of carving a subpopulation (see
  `docs/project_trajectory.md` 2026-07-01). Follow-up: `c ∈ {1.5,3,6}` rescue-difficulty sweep. Hunch: mostly
  prevention; possibly some resurrection for `h`.
- **Logged training:** per-unit activity every ~500 iters for `none` vs `frm` (h & s) — *when* during training the
  silencing happens, and whether any unit recovers. Together these explain *why* `frm` rescues but `rws` does not.

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
