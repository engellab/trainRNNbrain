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
- **Force-init-silent test (Pavel's original design) — DONE (job `5103664`, `CDDM_d9e0ec_g0_silentinit`).** Targeted
  inhibition over a fixed random 25% (set S, reconstructable from seed) via `model.inhibitory_boost=2.0`; 20 jobs =
  2 eq × {none, frm} × 5 seeds. **Result: `frm` RESURRECTS** — 100% of dead-at-init S units become active under
  `frm` (both eq), vs ~50–57% under `none`; and under `none` S ends statistically identical to non-S, so the init
  silencing is not durable (fate decided during training). See `plot_silentinit_rescue.py` +
  `docs/project_trajectory.md` (2026-07-02). (Global inhibitory boost was rejected — dims the homogeneous init
  collectively instead of carving a subpopulation.)
- **Master-inhibitor test (replaces the c-sweep) — SUBMITTED (job `5108070`, `CDDM_731df4_g0_masterinhib`).** The
  c-sweep was dropped: cranking `inhibitory_boost` changes weight magnitude but not init silence (a t=1 transient
  survives any c), so it never makes units truly gradient-dead. Instead: one inhibitory unit, driven only by the
  two context cues, holds a fixed fraction silent via deep context-locked inhibition that gradients cannot reach
  (`model.master_inhib_frac ∈ {0.25,0.5,0.75,1.0}`; `rnn_relu_Dale_masterinhib.yaml`,
  `SilentReLU_masterinhib_gamma0_N1000.slurm`; 48 jobs = 2 eq × 4 frac × {none,frm} × 3 seeds). Clean
  "no gradient → no rescue" test; frac=1.0 = all-but-master dead (predict R²≈0, no rescue). Analyse with
  `plot_silentinit_rescue.py` machinery. See `docs/project_trajectory.md` (2026-07-02).
  **RESULT (2026-07-03): prediction wrong — frm rescues 100% at every fraction (incl. 1.0), task solved
  (R²≈0.85).** frm does it *indirectly*: the master is over-cap and active, so frm penalizes and suppresses it
  (peak 1–4.5 → ~0.4), releasing the clamp; the dead targets are never lifted directly. Under `none` the clamp
  holds (~15% recover). So the construction did NOT make truly-unrescuable units — the inhibitor is tamable.
  ~13/48 nets diverged to NaN (excluded). `plot_masterinhib_rescue.py`; docs 2026-07-03.
- **Frozen master inhibitor (truly-unrescuable test) — SUBMITTED (job `5115568`, `CDDM_931680_g0_masterinhib_frozen`).**
  `RNN_torch(freeze_master=True)` freezes the master's I/O (grad hook + forward-pre-hook restore; the grad hook
  alone fails due to Adam+weight_decay). Verified frozen (`verify_master_freeze.py`: master grad 0.374→0.000).
  Same grid as unfrozen (48 jobs). Predict: targets stay silent even under frm; frac=1.0 fails the task (R²≈0).
  Analyse with `plot_masterinhib_rescue.py <folder>`. **RESULT (2026-07-06): freeze verified (master peak pinned
  1.0); `none` arm clean — frozen clamp HOLDS for h (~0% active vs ~10-29% unfrozen), leaks ~15% for s; frac=1.0
  breaks the task (R²≈-0.38). But the `frm` arm is UNINTERPRETABLE — 22/48 diverged to NaN, almost all in frm
  (1 valid net/cond). Central question (can frm rescue a gradient-proof clamp?) NOT answered.** docs 2026-07-06.
- **Stabilized frozen-master rerun — SUBMITTED (job `5116848`, `CDDM_f4b706_masterinhib_frozen_gamma`).** Fights the
  ~46% divergence by turning the cubic term back ON (`gamma=0.1`, `rnn_relu_Dale_masterinhib_frozen_gamma.yaml`,
  `SilentReLU_masterinhib_frozen_gamma_N1000.slurm`); the soft-saturation bounds runaway activity but only bites at
  large positive x, so it doesn't lift the dead targets -> stabilizes without changing the rescue question. Smoke
  test: targets 100% silent, master peak 0.92 (saturating). Goal: >=3 valid frm nets/cond. Analyse:
  `plot_masterinhib_rescue.py CDDM_f4b706_masterinhib_frozen_gamma`. If gamma alone doesn't suffice, next levers:
  lower master_ctx_drive to ~0.3 (not over-cap), tighten max_grad_norm (50->~5), more seeds.
  **RESULT (2026-07-07): gamma helped (18 NaN vs 22) but frm arm still thin (~1 valid/cond). ANSWER: frm overcomes
  even the frozen gradient-proof clamp at frac<1.0 — targets genuinely active (part ~0.1) + task solved (R²~0.83);
  it recruits the non-clamped units to build compensating excitation onto the dead targets. ONLY frac=1.0 (whole
  net clamped) resists: no scaffold left, task fails (R²≈-0.38, degenerate) under none AND frm — confirms the
  thought experiment. Synthesis: silence can't survive frm unless the rest of the net is disabled too. Caveat:
  frac<1.0 frm cells n=1 (consistent across conds + prior runs); frac=1.0 solid (s n=3).** docs 2026-07-07.
- **Stabilized full rescue run — SUBMITTED (job `5127540`, `CDDM_b5fafb_masterinhib_frozen_dt05`).** Fix found by
  cheap frm-only tests: clamp -2 didn't help (~50%, just delayed); `dt=0.5+gamma=0.1` -> ~19% NaN (dt=0.5 alone was
  100%, need BOTH). Full run: dt=0.5+gamma=0.1, frozen, clamp -5, 64 jobs = 2 eq x 4 frac x (3 none + 5 frm) seeds,
  wall 12h (~6h/job). Analyse: `plot_masterinhib_rescue.py CDDM_b5fafb_masterinhib_frozen_dt05`. Should give a
  well-powered read on whether frm rescues a gradient-proof clamp (prior thin answer: yes frac<1.0, no frac=1.0).
- **(Superseded) firm-up pass notes.** NaN root cause
  (docs 2026-07-07, corrected): a forward-dynamics instability, NOT the cubic (gamma=0 runs diverged the same way).
  frm builds a self-exciting recurrent loop (gain>1) onto the targets to overcome the -5 clamp; gamma=0 -> unbounded
  growth, gamma=0.1 -> cubic bounds it but overshoots via explicit Euler for |x|>~14. Grads were ~1 (clip=50
  irrelevant); sudden 1-step NaN. Fixes, most-direct first: (1) milder master_inhib_strength (-5->-2, less
  compensating excitation needed); (2) hard-bound the state in forward(); (3) smaller integration step (dt/tau or
  sub-step); (4) lower lambda_frm. Grad clipping / gamma tweaks act on the wrong layer.
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
