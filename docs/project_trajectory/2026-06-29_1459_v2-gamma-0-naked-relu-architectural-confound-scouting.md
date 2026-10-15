## 2026-06-29 — v2: gamma=0 ("naked" ReLU) + architectural-confound scouting

### Why

Before over-interpreting the v1 results, we scouted the model/training code for architectural and
parameter choices that could contaminate the "naked, field-standard ReLU-Dale RNN" reading. Three
matter; one is removed in this v2 run, two are flagged for follow-up.

### Confounds found (code refs: `trainRNNbrain/rnns/RNN_torch.py`, `trainRNNbrain/trainer/Trainer.py`)

1. **Cubic term `−gamma·x³` in the dynamics** (`RNN_torch.rhs`, active when `gamma > 1e-8`; config
   default `gamma=0.1`). It is a soft saturation — a *built-in activity-magnitude limiter baked into
   the dynamics* — which confounds the `lambda_frm` (firing-rate-magnitude) comparison: the v1
   baseline already had some magnitude control. **Removed in v2 via `model.gamma=0`** (config
   override, no code change; cubic skipped).

2. **Dale sign-projection is a clamp-to-eps** (`Trainer.enforce_dale_`, run after *every*
   `optimizer.step()`): a `W_rec` weight whose sign violates Dale is hard-set to `±1e-12`, i.e.
   **pinned at ~0, not reflected**. This is projected gradient descent onto the Dale orthant; a
   weight whose gradient keeps pushing it into the forbidden sign **sticks at ~0**. Over 30k steps
   this causes **emergent sparsification of `W_rec` independent of `lambda_rws`**, so part of the
   baseline's dead-unit / sparsity signature is a projection artifact rather than intrinsic to
   "naked ReLU". **Not changed in v2** (it is part of the Dale model); see follow-up below.

3. **Bias fixed at 0** (`bias_range=[0,0]`, `bias_trainable=False`): a ReLU unit with net-negative
   input across all conditions is silent with **no bias to lift it** into the active regime, which
   likely *inflates* the silent-unit count. A trainable bias is arguably more standard. **Not
   changed in v2** (kept for a clean one-variable comparison).

   Secondary (unchanged, lower expected impact): exc-only readout with `W_out ≥ 0` (inhibitory
   units never directly drive the output); no self-connections (`W_rec` diagonal zeroed each step);
   `spectral_rad=1.2` init; training noise `σ_rec=σ_inp=0.05`. Penalties are applied via a
   task-safe gradient projection (penalty-gradient components opposing the task gradient are
   dropped) — irrelevant to the no-penalty baseline.

### v2 experiment (this run)

**Single change from v1: `model.gamma=0`.** Everything else identical, **N=1000 only**.

- Grid (40 jobs) = 2 eq {h, s} × 2 `lambda_rws` {0, 0.05} × 2 `lambda_frm` {0, 0.2} × 5 nets.
- Code commit `4a031e` (gamma is a config override). Output root folder **`CDDM_4a031e_g0/`**
  (`_g0` = gamma 0), leaf `EqType=<eq>_N=1000_LmbdRWS=<rws>_LmbdFR=<frm>/`.
- Script: [`slurm/SilentReLU_ReluDale_gamma0_N1000.slurm`](../slurm/SilentReLU_ReluDale_gamma0_N1000.slurm)
  (`--mem=32G` — all N=1000; `--array=1-40`, no throttle). Submitted **2026-06-29 ~14:50 EDT**,
  SLURM array job **`5096100`** (smoke test `5095648` COMPLETED; confirmed `model.gamma=0` in the
  saved config). Re-run the v1 analysis scripts pointed at `CDDM_4a031e_g0` to compare.

#### v2 results — gamma=0 (sticky), N=1000

The full v1 figure set regenerated on `CDDM_4a031e_g0` (the analysis scripts now take a sweep-folder
arg; figures suffixed `_4a031e_g0`). gamma=0 was run for **N=1000 only**, so the N=100/500 groups in
the bar chart are empty. **Removing the cubic term barely changes the silent-unit story — the cubic
was not a confound here.** Silent units (dead, peak rate < 0.01), v1 (cubic=0.1) → v2 (gamma=0):

| eq · penalty (N=1000) | v1 (cubic=0.1) | v2 (gamma=0) |
|---|---|---|
| h · none | 47% | 44% |
| h · rws-only | 54% | 53% |
| s · none | 54% | 55% |
| s · rws-only | 56% | 56% |
| any · fr-only / both | 0% | 0% |

![Silent units per condition — gamma=0](../img/internal_figures/silent_units_per_condition_4a031e_g0.png)

![Participation histograms, h — gamma=0](../img/internal_figures/participation_histograms_h_4a031e_g0.png)

![Participation histograms, s — gamma=0](../img/internal_figures/participation_histograms_s_4a031e_g0.png)

![R² vs 1/HHI, N=1000 — gamma=0](../img/internal_figures/r2_vs_hhi_N1000_4a031e_g0.png)

![Least-participating unit activity, h — gamma=0](../img/internal_figures/least_unit_activity_h_N1000_4a031e_g0.png)

![Least-participating unit activity, s — gamma=0](../img/internal_figures/least_unit_activity_s_N1000_4a031e_g0.png)

> Repro note: the runtime `git rev-parse` folder hash scattered these 40 nets across
> `CDDM_4a031e_g0` and `CDDM_2bc3c1_g0` (Spock's checkout was advanced mid-sweep for the reflective
> code) — all gamma=0 sticky, behaviour-identical; consolidated into `CDDM_4a031e_g0` for analysis.
> Lesson: pin the hash in `experiment_tag` instead of computing it per-job at runtime.

### Planned follow-up (to discuss / next)

Make the Dale boundary behaviour an explicit, switchable config parameter rather than a hard-coded
clamp — e.g. **`weight_boundary: sticky | reflective`**:
- `sticky` = current behaviour (clamp sign-violating weights to `±eps`, pinned at 0).
- `reflective` = reflect across zero (`W ← |W| · sign_mask`) or reparametrize `W_rec = |θ| · sign_mask`,
  so weights are not pinned at the boundary.
Then re-run the gamma=0 sweep under both to quantify how much of the baseline silence is an artifact
of the `sticky` projection. (Also still open from v1: trainable-bias variant; decoding/selectivity
of the marginally-revived units.)

### v2b — `weight_boundary` implemented + reflective gamma=0 sweep

The `weight_boundary` parameter is now in the model (commits `7fdf63`, `2bc3c16`):
- **`sticky`** (default, legacy): raw `W_rec`; the Trainer clamps sign-violating entries to
  `±weight_boundary_eps` (default `1e-12`) each step. Behaviour is byte-identical to before
  (validated by `trainRNNbrain/rnns/check_weight_boundary.py`).
- **`reflective`**: the effective weight is `|param|·sign·mask`, enforced in `RNN_torch.forward`;
  the Trainer skips the post-step projections, so weights are never pinned at the boundary.
- `get_params` exports the **effective** (Dale-compliant) weights, so `RNN_numpy` and every analysis
  script are boundary-agnostic and reconstruction is correct even if the mode is unknown; `__init__`
  / `set_params` default to `sticky`/`1e-12` (legacy fallback for pre-existing nets).
- Config fields: `model.weight_boundary`, `model.weight_boundary_eps` (in `rnn_relu_Dale.yaml`).

**Reflective sweep (this run).** Same grid as the gamma=0 sweep (2 eq × 2 `lambda_rws` × 2
`lambda_frm` × 5 nets, N=1000, gamma=0) **plus `model.weight_boundary=reflective`**. Code commit
`2bc3c16`, output folder **`CDDM_2bc3c1_g0_reflective/`**. Script:
[`slurm/SilentReLU_ReluDale_gamma0_reflective_N1000.slurm`](../slurm/SilentReLU_ReluDale_gamma0_reflective_N1000.slurm).
Submitted **2026-06-29 ~15:35 EDT**, SLURM array **`5096453`**. Validated before launch by a local
20-iter run (config records `weight_boundary: reflective`, saved `W_rec` Dale-compliant, no eps-pile)
and by `check_weight_boundary` passing on the Spock L40S.

**Comparison axis:** `CDDM_4a031e_g0` (gamma=0, **sticky**) vs `CDDM_2bc3c1_g0_reflective`
(gamma=0, **reflective**). Note: because the `sticky` path is behaviour-preserving, the later jobs of
the `4a031e_g0` sweep actually executed at `7fdf63`/`2bc3c16` in sticky mode (Spock's checkout was
advanced for the reflective code) — identical results, only the recorded commit differs per job.

#### v2b results — gamma=0 reflective (N=1000)

Same figure set on `CDDM_2bc3c1_g0_reflective` (figures suffixed `_2bc3c1_g0_reflective`; N=1000
only). **The reflective boundary does NOT reduce silent units — it has as many or slightly *more*
than sticky.** So the silent-ReLU phenomenon is intrinsic to the unpenalized / rws-only ReLU-Dale
net, **not an artifact of the sticky clamp-to-eps projection.** This was a genuine falsification
test: reflective never pins weights at the boundary, so it *could* have shown far fewer silent
units — it didn't. The firing-rate-magnitude penalty still rescues to exactly 0 in every condition.

Silent units (dead, peak rate < 0.01), N=1000 — gamma=0 **sticky** vs **reflective**:

| eq · penalty | sticky | reflective |
|---|---|---|
| h · none | 44% | 46% |
| h · rws-only | 53% | 58% |
| s · none | 55% | 55% |
| s · rws-only | 56% | 63% |
| any · fr-only / both | 0% | 0% |

![Silent units — gamma=0 reflective](../img/internal_figures/silent_units_per_condition_2bc3c1_g0_reflective.png)

![Participation histograms, h — gamma=0 reflective](../img/internal_figures/participation_histograms_h_2bc3c1_g0_reflective.png)

![Participation histograms, s — gamma=0 reflective](../img/internal_figures/participation_histograms_s_2bc3c1_g0_reflective.png)

![R² vs 1/HHI, N=1000 — gamma=0 reflective](../img/internal_figures/r2_vs_hhi_N1000_2bc3c1_g0_reflective.png)

![Least-participating unit activity, h — gamma=0 reflective](../img/internal_figures/least_unit_activity_h_N1000_2bc3c1_g0_reflective.png)

![Least-participating unit activity, s — gamma=0 reflective](../img/internal_figures/least_unit_activity_s_N1000_2bc3c1_g0_reflective.png)

**Sanity check — reflective was active.** Pooled off-diagonal `|W_rec|` (h, N=1000, gamma=0):
sticky pins **19% (none) / 32% (both)** of recurrent weights at exactly the `eps=1e-12` clamp (the
spike at 1e-12); reflective has **0%** there (smallest weight ~4e-11, continuous) — confirming
`|param|·sign` was in effect with no boundary pinning. Script:
[`plot_weight_distribution.py`](../trainRNNbrain/experiments_and_analysis/plot_weight_distribution.py).

![|W_rec| distribution, sticky vs reflective](../img/internal_figures/weight_distribution_sticky_vs_reflective_h_N1000.png)

Notably, sticky pins a *large* fraction of recurrent weights at ~0, yet that does **not** produce
more silent units than reflective — so the unit-level silence is driven by the training dynamics,
not by which weights sit at the Dale boundary.

**Conclusion across the boundary controls:** neither removing the cubic term (gamma=0) nor
switching the Dale boundary (sticky → reflective) changes the picture — unpenalized / rws-only
ReLU-Dale nets are ~45–63% silent at N=1000, and the firing-rate-magnitude penalty is what rescues
them (to 0), regardless of these architectural choices. The silent-ReLU result is robust.

---

