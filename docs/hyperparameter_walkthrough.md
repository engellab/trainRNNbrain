# Hyperparameter walkthrough

Every configuration parameter, what it controls, and — just as important — **whether it is still
load-bearing**. Written 2026-07-27, after the standardisation pass added six new switches and it
became hard to tell live knobs from fossils.

Each parameter carries a status:

| | meaning |
|---|---|
| **CORE** | part of every run; changing it changes the model or the training |
| **SWEPT** | deliberately varied between conditions of an experiment |
| **DORMANT** | present and functional, but a no-op at its default; used by one past experiment |
| **UNUSED** | never set to a non-default value in any config or launcher in this repo |

A **cleanup ledger** with concrete removal candidates is at the end (§6). Nothing has been deleted —
that is a separate, deliberate step.

---

## 1. Global (`configs/base.yaml`)

| Parameter | Status | What it controls |
|---|---|---|
| `seed` | CORE | RNG seed for python/numpy/torch. `"random"` draws from the clock and is written back into the saved config, so a run is always reproducible from its own record. The per-network RNN seed is `seed + 3508` for the single-net case. |
| `n_nets` | CORE | Networks trained per job. **Always 1** in this project — each network is its own SLURM task, which is what makes the init reconstructable from the saved seed. |
| `experiment_tag` | CORE | Output folder under `paths.save_to`. Slashes create nesting: `"std_g0/EqType=h_N=1000_..."` → `CDDM_std_g0/EqType=.../`. |
| `display_figures` | CORE | `plt.show()` on/off. False on clusters. |
| `light_outputs` | CORE (new) | Skips **all** numpy/CPU post-training analysis and writes parameters as `.npz` instead of JSON. Needed above N≈5000, where the analysis costs more than the training (~3 h single-threaded at N=10000, a 10.8 GB float64 array, PCA over 10000×135000, ~4 GB of JSON). Retains config, parameters, participation trace, loss curve, `participation_trace.png`. |
| `paths.save_to` | CORE | Output root. `paths_local` / `paths_DELLA`; Spock and Della scratch are passed on the CLI. |

---

## 2. Model (`configs/model/rnn_relu_*.yaml` → `RNN_torch`)

### 2.1 The dynamics

`dx/dt = −x + W_rec·f(x) + W_inp·u + b − γx³`, integrated by forward Euler with `α = dt/τ`.

| Parameter | Status | Default | What it controls |
|---|---|---|---|
| `N` | SWEPT | 1000 | Number of units. The central independent variable of the size sweep. |
| `activation_args` | SWEPT | `{name: relu, slope: 1.0}` | Activation and its parameter. Used: `relu`, `softplus` (β), `leaky_relu`, `tanh` (slope), `gelu`, `sigmoid`. |
| `equation_type` | SWEPT | `h` | `h` = state is pre-activation, output read from `f(x)`. `s` = state is the rate, activation applied inside the recurrent drive. Both are reported throughout. |
| `dt`, `tau` | CORE | 1, 10 | Only the ratio `α = dt/τ = 0.1` matters. **Lowering `dt` to 0.5 was the fix for the explicit-Euler divergence** in the frozen-clamp experiments — it raises the stability boundary on recurrent loop gain. |
| `gamma` | CORE | **0** | Coefficient of the cubic saturation `−γx³`. **0 = off, the standard regime.** Originally 0.1; removed as a possible confound (it is an activity limiter baked into the dynamics) and it changed nothing. Re-enabled only to stabilise the frozen-clamp runs. |
| `spectral_rad` | CORE | 1.2 | `W_rec` is rescaled to this spectral radius at init. Checked at init across 0.6–1.6 (0% silent everywhere); **never yet varied in trained networks** — a planned rescue axis. |
| `connectivity_density_rec` | CORE | 1.0 | Fraction of non-zero recurrent weights. 1.0 = dense throughout; sparse is a planned axis. |
| `sigma_rec`, `sigma_inp`, `sigma_out` | CORE | 0.05, 0.05, 0.03 | Training noise, scaled as `√(2/α)·σ` (Euler–Maruyama). **The one deliberate non-vanilla choice** in the standard architecture — standard in neuroscience RNNs, absent in vanilla ML ones, and stated in Methods. All evaluation is noise-free. |
| `y_init` | CORE | zeros | Initial state. Its exponential decay is why "silent" units in the `s` equation sit at a shared floor of ~1.7e-14 rather than exactly 0: `(1−α)^300 ≈ 1.9e-14`. |
| `n_inputs`, `n_outputs` | CORE | 6, 2 | Set by the task, not by hand. |
| `RNN_tag` | CORE | `""` | Optional suffix on the output folder. Empty everywhere. |

### 2.2 Architecture switches — the ones added during standardisation

All default to the **legacy** value, so every pre-existing config behaves exactly as before.

| Parameter | Status | Legacy default | Standard-RNN value | What it controls |
|---|---|---|---|---|
| `dale` | SWEPT | `true` | **`false`** | Dale's law: sign-split `W_rec` (E units excite, I units inhibit, ratio `exc2inhR`), the sign re-imposed after every step, and a readout restricted to the excitatory subpopulation. `false` uses `get_connectivity_unconstrained` — signed zero-mean weights, no E/I split, `dale_mask=None`, every unit reads out. |
| `io_nonnegativity` | SWEPT | `true` | **`false`** | Clamps `W_inp ≥ 0` and `W_out ≥ 0`. **Independent of `dale`** — and the distinction matters: this switch, not Dale's law, is what conceals hard zeros. With non-negative input weights and non-negative task inputs, every unit gets a positive push at every timestep, so silent units sit at 1e-9…1e-2 instead of exactly 0. Same silent population, different floor. |
| `self_connections` | SWEPT | `false` | **`true`** | Whether a unit may connect to itself. `false` zeroes the `W_rec` diagonal at init *and* re-zeroes it after every optimizer step via `recurrent_mask`. `true` leaves it free and trainable, as in a standard RNN. |
| `bias_range` | SWEPT | `[0, 0]` | **`[-1, 1]`** | A degenerate range gives a fixed, non-trainable zero bias. A non-degenerate range makes the bias a trainable Parameter, clamped to the range after every step. **±1 is a rail, not a prior**: silent units sit at `x ≈ −0.19` (min −0.296), so `b = +0.30` would lift all of them — the optimum is well inside the bound. |
| `bias_init` | CORE | `"uniform"` | **`"zeros"`** | Only relevant when the bias is trainable. `uniform` (legacy) seeds it uniformly over `bias_range` — which means widening the range also changes the *initial condition*. `zeros` starts every bias at 0, so the initial weights are bit-identical to the bias-free network at the same seed and the manipulation is one-variable. |
| `exc2inhR` | CORE (Dale only) | 4.0 | — | Excitatory:inhibitory ratio. **Ignored entirely when `dale: false`**, though still present in the configs. It sets the E/I redundancy asymmetry: 200 inhibitory units carry the whole network's inhibition at 4× weight and are almost never silent, while 800 excitatory units are mutually redundant and half go quiet. |
| `weight_boundary` | DORMANT | `"sticky"` | inert | How Dale's sign constraint is enforced. `sticky` clamps violating weights to ±`eps` (they pile up at ~0); `reflective` uses `|param|·sign` so nothing is pinned. Tested as a possible artifact — it changed nothing. **Inert once `dale` and `io_nonnegativity` are both false.** |
| `weight_boundary_eps` | DORMANT | 1e-12 | inert | The clamp value. Also the source of the `±1e-12` entries in a Dale network's saved `W_out`. |

### 2.3 One-off experiment machinery — DORMANT, all no-ops at their defaults

These implement the silencing constructions from the prevention-vs-rescue arc. They are **superseded**
by the participation traces, which answer the same question on the natural initialization.

| Parameter | Default | What it did |
|---|---|---|
| `inhibitory_boost` | `None` | Multiplies the inhibitory columns of a fixed random `silent_init_frac` of `W_rec` rows, driving those units net-negative → silent at init. Used to test whether `frm` can resurrect dead-at-init units. |
| `silent_init_frac` | 0.25 | Size of that forced-silent set (drawn from the net's seed, so reconstructable). |
| `master_inhib_frac` | `None` | Fraction of units held silent by a single "master inhibitor" unit. |
| `master_inhib_strength` | 5.0 | Depth of that inhibition. |
| `master_ctx_drive` | 1.0 | Context-cue drive onto the master, its only input. |
| `freeze_master` | `False` | Freezes the master's weights (gradient hook + forward-pre-hook restore) so the penalty cannot tame it. The hook alone is insufficient — Adam plus weight decay reintroduces an effective update. |

---

## 3. Trainer (`configs/trainer/trainer*.yaml` → `Trainer`)

### 3.1 Optimisation

| Parameter | Status | Default | What it controls |
|---|---|---|---|
| `max_iter` | CORE | 30000 | Training iterations. The silent fraction is decided by ~400–600 and frozen, so 5000 suffices for that readout — the basis for truncating the large-N runs (and verified by explicit controls). |
| `lr`, `lr_scale_exp` | CORE | 1e-3, 1/3 | Effective learning rate is `lr·(100/N)^lr_scale_exp`. **N and lr therefore co-vary in any size sweep** — kept for comparability with the original 120-net sweep, and declared in Methods. |
| `weight_decay` | CORE | 1e-6 | Adam weight decay. Applies to the bias too. |
| `max_grad_norm` | CORE | 50 | Gradient clipping. Irrelevant to the NaN divergences that occurred: those were a **forward-dynamics** instability with gradient norms ~1. |
| `same_batch` | CORE | `True` | Train on one fixed batch (all 450 task conditions) rather than resampling. |
| `anneal_noise` | UNUSED | `False` | Would ramp σ up on a sigmoid schedule during training. Never enabled. |
| `trainer_tag` | CORE | `""` | Folder suffix; empty suppresses the default `_BaseTrainer`. |

### 3.2 The switches added during standardisation

| Parameter | Status | Legacy | Standard | What it controls |
|---|---|---|---|---|
| `task_safe_gradients` | SWEPT | `true` | **`false`** | `true` projects out penalty-gradient components that oppose the task gradient, so a penalty can never hurt task performance. `false` is plain descent on `task + Σλ_k·penalty_k`. **Provably a no-op when no penalty is active** — the two paths give bit-identical results in that case. |
| `monitor` | SWEPT | `true` | **`false`** | Per-penalty loss and gradient-norm logging. Costs **one extra backward pass per active penalty**, so switching it off is a real speedup. The total loss curve is saved regardless, as `*_TrainLosses.json`. |
| `track_participation` | SWEPT | `false` | **`true`** | Logs per-unit participation on a noise-free forward pass every `track_every` iterations → `*_ParticipationTrace.pkl`. ~3% overhead. This is what turned "prevention or rescue?" from a contrived experiment into a direct measurement. |
| `track_every` | CORE | 10 | Iterations between snapshots. 10 over 30000 gives 3000 × N float32 ≈ 12 MB at N=1000. |

### 3.3 Penalties

Each `lambda_X` scales a term; each `X_args` passes its keyword arguments. **A λ of 0 means the term
is never even evaluated.**

**In active use:**

| λ | Status | Term | Notes |
|---|---|---|---|
| `lambda_frm` | SWEPT (0 / 0.2) | Firing-rate magnitude: drives a soft-max-over-time rate toward `cap_fr`, penalising both over- and under-shoot | **The one intervention that works** — silent fraction to exactly 0 in every architecture at no task cost. `frm_args`: `cap_fr` 0.3 (scaled ~1/log N), `tau` 0.1 (softmax temperature), `g_top`/`g_bot` 3.0 (exponents above/below cap), `alpha`/`beta` 1.0 (under/over weights), `aggregation` mean vs logsumexp with `tau_n`. |
| `lambda_rws` | SWEPT (0 / 0.05) | Recurrent-weight sparsity: penalises effective in-degree above `tg_deg` (20) | Does **not** rescue. In Dale nets it is worse than baseline; in standard nets it lifts units off exact zero without making them participate — visible only because two metrics are reported. |
| `lambda_met` | SWEPT (0.01–10) | Metabolic cost, `mean(fr²)` — the field-standard form | Being swept now over four decades. Predicted to *deepen* silence, since it penalises rate magnitude. |

**Defined but never used** — all zero in every config and launcher in this repo:

`lambda_orth` (input/output channel overlap) · `lambda_iwm`, `lambda_rwm`, `lambda_owm` (input/recurrent/output weight magnitude caps) · `lambda_hm` (hidden-state magnitude) · `lambda_tv` (across-trial output variability) · `lambda_fri`, `lambda_hi` (Gini/HHI inequality of rates or of hidden drive) · `lambda_htvar`, `lambda_hlvar` (temporal / local variance of hidden drive) · `lambda_cl` (clustering) · `lambda_effdim` (eigenvalue tail energy).

Two of them would **break** on `dale: false`: `rec_weights_magnitude_penalty` with `account4dale: true`
and `h_local_variance_penalty` both index `dale_mask`, which is `None` without Dale. Harmless while
λ=0, since the term is never evaluated — but a trap for anyone who enables them.

### 3.4 Dropout — UNUSED

`dropout: False` everywhere, with a `dropout_args` block (`dropout_kind`, `sampling_method`, `eta`,
`drop_rate`, `dropout_beta`, `activity_q`) and ~17 lines of `Trainer` code including a
participation-based sampling mode. Never enabled in this project. Note `Trainer.get_participation_`
belongs to *this* machinery and is **not** the metric used for the participation traces — that one is
`participation_from_states_`, which matches the offline analysis exactly.

---

## 4. Task (`configs/task/CDDM.yaml`)

`T` 300 steps; cue on 0–300; stimulus 100–300; decision 200–300; loss masked to `(0,100)` and
`(200,300)`; 15 coherences; 6 inputs, 2 outputs → a fixed batch of **450 conditions**. Other tasks
(DMTS, GoNoGo, MemoryAngle) exist in the repo and are untouched — generality is still open.

---

## 5. Experiment group configs (`configs/experiment/`)

These pin the parameters that are *constant across a whole experiment line*, so launchers pass only
swept variables — the rule being that fixed values live in config, and the launcher varies only what
the sweep varies.

| Config | Pins |
|---|---|
| `silent_units_N1000` | N=1000, `max_iter` 30000, empty trainer tag — the original Dale line |
| `silent_units_std` | same, for the standard-RNN line |
| `silent_units_largeN` | `light_outputs: true`, empty trainer tag; N and `max_iter` come from the launcher because both are swept |

---

## 6. Cleanup ledger — candidates for removal

Nothing here has been deleted. Ordered by how much clutter it removes per unit of risk.

| # | Candidate | Risk | Rationale |
|---|---|---|---|
| 1 | **12 unused penalties** (`orth`, `iwm`, `rwm`, `owm`, `hm`, `tv`, `fri`, `hi`, `htvar`, `hlvar`, `cl`, `effdim`) with their `*_args` blocks and ~180 lines of `Penalties` methods | LOW–MED | Never used here, but two are Dale-broken and would mislead anyone who enabled them. **Recommendation: keep the methods, delete the λ/args entries from the trainer configs** and list them in a single "available but inactive" comment. That removes ~60 lines of config noise while preserving anything a collaborator may rely on. |
| 2 | **Dropout** (`dropout`, `dropout_args`, ~17 lines of Trainer code, `get_participation_`) | MED | Never used, but it is a coherent feature someone may have built for another project. Do not delete unilaterally — confirm first. |
| 3 | **6 master-inhibitor / silent-init model configs** and their model parameters | LOW | The experiments they served are superseded by the participation traces. The configs are one-line files, harmless; the six `RNN_torch` parameters are no-ops by default. **Recommendation: keep the model parameters** (they document a real experiment that is in the trajectory) but note in each config that the line is closed. |
| 4 | **10 unreferenced model configs** — `rnn_gelu_alpha{2,5}`, `rnn_tanh_slope{15,20,30}`, `rnn_sigmoid*`, `rnn_softplus_Dale`, `rnn_tanh_Dale`, `rnn_relu_Dale_SE` | LOW | Not referenced by any committed launcher. But `CDDM_tanh_slope=*` and `CDDM_TauNSweep-gelu2-*` **data exists on Della scratch**, so these configs are the only record of how it was produced. **Recommendation: keep, and commit the untracked ones** rather than delete. |
| 5 | `anneal_noise` | LOW | One line, never enabled. Keep or drop; no consequence either way. |
| 6 | `weight_boundary`, `weight_boundary_eps` in the *standard* model configs | LOW | Inert once `dale` and `io_nonnegativity` are false. Currently carried "for symmetry" with a comment saying so. Harmless, mildly confusing. |

**The honest summary:** the real clutter is §3.3 — twelve penalty knobs in every trainer config that
have never been used, two of which are quietly incompatible with the architecture the project has
standardised on. Everything else is either genuinely load-bearing or a small, documented fossil.
