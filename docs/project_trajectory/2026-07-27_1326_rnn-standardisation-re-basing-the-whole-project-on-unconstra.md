## 2026-07-27 — RNN standardisation: re-basing the whole project on unconstrained networks

### Why

Pavel's call, and it is a framing decision rather than a control: **every constraint in the model
makes the result easier to dismiss as niche.** The silent-unit effect had been established only in
Dale-constrained networks with non-negative I/O weights, no bias, no self-connections, and a
custom gradient rule. A reader can wave all of that away as "a biologically-motivated special case".
Most of the RNN literature trains plain unconstrained networks, so from now on **the reference
architecture is a vanilla RNN** and the constrained variants become a supplementary comparison.

The 2026-07-26/27 results made this affordable: silence persists at 42–58% without Dale's law and
without I/O positivity, and a trainable bias changes nothing, so nothing is lost by dropping them.

### The audit — what was still non-standard

Auditing the *saved config of an actually-running net* (not the intent) turned up exactly four
deviations from a textbook continuous-time RNN, plus one deliberate choice:

| Deviation | Status |
|---|---|
| Dale's law on `W_rec` + excitatory-only readout | already switchable (`dale`), now **off** |
| `W_inp`, `W_out` clamped non-negative | already switchable (`io_nonnegativity`), now **off** |
| `W_rec` diagonal zeroed every step — **no self-connections** | new switch, now **on** |
| Penalty gradients projected to never oppose the task gradient | new switch, now **off** |
| Training noise σ_rec = σ_inp = 0.05, σ_out = 0.03 | **kept** — standard in neuroscience RNNs, stated in Methods |

Confirmed already standard: `dx/dt = −x + W_rec·ReLU(x) + W_inp·u + b` with α = dt/τ = 0.1, **γ = 0
so no cubic term**, spectral radius 1.2 init, `y_init = 0`, Adam 1e-3, weight decay 1e-6, grad clip
50, and no active penalties in the `none` cells.

### Parameters introduced (all default to the legacy value, so nothing already run is affected)

| Parameter | Default | Standard-RNN value | Effect |
|---|---|---|---|
| `model.dale` | `true` | **`false`** | signed `W_rec`, no E/I split, every unit reads out, `dale_mask=None` |
| `model.io_nonnegativity` | `true` | **`false`** | no `W_inp ≥ 0`, `W_out ≥ 0` clamps |
| `model.self_connections` | `false` | **`true`** | `W_rec` diagonal free and trainable instead of re-zeroed each step |
| `model.bias_init` | `"uniform"` | **`"zeros"`** | bias starts at 0, so widening `bias_range` adds a degree of freedom without changing the init |
| `trainer.task_safe_gradients` | `true` | **`false`** | plain descent on `task + Σ λ_k·penalty_k` |
| `trainer.monitor` | `true` | **`false`** | drops the per-penalty loss/gradient breakdown, saving one backward pass per active term |

New configs: `configs/model/rnn_relu_standard.yaml`, `configs/trainer/trainer_ptrack_plain.yaml`,
`configs/experiment/silent_units_std.yaml`. Because `monitor=false` was the only thing recording the
loss trajectory, `run_experiment` now **always** writes `{score}_TrainLosses.json` (~300 KB) so the
participation trace can still be aligned against task learning.

**Verification (each a test that could have failed).** Legacy defaults are **bit-identical** to a
pre-change snapshot at the same seed. With `self_connections=true` the diagonal trains to 0.397
instead of returning to zero. `task_safe_gradients` on vs off is **identical** when no penalty is
active — the projection is a no-op there — and differs once `frm` is on (max ΔW_rec 0.0045 over 15
iterations), proving the switch touches only penalty combination. Both paths smoke-tested end to end
through `run_experiment` with `monitor=false`.

### Re-running the experiments (submitted 2026-07-27 ~13:20 EDT, commit `1325549`)

| Array | Folder | Grid | Jobs |
|---|---|---|---|
| `11672037` | `CDDM_std_g0` | 2 eq × 2 λ_rws × 2 λ_frm × 5 seeds, N=1000 | 40 |
| `11672038` | `CDDM_std_g0_Nsweep` | 3 sizes {100, 250, 500} × 2 eq × {none, frm} × 5 seeds | 60 |

Both from their own git worktrees (`~/trainRNNbrain_std`, `~/trainRNNbrain_stdN`) with the
PYTHONPATH guard. `lr` keeps its runtime `(100/N)^0.333` rescaling so the new size curve is directly
comparable to the original 120-net sweep — N and lr co-vary by design, to be stated in Methods.
N=1000 for the size curve comes from the reference sweep's `λ_rws=0` cells.

**Predictions.** Silence persists at ~40–55% under `none`/`rws` with `frm` at 0% and R² ≈ 0.85;
self-connections rescue little, since a positive diagonal only helps a unit that is *already* above
threshold. The size scaling holds (near-0% at N=100 → ~50% at N=1000). **Falsifiers:** under `none`,
below ~15% silent would mean one of the just-removed constraints (most plausibly the zeroed
diagonal) was producing the effect; a flat curve across N would break the "recruits what it needs"
reading entirely.

**Design caveat, recorded now.** `CDDM_std_g0` differs from `CDDM_ptrack_g0_nodale_trainablebias` in
**two** ways at once — self-connections on *and* the gradient projection off. If the two sweeps
disagree, that comparison alone cannot say which change is responsible; isolating it would need one
extra cell (self-connections on, projection still on).

### Planned Round 2 — the remaining rescue pathways

To run on the standard architecture once Round 0 confirms the baseline. Each is a route by which
silent units could plausibly be revived; all use `none` unless stated, 5 seeds, N=1000.

| Axis | Grid | Jobs | Question |
|---|---|---|---|
| **Connectivity scale** | `spectral_rad` ∈ {0.6, 0.9, 1.6, 2.0} × 2 eq | 40 | Pavel's suggestion — is silence simply weak recurrent drive? Only ever checked *at init* (0% silent at every radius), never in trained networks |
| **Connectivity density** | `connectivity_density_rec` ∈ {0.25, 0.5} × 2 eq | 20 | sparser connectivity → fewer, more variable inputs per unit |
| **Activation** | softplus(β=25), leaky-ReLU × 2 eq | 20 | sharper than the Dale version: without I/O positivity, hard zeros are impossible *by construction*, so this isolates whether the low-activity population survives when exact zeros cannot occur |
| **Recurrent noise** | σ_rec ∈ {0, 0.01, 0.1} × 2 eq | 30 | σ=0 was a distinct ~80%-silent regime in Dale nets |

~110 jobs, ~150 GPU-hours. Trimming to 3 seeds on the exploratory axes roughly halves that; the
sensible order is to run whichever axis Round 0 makes most urgent rather than all four at once.

Still open and untouched by any of this: the **spare-capacity** question (§5.1 of `paper.md`) —
scale the *task* rather than the network, and show whether the distributed solution buys anything
(lesion/noise robustness, generalisation) using networks we already have.

