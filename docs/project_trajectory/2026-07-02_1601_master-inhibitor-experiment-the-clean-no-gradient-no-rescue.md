## 2026-07-02 — master-inhibitor experiment: the clean "no gradient → no rescue" test (submitted)

**Why the inhibitory-boost result needs a follow-up — the gradient argument.** A truly-dead ReLU unit (input
negative at every timestep) has `ReLU′ = 0` throughout, so the gradient to its incoming weights is zero —
*including* from the `frm` under-penalty (which rewards sub-`cap` units for firing more, but that reward still
flows through `ReLU′`). So **no firing → no gradient → no rescue** — a fully-dead unit cannot be pulled up by its
own weight updates. Why, then, did the `inhibitory_boost` units get rescued? Two reasons, both of which we can
now see are artefacts of that construction: (i) their "silence" left a **t=1 initial-condition transient** (peak
~0.003, occurring at the first timestep before the recurrent inhibition acts — verified c-invariant from c=2 to
c=64), so they were never truly gradient-dead; and (ii) the inhibition onto them came from the **general
I-population, whose activity drifts as it trains**, so the clamp weakened and the active network could lift them.
The originally-planned `c`-sweep does not fix this — cranking `c` changes the inhibitory *weight magnitude* but
not the init silence (still that t=1 transient), so it would never produce genuinely gradient-dead units. **The
`c`-sweep is therefore replaced by this cleaner design.**

**The construction (no biases).** One inhibitory unit — the "master inhibitor" — is made a **context-locked
clamp**: it is driven ONLY by the two CDDM context cues (input channels 0 and 1, on throughout every trial) with
weight `master_ctx_drive`, receives **no recurrent input** (so the network cannot silence it), and projects deep
inhibition (`-master_inhib_strength`, default 5) onto a fixed random fraction `master_inhib_frac` of the other
units **and to no one else**. The target set is drawn from the net's seed (reconstructable). Verified at init
(`calibrate_master_inhibitor.py`): master active (peak = `ctx_drive`), **targets 100% silent, non-targets
untouched**, for every fraction. Implemented as a no-op-by-default block in `RNN_torch.__init__`.

**Why this is the clean test.** The clamp is **frozen against gradient descent**: the two weights that could
release a target — master→target, and context→master — both influence the loss only *through the dead target's
zero-derivative ReLU*, so both get ~zero gradient; and the master itself has no recurrent input to be silenced by.
The inhibition is therefore sustained across the whole trial and unreachable by training — the genuine
"no gradient → no rescue" condition, unlike the drifting, transient-leaking `inhibitory_boost`.

**The run (submitted — job `5108070`, commit `731df4`).** ReLU-Dale, γ=0, N=1000. Grid = 48 jobs: 2 equations
(h, s) × **4 silenced fractions** `master_inhib_frac ∈ {0.25, 0.5, 0.75, 1.0}` × 2 penalties (`none`, `frm=0.2`) ×
3 seeds. `frac = 1.0` = every unit except the master silenced — Pavel's extreme thought experiment (predict:
no gradient anywhere, R²≈0, nothing revives). Config `configs/model/rnn_relu_Dale_masterinhib.yaml`; launcher
`slurm/SilentReLU_masterinhib_gamma0_N1000.slurm`. Output → `data/trained_RNNs/CDDM_731df4_g0_masterinhib/`.

**Question / predictions.** Does `frm` revive the master-clamped units? Prediction: **no** (or far less than the
boost case) — with the clamp frozen and no firing to seed a gradient, the targets should stay silent even under
`frm`, and the effect should worsen with fraction; at `frac = 1.0` the network cannot learn the task at all
(R²≈0). Contrast with the `inhibitory_boost` result (100% rescued) would confirm that rescue there depended on
the transient/drift footholds, and that genuinely gradient-isolated units are unrescuable — pinning down the
exact boundary of `frm`'s reach.

**Read-out (planned).** Reconstruct init, regenerate the target set from the seed, and (as in
`plot_silentinit_rescue.py`) compare target-active% under `none` vs `frm` across the four fractions, plus task R².

### Result (2026-07-03) — the prediction was WRONG: frm rescues by killing the inhibitor, not the targets

Analysed by [`plot_masterinhib_rescue.py`](../trainRNNbrain/experiments_and_analysis/plot_masterinhib_rescue.py)
(35 valid nets; ~13 of 48 diverged to NaN and were excluded — see caveat). Init check passes: targets 100% silent,
master active. Trained target-active% (peak ≥ 0.01) and the master's own peak, mean over valid nets:

| eq | frac | `none`: master peak / T active | `frm`: master peak / T active | R² (both) |
|----|----|----|----|----|
| h | 0.25 | 0.94 / 12.7% | 0.62 / **100%** | ~0.85 |
| h | 0.50 | 1.43 / 10.2% | 0.12 / **100%** | ~0.85 |
| h | 0.75 | 2.05 /  9.6% | 0.43 / **100%** | ~0.85 |
| h | 1.00 | 4.53 / 29.3% | 0.44 / **100%** | ~0.85 |
| s | 0.25–0.75 | 1.5–1.8 / ~18% | 0.38–0.45 / **100%** | ~0.85 |

![Master-inhibitor rescue](../img/internal_figures/masterinhib_rescue.png)

**Contrary to the prediction, `frm` fully rescues the master-clamped units (100% at every fraction, including
frac=1.0), and the task is solved throughout (R²≈0.85).** The gradient argument was *locally* correct — the dead
targets get ~no direct gradient — but it missed the escape route:

- **`frm` rescues indirectly, by suppressing the inhibitor itself.** The master is *active* and, at ~1–4.5, sits
  far **above** the `frm` target `cap` (~0.3). `frm` penalizes over-cap units, and because the master fires
  (`ReLU′ > 0`) that gradient is real — it reduces the master's context drive, collapsing its peak to ~0.1–0.6.
  With the clamp's source suppressed, the targets' input rises, they start firing, and *then* their own gradients
  flow and they climb into the active mode. The dead targets are never lifted directly; `frm` removes the cause.
- **Under `none` the clamp holds:** the master stays hyperactive (peak 1–4.5, nothing pressures it down) and only
  ~10–30% of targets recover (via incidental/task-gradient footholds), roughly flat across fraction.
- **Even frac=1.0 recovers under `frm`:** the single always-active master is `frm`-attackable, so taming it
  releases everyone and the net solves the task — the "no gradient anywhere" prediction fails precisely because
  the inhibitor is an active, penalizable unit.

**What this means.** `frm`'s reach is broader than "revive units that occasionally fire": it will also dismantle a
*structured* silencing mechanism if that mechanism runs through an active, penalizable unit. The construction did
**not** produce truly-unrescuable units, because the master's weights are trainable and its over-cap activity
makes it a `frm` target. The genuinely-unrescuable case (Pavel's thought experiment) requires the inhibition
source to be **un-tamable** — e.g. freeze the master's weights (non-trainable `W_inp`/`W_rec` for that unit) so
`frm` cannot reduce it. That is the clean follow-up.

**Caveat — divergence.** ~13/48 nets trained to NaN (folder score `nan`), disproportionately under `frm`; the
master's hyperactivity (peak up to 4.5) plus the `frm` pressure likely destabilizes training. Excluded from the
means above; the surviving nets are internally consistent (every condition retained ≥1, most ≥2–3). A rerun with
gradient clipping or a smaller `master_ctx_drive`/`master_inhib_strength` would firm up the counts, but the
qualitative result (frm 100% vs none ~15%) is unambiguous.

