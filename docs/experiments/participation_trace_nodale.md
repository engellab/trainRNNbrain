# Experiment: unconstrained (non-Dale) networks — is the silent mode a Dale artifact?

*This file is copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

**Sibling sweeps** (same grid, same code commit, same seeds procedure — do not mix):

| Folder | Model |
|---|---|
| `CDDM_ptrack_g0` | Dale + non-negative I/O, no bias — the reference |
| `CDDM_ptrack_g0_trainablebias` | as above + trainable bias `[-1,1]` |
| `CDDM_ptrack_g0_nodale` | **this one**: `dale: false`, `io_nonnegativity: false` |

Trace file format and metric definitions: [participation_trace.md](participation_trace.md).

## Question

Every silent-unit result in this project was obtained in **Dale-constrained** networks: `W_rec` is
sign-split (excitatory units excite, inhibitory units inhibit, ratio `exc2inhR = 4`), the sign is
re-imposed after every optimizer step, the readout is restricted to the excitatory subpopulation, and
`W_inp`/`W_out` are clamped non-negative. That is a strong and unusual set of restrictions, and most
of the RNN literature uses unconstrained networks. Two things follow:

1. **Reach.** As it stands the result describes Dale networks and can be dismissed as a Dale curiosity.
2. **Cause.** The constraint could plausibly *create* the effect — a sign-constrained weight cannot
   change its role, and a unit whose useful contribution would require mixed signs is stuck.

Previous work tested only two *implementations of the Dale boundary* (`sticky` clamp-to-eps vs
`reflective` `|w|·sign`), which changed nothing. This sweep removes the constraints themselves.

## What motivated it: silence is an excitatory phenomenon

Measured on trained `h`, N=1000, γ=0 nets from `CDDM_4a031e_g0` (3 nets per condition), splitting the
silent population by unit type:

| Condition | overall silent | E units (800) | I units (200) |
|---|---|---|---|
| `none` | 43–45% | **53–55%** | **3.5–5.0%** |
| `rws` | 52–54% | 58–60% | 27–29% |

Under `none`, inhibitory units are almost **never** silent while more than half the excitatory units
are. This **falsifies** the natural hypothesis that the excitatory-only readout starves inhibitory
units of gradient and silences them — the opposite happens. The likely reason is load-bearing
redundancy: with `exc2inhR = 4`, 200 inhibitory units supply the inhibition for the whole network at
4× weight, so each is individually indispensable, while 800 excitatory units are mutually redundant
and half can be dropped.

That asymmetry is itself a Dale-specific structure. In an unconstrained network there is no E pool and
no I pool, so no subpopulation is structurally redundant in the same way — which is exactly why this
experiment is needed.

## Implementation

Two independent switches added to `RNN_torch` (both default `true`, so every prior config is
unaffected — verified bit-identical initial weights at the same seed):

- **`model.dale`** — `false` uses `get_connectivity_unconstrained`: signed zero-mean weights, no E/I
  split, `dale_mask = None`, **every** unit reads out; same 1/√N scale, zero diagonal and
  spectral-radius rescaling as the Dale version. `Trainer.enforce_dale_` is skipped.
- **`model.io_nonnegativity`** — `false` skips the `W_inp ≥ 0`, `W_out ≥ 0` clamps.

This sweep sets **both to `false`** (`configs/model/rnn_relu_noDale.yaml`). The intermediate cells
(Dale recurrence with a signed readout, or unconstrained recurrence with positive I/O) are one
override away if the result warrants isolating them.

## Prediction

Silence **persists** at a broadly similar level (~40–55% under `none`/`rws`), because the driver
appears to be that CDDM needs far fewer units than the network has (unpenalized nets solve it with an
effective ~60–150 units regardless of N), and that is untouched by removing sign constraints. The E/I
asymmetry disappears by construction; the distribution may be somewhat less extreme.

**Falsifier:** if the silent fraction under `none` drops below ~15%, Dale was doing the work and every
claim in the project narrows to Dale-constrained networks — a much smaller result, but the honest one.

## Analysis caveats specific to this sweep

- **E/I breakdowns do not apply.** The replacement question: is the silent set random, or does it align
  with some other structure (readout weight magnitude, input drive, in-degree)?
- `run_experiment.py` groups units by the sign of their outgoing recurrent weights for the sorted /
  clustered figures; that is meaningless without Dale, so it now falls back to a single group when
  `model.dale` is false. `sorted_matrices.png`, `avg_responses.png` and
  `intercluster_connectivity_matrices.png` should be read accordingly.
- Silent counts, participation histograms, HHI/R², least-unit heatmaps and the participation trace are
  all Dale-agnostic and directly comparable to the sibling sweeps.
- Two penalties that consume `dale_mask` (`rec_weights_magnitude` with `account4dale`,
  `h_local_variance`) are inactive here (λ = 0); they would need a guard before being used without Dale.

## Grid — 40 jobs

`2 equation types {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds`, N=1000, γ=0, 30000
iterations, participation tracked every 10 iterations. Identical to `CDDM_ptrack_g0` except
`model=rnn_relu_noDale`.

Launcher: `slurm/SilentReLU_ptrack_nodale_gamma0_N1000_della.slurm`, submitted from a **separate git
worktree** on Della so that the in-flight sibling sweeps keep running the code they started with.
