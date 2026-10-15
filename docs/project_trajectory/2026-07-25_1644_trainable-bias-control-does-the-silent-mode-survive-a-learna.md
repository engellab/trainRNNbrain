## 2026-07-25 — trainable-bias control: does the silent mode survive a learnable offset? (CANCELLED — replaced by the unconstrained version, see the 17:00 entry)

> **Status: cancelled and superseded.** Array `11610299` was cancelled at ~16:50 EDT after 4 of 40 tasks
> had run 9–13 minutes; no data was written (all outputs are saved at the end of training), so nothing
> was lost. The reasoning below — why a trainable bias is the right control, how the ±1 range was
> measured, the `bias_init` implementation and the DC-offset caveat — is unchanged and carries over to
> the replacement sweep, which runs the same bias manipulation on **unconstrained** networks
> (`CDDM_ptrack_g0_nodale_trainablebias`). The Dale+bias cell is simply not being collected.

### Why

Every silent-unit result in this document was obtained with `bias_range=[0,0]` — **no bias at all**. A ReLU unit
whose total input is negative at every timestep then has nothing to lift it into the active range, so the standing
objection is that the ~45–55% silent population is an artifact of an unusually constrained model rather than a
property of trained RNNs. A trainable bias is arguably the more standard architecture. This is the MED-priority
"trainable bias" item in [`TODO.md`](../TODO.md), and the most obvious reviewer question left open.

- If silence collapses → every claim narrows honestly to *bias-free* ReLU-Dale networks.
- If it persists → the result is much stronger and the objection is closed.

### Choosing the bias range — measured, not guessed

The `h` dynamics are `dx/dt = −x + W_rec·r + W_inp·I + b`, so at steady state `x* = drive + b`: the bias is
commensurate with the state and directly comparable to how negative a silent unit sits. Measured on a trained
`h`/`none` N=1000 net from `CDDM_4a031e_g0` (43.3% silent):

| | time-averaged state `x` |
|---|---|
| silent units (n=433) | median **−0.187**, p5–p95 −0.238 … −0.156, most negative **−0.296** |
| active units (n=567) | median −0.154, p95 +0.087 |

So **`b = +0.30` lifts 100% of the silent units to threshold** (+0.24 lifts 95%), against an active-unit median
peak rate of 0.23 and `cap_fr = 0.3`. The range is set to **±1** — a rail, not a prior: the optimum sits at
~0.2–0.3, well inside, so training is unconstrained in practice and "the bias could not reach far enough" is not
an available objection. (±0.3 would have been the measured-sufficient alternative; ±1 was chosen for robustness
to that critique.)

### `bias_init` — new option, so this stays a one-variable change

`RNN_torch` previously tied the bias *initialisation* to the range: any non-degenerate `bias_range` seeded the
biases **uniformly over that range** (`RNN_torch.__init__`). With ±1 that would start the network with offsets 5×
the drive scale — a different initial condition, breaking both the one-variable comparison and the established
"0% silent at init" fact. Added `bias_init` (default `"uniform"` = legacy behaviour, unchanged for every prior
config; `"zeros"` used here). Verified: with `bias_init="zeros"` the initial weights are **bit-identical** to the
bias-free baseline at the same seed, the bias is a trainable Parameter starting at exactly 0, an invalid value
raises, and in a 30-iteration run every bias moves off zero and stays inside the clamp
(`Trainer.enforce_bias_range_`).

### Pre-registered readout — a bias can fake participation

A unit with `b = 0.3` and no task input fires at a **constant** 0.3: peak rate 0.3, `q0.9(|fr|)` 0.3 — healthy
participation — while `std(fr) ≈ 0` and it carries no information. That is a DC offset, not participation, and
with `frm` on it is the *cheapest* way to satisfy the penalty, so the `frm` cells are expected to do exactly this.
Recorded before results: **report `std(fr)` over (time, conditions) separately from participation, and count a
unit as rescued only if its `std` rises into the active population's range.** A "0% silent" number based on peak
rate alone is uninterpretable in this sweep.

### The sweep (submitted — Della array `11610299`, commit `1492041`)

40 jobs = 2 equations {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds, N=1000, γ=0, 30000 iterations,
participation tracked every 10 iterations. Identical to the `CDDM_ptrack_g0` grid submitted earlier today, with
`model=rnn_relu_Dale_trainablebias` (`bias_range: [-1, 1]`, `bias_init: zeros`) as the **only** difference — so
that sweep is this one's control.

Config `configs/model/rnn_relu_Dale_trainablebias.yaml`; launcher
[`slurm/SilentReLU_ptrack_bias_gamma0_N1000_della.slurm`](../slurm/SilentReLU_ptrack_bias_gamma0_N1000_della.slurm);
descriptor [`docs/experiments/participation_trace_bias.md`](experiments/participation_trace_bias.md).

**Folder separation (the two sweeps must not be mixed):**

| Folder | Model |
|---|---|
| `CDDM_ptrack_g0` | bias-free (`bias_range: [0,0]`) — the control |
| `CDDM_ptrack_g0_trainablebias` | trainable bias (`bias_range: [-1,1]`, zero init) |

both under `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs/`, each with its own
`EXPERIMENT.md` written by array task 1.

### Prediction

Silence **persists** at a broadly similar level (~40–55% under `none`/`rws`), because the 2026-07-01 tracking
showed the silencing is created during training and only weakly predicted by init activity (`corr ≈ 0.3–0.5`) —
a learnable offset gives units a way up, but nothing pushes them to take it. `frm`/`both` stay at 0%, partly for
the trivial DC-offset reason above.

**Falsifier:** if the silent fraction under `none` drops substantially (below ~15%), the bias-free constraint was
doing the work, and every claim in this document narrows to bias-free networks.

> **Repro caveat (same failure mode as the 2026-06-29 note; fixed for the next sweep — see the
> worktree note in the 16:45 entry).** Della's single working copy was advanced
> `d29ed5a` → `1492041` at 16:20 EDT to submit this sweep, while array `11609846` (the bias-free sibling) still had
> 6 tasks pending — those tasks therefore imported `1492041`. The diff is a **no-op for them**: the only code change
> is the new `bias_init` branch, which a degenerate `bias_range=[0,0]` never reaches, and `rnn_relu_Dale.yaml` is
> untouched. Already-running tasks imported their source at start and are unaffected either way. The durable fix is
> one `git worktree` per sweep on the cluster rather than a single shared checkout.

