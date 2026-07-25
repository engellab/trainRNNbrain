# Experiment: trainable-bias control (does the silent mode survive a learnable offset?)

*This file is copied into the sweep's data folder as `EXPERIMENT.md` by the launcher.*

**Sibling sweep:** `CDDM_ptrack_g0` is the identical grid with the standard **bias-free** model
(`bias_range: [0, 0]`). This folder, `CDDM_ptrack_g0_trainablebias`, is the same thing with a
**trainable bias**. The two are directly comparable — same seeds procedure, same code commit, same
everything else — and must not be mixed. Tracking file format and metric definitions:
[participation_trace.md](participation_trace.md).

## Question

Every silent-unit result in this project was obtained with `bias_range=[0,0]`, i.e. **no bias at
all**. A ReLU unit whose total input is negative at every timestep then has nothing to lift it into
the active range, so the natural objection is that the ~45–55% silent population is an artifact of
an unusually constrained model rather than a property of trained RNNs. A trainable bias is arguably
the more standard architecture. This sweep asks:

**Does the low-activity population survive when every unit can learn its own offset?**

- If silence collapses → the claim narrows honestly to *bias-free* ReLU-Dale networks.
- If it persists → the result is much stronger, and the most obvious reviewer objection is closed.

## Why the bias range is ±1, and why the bias starts at zero

Measured on a trained `h`/`none` N=1000 net from `CDDM_4a031e_g0` (43.3% silent). The `h` dynamics
are `dx/dt = −x + W_rec·r + W_inp·I + b`, so at steady state `x* = drive + b` — the bias is
commensurate with the state and directly comparable to how negative a silent unit sits:

| | time-averaged state `x` |
|---|---|
| silent units (n=433) | median **−0.187**, p5–p95 −0.238 … −0.156, most negative **−0.296** |
| active units (n=567) | median −0.154, p95 +0.087 |

So **b = +0.30 would lift 100% of the silent units to threshold** (+0.24 lifts 95%), against an
active-unit median peak rate of 0.23 and `cap_fr = 0.3`. The range is set to **±1**: a rail rather
than a prior. The optimum sits at ~0.2–0.3, well inside it, so training is unconstrained in
practice and "the bias could not reach far enough" is not an available objection.

`bias_init: "zeros"` (new `RNN_torch` option; the legacy default `"uniform"` draws the initial bias
uniformly over `bias_range`). With a ±1 range, a uniform init would start the network with offsets
5× the drive scale — a different initial condition, breaking both the one-variable comparison and
the established "0% silent at init" fact. With zeros, the initial weights are **bit-identical** to
the bias-free baseline at the same seed (verified), and the only change is that each unit now has
an offset it can learn.

## Pre-registered readout — a bias can fake participation

A unit with `b = 0.3` and no task input fires at a **constant** 0.3: its peak rate is 0.3 and
`q0.9(|fr|)` is 0.3, so participation looks healthy while `std(fr) ≈ 0` and the unit carries no
information. This is a DC offset, not participation, and with `frm` on it is the *cheapest* way to
satisfy the penalty — expect the `frm` cells to do exactly this.

Therefore, recorded **before** the results:

- Report **`std(fr)` over (time, conditions) separately from participation**, for every condition.
- A unit counts as **rescued only if its `std` rises** into the active population's range — not
  merely its peak rate or participation.
- A "0% silent" result reported on peak rate alone is uninterpretable in this sweep.

## Prediction

Silence **persists** at a broadly similar level (~40–55% under `none`/`rws` at N=1000), because the
2026-07-01 result showed the silencing is created during training and only weakly predicted by init
activity — a learnable offset gives units a way up, but nothing pushes them to take it. The
`frm`/`both` cells stay at 0% silent, partly for the trivial DC-offset reason above.

**Falsifier:** if the silent fraction under `none` drops substantially (say below ~15%), the
bias-free constraint was doing the work and every claim in the project narrows to bias-free
networks.

## Grid — 40 jobs

`2 equation types {h, s} × 4 penalties {none, rws, frm, both} × 5 seeds`, N=1000, γ=0.

Identical to the `CDDM_ptrack_g0` grid, with `model=rnn_relu_Dale_trainablebias`
(`bias_range: [-1, 1]`, `bias_init: zeros`) as the only difference. Participation tracking is on
(`track_every=10` → 3000 snapshots/net), so the trajectory question is answered for this
architecture too, at no extra cost.

Launcher: `slurm/SilentReLU_ptrack_bias_gamma0_N1000_della.slurm`.
