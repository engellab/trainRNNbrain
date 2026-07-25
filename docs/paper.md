# Paper structure — silent units in trained RNNs

Working document. The full experimental record with configs, job IDs and repro instructions is
[`project_trajectory.md`](project_trajectory.md); this file is the *argument*: what we claim, what
supports each claim, and what is still missing. Every claim below carries an **evidence status** so
that interpretation never silently becomes result.

**Working titles**

- *Trained recurrent networks concentrate computation onto a minority of units — and how to stop them*
- *Silent units in trained RNNs: not a ReLU pathology, and not fixed by architecture*

**Thesis (one paragraph).** Recurrent networks trained on a standard cognitive task (context-dependent
decision making, CDDM) drive roughly half of their units into a near-silent, low-activity mode. This
is not an initialisation artifact, not a dead-gradient artifact of the ReLU, not noise-induced, and not
a consequence of the Dale projection or of the saturating term in the dynamics — it survives every
architectural intervention we tried and it worsens with network size. It is removed by an explicit
activity constraint: a firing-rate-magnitude penalty (`frm`) drives the silent fraction to exactly zero
at no cost in task performance. But `frm` alone is satisfied cheaply — the marginally-rescued units
fire weak, brief transients that meet the constraint without obviously doing work. Adding a recurrent
weight-sparsity penalty (`rws`) on top consolidates those units into stronger, more sustained,
condition-structured responses, even though `rws` alone cannot prevent silence at all.

---

## 1. The problem

### 1.1 What is observed

Networks: ReLU, Dale-constrained recurrent networks (excitatory/inhibitory sign-constrained `W_rec`,
excitatory-only readout), trained on CDDM to R² ≈ 0.85. After training, the per-unit peak firing rate
over a noise-free validation batch is **bimodal**: a tight active mode (~0.3–0.5) and a broad
low-activity mode centred near 1e-3 and spanning 1e-5–1e-1.

**Definitions used throughout** (all measured on the noise-free batch, on the trained weights):

| Quantity | Definition | Why |
|---|---|---|
| peak rate | `max` over (time, condition) of `\|fr_i\|` | unit *i*'s most active moment anywhere in the task |
| `dead<0.01` | fraction with peak < 0.01 | absolute floor; fine for ReLU, **unfair across activations** (softplus has a positive floor) |
| `silent<5%p95` | fraction with peak < 0.05·p95 of the net | within-network, scale-free; **the cross-activation comparison number** |
| participation | `std(fr_i) + q_0.9(\|fr_i\|)` over (t, c) | graded activity measure; basis of the histograms, HHI and per-unit tracking |
| 1/HHI | `1/Σ(p_i/Σp)²` | effective number of participating units (= N when perfectly even) |

**Report the distribution, not one thresholded number.** Only ~2.5% of units are *truly* dead
(peak ≈ 0); the "silent fraction" ranges 13% (<1e-4) → 44% (<1e-2) → 49% (<5e-2) in the same networks.
The principled cut is the inter-mode dip (~0.05), giving ~49%. The phenomenon is a **low-activity
mode**, not literal dead ReLUs — this framing matters and should be set in the first figure.

### 1.2 When it is observed

| Condition | Silent fraction (N=1000, h / s) | Note |
|---|---|---|
| N = 100 | 0% / 0.2% | no effect at small scale |
| N = 500 | 24% / 38% | |
| N = 1000 | 44–47% / 54–55% | **grows with network size** |
| σ_rec = 0 (h) | ~80% | noise-free training is a distinct, worse regime |
| at initialisation | **0%**, every condition, every `spectral_rad` ∈ {0.6, 1, 1.2, 1.6} | silence is created by training |

The size dependence is the key descriptive fact and the one that most constrains interpretation
(see §5.1): unpenalized nets solve CDDM using an effective **~60–150 units** (1/HHI) regardless of N.

### 1.3 What it is not

| Alternative explanation | Test | Result |
|---|---|---|
| A code bug | independently written from-scratch Euler integrator, no shared code | reproduces production per-unit peak rates **exactly** (max abs diff 0.0, silent-set Jaccard 1.000) |
| An initialisation artifact | score fresh untrained nets | **0% silent** at every `spectral_rad` |
| Literal dead ReLUs | peak-rate distribution | only ~2.5% truly dead; the rest is a graded low-activity mode |

---

## 2. Everything that failed to rescue participation

Each row is an intervention chosen because it *could* plausibly have removed the effect. All of them
left it essentially unchanged — which is what makes the phenomenon robust rather than incidental.

| # | Intervention | Why we tried it | Outcome | Evidence |
|---|---|---|---|---|
| 1 | **Equation type** `h` (pre-activation state) vs `s` (rate state) | the two standard conventions differ in where the nonlinearity sits; the effect could be specific to one | both show it; `s` marginally worse at large N | `CDDM_4a031e`, 120 nets |
| 2 | **Cubic saturation** `−γx³`, γ = 0.1 → **0** | γ is a built-in activity-magnitude limiter baked into the dynamics — it confounds the `frm` comparison and could itself cause silencing | no change (h/none 47% → 44%) | `CDDM_4a031e_g0` |
| 3 | **Dale boundary** sticky → **reflective** | the sticky projection hard-clamps sign-violating weights to ±1e-12, so weights *stick* at zero — an emergent sparsifier independent of any penalty | no rescue; if anything **more** silent (h/rws 53% → 58%). Sticky pins 19–32% of weights at the eps floor, reflective 0% — yet unit-level silence is identical | `CDDM_2bc3c1_g0_reflective` |
| 4 | **Activation**: ReLU → **softplus(β=25)**, **leaky-ReLU** | the decisive test of the dead-gradient story: both have nonzero gradient everywhere, so a unit can always be pulled back | persists, 40–64% on the scale-free criterion. **Softplus-h is the sharpest case: `dead<0.01` = exactly 0%, yet 41–64% relatively silent** — the population doesn't vanish, it reorganises from exact zeros into a soft low-activity continuum at the same ~half-of-N mass | `CDDM_fb2792_g0_softplus25`, `..._leakyrelu` |
| 5 | **Recurrent noise** σ_rec ∈ {0, 0.01, 0.05, 0.1} | noise could be silencing units, or masking them | never increases silence; **σ=0 is the worst case** (~80% for h), any noise drops it to ~46%. For `s`, flat ~55–60% | `CDDM_fb2792_g0_noise` |
| 6 | **Trainable bias**, `bias_range=[-1,1]`, zero init | with no bias, a unit whose input is always negative has nothing to lift it — the most obvious architectural constraint | **RUNNING** (Della `11610299`). Measured: silent units sit at `x` ≈ −0.19 (min −0.296), so `b = +0.30` would lift all of them; ±1 is a deliberately generous rail | `CDDM_ptrack_g0_trainablebias` |
| 7 | **Sparsity penalty `rws` alone** (λ=0.05) | an activity-shaping penalty; the natural counterpart to `frm` | **does not rescue — slightly worse** than baseline (h/N=1000: 54% vs 47%). A recurrent-weight sparsifier concentrates participation further | `CDDM_4a031e` |

**Summary sentence for the paper:** *no architectural change we tested — dynamics convention, saturation
term, Dale boundary implementation, activation function, noise level, or a learnable bias — removes the
low-activity population. It is a property of the trained solution, not of any one modelling choice.*

---

## 3. What actually works: explicit activity constraints

### 3.1 `frm` removes silence, for free

The firing-rate-magnitude penalty (soft-max-over-time rate driven toward a cap, `cap_fr = 0.3` scaled
as 1/log N; over- and under-cap terms) at λ = 0.2:

- **Silent units → exactly 0** in every cell: both equation types, all N, with or without `rws`, and in
  **every activation** (ReLU, softplus, leaky-ReLU).
- **1/HHI reaches the even ceiling** (~700–900 of 1000 effective units, vs ~60–150 unpenalized).
- **R² unchanged** (~0.83–0.87). Concentrating computation is *not* required for performance.

> **Framing risk — name it before a reviewer does.** "A penalty on sub-cap firing eliminates sub-cap
> units" is close to tautological. The non-trivial content is: (i) it costs **nothing** in task
> performance; (ii) unpenalized networks concentrate onto ~10% of units *when nothing pushes back*;
> (iii) `rws`, an equally plausible activity-shaping penalty, **fails**; (iv) the rescue is a
> network-level phenomenon that survives adversarial silencing constructions (§4). Lead with those.

### 3.2 …but `frm` is gamed at the margin, and `rws` fixes that

The *bulk* of `frm`-rescued units form a healthy participation mode (~0.4–0.6). The problem is the tail:

| | `fr only` | `both` (`frm` + `rws`) |
|---|---|---|
| worst unit's participation (N=1000) | 0.05–0.06 | **0.09–0.12** |
| 5th-pct participation of revived units | ≈0.06 | **≈0.13–0.14** |
| mean 1/HHI (h) | 746 | **887** (+19%) |
| mean 1/HHI (s) | 810 | **912** (+13%) |

Under `frm` alone the least-participating units carry **weak, brief, localized transients** — activity
that satisfies the penalty without an obviously sustained role. Under `frm`+`rws` the same marginal
units show stronger, more sustained, time- and condition-structured responses. The floor-lift is
essentially identical in `h` and `s`.

> **Evidence status — this is the paper's weakest link.** The transient-vs-structured contrast rests on
> (a) the participation statistics above (solid, quantitative) and (b) **visual inspection of the
> single worst unit's activity heatmap** (interpretation, not measurement). The claim "`rws` shapes
> neurons into meaningful computational roles" is **not yet supported by any test of computational
> role.** Required before this can be written as a result (§6): decode task variables (context,
> motion/colour evidence, choice) from the marginally-revived population and compare `fr only` vs
> `both`; quantify selectivity and cluster structure; ablate the revived units and measure the R² cost.
> All three use existing trained networks — no new training.

---

## 4. How far does the rescue reach? (mechanism)

A per-unit gradient argument says rescue should be impossible: a truly dead ReLU has `ReLU′ = 0`
everywhere, so no gradient — including the `frm` under-penalty — reaches its incoming weights. The
experiments show the argument is locally correct but globally wrong.

| Construction | Question | Result |
|---|---|---|
| Natural init, per-unit tracking init→trained | is `frm` preventing or resurrecting? | **~0% of units are silent at init**, so on the natural init the mechanism is **prevention**. Training *splits* a homogeneous population (`corr(log init, log trained) ≈ 0.3–0.5` — init activity barely predicts fate) |
| **Forced silent at init**: targeted over-inhibition of a random 25% (`inhibitory_boost = 2`) | can `frm` revive units that start dead? | **Yes — 100% of them**, both equation types. Also: under `none` those units end statistically **identical** to never-silenced units — init silencing is not durable |
| **Master inhibitor**: one context-locked unit holding a fraction of the net silent | a clamp gradients supposedly cannot reach | `frm` rescues **100% at every fraction** — indirectly, by **suppressing the inhibitor itself** (it is active and far over cap, so it *is* penalizable). Prediction falsified |
| **Frozen master inhibitor** (weights held at init by grad hook + forward-pre-hook) | remove that escape route entirely | `frm` **still rescues at frac < 1.0** (100% active, participation 0.02–0.10, R² ≈ 0.85) by recruiting the *non-clamped* units to build compensating excitation |
| **frac = 1.0** (entire network clamped) | the limiting case | **Fails** — no scaffold remains, task unsolvable under either penalty (R² = −0.38 / 0.08). The only regime where silence survives `frm` |

**Conclusion.** Silence is a **network-level** property, not a per-unit one. `frm` makes "all units
active" the trained solution and reaches it from any initial condition, dismantling even a structured,
gradient-proof silencing mechanism — unless the network's capacity to compute is removed entirely,
which also destroys the task.

**Methods note worth its own paragraph.** Getting a clean answer required diagnosing a NaN divergence
that was *not* a gradient explosion: `frm` builds a self-exciting recurrent loop (gain > 1) to overcome
the clamp, and explicit-Euler integration of that loop overflows within a 300-step trial. Gradient
norms were ~1 against a clip of 50 — clipping was irrelevant, and so was clamp magnitude. Fix:
`dt = 0.5` **and** γ = 0.1 together (γ = 0 diverged 100% of the time at dt = 0.5; dt = 1 with γ = 0.1
diverged ~50%). Divergence rate 50% → 10%.

---

## 5. Open framing risks

### 5.1 "That's just spare capacity" — the objection that decides the paper

A 1000-unit network on a low-dimensional task trivially has units to spare, and our own data supports
the deflationary reading: 0% silent at N=100, ~50% at N=1000, and unpenalized nets solve the task with
an effective ~60–150 units at full R². A reviewer will say the network recruits what it needs and we
then forced the rest to fire for no benefit.

Two ways to answer, both currently missing:

1. **Scale the task, not the network.** Harder/higher-dimensional CDDM variants (more contexts, more
   stimulus dimensions, compositional structure). If the silent fraction shrinks as the task could use
   more units → spare capacity, and the paper becomes "RNNs recruit minimally". If it persists → genuine
   pathology. Either is publishable; not knowing is not.
2. **Show the distributed solution buys something** — lesion robustness, noise robustness,
   generalisation to held-out coherences. Needs **no new training**. If `frm` nets degrade more
   gracefully under lesions, "no task cost" upgrades to "strictly better" and the objection dies.

### 5.2 Generality

Everything is CDDM. DMTS / GoNoGo / MemoryAngle configs exist in the repo; one task is one anecdote.

### 5.3 Reporting hygiene

Most tables report means over 5 nets with no spread — per-net points or CIs before submission. Two
softplus nets are missing from a 40-net sweep (38/40); confirm they are not concentrated in one cell.
A subset of nets recorded a commit hash that differs from their sweep's (a runtime `git rev-parse` in
the folder tag) — state it in Methods rather than leaving it as a repro note.

---

## 6. Figure plan

| Fig | Content | Status |
|---|---|---|
| 1 | The phenomenon: peak-rate / participation distributions, bimodality, size scaling (N = 100/500/1000, h and s) | data exists |
| 2 | It is not an artifact: init vs trained per-unit scatter; independent-integrator agreement; sticky vs reflective `\|W_rec\|` distribution | data exists |
| 3 | Architecture doesn't matter: silent fraction across activations (scale-free criterion), noise levels, boundary, γ, **bias** | bias sweep running |
| 4 | Penalties: R² vs 1/HHI scatter (two clean clusters, equal R²); silent counts per condition | data exists |
| 5 | The margin: least-participating-unit activity heatmaps, `fr only` vs `both`; floor-lift statistics | data exists; **needs the quantitative role analysis of §3.2** |
| 6 | Mechanism: forced-silent and (frozen) master-inhibitor rescue across fractions, with the frac = 1.0 failure | data exists |
| 7 | Time course: participation trajectories during training — when the split happens, and whether it reverses | **running** (Della `11609846`) |

Preliminary from the 3000-iteration test run: the bifurcation completes by iteration **~400–600** of
30000 and is then frozen; it is preceded by a **global collapse** (every unit quiet within ~20
iterations) from which only the eventual-active subset recovers. If that holds in the full sweep it is
a Figure 7 in its own right, and it was not predicted by anything in the endpoint data.

---

## 7. Priority list before writing

1. **Quantify the `rws` role claim** (§3.2) — decoding + selectivity + ablation on existing nets. Without
   it, the second half of the thesis is an interpretation.
2. **Task-scaling experiment** (§5.1) — needs design; highest value.
3. **Lesion/robustness analysis** (§5.1) — no new training; converts "no cost" into "a benefit".
4. Second task (§5.2).
5. Finish the two running sweeps (bias control, participation trajectories).
6. Reporting hygiene (§5.3).
