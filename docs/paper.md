# Paper plan — silent units in trained RNNs

Working document: the *argument*, what supports each claim, and what is missing. The full
experimental record (configs, job IDs, repro instructions) is [`project_trajectory.md`](project_trajectory.md).

**Every claim carries an evidence status** — ✅ measured, 🟡 preliminary, ⬜ planned — so that
interpretation never quietly becomes result.

**Architecture policy — main text vs supplement.** Every main-text result is stated for **standard
unconstrained RNNs**: no Dale's law, no I/O sign constraints, self-connections allowed, trainable
bias, no cubic term, plain multi-objective gradients. **Dale-constrained and I/O-positive networks
are supplementary throughout** (§S1) — a robustness check on the main claims, never the basis of
one. A result stated only for constrained networks reads as niche and is easy to dismiss.

Where a main-text number is currently derived from constrained networks because that is the only
data with the relevant manipulation, it is marked ⚠️ **Dale-derived, standard-RNN version pending**
rather than presented as settled.

**Working titles**

- *Most units in trained RNNs are silent, and the standard regularizer makes it worse*
- *Silent units distort what RNN models tell us about neural populations*

---

## 1. The problem

RNNs are used as models of cortical circuits, and the comparison to data is made at the level of the
**population**: dimensionality, selectivity distributions, functional cell classes, correlation
structure. That comparison assumes the model's population is the thing being modelled.

**It usually isn't.** In standard ReLU RNNs trained on a context-dependent decision task (CDDM), a
large fraction of units never fire:

| N = 1000, unconstrained + trainable bias | silent (peak rate < 0.01) | ✅ |
|---|---|---|
| h equation, no penalty | **45.6%** | 5 nets |
| s equation, no penalty | **41.9%** | 5 nets |
| scale-free criterion (peak < 5% of p95) | 48.0% / 46.2% | 5 nets |

These are not merely quiet units. **31–42% of units are at exactly 0.0** — literally switched off,
never crossing threshold at any timestep in any condition ✅. (In constrained networks this is
hidden: non-negative input weights guarantee every unit a small positive push, so silent units sit
at 1e-9…1e-2 instead of zero. Same population, concealed depth — see §S1.)

**It gets worse with size** ⚠️ *Dale-derived, standard-RNN version running* (`CDDM_std_g0_Nsweep`,
N ∈ {100, 250, 500, 1000}):

| | N=100 | N=500 | N=1000 |
|---|---|---|---|
| silent | 0.0% | 23.7% | 47.1% |
| **active units** | 100 | 382 | **529** |

**The active count grows sublinearly — and it may be heading for a ceiling** ⚠️. Two models fit
these three points about equally well and diverge sharply beyond them:

| Model | Fit | N=2000 | N=5000 | N=10000 |
|---|---|---|---|---|
| power law | `active = 3.57·N^0.723` | 873 | 1695 | **2798** |
| saturating | `active = 1011·N/(N+911)` | 695 | 855 | **926** |

The local exponent is already falling — 0.833 between N=100 and 500, then 0.470 between 500 and
1000 — which leans toward a **task-determined ceiling of order 1000 active units**.

**This is the paper's answer to the sharpest objection it faces:** *"why make every unit compute —
just train a bigger network and prune the silent ones?"* If active units saturate, pruning **cannot**
deliver a large active population at any size, because the ceiling is set by the task rather than the
budget, and activity regularization is the only route there. If they keep growing, pruning works but
costs a rapidly worsening ratio. ⬜ Decided by the large-N runs (N ∈ {2000, 5000, 10000}), currently
being sized for wall time and memory (see `project_trajectory.md`, 2026-07-27 15:30).

**Most of the silence is created by training** ✅ — but, corrected 2026-07-28, **not all of it**. In
standard unconstrained networks **4.4% of units are already hard-silent at initialization and 21% sit
below the 0.01 line** before a single training step: with signed input weights a unit can start
net-negative and never fire. Training takes it from 4.4% to 41.5%. The older "0% silent at init"
result is **Dale-specific** — I/O positivity guarantees every unit a positive push, so nothing can be
exactly zero there. It also isn't a bug: an independently written Euler integrator reproduces the
per-unit peak rates exactly (max abs diff 0.0) ✅.

---

## 2. Standard activity regularizers do not fix it — and at strength make it worse

Activity regularization in RNN training is routine — metabolic-cost terms are standard practice in
this literature. But the usual penalties act on the wrong side of the distribution: they penalize
*high* rates, pushing the whole population down. **Status (2026-07-28): both halves are now
measured** — the λ_met sweep is complete (36 networks, 4 decades × 3 sizes × 3 seeds).

| Penalty | Effect on silence |
|---|---|
| **`rws`** — recurrent-weight sparsity | **Does not rescue**: 42.7% → 39.9% ✅ — a marginal change against a ~43% baseline, nowhere near the 0% that `frm` reaches. (In constrained networks it is actively *worse*; see §S1.) |
| **`met`** — metabolic cost, `mean(fr²)`, the field-standard form | ✅ **Measured: never rescues, and at strength makes it worse.** Across λ ∈ {0.01, 0.1, 1, 10} and N ∈ {100, 500, 1000} the silent fraction never falls below baseline. At λ=10 it rises sharply where there is headroom — N=100: **12% → 59%** scale-free silent; N=500: 41% → 69% — while R² stays 0.81–0.87, so this is not a penalty destroying the task. Flat at N=1000, where the baseline is already ~42% and little room remains. |

⬜ **This experiment is the paper's hook and must be run first.** Sweep λ_met across ~3 decades at
N=100 and N=1000, reporting silent fraction *as a function of λ*, so the result cannot be dismissed
as a bad choice of λ. If standard practice actively worsens the problem, that is the finding — far
stronger than "here is another penalty".

`rws` also silences on a **completely different timescale**: median first crossing at iteration 3287
versus 16 for the unpenalized net ✅. It does not participate in the early collapse; it kills units
slowly, late in training.

---

## 3. What we tried, and what worked

Every intervention below was chosen because it *could* plausibly have removed the effect.

| # | Intervention | Result |
|---|---|---|
| 1 | Equation type `h` vs `s` | both affected ✅ |
| 2 | Cubic saturation term γ: 0.1 → 0 | no change ✅ |
| 3 | Dale boundary: sticky → reflective | no change ✅ |
| 4 | **Removing Dale's law entirely** | no change in the silent fraction ✅ |
| 5 | **Removing I/O positivity** | no change in the fraction; converts soft floors into hard zeros ✅ (§S1) |
| 6 | **Trainable bias** (`[-1,1]`, zero init) | no rescue ✅ (55.3% → 54.9%) |
| 7 | **Self-connections** allowed | no rescue ✅ — and the network trains the diagonal into self-*inhibition*: corr(self-weight, log participation) = **−0.51**, active units at −0.060 vs silent at −0.007. The cheapest escape from silence is one it declines to take. |
| 8 | Activation: softplus(β=25), leaky-ReLU | persists, 40–64% (Dale) ✅; ⬜ standard-RNN rerun |
| 9 | Recurrent noise σ_rec ∈ {0, .01, .05, .1} | never helps; σ=0 is the *worst* regime (~80%) ✅; ⬜ rerun |
| 10 | `rws` sparsity penalty | fails (§2) ✅ |
| 11 | **Connectivity scale** (spectral radius, density) | ⬜ planned — is silence simply weak recurrent drive? |
| 12 | **`frm` firing-rate-magnitude penalty** | ✅ **works** |

**`frm` is the one that works, and it works completely** ✅:

- Silent fraction → **exactly 0**, in every cell, every architecture (Dale, unconstrained, ±bias),
  and every activation tested.
- No tail whatsoever: the *minimum* participation of any unit in any `frm` net is 4.8e-2 — about a
  sixth of the median, with nothing approaching zero.
- **No task cost**: R² ≈ 0.83–0.87, indistinguishable from unpenalized networks.
- Effective participating units (1/HHI) rise from ~60–150 to ~700–900 of 1000.

> **Name the tautology before a referee does.** "A penalty on sub-cap firing removes sub-cap units"
> is nearly circular. The non-trivial content is: it costs *nothing* in performance; unpenalized nets
> concentrate onto ~10% of units when nothing pushes back; the standard regularizer fails or makes it
> worse; and — §5 — the resulting populations differ in ways that change scientific conclusions.

---

## 4. Does the rescue prevent, or resurrect?

A per-unit gradient argument says rescue should be impossible: a dead ReLU has zero derivative
everywhere, so no gradient — including from the penalty — reaches its incoming weights.

Answered directly from the participation traces (per-unit participation logged every 10 iterations
through all 30000), h equation, per 1000 units ✅:

| | ever dips below 0.01 | ends below | **silent ≥500 iters, then recovers** |
|---|---|---|---|
| standard, none | 83.4% | 54.9% | 95.8 |
| standard, **frm** | 48.9% | **0%** | **0.6** |
| Dale, none | 96.0% | 53.6% | 85.0 |
| Dale, **frm** | 94.4% | **0%** | **369.4** |

**Both, and which one depends on the architecture.** In standard RNNs `frm` **prevents** — units dip
briefly during the early collapse and are caught within a few hundred iterations; essentially none
endure a long silent episode. In Dale networks it genuinely **resurrects** — 369 units per network
were silent for ≥500 consecutive iterations and returned.

Two by-products of the same analysis ✅: silence is **not strictly irreversible** even without
penalties (~96 units per network recover spontaneously), and the split *begins* within the first few
hundred iterations, preceded by a **global collapse** in which the entire population goes quiet
within ~20 iterations and only the eventual-active subset climbs back out. Note it begins early but
does **not** finish early — see §6.1: the silent fraction is still growing at iteration 30000.

⬜ Optional sharpening, 20 jobs: force a random 25% of units silent at init (bias = −1, frozen) and
follow those specific units. Only needed if a referee insists on "can it revive a unit dead from the
very start". The Dale resurrection number already carries most of that weight.

*(The earlier master-inhibitor / frozen-clamp experiments asked this same question through
hand-built silencing constructions. They are superseded by the trace analysis, which answers it on
the natural initialization in the standard architecture. Keep at most as supplementary.)*

---

## 5. Why it matters: population-level signatures

The argument that turns this from tidiness into a validity problem. **Recorded cortical populations
do not contain 45% of neurons that never fire during a task** — so an RNN that does is not the
population being claimed as a model, and every population-level comparison inherits the distortion.

✅ **Measured 2026-08-17** (N=1000, 5 nets per cell, all conditions at R² = 0.84–0.87 — see
`project_trajectory.md`). Networks that perform identically differ by **3.5× in effective
dimensionality** (PR 2.22 → 7.74), **1.7× in choice selectivity measured over active units** (33.5% → 58.3%; the 3× figure from
all-unit fractions is inflated by dilution, since silent units are non-selective by construction —
and context selectivity actually *falls* among active units, 42.3% → 31.1%),
**2× in total metabolic cost** (31.1 → 14.6, the concentrated solution being the *more* expensive
one), and **100× in energy concentration** (HHI 0.123 → 0.0012, i.e. cost carried by an effective 8
units versus ~850). Separately, within unpenalized networks the same selectivity statistic reads
24.3% over all units but 42.3% over active units only — and a recording experiment sees only the
latter.

| Statistic | Why it is distorted |
|---|---|
| **Dimensionality / participation ratio** | computed over an effectively ~500-unit circuit while reported as N=1000 |
| **Selectivity distributions** (context, motion, colour, choice) | silent units are untuned by construction, diluting every proportion by an arbitrary factor |
| **Functional clustering / cell classes** | a large "unresponsive" cluster with no counterpart in a recorded population |
| **Metabolic cost of the computation** | unpenalized nets concentrate high rates in few units; `frm` nets spread moderate rates over all. Total energy may match while the distribution differs entirely |
| **Correlation structure / eigenspectrum** | inherits the above |

**Deliverable:** run these on `frm` vs `none` networks and show the numbers differ *materially* —
ideally by as much as the model-vs-data differences people publish. The claim then becomes: *your
conclusion about the circuit depends on whether you regularized activity, and nobody reports it.*

⬜ **Does the rescued network function better?**

- **Noise robustness** — raise σ_rec at evaluation, compare R² decay.
- **Lesion robustness of the top contributors** — ablate the highest-participation units and measure
  degradation. *Design note:* naive random ablation is misleading, because in a half-dead network
  half the lesions hit units that do nothing, making it look artificially robust. Ablate by
  participation rank, or express degradation per *active* unit removed.
- **Generalization** — held-out coherences or a shifted stimulus distribution.

---

## 6. Open questions

The four that decide what this paper can claim, in dependency order — 6.1 gates 6.2, and 6.4 gates
the framing of the whole thing.

### 6.1 Does training converge, or do the parameters (and the silent units) keep drifting?

**Status: running** (Della `11706899`, 12 jobs, 100000 iterations).

Every number in this project was measured at a fixed 30000 iterations, on the assumption that
training had settled. It has not. The **loss** is flat over the final 10% at every size (−0.4 to
−0.6%), but the **silent fraction is still climbing** — +3.7 pp over the last 5000 iterations at
N=1000, and 7.4% → 41.5% hard-silent between iterations 5000 and 30000. The network solves the task
early and then goes on switching units off: drift along a flat loss manifold.

Being measured: **new silent units per 1000 iterations** (from the participation trace, which needs
no new logging) and **relative parameter drift** `‖W(t)−W(t−Δ)‖_F/‖W(t)‖_F` (newly logged), at
N ∈ {1000, 2000} × `weight_decay` ∈ {1e-6, 0} × 3 seeds.

**Why it can change the paper rather than just the methods:** if the curves are still moving at
100000, then every silent-fraction number carries the caveat *"at 30000 iterations"* rather than
*"at convergence"*, and the size sweep needs re-running to a convergence **criterion**. And if
`weight_decay=0` removes the late drift, the late-phase silencing is a **regularization artifact**
rather than something the task demands — which would not erase the result (the silence is there at
30000 either way, and `frm` still removes it) but would change what the phenomenon *is*.

### 6.2 Does the active-unit count saturate, or grow without bound?

**Status: blocked on 6.1.** This is the answer to the sharpest objection the paper faces — *"why make
every unit compute, just train a bigger network and prune the silent ones?"*

Measured (standard RNNs, h, no penalty, 30000 iterations): active units 97 → 227 → 390 → 580 → **862**
across N = 100 → 2000, with the local exponent falling 0.93 → 0.78 → 0.57 → **0.57** — it declined,
then stopped declining. Two fits to N ≤ 1000 diverge 3× by N=10000 (power law 2069 vs saturating
702), and the N=2000 point (862) sits **above both**, so the current evidence points to **growth**,
against my pre-registered prediction of saturation.

**But that evidence is compromised by 6.1.** Residual drift grows with N, so larger networks are
further from their asymptote at any fixed iteration count. That undercounts silence at large N,
overcounts active units at large N, and biases the curve toward exactly the "growth" answer we got.
The N=5000/10000 runs must wait until the horizon is known — running them at 30000 iterations would
buy a differently-wrong answer for ~185 GPU-hours.

**If saturation:** pruning cannot deliver a large active population at any size, and activity
regularization is the only route — `frm` at N=1000 already gives 1000 active units.
**If growth:** pruning works, and the paper's argument must rest entirely on the population-level
differences of §5 rather than on reachability.

### 6.3 Is the rescue preventive or genuinely restorative?

**Status: answered, architecture-dependent** ✅ (§4). In standard RNNs `frm` **prevents** — 0.6 units
per network recover from a long silent episode. In Dale networks it **resurrects** — 369 per network
were silent ≥500 consecutive iterations and returned. Both from the participation traces, on the
natural initialization, with no contrived construction.

What remains open is the harder case: can it revive a unit that is dead **from initialization and
stays dead**? On the natural init there are few such units to test — though 6.1's finding that 4.4%
*are* hard-silent at init in standard networks means the material now exists. ⬜ 20 jobs would settle
it (force a random 25% silent via a frozen bias of −1, then follow those units in the trace).
Optional; the Dale resurrection number already carries most of the weight.

### 6.4 Is the silence just spare capacity — is the task too easy?

**Status: open, needs design.** The deflationary reading, and the one a referee will default to: CDDM
is low-dimensional, unpenalized networks solve it with ~60–150 effective units, and a 1000-unit
network trivially has units to spare. Nothing measured so far excludes it.

The test is to scale the **task**, not the network — more contexts, more stimulus dimensions,
compositional variants — and ask whether the active fraction rises. **Either answer is publishable,
but they are different papers.** If the active count rises with task complexity, the honest
conclusion is partly a recommendation: *don't train 1000-unit RNNs on simple tasks and then analyse
the population, because most of it is unused* — which composes with 6.2, since if active units also
grow only sublinearly you cannot cheaply buy a large active population by enlarging the network
either. If the active fraction is flat regardless of task complexity, it is a genuine pathology and
the paper is stronger.

### 6.5 Smaller open items

- **Connectivity scale** ⬜ — is silence simply insufficient recurrent drive? Spectral radius was only
  ever checked *at initialization* (0% silent at every radius), never in trained networks.
- **Generality** ⬜ — everything is CDDM. DMTS / GoNoGo / MemoryAngle configs exist in the repo.
- **Activation and noise reruns** ⬜ — established in Dale networks, not yet repeated in standard ones.

## 7. Methods points that must be stated

- Training noise (σ_rec = σ_inp = 0.05, σ_out = 0.03) — retained deliberately; standard in
  neuroscience RNNs, absent in vanilla ML ones.
- `lr` is rescaled as `lr × (100/N)^0.333`, so **N and learning rate co-vary** in the size sweep.
- Penalized runs use plain multi-objective descent (the earlier task-safe gradient projection is off).
- **A practical result worth its own paragraph** ✅: strong activity penalties can make training
  diverge, and *not* through gradient explosion. `frm` builds a self-exciting recurrent loop with
  gain > 1 to overcome inhibition; explicit Euler then overflows within the trial while gradient
  norms remain ~1, so clipping is useless. The fix is a smaller integration step plus a bounding
  nonlinearity (dt 1 → 0.5 with γ = 0.1 cut the divergence rate from ~50% to ~10%). This is exactly
  the kind of finding a tools-and-methods paper should carry.
- Report distributions, not single thresholds: the silent fraction is threshold-dependent
  (13% at <1e-4, 44% at <1e-2, 49% at <5e-2 in the same networks).
- Per-net spread or CIs, not just means over 5 nets.

---

## 8. Venue and priorities

**Target: PLOS Computational Biology (Methods).** It requires correct, complete, useful work rather
than novelty impact — which fits. Fallbacks: eNeuro (Research Methods and New Tools), NBDT (welcomes
careful, unglamorous, control-heavy work). bioRxiv preprint immediately. **Not** eLife until §5
exists — under its model a thin paper earns a permanent public "useful, incomplete" assessment.
JOSS for the package as a companion citation.

**Priority order:**

1. ⬜ **Metabolic-cost sweep** (§2) — cheap, and likely the paper's hook.
2. ⬜ **Population-distortion analyses** (§5) — no new training; converts tidiness into validity.
3. ⬜ **Noise + top-unit lesion robustness** (§5) — no new training.
4. 🟡 **Standard-RNN reference + N sweep** — running; fixes the active-unit exponent.
5. ⬜ **Large-N runs** (N ∈ {2000, 5000, 10000}) — settles saturation vs growth, i.e. whether the
   "just prune" objection has an answer. Being sized now; needs `.npz` parameter saving and
   truncated training validated against an N=1000 control.
6. ⬜ **Task scaling** (§6.1) — highest scientific value, needs design.
7. ⬜ Second task; activation/noise/connectivity-scale reruns.

---

## S1. Supplementary: Dale-constrained and I/O-positive networks

**Not part of the main argument.** Every claim in §§1–6 is established in standard unconstrained
RNNs. This section exists for two narrow purposes: to show the phenomenon is *not* an artifact of
the constrained architecture much of this literature uses, and to record the specific places where
the constrained case behaves differently. Readers who do not work with Dale-constrained models can
skip it.

1. **Silence is overwhelmingly excitatory** ✅: 53–55% of excitatory units vs 3.5–5.0% of inhibitory
   units silent (h, none). This **falsifies** the natural hypothesis that the excitatory-only readout
   starves inhibitory units of gradient — the opposite happens. Likely load-bearing redundancy: 200
   inhibitory units carry the whole network's inhibition at 4× weight and are individually
   indispensable, while 800 excitatory units are mutually redundant.
2. **I/O positivity conceals how dead the units are** ✅: with `W_inp ≥ 0` and non-negative inputs,
   every unit gets a positive push at every timestep, so no unit is *exactly* zero (2.2% hard-dead vs
   31.4% unconstrained) — while the total silent fraction is unchanged (44.2% vs 42.7%). Every one of
   the 285 hard-zero units in an unconstrained net has Σ`W_inp` < 0 (100%, no exceptions).
3. **`rws` reverses sign** ✅: worse than baseline under Dale, mildly better unconstrained.
4. **`frm` resurrects rather than prevents** ✅ (§4) — in standard RNNs it prevents.
5. **The `s` equation is more sensitive to constraints** ✅: 55.1% → 32.6% silent when Dale and I/O
   positivity are removed, while `h` is unchanged.

---
