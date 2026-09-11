# Research directions — what the paper still needs, and what it does not

Companion to [`paper.md`](paper.md) (the argument) and [`project_trajectory.md`](project_trajectory.md)
(the record). This file exists to keep the paper from becoming a dump of everything that was
measured. **Rule: a result goes into the main text only if removing it would break a link in the
storyline of `paper.md`.** Everything else is supplementary or out of scope, and is listed as such
so it is not re-proposed.

Each item states: the question, which claim in the paper it feeds, the design, the cost, and — the
part that matters — **what outcome would change the paper**. An experiment with no outcome that
would change the paper is not run.

Constraint as of 2026-09-10: **no new cluster training until the analysis-only items are done.**
Most of what the main line needs is computable from the data in `dead_ReLU_data/`.

---

## A. Main line, analysis only (local data)

| # | question | feeds | status |
|---|---|---|---|
| A1 | Four Hoyer axes (dimensionality, participation, temporal, selectivity) per penalty, both tasks | paper §4–5 characterisation | ✅ `characterize.py` |
| A2 | Joint vs best-single-variable R² per condition — is "mixed" a matter of degree? | paper §4 (frm is diffuse) | ✅ see trajectory 2026-09-10 |
| A3 | Assembly share and wiring modularity on CDDM | paper §5 (rws restores assemblies) — currently flip-flop only | ⬜ |
| A4 | Task-free purity (NMF) on CDDM | paper §5 | ⬜ |
| A5 | Training cost: iterations to 1.01× and 1.03× floor per penalty, both tasks | paper §3 "no task cost" — must include time | ⬜ flip-flop numbers exist in trajectory; CDDM losses on disk |
| A6 | Distortion table (dimensionality, selectivity fractions, σ_log) on flip-flop | paper §6 — currently CDDM only | ⬜ half exists (`flipflop_dimensionality.py`, `flipflop_arms.py`) |

| A7 | Why W_inp keeps drifting under the penalties: evaluate task, frm, rws and weight-decay gradients on the final weights — do they cancel while each is large? | paper §5 (nature of the `both` solution); trajectory 2026-09-11 09:02 | ⬜ one afternoon, local |

**A3 design.** CDDM has four natural states (context × choice). Two versions, as on the flip-flop:
hard assignment of each tuned unit to its dominant state and the fraction of its recurrent input
mass from same-state units, against a row-shuffle null; and the label-free tuning-overlap version.
Wiring and activity modularity at matched n, each against its own null, exactly as in
`flipflop_modularity` (2026-09-10 16:17 entry). *Outcome that changes the paper:* if `frm`'s wiring is
NOT the outlier on CDDM, the composition mechanism is flip-flop-specific and §5 is scoped down.

**A4 design.** `flipflop_mixedsel.py` with the CDDM loader; d swept 2..16; calibration with the
CDDM condition structure (pure = one context × choice cell). *Outcome:* same as A3.

**A5 design.** For each run, iteration at which the smoothed noise-free loss first stays within
1.01× / 1.03× of its final floor (the `flipflop_time_to_floor.py` machinery; Tobit-censored where the
run ends first). *Outcome that changes the paper:* if `both` needs > 2× the iterations of `none` at
matched N, "no task cost" becomes "no task cost, at a training-time cost of X", stated up front.

---

## B. Main line, needs training (cluster) — proposed, not launched

These are the experiments a referee will ask for. Ordered by how much of the paper they hold up.

### T1. Activation without a dead zone — leaky-ReLU and softplus, unconstrained
> **Status 2026-09-11 08:50 — ANSWERED, T1 closed.** Softplus and leaky-ReLU (June Dale sweeps, rates
> computed correctly) match ReLU on every axis. The sigmoid baseline (Spock `6147881`, 3-bit
> flip-flop, N = 500/1000 × 3, `sigmoid(7.5(x−0.3))`, 150k it.) silences 0.75–0.77 of units at
> N = 1000 vs ReLU's 0.72–0.75 at the same iteration, units parked at the lower asymptote. The
> phenomenon is general to positive activations; paper §2 rewritten accordingly. Details in the
> trajectory entry of that timestamp.
- **Question.** Is silence a *gradient trap* (a unit at zero gets no gradient and cannot return) or
  an *attractor of the optimization* (units are pushed to zero even when gradient flows)?
- **Why it mattered.** Paper §2 originally explained silence through a ReLU-specific mechanism (the
  dead gradient at zero, and for a while the scale symmetry — both now retracted). The "what we
  tried" table recorded 40–64% silence under softplus and leaky-ReLU in sign-constrained networks;
  the question was whether any activation-specific account survives. None does.
- **Design.** `none`, N = 2000, unconstrained, trainable bias, both tasks; leaky-ReLU (slope 0.01),
  softplus (β = 25), **sigmoid and tanh**; 3 seeds each → 24 runs, 200k iterations. ⚠️ Every sigmoid
  and tanh network on disk (`CDDM_SparsityAndFRMagnitude-sigmoid_shifted*`, `CDDM_tanh_slope=*`) was
  trained with `frm+rws` ON under Dale, so there is no unpenalized bounded-activation baseline to
  reuse. ⚠️ For bounded activations silence must be read with a **modulation criterion** (temporal
  std of the rate below a scale-free fraction of the population's), because a unit saturated at a
  constant `sigmoid(bias)` is functionally dead yet passes the participation criterion (q90 > 0).
  Report both criteria for every activation so the two notions of silence can be compared.
- **Outcome (realised).** Sigmoid showed > 30% unmodulated units (62–76%): the phenomenon is
  "nothing in the objective keeps a unit alive", general to activations, and §2 was rewritten that
  way. tanh was not run and is not needed for the claim.

### T2. What walks the units down — weight decay sweep
- **Question.** §2 says units the solution does not need are walked to the floor by weight decay and
  noise. Is weight decay the walker?
- **Why it matters.** The walk is measured (four activations, same silent fraction); its driver is
  asserted. A referee will ask what happens at weight decay 0.
- **Design.** `none`, N = 2000, k = 3, weight decay ∈ {0, 1e-6, 1e-5, 1e-4, 1e-3}, 3 seeds → 15
  runs, both tasks if cheap. Read: silent fraction and participation Hoyer along training.
- **Outcome that changes the paper.** Silence vanishes at WD = 0 → weight decay is the driver and the
  paper names it. Silence persists at WD = 0 → the walk is driven by noise and the task gradient's
  own dynamics, and §2 says "the objective does not hold a unit up, and training does not need to
  push it down for it to fall". Either way §2 gains its missing measurement.
- *(A gain-normalisation control that rescales each unit's incoming and outgoing weights was
  proposed here while the ReLU scale symmetry was thought to be the cause. It is only
  function-preserving for homogeneous activations and is withdrawn with that explanation.)*

### T5. Is the silence partly an initialisation artefact? — W_inp initial scale
> **Status 2026-09-11 09:44 — RUNNING.** Spock array `6154727` (24 tasks: s ∈ {0.5, 2, 5, 20} per-row
> norm × N ∈ {500, 1000} × 3 seeds, 150k it.), read-out job `6154728`. Design, read-out and decision
> rule in the trajectory entry of this timestamp. Knob: `model.input_row_norm` (commit `06082b1`).
- **Question.** W_inp is initialised at std 1/√N per entry (row norm 0.039 at N = 2000) and grows
  50× in the first 100k iterations to reach the task's operating scale; the units that survive are
  those whose input rows get amplified, and silent units' rows decay to 0.003. Would a W_inp
  initialised at the operating scale change the silent fraction?
- **Why it matters.** Every network in the project — all four activations included — shares this
  init. If a correctly scaled init cuts silence substantially, part of the phenomenon is a
  training-dynamics artefact of an under-scaled input, and the paper must say so; if it does not,
  the objective-level explanation stands with one more alternative excluded. Either outcome changes
  §2.
- **Design.** Add `model.input_gain` (multiplies the drawn W_inp; default 1.0 so every existing run
  is unchanged). `none`, 3-bit flip-flop, N = 500 and 1000, 3 seeds, 150k iterations, standard ReLU
  RNN, `input_gain` ∈ {1 (control), 10, 50, 150} — 150× is the live rows' final scale; 10× puts a
  pulse's drive at the noise level. 24 runs; ~6–9 h each on Spock. Read: silent fraction along
  training (does the early global collapse still happen?), participation Hoyer, the four axes.
- **Outcome that changes the paper.** Silence < 20% at any gain with task R² ≥ 0.9 → the init scale
  is a cause and the recommendation is a scaled input init before any penalty; §2 rewritten.
  Silence unchanged → excluded, one sentence in §2.2. Intermediate → reported as a dependence.

### T3. Is "modular" a property of the constraint or of the number 20?
- **Question.** Does the assembly / purity result depend on the in-degree target?
- **Why it matters.** §5's composition claim is stated for `S* = 20`. A referee will ask about 200.
- **Design.** `both`, N = 2000, k = 3, `S*` ∈ {5, 20, 50, 200, 800 (= no constraint)}, 3 seeds → 15
  runs. Read: assembly share, wiring Q above null, purity (Hoyer and NMF), temporal Hoyer.
- **Outcome that changes the paper.** Monotone in `S*` → the claim becomes "the tighter the
  in-degree, the more modular", which is stronger. Non-monotone or flat above 20 → the number is
  reported as a tuned choice with its sensitivity.

### T4. The `frm` cap sweep — rate heterogeneity as a tunable, not a limitation
- **Question.** Can the σ_log collapse (0.26 under `frm` vs ~1 in cortex) be traded off against
  the silent fraction by the cap and temperature?
- **Why it matters.** §6 admits `frm` produces a population that is too uniform. Without this sweep
  that stays a limitation; with it, it becomes a recommendation.
- **Design.** `frm`, N = 1000, CDDM, `cap_fr` × `τ` over a 3 × 3 grid, 3 seeds → 27 runs, 100k
  iterations.
- **Outcome that changes the paper.** A setting with < 5% silent and σ_log > 0.6 exists → §6
  gains a tuned recommendation. None exists → the limitation stands and is stated as a trade-off
  curve.

---

## C. Supplementary — supports the story, does not carry it

| # | item | what it adds | status |
|---|---|---|---|
| S1 | Dale-constrained and I/O-positive networks | phenomenon is not an artefact of the constrained architecture | ✅ paper §S1 |
| S2 | Matched-performance protocol; floor is size-independent; seven convergence criteria fail | licenses every cross-N comparison | ✅ methods paragraph |
| S3 | Saturating vs power-law model comparison for M(N) | one sentence: "decelerating; ceiling estimate depends on the matched level" | ✅ supplement |
| S4 | Two silence criteria disagree; `rws` parks units just above 1e-6 | why two criteria are always reported | ✅ methods + one figure |
| S5 | Penalty-switch intervention: full four-arm statistics, churn at stated sampling rates, retracted `corr(Δ, start)` | causal support for §5 | ✅ supplement; main text keeps the two headline reversals |
| S6 | Task-free purity (NMF, nnICA), calibration record | selectivity result without task variables | ✅ supplement; one sentence in main |
| S7 | Wiring–activity ARI, matched n; eigenvalue outliers do not count assemblies; Louvain | structure–function correspondence | ✅ supplement |
| S8 | Rigotti-style interaction terms in the regression | bounds nonlinear mixing at ~0 on this task | ⬜ analysis, one afternoon |
| S9 | Is `frm`'s selectivity carried by `W_inp`? | completes "function without block structure in `W_rec`" | ⬜ analysis |
| S10 | Cluster-count sweep around 2k for the modularity table | closes the last methods question there | ⬜ analysis |
| S11 | Identifiability demo: subsample K units, fit a population model, compare recovery | the "so what" for modellers | ⬜ analysis, no training |
| S12 | Lesion robustness by participation rank | does the rescued network *function* better | ⬜ analysis |
| S13 | Time-to-floor T(k) curves and the 1.01× / 1.03× budgets | training-cost supplement to A5 | ✅ flip-flop; ⬜ CDDM |
| S14 | Readout: `W_out` targets clean units 4× under `frm` but is not sparser | minor support for §5 | ✅ supplement |

---

## D. Out of scope for this paper (do not re-propose)

- **Second task family** (DMTS, GoNoGo, MemoryAngle). Two tasks with every claim shown on both is
  the bar for this paper; a third task is a follow-up.
- **Heavy-tail statistics, Mardia kurtosis, intrinsic dimension, ePairs, cluster-elbow sweeps.**
  All retracted (trajectory 2026-09-09); the star geometry made every one of them fit the wrong
  model.
- **Master-inhibitor and frozen-clamp constructions.** Superseded by the participation-trace
  analysis.
- **Prevent-vs-resurrect from a hard-silent initialisation** (20 jobs). Only if a referee insists.
- **N = 4000 `frm`/`both` on the flip-flop.** `frm` at N = 4000 does not fit a 46 GB GPU; the
  argument does not need it.
- **Biological realism of the rescued population.** The paper's claim is scoped to
  *identifiability and analysis validity*; whether a pure-selective, rate-uniform population is
  cortex-like is explicitly not claimed (mixed selectivity is a feature of cortex in the literature
  the paper will be reviewed by).
