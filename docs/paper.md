# Paper plan — silent units in trained ReLU RNNs

Restructured 2026-09-10 (the previous structure is in git history before commit `a2541c2`'s
successor). Working document: the *argument*, what supports each claim, what is missing. The
experimental record is [`project_trajectory.md`](project_trajectory.md); what is proposed but not
done, and what is deliberately left out, is [`research_directions.md`](research_directions.md).

**Every claim carries an evidence status** — ✅ measured, 🟡 preliminary, ⬜ planned — so that
interpretation never quietly becomes result. **Every main-text claim is shown on both tasks** (CDDM
and the n-bit flip-flop) or is explicitly marked as task-specific. Sign-constrained architectures
are supplementary throughout (§S1).

**Working title.** *Why trained RNNs leave most units silent, and what it takes to make every unit
count.*

**The argument in five sentences.** (1) A ReLU RNN trained on a neuroscience task leaves most of its
units silent, and enlarging the network does not fix this: the active count grows as roughly the
square root of N on both tasks, so a thousand active units costs a network of ten thousand. (2) The
cause is that the task loss has no term that keeps any unit active: units the solution does not
need are walked down to the activation's floor and left there, and the same silent fraction appears
at the same size and iteration under ReLU, softplus, leaky-ReLU and a bounded sigmoid, so it is not
a property of any activation; standard regularizers either do not touch it or make it worse. (3) A
floor on each unit's activity (`frm`) supplies the missing term and every unit stays alive, at no
task cost, but the
units it keeps alive are diffuse: they listen to the whole population, mix several task variables,
and their tuning degrades as N grows. (4) A cap on each unit's effective in-degree (`rws`), useless
on its own, forces the recruited units into the modular, single-variable, assembly-wired solution
that the unpenalized network builds for its few survivors — and this is causal and reversible within
a network. (5) Which population an RNN model presents to analysis is therefore a training choice, and
we characterize the four resulting populations on both tasks with one bounded statistic.

---

## 1. Most units are silent, and size does not fix it ✅ (both tasks)

### 1.1 The phenomenon

In standard ReLU RNNs (unconstrained weights, trainable bias, `h` equation) trained on CDDM,
**41–46% of units never fire** at N = 1000; 31–42% sit at exactly 0.0 at every timestep of every
condition. Training creates most of it (4.4% are hard-silent at initialization) and does not stop
creating it: silence keeps climbing at 10⁵ iterations while the loss is flat (§S2). On the flip-flop
at N = 2000, k = 3, **86% of units are silent** under the task-calibrated criterion.

> Two silence criteria are reported everywhere — hard (`p < 10⁻⁶`) and scale-free
> (`p < 0.05·q95(p)`) — because they disagree, and the disagreement has flipped a conclusion in this
> project (§2.3). Never one alone. (§S4)

### 1.2 The active count grows as ~√N, on both tasks

End-of-training networks, unpenalized, three seeds per size (`characterize.py`):

| task | sizes | `M ∝ N^b` | N for 1,000 active units | for 2,000 |
|---|---|---|---|---|
| CDDM | 100 – 10,000 | b = 0.53 | **1.3 × 10⁴** | 4.6 × 10⁴ |
| flip-flop (k=3) | 500 – 4,000 | b = 0.58 | **1.4 × 10⁴** | 4.6 × 10⁴ |

Under matched performance rather than end of training (§S2, CDDM) the exponent is lower still,
0.31–0.36, and the growth is significantly decelerating (curvature p = 0.001). The end-of-training
numbers above are *conservative* for the claim — large networks were trained for fewer iterations
and are read earlier in their silencing — and they agree across tasks. **Whether or not the count
saturates, a thousand active units costs a network of order 10⁴.** The active *fraction* falls
monotonically with size on both tasks (CDDM 0.74 → 0.09 from N = 100 to 10,000; flip-flop 0.32 →
0.14 from 500 to 4,000): bigger networks are emptier.

### 1.3 It is not spare capacity ✅ (flip-flop)

If silence were the network idling on an easy task, the active count should track task demand. Over
k = 1…8 bits crossed with N = 500…4000, fitting `M = A·N^b·k^c`: **c ∈ [−0.05, +0.04] in every
condition**, an 11% change over an eight-fold increase in memory demand, with inconsistent sign.
Meanwhile the activity's dimensionality does rise with k (≈ 1.8k). *More task dimensions, the same
number of units.* Trained RNNs recruit a number of units set by neither the size nor the task.

---

## 2. Why: nothing in the loss keeps a unit alive

### 2.1 The mechanism ✅ activation-general, ⬜ one direct test proposed

The task loss has no term that keeps any particular unit active. A unit the solution does not need
receives no gradient that holds it up; weight decay and noise walk it down to the activation's
floor, and it stays there because the loss does not care and the gradient at the floor is small
(zero for ReLU, ~0.01 for leaky-ReLU, ~0.02 for softplus and the sigmoid at its lower asymptote).
This is a statement about the objective, not about the activation, and the data say so.

> **Retracted explanation.** An earlier version of this section attributed silence to the ReLU
> scale symmetry (`relu(a·x) = a·relu(x)`, which makes a unit's gain a flat direction of the loss).
> A bounded sigmoid has no such symmetry and silences the same fraction of units at the same size
> and iteration (below). The symmetry is not the cause and is not used anywhere in the argument.

**Four activations, one silent fraction** ✅. Unpenalized, same size, read at the same iteration:
ReLU, softplus (β = 25), leaky-ReLU (leak 0.01) on CDDM at N = 1000 (Dale) are indistinguishable on
every axis (live 0.44 / 0.45 / 0.45; participation sparsity 0.74 / 0.73 / 0.74). On the flip-flop,
a bounded `sigmoid(7.5(x − 0.3))` standard RNN silences 0.75–0.77 of its units at N = 1000 against
ReLU's 0.72–0.75 at the same iteration, with the silent units parked at the lower asymptote (mean
rate 0.002, none saturated high) and still silencing at 150k iterations exactly as ReLU does. A
bounded, nonlinear positive part does not remove the phenomenon. Its concentration
is milder in threshold-free terms (participation sparsity 0.54 vs 0.73) only because the floor is
soft: silent units sit at 0.002 instead of 0.

Evidence that this is the right picture: the collapse is *global and early* — the whole population
goes quiet within ~20 iterations and only the eventual survivors climb back (§S5); silence grows
monotonically with training time (30k → 300k iterations makes it worse) and with N; the network
declines even the cheapest escape (with self-connections allowed it trains the diagonal into
self-*inhibition*, corr(self-weight, log participation) = −0.51). And the one intervention that
works (§3) is the one that adds to the objective exactly the term it lacks: a floor on activity.

> ⚠️ **What remains untested.** Weight decay is asserted to be the walker; a sweep over it,
> including weight decay = 0, has not been run (`research_directions.md` T2). Until then the *driver*
> of the walk is the best-supported reading; that the walk happens and is activation-general is
> measured.

### 2.2 What does not work ✅ (CDDM; activation rows in constrained nets, §S1)

Fifteen interventions, condensed. Architecture (`h`/`s` equation, cubic term, boundary handling,
sign constraints, I/O positivity, trainable bias, self-connections): no change. Activation
(softplus, leaky-ReLU, sigmoid): persists at the ReLU level (§2.1). Recurrent noise: never
helps, σ = 0 is worst. Longer training and larger networks: worse, monotonically. The field-standard
metabolic cost `mean(fr²)` over four decades of λ: never rescues, and at λ = 10 makes it worse
(N = 100: 12% → 59% silent). Full table in §S3.

### 2.3 The sparsity penalty does not rescue, and looks like it does ✅ (CDDM)

`rws` — a penalty driving each unit's effective in-degree `S = (Σ|W|)²/ΣW²` toward 20 — is the
one standard-looking regularizer that *appears* to help: at N = 2000 hard-silent falls from 79% to
60%. It is a threshold artifact. Under the scale-free criterion `rws` is **worse** than baseline
(86% vs 81%), the gap grows with N (12 → 21 → 26 pp at N = 500/1000/2000), and the "rescued" units
sit just above 10⁻⁶ doing nothing. `rws` constrains wiring, not activity — nothing in it holds a
unit up — so it cannot rescue; what it does do becomes visible only in §5.

---

## 3. The fix: a floor on each unit's activity ✅ (both tasks)

`frm` drives each unit's soft-max firing rate toward a cap (`τ·logsumexp(r/τ)` with τ = 0.1, cap
`0.3·log1p(100)/log1p(N)`, λ = 0.1). It is the one intervention in §2.2 that gives the objective a
term that holds every unit up, and it is the one that works, on both tasks:

| | CDDM | flip-flop k=3 |
|---|---|---|
| silent fraction under `frm`, every N tested | **0.0%** (both criteria; min participation 4.8 × 10⁻², nothing near zero) | **0.0–14%** (N = 500 → 2000; §4) |
| active count law | `M ∝ N^1.00` | `M ∝ N^0.90` |
| task cost (noise-free loss, N = 2000) | **14% better** than unpenalized; `both` 35% better | no cost in the switch experiment (§5.3); ⬜ grid-wide time-to-floor (A5) |

> **Name the tautology before a referee does.** "A floor on firing removes sub-floor units" is
> nearly circular. The content is elsewhere: it costs nothing and *helps* at large N; unpenalized
> networks concentrate onto a task-independent absolute number of units; the standard regularizers
> fail or make it worse; and the population it produces differs from the unpenalized one in ways
> that change conclusions (§6).

**Prevention, not resurrection** ✅ (CDDM, unconstrained). From participation traces logged every
10 iterations: unpenalized, 96 units per 1,000 endure ≥ 500 silent iterations and recover; under
`frm`, 0.6. Units dip in the early collapse and are caught within a few hundred iterations. (In
sign-constrained networks `frm` genuinely resurrects, §S1.)

**The rescued units are task-tuned** ✅ (both tasks). Regressing each unit on rectified task
variables (§7 for the basis), per N = 2000 network: flip-flop, tuned units 241 (none) → 1,401 (`frm`)
at median R² 0.61 → 0.84; CDDM, tuned fraction of live units 0.85 → 0.42 but live units 308 →
2,000, so 262 → 840 tuned units at R² 0.32 → 0.69. `frm` is not buying non-zero units that do
nothing.

---

## 4. What the floor alone produces: alive, diffuse, and degrading with N ✅ (both tasks)

The rescued population is not the unpenalized population scaled up. Four properties, each a Hoyer
sparsity (0 = spread evenly, 1 = concentrated on one; §7), N = 2000, three seeds:

| axis | vector | CDDM: none / frm | flip-flop: none / frm |
|---|---|---|---|
| **participation** | per-unit participation, d = N | 0.93 / **0.22** | 0.77 / **0.21** |
| **selectivity** | \|tuning coefficients\|, d = 7 or 2k | 0.48 / **0.32** | 0.76 / **0.63** |
| temporal | a unit's trace over samples | 0.73 / 0.71 | 0.41 / **0.53** |
| dimensionality (D_PR) | covariance eigenvalues | 2.3 / **8.2** | 6.2 / **8.9** |

Read down the `frm` column:

- **It spreads activity over units** — participation sparsity falls from ~0.8–0.9 to ~0.2 on both
  tasks. That is the rescue, seen as a distribution rather than a count.
- **Its units are less selective, and get less selective as N grows**, on both tasks: flip-flop
  0.86 → 0.82 → 0.63 and CDDM 0.37 → 0.36 → 0.32 from N = 500 to 2000, while `none` holds level.
  In the joint-versus-single-variable regression the median tuned unit is single-bit in *every*
  condition (gap 0.001–0.002), but **24–38% of `frm`'s tuned units gain > 0.1 R² from a second bit,
  against 0.5–10% under the other three** (flip-flop, N = 2000, k = 3). Mixed selectivity under `frm`
  is a distinct subpopulation, not the typical unit. Task-free, the same: `frm` has the lowest fraction
  of units following one population factor (NMF, 0.31–0.56 vs 0.77–0.88 for `both`; §S6).
- **It listens to everyone.** Effective in-degree `S ≈ 800` at N = 2000 against ~20 with `rws`.
  The recurrent wiring has no block structure (modularity above null 0.14 vs 0.33 unpenalized) while
  the *activity* is as modular as anyone's (0.29 vs 0.34): function present, not written into `W_rec`
  (§S7).
- **It raises dimensionality with N** on both tasks (CDDM D_PR 5 → 10 over N = 500 → 5000; flip-flop
  5.6 → 8.9 over 500 → 2000) where the unpenalized network's dimensionality is set by the task and
  flat in N.
- **On the flip-flop it is also transient and unstable**: units burst (temporal sparsity 0.53 vs
  0.41), ~10% are dead at any moment but not the same 10% (2.9 dead↔alive crossings per unit over
  50k iterations), and its live count falls with N (0.99 → 0.86 of N). ⚠️ **Task-specific**: on CDDM
  `frm`'s temporal sparsity equals the unpenalized value and its live fraction is 1.0 at every N.

The cheap way to satisfy "be active" when you may listen to anyone is to listen to a little of
everything. `frm` alone takes it.

---

## 5. What the in-degree cap adds: the modular solution, extended to every unit ✅ (both tasks, one part flip-flop only)

### 5.1 The four populations, one statistic (both tasks)

Same table as §4 with all four conditions, N = 2000, mean of three seeds (`characterize_matrix.png`):

| axis | CDDM none / rws / frm / **both** | flip-flop none / rws / frm / **both** |
|---|---|---|
| active fraction | 0.15 / 0.14 / 1.00 / **1.00** | 0.14 / 0.15 / 0.86 / **1.00** |
| participation sparsity | 0.93 / 0.92 / 0.22 / **0.23** | 0.77 / 0.78 / 0.21 / **0.03** |
| selectivity sparsity | 0.48 / 0.41 / 0.32 / **0.45** | 0.76 / 0.80 / 0.63 / **0.90** |
| temporal sparsity | 0.73 / 0.69 / 0.71 / 0.74 | 0.41 / 0.43 / 0.53 / **0.39** |
| D_PR | 2.3 / 2.2 / 8.2 / 8.5 | 6.2 / 6.1 / 8.9 / **5.5** |

What replicates across tasks, in every seed: **`both` keeps all units active *and* restores the
selectivity of the unpenalized survivors** (CDDM 0.45 vs `frm` 0.32; flip-flop 0.90 vs 0.63), and
its selectivity does *not* degrade with N where `frm`'s does (flip-flop `both` 0.84 → 0.86 → 0.90;
CDDM 0.40 → 0.41 → 0.45 over N = 500 → 2000). The difference between `frm` and `both` is not how
many units are alive; it is what kind of unit they are.

What is flip-flop-specific: the participation floor (`both` 0.03 vs `frm` 0.21; on CDDM they are
equal), the temporal stabilization (§5.3), and the dimensionality reduction (`both` 5.5 below
`none` 6.2; on CDDM `both` raises it like `frm`). ⚠️ These are reported as flip-flop results.

### 5.2 Mechanism: `rws` forbids the diffuse solution and the task's own assemblies return (flip-flop; ⬜ CDDM = A3)

A k-bit flip-flop has 2k states and its natural recurrent memory is an assembly per state. The
fraction of a tuned unit's recurrent input coming from same-state units, over chance:

| | k = 3 | k = 8 |
|---|---|---|
| none | 3.1× | 5.1× |
| rws | 3.7× | 8.8× |
| **frm** | **1.7×** | **3.2×** |
| both | 4.0× | 11.5× |

The unpenalized network builds assemblies unprompted among its survivors; `frm` is the one condition
that dissolves them; `rws` restores them, more strongly the harder the task. Wiring modularity above
its null: 0.33 / 0.40 / **0.14** / 0.20; activity modularity the same in all four (§S7). A unit in an
assembly inherits its activity from the task: a bit is in a given state 36% of the time by design,
and the modal occupancy of a `both` unit is 0.35–0.38 at every k from 1 to 8.

> **The honest magnitude.** `both` is *not* more modular than the unpenalized network. It has the
> same modularity extended from 275 units to 2,000. The gain from `rws` is structure in every unit
> where `frm` alone has none in the wiring — not more structure per unit.

> **Why the penalties need each other.** `frm` demands every unit be active but says nothing about
> how. With thousands of units and a handful of states, the cheap way is to listen to everyone;
> `frm` alone takes it. With the in-degree capped, a unit cannot, and the only remaining way to
> satisfy the floor is to draw its ~20 inputs from units carrying the same signal — an assembly.
> `frm` alone: active but diffuse. `rws` alone: modular but mostly dead (§2.3). Together: active and
> modular.

### 5.3 It is causal and reversible ✅ (flip-flop, N = 2000, k = 3, 3 seeds)

Four arms warm-started from converged 400k networks and trained 50k more; paired same-penalty
controls.

| arm | switch | median temporal PR | dead fraction | assembly share |
|---|---|---|---|---|
| A1 | `frm` → `frm+rws` | 0.157 → **0.353** | 0.106 → **0.000** | 0.28 → **0.64** |
| A2 | `frm` → `frm` (control) | 0.159 → 0.209 | 0.106 → 0.094 | — |
| A3 | `both` → `frm` | 0.353 → **0.228** | 0.000 → **0.088** | 0.67 → **0.28** |
| A4 | `both` → `both` (control) | 0.354 → 0.359 | 0.000 → 0.000 | — |

Forward and reverse effects are equal and opposite; controls are inert (A4) or drift mildly (A2 — the
control that makes reading A1 against zero wrong). Dead↔alive crossings per unit at 250-iteration
sampling: 0.35 with `rws` active after the switch, 2.70 without. No task cost (loss 0.027–0.030 in
every arm). The pre-registered hypothesis — that `rws` *molds* transient units into sustained ones —
**failed** both its criteria (identity ρ = 0.26 < 0.5; the `corr(Δ, start)` signature vanishes
against its null, §S5). The supported claim is narrower and better: **`frm` cannot hold units alive
on this task; `rws` stabilizes them there, and the structure is a state set by whichever penalty is
active, not by training history.**

---

## 6. Consequences: the population an RNN presents to analysis is a training choice ✅ (CDDM; ⬜ flip-flop = A6)

Networks that perform identically (R² 0.84–0.87, N = 1000) differ by 3.5× in effective
dimensionality (2.2 → 7.7), 1.7× in choice selectivity *among active units* (34% → 58%), 2× in total
metabolic cost (the concentrated solution is the expensive one), and 100× in how concentrated that
cost is (carried by an effective 8 units vs ~850). Every population-level comparison with data —
dimensionality, selectivity fractions, cell classes, correlation structure — inherits this, and by
§1.2 it is worst in exactly the large networks the field is moving toward.

**Two limitations that belong in the main text.**

- **Rate heterogeneity goes the wrong way.** σ_log (std of log₁₀ mean rate across active units) is
  1.2 unpenalized — the cortical value — and 0.26 under `frm`: one unrealism traded for another.
  Within-trial modulation survives (temporal CV 1.29 vs 1.43), so the penalty is not satisfied by
  tonic firing, but `both` flattens modulation further (0.96). ⬜ The cap × temperature sweep that
  would turn this into a recommendation is T4 in `research_directions.md`.
- **Pure selectivity is not a claim of realism.** Mixed selectivity is a computational feature of
  cortex in the literature this paper will be reviewed by. Our claim is scoped to *identifiability
  and analysis validity*: a population whose units each follow one task variable, wired in blocks
  one can read off `W_rec`, is one whose computation can be recovered from its weights. Whether
  cortex is like that is not claimed. Likewise, "cortex does not have 45–85% silent neurons" must
  engage the dark-neuron literature: the RNN silence is exact zeros with zero gradient, not sparse
  firing, and the argument is about the analysis pipeline.

---

## 7. Methods that must be stated

- **One statistic for four axes.** Hoyer sparsity `(√d − ‖v‖₁/‖v‖₂)/(√d − 1)` of a non-negative
  vector; `‖v‖₁/‖v‖₂ = √PR`, so it is the participation ratio normalized to [0, 1] and flipped.
  Applied to covariance eigenvalues (dimensionality; ⚠️ with d = N this saturates near 1 because
  D_PR ≪ N everywhere, so D_PR is the informative number and Hoyer is reported for completeness),
  per-unit participation (concentration over units), one unit's trace (temporal sparseness; the dual
  of temporal PR), and one unit's tuning coefficients (selectivity; **mixed selectivity is defined as
  1 − this**).
- **The regression basis is half-wave rectified**: `relu(+x), relu(−x)` for each signed task
  variable plus an intercept. ReLU units fall to exactly zero for one sign of a variable, which a line
  through three levels cannot fit; the signed basis understates R² by ~0.14 and manufactures a
  spurious untuned population under `frm`. No regularization (condition number 6.7, CV gap 0.0005).
  Flip-flop: 2k regressors, time-resolved. CDDM: 7 regressors (context, ± motion, ± colour, ± choice)
  on the decision-epoch mean per condition. R² and tuned fractions are not comparable across tasks;
  orderings across penalties within a task are.
- **Silence criteria.** CDDM: hard 10⁻⁶ and scale-free `0.05·q95`; flip-flop: `4 × 10⁻²`, Otsu-
  calibrated on log participation (the CDDM threshold sits below both flip-flop modes and reports
  0% silence there). Participation `p_i = std(r_i) + q₀.₉(|r_i|)`.
- **Matched-performance protocol** for cross-N comparison on CDDM (§S2): read each size at the
  iteration where the smoothed noise-free loss stably crosses a common level; licensed by a floor that
  is size-independent to 1.6%. The learning rate is rescaled `lr × (100/N)^{1/3}`, so N and lr
  co-vary and fixed-iteration comparison is biased toward *under*-estimating silence at large N.
- **Loss is reported noise-free on the task term only.** `TrainLosses.json` is task + λ·penalty with
  noise on; comparing it across penalties has produced a wrong conclusion twice.
- Penalties: `frm` λ = 0.1, τ = 0.1, cap `0.3·log1p(100)/log1p(N)`; `rws` λ = 0.05, target in-degree
  20; weight decay 10⁻⁶; training noise σ_rec = σ_inp = 0.05, σ_out = 0.03. ⚠️ None of the penalty
  hyper-parameters has a sensitivity sweep in the main text (T3, T4).
- **A practical result** ✅: strong activity penalties can make training diverge without gradient
  explosion (`frm` builds a self-exciting loop with gain > 1; explicit Euler overflows within the
  trial while gradient norms stay ~1). dt 1 → 0.5 with a bounding nonlinearity cut divergence from
  ~50% to ~10%.
- Per-network spread in every table; n = 3 seeds per cell throughout, 5 for the CDDM distortion set.
  Pre-registered vs post-hoc is marked: §5.3's molding hypothesis was pre-registered and failed;
  §4's mixed-selectivity subpopulation and §5.1's cross-task table were exploratory.

---

## 8. Venue

Target **Nature Communications** (repository set up per `paper_repo_setup.md`); the story is a
mechanism with a fix and a characterization, shown on two tasks. Honest fallback if T2 is not
run before submission: **PLOS Computational Biology**, with §2.1 stated as interpretation. bioRxiv
immediately.

**What decides which:** T2 (weight-decay sweep and gain-normalization control). T1 is done: the
phenomenon is activation-general. With T2, §2 is a measured mechanism; without it, the walk is
measured and its driver is the best-supported reading.

---

## S1. Supplementary: Dale-constrained and I/O-positive networks ✅

Not part of the main argument. Silence is overwhelmingly excitatory (53–55% of E units vs 3.5–5% of
I units, falsifying the readout-starvation hypothesis); I/O positivity conceals how dead the units
are (2.2% hard-dead vs 31.4% unconstrained at the same total silent fraction; every hard-zero unit
in an unconstrained net has Σ`W_inp` < 0); `rws` reverses sign under Dale; `frm` resurrects rather
than prevents; the `s` equation is more sensitive to constraints than `h`.

## S2. Supplementary: convergence and the matched-performance protocol ✅

Training does not converge in any strict sense: seven stopping criteria fail (parameter change decays
as a power law, exponent −0.29 to −0.57, so a 1% criterion extrapolates to 10⁶ iterations; loss-
trace criteria fire 7× too early; silent-count criteria never fire). The replacement is matched
performance, with the floor shown size-independent to 1.6% by three decay families; the floor itself
is task structure (98% of residual loss from |coh| < 0.05 where the target is discontinuous), not
capacity. `T_N ∝ N^0.27`. Under this protocol M(N) is significantly decelerating (curvature
c = −0.118 ± 0.033, p = 0.001), the saturating fit beats the power law under the hard criterion
(ΔAICc −9 to −18) but not the scale-free one, and the implied ceiling depends on the matched level
(3972 → 880 as L* deepens). The main text therefore claims the cost of active units, not a ceiling.

## S3. Supplementary: the full "what we tried" table ✅

Fifteen rows (§2.2), with the self-inhibition result (self-weights trained negative, active units at
−0.060 vs silent at −0.007) and the divergence mechanism (§7).

## S4. Supplementary: two silence criteria, and the `rws` artifact ✅

The hard/scale-free gap is 11.9 / 20.7 / 25.5 pp under `rws` at N = 500 / 1000 / 2000 and ≤ 2 pp in
every other condition; the apparent late-training "resurrection" under `rws` (hard-silent 75.5% →
60.5%, peaking at iteration 76k) is units crossing 10⁻⁶ without becoming active. `rws` silences on
a different timescale (median first crossing at iteration 3287 vs 16).

## S5. Supplementary: the penalty-switch intervention in full ✅

Churn at both sampling rates (1.7 vs 3.2 at every 10 iterations; 0.30 vs 2.94 at every 250 — different
phenomena: fast flicker under `both`, slow persistent switching under `frm`); the identity statistics
and why rank correlation is not the right readout (IQR compression 0.330 → 0.179); the
`corr(Δ, start)` statistic and its regression-to-the-mean bias (excess over per-arm null +0.11 vs
+0.29, the *opposite* of the naive reading); the one seed with the `frm` gradient-spike problem (498
skipped updates), retained as a documented example. The global early collapse and spontaneous
recovery statistics (96 units per 1,000 unpenalized) also live here.

## S6. Supplementary: selectivity without task variables ✅

NMF of the rate matrix, factors at unit norm, effective number of factors per unit `(Σh²)²/Σh⁴`;
calibrated before any real network was read (pure 1.000; equal two-arm sums 1.997 vs truth 2); the
L1 form of the statistic and a random-angle mixed synthetic were both rejected by calibration and
why; non-negative ICA cross-check at k = 3 (it does not identify the arms in 16 dimensions). Result:
`frm` lowest pure fraction in every seed at k = 3 and 8 under both methods; `both` highest under NMF
by a wide margin at k = 3 and a narrow one at k = 8. What "pure" means (relative to the finest
anchored population pattern; a conjunction unit reads pure) and how it differs from Rigotti-style
nonlinear mixing (interaction terms ⬜ S8).

## S7. Supplementary: modularity, wiring–activity correspondence, readout ✅

Matched n (275), 2k clusters, each partition against its own null: activity Q above null 0.34 /
0.33 / 0.29 / 0.33, wiring 0.33 / 0.40 / 0.14 / 0.20; `both`'s raw wiring Q inflated by sparsity
(null 0.32 vs 0.06–0.15); ARI(wiring, activity) 0.61 / 0.62 / 0.30 / 0.48 ± 0.15 — `frm` the outlier,
the rest comparable. Eigenvalue outliers of `W_rec` do not count assemblies (they are E/I-balanced);
Louvain does not recover 2k unprompted. `W_out` targets clean units 4× under `frm` but is not
sparser. Prior work to verify: like-to-like connectivity (Ko 2011, Cossell 2015); weight vs
activation clusterability (Hod/Casper/Filan ~2021); structure from connectivity (Dubreuil 2022) and
from function (Yang 2019).

## Retracted claims (kept so they are not re-derived)

- "Median of the 101 lowest losses" as a performance statistic — noise lottery.
- "93–96% of the loss is irreducible" — model-dependent.
- A size effect in noise tolerance — failed to localize.
- "Training never reaches a stationary regime" — overstated; motion is along a flat manifold.
- "`frm` costs ~5% in task loss" — computed from the total objective; on the task term it is level or
  better. **Never quote `TrainLosses.json` across penalty conditions.**
- "`rws` partially rescues at large N" — hard-threshold artifact (§S4).
- Monotone decay exponent with N — mismatched fit ranges; this manufactured a trend three times.
- "7–9 functional cell types", silhouette without null, intrinsic-dimension ordering, Mardia
  kurtosis ordering — the selectivity space is a 2k-armed star and each fitted the wrong model.
- "`both` is the most task-aligned / most modular" — unmatched-n artifact; matched, `both` has the
  unpenalized modularity extended to all units.
- "`rws` molds transient units into sustained ones" — pre-registered, both criteria failed; it
  stabilizes.
- "`frm` has an untuned quarter" — misspecified (signed) regression basis.
- **"Silence is caused by the ReLU scale symmetry `relu(a·x) = a·relu(x)`"** — a bounded sigmoid,
  which has no such symmetry, silences the same fraction of units at the same size and iteration
  (0.75–0.77 vs 0.72–0.75 at N = 1000, flip-flop). The cause is the objective's missing term, not
  the activation's homogeneity. Removed from every section on 2026-09-11.
