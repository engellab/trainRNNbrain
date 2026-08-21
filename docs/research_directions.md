# Silencing and effective size in task-trained recurrent networks

Research directions, drafted 2026-08-20 18:51. Separate from [`paper.md`](paper.md), which is the
argument of the *current* paper, and from [`project_trajectory.md`](project_trajectory.md), which is
the experimental record. This file is for work that goes beyond the current paper.

**Evidence status is marked throughout** — ✅ measured, 🟡 preliminary, ⬜ planned, ❓ assumed but
never tested — because several claims that read as established in the first draft are in fact
hypotheses, and one of them is the load-bearing claim of Direction 2.

---

## Shared background

Computational neuroscience routinely trains recurrent neural networks on cognitive tasks and treats
the trained network as a model of a cortical circuit. Population dynamics are analyzed, fixed points
extracted, dimensionality measured, and conclusions drawn about neural computation. Network size N is
reported as a property of the model.

The observation motivating all three directions: when RNNs with ReLU-family nonlinearities are
trained on cognitive tasks, **the number of units that end up active does not scale with N** ✅. At
N = 5000 roughly 730 units are active (731.7 ± 31.1 at matched performance L\* = 0.023). So a network
reported as N = 5000 is computationally a network of ~730.

❓ **"The silent units are genuinely inert — ablating them does not change behavior."** Split this in
two. For **hard-silent** units (`p < 1e-6`, i.e. r ≡ 0 at every timestep in every condition) it is a
mathematical identity, not an empirical claim: a unit whose rate is identically zero contributes
nothing to `W_rec @ r` or `W_out @ r`, so ablation provably cannot change the output. For the
**scale-free** band (`p < 0.05·q95`, non-zero but tiny) it is an untested assumption, and that band
is large — at N=2000 it is 76.4% versus 70.3% hard. No ablation experiment has been run. Do not
write the general sentence without either restricting it to hard-silent units or running the test.

⚠️ **Correction to the draft: K is not flat.** It grows monotonically — 99 → 369 → 492 → 593 → 732
across N = 100 → 5000 ✅. What is undecided is **bounded versus unbounded**: whether K saturates at a
ceiling (hyperbolic fit gives M\* ≈ 880, 95% upper bound 939) or continues as an unbounded power law
`K ∝ N^0.31`. Curvature in log-log is significantly negative (c = −0.118 ± 0.033, F(1,42) = 12.79,
p = 0.001) ✅ and the saturating fit beats the power law at every matched level under the hard
criterion, but the verdict flips under the scale-free criterion at shallow levels. The exponent falls
with N, and the marginal yield of active units per added unit is steep.

### The two regularizers, stated correctly

⚠️ **Correction to the draft: FRM is not `(mean activity − cap)²`.** The implemented form matters for
every mechanism claim below:

```python
activity = tau * (logsumexp(x/tau, dim=1) - log(T*B))    # tau = 0.1, dim=1 spans time x conditions
p_under  = (relu(cap - activity) / cap) ** g_bot          # g_bot = 3
p_over   = (relu(activity - cap) / cap) ** g_top          # g_top = 3
penalty  = (alpha * p_under + beta * p_over).mean()       # alpha = beta = 1
```

Three differences that carry weight:

1. **It acts on a soft maximum over time and conditions, not a mean.** At τ = 0.1 this is
   effectively the unit's *peak*. A unit satisfies the penalty with one large transient in one
   condition. This is the origin of the transient hypothesis below — it is a property of the
   objective, readable off the code, not an empirical surprise.
2. **It is two-sided.** The `beta`-weighted `p_over` term actively pushes *high*-firing units back
   down to the cap. That is a plausible mechanism for the σ_log collapse, and it is independent of
   the silence rescue.
3. **The exponent is 3, not 2**, and the cap scales as O(1/log N).

- **FRM** brings all units above threshold ✅ (silent fraction exactly 0 under both criteria, every
  architecture, through 200k iterations).
- 🟡 **FRM alone produces inert transients — active units with no causal role.** *This is a
  hypothesis, not a result.* It follows naturally from point 1 above, but it has not been measured.
  The evidence previously cited for the opposite conclusion (within-trial temporal CV 1.29 under FRM
  vs 1.43 unpenalized, recorded in `paper.md` §1.3 as showing the units are "genuinely modulated")
  does **not** discriminate: CV separates tonic from modulated, and a single large transient is the
  *highest*-CV signal a unit can produce. The discriminating statistic is peak-to-mean.
  Pre-registered analysis plan, with the falsifier fixed in advance:
  [`experiments/frm_rws_heterogeneity.md`](experiments/frm_rws_heterogeneity.md).
- 🟡 **FRM + RWS converts those units into ones with genuine computational roles.** Same status —
  this is the hypothesis under test in the same pre-registration, not an established result.
- ✅ **RWS alone barely changes anything** — but the mechanism is worth stating, because the naive
  reading is wrong. At 200k iterations and matched budget, N=2000: `rws` reads **60.5% hard-silent**
  against `none`'s 78.9%, which looks like a partial rescue, while under the scale-free criterion it
  reads **86.0%** against `none`'s 80.9% — *worse* than baseline. It lifts units just across the
  1e-6 line without making them active. The hard/scale-free gap grows with N (11.9 / 20.7 / 25.5 pp
  at N = 500 / 1000 / 2000) and is absent in every other condition.

🟡 Preliminary sweep (CDDM, N = 1000, 5 networks per cell):

| condition | PR (h) | PR (s) | n_active | σ_log(h) |
|---|---|---|---|---|
| none | 2.22 ± 0.05 | 3.03 ± 0.26 | 574 | 1.20 |
| rws | 2.39 ± 0.05 | 3.33 ± 0.06 | 600 | 1.27 |
| frm | 7.74 ± 0.15 | 7.06 ± 0.08 | 1000 | 0.26 |
| both | 6.14 ± 0.13 | 5.50 ± 0.07 | 1000 | 0.19 |

Unpenalized networks reach ~70% silent at 200k iterations, so the gaps above are likely conservative.
Note `PR` and `PR` over active units only agree to every decimal in all eight cells ✅ — silent units
contribute zero-variance eigenvalues and so change neither Σλ nor Σλ². PR is therefore immune to the
dilution artifact that inflates selectivity fractions. Inert *transients* are not harmless in the
same way, which is the point of Direction 2.

---

## Direction 1 — Recurrence breaks the known theory of dying ReLU

**Background.** The only theoretical treatment of dying ReLU is Lu, Shin, Su & Karniadakis (2019),
*Dying ReLU and Initialization*. It concerns **feedforward** networks and the worst case where the
entire network collapses to a constant function ("born dead"). Their scale is tiny: the running
example is approximating f(x) = |x| with a 10-layer, width-2 network; figures sweep width from 2 to
5. Three results matter here:

- Theorem 3.1: born-dead probability → 1 as depth → ∞.
- Theorem 3.3: at fixed depth, **born-dead probability → 0 as width → ∞**. They conclude that modern
  networks are wide enough for this not to be a practical concern.
- Theorem 3.7: once born dead, for any loss and any gradient-based optimizer, the network is
  optimized to a constant. Death is absorbing, because a dead unit receives exactly zero gradient.

**The contradiction.** Both load-bearing claims fail in the recurrent setting. Width does not
rescue: at N = 5000, 85.4% of units are hard-silent (87.0% scale-free) ✅. And death is not
absorbing ✅.

**The evidence that death is not absorbing is stronger than the draft claims, and comes from a
different place.** Stating it precisely, because the obvious version is architecture-dependent:

- ✅ **Spontaneous recovery without any penalty.** In standard unpenalized RNNs, **95.8 units per
  1000** are silent for ≥500 consecutive iterations and then recover. No penalty, no intervention —
  the absorbing-state property simply fails on its own. **This is the cleanest possible support for
  the proposed mechanism** and it should be the headline of this direction, not the penalty result.
- ✅ Under FRM in **Dale** networks, 369.4 units per 1000 endure a long silent episode and return —
  genuine large-scale resurrection.
- ⚠️ Under FRM in **standard** networks, FRM **prevents** rather than resurrects: only 0.6 units per
  network recover from a long silent episode, because units are caught within a few hundred
  iterations of dipping. So "hard-ReLU units revive when an activity penalty is applied" is true in
  Dale networks and misleading in standard ones. The unpenalized spontaneous-recovery number is what
  carries this argument in the main architecture.

**Proposed mechanism for revival.** The absorbing-state argument assumes a dead unit gets zero
gradient and therefore cannot change. In a recurrent network this is insufficient, because the
unit's presynaptic partners continue changing for their own reasons. Its pre-activation distribution
drifts, and it can cross threshold again with no gradient having flowed through it. Recurrence
supplies a channel feedforward architectures lack. The 95.8 spontaneous recoveries per 1000 are
exactly this: no penalty was applied, so nothing but partner drift can explain them.

**Specific question.** How does the number of active units K scale with network size N in
task-trained recurrent networks, and does the sublinearity survive variation in nonlinearity,
initialization, optimizer, and training budget?

**Setup.**
- K(N) sweep across N, with matched task performance ✅ (protocol established; see `paper.md` §6.1).
- Nonlinearity family: ReLU, softplus, sigmoid. Exclude tanh — no threshold, no silencing, wrong
  object. **Softplus is the critical control**: it cannot hard-die, so if sublinearity persists
  there, the phenomenon is not the classical dying-ReLU pathology. (Softplus β=25 and leaky-ReLU
  have been run in *Dale* networks only, 40–64% silent 🟡; the standard-RNN rerun is ⬜.)
- Initialization: symmetric (He) versus the randomized asymmetric initialization Lu et al. propose
  as their fix.
- Recurrent connectivity density sweep ⬜. The FRM/RWS dissociation implicates density: with
  unconstrained density a small core can route the computation and shadow the rest, while sparsity
  removes that option. Prediction is that silencing scales with density independently of
  nonlinearity. **Note this has only ever been checked at initialization** (0% silent at every
  spectral radius), never in trained networks.
- Revival experiment: track individual dead units through penalty onset, confirm zero gradient
  through the unit, and test whether revival also occurs in softplus (if yes, revival is not about
  crossing a hard threshold).

**Why it matters.** The existing theory tells the field that wide networks are safe. Wide recurrent
networks are not, and nearly all task-trained RNN neuroscience uses wide recurrent networks with
ReLU-family units.

---

## Direction 2 — Activity is not participation, and participation ratio inflates when units are inert

**Background.** Firing-rate regularization is standard practice for making trained RNNs
"biologically plausible." Dimensionality is standardly measured by participation ratio,
PR = (Σλ)² / Σλ², over the population activity covariance.

**Two contradictions.** Both currently rest on the 🟡 transient hypothesis; if that is falsified,
this direction dissolves and should be dropped rather than rescued.

*First:* FRM alone satisfies the activity statistic while producing transients with no causal role.
A network can pass every activity-based plausibility check while its added population is decorative.
An unknown fraction of published "biologically regularized" RNNs may be in this state. **The
objective-level argument for this is solid** — a soft-max-over-time penalty is provably satisfiable
by a single transient — **but that a trained network actually exploits the loophole is unmeasured.**

*Second:* PR is highest in exactly the condition where units are (hypothesised to be) inert. FRM
alone gives 7.74; FRM + RWS gives 6.14. If PR tracked computation the ordering would reverse. Silent
units are harmless to PR ✅ — zero-variance eigenvalues, which is why `pr` and `pr_active` agree to
every decimal. Inert *transients* are not harmless: they carry variance without function, and PR
counts variance.

**Specific question.** Does participation ratio overstate effective dimensionality when a population
contains active but causally inert units, and does correcting for this reverse the FRM-versus-FRM+RWS
ordering?

**Setup.**
- Classify units as load-bearing or not by ablation, per condition.
- Recompute the covariance and PR on the load-bearing subpopulation only.
- Compare corrected against uncorrected PR across all four conditions.
- Report the inert fraction per condition as a quantity in its own right.
- **Cheap precursor, no ablation needed:** the peak-to-mean / duty-cycle / condition-breadth metrics
  in [`experiments/frm_rws_heterogeneity.md`](experiments/frm_rws_heterogeneity.md) identify
  candidate inert units from the rate tensor alone. Run that first; it costs no compute and it
  decides whether the ablation study is worth doing.

**Interpretation.** If FRM's advantage disappears and FRM + RWS comes out ahead, PR is inflated by
causally inert variance and the corrected measure should be adopted. If FRM's advantage survives
ablation, the fluff/function account needs revising — which is also informative.

**Also worth tracking.** σ_log(h) collapses from 1.20 unpenalized to 0.26 under FRM ✅. Cortical
log-firing-rate distributions are famously broad. If penalized networks are more cortex-like in
dimensionality but less cortex-like in rate heterogeneity, that tension is a finding rather than a
nuisance, and it suggests no single regularizer buys biological realism wholesale. The `beta` term
identified above is the obvious knob and has never been swept ⬜.

---

## Direction 3 — Silencing and low dimensionality are one phenomenon

**Background.** Stringer, Pachitariu, Steinmetz, Carandini & Harris (2019) recorded ~10,000 mouse V1
neurons under natural image ensembles and found the stimulus-response eigenspectrum follows a power
law with exponent slightly above 1 — variance spread across very many dimensions with slow decay.
They also showed that for a d-dimensional stimulus ensemble the exponent must exceed 1 + 2/d or the
population code becomes non-smooth, so cortex sits near the highest dimensionality compatible with
smoothness.

Task-trained RNNs look nothing like this. PR values of 2 to 8 are typical. The gap is large and
unexplained.

**Competing hypotheses.**
- **H1, task poverty.** Lab tasks are trivial relative to natural vision. Dimensionality is bounded
  by task demands, so a low-dimensional RNN on a simple task is correct, not deficient.
- **H2, architecture.** RNNs are the wrong model class, and no amount of task complexity closes the
  gap.
- **H3, effective size.** The models are too small to be high-dimensional — not in N, but in the
  population actually participating.

H3 has been untestable in principle, because everyone reports N and nobody measures K. It survives
as an unexamined escape hatch: any low-dimensionality result can be waved away with "maybe the
network was effectively too small," and nobody can check.

**What the preliminary data show.** Moving K from 574 to 1000 with the task and architecture fixed
raised PR from 2.22 to 7.74 🟡. Task held constant, dimensionality nearly quadrupled. H3 has real
effect. But 7.74 is still orders of magnitude short of cortex, so H3 is a contributor, not the
explanation.

⚠️ **This attribution is confounded and the confound is not small.** FRM does not vary K alone — it
also compresses the rate distribution (σ_log 1.20 → 0.26), changes the dynamics, and imposes a cap
scaling as O(1/log N). Any of those could raise PR without K mattering at all. To attribute the PR
change to K specifically, K has to be varied by a route that does not also reshape the rate
distribution — e.g. comparing an unpenalized network of size N to an unpenalized network of size N′
chosen so that K(N′) matches the penalized network's K, or ablating a penalized network down to K
units and re-measuring. Until then, "moving K from 574 to 1000 raised PR" is a statement about
applying FRM, not about K.

**Specific question.** How much of the dimensionality gap between task-trained RNNs and cortex is
attributable to effective network size K, how much to task complexity, and how much remains for
architecture?

**Setup.**
- A 2D sweep: K (via regularization) crossed with task complexity, measuring dimensionality in every
  cell. The residual after accounting for K and task is the honest estimate of what the architecture
  hypothesis must explain.
- Task selection matters. n-bit flip-flop is n independent 1-bit flip-flops with known solution
  structure (2ⁿ hypercube corners, Sussillo & Barak) and dimensionality exactly n. If K scales with
  n, that is close to tautological — n subproblems need c·n units. The informative case is K scaling
  with 2ⁿ, meaning attractor count rather than dimensionality drives the requirement. Either way, at
  least one non-decomposable family is needed for interpretation: context-dependent integration with
  k contexts, or path integration in d dimensions.
- **Critical constraint on the cortical comparison.** Flip-flop cannot support a power-law exponent
  estimate. At n = 6 there are 64 attractor states — not enough to fit a slope. Matching Stringer's
  measurement requires many thousands of distinct inputs and a spectrum computed over a
  stimulus-response matrix, not over trials of a small task. Report PR and K for flip-flop; defer any
  exponent claim to a task with a rich input ensemble.

**Why it matters.** It converts a hand-wave into a number. "Our network is low-dimensional because
the task is simple" is currently asserted without evidence, and the K contribution has never been
measured because K has never been measured.

**Status note.** ⛔ The first flip-flop k-sweep (k = 2…6, 300k iterations) is **RETRACTED**: it ran
`same_batch=True`, so those networks saw 256 frozen trials for 300k iterations and measured
memorisation rather than the task. A corrected sweep (k = 2…8 × N = 500/1000/2000 × 3 seeds, fresh
batches of 1024, 400–500k iterations) is running as of 2026-08-21. It will supply the
task-complexity axis of this design for the flip-flop family. It does *not* supply the K axis —
those runs are all unpenalized.

---

## Gating requirement

The preliminary table mixes ~30k and 200k iteration budgets, and silent fraction rises substantially
with training (42.6% → 70.7% at N=1000 between those two budgets). Every ratio above is provisional
and effect sizes will move. **Rerun the sweep at a single reported budget before building any
argument on these numbers.**

Two further gates specific to the above:

1. **Direction 2 is gated on the transient hypothesis.** Run
   [`experiments/frm_rws_heterogeneity.md`](experiments/frm_rws_heterogeneity.md) first — it needs no
   compute, only the networks already on disk, and its falsifier is fixed in advance. If FRM and
   FRM+RWS are indistinguishable on peak-to-mean, Direction 2 has no premise.
2. **Direction 3's K-attribution needs a confound-free route to varying K** (see above), otherwise
   its central quantitative claim measures FRM rather than K.
