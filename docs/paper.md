# Paper plan — silent units in trained ReLU RNNs

Restructured 2026-09-10 (the previous structure is in git history before commit `a2541c2`'s
successor). Working document: the *argument*, what supports each claim, what is missing. The
experimental record is [`project_trajectory.md`](project_trajectory.md); what is proposed but not
done, and what is deliberately left out, is [`research_directions.md`](research_directions.md).

**Manuscript figures are vector.** `fig_paper_*.py` and `fig_supp_*.py` write PDF (the submission
asset) and SVG (so this document can show them inline) through `paperstyle.save`; they no longer
write PNG. The diagnostic figures made with `plotstyle.py` are still PNG — they are read on screen,
not printed.

**Every claim carries an evidence status** — ✅ measured, 🟡 preliminary, ⬜ planned — so that
interpretation never quietly becomes result. **Every main-text claim is shown on both tasks** (CDDM
and the n-bit flip-flop) or is explicitly marked as task-specific. Sign-constrained architectures
are supplementary throughout (§S1).

**Working title.** *Why trained RNNs leave most units silent, and what it takes to make every unit
count.*

**Working abstract (internal, ~450 words; 2026-09-11).**

*Motivation and problem.* Recurrent neural networks trained on cognitive tasks serve as model
organisms for systems neuroscience: digital circuits with known ground truth, on which analysis
methods are developed, tested and honed. Larger networks are trained to bring this testbed closer
to the scale of real circuits. However, a trained RNN with a non-negative activation leaves most of
its units silent: the number of active units scales between the cube root and the square root of
the total number of units, and task complexity changes it only marginally. As a result, a thousand
active units costs training a network of ten thousand. Most of the digital organism is dead weight,
and the population that remains is the small subset that training happened to keep.

*Characterization of the problem.* We show that silencing afflicts RNNs trained with any
non-negative activation function, and that it worsens with longer training and larger networks. We
tried many interventions to eliminate it, from architecture and noise to training length and the
standard regularizers. Most failed, and the majority of units persistently remained silent. The
reason they stay silent is that the task loss has no term that rewards keeping a unit active: early
in training the whole population is suppressed, only the units the solution needs recover, and
nothing in the objective ever pulls the rest back up.

*Solution.* The intervention that works is a convex penalty designed to keep each unit's
participation (a measure of how much the unit contributes to the network's activity) at a specific
level, so that every unit takes part in the computation. On both tasks (context-dependent decision
making and the k-bit flip-flop) it keeps every unit active at no cost in performance. We later found
that this penalty is easily taken advantage of: the cheapest way for a unit to stay active is to
receive weak input from the whole population, which satisfies the participation demand
superficially. Such units mix task variables and become less selective as the network grows, and on
the flip-flop task they respond only transiently, flickering in and out of silence.

*Characterization of the solution.* We characterize the trained networks along a continuum from
pure to mixed selectivity, together with their dimensionality and temporal structure. The
unpenalized network organizes the few units it keeps into highly selective ones, each following a
single task variable and wired to other units performing a similar computational role; the
participation penalty alone produces a population of mixed, diffusely connected units. A second
intervention, a penalty that encourages sparse connectivity, does nothing on its own to prevent
silence. On top of the participation penalty, however, it closes the loophole and moves the network
to the pure end of the continuum: every unit becomes selective for a single task variable, wired to
units performing the same computational role, and stays persistently active. Turning the sparsity
penalty on or off in a trained network makes the population more or less sharply selective,
reversibly. Networks with identical performance therefore differ several-fold in dimensionality,
selectivity, and whether units fire persistently or transiently.

*Takeaways.* First, two penalties solve the silent-unit problem afflicting large RNNs: the
participation penalty keeps every unit active at no cost in performance, on two tasks, and the
sparsity penalty keeps the rescued units stably active. Second, adding the sparsity penalty on the
connectivity makes the networks compute with highly selective units, each following one task
variable and connected to units that follow the same one. Third, the penalties shape the network's
solution in ways that really matter: dimensionality, selectivity and temporal structure are decided
by the training objective, and it should be chosen with that in mind.

---

## Elaboration of the abstract — every claim, one figure

Layer two of the document: abstract → this → the full paper below. **One statement, one figure**,
two panels where both tasks have the data (CDDM left, flip-flop right), no panel that does not serve
the statement. Each figure is made by one script from cached outputs of the analysis already on
disk. Status: ✅ figure built and checked; 🟡 measured but the code is not in the repo (numbers live
only in the trajectory); ⬜ figure or experiment still to make. A claim marked 🟡 is loose until its
script exists; the list at the end says what to build.

### Motivation and problem — one figure

**P. Trained RNNs keep few units active; the number grows as N^b with b between 1/3 and 1/2, is
almost independent of task complexity, and a thousand active units cost a network of ten
thousand.** ✅ [`fig_P_active_units.py`](../trainRNNbrain/experiments_and_analysis/fig_P_active_units.py) →
[`fig_P_active_units.png`](../img/internal_figures/fig_P_active_units.png). Active units M vs N, unpenalized ReLU, each
network read at a matched state (never end of training), under both silence criteria of each task,
with the fitted law, bootstrap confidence intervals on the exponents, and the extrapolation to
M = 1000 drawn on the panel. Left: CDDM, N = 500–5000 (N = 100 is excluded everywhere in the paper: it is fully active and exists
for one condition only). Right: k-bit flip-flop, k = 1–8, N = 500–4000. Top row at matched state (CDDM at matched performance; flip-flop at 1.10× each run's
own loss floor); bottom row at matched compute (iteration 100k). Four subclaims, all read off the
top row:

![fig_P_active_units.png](../img/internal_figures/fig_P_active_units.png)

| | CDDM | flip-flop (joint fit over k = 1–8) |
|---|---|---|
| **(a) most units silent** — points far below M = N | 297 of 500 → 684 of 5000 active (scale-free) | 262 of 500 → 654 of 4000 at k = 3 |
| **(b) M ∝ N^b, b between 1/3 and 1/2** | b = 0.36 [0.34, 0.38] scale-free; 0.31 [0.29, 0.33] hard | b = 0.40 [0.37, 0.44] scale-free; 0.37 [0.34, 0.40] absolute |
| **(c) k barely matters** | (one complexity level) | c = −0.06 [−0.10, −0.03] scale-free; −0.05 [−0.08, −0.02] absolute: 8× the bits → 12% *fewer* units |
| **(d) 1000 active units cost N ≈ 10⁴** | 1.4 × 10⁴ scale-free; 9.8 × 10³ hard | 1.1 × 10⁴ at k = 3, both criteria |

The bottom row reads every network at the same iteration, 100k (the largest budget every cell
reaches: CDDM N = 5000 and flip-flop N = 4000 both stopped there), so the two ways of comparing
sizes sit one above the other:

| read-out | CDDM b (scale-free / hard) | flip-flop b (scale-free / absolute) | flip-flop c | N for 1000 active, k = 3 |
|---|---|---|---|---|
| matched state (top) | 0.36 [0.34, 0.38] / 0.31 [0.29, 0.33] | 0.40 [0.37, 0.44] / 0.37 [0.34, 0.40] | −0.06 [−0.10, −0.03] / −0.05 [−0.08, −0.02] | 1.1 × 10⁴ (CDDM 1.0–1.4 × 10⁴) |
| iteration 100k (bottom) | 0.44 [0.42, 0.45] / 0.42 [0.40, 0.46] | 0.41 [0.38, 0.44] / 0.35 [0.33, 0.39] | +0.08 [−0.02, +0.16] / +0.09 [−0.01, +0.17] | 1.8–2.2 × 10⁴ (CDDM 1.1–1.4 × 10⁴) |

Notes. The criteria disagree on the count but not on the law; both are shown so neither is quoted
alone (§S4). The size exponent is 0.31–0.44 under every read-out and criterion — between 1/3 and 1/2
either way; end-of-training fits give 0.53–0.58 (§1.2) because large networks are then read earlier
in their silencing. On k: at fixed compute the sign flips to positive (c ≈ +0.08, CI touching 0 at
100k; +0.18 at 150k without N = 4000) and the difference is convergence depth — high-k networks
settle 2.6× slower; the participation ratio does rise with k; the N × k interaction is unresolved
(§1.3). The cost of 1000 active units is 10⁴ at matched state and 1–2 × 10⁴ at matched compute.
⚠️ (c) is flip-flop only. Subclaim (c) on its own: [`fig_P3_complexity.py`](../trainRNNbrain/experiments_and_analysis/fig_P3_complexity.py) → [`fig_P3_complexity.png`](../img/internal_figures/fig_P3_complexity.png) —
active units divided by the fitted size dependence, M/(A N^b), against k; c = +0.08 [−0.02, +0.16] at
matched compute and −0.06 [−0.10, −0.03] at matched state.

![fig_P3_complexity.png](../img/internal_figures/fig_P3_complexity.png)

### Characterization of the problem

**C1. Silencing afflicts any non-negative activation.** ✅
[`fig_C1_activations.py`](../trainRNNbrain/experiments_and_analysis/fig_C1_activations.py) → [`fig_C1_activations.png`](../img/internal_figures/fig_C1_activations.png). CDDM N = 1000, no penalty: 510 (ReLU) /
594 (softplus) / 548 (leaky-ReLU) active units of 1000 under the scale-free criterion; the hard
criterion gives 540 / 1000 / 675 — softplus has no hard-silent unit because its floor is soft, so
the scale-free criterion is the one that travels. Flip-flop k = 3, N = 1000: ReLU and a bounded
sigmoid(7.5(x − 0.3)) follow the same curve, 247–282 vs 226–252 active at the matched 150k
iteration, both still falling. The sigmoid was trained on the flip-flop (N = 500 and 1000, 3 seeds,
150k); on CDDM the three non-ReLU activations come from the 2026-07-01 sweeps on the earlier
architecture. ⬜ **E1 submitted 2026-09-11 13:39 (Spock 6163782, 9 jobs):** softplus, leaky-ReLU
and sigmoid on the standard CDDM network, N = 1000, 3 seeds, the drift-sweep recipe; the CDDM
panel is redrawn from it when it lands (~8 h).

![fig_C1_activations.png](../img/internal_figures/fig_C1_activations.png)

**C2. It worsens with longer training and larger networks.** ✅
[`fig_C2_training.py`](../trainRNNbrain/experiments_and_analysis/fig_C2_training.py) → [`fig_C2_training.png`](../img/internal_figures/fig_C2_training.png) for training time (active units vs iteration,
N = 1000, both tasks, still falling at the end of every run: CDDM 489–542 at 10k → 259–284 at 200k;
flip-flop 389–402 → 185–195 at 500k); for size, fig_P: the active count grows
only as N^0.3–0.45, so the silent count grows faster than the active one on both tasks.

![fig_C2_training.png](../img/internal_figures/fig_C2_training.png)

**C3. Many interventions failed.** ✅ Everything below was tried
on unpenalized networks and left most units silent:

- **trainable bias** (range [−1, 1], initialized at 0; the standard network's default): no change.
- **self-connections** (free, trainable diagonal): the network trains them *negative* — active units
  at −0.060, silent units at −0.007; corr(self-weight, log participation) = −0.51. No rescue.
- **activation**: softplus (β = 25), leaky-ReLU (leak 0.01), bounded sigmoid(7.5(x − 0.3)): the same
  active count as ReLU at the same size and iteration (C1).
- **recurrent noise** σ_rec ∈ {0, …}: never helps; σ = 0 is the worst.
- **longer training**: 30k → 300k iterations loses more units, monotonically (C2).
- **larger networks**: the active count grows as N^0.3–0.45, so the silent count grows faster (P).
- **metabolic cost** `mean(fr²)` over four decades of λ: never rescues; at λ = 10 it makes it worse
  (N = 100: 88 → 41 active units of 100).
- **input weight scale** (T5, running 2026-09-11): W_inp rows initialized at norm 0.5–20 instead of
  0.039. Logged twins at s = 5 and s = 1 reach the default active count by 50k iterations after a
  spike-driven collapse (trajectory 2026-09-11 entries); sweep read-out pending.
- **weight decay** 0 vs 10⁻⁶ (2026-07-28, N = 1000 and 2000, 3 seeds, 100k iterations, Della
  11706899): run, but its active-unit read-out was never recorded and only the loss files were
  synced to the laptop (task R² 0.85–0.88 in both arms). ⬜ Sync the traces and read them (T2).

✅ [`fig_C3_interventions.py`](../trainRNNbrain/experiments_and_analysis/fig_C3_interventions.py) → [`fig_C3_interventions.png`](../img/internal_figures/fig_C3_interventions.png):
active units at N = 1000 under each intervention, every bar labelled with its read-out iteration.
Left panel, standard network, participation criterion, all read at 30k iterations unless stated:
baseline (self-connections on, bias trainable) 414 ± 18; the same networks at 5k and 200k iterations
587 and 272; self-connections off with the bias trainable 404; self-connections off with the bias
fixed at 0 409 — so neither self-connections nor a trainable bias changes the count; metabolic cost
λ = 0.01 / 0.1 / 1 / 10: 393 / 431 / 432 / 380. Right panel, the 2026-07-01 sweeps at 30k on their
own peak-rate criterion with their own reference (not comparable to the left panel): recurrent
noise σ = 0 / 0.01 / 0.05 / 0.1: 184 / 521 / 524 / 540 (σ = 0 is the worst); ReLU / softplus /
leaky-ReLU 510 / 594 / 548. Nothing gets past ~600 of the 1000 units. The activation bars are
redrawn from E1 on the standard network when it lands.

![fig_C3_interventions.png](../img/internal_figures/fig_C3_interventions.png)

**C4. Why: the loss has no term that keeps a unit active; the population is suppressed early, only
needed units recover, nothing pulls the rest back.** ✅ (a) measured with a script now, (c) driver
still open. [`participation_recovery.py`](../trainRNNbrain/experiments_and_analysis/participation_recovery.py) → [`fig_C4_recovery.png`](../img/internal_figures/fig_C4_recovery.png): participation quantiles vs
iteration (N = 1000, both tasks) and, per network, the number of units that were silent for ≥ 500
iterations and are active at the end versus the number silent at the end. No penalty: CDDM 99 ± 4
recover and 728 ± 10 stay down; flip-flop 82 ± 6 recover, 793 ± 4 stay down — once a unit is down it
almost always stays down. Participation penalty: CDDM 0 silent at the end (2 ever endured 500 silent
iterations); flip-flop 23 ± 16 silent at the end, but 784 ± 18 units endured ≥ 500 silent iterations
and are active again — the flicker of S5, units repeatedly dipping below threshold and being pulled
back. ⚠️ These traces are stored every 100 iterations, so the ~20-iteration global collapse recorded
on 2026-07-25 from the every-10 sweeps is below this figure's resolution; the every-10 traces of
those sweeps are not on the laptop. (b) Activation-generality (C1) is what licenses "a property of
the objective, not the activation". (c) What *drives* units down is untested: weight decay — **E3
submitted 2026-09-11** (wd 0 / 10⁻⁵ / 10⁻⁴ vs the 10⁻⁶ baseline, N = 1000, 3 seeds) — and the
initialization scale (T5, running). The abstract says (a)+(b) only.

![fig_C4_recovery.png](../img/internal_figures/fig_C4_recovery.png)

**C5. Sparsity penalties do not rescue.** ✅ [`fig_C5_criteria.py`](../trainRNNbrain/experiments_and_analysis/fig_C5_criteria.py) → [`fig_C5_criteria.png`](../img/internal_figures/fig_C5_criteria.png):
active units, no penalty vs sparsity penalty, under each task's two criteria. CDDM (read at the
sparsity run's budget): under the hard criterion the sparsity penalty *raises* the count (790 vs 423
at N = 2000, 2086 vs 711 at N = 5000) and under the scale-free criterion it *lowers* it (279 vs 383;
440 vs 629) — the extra "active" units sit just above 10⁻⁶. Flip-flop (1.10× own floor): the CDDM
threshold 10⁻⁶ counts nearly every flip-flop unit as active (1995 of 2000 unpenalized), which is why
the task-calibrated 4·10⁻² threshold exists; under it and under the scale-free rule the sparsity
penalty gives a modest, real increase (708 vs 583 and 638 vs 551 at N = 2000) — the lifted size
exponent of the 2026-08-27 entry, not a rescue. §2.3, §S4.

![fig_C5_criteria.png](../img/internal_figures/fig_C5_criteria.png)

### Solution

**S1. The participation penalty keeps nearly every unit active on both tasks.** ✅
[`fig_cache_axis.py`](../trainRNNbrain/experiments_and_analysis/fig_cache_axis.py) → [`fig_S1_active.png`](../img/internal_figures/fig_S1_active.png) `active`. Active units vs N on log-log axes with
the M = N diagonal, four conditions, end of training. CDDM: every unit at every N (500 / 1000 /
2000 / 4996 of 5000). Flip-flop k = 3: 493 / 977 / 1728 of 500 / 1000 / 2000 with the participation
penalty alone, 500 / 999 / 2000 with both penalties. The no-penalty and sparsity-only lines sit a
factor 3–8 below the diagonal and diverge from it with N.

![fig_S1_active.png](../img/internal_figures/fig_S1_active.png)

**S2. At no cost in performance — TRUE ON CDDM, NOT ON THE FLIP-FLOP.** ✅ measured, ⚠️ the
abstract overstates it. [`fig_S2_cost.py`](../trainRNNbrain/experiments_and_analysis/fig_S2_cost.py) → [`fig_S2_cost.png`](../img/internal_figures/fig_S2_cost.png).
Task loss of the four conditions vs N, task term only, noise off (never `TrainLosses.json`).

![fig_S2_cost.png](../img/internal_figures/fig_S2_cost.png)

| | no penalty | sparsity only | participation only | both |
|---|---|---|---|---|
| CDDM, noise-free MSE of final weights, N = 2000 | 0.0089 ± 0.0009 | 0.0071 | 0.0076 | **0.0058** |
| CDDM, N = 5000 | 0.0085 | 0.0068 | **0.0053** | 0.0058 |
| flip-flop k = 3, fitted loss floor, N = 1000 | 0.0256 ± 0.0000 | 0.0260 | **0.0275** | **0.0305 ± 0.0024** |
| flip-flop, N = 2000 | 0.0257 | 0.0259 | **0.0291** | **0.0287** |

CDDM: no cost at any N and a gain at N ≥ 2000 (participation penalty 14% below the unpenalized
loss at N = 2000, both penalties 35% below). **Flip-flop: the participation penalty, alone or with
the sparsity penalty, converges to a loss floor 7–19% above the unpenalized one at every N**
(fitted over each run's own budget, 400k for the penalized runs, so this is not the 150k
non-convergence that was retracted on 2026-08-25). In R² terms it is about one point (folder-name
R² 0.953 unpenalized vs 0.941–0.948 penalized at N = 2000). The switch experiment's 0.027–0.030 in
every arm is the same penalized floor; it had no unpenalized arm, so it could not see this.
**The abstract's "at no cost in performance" must become "at no cost on CDDM and a ~1% R² cost on
the flip-flop"** — see the wording list. Unpenalized flip-flop networks that run only to their
own excess criterion, not 400k, would be the fairer comparison for the sparsity-only row; the
participation-penalty rows are converged and the gap is real.

**S3. The penalty is gamed by transients: the cheapest way to satisfy a soft-max activity floor is
to burst.** ✅ [`fig_S3_transients.py`](../trainRNNbrain/experiments_and_analysis/fig_S3_transients.py) → [`fig_S3_transients.png`](../img/internal_figures/fig_S3_transients.png). Per-unit temporal
participation ratio over the sample count (the fraction of time a unit is effectively active),
flip-flop k = 3, N = 2000, four conditions, 3 seeds. Panel (a): the distribution over live units; a
unit following one bit state sits at the task's duty cycle, 0.36. Panel (b): the median unit is at
the duty cycle in every condition except the participation penalty alone (0.24 ± 0.06). Panel (c),
the count that matters: **588 ± 80 burst units** (live but active less than 5% of the time) of 2000
with the participation penalty alone, against 46 / 35 / 93 for none / sparsity only / both. The
penalty is satisfied by bursts; the sparsity penalty added on top removes them, and 1999 of 2000
units are then live at the task's duty cycle. The wiring side of the loophole (in-degree ≈ 700 vs
≈ 20) is measured in CS1 (fig_CS1_wiring).

![fig_S3_transients.png](../img/internal_figures/fig_S3_transients.png)

**S4. Such units mix task variables — TRUE ON THE FLIP-FLOP; ON CDDM EVERY CONDITION MIXES, AND THE
SPARSITY PENALTY MAKES IT WORSE.** ✅ measured, ⚠️ the abstract overstates it. Three figures: the
picture on each task, then the count.

[`fig_S4_cloud.py`](../trainRNNbrain/experiments_and_analysis/fig_S4_cloud.py) → [`fig_S4_cloud.png`](../img/internal_figures/fig_S4_cloud.png): flip-flop, every live unit at its three regression
loadings on the three bits (the exact selectivity space at k = 3), N = 2000. No penalty and sparsity
only: a six-armed star of pure units (275 / 297 live). Participation penalty alone: 1634 live units
filling the space between the arms. Both penalties: the star, with all 2000 units on it.

![fig_S4_cloud.png](../img/internal_figures/fig_S4_cloud.png)

[`unit_stats.py`](../trainRNNbrain/experiments_and_analysis/unit_stats.py) → [`fig_S4_cloud_cddm.png`](../img/internal_figures/fig_S4_cloud_cddm.png) `cloud`: the mirror on CDDM, units at their (motion,
colour, choice) loadings, N = 2000. A few high-participation units sit on the axes in the
unpenalized network; the bulk is a core of mixed units in every condition, because the task is
context-dependent and a unit that reads one input alone cannot do it.

![fig_S4_cloud_cddm.png](../img/internal_figures/fig_S4_cloud_cddm.png)

[`unit_stats.py`](../trainRNNbrain/experiments_and_analysis/unit_stats.py) → [`fig_S4_mixed.png`](../img/internal_figures/fig_S4_mixed.png) `mixed`: the count. A unit is mixed-selective if it is
tuned (joint rectified regression R² ≥ 0.15) and the joint model beats the best single-variable
model by more than 0.1 R². Flip-flop, N = 2000: **322 ± 68** mixed units with the participation
penalty alone vs 7 / 3 / 102 for none / sparsity only / both (N = 500 → 2000: 42 → 71 → 322 alone,
0 → 18 → 102 with both). CDDM, N = 2000: none 225 (of 308 live: most unpenalized CDDM units are
mixed), sparsity only 188, participation alone 393, **both penalties 829 ± 104**, and at N = 5000
1087 alone vs 2288 with both. **On CDDM the sparsity penalty does not restore single-variable
selectivity; it doubles the number of mixed units.** The Hoyer selectivity curves
([`fig_S4_selectivity.png`](../img/internal_figures/fig_S4_selectivity.png), from `fig_cache_axis.py sel`) say the same
thing weakly (both ≈ none on CDDM) and are superseded by the count.

![fig_S4_mixed.png](../img/internal_figures/fig_S4_mixed.png)

[`unit_stats.py`](../trainRNNbrain/experiments_and_analysis/unit_stats.py) `tuning` → [`fig_S4_tuning.png`](../img/internal_figures/fig_S4_tuning.png): the two
per-unit quantities behind the counts, as distributions at N = 2000 (seeds pooled). **How they are
computed.** Each live unit's response is fitted by least squares on the task's rectified factors
plus a constant — flip-flop: relu(+b_j), relu(−b_j) for the three remembered bits, six factors,
time-resolved; CDDM: context, relu(±motion), relu(±colour), relu(±choice), seven factors, on the
decision-epoch mean per condition. Left: the R² of that fit per live unit. Right: the Hoyer sparsity
of the six (seven) coefficients per tuned unit (R² ≥ 0.15): 1 = one factor, 0 = all equal. The
medians in fig_S4_selectivity were these medians, which is why that figure looked flat; the
distributions do not.

| N = 2000 | none | sparsity only | participation only | both |
|---|---|---|---|---|
| flip-flop: median R² (live units) | 0.60 (282) | 0.58 (301) | **0.47** (1728) | **0.87** (1999) |
| flip-flop: median Hoyer (tuned units) | 0.83 (229) | 0.81 (254) | **0.62** (1149) | **0.89** (1855) |
| CDDM: median R² (live units) | 0.32 (307) | 0.36 (278) | 0.68 (2000) | 0.57 (2000) |
| CDDM: median Hoyer (tuned units) | 0.49 (260) | 0.41 (215) | **0.32** (834) | 0.44 (1383) |

Flip-flop: with the participation penalty alone a quarter of the live units are barely explained by
the bits at all (the R² spike at 0.1: the burst units) and the tuned ones spread across the whole
Hoyer range; with both penalties nearly every unit is explained (R² 0.87) and the Hoyer
distribution has a spike at 1.0 — one bit per unit. CDDM: the participation penalty alone gives a
unimodal Hoyer distribution around 0.3 (every unit mixes); with both penalties the distribution is
**bimodal** — a spike of pure units at 0.88 and a cluster of very mixed units at 0.15 — which is
how the mixed *count* doubles while the median barely moves. So on CDDM the sparsity penalty does
create single-factor units, but next to a larger population of more thoroughly mixed ones. (On
CDDM the tuned count is well below the live count under the participation penalty — 834 of 2000 —
because ~1000 live units have *no variance* across conditions in their decision-epoch mean: the
onset-burst units of fig_S5_examples fire the same pulse in every condition, so the fit is
undefined for them and they are neither tuned nor mixed.)

![fig_S4_tuning.png](../img/internal_figures/fig_S4_tuning.png)

Effective dimensionality as the actual participation ratio of the rate covariance (not a Hoyer
value) is in CS5, [`fig_CS5_dimensionality.png`](../img/internal_figures/fig_CS5_dimensionality.png): CDDM 2.1–2.4
unpenalized at every N vs 8.2 (participation alone) and 8.5 (both) at N = 2000; flip-flop 6.2 vs
8.9 and 5.5.

**Consequence for the abstract:** "every unit becomes selective for a single task variable" and
takeaway 2 hold on the flip-flop only. On CDDM the honest statement is that the penalties change
*which* mixed units the network builds, not whether it builds them. See the wording list.

**S5. They respond only transiently — the participation penalty makes burst units on BOTH tasks; the
sparsity penalty removes them on the flip-flop and NOT on CDDM.** ✅
[`unit_stats.py`](../trainRNNbrain/experiments_and_analysis/unit_stats.py) → [`fig_S5_bursts.png`](../img/internal_figures/fig_S5_bursts.png) `bursts`: number of live units active less than 5% of
the time (temporal PR / n < 0.05) vs N. Flip-flop: 22 → 248 → **591 ± 78** burst units (N = 500 →
2000) with the participation penalty alone, 13 → 50 → 94 with both, 31–46 unpenalized. CDDM: 67 →
285 → **811** → 2618 (N = 500 → 5000) alone, and 108 → 233 → **787** → 1103 with both — the sparsity
penalty does not remove bursts on CDDM at N ≤ 2000. This replaces the Hoyer temporal-sparsity curves
([`fig_S5_temporal.png`](../img/internal_figures/fig_S5_temporal.png)), which barely separated the conditions.

![fig_S5_bursts.png](../img/internal_figures/fig_S5_bursts.png)

[`unit_stats.py`](../trainRNNbrain/experiments_and_analysis/unit_stats.py) → [`fig_S5_examples.png`](../img/internal_figures/fig_S5_examples.png) `examples`: what the two kinds of unit look like — the
four lowest- and four highest-tPR live units of a participation-penalty network on each task, six
trials each.

![fig_S5_examples.png](../img/internal_figures/fig_S5_examples.png)

The distribution behind the flip-flop numbers is fig_S3. The flicker statistics (~200 of 2000
units dead at any moment, not the same 200; 2.9 dead↔alive crossings per unit) are in fig CS4's
trajectories. ⚠️ The abstract's "on the flip-flop task they respond only transiently" is right
about where the *fix* works, but the transients themselves appear on both tasks.

### Characterization of the solution

**CS1. The unpenalized network's survivors are highly selective and wired to units of the same
role; the participation penalty alone gives mixed, diffusely connected units.** ✅ Selectivity:
fig_S4_tuning and fig_S4_mixed. Wiring: [`wiring_structure.py`](../trainRNNbrain/experiments_and_analysis/wiring_structure.py) → [`fig_CS1_wiring.png`](../img/internal_figures/fig_CS1_wiring.png), N = 2000,
3 seeds, both tasks. Effective in-degree per tuned unit (median): flip-flop 80 / 20 / **726** / 20
for none / sparsity / participation / both; CDDM 236 / 20 / **577** / 20. Same-role share of a
tuned unit's input weight over chance (full row): flip-flop 3.1× / 3.6× / **1.6×** / 3.8×; CDDM
0.93× / 1.1× / 1.2× / **2.1×**. Matched-n (275 live units) wiring modularity above its row-shuffle
null: flip-flop 0.33 / 0.39 / **0.13** / 0.13; CDDM 0.01 / 0.10 / 0.09 / 0.19; activity modularity
above its null 0.28–0.38 in every condition on both tasks; ARI(wiring, activity) flip-flop 0.61 /
0.67 / **0.35** / 0.57, CDDM 0.57 / 0.52 / 0.63 / 0.47. **Flip-flop:** the unpenalized survivors
wire by role (3.1× chance) and the participation penalty alone dissolves that (1.6×, in-degree
726, wiring modularity a third of the others) while activity stays as modular as anyone's.
**CDDM:** the unpenalized survivors do *not* wire by role (0.93×, wiring modularity at null) — the
task's natural solution is mixed and there is no assembly structure to lose; both penalties
*create* same-role wiring (2.1×) in the subpopulation of pure units that fig_S4_tuning shows.

![fig_CS1_wiring.png](../img/internal_figures/fig_CS1_wiring.png)

Five wiring statistics of different construction, matched n = 275, N = 2000, 3 seeds (Q values are
excess over each statistic's own null):

| | ingredients | flip-flop none / rws / frm / both | CDDM none / rws / frm / both |
|---|---|---|---|
| spectral wiring modularity | clustering, no labels | 0.33 / 0.40 / **0.15** / 0.16 | 0.01 / 0.10 / 0.10 / 0.19 |
| task-role partition modularity | labels, no clustering | 0.33 / 0.44 / **0.12** / **0.49** | 0.00 / 0.08 / 0.04 / 0.14 |
| like-to-like (weight vs activity correlation, per pair) | neither | 0.53 / 0.45 / **0.17** / 0.33 | 0.09 / 0.16 / **0.61** / **0.58** |
| same-role share over chance | labels, per unit | 3.1 / 3.7 / **1.5** / 3.8 | 0.9 / 1.1 / 1.2 / **2.1** |
| ARI(wiring, activity partitions) | two partitions | 0.63 / 0.64 / **0.32** / 0.55 | 0.61 / 0.51 / 0.67 / 0.48 |

On the flip-flop all five agree that the participation penalty alone is the outlier and that the
sparsity penalty alone is at or above the unpenalized network; for `both` the label-based
instruments read above `none` and the label-free ones below (see the retracted-claims note). On
CDDM the like-to-like statistic is the surprise: the penalized networks wire strongly by activity
correlation (0.61, 0.58) with almost no task-role structure — the diffuse `frm` wiring is not random,
it follows correlation without following the task's factors.



**CS2. The sparsity penalty alone does nothing to prevent silence.** ✅ fig_S1: the sparsity-penalty
line sits on the unpenalized line on both tasks (0.14 vs 0.15 at N = 2000).

**CS3. On top of the participation penalty the sparsity penalty makes every unit selective for one
variable, wired to units of the same role, and persistently active — FLIP-FLOP ONLY.** Selectivity
✅ fig_S4_mixed: mixed units 322 → 102 at N = 2000 when the sparsity penalty is added (flip-flop);
on CDDM 393 → 829, the opposite. Persistence ✅ fig_S5_bursts: burst units 591 → 94 (flip-flop);
811 → 787 on CDDM, no change. Wiring ✅ fig_CS1_wiring: same-role share 3.8× chance with both penalties vs 1.6× alone (flip-flop); on CDDM 2.1× vs 1.2×, the one CS3 statistic that does move on CDDM. **Every part of CS3 is a flip-flop result; on CDDM
the sparsity penalty changes neither the burst count nor the mixing, and the abstract's
"every unit becomes selective for a single task variable" and takeaway 2 must say so.**

**CS4. Turning the sparsity penalty on or off in a trained network moves the population back and
forth, reversibly — persistence AND selectivity.** ✅ Four arms warm-started from converged 400k
networks (flip-flop, N = 2000, k = 3, 3 paired seeds). Persistence, from the traces:
[`flipflop_switch_summary.py`](../trainRNNbrain/experiments_and_analysis/flipflop_switch_summary.py) → [`switch_summary.png`](../img/internal_figures/switch_summary.png):
temporal PR 0.157 → 0.353 (penalty added) and 0.353 → 0.228 (removed), dead units 212 → 0 of 2000
and 0 → 176, controls inert. Selectivity, now measured at the parent and the endpoint of every arm:
[`flipflop_switch_selectivity.py`](../trainRNNbrain/experiments_and_analysis/flipflop_switch_selectivity.py) → [`fig_CS4_selectivity.png`](../img/internal_figures/fig_CS4_selectivity.png):

| arm | mixed-selective units | burst units | median Hoyer of tuning coefficients |
|---|---|---|---|
| A1 participation → + sparsity | 319 → **90** | 589 → **108** | 0.64 → **0.84** |
| A2 participation → participation (control) | 319 → 276 | 589 → 643 | 0.64 → 0.71 |
| A3 both → participation (sparsity removed) | 101 → **385** | 91 → **390** | 0.90 → **0.69** |
| A4 both → both (control) | 101 → 95 | 91 → 92 | 0.90 → 0.90 |

Adding the sparsity penalty removes three quarters of the mixed and burst units and sharpens the
tuning; removing it brings them back; the controls move little. The pre-registered molding
hypothesis failed (identity ρ = 0.26); the supported claim is that the state is set by whichever
penalty is active. ⚠️ Flip-flop only; **A6 (the same four arms on CDDM) submitted 2026-09-11.**

![switch_summary.png](../img/internal_figures/switch_summary.png)

![fig_CS4_selectivity.png](../img/internal_figures/fig_CS4_selectivity.png)

**CS5. Networks with identical performance differ several-fold in dimensionality, selectivity and
temporal structure.** ✅ [`fig_cache_axis.py`](../trainRNNbrain/experiments_and_analysis/fig_cache_axis.py) `d_pr` → [`fig_CS5_dimensionality.png`](../img/internal_figures/fig_CS5_dimensionality.png): D_PR vs N.
CDDM 2.1–2.4 unpenalized at every N vs 5.0 → 10.3 under the participation penalty (both penalties
4.7 → 8.5, then 6.8 at N = 5000); flip-flop 6.1–6.3 vs 5.6 → 8.9 alone and 5.0–5.5 with both.
Selectivity and temporal structure: fig_S4, fig_S5. The matched-performance CDDM comparison with
metabolic cost and rate heterogeneity (σ_log 1.2 → 0.26, the limitation) is
[`population_distortion.py`](../trainRNNbrain/experiments_and_analysis/population_distortion.py) → [`population_distortion.png`](../img/internal_figures/population_distortion.png).

![fig_CS5_dimensionality.png](../img/internal_figures/fig_CS5_dimensionality.png)

![population_distortion.png](../img/internal_figures/population_distortion.png)

### What to build — status after the autonomous round of 2026-09-11

Done this round (all one script, one statement, from data on disk):

- **F1** P(c) → `fig_P3_complexity.py`. **F3** C4 → `participation_recovery.py`. **F4** CS1/CS3 →
  `wiring_structure.py`. **F5** C5 → `fig_C5_criteria.py`.
- **S1** `wiring_structure.py` (in-degree, same-role share vs chance, matched-n wiring/activity
  modularity against their own nulls, ARI; both tasks). **S2** `participation_recovery.py`. **S3**
  `flipflop_switch_selectivity.py` (mixed / burst / Hoyer at parent and endpoint of every arm).
  **S4** superseded: the joint-vs-single statistic lives in `unit_stats.py`; the self-connection
  result is the 2026-07-28 within-network test (no script survives; the cross-sweep numbers are
  in fig_C3).
- **E2 = A5** closed by `fig_S2_cost.py` (flip-flop floors per condition).

Running on Spock (2026-09-11):

- **E1** activations on the standard CDDM network — array 6163782, 9 jobs (C1's CDDM panel, C3's
  activation bars).
- **E3 = T2** weight decay 0 / 10⁻⁵ / 10⁻⁴ vs the 10⁻⁶ baseline, N = 1000, 3 seeds —
  `slurm/SilentReLU_std_weightdecay_spock.slurm`, array 6163899 (C4c).
- **A6** the four switch arms on CDDM, N = 2000, 50k iterations from the 150k penalty nets —
  `slurm/SilentReLU_cddm_switch_spock.slurm`, array 6163900 (CS3/CS4 as two-task claims).
- **T5** input-scale sweep (C3, C4c).

Still open:

- **A3** is done as far as wiring goes (fig_CS1 has CDDM); the CDDM *switch* is A6 above.
- The every-10 participation traces of the 2026-07-25 sweeps (the ~20-iteration collapse) are not
  on the laptop; C4(a) cites the record for that number.
- Read-outs to write when the runs land: C1/C3 activation panels from E1; a weight-decay panel for
  C4 from E3; a CDDM version of `flipflop_switch_selectivity.py` for A6.

**Reporting rule (Pavel, 2026-09-11): never a silent fraction — always the number of active units.**
Wording to settle in the abstract once the above is done: **"at no cost in performance" → "at no cost
on CDDM; on the flip-flop the loss floor is 7–19% higher, about one point of R²" (S2, measured
2026-09-11, both in the solution paragraph and in takeaway 1); "every unit becomes selective for a
single task variable" and takeaway 2 → flip-flop only — on CDDM every condition builds
mixed-selective units and the sparsity penalty doubles their number (S4, CS3, measured 2026-09-11);
"on the flip-flop task they respond only transiently" → the transients appear on both tasks, the
sparsity penalty removes them on the flip-flop only (S5);** "every unit" → "nearly every unit" (S1);
"less selective as the network grows" → flip-flop, or "less selective than any other condition"
(S4); "persistently or transiently" → flip-flop (S5, CS3); "more or less sharply selective" →
measured or reworded (CS4).

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

### 1.3 Task demand recruits far fewer units than size does ✅ (flip-flop)

If silence were the network idling on an easy task, the active count should track task demand. Over
k = 1…8 bits crossed with N = 500…4000, fitting `M = A·N^b·k^c`, the answer depends on how the
networks are compared, and the paper must say so:

| comparison | c (k exponent) | M(k=8)/M(k=1) |
|---|---|---|
| fixed compute (150k iterations) | +0.18 [0.15, 0.21] | 1.45× |
| matched performance / matched weight dynamics (five criteria); excess criterion on the full grid incl. N = 4000: −0.06 [−0.10, −0.03] | −0.06 … +0.02 | 0.88–0.95× |
| under `rws`, matched dynamics | +0.08 [0.04, 0.12] | 1.19× |

The fixed-compute effect is a convergence artifact: high-k networks settle 2.6× slower
(T ∝ k^0.46) and less-converged networks carry more active units; reading at a k-compensated
iteration removes it entirely (c 0.18 → 0.00). At matched state, an eight-fold increase in memory
demand changes the count by −12% to −5% (CI −15% to +4%), against ~2.5× for an eight-fold increase in N
(b = 0.44). Two refinements keep this from being overstated: (i) the participation ratio *does*
rise with k (c ≈ 0.05–0.12, CI excludes 0 under all six criteria) — the same units are used more
evenly, and the activity's dimensionality rises ≈ 1.8k — and (ii) the N × k interaction under
matched loss (+7% at N = 500, −15% at N = 2000) is unresolved at three seeds per cell. *More task
dimensions, marginally more units, far fewer than size buys.* The number of units a trained RNN
recruits is set mainly by its size and only weakly by what it is asked to do.

---

## 2. Why: nothing in the loss keeps a unit alive

### 2.1 The mechanism ✅ activation-general, ⬜ one direct test proposed

The task loss has no term that keeps any particular unit active. A unit the solution does not need
receives no gradient that holds it up; weight decay and noise walk it down to the activation's
floor — or a transient instability drops a few hundred units there at once (§2.2) — and it stays
there because the loss does not care and the gradient at the floor is small (zero for ReLU, ~0.01
for leaky-ReLU, ~0.02 for softplus and the sigmoid at its lower asymptote).
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

**Nor is it the input initialization** ✅ (flip-flop). W_inp is drawn at std 1/√N, far below the
scale the task needs (trained live units end with input rows at norm ~5.4 from 0.04–0.06). Giving
every unit an input row of norm 0.5, 2, 5 or 20 at initialization (N = 500 and 1000, 3 seeds each)
removes the collapse in the first iterations but not the silence: at 150k iterations 53–57% of
units are silent at N = 500 and 66–70% at N = 1000, against 62% and 74% for the default draw at the
same iteration, with no dependence on the scale from 0.5 to 20 and the same task performance. The
network converges to the same input weights from any start (‖W_inp‖_F = 93–95 in every condition;
silent units' rows decay to 0.004 whether they started at 0.05 or 20). What the scale changes is
*when* the units go: **in every run, most of the silencing happens in one to three discrete
events, a rise of 10–50% of the population within a thousand iterations coincident with a transient
loss excursion**, at 2–3k iterations for the default draw and anywhere from 5k to 75k for the
scaled ones. Silence is not only a slow walk; it is also an avalanche, and the unpenalized network
takes both routes from any initialization.

Sixteen interventions, condensed. Input initialization scale (above): removes the early collapse,
not the silence. Architecture (`h`/`s` equation, cubic term, boundary handling,
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

## S0. Supplementary: the three tasks ✅

The task definitions, moved out of Figure 1. Silence is not a property of any one task, so putting a
trial structure in the motivation figure made it read as though it were; and a reader cannot learn
CDDM, the flip-flop and DMTS from three grey steps overlaid on a trace. One row per task:
a pictogram of what the trial asks, and beside it the actual input and target channels, generated
by the task objects in `trainRNNbrain/tasks/` from the same `configs/task/*.yaml` the training runs
used, so every epoch time on the panel can be checked against the config.
[`fig_supp_tasks.py`](../trainRNNbrain/experiments_and_analysis/fig_supp_tasks.py) →
[`fig_supp_tasks.pdf`](../img/internal_figures/fig_supp_tasks.pdf)
([svg](../img/internal_figures/fig_supp_tasks.svg)).

![fig_supp_tasks.svg](../img/internal_figures/fig_supp_tasks.svg)

CDDM: 6 inputs (2 context cues, 2 motion, 2 colour) → 2 outputs, 300 steps (30 τ), shown on a
conflict trial. k-bit flip-flop: k pulse channels → k held-state channels, 300 steps, Poisson pulse
times so a trial has no epochs. DMTS: 2 stimulus channels + a go cue → 1 match output, 140 steps,
shown on a match trial, stimulus times jittered by ±10 steps.

⚠️ Numbered S0 so that S1–S7 keep the numbers they are cited by elsewhere in this document; renumber
the whole block at submission.

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

Sixteen rows (§2.2; row 16 = W_inp initialization scale, 9–360× the default, flip-flop: 53–70%
silent at 150k vs 62–74% default, flat in the scale), with the self-inhibition result (self-weights trained negative, active units at
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
- "`both` is the most task-aligned / most modular" — unmatched-n artifact. Re-measured 2026-09-11
  with five wiring statistics of different construction, matched n = 275, flip-flop N = 2000
  (`wiring_structure.py`): the answer depends on whether the instrument uses the task labels.
  Label-based: modularity of the task-role partition above its null 0.33 (`none`) vs **0.49**
  (`both`); same-role share over chance 3.1× vs 3.8×. Label-free: spectral wiring modularity above
  its row-shuffle null 0.33 vs **0.16**; like-to-like correlation between a pair's weight and its
  activity correlation 0.53 vs **0.33**; ARI(wiring, activity) 0.63 vs 0.55. What every instrument
  agrees on: `frm` is the outlier (0.12 / 1.5× / 0.15 / 0.17 / 0.32), `rws` alone is at or above
  `none` on all five. So `both` is *more* organized by task role than the unpenalized network and
  *less* organized by the label-free measures, both of which are sensitive to its sparsity (S ≈ 20
  rows leave most pair weights at zero). The defensible sentence: `both` extends role-organized
  wiring to all 2000 units; it is not "more modular" in any label-free sense.
- "`rws` molds transient units into sustained ones" — pre-registered, both criteria failed (identity
  ρ = 0.26 < 0.5; corr(Δ, start) no better than its null). What it does instead, measured in the
  switch (2026-09-10): it **stabilizes membership of the active set** (dead↔alive transitions per
  unit per 250 iterations 2.94 under the participation penalty alone → 0.30 with the sparsity
  penalty; units crossing ≥ 4 times 38.7% → 1.7%; the ~10% dead under the participation penalty is
  a dynamic equilibrium of different units, not a fixed set), it **levels and pins each unit's
  temporal occupancy** (median tPR/n 0.157 → 0.353, IQR 0.330 → 0.179, the mode at the task's duty
  cycle), and it **lets the weights settle** (W_inp settles at ~55k iterations with `rws`, never
  under `frm` alone; 2026-08-26). All three are dynamically maintained — remove `rws` and they
  revert (A3) — so it is a state held by the active penalty, not a structural change.
- "`frm` has an untuned quarter" — misspecified (signed) regression basis.
- **"Silence is caused by the ReLU scale symmetry `relu(a·x) = a·relu(x)`"** — a bounded sigmoid,
  which has no such symmetry, silences the same fraction of units at the same size and iteration
  (0.75–0.77 vs 0.72–0.75 at N = 1000, flip-flop). The cause is the objective's missing term, not
  the activation's homogeneity. Removed from every section on 2026-09-11.
