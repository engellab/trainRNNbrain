# Dormant ReLU units — talk track

One claim per slide, one panel per slide. Every figure is written by
`trainRNNbrain/experiments_and_analysis/fig_slides.py`, which calls the manuscript's own panel
functions and loaders — if a number here disagrees with the paper, that is a bug, not a second
opinion.

`⚠` marks a slide whose figure does not exist yet.

Every figure is centred at one fixed width (760 px) so the deck reads at a constant scale — add new
ones as `<p align="center"><img src="..." width="760"></p>`, not as Markdown image syntax, which
cannot be centred.

---

## THE PROBLEM

### 1. Most units of a trained ReLU RNN never fire
<p align="center"><img src="../img/internal_figures/slide_01_schematic.svg" width="760"></p>

### 2. It is not a threshold artefact — the distribution is bimodal, on every task
<p align="center"><img src="../img/internal_figures/slide_02_participation_by_task_matched.svg" width="760"></p>
One network per task at N = 1000, all three read at the **same 40,000 iterations**. Active: 383,
269, 276 — the three tasks look alike at a matched budget.

### 2b. …and they diverge as training continues
<p align="center"><img src="../img/internal_figures/slide_02_participation_by_task.svg" width="760"></p>
The same three networks at the end of their own budgets. Active: 318, 269, 175. CDDM loses a further
65 units between 40k and 100k, DMTS a further 101 between 40k and 150k — the flip-flop panel is
unchanged because 40k is where it ends.

p_i = std(r_i) + q₀.₉(|r_i|). Active when p_i ≥ 0.05·q₀.₉₅(p) — relative to each network, which is
why the dashed line moves between panels. Shared x, separate count axes.

### 3. Units keep going silent long after the loss has stopped moving
<p align="center"><img src="../img/internal_figures/slide_03_silencing_vs_training.svg" width="760"></p>
Three tasks, every seed. DMTS does not follow the other two — shown, not hidden.

Vanilla networks — no dropout, no penalty, no augmentation. Kept: every run whose training loss was
recorded; nothing else filtered, no seed averaged away, no unsolved seed removed. Grey is the raw
loss, black a running median over y[i−h … i+h], h = min(200, ⌊0.03·(i+1)⌋). The median removes 99.6%
of the step-to-step wiggle, which is why the raw is drawn under it. Loss normalised by its own first
value. The dashed rule is where the **raw** loss first comes within 7% of its final value.

The coloured silent-unit curve is **not smoothed at all** — it is the raw count, criterion from
slide 2, at every 100-iteration probe. It is simply that quiet: the median change between probes is
1–3 units. The occasional jumps of 200–300 are the relative criterion moving, not units switching
off together — when overall activity dips, 0.05·q₀.₉₅(p) falls with it and many units cross at once.

---

## WHY ITERATION COUNT IS THE WRONG CLOCK

### 4. The parameters never stop moving — all three tasks
<p align="center"><img src="../img/internal_figures/slide_04_drift_trajectories.svg" width="760"></p>
‖W(t) − W(t−L)‖_F / ‖W(t)‖_F at L = 10,000, bias excluded, every seed. Still 10–100% of the weights'
own magnitude at 140,000 iterations.

### 5. A single iteration count cannot serve every condition
<p align="center"><img src="../img/internal_figures/slide_05_readout_time.svg" width="760"></p>
Iterations to reach 1.07× that run's **own** fitted floor, two tasks, every run drawn. Within a task
size barely moves it: the flip-flop goes 35k → 39k → 37k → 43k over an 8× range in N, CDDM 15k → 16k
→ 18k → 19k over a 10× range. Across tasks it does move: at N = 1000 CDDM reaches its floor at 16k
and the flip-flop at 39k, so the same rule lands at times differing by a factor of 2.4 on networks of
the same size. That is the argument for not fixing an iteration count.

CDDM is the cleaner of the two: its read-out rises monotonically with N, where the flip-flop's N=2000
sits below its N=1000 and the within-size scatter overlaps. The pooled fit over the whole unpenalised
flip-flop grid gives T ∝ N^0.143 [0.065, 0.244], which is 1.35× over an 8× size range.

Each panel prints its per-size budget because they are not equal, and each floor is fitted over its
own run's whole trace: a longer trace pins the floor down better and so crosses slightly later.

Six of the 96 unpenalised flip-flop runs never come within 7% of their own fitted floor and are
absent — at the looser 10% it was four. A run goes missing when its fitted floor sits a little below
what it actually reaches, so the threshold falls under its whole loss curve.

### 6. So: read every network where its own loss stops falling
<p align="center"><img src="../img/internal_figures/slide_06_readout_rule.svg" width="760"></p>
Four sizes per task, both tasks, every run drawn. Each size has its own fitted floor (dotted) and its
own crossing of 1.07× it (dashed, dot); the read-out iteration is in the legend. That iteration is
the read-out — not a number fixed in advance. Every count in this talk is taken there.

The four markers sit almost on top of each other because on both tasks the four sizes reach floors
within 2% of one another. The floors being that close is the point: what separates the networks is
when they get there, not where they stop. Drawn from iteration 100, since CDDM's loss is logged from
iteration 0 where an untrained network sits near 10³.

---

## THE SCALING

### 7. Active units grow as N^0.32–0.45 — so the fraction falls
<p align="center"><img src="../img/internal_figures/slide_06_scaling.svg" width="760"></p>
Both silence criteria, both task families. Fitted exponents: 3-bit flip-flop 0.45, 6-bit flip-flop
0.32, CDDM 0.43, DMTS 0.37. Each network is read a matched number of iterations after it crossed its
task's convergence bar, not at a fixed iteration.

---

## WHAT DOES NOT WORK — one knob at a time

### 8. A different activation does not help
<p align="center"><img src="../img/internal_figures/slide_x_activation_cddm.svg" width="760"></p>
No activation raises the count, but two of the three lower it: of 1000 units, ReLU holds 272 active
(259/272/284 across seeds), leaky ReLU 278, softplus 249 and sigmoid 205. Leaky ReLU sits inside the
reference's own seed spread; softplus's three seeds all fall below the reference's lowest.

### 8b. And the units they shed were not doing anything
<p align="center"><img src="../img/internal_figures/slide_x_activation_cddm_r2.svg" width="760"></p>
The same 12 networks. Across a 191–284 range of active units every one sits within 0.016 of
R² = 0.950: ReLU 0.951, leaky ReLU 0.956, softplus 0.948, sigmoid 0.949. Sigmoid gives up a quarter
of its live units and pays 0.002 of R² for it, which is less than the measurement's own noise.

R² here is **noise-free**, and on this panel that choice is not cosmetic. The score in the folder
name comes from a forward pass with the noise **on**, and that penalty is activation-dependent:
+0.063 for leaky ReLU, +0.062 for softplus, but +0.074 for sigmoid. On the folder score sigmoid is
the worst arm (0.869 against ReLU's 0.874); noise-free it is level with the rest. A bounded sigmoid
does not absorb injected noise the way a ReLU net does, so the ranking flips with the choice.

The ReLU reference predates the Trainer's clean-loss probe, so all four arms are re-scored offline
instead — rebuilt from their saved parameters and run on a noise-free batch, one route for every
arm. On the three arms that do carry the probe, each offline number lands inside the range that
probe spans over the last 10,000 iterations. That range is wide (per-probe standard deviation
0.004–0.018) because the weights are still moving at 200k, which is slide 4's point: a single probe
is a sample of a drifting network, not a fixed property.

### 9. Nor on the other task
<p align="center"><img src="../img/internal_figures/slide_x_activation_ff.svg" width="760"></p>
ReLU 263 active of 1000 (282/261/247), sigmoid 240 — the seed ranges overlap, so this one is inside
scatter. ⏳ Softplus and leaky ReLU are training at 150k (Della, 2 sizes × 3 seeds each) so this panel
carries the same three activations as slide 8; until they land it rests on sigmoid alone.

### 9b. Nor does it cost anything here
<p align="center"><img src="../img/internal_figures/slide_x_activation_ff_r2.svg" width="760"></p>
Six networks so far, both axes read at 150,000 iterations, spanning 226–282 active units within
0.005 of R² = 0.967. Sigmoid holds 23 fewer units than ReLU and scores slightly **higher** (0.969
against 0.965). The flip-flop's clean probe is much quieter than CDDM's — per-probe standard
deviation 0.0004–0.003 near the read-out — so single-probe values are used directly here, as on
slide 11b. ⏳ The leaky-ReLU and softplus arms join this panel when they finish.

### 10. Weight decay makes it monotonically worse
<p align="center"><img src="../img/internal_figures/slide_x_weightdecay.svg" width="760"></p>

### 10b. …and the task does not notice the units it takes
<p align="center"><img src="../img/internal_figures/slide_x_weightdecay_r2.svg" width="760"></p>
The same 12 networks, R² against active units. Across the ladder the mean active count falls 395 →
272 → 171 → 84, a 4.7-fold drop, while R² sits at 0.950, 0.951, 0.953, 0.940. Three quarters of the
units go and the task is scored no worse.

The one rung that does cost something is the strongest, λ = 10⁻⁴: 0.940 against the default's 0.951,
a gap of 0.011 against a standard error of 0.004 on that cell. Small, but it is the only rung where
the sign is consistent across all three seeds, so it is a real cost and not scatter.

R² is the **noise-free** re-score of each net's final parameters, simulated offline — not the score
in the folder name, which is a single forward pass with the noise on. Active units are read at
199,900 iterations under the same scale-free rule as every other panel.

### 11. A bigger input scale adds 40–75 units of 1000, and not monotonically
<p align="center"><img src="../img/internal_figures/slide_x_inputscale.svg" width="760"></p>
`model.input_row_norm = s` sets every row of W_inp to the **absolute** L2 norm `s` at
initialisation. The default draw puts its rows at √(n_inputs/N) = 0.050 at N = 1000, so the four
rungs are 10×, 40×, 100× and 400× the default — the reference is the bottom of the ladder, not
its middle. Labelling it "×1" was what made the curve look like it zigzagged: put every rung at its
true scale and it is single-peaked, 263 → 302 → 339 → 324 → 306.

The peak is not seed scatter. The N = 500 cells, independent seeds and never plotted, give the same
ordering: 191 → 219 → 237 → 226 → 213.

Why it turns over. (i) The knob is an **initialisation**, not a constraint — W_inp is trainable and
nothing holds the row norm. Every arm up to row norm 5 ends at the same total, ‖W_inp‖_F ≈ 94, from
starting totals of 1.7, 16, 63 and 158. Only row norm 20 is still above it at 150k (377), so that arm
is not even at a comparable state. (ii) Starting everyone at the same middling norm leaves the trained
W_inp least concentrated — mean row norm 1.67 at rung 2 against 1.09 for the default draw, with the
same total to share — and a less concentrated W_inp is more units with input drive. (iii) Every arm is
still falling at the read-out. The bigger the init, the later the collapse starts and the steeper it is
when it comes: at 40k the five arms read 378 / 446 / 517 / 506 / **758**, and rung 20 crosses below
rung 2 somewhere between 100k and 130k. The 150,000-iteration read-out catches five curves that are
still crossing.

The size of the whole effect, for scale: the reference cell has a 500k budget, and training it on past
150k with nothing changed takes it from 263 to 190. Every input scale we tried buys less than the next
350,000 iterations take away.

### 11b. And the units it buys are free
<p align="center"><img src="../img/internal_figures/slide_x_inputscale_r2.svg" width="760"></p>
The same 15 networks, both axes read at 150,000 iterations. Across a 247–342 range of active units
every one sits within 0.002 of R² = 0.964: whatever the extra units are doing, the task does not need
them, and the rungs that shed them do not pay for it.

R² here is the **noise-free** probe the Trainer records beside the participation vector, not the score
in the folder name — the reference cell has a 500k budget, so its folder score is the best over a run
3.3× longer than the rungs it is compared with, while the count beside it is read at 150k. The two
differ by a uniform offset: re-scoring every cell offline with the noise on and off gives 0.0047 at
all five rungs, so the choice favours none of them.

### 12. The field-standard metabolic penalty moves nothing beyond seed scatter
<p align="center"><img src="../img/internal_figures/slide_x_metabolic.svg" width="760"></p>

### 12b. …and it does not break the task either
<p align="center"><img src="../img/internal_figures/slide_x_metabolic_r2.svg" width="760"></p>
The same 17 networks as the panel above, read from their own run folders rather than the archived
CSV, which has no r² column. Active counts agree element for element (414 / 393 / 431 / 432 / 380),
so the two paths are reading the same networks. r² never leaves 0.83–0.88, which closes the first
objection to a null result: the penalty did not fail to move the count because it was busy breaking
CDDM.

The cells with more live units also score slightly higher — across the five cell means r = 0.94,
but that is five points and the whole r² span is 0.05, so it is an observation, not a result.
Within a cell there is nothing (r = 0.13 over 17 networks).

**The ladder is not non-monotonic in what the penalty controls.** `mean(fr²)` penalises rate
magnitude, and it delivers: q₉₅ of the participation distribution falls 0.33 → 0.33 → 0.23 → 0.14 →
0.05 across λ = 0 → 10, a 6× shrink with no reversal. The plotted criterion counts units above
0.05·q₉₅(p), so the bar falls by the same 6× and divides the effect out by construction. What is
left on the axis is a race between the bar and the body of the distribution, and at λ ≤ 1 the two
shrink factors differ by under 30% — so seed noise picks the sign. Only at λ = 10 does the gap open
(bar 6.0×, body 8.1×) and the count fall for real, by 34 units at 3.9 standard errors.

On a **fixed** bar the same networks go 448 → 427 → 452 → 371 → 169 active, a 2.7× collapse. The
units are not dying, they are being turned down: at λ = 10, 568 units still exceed 10⁻⁶ against the
reference's 576, while only 169 exceed 10⁻². N = 1000 is also the size with the least headroom — at
N = 100 the same ladder runs 12 → 12 → 17 → 33 → 59% silent.

### 13. Nor the equation form, nor a trainable bias
<p align="center"><img src="../img/internal_figures/slide_x_architecture.svg" width="760"></p>

### 14. Removing recurrent noise is the largest effect — and it is negative
<p align="center"><img src="../img/internal_figures/slide_x_recnoise.svg" width="760"></p>
Mean and 95% interval: this sweep saved no per-seed rows.

### 15. Everything above, on one axis
<p align="center"><img src="../img/internal_figures/fig_paper_F1.svg" width="760"></p>
Panel d. Each against its **own** matched reference. ⚠ wants its own export.

---

## WHAT DOES WORK

### 16. Five interventions, and where each one acts
<p align="center"><img src="../img/internal_figures/slide_rules.svg" width="760"></p>

### 17. Active units
<p align="center"><img src="../img/internal_figures/slide_f2_active.svg" width="760"></p>

### 18. Performance
<p align="center"><img src="../img/internal_figures/slide_f2_r2.svg" width="760"></p>

### 19. Dimensionality
<p align="center"><img src="../img/internal_figures/slide_f2_dims.svg" width="760"></p>

### 20. Weight distribution — lognormal, and which way it errs
<p align="center"><img src="../img/internal_figures/slide_f2_weights.svg" width="760"></p>

### 21. Does it survive a change of size? — units
<p align="center"><img src="../img/internal_figures/slide_f2_size_active.svg" width="760"></p>

### 22. …and performance
<p align="center"><img src="../img/internal_figures/slide_f2_size_r2.svg" width="760"></p>

---

## PER-INTERVENTION DETAIL

### 23. Dropout, along training
<p align="center"><img src="../img/internal_figures/dropout_live_vs_iter.png" width="760"></p>

### 24. Dropout is capped: the sampler cannot see firing
<p align="center"><img src="../img/internal_figures/dropout_sampler_blindness.png" width="760"></p>

### 25. Prune-and-duplicate ⚠
Needs its own figure: recruitment against jitter, and the output unchanged at the moment of surgery.

### 26. Synaptic noise ⚠
Needs its own figure: the σ_w ladder, active units and clean r² against noise level.

### 27. The penalty pair ⚠
`fig_paper_F3.pdf` exists but will not render in Markdown — needs an SVG export.

---

## Open

- The four-measure comparison is one task. CDDM and DMTS size series are training.
- Three figures still to build (the last three slides) and one panel to export (slide 14).
