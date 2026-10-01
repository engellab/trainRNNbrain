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
Iterations to reach 1.07× that run's **own** fitted floor. Within a task, size barely matters: 35k
to 43k over an 8× range (pooled fit T ∝ N^0.143 [0.065, 0.244], which is 1.35× over 8×). Across tasks
it does: the flip-flop reaches its floor near 35k, CDDM needs 100k, DMTS 150k.

Six of the 96 unpenalised runs never come within 7% of their own fitted floor and are absent from the
panel — at the looser 10% it was four. A run goes missing when its fitted floor sits a little below
what it actually reaches, so the threshold falls under its whole loss curve.

### 6. So: read every network where its own loss stops falling
<p align="center"><img src="../img/internal_figures/slide_06_readout_rule.svg" width="760"></p>
Each run has its own fitted floor (dotted) and its own crossing of 1.07× it (dashed, dot). That
iteration is the read-out — not a number fixed in advance. Every count in this talk is taken there.

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

### 9. Nor on the other task
<p align="center"><img src="../img/internal_figures/slide_x_activation_ff.svg" width="760"></p>

### 10. Weight decay makes it monotonically worse
<p align="center"><img src="../img/internal_figures/slide_x_weightdecay.svg" width="760"></p>

### 11. Scaling the input weights does not help
<p align="center"><img src="../img/internal_figures/slide_x_inputscale.svg" width="760"></p>

### 12. The field-standard metabolic penalty moves nothing beyond seed scatter
<p align="center"><img src="../img/internal_figures/slide_x_metabolic.svg" width="760"></p>

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
