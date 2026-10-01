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
<p align="center"><img src="../img/internal_figures/slide_02_participation_by_task.svg" width="760"></p>
One network per task at N = 1000, each at the end of its own budget. Active: 318, 269, 175.

**Participation** p_i = std(r_i) + q₀.₉(|r_i|) — how much unit *i*'s rate moves over a trial, and how
high it gets. **Criterion**: a unit is active when p_i ≥ 0.05 · q₀.₉₅(p), five per cent of the 95th
percentile of that network's own participation distribution. It is relative, so it does not assume a
scale — which is why the dashed line sits at a different place in each panel.

Shared x; each panel keeps its own count axis because CDDM puts 600 of its 1,000 units in one bin.

### 3. Units keep going silent long after the loss has stopped moving
<p align="center"><img src="../img/internal_figures/slide_03_silencing_vs_training.svg" width="760"></p>
Three tasks, every seed. DMTS does not follow the other two — shown, not hidden.

---

## WHY ITERATION COUNT IS THE WRONG CLOCK

### 4. The parameters never stop moving — all three tasks
<p align="center"><img src="../img/internal_figures/slide_04_drift_trajectories.svg" width="760"></p>
Relative weight change over a 10,000-iteration lag, every seed. Still 10–100% of the weights' own
magnitude at 140,000 iterations. The budgets differ because the tasks do — DMTS needs 150k to be
solved, CDDM 100k; the flip-flop's longer runs predate drift logging.

### 5. But bigger networks need longer to reach their loss floor
<p align="center"><img src="../img/internal_figures/excess_time_matrix.png" width="760"></p>
Iterations at 1.10× each run's own loss floor. Matching iterations compares a converged small
network with an unconverged large one — so every read-out below is at a matched loss, not a matched
step count.

---

## THE SCALING

### 6. Active units grow as N^0.3–0.4 — so the fraction falls
<p align="center"><img src="../img/internal_figures/slide_06_scaling.svg" width="760"></p>
Both silence criteria, both task families.

---

## WHAT DOES NOT WORK — one knob at a time

### 7. A different activation does not help
<p align="center"><img src="../img/internal_figures/slide_x_activation_cddm.svg" width="760"></p>

### 8. Nor on the other task
<p align="center"><img src="../img/internal_figures/slide_x_activation_ff.svg" width="760"></p>

### 9. Weight decay makes it monotonically worse
<p align="center"><img src="../img/internal_figures/slide_x_weightdecay.svg" width="760"></p>

### 10. Scaling the input weights does not help
<p align="center"><img src="../img/internal_figures/slide_x_inputscale.svg" width="760"></p>

### 11. The field-standard metabolic penalty moves nothing beyond seed scatter
<p align="center"><img src="../img/internal_figures/slide_x_metabolic.svg" width="760"></p>

### 12. Nor the equation form, nor a trainable bias
<p align="center"><img src="../img/internal_figures/slide_x_architecture.svg" width="760"></p>

### 13. Removing recurrent noise is the largest effect — and it is negative
<p align="center"><img src="../img/internal_figures/slide_x_recnoise.svg" width="760"></p>
Mean and 95% interval: this sweep saved no per-seed rows.

### 14. Everything above, on one axis
<p align="center"><img src="../img/internal_figures/fig_paper_F1.svg" width="760"></p>
Panel d. Each against its **own** matched reference. ⚠ wants its own export.

---

## WHAT DOES WORK

### 15. Five interventions, and where each one acts
<p align="center"><img src="../img/internal_figures/slide_rules.svg" width="760"></p>

### 16. Active units
<p align="center"><img src="../img/internal_figures/slide_f2_active.svg" width="760"></p>

### 17. Performance
<p align="center"><img src="../img/internal_figures/slide_f2_r2.svg" width="760"></p>

### 18. Dimensionality
<p align="center"><img src="../img/internal_figures/slide_f2_dims.svg" width="760"></p>

### 19. Weight distribution — lognormal, and which way it errs
<p align="center"><img src="../img/internal_figures/slide_f2_weights.svg" width="760"></p>

### 20. Does it survive a change of size? — units
<p align="center"><img src="../img/internal_figures/slide_f2_size_active.svg" width="760"></p>

### 21. …and performance
<p align="center"><img src="../img/internal_figures/slide_f2_size_r2.svg" width="760"></p>

---

## PER-INTERVENTION DETAIL

### 22. Dropout, along training
<p align="center"><img src="../img/internal_figures/dropout_live_vs_iter.png" width="760"></p>

### 23. Dropout is capped: the sampler cannot see firing
<p align="center"><img src="../img/internal_figures/dropout_sampler_blindness.png" width="760"></p>

### 24. Prune-and-duplicate ⚠
Needs its own figure: recruitment against jitter, and the output unchanged at the moment of surgery.

### 25. Synaptic noise ⚠
Needs its own figure: the σ_w ladder, active units and clean r² against noise level.

### 26. The penalty pair ⚠
`fig_paper_F3.pdf` exists but will not render in Markdown — needs an SVG export.

---

## Open

- The four-measure comparison is one task. CDDM and DMTS size series are training.
- Three figures still to build (the last three slides) and one panel to export (slide 14).
