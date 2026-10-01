# Dormant ReLU units — talk track

One claim per slide, one panel per slide. Every figure is written by
`trainRNNbrain/experiments_and_analysis/fig_slides.py`, which calls the manuscript's own panel
functions and loaders — if a number here disagrees with the paper, that is a bug, not a second
opinion.

`⚠` marks a slide whose figure does not exist yet.

---

## THE PROBLEM

### 1. Most units of a trained ReLU RNN never fire
![](../img/internal_figures/slide_01_schematic.svg)

### 2. It is not a threshold artefact — the distribution is bimodal
![](../img/internal_figures/slide_02_participation.svg)

### 3. Units keep going silent long after the loss has stopped moving
![](../img/internal_figures/slide_03_silencing_vs_training.svg)
Three tasks, every seed. DMTS does not follow the other two — shown, not hidden.

---

## WHY ITERATION COUNT IS THE WRONG CLOCK

### 4. The input weights settle
![](../img/internal_figures/slide_04_drift_W_inp.svg)

### 5. The recurrent weights settle
![](../img/internal_figures/slide_04_drift_W_rec.svg)

### 6. The read-out weights settle hardest
![](../img/internal_figures/slide_04_drift_W_out.svg)
All three are mean-reverting by the end. **The weights stop moving and units keep going silent** —
so silencing is not weight drift.

### 7. But bigger networks need longer to get there
![](../img/internal_figures/excess_time_matrix.png)
Iterations at 1.10× each run's own loss floor. Matching iterations compares a converged small
network with an unconverged large one — so every read-out below is at a matched loss, not a matched
step count.

---

## THE SCALING

### 8. Active units grow as N^0.3–0.4 — so the fraction falls
![](../img/internal_figures/slide_06_scaling.svg)
Both silence criteria, both task families.

---

## WHAT DOES NOT WORK — one knob at a time

### 9. A different activation does not help
![](../img/internal_figures/slide_x_activation_cddm.svg)

### 10. Nor on the other task
![](../img/internal_figures/slide_x_activation_ff.svg)

### 11. Weight decay makes it monotonically worse
![](../img/internal_figures/slide_x_weightdecay.svg)

### 12. Scaling the input weights does not help
![](../img/internal_figures/slide_x_inputscale.svg)

### 13. The field-standard metabolic penalty moves nothing beyond seed scatter
![](../img/internal_figures/slide_x_metabolic.svg)

### 14. Nor the equation form, nor a trainable bias
![](../img/internal_figures/slide_x_architecture.svg)

### 15. Removing recurrent noise is the largest effect — and it is negative
![](../img/internal_figures/slide_x_recnoise.svg)
Mean and 95% interval: this sweep saved no per-seed rows.

### 16. Everything above, on one axis
![](../img/internal_figures/fig_paper_F1.svg)
Panel d. Each against its **own** matched reference. ⚠ wants its own export.

---

## WHAT DOES WORK

### 17. Five interventions, and where each one acts
![](../img/internal_figures/slide_rules.svg)

### 18. Active units
![](../img/internal_figures/slide_f2_active.svg)

### 19. Performance
![](../img/internal_figures/slide_f2_r2.svg)

### 20. Dimensionality
![](../img/internal_figures/slide_f2_dims.svg)

### 21. Weight distribution — lognormal, and which way it errs
![](../img/internal_figures/slide_f2_weights.svg)

### 22. Does it survive a change of size? — units
![](../img/internal_figures/slide_f2_size_active.svg)

### 23. …and performance
![](../img/internal_figures/slide_f2_size_r2.svg)

---

## PER-INTERVENTION DETAIL

### 24. Dropout, along training
![](../img/internal_figures/dropout_live_vs_iter.png)

### 25. Dropout is capped: the sampler cannot see firing
![](../img/internal_figures/dropout_sampler_blindness.png)

### 26. Prune-and-duplicate ⚠
Needs its own figure: recruitment against jitter, and the output unchanged at the moment of surgery.

### 27. Synaptic noise ⚠
Needs its own figure: the σ_w ladder, active units and clean r² against noise level.

### 28. The penalty pair ⚠
`fig_paper_F3.pdf` exists but will not render in Markdown — needs an SVG export.

---

## Open

- The four-measure comparison is one task. CDDM and DMTS size series are training.
- Three figures still to build (26, 27, 28) and one panel to export (16).
