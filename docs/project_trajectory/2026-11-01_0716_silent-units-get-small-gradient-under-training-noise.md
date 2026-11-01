## 2026-11-01 07:16 — Silent CDDM units get a small, nonzero gradient under the training noise (correction to 2026-10-31_2207)

Corrects `2026-10-31_2207_c4-at-the-current-floor-passes-its-rule-figure-4-built.md`, which said
silent units' incoming weights "change only through weight decay". Found by the independent review
of dead-relu-paper PR #60 (point 1) and re-measured here.

**What was measured.** The zero gradient of silent units holds on the noise-free reference batch,
which is where silent is defined. `c4_gradient_mechanism.py` also keeps 10 training-like batches
per network under the run's own noise (σ_rec = σ_inp = 0.05, σ_out = 0.03; keys `noisy_g_in`,
`noisy_n_on`, Della job 16566988). Read by `fig_paper_F4_noreturn.py draw` (`report_c4`), each
batch's g_in divided by that batch's 95th percentile:

| CDDM network | Silent units | Steps with x > 0 per batch (median) | Per-batch gradients exactly 0 | Median g_in / q95 | Times below the reach threshold | Units at or above the threshold in some batch |
|---|---|---|---|---|---|---|
| s1423687489 | 313 | 1.65% | 0 of 3,130 | 5.9e-4 | 85 | 5 |
| s2657899762 | 293 | 1.58% | 0 of 2,930 | 1.7e-4 | 291 | 0 |
| s2670295995 | 260 | 1.54% | 0 of 2,600 | 8.3e-4 | 60 | 3 |

So under the training noise every silent unit is above zero on about 1.6% of steps and receives a
gradient 60-290 times (median per network) below the reach threshold; 8 of 866 cross it in at least
one of 10 batches. Adam rescales small but consistent gradients, so the 39-41% shrinkage of these
units' incoming rows over 5,000 iterations (c4_snapshot_drift.py) cannot be assigned to weight
decay alone. What stays measured: 5,000 iterations after the floor crossing, at most 1 silent unit
per network is active (0, 0, 1).

**What changes.** Results 4, the Figure 4 caption and Extended Data (gradient reach) now scope
"exactly zero" to the noise-free batch and give the noisy-batch numbers; the phrase "only through
weight decay" is removed. Extended Data mechanism panel a draws the silent units' noisy-batch
gradients beside the other classes. Board C4 paragraph updated.

**Figures redrawn in the shared style** (trainRNNbrain main e28b86c merged into
wp/fig4-no-return): 165.1 mm wide, Helvetica only, text at least 6 pt; `tests/test_paperstyle.py`
passes on all three PDFs (median printed text 6.0 pt, letters 8.5 pt bold). Colours: active and
inactive units in neutrals (silent hollow), tasks only in the panels that compare tasks, tanh as a
standard training setting against the ReLU default. Panel e now labels the median S0 of each group
(N = 1,000: 809 old, 709 young; N = 4,000: 3,648 old, 3,456 young). The CDDM active example unit of
panel a is now the one at the 75th percentile of participation (x up to +0.06 on the trial drawn;
the median one reached +0.04).
