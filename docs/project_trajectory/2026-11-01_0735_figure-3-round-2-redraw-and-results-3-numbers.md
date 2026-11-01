## 2026-11-01 07:35 — Figure 3 and Extended Data Fig. 4 redrawn at print width; the numbers Results 3 now uses

Work package fig3-r2 (dead-relu-paper#54), trainRNNbrain branch `wp/fig3-r2` (from main e28b86c,
the merged style sheet). No new runs: every number below comes from the caches of entry
2026-10-31_2010 (`data/fig_paper_F2_fails_cache.pkl`, `data/b5_read_cache.pkl`), printed by
`fig_paper_F3_settings.py` and `fig_paper_F2_fails.py` (draw mode) on the Mac.

**Figures.** Both drawn at 165.1 mm (Figure 3: 128 mm tall; Extended Data Fig. 4: 178 mm), so they
print at 100%; Helvetica only; median printed text 6.0 pt; layout checks 0 problems; `tests/test_paperstyle.py`
passes on both. Colours by meaning (`paperstyle`): default grey, other values on the teal setting
ramp, the rate and sparsity penalties in the violet rule family; no task colour anywhere.
- Figure 3a now has 9 rows (was 40): CDDM default, weight decay 0 and 1e-4, metabolic penalty 10,
  the means of the 15 other CDDM values, the rate penalty alone and with the sparsity penalty
  (labelled "per-unit rule"), 3BFF default and initial input-weight scale 20. Panel d leaves out
  SGD at learning rate 3 (in Extended Data Fig. 4l). Panel e is named "revival methods" (glossary).
- Extended Data Fig. 4 panel l is the optimizer comparison at the floor crossing and at 200,000
  iterations (the old Extended Data Fig. 5 without its 75,000 panel, which repeated Fig. 3d). The
  old bottom-line panel l is gone.

**Figure 3a, at 75,000 iterations** (mean active units; held-out R² minus the same sweep's default
mean):

| row | networks | active (range, mean) | R² minus default (mean) | R² |
|---|---|---|---|---|
| CDDM default (weight-decay sweep) | 3 | 325-345, 337 | 0 | estimated |
| weight decay 0 | 3 | 442-469, 453 | −0.001 | estimated |
| weight decay 1e-4 | 3 | 154-166, 160 | −0.011 | estimated |
| metabolic penalty 10 | 3 | 321-342, 330 | −0.046 | measured |
| 15 other CDDM values (means) | 15 means | 217-448, 335 | −0.014 to +0.002 | both |
| rate penalty 0.2 | 5 | 1,000 | −0.004 | measured |
| rate + sparsity penalties | 5 | 1,000 | −0.008 | measured |
| 3BFF default | 3 | 245-296, 275 | 0 | estimated |
| 3BFF input-weight scale 20 | 3 | 455-500, 477 | +0.002 | estimated |

So every CDDM standard-setting value other than the metabolic penalty 10 stays within 0.02 of its
default's R² (lowest: metabolic penalty 1, −0.014). The 3BFF scale-20 networks have 329 active
units at 150,000 iterations against 206 for the default (means; ranges 312-340 and 189-235).

**The fall under every setting, checked per network** (the claim tested: every network with a floor
crossing has fewer active units at the end of its sweep's budget than at its crossing). Of
the 140 networks of the nine sweeps and the two revival screens, 129 have a floor crossing (11 do
not: the sparsity and rate + sparsity networks and others listed by `--probe`). All 124 that are
not rate-penalty networks have fewer active units at the end of the sweep's budget than at their
crossing; the 5 rate-penalty networks stay at 1,000. Measured between the crossing and 75,000
iterations instead, 3 networks do not fall (3BFF: one default network of the input-scale sweep,
283 and 283; one default and one outgoing-zeroed network of the re-initialising screen, 277 to 279 and
260 and 260); their crossings are at 57,000-68,000 iterations, within 18,000 iterations of 75,000.
Results 3 therefore states the end-of-training version.

**Revival methods at 75,000 iterations** (unchanged from entry 2026-10-31_2035): input-weight cap
395 against 335 (+60); re-initialising +3 on CDDM (338), +3 on 3BFF with outgoing weights zeroed
(264 against 261), −33 with outgoing weights kept (228).

**The bridge sentence of Results 3.** Fig. 1c's 431 (CDDM, N = 1,000, at the floor crossing) is the
size series' `CDDM_paper_grid` control (100,000 iterations); the 337 at 75,000 iterations is the
weight-decay sweep's default, `CDDM_std_g0_drift_bias0` (the networks of Fig. 2d, crossings at
28,249-30,139). These are different networks of the default configuration with one difference in
their saved configs: `same_batch: true` for the sweep networks (one fixed batch of all task
conditions every iteration) against `false` for the size series. At their own floor crossing the
sweep's default networks have 416-424 active units (mean 420), against 431 ± 10 for Fig. 1c. Results 3
words the sentence as "default CDDM networks have 431 ... and the default networks of the CDDM
sweeps below have 337", and Methods (M08) states the fixed batch of the CDDM sweeps.
