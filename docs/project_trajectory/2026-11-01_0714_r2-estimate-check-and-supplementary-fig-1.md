## 2026-11-01 07:14 — Estimated R² at the floor crossing is within 0.004 of the measured value on the nine Fig. 1d networks; Supplementary Fig. 1 redrawn in the house style

Work package `methods-r2` (ptolmachev/dead-relu-paper#51), trainRNNbrain branch `wp/methods-r2`
(rebased onto `wp/figure-style` 3b39d86 for `paperstyle.py`).

**R² estimate.** Methods M05 used to quote a check of the estimate
R²(t\*) = 1 − (L(t\*)/L_end)(1 − R²_end) that was made at a different crossing and with 9-draw scores
(entry `2026-10-13_1506`). Recomputed under the current crossing (`common.read_run`) and 36-draw
scores: for every network of the Figure 2 collect record whose R² at t\* was measured on a kept
snapshot (the nine N = 1,000 networks of Fig. 1d), the estimate from the same record's R²_end.
Descriptive check; no threshold, and nothing was chosen from it.

Della CPU job 16568910 (`sbatch --wrap`, torch-env, PYTHONPATH `~/trainRNNbrain_fig2size` at
8bb2939), script `r2_estimate_check.py` at c837417 (b2594cb after the rebase, same content),
record `~/trainRNNbrain_fig2size/data/fig_paper_F2_size_collect.pkl` (MD5 a7e9cdd1…).

```
3BFF  N=1000  s 2461724787 t*=   50073 measured 0.9504  end 0.9519  L*/Lend 1.050  estimate 0.9496  diff -0.0008
3BFF  N=1000  s 4113307577 t*=   58035 measured 0.9503  end 0.9520  L*/Lend 1.050  estimate 0.9496  diff -0.0007
3BFF  N=1000  s 1258717025 t*=   67191 measured 0.9505  end 0.9526  L*/Lend 1.049  estimate 0.9503  diff -0.0002
CDDM  N=1000  s 2657899762 t*=   26028 measured 0.8245  end 0.8353  L*/Lend 1.042  estimate 0.8284  diff +0.0040
CDDM  N=1000  s 1423687489 t*=   25020 measured 0.8308  end 0.8364  L*/Lend 1.045  estimate 0.8291  diff -0.0017
CDDM  N=1000  s 2670295995 t*=   24219 measured 0.8308  end 0.8361  L*/Lend 1.045  estimate 0.8287  diff -0.0021
DMTS  N=1000  s 3322309204 t*=   20477 measured 0.9992  end 0.9994  L*/Lend 0.992  estimate 0.9994  diff +0.0002
DMTS  N=1000  s  198893703 t*=   23106 measured 0.9991  end 0.9994  L*/Lend 0.951  estimate 0.9994  diff +0.0003
DMTS  N=1000  s 3753404946 t*=   65778 measured 0.9990  end 0.9994  L*/Lend 0.920  estimate 0.9995  diff +0.0005
9 networks; largest |estimate - measured| = 0.0040
```

Largest |estimate − measured| 0.004 (CDDM s2657899762); 3BFF and DMTS within 0.001. M05 now states
"within 0.004 of the measured value" on these nine networks.

**Supplementary Fig. 1** (`fig_supp_tasks.py`): drawn at 165.1 × 140 mm (prints at 100%), every text
at 6.0-6.5 pt (bold row titles 6.5 pt, panel letters 8.5 pt), GridSpec margins set explicitly; the
script checks text size and overlap (none under 5 pt, no overlapping text). Targets in the task
colours of `paperstyle.TASK_COLORS`; inputs in black (sensory), grey (cue) and light grey (constant);
the CDDM colour feature drawn as dark against light grey arrows, because every hue of the palette is
reserved. `pdffonts`: Helvetica, Helvetica-Bold, Helvetica-Oblique only. On the built page
(dead-relu-paper `wp/methods-r2`), median text 6.0 pt, smallest 6.0 pt (PyMuPDF spans).
Figure: `img/internal_figures/fig_supp_tasks.{pdf,svg}`.
