## 2026-10-31 17:55 — Figure 1 rebuilt (four panels); panel d re-run at the current floor, C1 passes its rule

Follows `2026-10-31_1702_figure-1-panel-d-ablations-redone-at-the-current-floor-pre-registered.md`,
which set the rule before these jobs ran. Work package fig1-inactive (ptolmachev/dead-relu-paper#5).

**Jobs.** From `~/trainRNNbrain_fig1` on Della, detached at 538bd77 (`ablate_silent_units.py`
unchanged from main 484e0a4):
- 16555113, `floor` ablation, 9 networks, 2.5-3.5 min each on one GPU;
- 16555114, `retrain --at floor`, 9 networks, 22 min (CDDM, DMTS) and 34 min (3BFF) each;
- 16556248, `fig_paper_F1.py --collect` at 49fa02d (CPU, 8.5 min), after both.

Total about 4.3 GPU-h. Results in `/scratch/gpfs/TENGEL/pt1290/c1_ablation_79932ef` and
`c1_retrain_79932ef`; job lists in `c1_ablation_79932ef/lists/`. Every ablated snapshot now sits
within 250 iterations of its network's crossing under read-out 79932ef (before: up to 24k).

**Panel d, at the current floor** (N = 1000, three networks per task; held-out R², nine draws):

| task | inactive units | R² intact | every inactive unit removed | as many removed at random (mean of 5) | retrained 3,000 it: inactive removed | intact | random removed |
|---|---|---|---|---|---|---|---|
| 3BFF | 733, 658, 713 | 0.950 | 0.260, 0.347, 0.517 | 0.012, 0.084, 0.015 | 0.945, 0.946, 0.948 | 0.949, 0.950, 0.950 | 0.920, 0.927, 0.930 |
| CDDM | 570, 584, 562 | 0.833, 0.830, 0.827 | 0.832, 0.830, 0.824 | 0.075, −0.149, −0.024 | 0.829, 0.828, 0.825 | 0.829, 0.827, 0.825 | 0.828, 0.821, 0.828 |
| DMTS 5τ | 678, 589, 739 | 0.999 | 0.944, 0.746, 0.988 | −0.111, −0.126, −0.125 | 0.999, 0.999, 0.999 | 0.999, 0.999, 0.999 | 0.998, 0.419, 0.998 |

Rows list networks in the order of their seeds' file names. Against the 14 Oct ablation the means move
by at most 0.06 (DMTS with every inactive unit removed: 0.95 → 0.89, because s3753404946 is now read
at 66k instead of 42k).

**Verdict against the rule: C1 passes in all nine networks.**
1. CDDM, every inactive unit removed: R² changes by −0.001, 0.000 and −0.003 (bar 0.01).
2. Pruned of the inactive units and retrained, against the intact network retrained: −0.004, −0.004,
   −0.002 on 3BFF; 0.000, +0.001, 0.000 on CDDM; 0.000 on DMTS (bar 0.01).
3. Pruned at random and retrained, against pruned of the inactive units and retrained: lower on 3BFF
   (by 0.018-0.025) and DMTS (by 0.001, 0.580, 0.001); on CDDM −0.001, −0.007, +0.003, all within 0.01.

Random pruning followed by retraining costs nothing on CDDM either (0.821-0.828 against 0.825-0.829):
on CDDM the retrain test separates the two removals only before retraining.

**Figure.** `fig_paper_F1.py` at 49fa02d, drawn on the Mac from the record of job 16556248:
[img/internal_figures/fig_paper_F1.pdf](../../img/internal_figures/fig_paper_F1.pdf) (+ .svg; no
rasters; the overlap check reports no overlapping text among 77 text artists). Panels: (a) example
network, (b) participation before and after training, (c) active units at N = 1000 per task at the
floor with held-out R² (the read of entry 1702: 3BFF 278 ± 33, CDDM 431 ± 10, DMTS 218 and 422),
(d) the table above. The size and training-time panels moved to Figure 2 (package fig2-size-time);
their functions (`panel_c`, `panel_e`) stay in the module.
