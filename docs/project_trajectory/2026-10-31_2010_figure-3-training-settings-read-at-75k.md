## 2026-10-31 20:10 — Figure 3: every training-setting ladder and B5 read at 75,000 iterations

Work package fig3-settings (dead-relu-paper#7), branch `wp/fig3-settings`.

**What was run.**
- `fig_paper_F2_fails.py --collect` on Della, CPU job 16562362 (8 cores, 20 min wall, 1.3 CPU-h, no
  GPU), from a pinned worktree `~/trainRNNbrain_fig3settings` at 919fc84 with its own copy of the
  held-out R² cache. It reads every ladder and revival run at its floor (unchanged from Figure 2 of
  16 Oct) and at 75,000 iterations (`read_fixed`): the active count at the nearest probe, and
  held-out R² (36 seeded draws) measured on weights from that iteration where they exist (the run's
  own final weights, its warm-start parent, a kept snapshot), else estimated by the loss ratio. The
  CDDM input-scale ladder ends at 50k and is read there.
- `b5_read.py` on the Mac, the same 75k read for B5. The SGD lr 3, wd 0 arm finished on Spock
  (`~/trainRNNbrain_anal`) and was copied to the Mac without its Adam states: 2 runs at R² 0.21-0.25,
  1 diverged (folder `nan_...`, skipped by `runs_with_loss`).
- `fig_paper_F3_settings.py` (new) draws Figure 3 and Extended Data Fig. 5 from the two caches.

Pass rule set before drawing (from the brief): if a setting keeps more units active than the default
at equal R² at 75k, open a needs-pavel issue on the C5 wording.

**CDDM, N = 1000, at 75k** (ranges over runs; mean in brackets):

| ladder | default | other values |
|---|---|---|
| weight decay | 1e-6: 325-345 (337) | 0: 442-469 (453); 1e-5: 260-263; 1e-4: 154-166 (160) |
| recurrent noise | 0.05: 311-340 | 0: 245-258; 0.01: 249-279; 0.1: 378-412 |
| metabolic λ | 0: 325-345 | 0.01: 324-344; 0.1: 347-356; 1: 360-387; 10: 321-342 at R² 0.787-0.789 |
| spectral radius | 1.2: 333-347 | 0.5: 355-367; 2.0: 308-317 |
| W_rec penalty λ | 0: 333-347 | 0.05: 434-441; 0.2: 445-450 |
| activation | ReLU: 325-345 | leaky 348-385; softplus 372-376; sigmoid 198-234 |
| rate / sparsity | none: 320-334 | rate 0.2: 1000; sparsity 0.05: 248-281; both: 1000 |
| input scale (50k) | 355-376 | 0.5: 334-336; 2: 247-268; 5: 201-216; 20: 166-185 |
| revival | none 332-340 | cap 0.3: 390-400; re-initialise 330-346 |

R² at 75k is 0.823-0.839 everywhere except the metabolic penalty at λ ≥ 1 (0.819-0.823 and
0.787-0.789) and the rate penalty (0.822-0.830). Weight decay 0 still falls: 466-494 at the floor,
442-469 at 75k, 400-428 at 200k; weight decay 1e-4: 378-386, 154-166, 83-88.

**3BFF, N = 1000:** input-weight scale (row norm), floor / 75k / 150k: default 268-297 / 245-296 /
189-235; 0.5: 370-411 / 284-343 / 224-236; 2: 441-526 / 387-424 / 273-343; 5: 560-663 / 349-443 /
267-330; 20: 615-920 / 455-500 / 312-340. R² at 75k 0.950-0.954 (estimated). The scale-20 gain is
2.5-fold at the floor (means 718 vs 283), 1.7-fold at 75k (477 vs 275), 1.6-fold at 150k (329 vs 206).
Revival at 75k: none 236-288 (4 runs; FFseed4 now included), zero-out 258-274, random redraw 205-245.

**Verdict on the rule:** the 3BFF input-weight scale keeps more units at 75k at equal R², so the
issue was opened: dead-relu-paper#24. Recommended (and followed in the PR until Pavel answers):
retitle Results 3 to "No standard training setting stops the loss of active units". The count falls
under every setting; no CDDM value raises the mean by more than 116 units.

**B5 at 75k** (3BFF, N = 1000, 3 runs per arm): Adam wd 1e-6 233-270 at R² 0.951; Adam wd 0
252-367 at 0.956-0.959; AdamW 306-358 at 0.957-0.958; SGD lr 0.03 721-759 at 0.873-0.875; lr 0.1
569-620 at 0.880-0.899; lr 0.3 230-285 at 0.892-0.915; lr 1 212-281 at 0.877-0.906 plus one run at
0.578 (a loss burst at 74-75k: median loss 0.31 in the window against 0.086 over 60-90k); lr 3 does not
learn (R² 0.21-0.44 at 200k; four runs have no 75k estimate because their loss is not finite in the
last 10%).

**Correction to entry 2026-10-14_1301 (B5 at the floor).** Under the current read-out (79932ef) most
SGD runs have no floor read: 1 of 24 SGD runs at lr 0.03-1 reads (lr 0.3, wd 0: 236 active, R² 0.917);
the rest never stay under the bar or end below the fitted floor. Adam wd 1e-6 now reads 187-280 at
the floor. The floor table of that entry is withdrawn; the board's C3 now quotes the 75k read.

**Figures.** `img/internal_figures/fig_paper_F3_settings.{pdf,svg}` (Figure 3),
`fig_paper_F2_fails.{pdf,svg}` (now Extended Data Fig. 4, floor read open, 75k filled),
`fig_ed_optimizer.{pdf,svg}` (Extended Data Fig. 5). Layout checks: 0 problems each. Caches:
`data/fig_paper_F2_fails_cache.pkl` (the 16 Oct cache kept as `fig_paper_F2_fails_cache_20261016.pkl`),
`data/b5_read_cache.pkl`.
