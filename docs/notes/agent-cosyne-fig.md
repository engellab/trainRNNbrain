# Agent cosyne-fig: the one-page COSYNE 2027 figure

Builder: `trainRNNbrain/experiments_and_analysis/fig_cosyne2027.py`. Output:
`img/internal_figures/fig_cosyne2027.{pdf,svg}` and a `qlmanage` PNG preview. The page is 7.47 x
6.35 in (the PDF MediaBox), against a budget of 7.5 x 6.4 in. Branch `cosyne2027-fig`, cut from
`main` at db10e7c. Della checkout `~/trainRNNbrain_cosyne`: its `data/` is a real folder holding
links to the scratch `trained_RNNs` and `c1_snapshots` and copies of the shared caches, so the
shared caches are never written.

## How it is built

The data sit on two machines, so the builder runs in three stages:

1. `--collect-della` (Della CPU job, about one minute). This stage runs `fig_paper_F1.panel_c` and
   `panel_d` once on a throwaway figure, records every input they read, and adds the example
   network, the panel E trajectories and the merged 75k cache. Output:
   `data/fig_cosyne2027_della.pkl`, 129 MB.
2. `--collect-b5` (Mac, where `NBitFlipFlop_b5_optimizer` sits). This stage runs
   `b5_read.read_arm` on every arm. Output: `data/fig_cosyne2027_b5.pkl`.
3. No flag (Mac). This stage calls the F1 panel functions unchanged, with their loaders replaced by
   replays of the record. It also reads `data/fig_paper_F2_fails_cache.pkl`, the Figure 2 cache of
   14 Oct 16:18. That cache already holds the revival top-ups (CDDM re-initialise 3 of 3 runs,
   flip-flop zero-out and random redraw 3 of 3 runs each).

**Layout check.** The check (`check_layout`) passes: 134 texts with no overlaps, nothing outside
the canvas, and no text under 7 pt. The F1 panels draw their text at 5-6.6 pt, and the builder
raises 18 of those texts to 7 pt. The content spans 0.05-7.43 in across and 0.04-6.31 in down a
7.46 x 6.36 in canvas.

**Anonymity.** The PDF metadata holds only `Creator`/`Producer` = Matplotlib, and the figure carries
no names.

**One change outside the new file.** `fig_paper_F1.RETRAIN_DIR` became `RETRAIN_DIRS`, searched in
order:
- `c1_retrain_seeded2/rep1` first. It holds the seeded-batch flip-flop retrains (FINETUNE_SEED,
  ff994a9), the ones the 2026-10-14 entry says Figure 1 should use.
- `c1_retrain_seeded` second. CDDM's retrains there are already deterministic, because its training
  batch is a fixed grid. DMTS has no seeded-batch rerun, so its stars still come from the unseeded
  fine-tune.

The next rebuild of Figure 1 picks up this change.

## Numbers drawn, per panel

Every count is active units, at N = 1000 unless stated.

**A.** This panel shows the 3-bit flip-flop control `NBitFlipFlop_ff_revive_c200000` N = 1000 after
200,000 iterations. It has 183 of 1000 units active and 817 silent. Of the 8 units drawn at random,
2 are active.

**B.** The same network has 945 of 1000 units active untrained and 183 after 200,000 iterations.

**C.** Each run is read at 1.05x its own noisy-loss floor. The flip-flop series is the exception:
it is drawn at the end of training, with open markers labelled "(end)".

| task | N | runs | active at the floor | at the end of training | fit |
|---|---|---|---|---|---|
| CDDM (`CDDM_paper_grid`, 100k) | 500 / 1000 / 2000 / 4000 | 3 each | 296-302 / 399-425 / 549-573 / 775-796 | | N^0.46 (floor) |
| DMTS 5τ (`DMTS_paper_grid_d5`, 150k) | 500 / 1000 / 2000 | 3 / 3 / 2 | 200-266 / 225-589 / 454-470 | | N^0.51 (floor) |
| flip-flop (`ff_revive_c200000`, 200k) | 500 / 1000 / 2000 | 6 / 3 / 3 | 187-237 / 287, 294 (2 of 3) / 363-410 | 130-156 / 170-181 / 250-255 | N^0.40 (end) |

- One DMTS run at N = 2000 never learned the task (R² below 0.8) and was dropped.
- The flip-flop series is drawn at the end of training because N = 4000 (`FFtop4000`) had not
  landed at build time. The rule draws a series at its floor only when every size has a floor
  read-out. When N = 4000 lands, re-run `--collect-della` and then draw.
- The table's floor column for the flip-flop is shown for reference; the panel does not draw it.

**D.** Ablation at the floor, three networks per task. The figures are means; "random" means the
same number of units removed at random. The shaded band is the silent share.

| task | iteration | silent | intact R² | silent removed | random removed | silent removed, then retrained 3k |
|---|---|---|---|---|---|---|
| flip-flop | 60k | 706 | 0.949 | 0.334 | 0.035 | 0.948 (seeded2: 0.947 / 0.948 / 0.948) |
| CDDM | 25k | 574 | 0.832 | 0.831 | -0.034 | 0.827 |
| DMTS | 27k | 652 | 0.999 | 0.949 | -0.117 | 0.999 |

**E.** Flip-flop and CDDM only. DMTS was dropped for space; it shows the same pattern (556 silent
at its floor, 833 at 150k). Each value is the mean over runs with a floor read-out.

| task | runs (with a floor) | crossing | silent at the crossing | silent at the end | held-out R² |
|---|---|---|---|---|---|
| flip-flop (200k) | 3 (2) | 64.7k | 710 | 826 at 200k | 0.952 |
| CDDM (`CDDM_std_g0_drift_bias0`, 200k) | 3 (3) | 28.1k | 582 | 734 at 200k | 0.839 |

So after the loss floor, the flip-flop's active count falls from 290 to 174, and CDDM's from 418 to
266.

**F.** The y axis is R² minus the R² of the run's own control. Each control is drawn at (its mean
count, 0).

Points per family:
- 79 knob runs. These are Figure 2's ladders, every rung at its own floor; the sparsity penalty and
  the CDDM input-weight cap count as knobs.
- 9 re-initialise/zero runs.
- 28 optimizer runs.
- 26 participation-drive runs: 10 CDDM at the floor and 16 flip-flop at 75k.

The 10 distinct controls:

| control | active | R² | runs |
|---|---|---|---|
| CDDM drift (WD, metabolic, activation ladders) | 418 | 0.829 | 3 |
| noise ladder σ 0.05 | 406 | 0.829 | 5 |
| recinit sr 1.2 | 423 | 0.828 | 2 |
| CDDM c1 snapshots | 430 | 0.831 | 3 |
| A4 no-penalty | 415 | 0.827 | 5 |
| screen3 none | 436 | 0.829 | 3 |
| flip-flop ksweep | 289 | 0.951 | 2 |
| flip-flop ff_revive 200k | 290 | 0.950 | 2 |
| B5 Adam wd 1e-6 | 250 | 0.951 | 3 |
| 75k control (open marker) | 275 | 0.952 | 6 |

Readings, each against its own control:
- **Rate penalty, CDDM (A4):** 1000 of 1000 active at its floor, at ΔR² of -0.004 to -0.006
  (5 runs). The trajectory quotes -0.005 to -0.007, measured on the 50k and 75k weights; the figure
  uses `fig_knobs_at_floor`'s floor estimate.
- **Rate + sparsity, CDDM:** 1000 active, ΔR² -0.014 to -0.022.
- **Flip-flop remedies at the end of 75k (open markers):**
  - dropout: 496-568 active, ΔR² -0.013 to -0.019
  - duplication: 784-828 active, ΔR² -0.004 to -0.006
  - rescaling: 365-479 active, ΔR² -0.010 to -0.014
  - synaptic noise: 551-591 active, ΔR² -0.019 to -0.026
  - rate + sparsity: 892-958 active, ΔR² -0.024 to -0.034
- **Optimizers:**
  - Adam wd 0: 225, 254 active, ΔR² +0.009 to +0.010
  - AdamW: 282, 292 active, ΔR² +0.010
  - SGD keeps more than 400 units only at ΔR² -0.046 to -0.069 (lr 0.03: 703-736; lr 0.1: 483-585).
  - At lr 0.3 and 1, SGD keeps 230-331 at ΔR² -0.035 to -0.059.
  - SGD lr 3 (wd 1e-6) loses the task (R² 0.33 at its floor). It is left out and named under the
    key.
  - The Mac copy of B5 now includes the two extra SGD wd 0 lr 1 runs from Spock (3 of 3).
- **Re-initialise/zero:**
  - CDDM re-initialise: 412-427 active, ΔR² -0.001.
  - Flip-flop zero-out: 258-276 active.
  - Flip-flop random redraw: 204-246 active, ΔR² about 0.
- **Standard knobs at the floor.** All knob runs fall between 235 and 720 active, except
  flip-flop input scale 20, and all but three sit within 0.02 of the control's R². The three are
  the two metabolic λ = 10 runs at ΔR² -0.051 and one CDDM input-scale-5 run at -0.030.

**G.** The multiplier is each remedy's mean active count over the 75k control's at the same N,
using the operating-point cell (`fig_paper_F2.pick_cell`). Number of networks in brackets;
control networks: 6, 6, 3, 3 at N = 500 / 1000 / 2000 / 4000.

| remedy | N = 500 | 1000 | 2000 | 4000 |
|---|---|---|---|---|
| dropout | 2.04 (3) | 1.95 (3) | 2.00 (3) | 2.01 (3) |
| duplication | 2.22 (3) | 2.92 (3) | 3.33 (3) | 3.70 (3) |
| rescaling | 1.54 (4) | 1.57 (4) | 1.86 (3) | 1.94 (6) |
| synaptic noise | 1.90 (3) | 2.08 (3) | 1.63 (3) | 1.65 (5) |
| rate + sparsity | 2.51 (1) | 3.39 (3) | 5.36 (3) | 6.80 (1) |

## Checks

Each check had its pass bar written before its numbers were read.

- **G against the trajectory (bar: every cell within 0.01 of the 2026-10-09 75k table).** Pass: all
  20 multipliers equal that table to two decimals.
- **B5 against the trajectory (bar: every arm's range equal to the 2026-10-14 and 2026-10-15
  tables).** Pass, for example Adam wd 1e-6 at 237-271, R² 0.950-0.951, and SGD lr 0.03 wd 0 at
  703-727.
- **D against the trajectory (bar: equal to agent-fix's numbers).** Pass: the flip-flop retrain
  values equal agent-fix's R3 numbers (0.94697 / 0.94751 / 0.94812), and silent-removed equals
  0.25 / 0.22 / 0.53 (mean 0.334).
- **Leftover in D.** The flip-flop's "active removed" mean prints as -2e9: the output diverges once
  every active unit is gone. Panel D clips it at 0, so it does not affect the figure.

## Not done, or open

- **Panel C waits on N = 4000.** The flip-flop series is still read at the end of training until
  `FFtop4000` lands (121k of 125k at 13:32 on 15 Oct). See the update below.
- **The remedies have no floor read-out.** The flip-flop remedies (F, G) are the end of 75k. Their R²
  is the 75k cache's unseeded 8-draw score under the training noise, and it is compared only with
  the 75k control scored the same way. The 200k top-ups (`RMtop200`) are still training.
- **Most R² values in F are estimates.** Every R² in F except the 75k points is
  `fig_knobs_at_floor`'s floor R². It is measured on a snapshot only for CDDM input scale;
  elsewhere it is estimated from the loss ratio. On CDDM, differences under about 0.01 are scoring
  noise.
- **Wording risk for the abstract.** Two knob points recruit units at their floor at no R² cost:
  - Flip-flop input scale 20: 614, 614 and 913 active.
  - CDDM input scale 2: 695-720 active.

  The 2026-10-14 F2 note shows both are floor reads that fall away by the end of the budget.
  Flip-flop input scale 20 ends at 329 at 150k. CDDM input scale 2 ends at 257 at 50k, below its
  control's 362. "No standard knob fixes it" therefore holds only for the count at the end of the
  budget, not for the count at the floor. F shows the floor read.
- **Dependencies for the main cleanup.** The builder imports `fig_paper_F2`, `ff75k_readout` and
  `b5_read`. `fig_paper_F1` imports `flipflop_diversity` and `flipflop_fixedpoints`. All five, plus
  `flipflop_dropout_readout`, must survive the cleanup of main; the coordinator has been told.
