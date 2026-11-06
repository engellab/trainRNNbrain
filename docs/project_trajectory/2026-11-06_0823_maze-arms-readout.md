## 2026-11-06 08:23 — Maze arms of the uniform-interventions programme: 37 arms read out, the verdicts, the settings figure, the duplicate arm in the rules figure (issue #6)

**Issue:** engellab/trainRNNbrain#6 (Pavel's instruction of 5 Nov: report what works on every task
first). **Scripts:** `trainRNNbrain/experiments_and_analysis/read_maze_arms.py` (new),
`fig_ed_maze_settings.py` (new), `fig_ed_maze_interventions.py` (the arm derivation extended, the
duplicate arm drawn, the stopped runs recognised from their records). **Data:**
`data/read_maze_arms.json` (both tables); the package 3 arms copied from Spock at 08:20
(`Maze_interventions/{cap0.3, rwm0.05, rwm0.2, met0.01, met0.1, met1, met10, adamw, sgd0.03,
sgd0.1, sgd0.3, sgd1, redraw, zeroout}`, array 7389398: 8 of 42 runs finished, 34 still training).
**Figures:** `img/internal_figures/fig_ed_maze_settings.{pdf,svg}` (new) and
`fig_ed_maze_interventions.{pdf,svg}` (re-rendered).

### What works on the maze

Three per-unit rules work at N = 1,000: the rate penalty at 0.03 (792 to 878 active units at the
end of training against the learned controls' 426 to 449, 1.93 times their mean of 437.5; mean
success 0.984, the controls' value), the rate penalty at 0.3 (812 to 962 active, 2.06 times;
success 0.003 above the controls') and prune-and-duplicate (712 to 747 active, 1.68 times; success
0.008 below). Four fail: the rate penalty at 0.1 (seed 1 ends at a success of 0.63; the arm's mean
is 0.12 below the controls'), rate + sparsity (two seeds end below 0.9; mean 0.19 below), read-out
dropout (413 to 461 active, no seed learned, success 0.19 below) and noisy synapses (451 to 478
active, every seed above the largest control, but success 0.06 below, beyond the 0.05 tolerance).
No rescaling variant passes: the paper's values stopped at updates 52 to 60 before a checkpoint,
the slow-tilt variant has one seed collapsed to 0 active units at update 800 and two seeds at 314
and 321 active, and the 5%-growing variant ends at 231 to 255 active with no seed learned.

Among the settings, one increases the number of active units: an input-weight row norm of 0.5
(549 to 593 active, 1.31 times the controls' mean) with every seed learned and success 0.016 above
the controls'. Five decrease it: weight decay 1e-5 (370 to 417; every seed learned) and 1e-4 (335
to 378; one seed learned, success 0.20 below), activity noise 0.1 (311 to 346; no seed learned),
an initial spectral radius of 0.5 (338 to 414; every seed learned, success 0.02 below) and the
sparsity penalty alone (378 to 399; one seed learned). Six leave it inside the controls' range:
weight decay 0, input row norm 2, activity noise 0 and 0.01, spectral radius 1.5 and the
input-weight cap 0.3. Input row norms of 5 and 20 drive the network to non-finite rates: five of
the six runs were stopped by the rollback guard at updates 200 to 600 with 0 active units, and the
sixth (norm 20, seed 1) ran to update 2,000 at a success of 0.00 with 331 active units. Across the
three tasks, prune-and-duplicate now passes on all three (3BFF 2.77 times and CDDM 2.11 times from
the status board; the maze 1.68 times); read-out dropout passes on 3BFF and CDDM and fails on the
maze; noisy synapses fail the tolerance on all three.

### The read-out rules

Pre-registered: issue #5 for the rules; entry 2026-11-05_2310 and `docs/interventions_matrix.md`
(section A, settings; section B, per-unit rules) for the classes. Every value is read at the end
of training, update 2,000; the learned point (`common.maze_learned_index`: the first probe from
which the clean probe's success stays at or above 0.9 for five probes) and the count there are
reported beside it. The reference is the two learned controls of the uniform series
(`Maze_suite/N=1000_act=relu` seeds 1 and 2: 426 and 449 active, mean 437.5; success 0.99 and
0.98, mean 0.984); seed 3 ended at a success of 0.73 and is excluded, as issue #3's
pre-registration excludes a run that does not learn. A rule works when every seed's count exceeds
449 and the arm's mean success is within 0.05 of 0.984; otherwise it fails, with the reason. A
setting increases or decreases the count when all three seeds lie above 449 or below 426; otherwise
there is no change; the success difference and the number of seeds that learned are reported with
it. An arm with fewer than three finished runs gets no verdict ("incomplete", with the numbers
finished, training and stopped).

### Table 2: one row per arm (the launcher's order)

| arm | class | runs finished / training / stopped | active at the end (seeds 1, 2, 3) | fold over 437.5 | success minus controls' 0.984 | seeds learned | verdict |
|---|---|---|---|---|---|---|---|
| control | control | 3 / 0 / 0 | 426, 449, 415 | 0.98 | -0.086 | 2 of 3 | reference: 2 learned controls, mean 437.5 active (426-449), success 0.984 |
| frm0.03 | rule | 3 / 0 / 0 | 878, 862, 792 | 1.93 | +0.000 | 3 of 3 | works |
| frm0.1 | rule | 3 / 0 / 0 | 763, 927, 982 | 2.04 | -0.120 | 2 of 3 | fails: success cost -0.120 beyond 0.05 |
| frm0.3 | rule | 3 / 0 / 0 | 962, 930, 812 | 2.06 | +0.003 | 3 of 3 | works |
| frm0.1+rws0.05 | rule | 3 / 0 / 0 | 901, 907, 946 | 2.10 | -0.193 | 1 of 3 | fails: success cost -0.193 beyond 0.05 |
| synnoise | rule | 3 / 0 / 0 | 451, 478, 454 | 1.05 | -0.062 | 2 of 3 | fails: success cost -0.062 beyond 0.05 |
| dropout | rule | 3 / 0 / 0 | 413, 461, 436 | 1.00 | -0.190 | 0 of 3 | fails: count not above every learned control's (449) in seeds 1, 3; success cost -0.190 beyond 0.05 |
| rescale | rule | 0 / 0 / 3 | - | - | - | - | no verdict (stopped) |
| rescale_mild | rule | 2 / 0 / 1 | s2 314, s3 321 | 0.73 | -0.293 | 0 of 2 | incomplete (2 of 3 finished, 1 stopped) |
| rescale_few | rule | 3 / 0 / 0 | 255, 231, 234 | 0.55 | -0.245 | 0 of 3 | fails: count not above every learned control's (449) in seeds 1, 2, 3; success cost -0.245 beyond 0.05 |
| wd0 | setting | 3 / 0 / 0 | 404, 452, 432 | 0.98 | +0.008 | 3 of 3 | no change in the count (404, 452, 432 against the controls' 426-449); success +0.008 from the controls' mean; all seeds learned |
| wd1e-5 | setting | 3 / 0 / 0 | 402, 417, 370 | 0.91 | -0.016 | 3 of 3 | decreases the count (402, 417, 370 against the controls' 426-449); success -0.016 from the controls' mean; all seeds learned |
| wd1e-4 | setting | 3 / 0 / 0 | 335, 341, 378 | 0.80 | -0.201 | 1 of 3 | decreases the count (335, 341, 378 against the controls' 426-449); success -0.201 from the controls' mean; 1 of 3 seeds learned |
| inorm0.5 | setting | 3 / 0 / 0 | 577, 593, 549 | 1.31 | +0.016 | 3 of 3 | increases the count (577, 593, 549 against the controls' 426-449); success +0.016 from the controls' mean; all seeds learned |
| inorm2 | setting | 3 / 0 / 0 | 290, 443, 357 | 0.83 | -0.094 | 2 of 3 | no change in the count (290, 443, 357 against the controls' 426-449); success -0.094 from the controls' mean; 2 of 3 seeds learned |
| inorm5 | setting | 0 / 0 / 3 | - | - | - | - | no verdict (stopped) |
| inorm20 | setting | 1 / 0 / 2 | 331 | 0.76 | -0.984 | 0 of 1 | incomplete (1 of 3 finished, 2 stopped) |
| duplicate | rule | 3 / 0 / 0 | 747, 741, 712 | 1.68 | -0.008 | 3 of 3 | works |
| srec0 | setting | 3 / 0 / 0 | 440, 425, 400 | 0.96 | +0.016 | 3 of 3 | no change in the count (440, 425, 400 against the controls' 426-449); success +0.016 from the controls' mean; all seeds learned |
| srec0.01 | setting | 3 / 0 / 0 | 456, 380, 436 | 0.97 | +0.016 | 3 of 3 | no change in the count (456, 380, 436 against the controls' 426-449); success +0.016 from the controls' mean; all seeds learned |
| srec0.1 | setting | 3 / 0 / 0 | 346, 317, 311 | 0.74 | -0.271 | 0 of 3 | decreases the count (346, 317, 311 against the controls' 426-449); success -0.271 from the controls' mean; no seed learned |
| srad0.5 | setting | 3 / 0 / 0 | 338, 393, 414 | 0.87 | -0.021 | 3 of 3 | decreases the count (338, 393, 414 against the controls' 426-449); success -0.021 from the controls' mean; all seeds learned |
| srad1.5 | setting | 3 / 0 / 0 | 350, 471, 319 | 0.87 | -0.062 | 2 of 3 | no change in the count (350, 471, 319 against the controls' 426-449); success -0.062 from the controls' mean; 2 of 3 seeds learned |
| rws0.05 | setting | 3 / 0 / 0 | 399, 378, 386 | 0.89 | -0.094 | 1 of 3 | decreases the count (399, 378, 386 against the controls' 426-449); success -0.094 from the controls' mean; 1 of 3 seeds learned |
| cap0.3 | setting | 3 / 0 / 0 | 434, 463, 380 | 0.97 | +0.003 | 3 of 3 | no change in the count (434, 463, 380 against the controls' 426-449); success +0.003 from the controls' mean; all seeds learned |
| rwm0.05 | setting | 2 / 1 / 0 | s1 481, s3 394 | 1.00 | -0.047 | 2 of 2 | incomplete (2 of 3 finished, 1 training) |
| rwm0.2 | setting | 1 / 2 / 0 | 517 | 1.18 | -0.031 | 1 of 1 | incomplete (1 of 3 finished, 2 training) |
| met0.01 | setting | 1 / 2 / 0 | s3 379 | 0.87 | +0.008 | 1 of 1 | incomplete (1 of 3 finished, 2 training) |
| met0.1 | setting | 1 / 2 / 0 | 431 | 0.99 | +0.008 | 1 of 1 | incomplete (1 of 3 finished, 2 training) |
| met1 | setting | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| met10 | setting | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| adamw | setting | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| sgd0.03 | setting | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| sgd0.1 | setting | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| sgd0.3 | setting | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| sgd1 | setting | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| redraw | rule | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |
| zeroout | rule | 0 / 3 / 0 | - | - | - | - | no verdict (unfinished) |

### The arm derivation, and a label that never reached a commit

Every arm is derived from the run's saved config (`fig_ed_maze_interventions.arm_of`), never from
the folder name; the folder is printed beside the key, and no folder disagreed with its config.
`arm_of` now reads `trainer.prune_args.reinit_mode`. Before this change it keyed every
`prune_reinit` run by its rescaling values alone, so a duplicate run (mode copy, whose prune_args
keep the config's rescale defaults alpha 1.002, cap 64, protect 0.2) was keyed "rescale": the old
script on today's tree reports "rescale works, 1.68x", which is the duplicate arm's result. The
duplicate folder reached the Mac at 04:06 today, after the figure's last commit (c9809b6, 22:26 on
5 Nov), so no committed figure or entry carried the wrong label. `arm_of` also reads the settings
(`trainer.weight_decay` against the default 1e-6, `model.input_row_norm`, `model.sigma_rec`
against 0.05, `model.spectral_rad` against 1.2, `lambda_rws` alone, `model.inp_weight_cap`,
`trainer.lambda_rwm`, `trainer.lambda_met`, `trainer.optimizer` with `trainer.lr`); a config with
several options on gets their keys joined by "+" ("frm0.1+rws0.05"). A run the rollback guard
stopped is recognised from its records (`stopped_by_guard`: an unfinished run whose last probe
counts 0 active units) in place of the hand-kept `STOPPED_RUNS` table; the six known cases (Spock
logs MazeInterv.7384816_25, 7389335_46-48, 7389335_50-51) match and no running run does.

### Table 1: one row per run

| class | arm (from config) | folder | seed | status | last update | success at the end | learned at | active at learned | active at the end | fold | options |
|---|---|---|---|---|---|---|---|---|---|---|---|
| control | control | Maze_suite/N=1000_act=relu | 1 | finished | 2,000 | 0.99 | 1050 | 451 | 426 | 0.97 | no option on |
| control | control | Maze_suite/N=1000_act=relu | 2 | finished | 2,000 | 0.98 | 760 | 517 | 449 | 1.03 | no option on |
| control | control | Maze_suite/N=1000_act=relu | 3 | finished | 2,000 | 0.73 | - | - | 415 | 0.95 | no option on |
| rule | frm0.03 | frm0.03 | 1 | finished | 2,000 | 1.00 | 1040 | 936 | 878 | 2.01 | lambda_frm 0.03 |
| rule | frm0.03 | frm0.03 | 2 | finished | 2,000 | 0.98 | 1030 | 877 | 862 | 1.97 | lambda_frm 0.03 |
| rule | frm0.03 | frm0.03 | 3 | finished | 2,000 | 0.97 | 1450 | 783 | 792 | 1.81 | lambda_frm 0.03 |
| rule | frm0.1 | frm0.1 | 1 | finished | 2,000 | 0.63 | - | - | 763 | 1.74 | lambda_frm 0.1 |
| rule | frm0.1 | frm0.1 | 2 | finished | 2,000 | 0.97 | 570 | 953 | 927 | 2.12 | lambda_frm 0.1 |
| rule | frm0.1 | frm0.1 | 3 | finished | 2,000 | 0.99 | 1450 | 941 | 982 | 2.24 | lambda_frm 0.1 |
| rule | frm0.3 | frm0.3 | 1 | finished | 2,000 | 0.96 | 1100 | 958 | 962 | 2.20 | lambda_frm 0.3 |
| rule | frm0.3 | frm0.3 | 2 | finished | 2,000 | 1.00 | 850 | 973 | 930 | 2.13 | lambda_frm 0.3 |
| rule | frm0.3 | frm0.3 | 3 | finished | 2,000 | 1.00 | 720 | 985 | 812 | 1.86 | lambda_frm 0.3 |
| rule | frm0.1+rws0.05 | frm0.1_rws0.05 | 1 | finished | 2,000 | 1.00 | 1610 | 950 | 901 | 2.06 | lambda_frm 0.1, lambda_rws 0.05 |
| rule | frm0.1+rws0.05 | frm0.1_rws0.05 | 2 | finished | 2,000 | 0.43 | - | - | 907 | 2.07 | lambda_frm 0.1, lambda_rws 0.05 |
| rule | frm0.1+rws0.05 | frm0.1_rws0.05 | 3 | finished | 2,000 | 0.95 | - | - | 946 | 2.16 | lambda_frm 0.1, lambda_rws 0.05 |
| rule | synnoise | sigmaw1.0 | 1 | finished | 2,000 | 0.91 | 1950 | 456 | 451 | 1.03 | sigma_w 1 |
| rule | synnoise | sigmaw1.0 | 2 | finished | 2,000 | 0.96 | 1230 | 458 | 478 | 1.09 | sigma_w 1 |
| rule | synnoise | sigmaw1.0 | 3 | finished | 2,000 | 0.90 | - | - | 454 | 1.04 | sigma_w 1 |
| rule | dropout | dropout0.2_beta4 | 1 | finished | 2,000 | 0.84 | - | - | 413 | 0.94 | dropout mute rate 0.2 beta 4.0 |
| rule | dropout | dropout0.2_beta4 | 2 | finished | 2,000 | 0.79 | - | - | 461 | 1.05 | dropout mute rate 0.2 beta 4.0 |
| rule | dropout | dropout0.2_beta4 | 3 | finished | 2,000 | 0.76 | - | - | 436 | 1.00 | dropout mute rate 0.2 beta 4.0 |
| rule | rescale_mild | rescale_mild | 1 | stopped@800 | 800 | - | - | - | - | - | rescaling alpha 1.0005 cap 8 protect 0.2 |
| rule | rescale_mild | rescale_mild | 2 | finished | 2,000 | 0.75 | - | - | 314 | 0.72 | rescaling alpha 1.0005 cap 8 protect 0.2 |
| rule | rescale_mild | rescale_mild | 3 | finished | 2,000 | 0.63 | - | - | 321 | 0.73 | rescaling alpha 1.0005 cap 8 protect 0.2 |
| rule | rescale_few | rescale_few | 1 | finished | 2,000 | 0.66 | - | - | 255 | 0.58 | rescaling alpha 1.002 cap 64 protect 0.05 |
| rule | rescale_few | rescale_few | 2 | finished | 2,000 | 0.75 | - | - | 231 | 0.53 | rescaling alpha 1.002 cap 64 protect 0.05 |
| rule | rescale_few | rescale_few | 3 | finished | 2,000 | 0.81 | - | - | 234 | 0.53 | rescaling alpha 1.002 cap 64 protect 0.05 |
| setting | wd0 | wd0 | 1 | finished | 2,000 | 0.98 | 1650 | 408 | 404 | 0.92 | weight_decay 0 |
| setting | wd0 | wd0 | 2 | finished | 2,000 | 1.00 | 900 | 514 | 452 | 1.03 | weight_decay 0 |
| setting | wd0 | wd0 | 3 | finished | 2,000 | 0.99 | 920 | 466 | 432 | 0.99 | weight_decay 0 |
| setting | wd1e-5 | wd1e-5 | 1 | finished | 2,000 | 0.93 | 1800 | 403 | 402 | 0.92 | weight_decay 1e-05 |
| setting | wd1e-5 | wd1e-5 | 2 | finished | 2,000 | 0.99 | 1790 | 413 | 417 | 0.95 | weight_decay 1e-05 |
| setting | wd1e-5 | wd1e-5 | 3 | finished | 2,000 | 0.98 | 1320 | 390 | 370 | 0.85 | weight_decay 1e-05 |
| setting | wd1e-4 | wd1e-4 | 1 | finished | 2,000 | 0.64 | - | - | 335 | 0.77 | weight_decay 0.0001 |
| setting | wd1e-4 | wd1e-4 | 2 | finished | 2,000 | 0.71 | - | - | 341 | 0.78 | weight_decay 0.0001 |
| setting | wd1e-4 | wd1e-4 | 3 | finished | 2,000 | 1.00 | 1320 | 370 | 378 | 0.86 | weight_decay 0.0001 |
| setting | inorm0.5 | inorm0.5 | 1 | finished | 2,000 | 1.00 | 310 | 650 | 577 | 1.32 | input_row_norm 0.5 |
| setting | inorm0.5 | inorm0.5 | 2 | finished | 2,000 | 1.00 | 140 | 648 | 593 | 1.36 | input_row_norm 0.5 |
| setting | inorm0.5 | inorm0.5 | 3 | finished | 2,000 | 1.00 | 470 | 575 | 549 | 1.25 | input_row_norm 0.5 |
| setting | inorm2 | inorm2 | 1 | finished | 2,000 | 0.68 | - | - | 290 | 0.66 | input_row_norm 2 |
| setting | inorm2 | inorm2 | 2 | finished | 2,000 | 1.00 | 140 | 449 | 443 | 1.01 | input_row_norm 2 |
| setting | inorm2 | inorm2 | 3 | finished | 2,000 | 0.99 | 1230 | 342 | 357 | 0.82 | input_row_norm 2 |
| setting | inorm5 | inorm5 | 1 | stopped@500 | 500 | - | - | - | - | - | input_row_norm 5 |
| setting | inorm5 | inorm5 | 2 | stopped@600 | 600 | - | - | - | - | - | input_row_norm 5 |
| setting | inorm5 | inorm5 | 3 | stopped@300 | 300 | - | - | - | - | - | input_row_norm 5 |
| setting | inorm20 | inorm20 | 1 | finished | 2,000 | 0.00 | - | - | 331 | 0.76 | input_row_norm 20 |
| setting | inorm20 | inorm20 | 2 | stopped@200 | 200 | - | - | - | - | - | input_row_norm 20 |
| setting | inorm20 | inorm20 | 3 | stopped@400 | 400 | - | - | - | - | - | input_row_norm 20 |
| rule | duplicate | duplicate | 1 | finished | 2,000 | 0.98 | 1060 | 695 | 747 | 1.71 | prune_reinit copy rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| rule | duplicate | duplicate | 2 | finished | 2,000 | 0.95 | 930 | 677 | 741 | 1.69 | prune_reinit copy rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| rule | duplicate | duplicate | 3 | finished | 2,000 | 0.99 | 840 | 683 | 712 | 1.63 | prune_reinit copy rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| setting | srec0 | srec0 | 1 | finished | 2,000 | 1.00 | 270 | 629 | 440 | 1.01 | sigma_rec 0 |
| setting | srec0 | srec0 | 2 | finished | 2,000 | 1.00 | 350 | 596 | 425 | 0.97 | sigma_rec 0 |
| setting | srec0 | srec0 | 3 | finished | 2,000 | 1.00 | 290 | 606 | 400 | 0.91 | sigma_rec 0 |
| setting | srec0.01 | srec0.01 | 1 | finished | 2,000 | 1.00 | 460 | 566 | 456 | 1.04 | sigma_rec 0.01 |
| setting | srec0.01 | srec0.01 | 2 | finished | 2,000 | 1.00 | 560 | 563 | 380 | 0.87 | sigma_rec 0.01 |
| setting | srec0.01 | srec0.01 | 3 | finished | 2,000 | 1.00 | 520 | 567 | 436 | 1.00 | sigma_rec 0.01 |
| setting | srec0.1 | srec0.1 | 1 | finished | 2,000 | 0.68 | - | - | 346 | 0.79 | sigma_rec 0.1 |
| setting | srec0.1 | srec0.1 | 2 | finished | 2,000 | 0.61 | - | - | 317 | 0.72 | sigma_rec 0.1 |
| setting | srec0.1 | srec0.1 | 3 | finished | 2,000 | 0.85 | - | - | 311 | 0.71 | sigma_rec 0.1 |
| setting | srad0.5 | srad0.5 | 1 | finished | 2,000 | 0.92 | 1960 | 339 | 338 | 0.77 | spectral_rad 0.5 |
| setting | srad0.5 | srad0.5 | 2 | finished | 2,000 | 0.99 | 1230 | 443 | 393 | 0.90 | spectral_rad 0.5 |
| setting | srad0.5 | srad0.5 | 3 | finished | 2,000 | 0.98 | 910 | 437 | 414 | 0.95 | spectral_rad 0.5 |
| setting | srad1.5 | srad1.5 | 1 | finished | 2,000 | 0.92 | 1590 | 369 | 350 | 0.80 | spectral_rad 1.5 |
| setting | srad1.5 | srad1.5 | 2 | finished | 2,000 | 0.97 | 1240 | 525 | 471 | 1.08 | spectral_rad 1.5 |
| setting | srad1.5 | srad1.5 | 3 | finished | 2,000 | 0.88 | - | - | 319 | 0.73 | spectral_rad 1.5 |
| setting | rws0.05 | rws0.05 | 1 | finished | 2,000 | 1.00 | 1190 | 447 | 399 | 0.91 | lambda_rws 0.05 |
| setting | rws0.05 | rws0.05 | 2 | finished | 2,000 | 0.82 | - | - | 378 | 0.86 | lambda_rws 0.05 |
| setting | rws0.05 | rws0.05 | 3 | finished | 2,000 | 0.85 | - | - | 386 | 0.88 | lambda_rws 0.05 |
| setting | cap0.3 | cap0.3 | 1 | finished | 2,000 | 0.98 | 1110 | 495 | 434 | 0.99 | inp_weight_cap 0.3 |
| setting | cap0.3 | cap0.3 | 2 | finished | 2,000 | 0.98 | 1400 | 487 | 463 | 1.06 | inp_weight_cap 0.3 |
| setting | cap0.3 | cap0.3 | 3 | finished | 2,000 | 1.00 | 1090 | 394 | 380 | 0.87 | inp_weight_cap 0.3 |
| setting | rwm0.05 | rwm0.05 | 1 | finished | 2,000 | 0.93 | 1170 | 431 | 481 | 1.10 | lambda_rwm 0.05 |
| setting | rwm0.05 | rwm0.05 | 2 | unfinished | 1,900 | - | - | - | - | - | lambda_rwm 0.05 |
| setting | rwm0.05 | rwm0.05 | 3 | finished | 2,000 | 0.95 | 1420 | 409 | 394 | 0.90 | lambda_rwm 0.05 |
| setting | rwm0.2 | rwm0.2 | 1 | finished | 2,000 | 0.95 | 790 | 505 | 517 | 1.18 | lambda_rwm 0.2 |
| setting | rwm0.2 | rwm0.2 | 2 | unfinished | 1,900 | - | - | - | - | - | lambda_rwm 0.2 |
| setting | rwm0.2 | rwm0.2 | 3 | unfinished | 1,800 | - | - | - | - | - | lambda_rwm 0.2 |
| setting | met0.01 | met0.01 | 1 | unfinished | 1,800 | - | - | - | - | - | lambda_met 0.01 |
| setting | met0.01 | met0.01 | 2 | unfinished | 1,600 | - | - | - | - | - | lambda_met 0.01 |
| setting | met0.01 | met0.01 | 3 | finished | 2,000 | 0.99 | 1220 | 377 | 379 | 0.87 | lambda_met 0.01 |
| setting | met0.1 | met0.1 | 1 | finished | 2,000 | 0.99 | 1120 | 468 | 431 | 0.99 | lambda_met 0.1 |
| setting | met0.1 | met0.1 | 2 | unfinished | 1,600 | - | - | - | - | - | lambda_met 0.1 |
| setting | met0.1 | met0.1 | 3 | unfinished | 1,500 | - | - | - | - | - | lambda_met 0.1 |
| setting | met1 | met1 | 1 | unfinished | 1,500 | - | - | - | - | - | lambda_met 1 |
| setting | met1 | met1 | 2 | unfinished | 1,600 | - | - | - | - | - | lambda_met 1 |
| setting | met1 | met1 | 3 | unfinished | 1,500 | - | - | - | - | - | lambda_met 1 |
| setting | met10 | met10 | 1 | unfinished | 1,500 | - | - | - | - | - | lambda_met 10 |
| setting | met10 | met10 | 2 | unfinished | 1,400 | - | - | - | - | - | lambda_met 10 |
| setting | met10 | met10 | 3 | unfinished | 1,500 | - | - | - | - | - | lambda_met 10 |
| setting | adamw | adamw | 1 | unfinished | 1,400 | - | - | - | - | - | optimizer adamw lr 0.001 |
| setting | adamw | adamw | 2 | unfinished | 1,400 | - | - | - | - | - | optimizer adamw lr 0.001 |
| setting | adamw | adamw | 3 | unfinished | 1,400 | - | - | - | - | - | optimizer adamw lr 0.001 |
| setting | sgd0.03 | sgd0.03 | 1 | unfinished | 1,400 | - | - | - | - | - | optimizer sgd lr 0.03 momentum 0.0 |
| setting | sgd0.03 | sgd0.03 | 2 | unfinished | 1,300 | - | - | - | - | - | optimizer sgd lr 0.03 momentum 0.0 |
| setting | sgd0.03 | sgd0.03 | 3 | unfinished | 1,400 | - | - | - | - | - | optimizer sgd lr 0.03 momentum 0.0 |
| setting | sgd0.1 | sgd0.1 | 1 | unfinished | 1,300 | - | - | - | - | - | optimizer sgd lr 0.1 momentum 0.0 |
| setting | sgd0.1 | sgd0.1 | 2 | unfinished | 1,300 | - | - | - | - | - | optimizer sgd lr 0.1 momentum 0.0 |
| setting | sgd0.1 | sgd0.1 | 3 | unfinished | 1,300 | - | - | - | - | - | optimizer sgd lr 0.1 momentum 0.0 |
| setting | sgd0.3 | sgd0.3 | 1 | unfinished | 1,300 | - | - | - | - | - | optimizer sgd lr 0.3 momentum 0.0 |
| setting | sgd0.3 | sgd0.3 | 2 | unfinished | 1,300 | - | - | - | - | - | optimizer sgd lr 0.3 momentum 0.0 |
| setting | sgd0.3 | sgd0.3 | 3 | unfinished | 1,200 | - | - | - | - | - | optimizer sgd lr 0.3 momentum 0.0 |
| setting | sgd1 | sgd1 | 1 | unfinished | 1,200 | - | - | - | - | - | optimizer sgd lr 1 momentum 0.0 |
| setting | sgd1 | sgd1 | 2 | unfinished | 1,100 | - | - | - | - | - | optimizer sgd lr 1 momentum 0.0 |
| setting | sgd1 | sgd1 | 3 | unfinished | 1,100 | - | - | - | - | - | optimizer sgd lr 1 momentum 0.0 |
| rule | redraw | redraw | 1 | unfinished | 1,300 | - | - | - | - | - | prune_reinit random rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| rule | redraw | redraw | 2 | unfinished | 1,100 | - | - | - | - | - | prune_reinit random rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| rule | redraw | redraw | 3 | unfinished | 1,200 | - | - | - | - | - | prune_reinit random rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| rule | zeroout | zeroout | 1 | unfinished | 1,100 | - | - | - | - | - | prune_reinit zero_out rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| rule | zeroout | zeroout | 2 | unfinished | 1,200 | - | - | - | - | - | prune_reinit zero_out rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| rule | zeroout | zeroout | 3 | unfinished | 1,000 | - | - | - | - | - | prune_reinit zero_out rate 0.025 maturity 1000 utility contribution every 100 patience 3 |
| ? | ? | rescale | 1 | stopped (no run files) | - | - | - | - | - | - | the folder holds no checkpoint: stopped at updates 52-60 (entry 2026-11-05_2226) |
| ? | ? | rescale | 2 | stopped (no run files) | - | - | - | - | - | - | the folder holds no checkpoint: stopped at updates 52-60 (entry 2026-11-05_2226) |
| ? | ? | rescale | 3 | stopped (no run files) | - | - | - | - | - | - | the folder holds no checkpoint: stopped at updates 52-60 (entry 2026-11-05_2226) |

### Figures

`fig_ed_maze_settings.py` (new): the maze counterpart of Figure 3a, 165.1 x 133 mm, four axes in
two blocks. (a, b) weight decay, input-weight row norm, activity noise, initial spectral radius and
the sparsity penalty alone; (c, d) the input-weight cap, the recurrent-weight magnitude penalty,
the metabolic penalty, AdamW, SGD at four learning rates and the two re-initialisation modes; each
block starts with the controls. The upper row of a block is the number of active units at the end:
one open marker per run (an open marker is a fixed iteration, paperstyle's rule), a bar at the mean
once all three runs finished, a cross for a run that did not learn, a cross at 0 with "stopped" for
a run the guard stopped, "unfinished" in a column with no finished run and "k of 3" where some runs
are still training; the learned controls' range (426 to 449) is a band and their mean a line. The
lower row is the success at the end, the 0.9 level dashed. Colours: the controls grey, a setting's
values on the teal setting ramp, light to dark in ladder order (Figure 3's convention). Printout:

```
controls' reference: 2 learned, active 437.5 (426-449), success 0.984
     control: s1 426 0.99; s2 449 0.98; s3 415 0.73 (not learned)
         wd0: s1 404 0.98; s2 452 1.00; s3 432 0.99
      wd1e-5: s1 402 0.93; s2 417 0.99; s3 370 0.98
      wd1e-4: s1 335 0.64 (not learned); s2 341 0.71 (not learned); s3 378 1.00
    inorm0.5: s1 577 1.00; s2 593 1.00; s3 549 1.00
      inorm2: s1 290 0.68 (not learned); s2 443 1.00; s3 357 0.99
      inorm5: s1 stopped@500; s2 stopped@600; s3 stopped@300
     inorm20: s1 331 0.00 (not learned); s2 stopped@200; s3 stopped@400
       srec0: s1 440 1.00; s2 425 1.00; s3 400 1.00
    srec0.01: s1 456 1.00; s2 380 1.00; s3 436 1.00
     srec0.1: s1 346 0.68 (not learned); s2 317 0.61 (not learned); s3 311 0.85 (not learned)
     srad0.5: s1 338 0.92; s2 393 0.99; s3 414 0.98
     srad1.5: s1 350 0.92; s2 471 0.97; s3 319 0.88 (not learned)
     rws0.05: s1 399 1.00; s2 378 0.82 (not learned); s3 386 0.85 (not learned)
     control: s1 426 0.99; s2 449 0.98; s3 415 0.73 (not learned)
      cap0.3: s1 434 0.98; s2 463 0.98; s3 380 1.00
     rwm0.05: s1 481 0.93; s3 394 0.95; 1 training
      rwm0.2: s1 517 0.95; 2 training
     met0.01: s3 379 0.99; 2 training
      met0.1: s1 431 0.99; 2 training
        met1: 3 training (at updates 1500, 1600, 1500)
       met10: 3 training (at updates 1500, 1400, 1500)
       adamw: 3 training (at updates 1400, 1400, 1400)
     sgd0.03: 3 training (at updates 1400, 1300, 1400)
      sgd0.1: 3 training (at updates 1300, 1300, 1300)
      sgd0.3: 3 training (at updates 1300, 1300, 1200)
        sgd1: 3 training (at updates 1200, 1100, 1100)
      redraw: 3 training (at updates 1300, 1100, 1200)
     zeroout: 3 training (at updates 1100, 1200, 1000)
  .. no overlapping text (78 text artists checked)
wrote /Users/pt1290/Documents/GitHub/trainRNNbrain/trainRNNbrain/experiments_and_analysis/../../img/internal_figures/fig_ed_maze_settings.pdf (+ .svg)
```

`fig_ed_maze_interventions.py`: the prune-and-duplicate arm joins the ladder (Figure 5's colour
`ps.DUPLICATE` and marker ^), so panel c has ten columns (90 mm, 9 mm each) and the key three rows
of four arms; the canvas is 113.5 mm tall (was 110). Panel d is 2.5 mm narrower so its "0.0" tick
label clears the last column label. Printout:

```
            arm     status  n          fold per seed mean fold succ cost  verdict (count above every learned control's in all seeds AND mean success within 0.05 of the controls')
        frm0.03   finished  3       2.01, 1.97, 1.81      1.93    +0.000  works
         frm0.1   finished  3       1.74, 2.12, 2.24      2.04    -0.120  fails: success cost -0.120 beyond 0.05
         frm0.3   finished  3       2.20, 2.13, 1.86      2.06    +0.003  works
 frm0.1+rws0.05   finished  3       2.06, 2.07, 2.16      2.10    -0.193  fails: success cost -0.193 beyond 0.05
        dropout   finished  3       0.94, 1.05, 1.00      1.00    -0.190  fails: count not above every learned control's (449) in seeds 1, 3; success cost -0.190 beyond 0.05
       synnoise   finished  3       1.03, 1.09, 1.04      1.05    -0.062  fails: success cost -0.062 beyond 0.05
        rescale    stopped  0                      -         -         -  no verdict (stopped)
    rescale_few   finished  3       0.58, 0.53, 0.53      0.55    -0.245  fails: count not above every learned control's (449) in seeds 1, 2, 3; success cost -0.245 beyond 0.05
   rescale_mild   finished  2             0.72, 0.73      0.73    -0.293  incomplete (2 of 3 runs finished)
      duplicate   finished  3       1.71, 1.69, 1.63      1.68    -0.008  works
5 technique ablation(s) under /Users/pt1290/Documents/GitHub/trainRNNbrain/data/rl_maze_ablation_interventions; control drawn: s1

Panel a: 29 runs drawn; panel b: 29 markers
Panel d: 
  !! panel b: synnoise s1 and synnoise s3 print 1.1 pt apart (marker 3.0 pt)
  .. no overlapping text (64 text artists checked)
wrote /Users/pt1290/Documents/GitHub/trainRNNbrain/trainRNNbrain/experiments_and_analysis/../../img/internal_figures/fig_ed_maze_interventions.pdf (+ .svg)
```

Both PDFs pass `tests/test_paperstyle.py` (Helvetica and Helvetica-Bold only, panel letters
8.5 pt, every y label at least 1.36 mm from its ticks) and `pdfimages -list` lists no raster.
`tests/test_no_function_imports.py`: 206 files; `tests/test_scripts_start.py`: 55 scripts start.

### Still training, and how to pick it up

At 08:20, 34 of the 42 runs of array 7389398 were at updates 1,000 to 1,600 of 2,000 (8 to 10 s
per update, so the last ones land about 2.5 h later): rwm0.05 seed 2; rwm0.2 seeds 2 and 3;
met0.01 seeds 1 and 2; met0.1 seeds 2 and 3; met1, met10, adamw, sgd0.03, sgd0.1, sgd0.3, sgd1,
redraw and zeroout (three seeds each). Their verdicts wait: whether the recurrent-weight and
metabolic penalties move the count (the finished runs sit at 481 and 394 active for rwm 0.05, 517
for rwm 0.2, 379 for met 0.01 and 431 for met 0.1, all inside or near the controls' range), whether
the maze reproduces 3BFF's optimiser pattern (AdamW and no weight decay keep more units at no
loss; SGD keeps more only at lower performance), and whether re-initialisation works by the rule
criterion. To pick them up: copy each arm (`rsync -a --exclude 'PolicyState.pt' --exclude
'*AdamState*' spock:/usr/people/pt1290/trainRNNbrain/data/trained_RNNs_rl_maze/Maze_interventions/<arm>/
data/trained_RNNs_rl_maze/Maze_interventions/<arm>/`), then run `read_maze_arms.py`,
`fig_ed_maze_settings.py` and `fig_ed_maze_interventions.py`; every status, verdict and column
follows the files on disk, and a run copied while training shows as "unfinished" at its last
update.

### Files

- `trainRNNbrain/experiments_and_analysis/read_maze_arms.py`: both tables and the verdicts;
  imports `load_records`, `control_reference` and `rule_verdict` from `fig_ed_maze_interventions`;
  `setting_verdict` and the class rule (`RULE_PREFIXES`) are its own; writes `data/read_maze_arms.json`.
- `trainRNNbrain/experiments_and_analysis/fig_ed_maze_settings.py`: the settings figure.
- `trainRNNbrain/experiments_and_analysis/fig_ed_maze_interventions.py`: `arm_of` extended and
  `fmt_value` (1e-05 printed as the folders write it, "1e-5"), `FOLDER_OF` (the launcher's arm
  names, for the mismatch check and the printouts' order), `stopped_by_guard`, `arm_status` and
  `rule_verdict` factored out of `arm_summaries`, `control_reference` gains `active_min`, the
  duplicate arm in `ARMS`, `SHORT` and the layout.
- `trainRNNbrain/experiments_and_analysis/README.md`: a row for the three scripts.
- `docs/paper_audit.md`: the C6 paragraph (the duplicate verdict, the maze settings, what still
  trains).
