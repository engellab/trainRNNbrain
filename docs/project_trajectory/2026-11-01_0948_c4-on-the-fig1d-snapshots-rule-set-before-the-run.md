## 2026-11-01 09:48 — C4 recomputed on the Figure 1d snapshots: rule set before the run (dead-relu-paper#91)

**Why.** Figure 4a-c and the C4 numbers in Results 4 were measured (entry
`2026-10-31_2207_c4-at-the-current-floor-passes-its-rule-figure-4-built.md`, Della job 16566988) on
the first kept snapshot at or after each network's floor crossing: CDDM 25,500, 26,500, 24,500 and
3BFF 67,500, 50,500, 58,500 iterations. Figure 1d, the removal analysis (Methods M06) and Extended
Data Fig. 2 (population statistics) use the nearest kept snapshot to the same crossings: CDDM 25,000,
26,000, 24,000 and 3BFF 67,000, 50,000, 58,000 (`data/pop_stats_nets/manifest.tsv`). The pipeline
crossings (`fig_paper_F2_size_collect.pkl['ablation']`, `t_star`) are CDDM 25,020, 26,028, 24,219 and
3BFF 67,191, 50,073, 58,035, so the Figure 4 weights were 281-480 iterations after the crossing and
the Figure 1d weights 20-219 before it. The Methods call both "at the floor crossing". The committee
counted silent units on the Figure 1d weights with its own numpy simulation and got 310, 288 and 253
against the paper's 260-313.

**What runs.** On the Mac (Della refuses the login today), from trainRNNbrain branch `wp/fig4-r3`:
`fig_paper_F4_noreturn.py fig1d`, which for each of the six networks
1. runs `c4_gradient_mechanism.run_one` (unchanged measurement code; it now accepts the weights file
   and iteration directly instead of finding them with `read_point`) on the `pop_stats_nets` weights,
   K = 10 training-noise batches, output `data/fig4_noreturn/c4_fig1d/`;
2. keeps the noise-free x(t) of the first 16 reference trials of those weights for panel a
   (`traces_fig1d/`);
3. recomputes participation on the full reference batch at the Figure 1d snapshot and at the two later
   snapshots kept on the Mac, which are 500 and 5,500 iterations after it (the Della read's `d0` and
   `d5000` copies in `snap/`), output `activity_fig1d/`.
Classes and thresholds are those of entry 2026-10-31_2137 (active, silent, low-active, reached at 5%
of q95 of g_in; noise-free reference batch, task seed 12345).

**Rule, written before the run.**
- *Oracle checks (block, not verdict).* (a) CDDM active counts equal the removal file's counts on the
  same weights (manifest column 5: 430, 416, 438) exactly; 3BFF active counts within 5 units of 267,
  342, 287 (Figure 1d's 3BFF batch is not the C4 reference batch). (b) CDDM silent counts equal the
  committee's independent count (310, 288, 253) exactly. (c) The full-batch participation pass at the
  Figure 1d snapshot reproduces the C4 classes in every unit. (d) Participation recomputed on the
  `d0` copies (25,500 etc.) reproduces the stored classes of job 16566988 in every unit, which
  confirms those copies are the snapshots 500 iterations later. Any failure is traced to its cause
  before a number is used.
- *Claim.* The C4 statement stands on the Figure 1d snapshots if all four clauses of entry
  2026-10-31_2137 hold, with "5,000 iterations later" read as 5,500: every 3BFF network has 0 silent
  units; every CDDM network has at least 200 silent units with largest noise-free g_in exactly 0;
  every 3BFF network has at least 100 reached low-active units; in every 3BFF network fewer than 10%
  of the reached low-active units are active 5,500 iterations later.
- If a clause fails, the paper reports the measured numbers and the statement is rewritten to match;
  nothing is re-run with other settings.

**Cost.** Six networks on the Mac CPU, expected under an hour.
