## 2026-11-01 07:16 — Figure 1 and Extended Data Fig. 1 redrawn for round 2; decision #20 applied

Work package fig1-r2 (ptolmachev/dead-relu-paper#52), trainRNNbrain branch `wp/fig1-r2` (from main
e28b86c, which carries the shared `paperstyle.py`). No new training and no new Della job: both
figures are drawn on the Mac from records that already existed.

### Decision dead-relu-paper#20 applied: four 3BFF networks in Fig. 1c

`fig_paper_F1.py` now has `EXCLUDED_RUNS = {}` and an `apply_decisions` step (the same one
`fig_paper_F2_size.py` uses), which re-flags the panel (c) rows of the collect record of Della job
16556248 (entry 2026-10-31_1755). The 3BFF network of seed 3374117298 was excluded as a straggler
under an earlier read-out; under read-out 79932ef it has a floor crossing (t* 58,968, 277 active,
estimated held-out R² 0.949; entry 2026-10-31_1702), so the reason for excluding it no longer holds.
The drawing stage now prints, for 3BFF at N = 1,000: 4 of 4 runs drawn, active 277, 296, 297, 240,
mean 277.5 ± 26.6 (s.d.), R² 0.949-0.950. Mean is unchanged from the three-network value (277.7);
the s.d. falls from 32.6 to 26.6. CDDM (431 ± 10, n = 3) and DMTS (218 and 422, n = 2; seed
2946835052 has no floor crossing) are unchanged.

### Figure 1 (`fig_paper_F1.py`, img/internal_figures/fig_paper_F1.pdf)

Drawn at 165.1 × 114 mm (prints at 100%). Panel letters and colours follow `paperstyle`: active
units near-black and inactive light grey in a, b and the band of d; task colours only in c and d;
text in ink. The red/blue excitatory-inhibitory code of panel a is gone (every sampled connection
in one grey). Panel b labels each class with its count instead of a legend; it makes no claim that
the criterion sits in an empty valley (dead-relu-paper#13 item 1; entry 2026-10-31_1657). Panel d
has its three tasks side by side with the legend in a strip above them, off the data.
Audit: `python tests/test_paperstyle.py img/internal_figures/fig_paper_F1.pdf` → ok (Helvetica,
Helvetica-Bold, Helvetica-Oblique; median 6.0 pt; letters 8.5 pt; the 12 runs under 5 pt are
sub- and superscripts at 4.2-4.55 pt). The overlap check reports no overlapping text.

### Extended Data Fig. 1 (`fig_supp_ablation.py`)

Reads the 36 `*__end.json` files of Della `/scratch/gpfs/TENGEL/pt1290/c1_ablation_seeded` (14 Oct),
copied unchanged to `data/c1_ablation_seeded/`. One column per task; row a puts N on the x axis
(marker size no longer encodes N), row b the quietest-first curves against units removed over the
network's inactive count. "inactive" everywhere (no "silent"), R² upper case.

Networks (from the runs' configs, read on Della): every 3BFF network is the 75,000-iteration end of
the second of its three consecutive training runs (40,000 + 35,000; the size series continues each
with a 125,000-iteration run to 200,000), N = 500 (6), 1,000 (3), 2,000 (3), 4,000 (3). CDDM at
100,000 (12 networks, N = 500-4,000), DMTS at 150,000 (9 networks, N = 500-2,000). DMTS seed
597040509 (N = 2,000, intact R² 0.418) never learned the task and is left out: 35 networks drawn.

Printed by the script:

| task | networks | intact R² | change with every inactive unit removed | as many removed at random (mean of 5) |
|---|---|---|---|---|
| 3BFF | 15 | 0.949-0.952 | −0.792 to −0.249 | −41.9 to 0.060 |
| CDDM | 12 | 0.823-0.838 | −0.010 to +0.001 | −0.297 to 0.019 |
| DMTS | 8 | 0.993-1.000 | −0.046 to −0.001 | −0.154 to −0.092 |

In every one of the 35 networks, random removal ends below removal of the inactive units (largest
difference, random minus inactive: −0.137, on 3BFF).
