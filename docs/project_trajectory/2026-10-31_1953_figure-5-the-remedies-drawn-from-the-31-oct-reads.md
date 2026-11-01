## 2026-10-31 19:53 — Figure 5 and its Extended Data drawn from the 31 Oct remedy reads

Package fig5-drive (dead-relu-paper#9). No new training or analysis jobs: every number below comes
from read outputs that already existed, re-drawn by two new scripts.

**Scripts.** `trainRNNbrain/experiments_and_analysis/fig_paper_F5_drive.py` (main Figure 5) and
`fig_ed_remedies.py` (ED Fig. 8, rate penalty alone; ED Fig. 9, remedies at three floor margins; the
rows of ED Table 3). Both print every number the paper quotes. Inputs, copied from Della on 31 Oct
into `data/remedy_reads_31oct/` (gitignored): `~/rm_read/remedies_catalogue.json` and `summary.txt`
(entry 2026-10-31_1732), `~/a3_read/a3_N*.json` (same entry), `~/b3b4_read/read_b3.json` and
`read_b2_b4.json` (entry 2026-10-31_1750), `~/rm_diag/rm_diag.json` and `~/rm_diag33/rm_diag.json`
(saved as `rm_diag33.json`). The nine B2 runs at N = 500-2000
(`NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=frm0.1`) were rsynced from
`della:~/trainRNNbrain_c1ana/data/trained_RNNs/` without weights (TrainLosses.json,
ParticipationTrace.pkl, config) and re-read on the Mac with `common.read_run`; the reasons for
exclusion match the catalogue run for run (N = 1000: one "loss ends below its floor", two "never
stays under the bar").

**Cell selection.** Each remedy's size series is the one (family, signature) present at three or
more sizes in the catalogue, the selection `a3_remedy_ablation.py` uses; the script asserts that
exactly one exists per family. Controls are the catalogue's `role == "control"` cells (Figure 1's;
four networks at N = 1000, including seed 3374117298, as dead-relu-paper#20 recommends).

**Figures** (vector; PNG renders checked by eye):
- [Figure 5](../../img/internal_figures/fig_paper_F5_drive.svg): (a) the five rules; (b) N = 1000 at
  75k, active units against held-out R², A3's intact scores (9 draws); (c) multiplier and R² cost
  against N, floor reads, with rescaling and read-out dropout drawn at 75k at every size
  (dead-relu-paper#16, option 1); (d) B2 at N = 1000; (e) A3; (f) B3 at the floor and at 75k.
- [ED Fig. 8](../../img/internal_figures/fig_ed_rate_alone.svg): B2 traces at N = 500, 1000, 2000 and
  R² at the floor against N.
- [ED Fig. 9](../../img/internal_figures/fig_ed_remedies_size.svg): multipliers at 1.05x, 1.10x and
  1.20x each run's floor.

**What the redraw adds to the 31 Oct entries.**
- Panel (b) needs R² at 75k for the rate penalty, which the catalogue does not hold (`r2_75k` is empty
  for the B2 cells). It uses A3's intact scores for every network instead: one scoring path, the same
  networks as panel (e). A3's counts are the offline counts at 75k (control 242-288; the catalogue's
  probe counts are 236-288).
- At looser margins rescaling and read-out dropout have reads where 1.05x has none at N ≥ 2000: 12
  of 16 networks at 1.10x and 15 of 16 at 1.20x. At 1.20x read-out dropout gives 1.60-1.90 and
  rescaling 1.23-1.81 across N = 500-4000; at 75k they give 2.00-2.05 and 1.55-1.95. The rate
  penalty alone gets reads at 1.10x at N = 500 (2 of 3, 2.28x) and at 1.20x at N = 1000 (1 of 3,
  3.09x), inside the swing of its active count.
- Off-scale held-out R² values, all on weights saved inside a loss burst or after divergence:
  rescaling at N = 2000, one network −14,335 at 75k; rate + sparsity at N = 500, one network −2,163
  at 75k; A3, a duplication network at N = 1000 scores −4,902 after its recruited units are removed,
  and one rescaling network's random-k removals have a median of −41,974. They are drawn at the axis
  edge and left out of the ED Table 3 means (marked).
- B2 at N = 2000 ends training with 950-1604 active units, after 1854-1893 at the floor.

**Decisions followed until Pavel answers.** #16 option 1 (75k for rescaling and read-out dropout,
1.20x in ED Fig. 9); #17 option 2 (Results 5 does not take B3 as support for combining rules; the
figure and ED Table 5 state both read-outs and the literal verdict); #20 option 2 (four controls at
N = 1000, as the catalogue already does).

**Not changed, found on the way.** The paper's Results 5 states the CDDM rate-penalty result (all
1000 units active at < 1% R² cost) with the caveat that it comes from an end-of-training sweep
(board, "Decisions taken"), not from a floor or 75k read. The Discussion calls read-out dropout
"targeted dropout", and the introduction and discussion still cite `fig:F3`/`fig:F5`, both
undefined; those files belong to other packages.
