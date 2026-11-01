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
four networks at N = 1000, including seed 3374117298, as decided in dead-relu-paper#20).

**Figures** (vector; PNG renders checked by eye):
- [Figure 5](../../img/internal_figures/fig_paper_F5_drive.svg): (a) the five rules; (b) N = 1000 at
  75k, active units against held-out R², A3's intact scores (9 draws); (c) multiplier and R² cost
  against N, floor reads, with rescaling and read-out dropout drawn at 75k at every size
  (dead-relu-paper#16, as decided); (d) B2 at N = 1000; (e) A3; (f) B3 at the floor and at 75k.
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

**Decisions applied** (manager, on Pavel's delegation, 31 Oct, on dead-relu-paper#16, #17 and #20):
#16, rescaling and read-out dropout at N ≥ 2000 at 75k, their 1.20x crossing in ED Fig. 9; #17, B3
between the bars (Results 5 does not take B3 as evidence that raising the active count requires
combining rules; the figure and ED Table 5 state both read-outs and the literal verdict); #20,
option 2 (four controls at N = 1000, as the catalogue already does).

**Not changed, found on the way.** The paper's Results 5 states the CDDM rate-penalty result (all
1000 units active at < 1% R² cost) with the caveat that it comes from an end-of-training sweep
(board, "Decisions taken"), not from a floor or 75k read. The Discussion calls read-out dropout
"targeted dropout", and the introduction and discussion still cite `fig:F3`/`fig:F5`, both
undefined; those files belong to other packages.

**Revision after the round-1 review (PR #22, 20:10).**
- A3, every rule (the first version of Results 5 said the random removal costs "as much" for the rate
  penalty, which the per-network values contradict). Median over networks of the median R² of 5
  random-k draws, against R² after removing the k added units (fig_paper_F5_drive.py printout, panel
  e). N = 1000: rate penalty 0.13 vs 0.22 (random lower in 3 of 3 networks), noisy synapses 0.16 vs
  0.58 (3 of 3), rescaling 0.06 vs 0.49 (4 of 4), prune-and-duplicate 0.06 vs 0.23 (2 of 3; the third
  scores −4,902 after removal), read-out dropout 0.18 vs 0.19 (2 of 3). N = 4000: rate penalty 0.12
  vs 0.18, noisy synapses 0.22 vs 0.62, rescaling 0.14 vs 0.39, prune-and-duplicate 0.06 vs 0.57,
  read-out dropout 0.19 vs 0.30 (random lower in every network of every rule). So removing k random
  active units costs more than removing the k added ones for four rules at N = 1000 and all five at
  N = 4000; read-out dropout at N = 1000 is the one case where the two cost the same.
- Loss bursts per network for ED Table 3: the RMtop200 per-run records (`della:~/rm_read/runs_*.json`,
  field `n_bursts`, 267 runs, one diverged run without a count) copied to
  `data/remedy_reads_31oct/rm_read_runs/`; `fig_ed_remedies.py` asserts a count for every run of
  every tabulated cell. Controls 4-8, 8-12, 6-10 and 10-13 at N = 500, 1000, 2000 and 4000; rate
  penalty 51-75 and rate + sparsity 62-81 at every size; rescaling 36-112; prune-and-duplicate 4-20
  up to N = 2000 and 150-155 at 4000; noisy synapses 0-8; read-out dropout 0-5. The B2 counts match
  the Mac re-read (N = 1000: 60, 66, 67).
- ED Fig. 9: networks read per cell at 1.10x and 1.20x go into its caption; a size with no read now
  breaks a family's line. Figure 5b x-axis to 1030 (rate-penalty markers at 990-995 were clipped);
  ED Fig. 8d legend moved inside the axes, checked with `get_tightbbox` (no overlap with any tick
  label or text).

**Revision after the second round-1 review (PR #22).**
- Why the rate penalty alone (B2) has no floor read at N ≤ 1000. The first two versions of Results 5
  and ED Fig. 8 gave the loss bursts as the reason. The bursts are not the reason: ED Table 3 has the
  rate penalty at N = 2000 with 68-75 bursts and a floor read in 3 of 3 networks, and rescaling at
  N = 500 with 74-82 bursts and 4 of 4, and the read-out masks bursts before the fit. The exclusion
  reasons, from `common.read_run` (`fig_ed_remedies.py`, ED8 lines): N = 500, two networks never stay
  under the 1.05x bar (60 and 58 bursts) and one ends 1.8x above its floor (57); N = 1000, two never
  stay under the bar (67 and 60) and one ends below its floor, still falling (66). Controls burst 4-8
  times at N = 500 and 8-12 at N = 1000. Results 5 and the ED Fig. 8 caption now state the reasons and
  give the burst counts as a separate observation.
- Figure 5 panel text: panel a, rescaling "strengthens an inactive unit's positive inputs", read-out
  dropout "hides active units from the read-out"; panel e y-label "k added units removed" (the
  caption, Results 5 and Methods say "added"). Text-overlap check over every drawn tick label, axis
  label, text and legend: no overlap, nothing off the canvas.
