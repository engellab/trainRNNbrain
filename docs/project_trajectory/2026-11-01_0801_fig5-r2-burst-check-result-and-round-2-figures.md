## 2026-11-01 08:01 — Burst check of Figure 5's R² values: result; Figure 5 and Extended Data Figs. 7-8 redrawn (round 2)

Package fig5-r2 (dead-relu-paper#55). Follows the rule fixed in `2026-11-01_0745` before the run.

**How it was run.** `trainRNNbrain/experiments_and_analysis/fig5_burst_check.py` (commit e36f638),
copied with `common.py` (identical, md5 ab9d9d70) to Della `~/fig5r2_code/`, run as Della CPU job
16592651 (8 CPUs, under 5 minutes):

    sbatch --job-name=F5burst --cpus-per-task=8 --mem=48G --time=01:30:00 --wrap "... python
    fig5_burst_check.py --out ~/fig5r2_read --cddm
    ~/trainRNNbrain_fig3settings/data/trained_RNNs/CDDM_std_g0_bias0_c75000 --procs 8"

Output `~/fig5r2_read/burst_check.json`, `traces_N1000.json`, `F5burst.16592651.out`, copied to
`data/remedy_reads_31oct/fig5r2_read/`. 391 networks, none unreadable. Cost about 0.5 CPU-h.

**Falsification check: passed.** Every intact R² below 0.5 among the checked values is marked
affected; the summary prints no "MISSED" line.

**Result, over everything checked** (values; "affected" = weights inside a padded loss burst or
non-finite stretch):

| Source | Value | Affected | Calm |
|---|---|---|---|
| per-unit-rule catalogue (all 83 cells) | R² at the floor crossing (measured or estimated) | 0 | 146 |
| per-unit-rule catalogue | R² at 75,000 iterations | 61 | 186 |
| removal analysis (all N) | every score (75,000-iteration weights) | 21 networks | 72 networks |
| combination test | R² at the floor crossing | 3 | 10 |
| combination test | R² at 75,000 iterations | 3 (one non-finite) | 19 |
| CDDM rate penalty λ = 0.2 and its unpenalised neighbour (Fig. 3a) | final weights at 75,000 | 0 | 10 |

**In the items the paper shows.**
- No R² measured or estimated at a floor crossing is affected, in any per-unit-rule cell. The
  rate-penalty size test's R² cost at N = 2,000 (0.926 against 0.950, three measured snapshots)
  stands as recorded.
- Figure 5b (N = 1,000, 75,000 iterations): 3 of 19 networks (one rate-penalty network, two
  rescaling networks), R² 0.930-0.941.
- Figure 5c: 14 of the 18 rescaling values at 75,000 iterations (N = 500: 4 of 4; 1,000: 2 of 4;
  2,000: 4 of 4; 4,000: 4 of 6), including the one at −14,335; the other 13 have R² 0.892-0.941. Read-out
  dropout: 0 of 12. The floor-crossing series (rate penalty, noisy synapses, prune-and-duplicate): 0.
- Figure 5e (N = 1,000 and 4,000): 9 of 36 rule networks (N = 1,000: rate penalty 1, rescaling 2;
  N = 4,000: rate penalty 1, prune-and-duplicate 1, rescaling 4).
- Figure 5f (combination test): 6 values, the four already reported as off scale in
  `2026-10-31_1750` (rate + noise at 120,000, −27.5; the triple at 4,000, −74.4; rate + rescaling at
  75,000, −61 and −1.7 × 10⁹, the last in a non-finite stretch) and two more: the triple's 0.844 at
  3,000 iterations and one rate-alone network's 0.931 at 75,000.
- Supplementary Table 2: 21 networks whose 75,000-iteration weights are affected (rescaling 14,
  rate penalty 3, rate + sparsity 3, prune-and-duplicate 1). The table gets a column with these
  counts.

So the affected values outside the combination test and the rate-penalty networks are rescaling
(Fig. 5c, Supplementary Table 2), one prune-and-duplicate network at N = 4,000 and the rate + sparsity
networks at N = 500. Rescaling bursts 36-112 times per network over 200,000 iterations, so a fixed
iteration often lands inside a padded burst. Apart from −14,335 (already left out of the table's
mean), the affected rescaling values are 0.892 (one network at N = 500) and 0.933-0.941 (12
networks), against 0.934-0.938 for the four calm ones; every one is below the control networks'
0.949-0.952, so the statement that rescaling costs R² does not rest on the affected values. No fold
change uses R². Reported on dead-relu-paper#18 (the decision named this case), with the choice to
mark the values and change no claim.

**Two more facts found on the way.**
- *CDDM:* no loss burst in any of the ten CDDM networks of Fig. 3a's rate-penalty value and its
  unpenalised neighbour, over their 75,000 iterations. On 3BFF at N = 1,000 the control networks
  burst 7-9 times and the rate-penalty networks 29-34 times in their first 75,000 iterations (66, 67
  and 60 over 200,000).
- *Rate + sparsity at N = 500:* two of the three 200,000-iteration networks
  (seeds 1526990995 and 2467430577) were continued from the same 75,000-iteration network (seed
  2729554139), so the cell's 75,000-iteration values hold that network twice (R² 0.591 both times).
  The only such pair in the catalogue (checked over every cell). Supplementary Table 2 says so.

**Figures** (vector, audited with `tests/test_paperstyle.py`: Helvetica only, bold letters 8.5 pt,
median text 6.0 pt, drawn at 165.1 mm; text-overlap check over every drawn text: none):
- [Figure 5](../../img/internal_figures/fig_paper_F5_drive.svg), 165.1 × 140 mm: a key row; (a)
  active units over training at N = 1,000, median over networks, every rule and the control networks
  (new; replaces the text table); (b) N = 1,000 at 75,000 iterations; (c) fold change and R² change
  against N; (d) the rate penalty alone at N = 1,000; (e) the removal analysis; (f) the combination
  test with a dot matrix of which rules each combination has. Ink rings mark the affected R² values;
  off-scale values are arrowheads inside the axes. Per-unit-rule palette from `paperstyle`.
- [Extended Data Fig. 7](../../img/internal_figures/fig_ed_rate_alone.svg), 165.1 × 82 mm, and
  [Extended Data Fig. 8](../../img/internal_figures/fig_ed_remedies_size.svg), 165.1 × 58 mm.

Panel (a), median active count over networks at N = 1,000 (`fig_paper_F5_drive.py` printout), at
the probe nearest 75,000 iterations and at the last probe (199,900): control networks 260 and 180;
rate penalty 985 and 938; noisy synapses 572 and 563; rescaling 441 and 401; prune-and-duplicate
793 and 802; read-out dropout 539 and 561.
