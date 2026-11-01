## 2026-11-01 10:46 — fig5-r3 correction: launch evidence, the noisy-synapse sequence, and the full search

This entry corrects `2026-11-01_1007_fig5-r3-how-each-per-unit-rule-setting-was-chosen.md`. The
corrections follow the independent review of dead-relu-paper PR #99 (comment 6562133913). Della
refused a non-interactive login, so no Slurm submit times (sacct) could be read; the evidence below
is the run files on the Mac.

### 1. Launch evidence: job end times, not commit times

Entry 1007 used commit times as launch times. A commit can come after its jobs were submitted, so
the evidence is now the time each run's files were written. Config and loss files are written
together at the end of a job, so these times are job end times. Two caveats:

- They are file modification times of the Mac copies. The copies keep Della's times; the
  spread within and across folders shows that.
- The Mac shows them in Pacific time, and Della and the commits use Eastern time. All times below
  are in Eastern time, which is the Mac time plus 3 hours.

The review compared Mac (Pacific) file times with Eastern commit times. Read on one clock, every job
below ends after the commit that the notebook and the commit messages say launched it. The order of
the choices in entry 1007 stands; only the evidence changes.

| What | Run files written (ET) | Commit (ET) |
|---|---|---|
| noisy synapses σ_w = 0.1, 0.3, 1, N = 1,000 | 30 Sep 03:20-03:31 | |
| synnoise_ext: σ_w = 1 at N = 500 | 30 Sep 09:36-09:37 | launched with 84b4485, 07:19 |
| synnoise_ext: σ_w = 2 and 3 at N = 1,000 | 30 Sep 10:14-10:42 | same launch |
| synnoise_ext: σ_w = 1 at N = 2,000 | 30 Sep 14:46-15:16 | same launch |
| size_series: σ_w = 1 at N = 4,000 | 1 Oct 11:58-16:41 | 99c877a, 30 Sep 11:05 |
| rescaling target ladder m = 6, 8, 10 | 30 Sep 04:37-05:04 | 85aea77, 01:17 |
| target ladder m = 14, 20, 30 | 30 Sep 10:48-11:16 | 84b4485, 07:19 |
| rescaling size series m = 10, N = 500 / 2,000 / 4,000 | 30 Sep 13:21-13:53 / 19:24-19:28 / 1 Oct 15:40-16:02 | 99c877a, 11:05 (N = 4,000: 0ddcdac, 13:08) |
| copy noise sweep (copy_perturb) | 25 Sep 00:53-03:35 | |
| paper_grid duplication N = 500 / 1,000 / 2,000 | 28 Sep 12:01-12:03 / 12:51-12:55 / 17:17-17:23 | 852a17f, 28 Sep 10:11 |
| whole-pass dropout grid at β = 4 | 22 Sep 16:54 to 23 Sep 09:49 | |
| dropout_sizes q = 0.2, β = 4, N = 500 / 2,000 | 23 Sep 13:03-13:07 / 24 Sep 03:44-03:50 | 8bfd030, 23 Sep 10:14 |

Two consequences:

- **Target ladder.** One network at m = 30 was written at 11:16, after commit 99c877a (11:05). That
  commit quotes the ladder's participation ratios and launches the size series at m = 10, so the
  m = 30 value it quotes rests on 3 of the 4 networks.
- **Noisy synapses.** σ_w = 1 at N = 500 and 2,000 was launched together with σ_w = 2 and 3, in one
  launch, before either level was read. A submit time from sacct would confirm this once Della is
  back.

### 2. The noisy-synapse sequence, completed

Entry 1007 said σ_w = 2 and 3 "were read afterwards". The full sequence on 30 Sep:

1. σ_w = 2 and 3 were read, at R² 0.784 and 0.655, which commit 99c877a records as "BROKEN".
2. 99c877a (11:05) launched N = 4,000 at σ_w = 1.
3. 65d6aa8 (12:07) drew σ_w = 3 in Figure 2 because it had the most active units (767).
4. 0ddcdac (13:08) returned the figure to σ_w = 1, because at the common read-out σ_w = 2 and 3
   "fail the bar" (R² 0.812 and 0.677).

So σ_w = 1 was also kept after σ_w = 2 and 3 had been read, on their R² cost. Methods (Per-unit
rules) now says so.

### 3. λ_frm = 0.03 and 0.01 under one read-out

The combination-test pilot (16 Oct) was read with read-out 353dc13:

- λ = 0.01: 2 of 3 networks reached a floor crossing, with 766 and 932 active units.
- λ = 0.03: 2 of 3 networks reached a floor crossing, with 629 and 810 active units.

Under the paper's read-out (79932ef), the λ = 0.03 networks have 0 of 3 crossings (entry
2026-10-31_1750). Pilot rule 6 needs two crossings, so it would not have chosen 0.03 on these
numbers. λ = 0.01 was trained to 200,000 iterations (Della 15598507-08) but was never read again.
Supplementary Table 3a now has a λ = 0.01 row marked "not read". Its footnote gives both pilot
values under the earlier read-out, and Methods (Combination test) states the change.

### 4. The full search: rules and settings not continued to 200,000 iterations

`trainRNNbrain/experiments_and_analysis/table_rule_screen.py` (wp/fig5-r3) walks every local 3BFF sweep
folder. It lists each cell that has a per-unit rule and no continuation in the catalogue: 60 cells.
For each cell it gives the mean active count at the probe nearest 40,000 iterations (75,000 for
longer runs). The fold change is against Figure 5's control networks at the same N and iteration
(γ = 0.1 cells against γ = 0.1 controls). Control means at 40,000 iterations: N = 1,000, 275.

| Rule (not among the five) | Settings | Fold change at 40,000 |
|---|---|---|
| synaptogenesis | 4 | 0.80-1.88 |
| disinhibition | 1 | 0.98 |
| orthogonal-row replacement | 1 | 1.07 |
| donor-mix replacement | 1 | 1.21 |
| bias offset | 1 | 1.36 |

These five rules were screened at N = 1,000 and stopped at 40,000 iterations. Counted with the five
reported rules, ten per-unit rules were tried, and Results 5 now says so.

The other cells are settings of the five rules:
- 16 settings of the whole-pass dropout variant, fold change 1.20-3.51.
- Duplication without the replacement cap at γ = 0.1, N = 500-4,000, 2.03-2.87.
- 11 duplication cells trained with the self-weight error, 1.35-2.60.
- Two copy controls.
- Rate-penalty and rate + sparsity cells trained to 150,000 or 400,000 iterations (std_pen,
  std_penlong, std_dropout), 2.49-5.48 at 75,000 iterations.
- Read-out dropout cells trained to 150,000 iterations (std_dropout, paper_grid_150k), 1.64-3.83.
- A duplication cell trained to 150,000 iterations, 2.99.
- Three variants of the zeroing revival method at other rates, 1.01-1.15.

Cells with the sparsity penalty alone are left out, because that penalty is a Results 3 setting.
My counts for the five screened rules match the reviewer's independent count: synaptogenesis
220-516 active, disinhibition 270, orthogonal 294, donor mix 333, bias offset 373.

### 5. The rate penalty: why λ_frm = 0.1

The B2 note (13 Oct) chose λ = 0.1 to match the rate term of the rate + sparsity pair. It also
recorded that λ = 0.1 networks trained earlier had been read: 979 of 1,000 units active at
N = 1,000, which is the std_pen and std_penlong cells of Supplementary Table 3b. Methods now gives
both reasons.

### Figure 5

The figure is now drawn at 141 mm, the main-float limit (MAX_H_MM), and the "float too large"
warning is gone. Rows of panels are at least 4 mm apart. Two drawing changes:

- The open markers at 75,000 iterations in panel a sit above the traces, with a white inside.
- An off-axis value is drawn as its own marker at the axis edge, with a short arrow, in place of a
  triangle that could be read as read-out dropout's marker.

The printout of every number is unchanged.
