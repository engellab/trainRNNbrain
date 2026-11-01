## 2026-10-31 17:32 — Remedies catalogue at the floor (RMtop200) and A3 ablations

**What was read.** RMtop200: every flip-flop remedy cell (and the controls next to it) continued from
75,000 to 200,000 iterations with kept weight snapshots, trained from `~/trainRNNbrain_redo` at
709c17d (Della job ids in `~/f2_logs/rm_submit_go.txt`; all 189 tasks completed). Cells were found by
listing Della's `NBitFlipFlop*_c200000/` directories, not from an old cell list: 83 cells, 267 runs,
including B2's rate-penalty-alone cells (`NBitFlipFlop_remedy_grid/..._arm=frm0.1`) and Figure 1's
controls. Each cell's family is set from its saved config (`read_remedies.classify`), and every cell
was checked against its control on the settings no remedy changes (noise, learning rate, weight
decay, bias, batch, task): no cell differs.

**How it was read.** `trainRNNbrain/experiments_and_analysis/read_remedies.py` (commit 2d32b3b), Della
array 16553347 (40 CPU shards) from `~/trainRNNbrain_rmread` at 2d32b3b; shard 33 sat on an
overloaded node (load 155 on 8 cores) and was rerun as 16555725 on another node, same code.
Output: Della `~/rm_read/runs_*.json`, the merged `~/rm_read/remedies_catalogue.json` and
`~/rm_read/summary.txt`. Each run is read where its smoothed noisy training loss reaches 1.05× its
own fitted floor and stays there through 1.2× that iteration (`common.read_run`, read-out 79932ef),
because remedies change both how fast and how low the loss goes. The *multiplier* is a cell's mean
active count divided by its control's mean active count at the same N, both read this way.
Controls are Figure 1's flip-flop cells (`ff_revive_c200000` at N = 500-2000, `paper_grid_c200000`
control at 4000), the same ones B2 was read against. Held-out R² is under the training noise, 36 draws
(`heldout_r2.floor_r2`): measured on the kept snapshot at the crossing where one exists within 1,000
iterations, else estimated from the end of the run as 1 − (L(t\*)/L_end)(1 − R²_end). Only 49 of the
146 floor reads have a measured R², because most crossings fall before 76k, where the top-ups start
keeping snapshots. Every run is also read at 75k (the same-size rule): active count at the probe
nearest 75k and held-out R² measured on the 75k weights.

Cost: about 59 allocated CPU-hours for the read and the diagnostic below; A3 about half a GPU-hour.

### Controls

| N | Runs read | Active at the floor | R² at the floor |
|---|---|---|---|
| 500 | 6 of 6 | 179-235 (mean 205) | 0.950-0.951 |
| 1000 | 4 of 4 | 240-297 (mean 278) | 0.949-0.950 |
| 2000 | 3 of 3 | 370-415 (mean 390) | 0.950 |
| 4000 | 3 of 3 | 517, 540, 598 | 0.950 |

Two more N = 1000 control cells, read the same way, give 1.07× (`c1_snapshots`, 264-344 active) and
1.02× (`std_dropfix`, 253-306) the Figure 1 control: control-to-control spread is up to 7%.

### The catalogue: the remedies on the paper's list, at the floor

Each entry: multiplier at 1.05× (runs read of runs found), then held-out R² at the floor minus the
control's (0.950). "No read" means no run of the cell has a floor crossing; the 75k multiplier is
given instead, labelled.

| Remedy | Cell | N = 500 | N = 1000 | N = 2000 | N = 4000 |
|---|---|---|---|---|---|
| Noisy synapses, σ_w = 1 | `synnoise_ext` (500, 2000), `synnoise` (1000), `size_series ..._synnoise_sw1.0` (4000) | 1.81× (3/3), −0.043 | 2.03× (3/3), −0.023 | 1.54× (3/3), −0.025 | 1.60× (5/5), −0.021 |
| Rescaling incoming weights, target 10 | `size_series ..._rescale_tgt10` (500, 2000, 4000), `target_ladder tgt=10.0` (1000) | 1.57× (4/4), −0.011 | 1.51× (1/4), −0.007 | no read (0/4); 1.73× at 75k | no read (0/6); 1.95× at 75k |
| Prune-and-duplicate | `paper_grid ..._arm=duplication` | 2.06× (3/3), −0.003 | 2.77× (3/3), −0.005 | 3.24× (3/3), −0.009 | 3.59× (1/3), −0.018 |
| Read-out dropout, rate 0.2, β 4 | `dropout_sizes` | 2.02× (1/3), −0.016 | 1.81× (1/3), −0.010 | no read (0/3); 2.00× at 75k | no read (0/3); 2.01× at 75k |
| Rate + sparsity penalties | `ff_both40k` | no read (0/3); 1.98× at 75k | 3.51× (3/3), −0.022 | 4.99× (1/3), −0.021 | 5.87× (1/1), −0.020 |
| Rate penalty alone (B2) | `remedy_grid frm0.1` | no read (0/3) | no read (0/3) | 4.81× (3/3), −0.024 | 5.64× (2/3), −0.011 |
| Redraw inactive units | `replace_variants mode=random` / `zero_out` | | 0.84× / 0.96× (3/3 each), +0.001 | | |

Active counts behind the multipliers: noisy synapses 365-385, 544-583, 592-607 and 838-963 active
at N = 500, 1000, 2000 and 4000; duplication 402-438, 745-788, 1248-1274 and 1979; rate + sparsity
961-986 at 1000, 1947 at 2000, 3237 at 4000. The multipliers are smaller at the looser margins for
most remedies (noisy synapses at N = 4000: 1.60× at 1.05×, 1.41× at 1.10×, 1.34× at 1.20×; duplication
at N = 2000: 3.24×, 2.75×, 2.41×).

Noisy synapses are the only remedy read at the floor at every size in every run (14 of 14). Their R²
cost is 0.021-0.025 at N ≥ 1000 and 0.043 at N = 500; R² measured on the 75k weights gives the same
picture (0.912-0.938 against the controls' 0.949-0.952). Larger σ_w costs more than it gains at
N = 1000: σ_w = 2 gives 1.89× at R² 0.75-0.78, σ_w = 3 gives 1.75× at 0.60-0.61; σ_w = 0.1 and 0.3 give
1.34× and 1.35× at a cost of at most 0.003. Duplication has the highest multiplier at the smallest R²
cost (≤ 0.009 at N ≤ 2000). Every run read at the floor is listed in `remedies_catalogue.json`, with
the variant cells at N = 1000 that the table above leaves out (8 other duplication settings, 12
read-out dropout settings, 25 other rescaling settings, 4 other σ_w).

### Why 121 of 267 runs have no floor read

Exclusions, by `common.read_run`'s reason: 92 "never stays under the bar", 12 "loss ends more than
1.5× above its floor", 9 "loss ends below its floor", 7 "fit pinned at its lower bound", 1 diverged.

The 92 "never stays under the bar" runs were diagnosed with
`trainRNNbrain/experiments_and_analysis/rm_exclusion_diag.py` (commits 484e0a4, cb36d82; Della job
16554963 and a rerun on shard 33's runs; output `~/rm_diag/rm_diag.json`, `~/rm_diag33/rm_diag.json`).
It recomputes `read_run` unchanged and compares the smoothed loss with its 1.05× bar:

- 84 runs end above the bar: over the last 10% of training their smoothed loss is still about 5% to
  50% above the floor the fit extrapolates (median over that stretch, 1.047-1.495× the floor).
- 8 runs reach the bar only at 169k-188k, too late for the hold through 1.2t to fit inside 200k.

The excluded runs are still descending at 200k. For two read-out dropout runs at N = 4000 the smoothed
loss over the fitted floor is 1.20, 1.15, 1.10 and 1.08 at 75k, 100k, 150k and 200k; for two
rescaling runs at N = 2000 it is 1.16-1.21, 1.12-1.16, 1.09-1.11 and 1.06-1.08. Their fitted
stretch exponents are β = 0.17-0.23, against 0.80-0.86 for two control runs at N = 2000, whose
curves are within 1% of their floor by 150k. The floor of a run with β ≈ 0.2 is an extrapolation
far past the data, so "5-8% above the floor at 200k" carries that uncertainty. At the looser 1.20×
margin, 70 of the 92 runs have a crossing; the active counts there are in `rm_diag.json`. For
read-out dropout at N = 4000 they agree with the 75k counts (1080-1119 at 1.20×, 1081-1118 at 75k);
for rescaling at N = 2000 the ranges overlap (485-633 at 1.20×, 476-701 at 75k).

What this means for the catalogue: under the binding 1.05× read-out, rescaling and read-out dropout
have no floor read at N = 2000 or 4000 (0 of 16 runs), because their noisy loss has not stopped
falling by 200k. Their multipliers at 75k (1.7-2.0×) are the only same-sized numbers.

### B2's third test is closed

B2's rule (entry 2026-10-31_1532) falsifies the rate penalty alone if "the multiplier falls below
synaptic noise's at the same N". At the two sizes where B2 has floor reads, the rate penalty's
multiplier is three to three and a half times synaptic noise's: 4.81× against 1.54× at N = 2000, and 5.64× against
1.60× at N = 4000. The third test does not falsify. B2's verdict stays **falsified**, on the two
counts already recorded (no floor read at N = 1000; R² cost 0.024 at N = 2000, past the 0.02 bar).

### A3: removing the units each remedy recruited

`trainRNNbrain/experiments_and_analysis/a3_remedy_ablation.py` (commit 30efe65), Della jobs
16553440-43 (one GPU job per N) from `~/trainRNNbrain_a3`; output `~/a3_read/a3_N<N>.json`. Networks:
every run of each remedy present at three or more sizes, and Figure 1's controls, all read at 75k (the
same-size rule; each network is compared with ablations of itself). A unit is removed by zeroing its
outgoing recurrent and output weights (`ablate_silent_units`). Held-out R² under the training noise,
9 draws, scored on the GPU (the intact scores match `heldout_r2` on CPU: N = 1000 controls 0.951-0.952
both ways).

*Recruited units* are defined operationally: k = the network's active count minus the control's mean
active count at the same N, and the recruited set is the network's k least-participating active
units. The networks are trained separately, so no unit-to-unit match to the control exists; this
choice is the set most likely to be surplus. The reading was fixed in the script's docstring before
the jobs ran: if the recruited units carry no task signal, removing them leaves R² near the intact
value while removing k random active units lowers it; if they do work, the two removals cost about
the same.

Medians over each cell's networks (each network's value is the median over 5 random sets):

| N | Remedy | Networks | Active | R² intact | −10% random | −30% | −50% | −k recruited | −k random active | Cut to control's count |
|---|---|---|---|---|---|---|---|---|---|---|
| 1000 | control | 3 | 278 | 0.952 | 0.48 | 0.21 | 0.06 | | | |
| 1000 | noisy synapses | 3 | 572 | 0.929 | 0.67 | 0.27 | 0.15 | 0.58 (k = 303) | 0.16 | 0.41 |
| 1000 | rescaling | 4 | 444 | 0.938 | 0.58 | 0.20 | 0.10 | 0.49 (k = 174) | 0.06 | 0.36 |
| 1000 | duplication | 3 | 799 | 0.947 | 0.69 | 0.26 | 0.13 | 0.23 (k = 530) | 0.06 | 0.24 |
| 1000 | read-out dropout | 3 | 539 | 0.933 | 0.86 | 0.29 | 0.18 | 0.19 (k = 270) | 0.18 | 0.19 |
| 1000 | rate alone | 3 | 985 | 0.930 | 0.70 | 0.30 | 0.18 | 0.22 (k = 716) | 0.13 | 0.22 |
| 1000 | rate + sparsity | 3 | 946 | 0.926 | 0.69 | 0.34 | 0.22 | 0.37 (k = 677) | 0.15 | 0.37 |
| 4000 | control | 3 | 544 | 0.951 | 0.65 | 0.18 | 0.13 | | | |
| 4000 | noisy synapses | 5 | 870 | 0.934 | 0.60 | 0.27 | 0.17 | 0.62 (k = 325) | 0.23 | 0.40 |
| 4000 | rescaling | 6 | 1080 | 0.934 | 0.63 | 0.22 | 0.12 | 0.39 (k = 536) | 0.14 | 0.24 |
| 4000 | duplication | 3 | 2027 | 0.930 | 0.79 | 0.41 | 0.23 | 0.57 (k = 1482) | 0.06 | 0.56 |
| 4000 | read-out dropout | 3 | 1094 | 0.939 | 0.89 | 0.39 | 0.18 | 0.30 (k = 549) | 0.19 | 0.26 |
| 4000 | rate alone | 3 | 3690 | 0.934 | 0.80 | 0.35 | 0.21 | 0.18 (k = 3145) | 0.12 | 0.18 |

N = 500 and 2000 are in the JSON and the A3 summary (`a3_remedy_ablation.py summary ~/a3_read`) and
show the same pattern; the N = 500 controls diverge when 10% of their units are removed (median R²
−22), and the N = 500 rate + sparsity networks are unstable at 75k (one at R² −2,142).

Results:
- **The recruited units do work, by the pre-set reading.** Removing them lowers R² from 0.93-0.95 to
  0.18-0.62 in every remedy at N = 1000 and 4000, far from the intact value (N = 2000: 0.15-0.72;
  N = 500: 0.17-0.70, and the rescaling networks diverge, median −2.1). For read-out dropout and the
  rate penalty the cost is close to that of removing as many random active units (N = 1000: 0.19
  against 0.18, and 0.22 against 0.13). For noisy synapses, rescaling and duplication, removing the
  recruited set costs less than a random set of the same size (noisy synapses at N = 1000: 0.58
  against 0.16, in each of the 3 networks), so the least-participating active units carry less of the
  task than the average active unit, but still much of it.
- **Cutting a remedy network down to the control's active count does not recover the control's
  performance**: R² 0.18-0.56 at N = 4000 against 0.951 for the intact control.
- **Remedy networks lose less to random removal.** Removing a random 10% of all units leaves median
  R² 0.58-0.86 at N = 1000 for the remedies against 0.48 for the control; read-out dropout holds best
  (0.86 at 1000, 0.89 at 4000). At 30% and 50% removal every cell's median is at or below 0.41.
- **Inactive units in remedy networks can be removed at small cost** (all inactive units removed:
  R² 0.92-0.94 for duplication, read-out dropout and the rate penalties, 0.84-0.86 for noisy
  synapses and rescaling at N = 1000), while removing the inactive units of the control costs
  0.952 → 0.38 at N = 1000 and 0.951 → 0.70 at 4000 (the paper already reports the N = 1000 cost).

Caveats: A3 is read at 75k, not at the floor, and includes networks with no floor read. The
recruited set is a proxy (least participating), so "the recruited units do work" means "the surplus
units, chosen as the least participating, carry task signal".

### What changes

- The board's remedy multipliers (75k, end of training) can be replaced by the floor reads above for
  noisy synapses, duplication, rate + sparsity and the rate penalty alone. For rescaling and read-out
  dropout the floor read exists only at N ≤ 1000; above that, the 75k multiplier stays, labelled.
- The open decision "which biological penalty goes in the main text" now has the synaptic-noise
  numbers: the only remedy read at the floor in all 14 runs across sizes, at 1.5-2.0× and an R² cost
  of 0.021-0.025 at N ≥ 1000.
