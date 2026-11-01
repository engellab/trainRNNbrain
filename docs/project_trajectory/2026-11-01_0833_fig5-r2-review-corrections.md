## 2026-11-01 08:33 — fig5-r2, review of PR #72: corrections to three entries

Package fig5-r2 (dead-relu-paper#55); review https://github.com/ptolmachev/dead-relu-paper/pull/72#issuecomment-6560718847.

**Correction to `2026-10-31_1532` (rate-penalty size test).** That entry says the R² cost at
N = 2,000, 0.926 against 0.950, was "both measured, the penalty's on kept snapshots". Only the rate
penalty's three values are measured on kept weights. Every control value at the floor crossing is
*estimated* from the end of training (`r2_measured` False for all 16 control networks at
N = 500-4,000 in `remedies_catalogue.json`; `fig_ed_remedies.py` prints it). The test still fails:
the cost is 0.024 against an allowed 0.02, and the estimate was within 0.006 of measured values for
eight of nine networks when it was checked (`common.r2_at_floor_estimate` docstring). Extended Data
Fig. 7d now draws the control values open, as estimated.

**Additions to `2026-11-01_0801` (burst check).**
- *Iteration of the scored weights.* The rule entry (`2026-11-01_0745`) set t_w = 75,000 for every
  75,000-iteration value. For the three combination-test networks without any rule (cell "none",
  `NBitFlipFlop_c1_snapshots_c200000` control), the snapshot actually scored is at 75,500
  iterations, and `fig5_burst_check.py` took t_w from that snapshot's path. All three are calm.
- *Affected values with R² above 0.9* (the rule asked for them to be counted separately): 69 of the
  193 affected values over everything checked.
- *Rate-penalty non-finite losses at N = 4,000* (from `burst_check.json`, field `n_nonfinite`):
  1,009, 605 and 500 of 200,000 iterations, as well as the 2,004-3,686 at N = 2,000 given in
  `2026-11-01_0813`. Extended Data Table 2 now gives both sizes.

**Correction to the Results 5 sentence on Fig. 5a** (first PR #72 version, not in a notebook entry):
"each rule's [median count] stays more than twice as high" as the control networks' is false. From
`traces_N1000.json` (this package's recompute, `fig_paper_F5_drive.py` traces): each rule's median
is above the control median from the probe at 5,400 iterations on (last probe at or below it:
rescaling at 5,300); after 10,000 iterations the ratio is at most 2 at 56% of probes for rescaling
(minimum 1.13), 29-30% for noisy synapses and read-out dropout (minimum 1.40), and never below 1.97
for the rate penalty; at the last probe it is 2.2 (rescaling) to 5.2 (rate penalty). Results 5 now
states those facts.

Figure 5a now carries each rule's marker every 25,000 iterations, so the two ochre traces (noisy
synapses, read-out dropout) differ in shape as well as lightness.
