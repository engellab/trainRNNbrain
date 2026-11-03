## 2026-11-02 22:18 — Confirmatory test of connectivity statistics on 39 unseen networks, and the issue #172 test: rule, fixed before any of these networks is computed

Work package r12-weights (dead-relu-paper #127, item 52), branch wp/r12-weights. The manager's decision
of 2 Nov on the first rule (`2026-11-02_2203_connectivity-stats-rule.md`, result
`2026-11-02_2210_connectivity-stats-result.md`): the first rule stands as failed, and the paper takes
no post-hoc sentence. This entry fixes a new rule for a confirmatory test on networks the first
analysis has not seen. It also fixes a test of the interpretation in dead-relu-paper#172. It is
committed before any weight of these networks is read into the analysis. Nothing below changes after
the values are seen.

### Networks

"Unseen" means every default network of N = 1,000 outside the nine of Fig. 1e. The list comes from
an inventory of the Della sweeps (`/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/trained_RNNs`,
CPU job 16711892, script `trainRNNbrain/experiments_and_analysis/connectivity_inventory.py`, run from
`~/trainRNNbrain_c1ana` at ed11dfa, `common.read_run` for the floor crossing t*). Every link of every
chain was checked against the standard network of `docs/AGENT_GUIDE.md`: ReLU, equation h, bias 0,
no Dale's law, γ = 0, self-connections, spectral radius 1.2, σ_rec = 0.05, weight decay 1e-6, no
penalty, no dropout, no pruning, no synaptic scaling. All 39 match it except in two fields:

- `same_batch: True` (one fixed training batch instead of a fresh batch every iteration) in 20
  CDDM networks, marked "same batch" below. These networks are the default arms of the CDDM sweeps of
  Fig. 3. They are included, so the test covers every default arm the paper uses.
- `init_draws: independent` in the three DMTS_std_settings networks (dead-relu-paper#154).

**Left out:**
- the nine networks of Fig. 1e (`CDDM_c1_snapshots`, `DMTS_c1_snapshots`, `NBitFlipFlop_c1_snapshots_c200000`
  control cells), because they were seen;
- `DMTS_paper_grid` control, because it is the `DMTS_d7` task variant (T = 210), not the paper's DMTS
  (T = 190);
- the parents of continuations: `CDDM_std_g0_bias0` and `CDDM_screen3`. Their networks enter once,
  through the chains of `CDDM_std_g0_bias0_c75000` and `CDDM_screen3_c75000`;
- the weight-decay-0 networks, which are not default. They enter only the #172 test below.

**Read point.** The kept snapshot at the floor crossing (`common.read_run(...)["snapshot"]`) exists
for two networks only. For each of the others, the read point is the saved weights nearest t*: a
kept snapshot, or the final weights of any link of the warm-start chain. For a network without a
floor crossing, it is the saved weights nearest 75,000 iterations, the paper's same-size read-out. The
gap between t* and the read point is large in most networks (up to 170,000 iterations). The test
therefore asks whether the statistic differs at the weights that exist. It does not ask about the
floor crossing as such.

| # | task | cell | network (final seed) | t* | read at | weights | note |
|---|---|---|---|---|---|---|---|
| 1 | 3BFF | NBitFlipFlop_ff_revive_c200000/EqType=h_k=3_N=1000_pen=none_arm=none | 3374117298 | 58,968 | 75,000 | link 3502253762 | |
| 2 | 3BFF | same | 1914910909 | 59,917 | 75,000 | link 3039548724 | |
| 3 | 3BFF | same | 3866906883 | 62,703 | 75,000 | link 3095970260 | |
| 4 | 3BFF | same | 4186074845 | 67,871 | 200,000 | own run | |
| 5 | 3BFF | NBitFlipFlop_std_dropfix_c200000/EqType=h_k=3_N=1000_pen=none_do=none | 3665415385 | 62,818 | 75,000 | link 2075057582 | |
| 6 | 3BFF | same | 1847344754 | 57,341 | 40,000 | link 3071103934 | |
| 7 | 3BFF | same | 2147366219 | 56,159 | 40,000 | link 1052986492 | |
| 8 | CDDM | CDDM_paper_grid/EqType=h_N=1000_arm=control | 3833537590 | 27,259 | 100,000 | own run | |
| 9 | CDDM | same | 1272763220 | 26,535 | 100,000 | own run | |
| 10 | CDDM | same | 457853 | 23,929 | 100,000 | own run | |
| 11 | CDDM | CDDM_std_g0_drift_bias0/EqType=h_N=1000_iters=200000 | 2490685872 | 28,249 | 200,000 | own run | same batch |
| 12 | CDDM | same | 970815651 | 30,139 | 200,000 | own run | same batch |
| 13 | CDDM | same | 3150611977 | 30,091 | 200,000 | own run | same batch |
| 14 | CDDM | CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05_iters=200000 | 1032176401 | 30,734 | 200,000 | own run | same batch |
| 15 | CDDM | same | 1072480682 | 29,261 | 200,000 | own run | same batch |
| 16 | CDDM | same | 98625316 | 28,260 | 200,000 | own run | same batch |
| 17 | CDDM | same | 2081117351 | 29,647 | 200,000 | own run | same batch |
| 18 | CDDM | same | 1214724849 | 30,327 | 200,000 | own run | same batch |
| 19 | CDDM | CDDM_77fbd9_g0_noise_ptrack/EqType=h_N=1000_sigrec=0.05 | 622914433 | 26,940 | 75,000 | own run | same batch |
| 20 | CDDM | same | 3398027559 | 26,457 | 75,000 | own run | same batch |
| 21 | CDDM | same | 2804838852 | 25,262 | 75,000 | own run | same batch |
| 22 | CDDM | same | 692468002 | 25,747 | 75,000 | own run | same batch |
| 23 | CDDM | same | 3106096154 | 21,408 | 75,000 | own run | same batch |
| 24 | CDDM | CDDM_std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000 | 2802579864 | 25,228 | 75,000 | own run | same batch |
| 25 | CDDM | same | 1542519098 | 24,604 | 75,000 | own run | same batch |
| 26 | CDDM | CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0 | 3182599426 | 22,293 | 50,000 | link 1137256420 | same batch |
| 27 | CDDM | same | 3090320285 | 24,765 | 50,000 | link 3632476009 | same batch |
| 28 | CDDM | same | 2583923803 | 27,447 | 50,000 | link 1994542782 | same batch |
| 29 | CDDM | same | 2161475543 | 24,208 | 50,000 | link 2990212620 | same batch |
| 30 | CDDM | same | 783414550 | 30,995 | 50,000 | link 348112504 | same batch |
| 31 | CDDM | CDDM_screen3_c75000/EqType=h_N=1000_pen=none_arm=none | 1504400307 | 22,819 | 30,000 | link 2837619054 | |
| 32 | CDDM | same | 341166437 | 28,139 | 30,000 | link 2775612459 | |
| 33 | CDDM | same | 1468600595 | 25,714 | 30,000 | link 381098289 | |
| 34 | DMTS | DMTS_paper_grid_d5/EqType=h_N=1000_arm=control | 86371460 | 18,599 | 150,000 | own run | |
| 35 | DMTS | same | 1167683015 | 75,721 | 150,000 | own run | |
| 36 | DMTS | same | 2946835052 | none (loss ends below its floor) | 150,000 | own run | read nearest 75,000 |
| 37 | DMTS | DMTS_std_settings/EqType=h_N=1000_arm=default_iters=150000 | 3546706548 | none (loss ends below its floor) | 150,000 | own run | read nearest 75,000; independent draws |
| 38 | DMTS | same | 2548178424 | 62,055 | 63,000 | kept snapshot | independent draws |
| 39 | DMTS | same | 2239198786 | 59,018 | 60,000 | kept snapshot | independent draws |

"link s" is the final weights (`*LastParams*.npz`) of the chain's run with seed s, which ended at the
"read at" iteration. Files are copied from Della to `data/connectivity_confirm/<#>_<task>_<seed>/`
with their Della paths in `source.txt`.

### Active units and initial weights

- **Active set A:** `population_stats.simulate` on the read weights, with the config of the link that
  wrote them. This is `common.active_mask` on the noise-free participation of the
  `population_stats.batch` batch, the mask of Extended Data Fig. 2.
- **Initial weights W0:** `gradient_participation.initial_params` on the config of the first run of
  the chain.
- **Check T (twin), per network:** the L2 norm of each row of the rebuilt W0 is compared with the
  `w_rec_row` vector stored at iteration 0 in the first run's own `*_ParticipationTrace.pkl`
  (`Trainer.incoming_norms_`, written before the first update). The twin is valid if the largest
  difference, relative to the mean row norm, is below 1e-4. Calibration, on the six seen CDDM and DMTS
  networks and before this entry: the true twin gives 1e-7 to 5e-6, a twin with seed + 1 gives 0.10 to
  0.13. A continuation run's trace does not carry the first run's iteration-0 norms, so the first
  run's own trace is used.

If check T fails for a network, θ for that network is 2 × 1.2/√N, and the network is left out of the
#172 test.

### Statistics (from the first rule, W2 and W5 dropped)

For a set S of units with k = |S|, on W[S, S] (row i holds unit i's incoming weights):

- **W1:** the s.d. of the k² recurrent weights.
- **W3:** the median over i in S of the normalised in-degree #{j in S : |W_ij| ≥ θ} / k.
- **W4:** the median over i in S of the normalised out-degree #{j in S : |W_ji| ≥ θ} / k.

θ = 2 × the s.d. of all N² entries of the network's own W0. rel = (S_all − S_act) / S_act. The code is
`connectivity_stats.statistics`, the function the first rule's M3 hand check verified.

### Pass condition

A statistic **differs** between all units and active units if, in every one of the 39 networks:

1. |rel| > 0.10; and
2. S_act lies outside the range [min, max] of S_R over 200 random subsets R of |A| units, drawn
   uniformly without replacement. The draws use `numpy.random.SeedSequence(20261103).spawn(39)`, one
   stream per network in the order of the table above, and `default_rng(stream).choice(N, |A|,
   replace=False)`.

An undefined rel (S_act = 0) fails condition 1. The sign of rel is reported per task, and a paper
sentence states the direction. If W1 passes, the manager receives one Introduction sentence and one
Results sentence, with numbers over the 39 networks.

### Checks on the code

- **M1 (mask against the trace):** per network, the active count from `simulate` is compared with the
  count at the probe of the stored participation trace nearest the read iteration (the trace of the
  run that wrote the read weights). On the seen nine networks, the two agreed within 1.2% (for example
  323 and 324, or 412 and 412). A disagreement above 10% in any network stops the analysis until its
  cause is found.
- **M3 (independent hand check):** W1, W3 and W4, over A and over all units, for network #1
  (3BFF 3374117298), recomputed by `connectivity_stats_handcheck.py` (pure Python, extended to W4 and
  to this test's files). The two implementations must agree to a relative 1e-6.

### Test of the interpretation in dead-relu-paper#172

Interpretation under test: Adam's coupled weight decay drives the weights between inactive units
toward zero.

- **Networks.** The six CDDM networks of N = 1,000 trained with weight decay 0:
  `CDDM_std_g0_weightdecay_bias0/EqType=h_N=1000_wd=0_iters=200000` (955478765, 1132724915,
  3370475588; final weights at 200,000) and `CDDM_std_g0_wd0_long/EqType=h_N=1000_wd=0_iters=300000`
  (4178827658, 2029632449, 1711863583; final weights at 300,000).
- **Comparison.** The three networks of the same sweep family with weight decay 1e-6: #11-#13
  (`CDDM_std_g0_drift_bias0`, final weights at 200,000). All nine use `same_batch: True` and bias 0,
  and they differ only in weight decay and length.
- **Statistic.** On the inactive–inactive block (I, the units not in A on the final weights), the
  regression slope b = Σ W_ij W0_ij / Σ W0_ij², and the s.d. of W[I, I] over the s.d. of W0[I, I].
- **Outcomes:**
  - *Supported* if b > 0.9 in all six weight-decay-0 networks and b < 0.1 in all three
    weight-decay-1e-6 networks.
  - *Falsified* (weight decay is not the cause) if b < 0.5 in all six weight-decay-0 networks.
  - *Partial* otherwise, with the numbers reported.
- **Secondary, outside the verdict.** The same slope in the three DMTS_std_settings weight-decay-0
  networks (870461262, 3675806006, 3701657385, final weights at 150,000).
- **Reporting.** The result is posted on #172. The issue is closed if the outcome is "supported" or
  "falsified", and left open with the result if "partial".

### Prediction

On the seen nine networks, W1 had rel of −0.47 to −0.69, and S_act was above every subset value.
The prediction is that W1, W3 and W4 differ under this rule in all 39 networks, with rel negative.
A falsifying result is any single network with |rel| ≤ 0.10, or with S_act inside the subset range.
The networks read at 30,000-50,000 iterations, with less training since their crossing, are the most
likely to produce one.

### Run

`connectivity_confirm.py` on the Mac, CPU, writing `data/connectivity_confirm.json`. The result entry
names the commit.
