## 2026-11-02 22:03 — Connectivity statistics over all units and over the active units: rule, fixed before any value is computed

Work package r12-weights (dead-relu-paper #127, ROUND12_BRIEF "Package r12-weights"; Pavel's item 52 of
1 Nov: "remove dimensionality, replace with some other affected statistics. Perhaps weight statistics,
or degree statistics of each unit; synaptic values as well could be"). Code branch wp/r12-weights,
worktree `~/Documents/Projects/paper_push/wt/r12-weights-code`, from main 627dc1e.

**Question.** Does a connectivity statistic of the recurrent weight matrix change when it is
computed over all N = 1,000 units instead of over the active units only? "Over the active units"
means over the active–active submatrix W_rec[A, A], where A is the set of active units.

This entry is committed before any of the statistics below is computed on any network. The pass
condition is not changed after the values are seen.

### Networks and active units

The nine networks of Fig. 1d/1e and Extended Data Fig. 2 (three per task, N = 1,000, default
settings), at the snapshot of each network's floor crossing, as listed in
`data/pop_stats_nets/manifest.tsv` (written for `population_stats.py`, entries 2026-10-31_2212 and
2026-10-31_2220). The trained recurrent matrix W is `W_rec` of that snapshot (stored as float16,
read as float64).

The active set A is the one Extended Data Fig. 2 uses: `population_stats.simulate(net)["act"]`,
which is `common.active_mask` on the noise-free participation p = std(r) + q0.9(|r|) of the
`population_stats.batch` batch. Check M1 (below) requires the count to equal the stored `n_active`
of `data/population_stats.json` exactly.

Orientation: the forward pass is `W_rec @ r` (`rnns/RNN_torch.py`), so row i of W holds unit i's
incoming weights and column i its outgoing weights.

### Initial weights

No snapshot at iteration 0 was kept. The initial matrix W0 is rebuilt with
`gradient_participation.initial_params(cfg)` from the config of the first run of the network's
warm-start chain. For CDDM and DMTS that run is the network's own run (`init_from: null`). For 3BFF it
is the 75,000-iteration parent named in `paths.init_from`: seeds 2670297180, 1542248578 and
2851654641, which are also the seeds of the floor snapshots in the manifest, so the floor snapshots
lie in those parent runs. Their configs are copied from Della.

The rebuild uses current code, so it can differ from the draw that was trained (the memory
"offline task rebuild needs a version pin"). Check M2 tests the rebuild against the kept snapshot at
iteration 500 of the same run (`c1_snapshots/.../s<seed>/it0000500/`).

### Statistics

For a set S of units (S = all units, or S = A) and k = |S|:

- **W1, s.d. of the recurrent weights:** the standard deviation of the k² entries of W[S, S].
- **W2, fraction of large weights:** the fraction of the k² entries of W[S, S] with |w| ≥ θ.
- **W3, median normalised in-degree:** for each unit i in S, in_i = #{j in S : |W_ij| ≥ θ} / k. The
  statistic is the median of in_i over i in S.
- **W4, median normalised out-degree:** out_i = #{j in S : |W_ji| ≥ θ} / k; median over i in S.
- **W5, median weight change from initialisation:** for each unit i in S,
  Δ_i = sqrt( Σ_{j in S} [ (W_ij − W0_ij)² + (W_ji − W0_ji)² ] / (2k) ), the root-mean-square change
  of its incoming and outgoing weights to the other units of S. The statistic is the median of Δ_i
  over i in S.

Dividing each degree by k, the number of possible partners in S, keeps the size of the submatrix from
deciding the result (an unnormalised degree over A is smaller than over all units by A/N for any
matrix). The degrees use the median because the mean normalised in-degree (and out-degree) over S is
exactly W2: (1/k) Σ_i (1/k) Σ_j 1[|W_ij| ≥ θ] is the fraction of entries above θ. A mean degree would
repeat W2.

**Threshold θ, set from the initial weights, before any trained value is seen:** θ = 2 σ0, with σ0
the standard deviation of all N² entries of the network's own rebuilt W0. The initial entries are
i.i.d. Gaussian draws rescaled to spectral radius 1.2 (`get_connectivity_unconstrained`), so about
4.6% of initial entries exceed θ for every set of units; any difference between S = all and S = A
in W2-W4 therefore arises in training. θ is the same for both sets of one network. As a secondary
check, W2-W4 are also reported at θ = 1 σ0 and 3 σ0; those values do not enter the pass condition.

### Comparison and pass condition

Per network and statistic, the relative difference is rel = (S_all − S_act) / S_act, the definition
of `population_stats.rel_diff`, so the values sit beside Extended Data Fig. 2.

A statistic **differs** between all units and active units if both hold:

1. |rel| > 0.10 in every one of the nine networks (the brief's threshold, kept; it is also the
   "changes" limit of the 2026-10-31_2212 population-statistics rule, so ED Fig. 2 uses one limit
   throughout);
2. the size control passes in every one of the nine networks: for 200 random subsets R of |A|
   units drawn uniformly without replacement (numpy `default_rng(20261102)`, one stream per network
   in manifest order), the 95th percentile over subsets of |(S_all − S_R) / S_R| is below 0.05. If a
   random subset of the same size already moves the statistic by 5%, its size and not its activity
   could produce the active-set difference, and the statistic fails.

The sign of rel is reported per task. A statistic whose sign differs between networks still
"differs" under conditions 1 and 2, but a paper sentence about it must state the direction per task.

The statistic that passes and goes into Extended Data Fig. 2a (replacing the dimensionality of rates)
is the first passing one in the order W1, W3, W4, W2, W5: W1 needs no threshold; the degrees W3, W4
are Pavel's "degree statistics of each unit"; W2 duplicates the mean degree; W5 depends on the rebuilt
initial weights. Every passing statistic is reported to the manager for the Introduction list.

If no statistic passes, the paper is not changed and a result entry says so.

### Checks on the code, each with the result that would mean it is broken

- **M1 (active mask).** The active count from `simulate` equals `n_active` in
  `data/population_stats.json` for every network (430, 416, 438, 323, 412, 261, 270, 340, 285). Any
  mismatch stops the analysis: the mask is then not the ED Fig. 2 mask.
- **M2 (rebuilt initial weights).** For every network, the Pearson correlation over all N² entries
  between the rebuilt W0 and the iteration-500 snapshot's W_rec is above 0.3, and the same
  correlation for a twin rebuilt with the seed plus 1 is below 0.01 in absolute value. A wrong draw
  correlates with the snapshot at about 0 (±0.001 over 10⁶ entries). If M2 fails for a network, W5
  is not computed for it, W5 cannot pass, and θ is instead 2 × 1.2/√N (the scale the initialiser
  aims for).
- **M3 (independent hand check).** For CDDM s1423687489, W1, W2, W3 and W5 over A are recomputed by
  separately written pure-Python code (lists and the `math` module, no numpy in the arithmetic, the
  matrix read once with `np.load` and converted with `.tolist()`). The two implementations must agree
  to a relative 1e-6. A larger gap means one of them is wrong; the analysis stops until it is found.
- **M4 (size control on the initial weights).** On W0 of every network, W1-W4 over A must differ
  from W1-W4 over all units by less than 5% (|rel| < 0.05): before training, the active set is not
  special. A larger initial difference means the comparison measures the draw, not training. This
  check is reported; it does not change the pass condition.

### Prediction, written down so the result can contradict it

Gradients with respect to the incoming and outgoing weights of a unit with near-zero rate are near
zero, so those weights stay close to W0. W[A, A] holds about 7-19% of the N² entries ((A/N)², A =
261-438). The prediction is that W1, W2 and W5 are smaller over all units than over the active
units by more than 10% in every network. The result that would falsify it: |rel| ≤ 0.10 in any one
network, or a size-control failure. If training changed the inactive units' weights as much as the
active units', rel would be near 0.

### Run

`python trainRNNbrain/experiments_and_analysis/connectivity_stats.py` on the Mac (CPU), writing
`data/connectivity_stats.json` and the per-network table. The result entry will name its commit.
