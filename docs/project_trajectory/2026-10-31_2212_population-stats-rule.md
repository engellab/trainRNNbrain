## 2026-10-31 22:12 — Population statistics over all units and over active units: the pre-registered rule

Work package pop-stats (ptolmachev/dead-relu-paper#56; decision on #34). This entry is committed on
trainRNNbrain branch `wp/pop-stats` before any statistic is computed on a trained network or on the
synthetic oracle. Part 1 is the editor's rule, copied verbatim from
https://github.com/ptolmachev/dead-relu-paper/issues/34#issuecomment-6553682701 (the same text is in #56).
Part 2 adds the implementation details the rule leaves open. Part 3 fixes the oracle tests and their
thresholds.

### Part 1. The rule, verbatim

**Pre-registered rule for the population statistics** (editor, 31 Oct 2026, 21:48; written before any value was computed on a trained network).

*Networks.* The nine *N* = 1,000 default networks of Fig. 1d, three per task: `NBitFlipFlop_c1_snapshots_c200000`, `CDDM_c1_snapshots` and `DMTS_c1_snapshots`, each `EqType=h_N=1000_arm=control` (Della, `~/trainRNNbrain_c1ana/data/trained_RNNs/`), at the weights the removal analysis of Fig. 1d used (the snapshot at each network's floor crossing; `fig_paper_F1.py` names them). Active and inactive units by the paper's criterion, on the same noise-free batch as the statistics. Optional addition: CDDM networks at *N* = 4,000 from the size series, only if weights kept within 5% of their floor crossing exist. If none exist, the Discussion's sentence about 4,000-unit CDDM networks is cut.

*Responses.* Noise-free rates on one fixed batch per task: CDDM, its 450 conditions; DMTS and 3BFF, 256 trials drawn with a fixed seed (DMTS: 64 per sample-test pair).

*Statistics*, each computed twice: over all *N* units and over the active units only.
- **S1, dimensionality:** participation ratio PR = (Σλ)²/Σλ² of the covariance of rates over every step and trial.
- **S2, dimensionality of normalised rates:** the same after dividing each unit's rates by their standard deviation. Units with zero variance (silent units) are dropped from both sets, as a recording contains no neuron that never fires.
- **S3, selective and mixed-selective units:** one response per trial and unit, the mean rate over the task epoch (CDDM steps 100-199; DMTS steps 110-130; 3BFF every step after each input has had a pulse, one sample per step). Regressors: CDDM context, motion coherence, colour coherence and choice; DMTS sample, test and match (±1 each); 3BFF the three bit states (±1). The partial *R*² of a regressor is the drop in residual variance when it is added last, over the residual variance without it. A unit is *selective* for a variable if that partial *R*² ≥ 0.10, and *mixed-selective* if it is selective for at least two variables. Zero-variance units are not selective. Reported as counts: selective and mixed-selective units among all *N* and among the active units, and the number of inactive units classified selective.
- **S4, functional classes:** k-means (20 restarts, fixed seed) on each unit's selectivity profile (its partial *R*² values, normalised to unit sum; zero-variance units excluded); *k* from 2 to 8 by the highest mean silhouette. Run over all non-silent units and over the active units; report *k\** and the share of units in each class, with classes matched between the two runs by nearest centroid.

*Decision rule, per statistic and task.* S_all and S_act are the values over all units and over the active units (S3: the share of units selective and the share mixed-selective; S4: *k\** and the matched class shares).
- **changes**: |S_all − S_act| ≥ 0.10 |S_act| in all three networks of the task (S4: *k\** differs, or a matched class share differs by at least 10 percentage points, in all three);
- **unchanged**: |S_all − S_act| < 0.02 |S_act| in all three (S4: the same *k\** and every matched share within 2 percentage points);
- **mixed**: anything else.

*What the paper then says.* Results 1 states each statistic's outcome, with its values over all units and over active units for one task and the range for the other two; every network is in Extended Data Fig. 2. The Introduction and the Discussion keep a general statement that inactive units change a population statistic only for statistics labelled *changes* on at least two of the three tasks, and they name the statistics labelled *unchanged*. If every statistic is *unchanged* on every task, the paper takes #34's option 2 (name only statistics whose value depends on *N* by construction) and reports the measurement.

*Predictions, written now so that the result can contradict them.* S1 unchanged on every task (an inactive unit's participation is about 17,000 times smaller than an active unit's, Fig. 1b). S2 and S3 may change on 3BFF, where no unit is silent and every inactive unit has a small nonzero rate. On CDDM, where about 300 of 1,000 units are silent, S3's shares change through the denominator alone.

*Validation before any trained network is measured* (thresholds fixed now). (1) PR from the eigenvalues and from tr(C)²/tr(C²) agree within 10⁻⁸ relative; a synthetic population of *k* independent unit-variance sources mixed into 200 units, plus 800 units at zero, gives PR within 5% of *k* for *k* = 3 and 10. (2) Planted units with a partial *R*² of 0.05, 0.2 and 0.5 for one regressor and 0 for the others are recovered within 0.02 and classified correctly. (3) Every statistic is printed per network, never only as a mean.

*Cost.* Under one CPU-hour: nine networks of 1,000 units, one batch each.


### Part 2. Implementation details (fixed before the run)

**Code.** `trainRNNbrain/experiments_and_analysis/population_stats.py` (statistics, oracle tests,
run over the nine networks; writes `data/population_stats.json`, gitignored) and
`trainRNNbrain/experiments_and_analysis/fig_ed_population.py` (Extended Data Fig. 2, drawn from that
file). CPU only, on the Mac (`/opt/miniconda3/envs/rnns`), after copying the nine weight files from
Della.

**Networks and weights.** The weight files are the ones the Fig. 1d ablation used, taken from the
`params_file` field of `/scratch/gpfs/TENGEL/pt1290/c1_ablation_79932ef/*__floor.json` (the
`ABLATION_DIR` of `fig_paper_F1.py`), with the architecture from each run's own `*_config.yaml`.
Local copies in `data/pop_stats_nets/<ablation label>/` with `manifest.tsv`. Iteration of the
weights / active count in the ablation file / first 16 hex digits of the weight file's SHA-256:

| network | iteration | active (ablation file) | sha256 |
|---|---|---|---|
| CDDM s1423687489 | 25,000 | 430 | 2f06601e189a4f5a |
| CDDM s2657899762 | 26,000 | 416 | 41ed42a2dc66ca74 |
| CDDM s2670295995 | 24,000 | 438 | e0f92c3a14a374e5 |
| DMTS s198893703 | 23,000 | 322 | b73ebd3e1d5837bb |
| DMTS s3322309204 | 20,500 | 411 | 77183814c4625996 |
| DMTS s3753404946 | 66,000 | 261 | 1d10504de8b9a984 |
| 3BFF s1258717025 | 67,000 | 267 | 083d705d1013dba5 |
| 3BFF s2461724787 | 50,000 | 342 | c3dd0775308645d8 |
| 3BFF s4113307577 | 58,000 | 287 | caeee57a43381a94 |

The network is built exactly as `ablate_silent_units.build` builds it (same assertion that it is
the default network), with `sigma_w = 0`.

**Batch.** Seed `POP_SEED = 20261031` for the task generator (`tcfg.seed`) and for `np.random` /
`torch` global seeds. CDDM: `task.get_batch()` on the trained coherence grid (15 values, 2 contexts,
450 conditions), unshuffled. DMTS: `num_rep = 64` (4 sample-test pairs x 64 = 256 trials; the
training config has 256 repeats). 3BFF: `batch_size = 256` (the training config has 1,024). One
noise-free forward pass, `rnn(inputs, w_noise=False)`; states are the net inputs h (N x T x trials);
rates r = relu(h).

**Active units.** p = std(r) + q0.9(|r|) per unit over every step and trial of that batch, computed
by `ablate_silent_units.participation`; active by `common.active_mask` (p >= 0.05 q95(p)). Gate: CDDM's
batch is the same 450 conditions the ablation ordered its units on, so each CDDM network's active
count must equal the ablation file's exactly; DMTS and 3BFF use a different draw and are reported
next to the ablation file's count without a gate.

**Zero-variance units.** A unit whose rate has variance exactly 0 (float64) over every step and
trial. With the bias fixed at 0 these are the units whose rate is 0 throughout.

**S1.** C = covariance (float64, `np.cov` over units, samples = every step of every trial). PR =
(sum lambda)^2 / sum lambda^2 from `np.linalg.eigvalsh(C)`, checked against tr(C)^2 / tr(C^2) on every
network. "All" = all 1,000 units; "active" = the active units.

**S2.** Zero-variance units dropped from both sets; each remaining unit's rates divided by their
standard deviation; PR as S1. "All" = every unit with nonzero variance; "active" = the active units
(no active unit has zero variance).

**S3.** One response per sample and unit. CDDM: mean rate over steps 100-199 (Python slice
100:200, stimulus onset to decision onset), one sample per trial, 450 samples. DMTS: mean rate over
steps 110-129 (slice 110:130, test-stimulus offset to decision onset, matching CDDM's convention that
the epoch ends at decision onset), one sample per trial, 256 samples. 3BFF: the rate at every step at
which all three bit targets are nonzero (from the onset of the first pulse of the last bit to receive
one, to the end of the trial), one sample per step and trial. Regressors (each as a column, plus an
intercept): CDDM context (+1 motion, -1 colour), motion coherence, colour coherence, choice
(`correct_choice`, +1 or -1; -1 when the relevant coherence is 0, as the task defines it); DMTS
sample (+1 channel 0, -1 channel 1), test (same coding), match (+1 same, -1 different); 3BFF the
three bit targets (+1 or -1). OLS by `np.linalg.lstsq`. Partial R^2 of regressor j = (RSS without j
- RSS full) / RSS without j. A unit whose response vector has variance exactly 0 has no partial R^2
(NaN) and is not selective. Selective: some partial R^2 >= 0.10; mixed-selective: at least two.
Shares: count / N for "all" (N = 1,000) and count / active count for "active". The decision rule is
applied separately to the share selective (S3a) and the share mixed-selective (S3b); both labels are
reported, not combined.

**S4.** Profile = a unit's partial R^2 vector divided by its sum. Excluded: units without partial
R^2 (S3) and units whose partial R^2 sum is 0. `sklearn.cluster.KMeans(n_clusters=k, n_init=20,
random_state=POP_SEED)` for k = 2..8; k* = argmax of `sklearn.metrics.silhouette_score` (Euclidean).
"All" = every unit with a profile; "active" = the active units with a profile. Class shares = class
size / number of units clustered in that run. Matching: the one-to-one assignment of active-run
classes to all-run classes minimising total Euclidean distance between centroids
(`scipy.optimize.linear_sum_assignment`), over min(k*_all, k*_active) pairs. The S4 difference of a
network is the largest absolute difference of matched shares, in percentage points; "changes" for a
network if k* differs or that difference is >= 10 points; "unchanged" if k* is equal and the
difference is < 2 points.

**Zero baselines.** If S_act = 0 and S_all = 0, the difference is 0 and counts as unchanged; if
S_act = 0 and S_all != 0, it counts as changes.

**Optional N = 4,000 CDDM networks: none exist.** No kept snapshots of CDDM at N = 4,000 exist on
Della (`data/c1_snapshots/` holds CDDM only at N = 1,000). The only N = 4,000 CDDM default networks
(`CDDM_paper_grid/EqType=h_N=4000_arm=control`, three seeds) were trained for 100,000 iterations
without `checkpoint_keep`, so the only weights are at 100,000 iterations; CDDM crosses its floor at
about 25,000 iterations at N = 1,000. The addition is therefore not made and, by the rule, the
Discussion's sentence about 4,000-unit CDDM networks is cut (for frame-discussion, #57). The
crossing of these three networks is computed in the run and recorded in the results entry, to
confirm that the end weights are not within 5% of it.

### Part 3. Oracle tests and their thresholds (fixed before they run)

Each test states the result that would show the code is broken.

**V1a, PR formula.** On a random 50 x 50 covariance and on every network's C: eigenvalue PR and
tr(C)^2 / tr(C^2) agree within 1e-8 relative. Broken if any pair differs by more.

**V1b, known dimensionality.** k = 3 and k = 10 independent unit-variance Gaussian sources, 20,000
samples, mixed into 200 units through a 200 x k matrix with orthonormal columns (the population
covariance then has exactly k equal nonzero eigenvalues, so its PR is k by construction), plus 800
units fixed at 0. Pass: S1 over all 1,000 units, S1 over the 200, and S2 over the units with nonzero
variance each within 5% of k; S1 over all 1,000 equals S1 over the 200 within 1e-8 relative (zero
units add zero eigenvalues). A Gaussian mixing matrix is also run and printed for information, with no
threshold, because its population PR is below k.

**V2, planted selectivity.** On the real CDDM design (450 conditions, four correlated regressors):
for each regressor j and each target value in {0.05, 0.2, 0.5}, a unit y = a x_j + e with e drawn
Gaussian and then projected orthogonal to the intercept and all four regressors, and a chosen so the
sample partial R^2 of x_j is exactly the target (a^2 |x_j residualised on the others|^2 = target /
(1 - target) |e|^2). Pass: recovered partial R^2 within 0.02 of the target for x_j and within 0.02 of
0 for the other three; classified not selective at 0.05, selective and not mixed-selective at 0.2
and 0.5. Plus two units with 0.3 on two regressors (motion and colour coherence; context and
choice): classified mixed-selective. Sampling version: the same planted values on a design of 20,000
samples of three independent +-1 regressors (3BFF-like), with e not orthogonalised: recovered within
0.02.

**V3, a population with known classes, plus all-zero units.** 300 units on the CDDM design in three
planted classes of 100 (selective for motion coherence only, colour coherence only, both), partial
R^2 0.4 on the planted regressors, plus 700 units at 0; the 300 are the "active" set. Predictions:
S1 and S2 identical between all and active (within 1e-8 relative); S3 counts identical and shares
over all = shares over active x 300 / 1,000; S4 k* = 3 in both runs with every class share 1/3
within 2 points, and identical between the runs. Broken if any of these fails.

**V4, weak non-zero units (the 3BFF case).** As V3, but the 700 extra units carry random
selectivity (partial R^2 0.4 on one random regressor) at 1e-3 times the amplitude of the 300.
Predictions: S1 over all within 2% of S1 over the 300 (their variance is 1e-6 times smaller);
S2 differs from the 300-unit value by at least 10% (normalisation gives the weak units full weight);
S3 counts the weak units as selective (partial R^2 does not depend on amplitude). These are the
statistics' expected behaviour, not tests of the networks.
