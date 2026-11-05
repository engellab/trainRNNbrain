## 2026-11-05 09:26 — Uniform read-outs at N = 1,000 and 75,000 iterations: the 3BFF input-weight cap fails at every value, CDDM noisy synapses fail on R², CDDM read-out dropout and prune-and-duplicate work (issue #6)

**What was read.** The three cell groups that `docs/interventions_matrix.md` lists under "Read-outs
that cost no GPU time", none of which any paper script had read: the 3BFF input-weight cap, CDDM
noisy synapses, and CDDM read-out dropout and prune-and-duplicate. Script
`trainRNNbrain/experiments_and_analysis/read_uniform_cells.py` (CPU, 19 minutes the first time, then
from the held-out cache); output `data/read_uniform_cells.json` and the tables below. Every cell is
N = 1,000, the standard network (ReLU, bias fixed at 0, weight decay 1e-6, Adam,
σ_rec = σ_inp = 0.05). Each arm's saved config was compared with its control's on the settings no
intervention changes (`read_remedies.SHARED_FIELDS`: noise, bias, equation, learning rate, weight
decay, batch, task); every arm agrees with its control on all of them.

1. **3BFF input-weight cap 0.1, 0.5, 2.0** (|W_inp| clamped after every optimiser step):
   `NBitFlipFlop_std_inpcap/EqType=h_k=3_N=1000_cap=<c>_pen=none`, 3 runs each, 150,000 iterations,
   trained 24 Sep on Spock (Mac copies). Control: `NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000`,
   3 runs, 150,000 iterations, trained 2 Oct on Della (the default rung of Figure 3's input-scale
   ladder).
2. **CDDM noisy synapses σ_w 1.0**: `CDDM_synnoise_ext/EqType=h_N=1000_sw=1.0_iters=100000`, 3 runs,
   100,000 iterations (Mac only).
3. **CDDM read-out dropout** (kind mute, rate 0.2, β 4) and **prune-and-duplicate** (reinit_mode copy,
   copy noise 1.0, rate 0.025, maturity 1,000): `CDDM_paper_grid/EqType=h_N=1000_arm={mute,duplication}`,
   3 runs each, 100,000 iterations. The Mac holds complete copies of all six run folders (config,
   participation trace with 1,000 probes, loss record, final weights), so nothing was copied from
   Della.

The control for groups 2 and 3 is `CDDM_paper_grid/EqType=h_N=1000_arm=control` (3 runs, 100,000
iterations: the same launcher and budget as the arms, weights on the Mac). The paper's Figure 3a
control, `CDDM_std_g0_bias0_c75000/EqType=h_N=1000_LmbdRWS=0_LmbdFR=0` (5 runs, 50,000 + 25,000
iterations), is read beside it from the Figure 2 cache; its parent run is not on the Mac, so its
loss chain cannot be joined locally. Every verdict below is the same against either control.

**Read-out.** `fig_paper_F2_fails.read_fixed` at 75,000 iterations (the same-size rule of the guide):
the number of active units (`common.active_mask` on the stored participation trace, which the
trainer measures on a noise-free pass) at the probe nearest 75,000 (the probe is at 75,000 for every
run except the Figure 3a control's, whose last probe is at 74,900); the held-out R² (`heldout_r2`:
the training σ_rec and σ_inp, σ_w = 0, 36 seeded draws), estimated from the final weights as
1 − (L(75k)/L_end)(1 − R²_end) with the noisy training loss, because no run kept weights at 75,000
(the Figure 3a control ends at 75,000, so its R² is measured). The 3BFF control's weights are not on
the Mac: its R² at 75k is the Figure 2 cache's value (`data/fig_paper_F2_fails_cache.pkl`, Della,
31 Oct, the same function); its active counts were re-read from the trace on disk and equal the
cache's. Cross-platform check, threshold set beforehand at 0.002 per run: the same `read_fixed` on
the Mac on the three `CDDM_std_g0_drift_bias0` N = 1,000 runs, which are in the cache and have
weights locally, gives R² 0.8350, 0.8330, 0.8337 and 325, 340, 345 active units, equal to the
cache's values to four decimals. For the noisy-synapse cell the stored score (one noisy pass with
σ_w on) is not used; the clean R² (w_noise=False: no activity noise and no synaptic noise, the
co-primary measure named in the header of `slurm/SilentReLU_size_series_della.slurm`) is given on
the final weights beside the project's measure.

**Criterion** (the per-unit-rule criterion of the matrix file, set before the read). An arm works if,
at 75,000 iterations, every one of its runs has more active units than the largest control count,
and its mean held-out R² is at least the controls' mean minus 0.01. An arm whose mean R² is above the
controls' passes the second condition. "Fold" below is the arm's mean number of active units divided
by the controls' mean.

**3BFF input-weight cap** (3-bit flip-flop, N = 1,000, read at 75,000 of 150,000 iterations)

| cell | seed | active units | held-out R² at 75k | R² of the final weights (150k) |
|---|---|---|---|---|
| cap 0.1 | 1243342348 | 303 | 0.817 (estimate) | 0.868 |
| cap 0.1 | 3575035851 | 429 | 0.873 (estimate) | 0.877 |
| cap 0.1 | 1773761193 | 402 | 0.850 (estimate) | 0.879 |
| cap 0.5 | 3873875029 | 362 | 0.918 (estimate) | 0.920 |
| cap 0.5 | 929894869 | 425 | 0.923 (estimate) | 0.923 |
| cap 0.5 | 2679000981 | 321 | 0.914 (estimate) | 0.925 |
| cap 2.0 | 2667022991 | 280 | 0.926 (estimate) | 0.935 |
| cap 2.0 | 3828651639 | 343 | 0.935 (estimate) | 0.937 |
| cap 2.0 | 533431211 | 352 | 0.936 (estimate) | 0.936 |
| control (ksweep_bias0) | 1093113168 | 296 | 0.950 (estimate, F2 cache) | not on the Mac |
| control (ksweep_bias0) | 1068656537 | 245 | 0.951 (estimate, F2 cache) | not on the Mac |
| control (ksweep_bias0) | 2425551384 | 283 | 0.952 (estimate, F2 cache) | not on the Mac |

| arm | active units (seeds) | mean | fold | mean R² | R² minus controls' | verdict |
|---|---|---|---|---|---|---|
| cap 0.1 | 303, 429, 402 | 378 | 1.38 | 0.846 | −0.105 | fails: R² gap beyond 0.01 |
| cap 0.5 | 362, 425, 321 | 369 | 1.34 | 0.918 | −0.033 | fails: R² gap beyond 0.01 |
| cap 2.0 | 280, 343, 352 | 325 | 1.18 | 0.932 | −0.019 | fails: seed 2667022991 at 280 is below the controls' largest, 296; R² gap beyond 0.01 |
| controls | 296, 245, 283 | 275 | 1 | 0.951 | | |

**CDDM noisy synapses** (CDDM, N = 1,000, read at 75,000 of 100,000 iterations)

| cell | seed | active units | held-out R² at 75k | R² of the final weights (100k) | clean R², final weights |
|---|---|---|---|---|---|
| σ_w 1.0 | 3139128930 | 486 | 0.820 (estimate) | 0.822 | 0.963 |
| σ_w 1.0 | 145705583 | 487 | 0.828 (estimate) | 0.830 | 0.960 |
| σ_w 1.0 | 2925113920 | 511 | 0.808 (estimate) | 0.809 | 0.888 |
| control (paper_grid) | 3833537590 | 342 | 0.831 (estimate) | 0.833 | 0.921 |
| control (paper_grid) | 1272763220 | 334 | 0.835 (estimate) | 0.836 | 0.923 |
| control (paper_grid) | 457853 | 333 | 0.835 (estimate) | 0.836 | 0.923 |
| Figure 3a control (c75000, F2 cache) | 3182599426 | 329 | 0.833 (measured) | | |
| Figure 3a control (c75000, F2 cache) | 3090320285 | 320 | 0.831 (measured) | | |
| Figure 3a control (c75000, F2 cache) | 2583923803 | 321 | 0.834 (measured) | | |
| Figure 3a control (c75000, F2 cache) | 2161475543 | 320 | 0.834 (measured) | | |
| Figure 3a control (c75000, F2 cache) | 783414550 | 334 | 0.834 (measured) | | |

**CDDM per-unit rules** (CDDM, N = 1,000, read at 75,000 of 100,000 iterations; the same controls)

| cell | seed | active units | held-out R² at 75k | R² of the final weights (100k) |
|---|---|---|---|---|
| read-out dropout (mute 0.2, β 4) | 3526318829 | 524 | 0.824 (estimate) | 0.827 |
| read-out dropout (mute 0.2, β 4) | 1823470335 | 505 | 0.825 (estimate) | 0.827 |
| read-out dropout (mute 0.2, β 4) | 140111091 | 523 | 0.830 (estimate) | 0.832 |
| prune-and-duplicate | 11084869 | 701 | 0.832 (estimate) | 0.834 |
| prune-and-duplicate | 4009237578 | 699 | 0.827 (estimate) | 0.829 |
| prune-and-duplicate | 2429277935 | 726 | 0.832 (estimate) | 0.834 |

| arm | active units (seeds) | mean | fold | mean R² | R² minus controls' (paper_grid; Figure 3a) | verdict |
|---|---|---|---|---|---|---|
| σ_w 1.0 | 486, 487, 511 | 495 | 1.47 | 0.819 | −0.015; −0.014 | fails: R² gap beyond 0.01 |
| read-out dropout (mute 0.2, β 4) | 524, 505, 523 | 517 | 1.54 | 0.826 | −0.008; −0.007 | works |
| prune-and-duplicate | 701, 699, 726 | 709 | 2.11 | 0.830 | −0.003; −0.003 | works |
| controls (paper_grid) | 342, 334, 333 | 336 | 1 | 0.834 | | |
| Figure 3a control (c75000) | 329, 320, 321, 320, 334 | 325 | 0.97 | 0.833 | | |

**Verdicts.**

- *3BFF input-weight cap: no value works.* Cap 0.1 increases the number of active units to 303, 429 and
  402 (1.38 times the controls' mean of 275) with held-out R² 0.817, 0.873 and 0.850, a mean 0.105
  below the controls' 0.951. Cap 0.5 gives 362, 425 and 321 (1.34 times; every run above the
  controls' largest, 296) at R² 0.918, 0.923 and 0.914, 0.033 below. Cap 2.0 gives 280, 343 and 352
  (1.18 times; one run below 296) at R² 0.926, 0.935 and 0.936, 0.019 below. The R² gaps are present
  at the end of training as well: the final weights score 0.868-0.879, 0.920-0.925 and 0.935-0.937,
  compared with about 0.95 for the controls (the controls' 75k estimates; their final weights are
  not on the Mac). On CDDM the cap at 0.3 is Figure 3e's revival screen (390-400 active compared
  with 332-340); no 3BFF cell at 0.3 exists (the matrix file lists one at 17 GPU-h).
- *CDDM noisy synapses σ_w 1.0: fails on R².* 486, 487 and 511 active units (1.47 times the
  controls' 336; every run above the controls' largest, 342) at held-out R² 0.820, 0.828 and 0.808,
  a mean 0.819, 0.015 below the controls' 0.834 (0.014 below the Figure 3a control's 0.833). On 3BFF
  the same σ_w at the floor gave 2.03 times the control's count at an R² 0.023 lower (the remedies
  catalogue, `docs/interventions_matrix.md`), so on both supervised tasks noisy synapses increase
  the number of active units with an R² gap beyond the 0.01 tolerance. Clean R² on the final
  weights (no noise of any kind): the σ_w networks score 0.963, 0.960 and 0.888, the controls 0.921,
  0.923 and 0.923. On CDDM every network scores higher without its training noise (the controls by
  about 0.09, two σ_w networks by 0.13-0.14, the third by 0.08), and the clean score does not rank
  the networks as the project's measure does: two σ_w networks are above every control and the
  third is below every control. The verdict uses the project's measure.
- *CDDM read-out dropout (mute 0.2, β 4): works.* 524, 505 and 523 active units (1.54 times;
  every run above 342) at R² 0.824, 0.825 and 0.830, a mean 0.826, 0.008 below the controls' mean.
  The gap is inside the 0.01 tolerance by 0.002, and inside what the guide calls scoring noise on
  CDDM; it is also within the error of the 75k estimate (0.006 on eight of nine validation
  networks, 0.0115 on one). A kept snapshot at 75,000 would settle the margin; `CDDM_paper_grid`
  kept none.
- *CDDM prune-and-duplicate: works.* 701, 699 and 726 active units (2.11 times) at R² 0.832, 0.827
  and 0.832, a mean 0.830, 0.003 below the controls'.

**Across the three tasks (the matrix's rule table, updated).** Prune-and-duplicate works on 3BFF
(2.77 times at the floor, R² 0.005 lower; catalogue) and on CDDM (2.11 times at 75k, 0.003 lower);
the maze trainer cannot run it yet. Read-out dropout works on 3BFF (1.81 times at the floor, one of
three runs read) and on CDDM (1.54 times), and fails on the maze (1.00 times, success 0.19 below the
learned controls; entry 2026-11-05_0812). Noisy synapses σ_w 1.0 increase the number of active
units on 3BFF (2.03 times) and CDDM (1.47 times) at R² gaps of 0.023 and 0.015, beyond the
tolerance on both; the maze arm is still running. The input-weight cap passes at no 3BFF value;
its CDDM value 0.3 is Figure 3e's.

**Caveats.** (1) The 3BFF cap arms (24 Sep, Spock) and their control (2 Oct, Della) were trained at
different commits. Their configs agree on every shared setting; the arms' configs predate the keys
`model.sigma_w`, `trainer.prune_reinit` and `trainer.prune_args`, which the control holds at their
off values (0.0, False). A trainer change between the two dates would be a confound, but the R²
gaps (0.019-0.105) are 2-10 times the validated error of the 75k estimate. (2) Every arm's R² at
75k is an estimate from its final weights, since none of these sweeps kept snapshots. (3) The two
CDDM controls differ in `same_batch` (below) and in budget; they give the same verdicts.

**Side finding, correcting an open point of `docs/interventions_matrix.md`.** `same_batch: true` is
not peculiar to `CDDM_std_g0_tanh`. Every CDDM cell of the `b0_*` launchers on the Mac has it
(`CDDM_std_g0_bias0`, `CDDM_std_g0_bias0_c75000`, `CDDM_std_g0_drift_bias0`,
`CDDM_std_g0_weightdecay_bias0`, `CDDM_std_g0_metabolic_bias0_c75000`,
`CDDM_std_g0_activations_bias0`), and so does `CDDM_77fbd9_g0_noise_ptrack`; `CDDM_paper_grid`,
`CDDM_screen3`, `CDDM_synnoise_ext` and every 3BFF cell have false. On CDDM the flag does not change
the inputs: `TaskCDDM.get_batch` builds the full condition grid in a fixed order on every call (it
permutes only with `shuffle=True`, which `run_experiment` never passes), so with the flag off the
trainer rebuilds the same batch every iteration; only the noise draws differ, and they differ with
the flag on as well. The comment in `configs/trainer/trainer_ptrack_freshbatch.yaml` says the same.

**Della.** Unreachable from this Mac (ssh timed out at 08:51, 08:57 and 09:13; a background probe
continues every 15 minutes). Nothing was needed from it: the six `CDDM_paper_grid` N = 1,000 run
folders are complete on the Mac. Della would add a check that the Mac copies match Della's file
sizes, and the 3BFF control's final weights, which would let its R² at 75k be re-derived locally
instead of taken from the cache (the cross-platform check above makes the two equivalent).

**Checks.** `tests/test_scripts_start.py`: 52 scripts start; `tests/test_no_function_imports.py`:
205 files; the new script has no nested import and every function has a docstring (AST check).
The script's config comparison printed only the interventions' own keys as differences. The
3BFF control's active counts re-read from the trace equal the cache's for all three runs.
