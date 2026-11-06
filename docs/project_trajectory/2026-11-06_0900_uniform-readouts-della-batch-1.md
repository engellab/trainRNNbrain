## 2026-11-06 09:00 — Uniform read-outs, Della batch 1: weight decay lowers the 3BFF count as on CDDM; CDDM rescaling fails on both conditions; CDDM AdamW and SGD have more active units; the rate penalty 0.1 works on CDDM; 47 tasks still pending (issue #6)

**What was read.** The 39 array tasks of the eleven Della arrays of entries 2026-11-05_0903
(WdB0 16955456, TgtLadder 16955457), _2256 (B5opt 16957418, StdPen 16957419) and _2259
(NoiseB0 16957556, StdRecInit 16957557 and 16957558, MetB0 16957559, FFpenD 16957560, ActB0
16957561, Screen3 16957562) that `sacct -X` reported COMPLETED at 08:05 on 6 Nov; the other 47
tasks (23 running, 24 pending behind the 24-job limit of gpu-short) are listed at the end and get a
second read-out when they finish. Every run is N = 1,000, the standard network (ReLU, bias fixed at
0, weight decay 1e-6 and Adam unless the arm says otherwise), 75,000 iterations, so the final
weights are the 75,000-iteration weights. The 39 run folders (config, loss record, participation
trace, final weights; no PNGs, optimiser states or checkpoints) were copied to `data/trained_RNNs/`
under the cluster's folder names, with the ten older seeds of `CDDM_std_g0_recinit` (job 14919348,
two per cell), which the matrix listed as on the Mac and which were not. A run folder exists only
once training has finished (its name starts with the final score), so every folder under a cell is
a finished run; `data/della_batch1_status.tsv` (one row per task: job task, Slurm state, seed and
experiment tag from the job log, the trainer's skipped-update line) ties each folder to its task.
Every completed task has its folder on the Mac, and no folder is unaccounted for.

**Script.** `trainRNNbrain/experiments_and_analysis/read_uniform_cells_2.py`, written beside
`read_uniform_cells.py` rather than inside it: it imports yesterday's loaders (`read_cell`,
`read_cell_from_f2_cache`) and rule criterion (`verdict`) and adds the two verdict kinds the new
cells need (settings: increase, decrease or no change; optimiser: the three classes of entry
_2256), a loader for the B5 cache, the status table and the pending list. Extending the old script
would have doubled its group table and its print function; the new file is 570 lines, of which the
cell table is 73. Output `data/read_uniform_cells_2.json` (every run, every verdict, the other
task's ladder beside each setting, the pending tasks) and the tables below. CPU, 49 minutes the
first time (36 noise draws per held-out R²), 21 seconds from the held-out cache.

**Read-out, as pre-registered.** At 75,000 iterations: the number of active units
(`common.active_mask` on the stored participation trace, measured on a noise-free pass, at the
probe nearest 75,000, which is 74,900 for these runs, their last probe), and held-out R² (the
variance of the target output explained by the network's output on a fixed batch of trials not
used in training, under the training noise; `heldout_r2`, 36 seeded draws, σ_w = 0) on the final
weights. Controls and comparison cells that end later are read as the caches hold them: the 3BFF
control `NBitFlipFlop_b5_optimizer/EqType=h_N=1000_opt=adam_wd=1e-6_lr=1e-3` and the B5 arms from
`data/b5_read_cache.pkl` (R² at 75k estimated from the 200k weights), the CDDM ladders of Figure 3
and the flip-flop's second reference from `data/fig_paper_F2_fails_cache.pkl`; the active counts of
every cached cell whose traces are on the Mac were re-read from disk and equal the cache. Four
checks were named before the run and all pass: no cached count differs from the trace on disk; the
CDDM control `CDDM_paper_grid/EqType=h_N=1000_arm=control` re-read today reproduces entry
2026-11-05_0926 (342, 334, 333 active; R² 0.831, 0.835, 0.835); the ten older
`CDDM_std_g0_recinit` seeds, read here from their own final weights, reproduce the Figure 2 cache's
count and R² run by run (a per-run check that does not go through the cache); no completed task
lacks a folder. No run is broken: no loss has a non-finite value, and the trainer skipped at most
499 of 75,000 gradient updates (0.67%) in any run, under the 5% exclusion bar. Each arm's saved
config was compared with its control's on the shared settings (`read_remedies.SHARED_FIELDS`) and
the intervention keys; the differences are the swept key, the budget and checkpoint interval where
the control ran longer (the B5 control: 200,000 iterations, snapshots every 5,000) and, compared
with the older paper-grid control, trainer keys that did not exist when it ran (`optimizer`,
`momentum`, `checkpoint_every`, the full `prune_args` block).

**Controls, read at 75,000.**

| control | runs | active units | mean | held-out R² (mean) |
|---|---|---|---|---|
| 3BFF: `NBitFlipFlop_b5_optimizer/..._opt=adam_wd=1e-6_lr=1e-3` (B5 cache, estimate) | 3 | 270, 234, 233 | 246 | 0.951 |
| 3BFF second reference: `NBitFlipFlop_std_ksweep_bias0/EqType=h_k=3_N=1000_iters=150000` (F2 cache, estimate) | 3 | 296, 245, 283 | 275 | 0.951 |
| CDDM: `CDDM_paper_grid/EqType=h_N=1000_arm=control` (local, estimate from the 100k weights) | 3 | 342, 334, 333 | 336 | 0.834 |
| CDDM second reference: `CDDM_std_g0_bias0_c75000/..._LmbdRWS=0_LmbdFR=0` (F2 cache, final) | 5 | 329, 320, 321, 320, 334 | 325 | 0.833 |
| CDDM recinit control: `CDDM_std_g0_recinit/EqType=h_N=1000_sr=1.2_iters=75000` (local, final; third seed 315 is new) | 3 | 315, 333, 347 | 332 | 0.833 |

### 1. 3BFF weight decay (entry _0903; control B5 Adam wd 1e-6, 233-270 active, R² 0.951)

| cell | seed | active | R² (how) | task |
|---|---|---|---|---|
| wd 0 (B5 cache) | 3004463764 / 4169212093 / 3183899476 | 252 / 296 / 367 | 0.956 / 0.956 / 0.959 (estimate) | Spock, 42e8811 |
| wd 1e-5 | 425111804 / 315144892 / 314492794 | 225 / 229 / 203 | 0.933 / 0.934 / 0.935 (final) | 16955456_4-6 |
| wd 1e-4 | 462658441 / 3629976113 / 3370308883 | 157 / 144 / 163 | 0.905 / 0.907 / 0.906 (final) | 16955456_8, 9, 7 |

| rung | 3BFF: active (mean), R² gap, verdict | CDDM ladder at 75k (Figure 3b; control `CDDM_std_g0_drift_bias0` 325-345, R² 0.834) | direction |
|---|---|---|---|
| wd 0 | 252-367 (305), +0.006, **no change** (252 lies inside the control's range) | 442-469 (453), −0.001, increase | differs by the rule; the mean rises on both tasks |
| wd 1e-5 | 203-229 (219), −0.017, **decrease** | 260-263 (261), −0.003, decrease | same |
| wd 1e-4 | 144-163 (155), −0.045, **decrease** | 154-166 (160), −0.011, decrease | same |

Weight decay 1e-5 and 1e-4 lower the number of active units on the 3-bit flip-flop as they do on
CDDM, and at N = 1,000 the counts at 1e-4 are the same on the two tasks (144-163 and 154-166). On
3BFF the R² falls with the count (0.934 and 0.906 compared with the control's 0.951); on CDDM it
falls by 0.003 and 0.011. At wd 0 the three 3BFF seeds spread from 252 to 367 and one lies inside
the control's range, so the pre-registered rule says "no change" where CDDM's rule says "increase"
(442-469); the means move the same way (305 compared with 246 on 3BFF, 453 compared with 337 on
CDDM).

### 2. CDDM rescaling, the paper's arm (entry _0903; control CDDM paper grid, 333-342 active, R² 0.834)

| cell | seed | active | R² (final) | task |
|---|---|---|---|---|
| `CDDM_target_ladder/EqType=h_N=1000_tgt=10.0` | 3366773033 / 803704226 / 511030482 | 196 / 183 / 169 | 0.819 / 0.813 / 0.824 | 16955457_9, 11, 10 |

**Fails both conditions.** Every seed has fewer active units than every control run (169-196
compared with 333-342; the mean, 183, is 0.54 of the control's 336), and the mean R² is 0.015
below the controls' (0.819 compared with 0.834), beyond the 0.01 tolerance. On the flip-flop the
same arm has 1.51 times the control's count at the floor (`NBitFlipFlop_target_ladder_c200000`,
the matrix); on CDDM it halves it.

### 3. CDDM optimiser comparison (entry _2256; control CDDM paper grid, 333-342 active, R² 0.834)

| arm | seeds | active | R² (final unless said) | task |
|---|---|---|---|---|
| Adam wd 0 (weight-decay ladder, F2 cache, estimate) | 955478765 / 1132724915 / 3370475588 | 469 / 448 / 442 | 0.833 / 0.834 / 0.831 | Della, 77fbd97 |
| AdamW wd 1e-6 | 1063251570 / 3161811021 / 4187879259 | 453 / 431 / 448 | 0.832 / 0.832 / 0.835 | 16957418_8, 7, 9 |
| SGD lr 0.03 | 1032671150 / 1083008248 / 1084166182 | 842 / 813 / 834 | 0.820 / 0.820 / 0.818 | 16957418_12, 11, 10 |
| SGD lr 0.1 | 2131628978 / 4208297148 / 2131637255 | 857 / 845 / 859 | 0.828 / 0.827 / 0.826 | 16957418_14, 13, 15 |
| SGD lr 0.3 | 1700903875 / 2107103656 / 3871697327 | 895 / 882 / 881 | 0.831 / 0.818 / 0.830 | 16957418_18, 16, 17 |
| SGD lr 1 | 3243312434 / 434938894 / 2550068419 | 846 / 821 / 767 | 0.817 / 0.826 / 0.819 | 16957418_19, 20, 21 |

| arm | CDDM: mean active, R² gap, class | 3BFF B5 at 75k (control 233-270, R² 0.951): mean active, R² gap, class |
|---|---|---|
| Adam wd 0 | 453, −0.001, **A** (more active units at no cost in R²) | 305, +0.006, C (one seed, 252, inside the control's range) |
| AdamW | 444, −0.001, **A** | 325, +0.007, A |
| SGD lr 0.03 | 830, −0.014, **B** (more active units only at lower R²) | 749, −0.077, B |
| SGD lr 0.1 | 854, −0.007, **A** | 595, −0.058, B |
| SGD lr 0.3 | 886, −0.007, **A** | 264, −0.051, C |
| SGD lr 1 | 811, −0.013, **B** | 243, −0.049 over the two runs that learned (one run at R² 0.578 excluded), C |

Every CDDM run learned (lowest R² 0.813). AdamW and Adam without weight decay have 1.3 times the
control's number of active units at the same R² (within 0.001), as AdamW does on the flip-flop.
Every SGD arm has 2.4-2.6 times the control's count (767-895 of 1,000 active), and its R² is
0.007-0.014 below the control's; at lr 0.1 and 0.3 that gap is within the 0.01 tolerance, so those
two arms are class A, and the pre-registered test ("every SGD arm whose count passes is in class
B") says **CDDM does not reproduce the 3BFF pattern**. What differs is the size of the R² gap, not
its sign: on the flip-flop SGD's R² is 0.05-0.08 below Adam's and only the two lowest learning
rates increase the count; on CDDM SGD's R² is 0.007-0.014 below Adam's and every learning rate from
0.03 to 1 increases the count. The 3BFF Adam wd 0 cell is class C here by the per-seed rule
(entry 2026-10-31_2010 called its mean, 305, "more at a higher R²"); its mean is 1.24 times the
control's.

### 4. CDDM rate penalty 0.1 alone and with the sparsity penalty 0.05 (entry _2256; control CDDM paper grid)

| cell | seed | active | R² (final) | task |
|---|---|---|---|---|
| rate 0.1 (`pen=frm0.1`) | 3986057758 / 3681082541 / 1954623883 | 1,000 / 1,000 / 1,000 | 0.825 / 0.829 / 0.827 | 16957419_4-6 |
| rate 0.1 + sparsity 0.05 (`pen=frm0.1_rws0.05`), 1 of 3 | 1037007987 | 1,000 | 0.823 | 16957419_7 |
| reference: rate 0.2 (F2 cache, 5 runs) | | 1,000 in all five | 0.828-0.830 (mean 0.829) | |
| reference: rate 0.2 + sparsity 0.05 (F2 cache, 5 runs) | | 1,000 in all five | 0.822-0.829 (mean 0.826) | |

**The rate penalty 0.1 works on CDDM**: all 1,000 units are active in each of the three seeds
(control 333-342), and the mean R², 0.827, is 0.007 below the controls' 0.834, within the 0.01
tolerance. The 0.2 cells of Figure 3a have the same count at R² 0.005 lower than the control.
**Rate 0.1 + sparsity 0.05, provisional (1 of 3 runs)**: 1,000 active, R² 0.823, a gap of 0.011
that fails the tolerance by 0.001; the verdict waits for tasks 16957419_8 and _9 (running). The
0.2 + 0.05 cells have R² 0.008 below the control.

### 5. 3BFF settings ladders (entry _2259; control B5 Adam wd 1e-6, 233-270 active, R² 0.951; the direction on CDDM is the same rule applied to the Figure 3 ladder and its own control)

| setting | runs read | active | R² (final) | verdict | CDDM at 75k | direction |
|---|---|---|---|---|---|---|
| σ_rec 0 | 3/3 (16957556_1-3) | 284, 289, 281 | 0.950, 0.951, 0.950 (gap −0.000) | **increase** (inside the second reference's 245-296) | 245-258 compared with 311-340 at σ_rec 0.05: decrease | differs |
| σ_rec 0.01 | 0/3 | | | not read | 249-279: decrease | |
| σ_rec 0.1 | 0/3 | | | not read | 378-412: increase | |
| spectral radius 0.5 | 2/3 (16957557_1, 2) | 295, 270 | 0.952, 0.952 (+0.001) | provisional: no change (270 equals the control's largest) | 355-368 compared with 315-347: increase | open |
| spectral radius 2.0 | 0/3 | | | not read | 308, 317: no change (2 of 3) | |
| λ_rwm 0.05 | 0/3 | | | not read | 434, 441: increase (2 of 3) | |
| λ_rwm 0.2 | 0/3 | | | not read | 445, 450: increase (2 of 3) | |
| λ_met 0.01 | 1/3 (16957559_7) | 351 | 0.949 (−0.002) | provisional: increase | 324-344 compared with 325-345: no change | open |
| λ_met 0.1, 1, 10 | 0/3 each | | | not read | 347-356 increase; 360-387 increase at R² −0.013; 321-342 no change at R² −0.046 | |
| λ_rws 0.05 alone | 1/3 (16957560_166) | 233 | 0.950 (−0.001) | provisional: no change (233 equals the control's smallest) | 248-281 compared with 320-334: decrease | open |
| softplus β 25 | 1/3 (16957561_1) | 244 | 0.949 (−0.002) | provisional: no change | 372-376 compared with 325-345: increase | open |
| leaky ReLU 0.01, shifted sigmoid, tanh | 0/3 each | | | not read | 348-385 increase; 198-234 decrease; no CDDM cell at 75k | |
| input-weight cap 0.3 | 1/3 (16957562_4) | 365 | 0.904 (−0.047) | provisional: increase | 390-400 compared with 332-340: increase at R² −0.004 | open |

Only σ_rec 0 is complete. Without activity noise the flip-flop networks have 281-289 active units
compared with the control's 233-270, at the same R²; the CDDM networks without noise have fewer
(245-258 compared with 311-340). The flip-flop counts lie inside the second reference's range
(245-296), so the result rests on which control is named; the pre-registration names the B5 cell.
The cap 0.3 seed has 365 active units at R² 0.904, a drop of 0.047, in line with yesterday's read
of caps 0.1, 0.5 and 2.0 (every one fails on R²); on CDDM the same cap drops R² by 0.004.

### 6. CDDM spectral radius and recurrent-weight penalty, third seeds (entry _2259; control the sr 1.2 cell, now 315, 333, 347 active, R² 0.833)

| cell | runs | active (new seed marked) | R² (final) | verdict |
|---|---|---|---|---|
| sr 0.5 | 3/3 | 355, **368** (16957558_1), 367 | 0.831, 0.835, 0.834 (+0.000) | **increase** |
| sr 2.0 | 2/3 | 308, 317 | 0.833, 0.836 (+0.001) | provisional: no change; third seed 16957558_3 running |
| λ_rwm 0.05 | 2/3 | 441, 434 | 0.832, 0.833 (−0.001) | provisional: increase; 16957558_4 running |
| λ_rwm 0.2 | 2/3 | 445, 450 | 0.826, 0.833 (−0.004) | provisional: increase; 16957558_5 pending |

The third sr 1.2 seed (315, task 16957558_2) widens the control's range downward from 333-347 to
315-347; with it, sr 2.0 (308, 317) is "no change" where the two-seed range said "decrease".

**Pending tasks (47 of 86), by cell.** Running at 08:05: `CDDM_std_g0_penalties_bias0` rate 0.1 +
sparsity 0.05 (16957419_8, 9); `NBitFlipFlop_std_g0_noise_bias0` σ_rec 0.01 (16957556_6-8) and
0.1 (_16, 17; _18 pending); `NBitFlipFlop_std_g0_recinit` sr 0.5 (16957557_3) and sr 2.0 (_7-9);
`CDDM_std_g0_recinit` sr 2.0 (16957558_3) and λ_rwm 0.05 (_4); `NBitFlipFlop_std_g0_metabolic_bias0`
λ_met 0.01 (16957559_8, 9) and 0.1 (_16); `NBitFlipFlop_std_penlong` λ_rws (16957560_167, 168);
`NBitFlipFlop_std_g0_activations_bias0` softplus (16957561_2, 3) and leaky ReLU (_4);
`NBitFlipFlop_screen3` cap 0.3 (16957562_5, 6). Pending (not started): 3BFF λ_rwm 0.05 and 0.2
(16957557_10-15), CDDM λ_rwm 0.2 (16957558_5), 3BFF λ_met 0.1 (16957559_17, 18), 1 (_25-27) and
10 (_34-36), leaky ReLU (16957561_5, 6), shifted sigmoid (_7-9), tanh (_10-12), σ_rec 0.1
(16957556_18). The tanh cell's saturation read (entry _2259) waits for those runs. The next
read-out is the same command once `sacct` says the tasks are complete, after an rsync of the new
folders; `data/della_batch1_status.tsv` is rebuilt from `sacct` and the logs first.

**Status board.** No paper claim changes. The interventions matrix's rows for 3BFF weight decay,
CDDM rescaling, the CDDM optimiser and the CDDM rate penalty 0.1 are answered by this entry (the
matrix file is not edited here); the settings rows stay open until the pending runs are read.
