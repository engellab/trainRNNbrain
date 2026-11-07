## 2026-11-06 18:30 — Uniform read-outs, Della batch 2: every supervised cell complete; the recurrent-weight penalty, the metabolic penalty at 0.1 and 1, the cap 0.3 and weight decay move the 3BFF count as on CDDM; activity noise, the activations, the sparsity penalty alone and spectral radius 0.5 do not; rate 0.1 + sparsity 0.05 works on CDDM; the tanh networks have 41-536 saturated units (issue #6)

**What was read.** The remaining 47 array tasks of the eleven Della arrays of entries
2026-11-05_0903, _2256 and _2259 (the first 39 are entry 2026-11-06_0900, "batch 1"), so every
cell of the supervised uniform programme is now complete: 86 of 86 tasks COMPLETED. The new run
folders were copied to `data/trained_RNNs/` as before (config, loss record, participation trace,
final weights). `data/della_batch2_status.tsv` is the status table rebuilt from `sacct` and the job
logs by the new `trainRNNbrain/experiments_and_analysis/della_batch_status.py` (the index decode of
every array and the join, which batch 1 did from a scratch snippet; run on batch 1's inputs it
reproduces `data/della_batch1_status.tsv` byte for byte). One correction to batch 1's table: the
launcher writes the metabolic values 1 and 10 as `LmbdMet=1.0` and `LmbdMet=10.0`, where entry
_2259's table and batch 1's decode wrote `LmbdMet=1` and `LmbdMet=10`; those tasks were pending in
batch 1, so no run was misread, and `read_uniform_cells_2.py` now names the folders as they are.

**Script.** `read_uniform_cells_2.py` as in batch 1, with the tanh saturation read added: on a
noise-free pass over the held-out batch a unit is responsive when the mean slope of its activation,
mean(1 − r²) over every time step and trial, exceeds 0.5, and saturated otherwise (pinned near +1
or −1, so its activation passes no gradient), the test of `fig_tanh_saturation.py`, with that
script's trace proxy (participation p < 1.6 at the last probe) beside it. Checked on the three
CDDM tanh networks of `CDDM_std_g0_tanh` (150,000 iterations): 13, 688 and 913 responsive units by
the slope test, 8, 489 and 911 by the proxy, the numbers that script's docstring gives (13 on the
calibration network; 8, 489, 911 for the proxy). Output `data/read_uniform_cells_2.json`
(overwritten; same layout, every arm now complete) and the tables below.

**Read-out, as pre-registered**, the same as batch 1: active units at the probe nearest 75,000
(74,900, the last probe), held-out R² on the final weights (36 draws), the B5 Adam wd 1e-6 cell as
the 3BFF control (233-270 active, R² 0.951; second reference the ksweep control, 245-296), the
Figure 3 ladders as the CDDM comparison, read with the same rule against their own control. A
setting changes the number of active units if all three seeds lie outside the control's range on
the same side; a cell with mean R² below 0.7 is reported, not interpreted. No new run is broken: no loss has a
non-finite value, and the most gradient updates the trainer skipped in any run is 1,489 of 75,000
(2.0%, softplus seed 357642158; batch 1's most was 499), under the 5% exclusion bar. Every cached
control count equals the trace on disk, and the final read-out reproduces the counts and R² of the
43 runs scored earlier in the day run by run.

**Controls, read at 75,000** (unchanged from batch 1): 3BFF B5 Adam wd 1e-6, 270, 234, 233 active
(mean 246), R² 0.951; the ksweep second reference 296, 245, 283, R² 0.951; CDDM paper grid 342,
334, 333, R² 0.834; CDDM sr 1.2, 315, 333, 347, R² 0.833. Batch 1's complete cells (3BFF weight
decay, CDDM rescaling, the CDDM optimiser, CDDM rate 0.1, 3BFF σ_rec 0, CDDM sr 0.5) are unchanged
and not repeated; their verdicts stand.

### 1. CDDM rate 0.1 + sparsity 0.05, now complete (control CDDM paper grid, 333-342 active, R² 0.834)

| cell | seed | active | R² (final) | task |
|---|---|---|---|---|
| rate 0.1 + sparsity 0.05 | 1037007987 / 3331603136 / 1077378740 | 1,000 / 1,000 / 1,000 | 0.823 / 0.829 / 0.827 | 16957419_7, 8, 9 |

**Works on CDDM**: all 1,000 units active in each seed, mean R² 0.826, 0.008 below the controls'
0.834 (batch 1's single seed, R² 0.823, had failed the tolerance by 0.001). The 0.2 + 0.05 cells
of Figure 3a have the same count at R² 0.008 below the control. So on CDDM the rate penalty works
alone and with the sparsity penalty, at 0.1 and at 0.2.

### 2. 3BFF settings ladders, complete (control B5 Adam wd 1e-6, 233-270 active, R² 0.951; the direction on CDDM is the same rule applied to the Figure 3 ladder and its own control)

| setting | seeds | active | R² (final) | verdict | CDDM at 75k (its control) | direction |
|---|---|---|---|---|---|---|
| σ_rec 0 | 2164248961 / 2095512490 / 2112247374 | 284 / 289 / 281 | 0.950 / 0.951 / 0.950 (−0.000) | **increase** | 245-258 (311-340): decrease | differs |
| σ_rec 0.01 | 114095993 / 2037472196 / 2113571904 | 263 / 276 / 289 | 0.950 / 0.951 / 0.952 (+0.000) | no change | 249-279: decrease | differs |
| σ_rec 0.1 | 2750111374 / 3087851235 / 2644588571 | 239 / 263 / 244 | 0.950 / 0.950 / 0.950 (−0.001) | no change | 378-412: increase | differs |
| spectral radius 0.5 | 3901667487 / 2286017443 / 3123952635 | 346 / 295 / 270 | 0.952 ×3 (+0.001) | no change (270 equals the control's largest) | 355-368 (315-347): increase | differs |
| spectral radius 2.0 | 2142213740 / 1240111490 / 651190503 | 208 / 248 / 280 | 0.949 / 0.951 / 0.951 (−0.000) | no change | 304-317: no change | same |
| λ_rwm 0.05 | 3099110300 / 534558521 / 1187087490 | 540 / 516 / 520 | 0.944 / 0.945 / 0.945 (−0.006) | **increase** | 434-459: increase | same |
| λ_rwm 0.2 | 3304714684 / 1051263503 / 2390508989 | 567 / 532 / 543 | 0.944 / 0.944 / 0.945 (−0.006) | **increase** | 436-450: increase | same |
| λ_met 0.01 | 2413978074 / 2274079521 / 2618310007 | 331 / 346 / 351 | 0.949 ×3 (−0.002) | **increase** | 324-344 (325-345): no change | differs |
| λ_met 0.1 | 1962395024 / 3808897215 / 734802261 | 281 / 352 / 347 | 0.944 / 0.947 / 0.945 (−0.006) | **increase** | 347-356: increase | same |
| λ_met 1 | 1085794566 / 3969087498 / 765993234 | 312 / 292 / 285 | 0.936 ×3 (−0.015) | **increase** | 360-387 (R² −0.013): increase | same |
| λ_met 10 | 1895021074 / 485230843 / 4052550188 | 197 / 188 / 206 | 0.903 / 0.904 / 0.903 (−0.048) | **decrease** | 321-342 (R² −0.046): no change | differs |
| λ_rws 0.05 alone | 3596355676 / 732025745 / 3382294632 | 233 / 221 / 221 | 0.950 / 0.950 / 0.951 (−0.001) | no change (233 equals the control's smallest) | 248-281 (320-334): decrease | differs |
| softplus β 25 | 289852694 / 1945563352 / 357642158 | 227 / 244 / 261 | 0.946 / 0.949 / 0.949 (−0.003) | no change | 372-376 (325-345): increase | differs |
| leaky ReLU 0.01 | 4276078699 / 1786743050 / 1435502468 | 268 / 315 / 349 | 0.951 / 0.951 / 0.952 (+0.001) | no change (268 inside the range) | 348-385: increase | differs |
| shifted sigmoid | 766522529 / 1158212453 / 1689362295 | 262 / 257 / 288 | 0.960 / 0.959 / 0.961 (+0.009) | no change | 198-234: decrease | differs |
| tanh | 2226065994 / 3176447870 / 1147288988 | 1,000 / 1,000 / 1,000 | 0.929 / 0.937 / 0.943 (−0.014) | increase by participation (every tanh unit scores active); see the saturation read | no CDDM cell at 75k | - |
| input-weight cap 0.3 | 1533507216 / 2121980320 / 3354746598 | 326 / 365 / 421 | 0.903 / 0.904 / 0.908 (−0.046) | **increase** | 390-400 (332-340, R² −0.004): increase | same |

Every mean R² is above 0.7, so every count is interpreted. Same direction on both tasks: weight
decay 1e-5 and 1e-4 (decrease, batch 1), the recurrent-weight penalty at 0.05 and 0.2 (increase:
516-567 active on 3BFF, 2.1-2.2 times the control's mean, at R² 0.006 lower; 434-459 on CDDM at
R² 0.002-0.003 lower), the metabolic penalty at 0.1 and 1 (increase on both; on 3BFF at R² 0.006
and 0.015 lower), the input-weight cap 0.3 (increase on both; on 3BFF 326-421 active at R² 0.046
lower, on CDDM 390-400 at R² 0.004 lower) and spectral radius 2.0 (no change on both). Different: the
metabolic penalty 10 (decrease on 3BFF, 188-206 active at R² 0.048 lower; no change on CDDM at
R² 0.046 lower), activity noise at every value (on 3BFF σ_rec 0 increases the count to 281-289 and 0.01 and 0.1
change nothing; on CDDM the count rises with the noise, 245-258 at 0, 378-412 at 0.1), spectral
radius 0.5 (no change on 3BFF, where one seed sits at the control's largest count; increase on
CDDM), the metabolic penalty 0.01 (increase on 3BFF, 331-351; no change on CDDM), the sparsity
penalty alone (no change on 3BFF, where one seed sits at the control's smallest; decrease on
CDDM), softplus and leaky ReLU (no change on 3BFF; increase on CDDM) and the shifted sigmoid (no
change on 3BFF, at the highest R² of any cell, 0.959-0.961; decrease on CDDM). Several of the
3BFF "no change" verdicts come from one seed inside a wide control range (233-270) while the other
two lie above it: leaky ReLU 268, 315, 349; spectral radius 0.5 270, 295, 346; the rule needs all
three outside. On the flip-flop, the only settings that move every seed are the two penalties on
the recurrent weights and on the rates, the cap, weight decay above 1e-6, and no noise.

**The tanh cell's saturation.** By participation all 1,000 units of each tanh network are active,
as in every tanh network of this project (`fig_tanh_saturation.py`: tanh has no zero region, so no
unit scores silent). On a noise-free pass over the held-out batch the slope test (a unit is
responsive when mean(1 − r²) over every step and trial exceeds 0.5) gives 464, 959 and 925
responsive units, so 536, 41 and 75 of 1,000 are saturated (pinned near +1 or −1); the trace proxy
(p < 1.6 at the last probe) gives 449, 955 and 918. The median slope is 0.42 in the first network
and 0.94-0.96 in the other two; |r| exceeds 0.9 on 36% of (unit, step, trial) samples in the first
and on 3-4% in the other two. The CDDM tanh networks at 150,000 iterations have 987, 312 and 87
saturated units by the same test. The three flip-flop tanh networks reach R² 0.929-0.943, 0.008-0.022
below the ReLU control's 0.951.

### 3. CDDM spectral radius and recurrent-weight penalty, complete with the third seeds (control sr 1.2, 315, 333, 347 active, R² 0.833)

| cell | seeds (new one marked) | active | R² (final) | verdict |
|---|---|---|---|---|
| sr 0.5 | 3896335710 / **2167248987** / 1550227334 | 355 / 368 / 367 | 0.831 / 0.835 / 0.834 (+0.000) | **increase** |
| sr 2.0 | 902800158 / **2657990103** / 4033149549 | 308 / 304 / 317 | 0.833 / 0.832 / 0.836 (+0.000) | no change (304-317 against 315-347: 317 inside) |
| λ_rwm 0.05 | 3982450458 / **1068748794** / 628548698 | 441 / 459 / 434 | 0.832 / 0.829 / 0.833 (−0.002) | **increase** |
| λ_rwm 0.2 | 2190601718 / **2472667323** / 325260688 | 445 / 436 / 450 | 0.826 / 0.833 / 0.833 (−0.003) | **increase** |

The third seeds leave every two-seed verdict of batch 1 in place except sr 2.0, which the third
sr 1.2 seed (315) had already moved from "decrease" to "no change"; its own third seed (304) is
below the control's range but 317 is not.

**Settings × tasks, the direction at 75,000 (3BFF | CDDM), every cell complete.** Weight decay 0:
no change | increase; 1e-5: decrease | decrease; 1e-4: decrease | decrease. Activity noise 0:
increase | decrease; 0.01: no change | decrease; 0.1: no change | increase. Spectral radius 0.5:
no change | increase; 2.0: no change | no change. Recurrent-weight penalty 0.05: increase |
increase; 0.2: increase | increase. Metabolic penalty 0.01: increase | no change; 0.1: increase |
increase; 1: increase | increase; 10: decrease | no change. Sparsity penalty alone 0.05: no change |
decrease. Softplus: no change | increase; leaky ReLU: no change | increase; shifted sigmoid: no
change | decrease; tanh: 1,000 active by participation on 3BFF (41-536 saturated), no CDDM cell at
75k. Input-weight cap 0.3: increase | increase. Optimiser (batch 1): AdamW class A on both tasks;
Adam wd 0 class A on CDDM and class C on 3BFF by the per-seed rule (one seed inside the control's
range; its mean, 305, is above the control's 246); SGD 0.03 class B on both; SGD 0.1 and 0.3 class
A on CDDM (R² 0.007 lower) and class B and C on 3BFF (R² 0.05-0.06 lower); SGD 1 class B on CDDM,
C on 3BFF. Of the 19
setting values with a cell on both tasks, 8 move the count the same way (weight decay 1e-5 and 1e-4,
spectral radius 2.0, the recurrent-weight penalty at both values, the metabolic penalty at 0.1 and
1, the cap) and 11 do not. `fig_paper_F3_tasks.py` draws these from the JSON.

**Pending.** Nothing: 86 of 86 tasks completed, 86 run folders on the Mac, every arm at 3 of 3
seeds. The uniform programme's supervised cells are complete; the maze cells are another agent's
(entries of 6 Nov under issue #6).

**Status board.** No paper claim changes. The interventions matrix's 3BFF settings rows, the CDDM
third seeds and the CDDM rate 0.1 + sparsity 0.05 row are answered by this entry (the matrix file
is not edited here).
