## 2026-11-08 08:59 — Round 18 training package, final results by the committed rules: technique verdicts on all three tasks, G11, G12, G17, G18

Rules applied:
- `2026-11-07_0110_round18-training-rules.md` (G11, G12, G17, G18);
- `2026-11-07_0136_round18-parity-rules.md` (the parity runs);
- `2026-11-07_0715_round18-g17-mirror-rules.md` (G17's second maze, the first maze mirrored left to right).

This entry completes `2026-11-07_0707_round18-training-interim-results.md`, whose G11, G12 and G18 numbers
are unchanged here.

The reads use `trainRNNbrain/experiments_and_analysis/round18_training_read.py` (f24b1b9):
- `maze`, `exponent` and `settings` on the Mac, on the rsynced trees;
- `cddm` on Della, CPU job 17046916.

The outputs are in `data/round18_training/`. Run folders were copied to the Mac without the optimizer-state
files. Every job of the package has ended.

**Terms.**
- *3BFF*: the 3-bit flip-flop task. *CDDM*: context-dependent decision making. *Maze*: the egocentric
  maze task trained by PPO (proximal policy optimisation).
- *Active unit*: participation p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05·q95(p).
- *Probe success*: the share of 128 noise-free Maze test episodes that reach the cheese.
- *Learned point*: the first probe from which probe success stays at or above 0.9 for five probes, in a run
  whose last probe is also at or above 0.9.
- *σ_w*: the relative s.d. of the multiplicative noise on each synapse at every step.
- *The criterion* (the paper's, Methods M09). A technique value meets it on a task when every network
  trained with it has more active units than every control network, and the networks' mean performance
  is at least the controls' mean minus a tolerance. The tolerance is 0.01 in held-out R² on 3BFF and
  CDDM, read at 75,000 iterations. On Maze it is 0.05 in success at update 2,000, no network may end
  below 0.9, and the controls are uniform-series seeds 1-3 (decision #194: 426, 449 and 415 active units,
  mean success 0.898).

### Jobs and run times

| Cell | Cluster, job | Runs | Time per run |
|---|---|---|---|
| G11 | Spock 7397701 | 10 | 4.0-5.3 h |
| G12 | Spock 7397702 | 3 | 5.5-5.6 h |
| G17 | Spock 7397983 (old), 7397984 (fresh) | 5 + 3 | 4.9-6.4 h |
| G18 | Della 17030263 (old), 17030264 (fresh) | 3 + 3 | 2.8-3.5 h |
| CDDM parity | Della 17031165 (σ_w), 17031166 (dropout), 17031167 (rescaling) | 12 | 3.8-7.9 h |
| Maze parity | Spock 7397761 (σ_w), 7397762 (dropout, rescaling) | 12 | σ_w 14.4-22.1 h; others 5.4-5.8 h |
| Calibration | Spock 7397777 (CPU) | 39 | 300 updates each at N = 100 |

The two σ_w 0.1 runs at seeds 2 and 3 took 21.9 and 22.1 h of their 24 h limit. Rescaling target 4 at
seed 2 on Maze was stopped by the rollback guard at update 800. The other 50 training runs finished
their budgets.

### The parity runs, per value

**CDDM, at 75k.** Controls: `CDDM_paper_grid` arm control, with 342, 334 and 333 active units and R²
0.8312, 0.8350 and 0.8349 (mean 0.8337). A value passes if every network has more than 342 active units
and the mean R² is at least 0.8237. The SE (standard error) of the difference uses the network-to-network
spread of both groups, sqrt(var_arm/3 + var_ctrl/3), as G2 computed it.

| Value | Active units | R² | Mean R² minus control mean (SE) | Result at 0.005 / 0.01 / 0.02 |
|---|---|---|---|---|
| σ_w 0.3 | 356, 360, 380 | 0.8345, 0.8344, 0.8345 | +0.0008 (0.0012) | pass / pass / pass |
| σ_w 0.1 | 331, 316, 333 | 0.8314, 0.8326, 0.8317 | −0.0018 (0.0013) | fail / fail / fail (count) |
| read-out dropout, rate 0.05, β 4 | 326, 334, 350 | 0.8305, 0.8287, 0.8316 | −0.0035 (0.0015) | fail / fail / fail (count) |
| rescaling, target 4.0 | 231, 230, 224 | 0.8273, 0.8274, 0.8266 | −0.0066 (0.0013) | fail / fail / fail (count) |

**The σ_w 0.3 pass on CDDM.**
- The mean R² is 0.0008 above the controls' mean, with an SE of 0.0012. That difference is within one SE
  of zero and 8.7 SEs inside the tolerance of 0.01.
- The count clause holds by 14 units: the fewest active units in a σ_w 0.3 network is 356, against the
  controls' largest, 342.
- The pass also holds against the second control, `CDDM_std_g0_bias0_c75000` (5 networks, 320-334
  active units, measured R² 0.831-0.834).
- One difference is not the technique: the σ_w 0.3 networks drew W_out independently of W_inp, and the
  controls used the old shared draws (parity rules entry).

**Maze, at update 2,000.** A value passes if every network has more than 449 active units, the mean
success is at least 0.848, and no network ends below 0.9. Index mapping checked from each run's saved
config: indices 115-117 are σ_w 0.1 seeds 1-3, 118-120 are σ_w 0.3 seeds 1-3.

| Value | Active units | Success | Result |
|---|---|---|---|
| σ_w 0.3 | 400, 432, 396 | 0.984, 1.000, 0.977 | **fails**: no network above 449 |
| σ_w 0.1 | 418, 416, 342 | 1.000, 0.984, 1.000 | **fails**: no network above 449 |
| read-out dropout, rate 0.05, β 4 | 455, 373, 420 | 0.930, 0.961, 1.000 | **fails**: two networks at or below 449 |
| rescaling, target 4 | 352; stopped at update 800; 197 | 0.812; 0.008 for the third | **fails** every clause |

The same four results hold against the five learned controls (394-449), the sensitivity check named in
the rules. Their largest count is also 449, and the three Maze σ_w 0.3 networks (396-432) lie inside
that range.

### Technique verdicts on all three tasks

Tolerance 0.01 on 3BFF and CDDM, 0.05 in success on Maze. The 3BFF column is G2's per-network test
(`data/round18/g1g2.json`). The CDDM and Maze columns combine the paper's existing values
(`fig_paper_F5_tasks.py`, `data/read_maze_arms.json`) with the parity values above. P passes, − fails.

**(a) The one-value rule.** A technique passes a task if any one of its values passes.

| Technique | 3BFF | CDDM | Maze | All three |
|---|---|---|---|---|
| prune-and-duplicate | P (9 of 9 values) | P | P | **yes** |
| rate penalty | − (P at 0.02 via λ 0.01) | P | P | no (yes at 0.02) |
| rate penalty + sparsity | − | P | − | no |
| noisy synapses | P (σ_w 0.1 and 0.3) | P (σ_w 0.3; σ_w 1.0 and 0.1 fail) | − (σ_w 1.0, 0.3 and 0.1 fail) | no |
| read-out dropout | P (4 of 13 values) | P (rate 0.2, β 4; rate 0.05 fails) | − (rate 0.2 and rate 0.05 fail) | no |
| rescaling | P (target 4.0, 0.7 SE inside) | − (targets 10 and 4.0 fail) | − (all variants and target 4 fail) | no |

**(b) Matched values.** A value counts here only if it was run on all three tasks; the parity runs added
four such values.

| Value | 3BFF | CDDM | Maze |
|---|---|---|---|
| prune-and-duplicate, copy noise 1, rate 0.025 | P | P | P |
| rate penalty 0.1 | − | P | − |
| rate penalty 0.1 + sparsity 0.05 | − | P | − |
| noisy synapses σ_w 1.0 | − | − | − |
| noisy synapses σ_w 0.3 (parity) | P | P | − |
| noisy synapses σ_w 0.1 (parity) | P | − | − |
| read-out dropout 0.2, β 4 | − | P | − |
| read-out dropout 0.05, β 4 (parity) | P | − | − |
| rescaling, target 10× | − | − | − |
| rescaling, target 4.0 (parity) | P | − | − |

**The headline.** Noisy synapses do not pass on all three tasks at σ_w 0.3 under the one-value rule.
At σ_w 0.3 they pass on 3BFF and on CDDM, and fail on Maze, where none of the three networks has more
active units than the largest control (449). Under both rules, prune-and-duplicate is the only technique
that meets the criterion on all three tasks at tolerance 0.01. The same holds at 0.005, because no parity
value passes on Maze at any tolerance; each fails the count clause. At 0.02 the rate penalty also passes
under (a), through λ 0.01 on 3BFF, as G2 found. The parity runs settle the two provisional points of the
status board without changing the headline.

What does change, from the per-value tables:
- Under (a), noisy synapses now pass on CDDM (σ_w 0.3), where the paper's value σ_w 1.0 fails at 0.01.
- Under (b), σ_w 0.3 passes on two of three tasks. Before the parity runs, the one matched noisy-synapse
  value, σ_w 1.0, passed on none of the three at 0.01.

### G12: the third learned Maze control, and the setting verdicts

All three new N = 1,000 controls learned:

| Seed | Success at update 2,000 | Active units at update 2,000 |
|---|---|---|
| 4 | 0.953 | 394 |
| 5 | 0.938 | 398 |
| 6 | 1.000 | 417 |

All three are at or below 417. By the rule, weight decay 1e-5 on Maze (370-417 active units) no longer
lies below every learned control, and the claim that only weight decay changes the number of active units
on all three tasks loses its Maze direction. The kept snapshots are complete: 20 per run, with the first
one at or after the learned point at most 70 updates after it.

**Every Maze setting verdict, re-read against the five learned controls (394-449 at update 2,000).**
`read_maze_arms.py` ran on a control root holding only the 2,000-update control cell, so the G11
continuations do not move the controls to update 4,000. A setting decreases or increases the number of
active units when all its learned networks lie below or above the control range. Four verdicts change,
from "decreases" to "no change":

| Setting | Active units (3 networks) |
|---|---|
| weight decay 1e-5 | 402, 417, 370 |
| spectral radius 0.5 | 338, 393, 414 |
| sparsity penalty 0.05 | 399, 378, 386 |
| metabolic penalty 0.01 | 537, 413, 379 |

These still change the number of active units:
- weight decay 1e-4 decreases it (378 in its one learned network);
- the metabolic penalty at 1 decreases it (390, 336, 359);
- input norm 0.5 increases it (577, 593, 549).

The other settings are unchanged: no change, no direction, or incomplete, as in `data/read_maze_arms.json`.
The technique verdicts of that file are unchanged as well. Output: `data/round18_training/round18_settings.json`.

### G11: the Maze size series at one budget, and the exponent refit

All ten continued runs (N = 250 seeds 1-4, N = 500 seeds 1-3, N = 1,000 seeds 1-3) learned by update 4,000.
In every run the number of active units at update 4,000 is at or below the number at 2,000, and below the
number at its learned point. Per-run numbers are in the interim entry.

**The exponent refit at N ≥ 500 on equal budgets.** Every size from N = 250 to 4,000 now has 4,000 updates.
The fit uses every learned uniform-series ReLU run whose record ends at update 4,000:

| N | 500 | 1,000 | 2,000 | 4,000 |
|---|---|---|---|---|
| Networks | 3 | 3 | 3 | 1 |

G12's 2,000-update controls are left out. The fit is `power_fit` and `boot_exponent` of
`fig_paper_F2_size.py`. **This fit replaces the mixed-budget fit** (2,000 updates at N ≤ 1,000, 4,000 above;
`data/round18/g10_maze_exponent.json`), which is reported beside it:

| Read-out point | Equal budgets, b (95% bootstrap) | Mixed budgets, b (95% bootstrap) |
|---|---|---|
| Learned point, N ≥ 500 | 0.351 (0.318-0.384) | 0.345 (0.314-0.381) |
| End of training, N ≥ 500 | 0.491 (0.463-0.520) | 0.410 (0.378-0.444) |
| Learned point, every size | 0.482 (0.444-0.521) | 0.473 (0.436-0.512) |
| End of training, every size | 0.554 (0.529-0.580) | 0.516 (0.486-0.548) |

Mean active units:
- at the learned point: 273, 465, 455 and 619 at N = 500, 1,000, 2,000 and 4,000;
- at update 4,000: 231, 360, 431 and 676.

N = 4,000 still has one learned network, because seeds 4 and 5 (G19, another agent's) were not on the Mac
at this read. The end-of-training exponent rises from 0.41 to 0.49 because the 2,000 extra updates lower
the counts at N ≤ 1,000. At N = 1,000 the end mean falls from 438 (two learned networks at update
2,000) to 360 (three at update 4,000). Output:
`data/round18_training/round18_exponent.json`.

### G17: the Maze transfer test on the mirrored maze

There are 5 old networks: uniform series seeds 1 and 2 and G12 seeds 4-6, each at update 2,000 on the first
maze, and every one learned that maze. There are 3 fresh networks, seeds 1-3. Each run had 2,000 updates on
the mirrored maze.

Columns:
- *S₀*: the units inactive in the probe after update 1.
- *Revived*: units of S₀ active in the first stored participation vector at or after the learned point.
- *Active in A\**: the active count in that stored vector; *Active at the learned probe*: the probe's count.

| Network | Learned at update | Active at transfer | S₀ | Revived (share of S₀) | Newly inactive | Active in A* | Active at the learned probe | S₀ active at 2,000 | Active at 2,000 |
|---|---|---|---|---|---|---|---|---|---|
| old s1 | 430 | 432 | 568 | 89 (15.7%) | 88 | 433 | 442 | 76 | 361 |
| old s2 | 410 | 430 | 570 | 104 (18.2%) | 111 | 423 | 431 | 85 | 375 |
| old s4 | 420 | 426 | 574 | 96 (16.7%) | 107 | 415 | 400 | 99 | 376 |
| old s5 | 1,020 | 394 | 606 | 83 (13.7%) | 96 | 381 | 390 | 84 | 361 |
| old s6 | 1,810 | 418 | 582 | 91 (15.6%) | 118 | 391 | 388 | 92 | 389 |
| fresh s1 | 1,080 | 632 | 368 | 162 | 341 | 453 | 461 | 163 | 411 |
| fresh s2 | 1,440 | 627 | 373 | 163 | 336 | 454 | 466 | 166 | 442 |
| fresh s3 | 1,330 | 582 | 418 | 130 | 332 | 380 | 383 | 138 | 378 |

Each row adds up: active at transfer − newly inactive + revived = active in A*.

**Gate fails: no B4 verdict.** The fresh networks have 437 ± 47 active units at their learned point, 17
above the old networks' 420 at transfer; the bar is 3 SD = 140. The rules entry anticipated this from the
known Maze counts, and B4 then gives no verdict.

**Per-unit result, which the rules said to report when the gate fails (not a B4 verdict).** In every old
network, 13.7-18.2% of S₀ is active at the learned point. That share is above B4's 10% bar and below its
30% bar in all five networks. The old networks' count at the learned point (410 mean) is within 1 SD of
the fresh networks' (437).

**Difficulty check (no threshold).** Fresh seeds 1-3 start from the same weights as uniform-series seeds
1-3. On the mirrored maze they learn at updates 1,080, 1,440 and 1,330. On the first maze the same seeds
learned at 1,050 and 760 and had not learned by update 2,000. Three of the five old networks re-learn the
mirrored maze by update 430; the other two take 1,020 and 1,810 updates.

### G18: the CDDM transfer test (unchanged from the interim entry)

The new task is CDDMHyper: CDDM's six inputs, with five outputs (the cued choice, the motion sign, the
colour sign and their product). There are 3 old networks (200k iterations on CDDM) and 3 fresh networks,
each trained 75k iterations on the new task.

- **Gate passes.** The fresh networks have 417 ± 8 active units at their floor crossing; the old networks
  have 265 at transfer.
- **Verdict at 1.05x the floor (binding): between the bars.** The old networks revive 12%, 12% and 25% of
  their inactive units. They reach the floor at 2,000-4,890 iterations, against 24,906-28,578 for the fresh
  networks. The 1.10x and 1.20x margins give the same verdict.
- **At 75k (not binding): "do not come back".** The old networks keep 4.8-6.0% of those units active.
  They end with 251-257 active units, against 349-362 for the fresh networks.
- R² is the same in both groups: 0.719-0.727 against 0.719-0.726.

### Status board

`docs/paper_audit.md` gets one paragraph covering:
- the technique verdicts under both rules, with the headline unchanged;
- noisy synapses passing on 3BFF and CDDM under (a);
- the four Maze setting verdicts that change;
- the equal-budget exponent, which replaces the mixed-budget fit;
- G17's failed gate and its per-unit share;
- G18.

The provisional notes of the C6 round-18 paragraph are settled by these reads.
