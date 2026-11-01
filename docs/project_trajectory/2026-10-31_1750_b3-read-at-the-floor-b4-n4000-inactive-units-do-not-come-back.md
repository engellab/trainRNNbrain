## 2026-10-31 17:50 — B4 at N = 4000: inactive units do not come back. B3 at the floor: the rule's literal verdict is "supports", resting only on its stability clause

**How it was read.** Della CPU job 16553903 (16 CPUs, 67 min, finished 31 Oct 20:38 Della time) from
`~/trainRNNbrain_c1ana` checked out at ed11dfa. It ran `read_b2_b4.py` and then
`read_b3_b4_4000.py --b4` (both in `trainRNNbrain/experiments_and_analysis/`). Output: Della
`~/b3b4_read/` (`read_b2_b4.json`, `read_b3.json`, `b4_verdict_4000.json`, `B3B4read.16553903.out`).
Every run is read where its noisy training loss reaches 1.05× its own fitted floor (read-out
79932ef; a penalised run's loss includes the penalty) and, for B3, also at a fixed 75,000
iterations. R² is held-out R² under the training noise, 36 draws, on the first kept snapshot at or
after the read point. PR (participation ratio) is the covariance participation ratio of noise-free
rates over the active units (`run_dims.measure`), averaged over 8 task batches. How the rules are
applied was fixed before the read in ptolmachev/dead-relu-paper#2.

The read reproduces the 31 Oct entry (2026-10-31_1532) run for run: B2, and B4 at N = 1000.

### B4 at N = 4000: inactive units do not come back

Rules: 2026-10-15_1342. S₀ = units inactive at iteration 0 of the new task (the 3-bit hyper
flip-flop); revived = units of S₀ active at the first participation probe at or after the crossing.
Source networks: the N = 4000 flip-flop controls trained 200k iterations (old) or stopped at their
floor (young); fresh networks are untrained. Della jobs 15655301-03, 100k iterations each.

| Source | Crossing | Active at transfer | Inactive at transfer (S₀) | Revived | Newly inactive | Active at the floor | R² at the floor |
|---|---|---|---|---|---|---|---|
| old | 22.4k | 317 | 3683 | 179 (4.9%) | 32 | 464 | 0.912 |
| old | 11.8k | 352 | 3648 | 284 (7.8%) | 23 | 613 | 0.912 |
| old | 17.7k | 387 | 3613 | 82 (2.3%) | 23 | 446 | 0.913 |
| young | 13.8k | 588 | 3412 | 140 (4.1%) | 61 | 667 | 0.911 |
| young | 33.3k | 502 | 3498 | 245 (7.0%) | 113 | 634 | 0.912 |
| young | 40.7k | 544 | 3456 | 166 (4.8%) | 136 | 574 | 0.914 |
| fresh | 43.4k | 3756 | 244 | 61 | 2861 | 956 | 0.909 |
| fresh | 55.4k | 3807 | 193 | 51 | 3033 | 825 | 0.912 |
| fresh | 56.1k | 3775 | 225 | 55 | 3024 | 806 | 0.911 |

All nine runs have a floor read and all nine R² values are measured on kept snapshots. Each row's
counts add up (active at transfer − newly inactive + revived = active at the floor).

**Gate passes:** fresh networks reach 862 ± 82 active (mean ± SD) at their floor, 510 above the old
networks' 352 at transfer; the bar is 3 SD = 245.
**Verdict: inactive units do not come back.** Every old network revives under 10% of S₀ (2.3-7.8%),
and the old networks end 355 units below the fresh ones (508 against 862), past the 3-SD bar of
245. I computed the gate and verdict by hand from the printed rows before `b4_verdict` printed its
own; both agree on every number. As at N = 1000, learning the new task costs nothing in R²: old,
young and fresh networks all reach 0.909-0.914. Young networks revive 4.1-7.0% of their inactive
units, inside the old networks' range; at N = 1000 they revived more (11-12% against 5-9%).

### B3: rate penalty × synaptic noise × rescaling at N = 1000

Cells: none = `NBitFlipFlop_c1_snapshots_c200000/EqType=h_N=1000_arm=control`; the rest are
`NBitFlipFlop_remedy_grid/EqType=h_N=1000_arm=<arm>` with rate = `frm0.03` (λ = 0.03), noise =
`sw1.0` (σ_w = 1.0), rescale = `rsc10`. Three seeds per cell. Two runs diverged (folder `nan_*`) and
count as found but not read: rate+rescale s498785141 and rate+noise+rescale s2773443343. Updates
skipped are from the training logs (Della `~/trainRNNbrain_logs/B3fact.<job>_<task>.out`,
`B3pilot.15598508_*`).

**At the floor (the rule binds here).**

| Cell | Runs read | Active at the floor | R² at the floor | PR | Updates skipped |
|---|---|---|---|---|---|
| none | 3 of 3 | 344, 286, 264 | 0.950-0.951 | 4.4-4.9 | not checked |
| rate | 0 of 3 | — | — | — | 1.2-4.6% |
| noise | 3 of 3 | 516, 594, 557 | 0.928-0.930 | 6.9-7.3 | no spike line logged |
| rescale | 0 of 3 | — | — | — | 1.3-3.4% |
| rate+noise | 2 of 3 | 976, 964 | 0.889; −27.5 | 10.7; 1.3 | 1.0-1.5% |
| rate+rescale | 1 of 3 | 787 (read at 4.8k) | 0.849 | 8.0 | 44-67% |
| noise+rescale | 2 of 3 | 466, 419 | 0.884, 0.880 | 7.3, 6.2 | 0.4-1.1% |
| rate+noise+rescale | 2 of 3 | 321 (read at 3.9k), 490 (at 2.1k) | −74.4; 0.844 | 1.0; 8.7 | 8-11% |

**At 75,000 iterations (the 16 Oct same-size rule; reported, not binding).**

| Cell | Runs read | Active at 75k | R² at 75k | PR |
|---|---|---|---|---|
| none | 3 of 3 | 309, 293, 259 | 0.951-0.952 | 4.0-4.8 |
| rate | 3 of 3 | 722, 606, 844 | 0.923-0.932 | 9.8-12.5 |
| noise | 3 of 3 | 518, 597, 571 | 0.930-0.934 | 6.6-7.3 |
| rescale | 3 of 3 | 408, 409, 353 | 0.936-0.941 | 5.2-7.8 |
| rate+noise | 3 of 3 | 948, 974, 930 | 0.886-0.893 | 11.2-12.4 |
| rate+rescale | 2 of 3 | 395, 783 | not measurable (−61 and −1.7×10⁹) | 6.7, 3.0 |
| noise+rescale | 3 of 3 | 472, 509, 466 | 0.889-0.895 | 6.5-7.8 |
| rate+noise+rescale | 2 of 3 | 856, 851 | 0.884-0.886 | 10.8, 11.9 |

**Why some reads fail or look wrong** ([figure](../../img/internal_figures/b3_loss_at_floor_n1000.png),
`plot_b3_loss_floor.py`: every remedy run's loss with the fitted floor, the 1.05× bar, the crossing
and the bursts):
- *Rate alone and rescale alone have no floor read because their loss is still falling at 200k.*
  Over the last 10% of training their smoothed loss sits 7-19% above the fitted floor (one rate fit
  is pinned at the fit's lower bound), with 47-61 (rate) and 65-103 (rescale) loss bursts per run.
  Neither cell diverged: 1-5% of updates skipped, and at 75k their R² is 0.92-0.94. Two of three
  rate runs stay under 1.20× their floor from about 116k; all three rescale runs stay under 1.20×
  from 61-165k, and two of them under 1.10× from about 126k.
- *Rate+rescale and the triple stall early.* Their smoothed loss flattens at 0.11-0.15 from about
  2,000 iterations on, against floors of 0.045-0.062 for every other cell, and the raw loss bursts to
  order 1 for the rest of training. The read-out reads that flat level, so the triple's two reads
  fall at 2,065 and 3,930 iterations. Rate+rescale skips 44-67% of its updates and the triple 8-11%.
  `common.noisy_floor` drops non-finite loss values before it counts bursts, so the 74,000-131,000
  non-finite iterations of the rate+rescale runs count toward no exclusion.
- *Four R² values are snapshots inside a burst or a non-finite stretch,* not the network's floor
  performance: rate+noise s919863771 at 120,000 (−27.5; its crossing is at 119,434), the triple's
  s300782315 at 4,000 (−74.4), and rate+rescale at 75,000 (−61 and −1.7×10⁹). Their PR (1.0-3.0)
  is likewise taken on a network that does not do the task.

**Verdict at the floor, literal (as fixed on dead-relu-paper#2): supports "several sources", through
the stability clause alone.** No combination passes the units-and-cost clause. Rate+noise beats its
best stable component, noise, by 3.26 on the logit of the active fraction (970 against 556 active;
bar 0.47), but its R² is 0.889 on its one usable snapshot against noise's 0.929, a cost of 0.04
against an allowed 0.01. The stability clause fires for three combinations, every time because rate
alone or rescale alone has no floor read:
- rate+noise (2 of 3 read, 1.0-1.5% skipped) against rate (0 of 3);
- noise+rescale (2 of 3) against rescale (0 of 3), though noise+rescale has fewer active units than
  noise (442 against 556 mean);
- the triple (2 of 3) against rate and rescale, though its reads are at 2,065 and 3,930 iterations on
  a stalled loss with 8-11% of updates skipped.

Under the B3 plan's own wording ("a pair or triple that matches the best single remedy on units but
trains stably where that remedy does not") and B2's definition of stable training (a floor read,
under 5% of updates skipped, a finite loss), the triple drops out (8-11% skipped) and noise+rescale
drops out (noise, the best single on units at the floor, is stable). Rate+noise against rate
remains if rate, which has the most active units of the three single remedies at 75k (724 against
562 and 390), counts as the best single remedy. If the best single remedy may only be one with a
floor read, it is noise for every combination, the clause never fires, and the floor verdict is
"between the bars". So the literal "supports" rests on one comparison, rate+noise against rate
alone, and on rate alone lacking a floor read because its loss has not finished falling at 200k,
not because it diverged.

**Verdict at 75k: between the bars.** Rate+noise beats rate on units (+2.00 on the logit, 951
against 724 active; bar 1.89) but costs 0.039 in R² (0.890 against 0.929). Noise+rescale and the
triple sit within 3 SD of their best component on units and PR. Rate+rescale is not within on PR
(6.2 below rate; bar 4.0) and its R² is not measurable.

The floor and 75k verdicts disagree. The rule binds at the floor. Whether a stability gain over a
remedy that has not converged, at a cost of 0.04 in R², is the gain the rule meant is a scientific
call, sent to Pavel (ptolmachev/dead-relu-paper#17). Stage 2 (N = 4000) is not run.

**The pilot's λ choice under the current read-out.** The 16 Oct pilot (read-out 353dc13) read the
λ = 0.03 rung at 2 of 3 runs (629 and 810 active). Under read-out 79932ef the same three runs give
0 of 3. Pilot rule 6 says a rung with fewer than 2 of 3 read-outs cannot be chosen on its numbers,
so under today's read-out λ = 0.03 would not have been chosen that way. The λ = 0.01 rung was not
re-read here.

**Two read-out limits found here** (ptolmachev/dead-relu-paper#18, not patched):
non-finite loss values are dropped before bursts are counted, so a run that skips most of its
updates can still be read; and "R² at the floor" is taken on the first kept snapshot after the
crossing even when that snapshot falls inside a burst.
