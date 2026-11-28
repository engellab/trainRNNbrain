# Result: entropy set point × prune-and-duplicate at N = 1,000 (2×2, 3 seeds per cell)

Branch `wp/entropy-prune-combo`. Rules: `2026-11-25_1245_rules-entropy-prune-combo.md`, amended by
`2026-11-25_1312_entropy-prune-combo-length-and-chain.md` (L = 175,000). Submission: `..._entropy-prune-combo-submitted.md`.

**The two techniques stack on 2 of 3 tasks (CDDM, Maze) and stall the late fall further than either alone on 1 of 3
(Maze).** On 3BFF they stack at 75,000 iterations but not at the end: one of the three networks trained with both
techniques diverged 200 iterations before the end, and under the rules that counts as interference. Neither verdict is
claimed in general, because neither holds on all three tasks.

Cells: D (default training), E (entropy set point alone, λ_ent = 0.1, target 0.75 bits), P (prune-and-duplicate alone,
M09's values) and EP (both). Tasks: 3BFF (3-bit flip-flop) and CDDM (context-dependent decision making), each trained
to 175,000 iterations in two stages (75,000 + 100,000, Adam state carried over); Maze (the reinforcement-learning maze,
trained with PPO, proximal policy optimization) to update 4,000 in two stages of 2,000. A unit is active when its
participation (std(r) + q0.9(|r|) on a noise-free pass) is at least 5% of the 95th percentile of participation over
the network's units.

## What ran

All 60 training jobs (18 stage 1 and 24 stage 2 on 3BFF and CDDM, 6 and 12 on Maze) and the three analysis jobs
completed within their time limits on Della; none was rerun, continued or cancelled. Total use: about 327 GPU-hours of
the 400-hour cap (3BFF and CDDM 209, Maze 117, pilots 1.1, reads under 0.1).
The Della connection was down for about a day while the chain ran; the jobs did not depend on it.

## Verdicts

| task | stack (b)+(d), both points | late fall (c) | stall further | best single | share of the best's inactive units that EP adds (read; end) |
|---|---|---|---|---|---|
| 3BFF | no: holds at 75,000; fails at the end (interference) | a single technique already stalls the fall | no | P | 0.71; −0.23 |
| CDDM | yes | a single technique already stalls the fall | no | P | 0.83; 0.74 |
| Maze | yes | stalled further | yes | P | 0.56; 0.70 |

The best single technique is P on all three tasks: both E and P are within tolerance of D's performance at the read
point, and P has more active units. No E or P network reaches the 950-unit ceiling, so no comparison is limited by it.

## Active units, per task (read point; end)

Each list is the three networks of a cell, in seed order.

**3BFF** (read at iteration 74,900, end at 174,900; tolerance 0.01 in held-out R²):
- D 236, 296, 290; 172, 200, 214. Mean R² 0.951; 0.953.
- E 381, 380, 453; 302, 340, 410. Mean R² 0.951; 0.952.
- P 784, 765, 803; 760, 801, 808. Mean R² 0.946; 0.947.
- EP 927, 938, 947; 939, 956, **326**. Mean R² 0.950; **−95,087** (the two intact networks: 0.951 and 0.951).

At the read point every EP network has more active units than every E and every P network, and EP's R² is within
0.01 of D and of P. At the end, network 4168822465 has 326 active units and R² −285,000, so (b) fails against both
E and P and (d) fails while P is within tolerance of D: interference under rule (e). Outside the rules, for
description only: the two intact EP networks end with 939 and 956 active units, above every E and P network.

**CDDM** (tolerance 0.01 in held-out R²):
- D 340, 335, 339; 267, 270, 276. Mean R² 0.832; 0.835.
- E 536, 568, 535; 506, 497, 497. Mean R² 0.832; 0.836.
- P 709, 696, 713; 739, 710, 717. Mean R² 0.831; 0.833.
- EP 940, 939, 968; 830, 988, 968. Mean R² 0.831; 0.834.

**Maze** (read at update 2,000, end at 4,000; performance is the probe success rate over 128 episodes; tolerance 0.05,
no network below 0.9):
- D 401, 432, 392; 346, 353, 378. Mean success 0.93; 1.00.
- E 634, 689, 661; 519, 634, 584. Mean success 0.98; 0.99.
- P 763, 757, 757; 750, 755, 765. Mean success 0.98; 1.00.
- EP 876, 899, 904; 902, 928, 948. Mean success 0.99; 1.00.

## The late fall (c)

F is the fall of the median count over the late window (3BFF and CDDM: iterations 75,000 to 175,000; Maze: updates
2,000 to 4,000), A(start) − A(end), where A is the median over the last 2,000 iterations (100 updates). A negative F
is a rise. Mean F per cell:

| task | D | E | P | EP |
|---|---|---|---|---|
| 3BFF | 76 | 56 | −16 | −1 |
| CDDM | 69 | 49 | −22 | 8 |
| Maze | 48 | 82 | 9 | −24 |

On 3BFF and CDDM the count in P networks does not fall over the late window (mean F under 20 units), so there is no
fall left for EP to stall: outcome 3. On 3BFF, A(end) of the diverged network is 937, because the divergence falls in
the last 200 iterations and A is a 2,000-iteration median; its F of 26 does not change the outcome. Two CDDM EP networks
(F −45 and −22, A(end) 983 and 982) and one 3BFF EP network (F −18, A(end) 956) are at the ceiling under rule (e).

On Maze every EP network's count rises over stage 2 (by 16, 24 and 31 units, about 2-3%), while every P network falls by up
to 21 units or rises by 4, and every E network falls by 57 to 109 units (8-17%). Mean F of EP is 32 units below P's and
106 below E's, both beyond the 20-unit threshold: outcome 1, stalled further, with success 1.00 in every EP network.

## What the count cannot show

**Replacements per check** (mean over stage 1 → mean over the late window; the cap is 25 units per check):

| task | P | EP |
|---|---|---|
| 3BFF | 23.4-24.4 → 22.0-22.9 | 4.1-4.5 → 2.9-3.4 |
| CDDM | 24.6-24.7 → 25.0 (at the cap) | 9.1-12.6 → 4.3-17.1 |
| Maze | 24.7 → 23.9-24.6 | 9.8-12.6 → 8.6-11.3 |

With the entropy term, prune-and-duplicate replaces fewer units per check over the late window: about a seventh as many
as without it on 3BFF, and about two fifths as many on CDDM and Maze.

**Unit death rate** over the late window (deaths per 100 active units per 1,000 iterations; Maze per 100 updates):

| task | D | E | P | EP |
|---|---|---|---|---|
| 3BFF | 1.9-3.4 | 4.8-6.1 | 11.5-13.0 | 6.1-7.3 |
| CDDM | 10.7-12.3 | 7.2-8.3 | 16.3-17.6 | 2.9-11.0 |
| Maze | 20.6-22.4 | 19.9-22.5 | 16.2-16.9 | 7.4-10.7 |

Wording fixed by the rules:
- Maze: **fewer units die with both techniques** (every EP rate below every E and P rate).
- CDDM: **the larger count is held by replacement**: EP units die at 2.9, 5.5 and 11.0 per 100 per 1,000 iterations,
  against 7.2-8.3 in E and 16.3-17.6 in P.
- 3BFF: (b) fails at the end, so neither sentence applies. EP's death rate (6.1-7.3) is below P's and above E's.

**Participation ratio** of the active units' noise-free rates at the end (dimensionality, (Σλ)²/Σλ²):

| task | E | P | EP |
|---|---|---|---|
| 3BFF | 4.3-6.3 | 7.4-10.6 | 8.0, 7.5, 1.0 (diverged) |
| CDDM | 2.8-3.1 | 4.6-5.1 | 7.4, 4.3, 4.2 |
| Maze | 4.8-10.6 | 13.0-13.6 | 13.6, 14.7, 16.8 |

Maze: **the added units add dimensions** (every EP ratio above every E and P ratio; the margin is small, 13.63 for the
lowest EP network against 13.62 for the highest P network). 3BFF and CDDM: **the added units
add no measurable dimensions**.

**Near-copies.** On 3BFF and CDDM many active units have a noise-free rate correlated above 0.95 with another active
unit's: at the end 43-85% of EP's active units on 3BFF and CDDM (the diverged network excluded), 69-78% in P, 36-76% in E and
52-64% in D. On Maze the fraction at the end is 0-3% in every cell. On the two supervised tasks the count of active units therefore overstates the number of
distinct rate profiles in every cell, EP included.

## The diverged 3BFF network

EP network 4168822465, stage 2 (job 18376387_12, seed 4168822466). Its R² was 0.95 up to iteration 174,800 (local
99,800). The prune-and-duplicate check at 174,800 replaced 18 units (cumulative replacements 3,340 → 3,358). The
training loss then rose from 0.035 to 0.35 over the next five updates and to 785,000 at local update 99,807. From
174,810 to the end the weight norms are constant and the cosine between successive weight changes is 0: the gradient-
spike guard skipped every remaining update (693 skipped in stage 2, about 193 of them after the divergence). The
guard's rollback to the last good weights acts after 500 consecutive skips (`restore_after`), and the run ended 193
updates after the divergence, so no rollback happened. The weights are finite and no rollback stopped the run, so under
the rules the network finished and enters every mean. The checkpoint at 174,000 is intact.

This is one event in 36 networks, and the only divergence; whether duplication under the entropy term makes such events
more likely is not tested by three seeds.

## Skipped updates (gradient-spike guard), per network

Nonzero counts only (stage 1; stage 2). 3BFF: D 4200852863 500; 0. D 1256299004 986; 0. E 4168822465 500; 498.
EP 4200852863 0; 395. EP 1256299004 16; 0. EP 4168822465 0; 693. CDDM: D 3116616303 0; 499. Maze: D seed 1, 15 skipped
steps or restored rollouts in stage 1. All are below 1.4% of updates.

## Checks

- **Independent recount** (`/Users/pt1290/firstmate/data/entropy-prune-combo/scripts/oracle_combo.py`, numpy only,
  separate code from the reader): active counts at both points, replacements per check and death counts with exposure
  for all 36 networks, 144 values, compared for exact equality. Mismatches: none. Its (b), (c) and death-wording
  verdicts agree with the reader's on all three tasks.
- **Dry run before any read**: the reader reproduced the recorded E counts and performance exactly on all three tasks
  (submission entry).
- **Verdict rerun**: `verdict` rerun on the Mac from the copied `reads.json` prints the same text as the Della job.
- **Figure**: `img/internal_figures/entropy_prune_combo_active.{pdf,png}`, active units against training for every
  network of every cell, read points marked. The Della job's figure plotted every fifth stored vector on 3BFF and CDDM
  and so dropped the last vectors, hiding the 3BFF divergence; `draw` now always includes the last vector, and the
  legend moved to an empty corner. Verdicts are unchanged by this.

## Status board

`docs/paper_audit.md` is not changed: no claim on it concerns combining the two techniques, and no single-technique
verdict changes (E and P here are within the ranges already recorded).

## Files

- Reads, verdict, recount, figure: `/scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/entropy_prune_combo_reads/`
  (Della) and `/Users/pt1290/firstmate/data/entropy-prune-combo/results/`.
- Networks: `data/trained_RNNs/{NBitFlipFlop,CDDM}_entropy_prune{,_c}/`, Maze under `trained_RNNs_rl_maze_epc`.
