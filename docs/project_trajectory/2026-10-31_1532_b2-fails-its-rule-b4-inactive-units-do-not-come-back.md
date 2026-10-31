## 2026-10-31 15:32 — B2 fails its pre-registered rule; B4 at N = 1000: inactive units do not come back

**How it was read.** `trainRNNbrain/experiments_and_analysis/read_b2_b4.py` (commit b86a623), run on
Della as job 16079644 (CPU, 1.5 h, finished 23 Oct 11:09) from `~/trainRNNbrain_c1ana` checked out
at b86a623. Output: Della `~/b2b4_read/read_b2_b4.json` and `B2B4read.16079644.out`; its own
held-out R² cache `~/b2b4_read/read_b2_b4_r2_cache.json`. Every run is read where its noisy training
loss reaches 1.05× its own fitted floor (read-out 79932ef). R² is held-out R² under the training
noise, 36 draws, measured on a kept snapshot where one exists at the crossing, else estimated from
the end of the run (`r2_at_floor_estimate`). Both comparisons cross sizes or were pre-registered at
the floor, so the 16 Oct same-size rule (read at 75k) does not apply.

Before the read, Della's analysis checkout held locally re-rendered Figure 1 and Figure 2 files
(`img/internal_figures/fig_paper_F1.*`, `fig_paper_F2_fails.*`). They are stashed there
(`git stash list`: "Della renders of F1/F2 before b86a623 read"), not deleted.

### B2: the rate penalty alone on the standard 3-bit flip-flop

λ_frm 0.1, default `frm_args`, 3 seeds per size, 200k iterations from scratch
(`NBitFlipFlop_remedy_grid/EqType=h_N=<N>_arm=frm0.1`; Della jobs 15598509-11 and 15527679).
Controls are Figure 1's flip-flop cells (`NBitFlipFlop_ff_revive_c200000` at N = 500-2000,
`NBitFlipFlop_paper_grid_c200000` at 4000), read the same way.

The rule, written 13 Oct before any run (agent-bplan §B2, file deleted in eec4a16, recoverable with
`git show eec4a16^:docs/notes/agent-bplan.md`):
- *Supports* the rate penalty as the main biological remedy if at every N at least 2 of 3 runs reach
  their floor with fewer than 5% of updates skipped and a finite loss; the active count at the floor
  is at least 1.5× the control's; and held-out R² at the floor is within 0.02 of the control's.
- *Falsifies* it if at any N ≥ 1000 fewer than 2 of 3 runs train stably, or the R² cost at the floor
  exceeds 0.02, or the multiplier falls below synaptic noise's at the same N (to be read from
  RMtop200).

| N | Runs with a floor read | Updates skipped | Active at the floor, penalty | Active at the floor, control | Multiplier | R², penalty | R², control |
|---|---|---|---|---|---|---|---|
| 500 | 0 of 3 | 1.7-1.8% | — | 179-235 (6 runs) | — | — | 0.950 |
| 1000 | 0 of 3 | 1.1-1.4% | — | 240-297 (4 runs) | — | — | 0.950 |
| 2000 | 3 of 3 | 3.4%, 4.9%, 6.4% | 1854, 1885, 1893 | 370, 385, 415 | 4.8× | 0.912-0.938 (measured) | 0.950 |
| 4000 | 2 of 3 | 4.5%, 4.6%, 5.1% | 2522, 3698 | 517, 540, 598 | 5.6× | 0.935, 0.943 (estimated) | 0.950 |

Skipped updates are from the training logs (`gradient spikes: … updates skipped`); a control run
(FFseed4) skips 0.25%.

**Verdict: falsified, on two counts.** At N = 1000 no run reaches a floor read (the rule needs 2 of
3), and at N = 2000 held-out R² at the floor is 0.024 below the control's (0.926 against 0.950, both
measured, the penalty's on kept snapshots), past the 0.02 bar. The third test (against synaptic
noise) waits for RMtop200 and cannot change the verdict.

**Why N = 500 and 1000 have no read.** Not a read-out artefact.
[Figure](../../img/internal_figures/b2_rate_alone_small_N_unstable.png): the raw training loss
bursts 57-67 times per run (raw values up to 10^21 at N = 500); the smoothed loss is still falling at
200k; and from about 3k iterations on the active count swings between roughly 300 and 1,000 at
N = 1000 (250-500 at N = 500). The end-of-training counts (884-964 of 1000, 484-487 of 500) are one
moment of that swing and are not reported as results. At N = 2000 the count also falls after the
floor: one run has 1,893 active at its crossing (58k) and 950 at 200k.

What this changes: the decision "the biological penalty in the main text is the rate penalty alone"
rests on CDDM, where the penalty holds all 1000 units active at < 1% R² cost. On the flip-flop the
same penalty trains unstably at N ≤ 1000 and costs 2.4 R² points at N = 2000. Moved to the open
decisions on the board.

### B4: plasticity transfer to the 3-bit hyper flip-flop, N = 1000

Rules: 2026-10-15_1342 entry. S₀ = units inactive at iteration 0 of the new task; revived = units of
S₀ active at the first participation probe at or after the crossing.

| Source | t* | Active at transfer | Inactive at transfer (S₀) | Revived | Newly inactive | Active at the floor | R² at the floor |
|---|---|---|---|---|---|---|---|
| old (trained 200k) | 26.6k | 191 | 809 | 63 (7.8%) | 13 | 241 | 0.912 |
| old | 17.7k | 191 | 809 | 70 (8.7%) | 2 | 259 | 0.912 |
| old | 20.0k | 194 | 806 | 42 (5.2%) | 4 | 232 | 0.912 |
| young (52-68k) | 18.5k | 336 | 664 | 73 (11%) | 2 | 407 | 0.913 |
| young | 19.5k | 291 | 709 | 81 (11%) | 7 | 365 | 0.909 |
| young | 25.3k | 268 | 732 | 85 (12%) | 28 | 325 | 0.912 |
| fresh (untrained) | 50.5k | 944 | 56 | 24 | 511 | 457 | 0.912 |
| fresh | 57.2k | 928 | 72 | 32 | 539 | 421 | 0.912 |
| fresh | 56.0k | 944 | 56 | 31 | 528 | 447 | 0.912 |

All nine R² values are measured on kept snapshots. Each row's counts add up (active at transfer −
newly inactive + revived = active at the floor).

**Gate passes:** fresh networks reach 442 ± 19 active (mean ± SD) at their floor, 250 above the old
networks' 192 at transfer; the bar was 3 SD = 56.
**Verdict: inactive units do not come back** (the pre-registered "silent units do not come back";
the rule's "silent" is the 5%-of-q95 criterion, now called inactive). In every old network fewer
than 10% of S₀ revive (5-9%), and the old networks end 198 units below the fresh ones (244 against
442), past the 3-SD bar of 56. Learning the new task costs nothing in R²: old, young and fresh all
reach 0.909-0.913.

Young networks, a comparison and not a verdict: 11-12% of their inactive units revive, against 5-9%
in old networks, and they end with 325-407 active.

**N = 4000** (1 of 9 runs finished; the rest of 15655301-03 are running or pending): one old network
revives 179 of its 3,683 inactive units (4.9%) and ends with 464 active (317 at transfer). No fresh
run at 4000 has finished, so there is no gate or verdict yet.

### Side note: the N = 1000 flip-flop control

FFseed4 (15657728_4) finished; the N = 1000 control cell now has four runs, all read at the floor:
240, 277, 296 and 297 active. Figure 1 was built with fewer; it needs a rebuild to include the new
seed, and the board's note on excluding the straggler needs checking against these four reads.
