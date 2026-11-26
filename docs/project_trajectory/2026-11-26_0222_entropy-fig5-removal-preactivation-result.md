## 2026-11-26 02:22 — Result: the entropy set point in Figure 5; unit removal and pre-activations at λ_ent = 0.1

Rules: `2026-11-25_1235_rules-entropy-removal-and-preactivation.md` (committed 4830b4c before any measurement).
Branch `wp/entropy-fig5`. No network was trained. Code: `entropy_fig5.py` (bbbb156, guard 242e3f6),
`slurm/entropy_fig5_della.slurm`; figure: `fig_paper_F5_tasks.py`. Della jobs 18374496 (supervised removal and
pre-activations), 18374497 and 18390168 (Maze removal; tasks 1 and 3 of 18374497 reached their 4 h limit and were
run again with 12 h), 18376468 (pre-activations again, with the criterion's Maze controls), 18376094-6, 18390169
(independent checks), 18390170 (summary), 18390171 and 18422177 (diagnostics of the Maze check, below).

### Summary

- **Figure 5 criterion.** The entropy set point at λ_ent = 0.1 works on 3BFF, CDDM and Maze. It is the second technique,
  after prune-and-duplicate, that works on all three tasks; the panel a and b columns of both are shaded.
- **Removal of the k least participating active units** (k = active units minus the controls' mean; M09's bar of a 0.10
  drop): *mixed* on 3BFF (drops 0.030, 0.013, 0.033), *mixed* on CDDM (0.095, 0.111, 0.096), *depends* on Maze
  (0.367, 0.191, 0.287). In all nine networks, removing k random active units lowers the score more.
- **Pre-activations.** In the inactive units the median actual on/off entropy is 0.09-0.37 bits on all three tasks, far
  from the 0.75 target: there the term sets only the ratio of mean to standard deviation of h. In the active units it is
  within 0.15 bits of 0.75 on all three tasks, but on CDDM and Maze the control networks' active units are nearer to 0.75
  than the entropy networks', so the term did not change the active units' on/off entropy on those two tasks.

### Gates and checks

- **Environment, supervised** (rule: same active count, R² within 0.002): the three 3BFF controls and one
  prune-and-duplicate network of A3 and the CDDM rate-penalty 0.1 network of G3 reproduce their stored active counts
  (278, 288, 242, 799, 1000) and every stored R² to 0.0000. Pass.
- **Environment, Maze** (rule: active units within 2, scores within 0.03): prune-and-duplicate s1, scored on the Mac with
  766 active units, has 768 on Della; intact, inactive-removed and k-removed success differ by 0.0010, 0.0059 and
  0.0176. Pass.
- **Independent check, supervised** (separate numpy script; rule: identical removed set, R² within 0.01 and 0.03): all
  six networks have the same active count, the same k and the same removed set, unit by unit; intact R² within 0.0017,
  k-removed R² within 0.0029. Pass.
- **Independent check, Maze** (the units' rates clamped to zero in a retyped RNN loop; rule set before the run: the same
  active count and k, success within 0.03):
  - s3: 660 active units, k 230 in both; intact 0.9336 vs 0.9307, k removed 0.6504 vs 0.6436. Pass.
  - s2: 689 active units in the check, 690 in the read (k 259 vs 260); intact 0.8740 vs 0.8799, k removed 0.6748 vs
    0.6885. **Fails** on the count.
  - s1: 635 vs 633 (k 205 vs 203); intact 0.8672 vs 0.8691, k removed 0.4453 vs 0.5020. **Fails** on the count and on the
    k-removed success (0.057 apart).
  - Cause, s2 (job 18390171): both codes replay the same 128 probe drops on CPU; per unit, their participation values
    differ by a median of 0.1% (largest 10%, one unit). One unit (937) sits at 1.017 times the active threshold in the
    read and 0.981 times in the check, so it changes label.
  - Cause, s1 (job 18422177): the two removed sets share 202 units; 1 is removed only in the read, 3 only in the check.
    Given the read's removed set, the check's own loop (rates clamped to zero) gives 0.5010 against the read's 0.5020;
    given its own set, 0.4453. On s2 the same test gives 0.6885 against 0.6885, and 0.6729 with its own set. The
    gap is therefore the 1 to 4 units on which the two sets differ, not the removal or the scoring: clamping rates
    and zeroing outgoing weights give the same success on the same set.
  - The Maze verdict does not depend on which count is used: every drop in the check is above 0.10 (s1 0.42, s2 0.20,
    s3 0.28). The stored training probe (GPU) has 634, 689 and 661 active units; the CPU read has 633, 690 and 660. The
    read is what Figure 5d draws, as for its 12 other Maze networks.
- **Independent check, pre-activations** (rule: within 10⁻⁶ per unit): one entropy network per task; largest per-unit
  difference 3.6 × 10⁻¹⁵. Pass.

### Part 1. Removal (M09's procedure and bar)

Scores: held-out R² (3BFF, CDDM, 75,000-iteration weights, 9 draws) and success rate under the training noise (Maze,
update 2,000, 256 drops × 4 draws). "Random" is the median over 5 sets of k random active units.

| Task | Network | Active | k | Intact | All inactive removed | k least removed | k random removed | Drop |
|---|---|---|---|---|---|---|---|---|
| 3BFF | seed 4200852863 | 381 | 112 | 0.9501 | 0.487 | 0.9206 | −956 | 0.030 |
| 3BFF | seed 1256299004 | 383 | 114 | 0.9513 | 0.486 | 0.9379 | −12,995 | 0.013 |
| 3BFF | seed 4168822465 | 453 | 184 | 0.9516 | 0.583 | 0.9185 | −1.4 × 10¹¹ | 0.033 |
| CDDM | seed 3116616303 | 536 | 207 | 0.8367 | 0.836 | 0.7416 | −0.135 | 0.095 |
| CDDM | seed 1299181146 | 568 | 239 | 0.8299 | 0.823 | 0.7194 | 0.320 | 0.111 |
| CDDM | seed 2184072010 | 535 | 206 | 0.8356 | 0.835 | 0.7398 | 0.296 | 0.096 |
| Maze | s1 | 633 | 203 | 0.869 | 0.836 | 0.502 | 0.364 | 0.367 |
| Maze | s2 | 690 | 260 | 0.880 | 0.882 | 0.688 | 0.504 | 0.191 |
| Maze | s3 | 660 | 230 | 0.931 | 0.930 | 0.644 | 0.566 | 0.287 |

Controls for k: 3BFF 269.3 (A3), CDDM 329.0 (the five Figure 3a controls at 75,000), Maze 430. The active counts
here are the offline counts of the removal code; the criterion's counts from the stored participation vectors are
384, 380, 454 (3BFF), 540, 566, 539 (CDDM) and 634, 689, 661 (Maze).

**Readings, as the rules state them.**
- 3BFF, *mixed*: every drop is above the 0.01 tolerance and below 0.10. The prediction written with the rules (k between
  110 and 185, 3BFF the task most likely to fall under 0.10) held: k is 112, 114 and 184.
- CDDM, *mixed*: one network of three has a drop of at least 0.10 (0.111); the other two are 0.095 and 0.096.
- Maze, *depends*.
- By the rules: the statement "the output depends on the added units" does not extend to the entropy term on 3BFF or
  CDDM, and the paper lists both with prune-and-duplicate on CDDM, until now the one mixed result. The M09 criterion is
  unchanged by this reading.
- Removing every inactive unit lowers 3BFF R² from 0.95 to 0.49-0.58 in the entropy networks. In the three 3BFF control
  networks the same removal gives 0.36-0.48 (gate values above), so 3BFF networks of both kinds use their inactive units.
- The 3BFF random removals make the dynamics diverge (R² down to −1.4 × 10¹¹); Figure 5d draws them at its floor with an
  arrowhead.

### Part 2. Pre-activations (descriptive)

Medians over units. "On" is the actual on-fraction (share of noisy samples with h > 0); H is its binary entropy in bits
(target 0.75); "skew" is the skewness of h. 3BFF and CDDM: held-out batch, training noise; Maze: 256 noisy training
rollouts. Controls: 3BFF the three default networks at 75,000; CDDM the criterion's three controls at 100,000 and the
five Figure 3a controls at 75,000; Maze the three default networks at update 2,000 (the criterion's networks, below).

| Task | Networks | Inactive: on | Inactive: H | Inactive: skew | Active: on | Active: H | Active: skew |
|---|---|---|---|---|---|---|---|
| 3BFF | entropy (616, 620, 546 inactive) | 0.042, 0.050, 0.070 | 0.252, 0.285, 0.367 | −1.67, −1.23, −1.31 | 0.337, 0.327, 0.305 | 0.864, 0.885, 0.865 | −0.13, −0.08, −0.24 |
| 3BFF | control | 0.018, 0.021, 0.017 | 0.129, 0.150, 0.127 | +0.68, +0.50, +0.82 | 0.420, 0.426, 0.581 | 0.921, 0.913, 0.913 | +0.24, +0.22, +0.16 |
| CDDM | entropy (460, 434, 461) | 0.025, 0.024, 0.024 | 0.170, 0.163, 0.164 | −1.56, −1.48, −1.52 | 0.725, 0.732, 0.728 | 0.796, 0.788, 0.797 | −0.01, −0.05, −0.05 |
| CDDM | control, criterion | 0.021, 0.026, 0.025 | 0.145, 0.172, 0.169 | −0.11, −0.07, −0.10 | 0.207, 0.201, 0.206 | 0.735, 0.723, 0.734 | −0.25, −0.27, −0.23 |
| CDDM | control, Figure 3a | 0.021-0.024 | 0.145-0.165 | −0.15 to −0.10 | 0.204-0.219 | 0.730-0.759 | −0.31 to −0.24 |
| Maze | entropy (366, 311, 339) | 0.014, 0.011, 0.014 | 0.104, 0.088, 0.104 | −4.03, −8.14, −4.16 | 0.182, 0.183, 0.187 | 0.677, 0.680, 0.691 | −1.80, −2.03, −1.88 |
| Maze | control | 0.001, 0.002, 0.001 | 0.014, 0.019, 0.012 | −0.30, −1.43, −0.23 | 0.271, 0.270, 0.264 | 0.797, 0.800, 0.806 | −0.25, −0.56, −0.17 |

H_i, the Gaussian entropy the term acts on, has a median of 0.73-0.75 in the entropy networks' inactive units (77-99% of
them within 0.05 of 0.75) and 0.75-0.80 in their active units. The per-unit file is
`firstmate/data/entropy-fig5/della/preact2/preactivation.json` (with quartiles, Φ(μ/s), the noise-free on-fraction and the
share of units within 0.05 of 0.75).

**Readings, as the rules state them.**
- Inactive units: the median actual entropy is not within 0.15 bits of 0.75 in any network on any task. The paper may say
  only that the term sets the ratio of mean to standard deviation of h in inactive units. The h of these units is skewed
  to the left (−1.2 to −8.1), where the control networks' inactive units have skews of −1.4 to +0.8, so a Gaussian with
  the target ratio overstates their on-fraction.
- Active units: within 0.15 bits of 0.75 in all nine networks.
- Nearer to 0.75 than the controls? 3BFF: yes in every network (0.11-0.14 bits away against 0.16-0.17 in the controls).
  CDDM: no in every network (0.04-0.05 against 0.00-0.03 in all eight controls), although the active on-fraction rose
  from about 0.21 to 0.73. Maze: no in every network (0.06-0.07 against 0.05-0.06). By the rules, the term did not
  change the active units' on/off entropy on CDDM or on Maze.

**Maze controls on Della are another training run.** Della's `trained_RNNs_rl_maze/Maze_suite/N=1000_act=relu`
s1-s3 have 455, 423 and 404 active units at update 2,000; the criterion's controls (on the Mac, the networks of Figure 5)
have 426, 449 and 415. The first pre-activation job read Della's copy. The Mac copy was uploaded (parameter files'
md5 checked) and job 18376468 read it; the numbers above are from that job, and `entropy_fig5.py preact` now refuses a
Maze control whose active count is not the criterion's. The entropy networks' rows of the two jobs are identical.

### Part 3. Figure 5

- Cells (`data/entropy_fig5/cells.json`, written by `entropy_fig5.py export` from the three stored reads, equal to the
  result entries): λ_ent = 0.1 has 384, 380, 454 active units on 3BFF (fold 1.55 over the control mean; R² change
  −0.0002), 540, 566, 539 on CDDM (fold 1.63; −0.0020) and 634, 689, 661 on Maze (fold 1.54; success +0.081).
  λ_ent = 0.01 (exported, listed, not drawn): 383, 424, 411 (fold 1.55; −0.0002), 388, 350, 418 (1.15; +0.0012),
  631, 667, 620 (1.49; +0.0625).
- Column order in panels a and b: entropy set point, prune-and-duplicate, noisy synapses, rate penalty, read-out dropout,
  rate + sparsity, rescaling. The entropy marker is an open pentagon. (A right-pointing triangle failed
  `tests/scan_open_markers.py` in panel d, where the three nearly equal 3BFF values put a triangle's tip on its
  neighbour's inside; every other triangle is taken.)
- Rule tables. One-value rule: a technique passes on all three tasks from the R² tolerance: entropy set point 0.0002
  (with λ_ent = 0.01 on 3BFF), prune-and-duplicate 0.0032, rate penalty 0.0161. Matched values: entropy set point 0.0002,
  prune-and-duplicate 0.0046, rate penalty 0.0213. λ_ent = 0.1 alone passes from 0.0020 (its CDDM R² change).
  At the paper's 0.01 both rules give the entropy set point and prune-and-duplicate.
- Panel d: 22 (3BFF), 17 (CDDM) and 15 (Maze) networks. 47 of 54 have a drop of at least 0.10 (43 of 45 before);
  in 51 of 54 the random removal lowers the score more (42 of 45 before). Median ticks (intact, k least, k random): 3BFF
  0.935, 0.364, 0.127; CDDM 0.830, 0.557, −0.024; Maze 0.931, 0.644, 0.366. The entropy networks' rows are not checked
  for loss bursts ("bursts not checked" in the printout).
- **Bit-identity test** (set in the rules; float-hex dump at 513623a before any edit): all 38 cells, 38 verdicts, 45
  panel d rows, the floor reads, the panel c CDDM rows and the tolerance tables are identical; 3 cells and 9 panel d rows
  are new. `tests/scan_open_markers.py fig_paper_F5_tasks`: 0 failures, no overlapping text.
  `tests/test_no_function_imports.py`: pass. `tests/check_f5_maze_reruns.py` fails the same way at 513623a, before these
  edits; left alone.
- Printout: `data/entropy_fig5/fig_paper_F5_tasks_printout.txt`.

### What the paper needs

- Figure 5 (internal figure `img/internal_figures/fig_paper_F5_tasks.pdf`): seven techniques; the entropy set point
  works on all three tasks at λ_ent = 0.1; "prune-and-duplicate is the only technique that works on all three tasks" is no
  longer true.
- Removal: mixed on 3BFF and CDDM, depends on Maze; the statement that the output depends on the added units is limited
  to Maze for this term.
- Pre-activations: the term sets the ratio of mean to standard deviation of h, not the on/off entropy, in inactive units
  (0.09-0.37 bits); in active units the on/off entropy is within 0.15 bits of 0.75 but is no nearer to 0.75 than in
  control networks on CDDM and Maze.
