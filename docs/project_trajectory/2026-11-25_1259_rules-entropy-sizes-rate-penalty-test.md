## 2026-11-25 12:59 — Rules, addition to the entropy size test: the rate-penalty size test as a pre-set read-out

Adds to `2026-11-25_1241_rules-entropy-setpoint-sizes.md` (8d31c36). Written and committed before any size run is
submitted; the pilot jobs (18374551-18374556) are still queued. Asked for by firstmate (25 Nov): a referee says that
prune-and-duplicate was never put through the paper's pre-registered rate-penalty size test.

**What has been seen already.** The reader's dry run (Della jobs 18374698-18374704) read the controls and the existing
N = 1,000 entropy networks, so their crossing reads were seen before this entry. On CDDM the three N = 1,000 networks
(75,000 iterations) have 607, 600 and 607 active units at the crossing, 1.40 times the controls' mean of 431. On 3BFF one of
the three has a crossing within 75,000 iterations. Of the Maze N = 1,000 and 2,000 networks, every count has been published
in this notebook. Every prune-and-duplicate number below was published before this entry. No network of the new sizes exists.

### (B) The test, from `dead-relu-paper` floats/EDT5_rules.tex, first row (a6c3f46), with λ_ent = 0.1 in place of the rate penalty

3BFF and CDDM, at the floor crossing (1.05×). A network is **stable** if it has a floor crossing, fewer than 5% of all its
updates were skipped (the `gradient spikes` line, summed over segments, over the budget), and its loss is finite.
*Finite loss*, strict reading, which binds: no non-finite value in the saved training loss of any segment. This is the
reading behind the table's "every network has non-finite loss values". *Lenient reading*, reported: the run's folder is not
named `nan`, as gap B read it.
- **Pass** if, at every N read: at least 2 of 3 networks are stable; the fold change (mean active units at the crossing over
  the controls' mean at theirs) is at least 1.5; and the mean held-out R² at the crossing is within 0.02 of the controls'.
- **Fail** if, at any N ≥ 1,000: fewer than 2 of 3 are stable; or R² is more than 0.02 below the controls'; or the fold
  change is below that of noisy synapses (σ_w = 1) at the same N, where that exists: 3BFF 2.03, 1.54 and 1.60 at N = 1,000,
  2,000 and 4,000 (Supplementary Table 4); CDDM 1.11 at N = 1,000; Maze 0.98 at N = 1,000 (gaps B and C).
- **Otherwise inconclusive.** A failed condition at N = 500 alone prevents a pass and is not a fail.

**Maze**, as Methods (M12) defines it: the learned point replaces the crossing; stable means the network reaches update 4,000
without being stopped, has a learned point by then, and has a skipped minibatch step or a restored rollout in fewer than 5%
of its updates; the success tolerance at update 4,000 is 0.05 instead of 0.02 in R².

**Sizes.**
- 3BFF: N = 500 and 2,000 (this task). N = 1,000 at 200,000 iterations belongs to `entropy-prune-combo`; its 75,000-iteration
  networks are reported and not judged here. N = 4,000 is cut. So on 3BFF this task can give a fail (at N = 2,000) or an
  outcome over N = 500 and 2,000, and the statement over every N needs N = 1,000 and 4,000.
- CDDM: N = 500, 2,000 and 4,000 (if run), and N = 1,000 from the existing networks: 75,000 iterations, the budget at which
  gap B ran this test on CDDM.
- Maze: N = 500 (new) and N = 2,000 (existing, 4,000 updates). N = 1,000 was trained for 2,000 updates only: its fold change
  at the learned point and its stability over 2,000 updates are reported, and its success at 4,000 does not exist.

Reported as well, without entering the verdict: the test at 1.10× and 1.20×. An outcome is printed per size and per task.

### Prune-and-duplicate, read from the published numbers

- **3BFF, from floats/EDT3_remedies.tex** (Supplementary Table 4): crossings 3/3, 3/3, 3/3 and 1/3 at N = 500, 1,000, 2,000 and
  4,000; fold 2.06, 2.77, 3.24 and 3.59; ΔR² at the crossing −0.003, −0.005, −0.009 and −0.018 (at 75,000: −0.023 at
  N = 4,000, outside this test, which reads the crossing). At N = 4,000 one network has a crossing, so at most one is stable,
  and the test needs two: **by the published table, prune-and-duplicate fails the test on 3BFF**, at N = 4,000, by its
  stability condition. The crossing, fold and R² conditions hold at N = 500 to 2,000. The table does not give skipped updates
  or non-finite losses, so stability at N ≤ 2,000 is not shown by it.
- **CDDM, from G6** (`data/round18/g6_reads.json`; the table's notes summarise it as 1.4- to 2.1-fold): crossings 3/3 at every
  size; fold 1.37, 1.52, 1.94 and 2.11; ΔR² −0.003, −0.002, +0.002 and −0.004. The fold at N = 500 is below 1.5, so it does
  **not pass**; no fail condition holds on these numbers (skipped updates and finite losses not read): inconclusive.
- **Maze, from gap A** (entry 2026-11-20_1600): learned points 3/3 at N = 500, 1,000 and 2,000; fold at the learned point 1.497,
  1.45 and 2.84; success difference +0.003, 0.000 and −0.016. The fold is below 1.5 at N = 500 and 1,000 and above noisy
  synapses' at 1,000: inconclusive. N = 4,000 does not exist.

### Expectation, stated before any size run

On CDDM the entropy networks probably do not pass (N = 1,000 fold 1.40 is already below 1.5) and do not fail. On 3BFF a fail
at N = 2,000 is possible through the loss bursts: three of the six 3BFF entropy networks at N = 1,000 (one of the three at λ = 0.1) had 499 to 998 skipped
updates (0.7% to 1.3% of 75,000), which is under 5%, but a skipped block may hold non-finite losses.
