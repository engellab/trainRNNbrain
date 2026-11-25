## 2026-11-24 18:48 — Entropy set point on CDDM: passes the M09 criterion at λ = 0.1 and at λ = 0.01, the latter by 8 units in one network

Rules: `2026-11-22_1831_rules-entropy-setpoint-cddm.md`, committed at 91e759a before any run. Branch `wp/entropy-cddm`.

**Verdict.** The per-unit activity-entropy set point passes on CDDM (the context-dependent decision-making task) in
networks of N = 1,000 units at both values of λ_ent, the weight of the term in the loss.
- *One-value rule* (M09): the term meets the criterion on CDDM.
- *Matched-values rule* (M09): λ = 0.01 and λ = 0.1 were both run on 3BFF, CDDM and Maze. Each value passes on all three
  tasks at N = 1,000 (3BFF: `2026-11-21_0755_...`; Maze: `2026-11-22_1808_...`).

**The λ = 0.01 pass is marginal.** One of its three networks has 350 active units at the read point, and the largest
control has 342. The number of active units in that network was at or below the largest control, compared at the same
probe, at 34 of the 99 probes between iterations 65,000 and 75,000. The λ = 0.1 pass is not marginal: at each of the
99 probes, each λ = 0.1 network has at least 174 more active units than the largest control.

*Active unit*: p = std(r) + q0.9(|r|) on a noise-free pass, active if p ≥ 0.05 × the 95th percentile of p. *H_i*: the
binary entropy, in bits, of Φ(μ_i/s_i), from the mean and standard deviation of unit i's pre-activation h.

### Runs

Della job 18172456, array 1-6, gpu-short, from `~/trainRNNbrain_entcddm` at 91e759a. All six tasks ended COMPLETED (exit 0)
after 3 h 14 min to 4 h 04 min, 20.5 GPU-hours in total of the 70 allowed. Every network reached iteration 75,000, so all
six are counted. No update was skipped in any of them, and there was no rollback. There was no pilot, and Spock was not
used. No separate submission entry was written; this paragraph records the submission:

    sbatch --chdir=$HOME/trainRNNbrain_entcddm --export=ALL --array=1-6 slurm/entropy_setpoint_cddm_della.slurm

The three controls were trained on 28 Sep 2026 at commit 852a17f (Della logs PaperGrid.14623697_4 to _6). The option
`init_draws` was added on 15 Oct (4d25ac0), so the controls used the shared draws, as the rules assumed.

### Read

`entropy_setpoint_read.py cddm-read` at 8b02722, Della CPU job 18323003; `cddm-verdict` on the Mac. The reads are in
`firstmate/data/entropy-cddm/cddm_reads.json`, outside the repository. The gate passes: the controls read 342, 334 and 333
active units and R² 0.8312, 0.8350 and 0.8349, as recorded for M09.

| Group | Active units per network | Mean | Over the controls' mean (336) | Mean held-out R² | Difference from 0.8337 | Verdict |
|---|---|---|---|---|---|---|
| controls (not retrained) | 342, 334, 333 | 336 | 1.00 | 0.8337 (estimate) | | |
| λ = 0.01 | 388, 350, 418 | 385 | 1.15 | 0.8349 (measured) | +0.0012 | passes |
| λ = 0.1 | 540, 566, 539 | 548 | 1.63 | 0.8317 (measured) | −0.0020 | passes |

The bar was more than 342 active units in every network and a mean R² of at least 0.8237.

| Group | Seed | Active units | Held-out R² | Median active units, 2,000 iterations before 50,000 → before 75,000 | Revived | Median \|H_i − 0.75\| (bits) |
|---|---|---|---|---|---|---|
| control | 3833537590 | 342 | 0.8312 | 374 → 346 (−28) | 174 | 0.349 |
| control | 1272763220 | 334 | 0.8350 | 368 → 338 (−29) | 145 | 0.389 |
| control | 457853 | 333 | 0.8349 | 370 → 333 (−36) | 152 | 0.387 |
| λ = 0.01 | 1418346959 | 388 | 0.8350 | 430 → 392 (−38) | 186 | 0.013 |
| λ = 0.01 | 2214393744 | 350 | 0.8338 | 410 → 347 (−64) | 182 | 0.013 |
| λ = 0.01 | 325297547 | 418 | 0.8361 | 456 → 419 (−36) | 191 | 0.011 |
| λ = 0.1 | 3116616303 | 540 | 0.8340 | 567 → 552 (−15) | 132 | 0.010 |
| λ = 0.1 | 1299181146 | 566 | 0.8268 | 585 → 569 (−16) | 135 | 0.010 |
| λ = 0.1 | 2184072010 | 539 | 0.8344 | 552 → 538 (−14) | 148 | 0.010 |

Differences are computed before rounding. The controls' |H_i − 0.75| is from their final (100,000-iteration) weights,
the only ones kept. *Revived*: inactive at some stored participation vector after iteration 1,000 and active at the read.

### Are active units still falling between 50,000 and 75,000?

**Yes, in all nine networks.** The median falls by 14 to 16 units in the λ = 0.1 networks, 36 to 64 in the λ = 0.01
networks and 28 to 36 in the controls. On 3BFF the six entropy networks lost 27 to 161 units over the same interval. The
λ = 0.1 networks lose units more slowly than the controls, so their lead over the controls was still widening at 75,000.
Two of the λ = 0.01 networks lose units faster than every control. No network was trained past 75,000 iterations, and
this read gives no verdict for later ones.

### Other descriptive read-outs

- **The target was met as defined.** Median |H_i − 0.75| is 0.010 to 0.013 bits in the six networks, and 0.35 to 0.39 in
  the controls. On 3BFF a post hoc check found that the term's Gaussian estimate overstated how often inactive units are
  on. That check was not run on CDDM.
- **The λ = 0.1 networks do not show more revival.** They revive 132 to 148 units, the controls 145 to 174 and the λ = 0.01
  networks 182 to 191. The revival measure does not separate slower loss of units from revival before iteration 1,000.
- **The two kinds of R² read (post hoc).** The controls' R² at 75,000 is an estimate from the loss. Their measured R² at
  100,000 averages 0.8349. Relative to 0.8349, the λ = 0.1 mean is 0.0032 lower and the λ = 0.01 mean is equal to four
  decimals, so both arms would also pass with that bar.

### Checks

- **Check 1 and 3 (configs), by value.** Each of the six saved configs differs from control seed 3833537590's only as the
  rules list. **Correction to the rules:** they say nine inert `prune_args` keys were added to the code since 28 Sep;
  there are ten.
- **Check 2.** Della job 18323434, `firstmate/data/entropy-cddm/scripts/check_reads_cddm.py`. The numpy recount equals the
  reader's number of active units in all nine networks. For one network per arm, the numpy and torch forward passes give
  noise-free R² within 6 × 10⁻⁹ of each other (threshold 10⁻⁵). That R² is computed over all steps without the task's
  mask (0.45 and 0.44), so it checks the forward pass and not the masked R² of the tables. The script shares the parameter
  loader, the held-out batch and the warm-start chain with the reader.
- **An incomplete read was discarded.** A poll with `sacct -j 18323003` returned a 2018 record with the same ID, in state
  COMPLETED, while the read was still running. The first copy of the reads therefore held two controls and no arm, and the
  gate did not notice. The gate now requires all three controls and three networks per arm. The verdict above is from the
  complete file.

### Figure

`img/internal_figures/entropy_setpoint_cddm_active.{pdf,png}`: active units against iterations, every network of both
arms and the three controls; the vertical line marks iteration 50,000.

### What follows

The term now meets M09's criterion on all three tasks at N = 1,000. λ = 0.1 is the value whose CDDM pass does not
depend on which probe is read. The paper and the status board are unchanged: the set-point terms are not among the paper's
techniques, and no claim in `docs/paper_audit.md` rests on this result.
