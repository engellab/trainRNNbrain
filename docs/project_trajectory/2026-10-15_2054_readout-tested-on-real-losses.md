## 2026-10-15 20:54 — The new read-out tested on real losses: truncation, seeds, and a snapshot oracle

Follows `2026-10-15_1400_readout-hampel-sustained-crossing-bursts.md`. Pavel asked for the rule to
be tested on real training losses as well as synthetic ones. Three tests; every pass threshold was
written down before its result was seen. Scripts were run from the session scratchpad and are not
kept in the repo.

**1. Truncation (1,775 local runs).** Each run was cut to 50% and 75% of its iterations and read
again. Only runs whose full-run read-out, plus its 1.2t hold, fits inside the cut are included.
Thresholds: median |t* shift| <= 5%, 90th percentile <= 20%, >= 90% still read, and no worse than
the old rule.

| Cut | Rule | Still read | Median shift | 90th pct shift |
|---|---|---|---|---|
| 75% | new | 90% | 3.0% | 11.9% |
| 75% | old | 90% | 2.9% | 21.3% |
| 50% | new | 87% | 7.6% | 23.4% |
| 50% | old | 87% | 8.5% | 30.0% |

- The 75% cut passes (the 0.1-point gap to the old rule's median is counted as a tie).
- The 50% cut fails all three thresholds, though the new rule is less wrong than the old.
- The failure has a direction: a run whose training ends soon after its read-out fits a higher
  floor and is read earlier. The median shift is -14% when training stopped before 1.5x t*, -4%
  at 2-3x and -0.7% beyond 5x.
- Requiring each run to have trained to at least 2x t* would keep 81% of current read-outs; 3x
  would keep 65%. Not adopted yet; the choice is Pavel's.

**2. Seed agreement (278 cells with 3+ seeds read by both rules).** The median within-cell CV of
t* is 0.080 under the new rule and 0.078 under the old, with the new rule lower in 55% of cells.
This fails the "no higher than old" criterion, and it is a tie: the seed-to-seed spread is mostly
real, so this test cannot tell the two rules apart.

**3. Snapshot oracle (26 runs with kept weights on Della).**
- The oracle evaluates each kept snapshot (every 1,000 iterations) on fresh batches under training
  noise, the way `Trainer.train_step` runs the network: 32 batches per snapshot on CDDM, 16 on
  DMTS, 8 on the flip-flop.
- It runs from the sweep's own checkout (`~/trainRNNbrain_c1abl`, b524ac7) and imports nothing from
  `common.py`.
- Sweeps: `CDDM_c1_snapshots` (control and four winp arms, 15 runs), `DMTS_c1_snapshots` (3 runs) and
  the flip-flop `c1_snapshots`, `_c200000` and `_c300000` chains (8 runs).
- Della job 15635966: 2,124 snapshots, 3 h on 32 CPUs. Output in
  `/scratch/gpfs/TENGEL/pt1290/oracle_readout/`.
- The oracle at t is the mean over the snapshots in [t/1.1, 1.1t], with snapshots inside detected
  bursts dropped. Each snapshot's value is the mean of its per-batch losses after dropping points
  more than 3 MADs from their median.
- Thresholds: the curve within 2% of the oracle at the median and 5% at the 90th percentile; t*
  within 10% of the oracle's own sustained crossing.

Results (figure: `img/internal_figures/readout_vs_snapshot_oracle.png`):
- **Flip-flop: passes.** The curve is within 0.1-0.4% of the oracle (median), and all 6 runs with
  a read-out are within 3% of the oracle crossing.
  - A first version of the comparison reported -21% to -75% errors on the top-up runs. The cause
    was a bug in the comparison script: the oracle used only the top-up's own snapshots and could
    not cross before 75k or 200k. Fixed by joining the parent's snapshots, as `noisy_loss_series`
    joins the loss.
- **CDDM: the level passes and the crossing time fails, for both rules.** The curve is within
  0.3-0.8% of the oracle on all 15 runs. t* misses the oracle by more than 10% on 7 of 11 runs, from
  -33% to +63%, and the old rule misses by the same amounts on the same runs.
  - Cause: the CDDM loss crosses the bar at a shallow angle, falling about 0.6-0.9% per 10% more
    iterations at the crossing. A level difference under 1% therefore moves the crossing by tens of
    percent.
  - The looser margins do not fix it: errors stay between -43% and +26% at 1.10x and 1.20x.
  - The oracle cannot pin the crossing either: from batch noise alone, its 90% bootstrap range at
    1.05x is 15-25k or 17-26k on two control runs. The rule's t* is inside that range on 8 of 15
    runs at 1.05x.
  - The misses that remain are early reads on the fast-training winp2 and winp5 arms: 4-7k against
    oracle crossings of 6-10k, near the 2k fit start, where the oracle has only one snapshot per
    window.
- **DMTS: fails.** The curve sits 3-13% below the oracle mean and 3-6% above the oracle median over
  snapshots. Snapshots taken during instabilities below the 2x burst threshold raise the mean; the
  Hampel filter trims those iterations from the curve. t* agrees within 1% on one run and is 20% early
  on another, where a sub-threshold instability keeps the oracle above the bar until 80k.

**Also found.** 44 of the 1,117 runs read at 1.05x end training more than 10% below their own fitted
floor: 23 flip-flop, 13 DMTS, 4 from the retracted sweep, 3 flip-flop hyperparameter runs and 1
CDDM. Figure: `img/internal_figures/readout_loss_ends_below_floor.png`, with the histogram in
`..._hist.png`.
- The fit misses a slow late decline (the SGD arm of B5: floor 0.072, loss 0.058 at 200k, read at
  35k) or a second learning phase (CDDM tanh).
- Their read-outs are early. The existing guard catches only losses that end more than 1.5x
  *above* the floor.
- Proposed to Pavel, not adopted: fall back to the smoothed-minimum floor for such runs, as already
  approved for beta on its bound, and exclude any run that still has no read-out.

**Ablation: is the Hampel smoother needed, or only the sustained crossing?** Pavel asked to commit
only what earns its complexity. The minimal variant keeps the old 2,001-iteration median and the
log-binned floor fit, and adds only the sustained crossing and the 1.10x and 1.20x margins. It was
run through the same three tests.
- **Synthetic traces.** Without bursts it passes: floor within 0.6%, t* within 6%. With planted
  bursts at realistic noise it fails: the floor is 2.5-3.9% high and t* 10-17% early. A burst
  thousands of iterations long fills a large share of a log-bin and moves its median.
- **Truncation.** The 90th-percentile shift is 18% at the 75% cut and 28% at the 50% cut, against
  12% and 23% for the full rule.
- **Oracle.**
  - Flip-flop: the same as the full rule.
  - CDDM controls: +86% and +73%, against +47% and +63% for the full rule.
  - DMTS: +77% on one run where the full rule is at -20%.
- **Verdict.** The full rule is kept. Its added code (the Hampel curve, burst detection and masking)
  is what handles bursts. The minimal variant would need the same machinery to fix the same cases.

**What this means for comparisons.**
- On the flip-flop the 1.05x read-out time is reliable to a few percent.
- On CDDM the read-out *level* is reliable but its *time* is not better than about ±30%, under any
  rule, because the loss changes too slowly near the bar. A property that keeps changing through
  that stretch (active units do) should be read as a function of the loss level, or reported
  together with its spread across the margins, rather than at one t*.
- On DMTS the loss level depends on whether brief instabilities count, by up to 13%.
