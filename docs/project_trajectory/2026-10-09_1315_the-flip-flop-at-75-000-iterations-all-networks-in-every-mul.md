## 2026-10-09 13:15 — the flip-flop at 75,000 iterations, all networks in: every multiplier grows

**All 3-bit flip-flop top-ups have landed.** Della `F2top75` 15181151 (N = 2000 mute, one net) finished
2026-10-07 21:02 and 15181152 (N = 4000 synaptic noise, four nets) 2026-10-08 14:00. 245 flip-flop
continuations sit under `~/trainRNNbrain/data/trained_RNNs/NBitFlipFlop_*_c75000`. The seven from
`topup_manifests/topup_rerun.txt` had no `parent_40k/`. They were nested on 2026-10-09 the same way as
the other 238: a copy of the parent named in the continuation's `paths.init_from`, plus
`PARENT_SOURCE.txt`. No parent is continued twice.

**The cache was rebuilt** as `/scratch/gpfs/TENGEL/pt1290/f2_75k/shards_v2` (Della 15274159, 16
shards, `f2cache75k_v2.slurm`). `shards/` is the 236-network run of 2026-10-07 and is kept. The
read-outs are `ff75k_readout_v2.txt` and `ff75k_traces_v2.txt` beside it. 242 networks are scored and
232 pass r2_common ≥ 0.8 at both read-outs.

⚠️ **Three continuations diverged and the cache gate drops them** (stored r2 at 75k): frm + rws at
N = 500 (−1876), rescale at N = 2000 (−14,869), and the rescale_active `f_a=1.002_tgt=2.5_act=true`
rerun (−45.5). The fourth net in that last cell already failed at 40k (−112.7) and stays bad at 75k
(0.549). That cell, `norm=f ... act=true`, is broken at both budgets and is not an operating point.

⚠️ **The cache is not reproducible to the unit.** The common-condition scoring draws its noise
unseeded (only `weight_shape` takes a fixed rng), so rebuilding the same networks moves active counts
by 1–5 and r2 in the third decimal. It also moved a borderline gate: `norm_mode ... median_active_tgt=6.0_act=true`
had one network at 40k r2 0.798 (dropped) on 2026-10-07 and is above 0.8 now (n = 2 → 3). Treat
differences under about 5 units as noise.

**Multiplier over the control, same N and same read-out** (`ff75k_readout.py` section 6, operating
point by `fig_paper_F2.pick_cell`). Every arm's networks are disjoint from the control's at both
read-outs, in every row.

| N | mute | duplicate | rescale | synnoise | frm + rws |
|---|---|---|---|---|---|
| 500 | 1.89 → 2.04x | 2.03 → 2.22x | 1.52 → 1.54x | 1.76 → 1.90x | 2.29 → 2.51x (n = 1) |
| 1000 | 1.76 → 1.95x | 2.49 → 2.92x | 1.53 → 1.57x | 1.84 → 2.08x | 3.16 → 3.39x |
| 2000 | 1.59 → 2.00x | 2.57 → 3.33x | 1.46 → 1.86x | 1.34 → 1.63x | 4.08 → 5.36x |
| 4000 | 1.75 → 2.01x | 2.83 → 3.70x | 1.69 → 1.94x | 1.36 → 1.65x (n = 5) | 6.05 → 6.80x (n = 1) |

The r2_common cost against the control stays between −0.1% and −4.3% and does not grow beyond
noise: at N = 2000 and 4000 it shrinks or holds within 0.1 point for every arm. The multipliers grow
mainly because the **control keeps silencing** (−9.8% of its 40k count overall, −20% at N = 2000,
−12% at N = 4000) while mute, synnoise and frm + rws change by −3% to +7% of their own 40k counts. Duplicate is the one arm that also gains in
absolute terms: +51 units on average, +258 at N = 4000.

N = 4000 synaptic noise, now 5 networks rather than 1: 845 → 901 active (paired +56, range +38..+87),
1.36x → 1.65x the control. The single-network figure on 2026-10-07 (1.33x → 1.61x) holds.

**Is the control still silencing at 75k?** (`ff75k_traces.py`: scale-free live count read off the
joined 0 → 75k trace, mean of snapshots within ±1500 iterations, not fitted.)

| control | 40k | 60k | 75k | 40→60 | 60→75 | still falling 60→75 |
|---|---|---|---|---|---|---|
| N = 500 | 211 | 206 | 200 | −5 | −6 | 5/6 |
| N = 1000 | 296 | 286 | 274 | −10 | −12 | 6/6 |
| N = 2000 | 452 | 370 | 360 | −82 | −10 | 1/3 |
| N = 4000 | 625 | 555 | 544 | −70 | −11 | 3/3 |

At N = 2000 and 4000 most of the loss happens between 40k and 60k, and it slows to about 10 units
per 15k after that. At N = 500 and 1000 the control loses units at a steady 5–12 per 15k across the
whole window. So the control has not stopped at 75k at any size. The 75k multipliers are a later
point on a curve that is still rising slowly, not a converged value. Of the arms, only rescale
(N = 500: −37 from 60k to 75k; N = 4000: −37, 4 of 6 falling) and frm + rws at N = 1000 (−30) lose
units over the last 15k. Mute and synaptic noise are flat or rising at every size.

### Correction to the entry above, the same day — the "diverged" networks did not diverge, and the table is the wrong read-out

**The three dropped continuations are not divergences.** Their typical clean loss stays normal to the
end: frm + rws N = 500 has a median of 0.073 at 70–75k, and rescale `size_series` `rescale_tgt10` N = 2000
about 0.11. What they show are brief spikes, up to 33 and 2.6e14. The rescale N = 2000 net was already
spiking at 30–40k (max 5.8e5) before its continuation started. The cache scores `LastParams`, so a
network whose last save lands inside an episode scores −1876 or −14,869. Only one of the 245 has its
last trace sample inside a spike (that rescale net, ~5e3 over its final 60 iterations). The third,
rescale_active `f_a=1.002…act=true` (−45.5), was broken before the top-up (parent −20.7 at 40k).
Spikes (> 10x the run's median clean loss) are confined to the rule-based arms and were as common
before 40k as after: target_ladder 24/24 before vs 22/24 after, recgrow 11/12 vs 10/12, norm_mode 10/12 vs 9/12,
ff_both40k 5/10 vs 8/10. After 40k there are none in std_bernoulli (36), dropout_sizes (12), synnoise (21) or the
controls. This is the same thing the DMTS read-out recorded (item 4 of the 2026-09 DMTS penalty entry):
"the last network is a lottery ticket". 75k simply gave two more networks the chance to be saved mid-episode.

**The multiplier table above is read at fixed budgets, and the project's rule is 1.07x each network's
own fitted floor** (Pavel). `ff75k_atfloor.py` applies that rule with `fig_knobs_at_floor`'s guards, and
on these sweeps **it is not usable on the noise-free probe.** The control's smoothed `loss_clean_train`
falls to ~0.075 by 5–10k, then rises 2–3x and settles at 0.13–0.18. Duplicate does the same; synaptic noise
does not (0.053–0.056 at 75k, final / min 1.00–1.12). The late-rise guard (final > 1.5x min) therefore
drops all 6 controls at N = 500 and 1000, 2 of 3 at N = 2000 and all 3 at N = 4000. Without the guard,
the first crossing falls on the initial descent, before the silencing. Meanwhile r2_common, scored
under the shared noise, IMPROVES 0.946 → 0.952 over the same window. So the noise-free loss and the
deck's score disagree on these networks. This is the same failure `fig_knobs_at_floor` documents for
the flip-flop input-scale ladder. No at-floor multiplier is reported until a loss series is chosen.
The fixed-budget table stands only as an end-of-training read-out.

