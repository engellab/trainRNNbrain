## 2026-09-21 12:01 — the dropout sampler was scoring the wrong variable; three fixes and a re-run

Pavel read the Figure 2b caption and asked whether it really meant the sampler works in `h` space
rather than firing-rate space. It did, and chasing that opened two further defects. All three are
fixed and the dropout sweep is being redone from scratch.

### Defect 1 — the score was even in h (`f151570`)

`Trainer.get_participation_` read the RAW states, so for `equation_type: h` the sampler scored
`q_0.9(|h_i|) + std(|h_i|)`. That expression depends on `h` only through `|h|`, so a unit pinned at
`h = -c` (permanently silent, `r = 0`) scored **identically** to one pinned at `h = +c` (maximally
active). Measured on the trained networks: silent units score 0.008 ± 0.016 under the rate-based
score against 0.847 ± 0.608 for active ones, where the old score gave 0.959 vs 1.507 — overlapping.
51 ± 4% of the drops landed on units that were already silent, where dropping is provably inert
(`max|Δh|` over every other unit = 0.000e+00, measured for both dropout kinds).

The fix deletes that scorer — its only caller was the dropout EMA — and points the sampler at
`participation_from_states_`, which already applied the activation for `h` and takes `s` states as
rates. It also gained the `q` argument it never had, so `dropout_args.activity_q` is finally
honoured; the trace call keeps `q = 0.9`, so every logged participation vector is bit-identical and
no existing analysis moves. One definition of participation now, which is what makes the bug
unrepeatable.

### Defect 2 — β was secretly a dose knob (`9a05f27`)

`p_i = clamp(drop_rate*N*softmax(β v)_i, 0, 0.999)`, independent Bernoulli per unit. Since the
softmax sums to one the intended dose is exactly `drop_rate*N`, but a saturated unit's excess above
0.999 was **discarded rather than redistributed**, so the dose collapsed as the softmax concentrated:
at `drop_rate = 0.05`, N = 1000 (nominal 50), β = 1 → 49.1 dropped, β = 2 → 33.3, β = 4 → **7.9**,
β = 8 → 2.1.

**This voids the drop-rate ladder's β = 4 arm.** That arm was not "sharper targeting is worse than
useless"; it was dropout almost switched off, which is exactly why it landed on the no-dropout count
(394.7 ± 71.7 against 387.0 ± 37.3). The targeting axis is *unexplored*, not explored-and-null, and
the manuscript sentence asserting otherwise has to go.

Pavel's suggestion replaced my water-filling patch and is better: draw **exactly
k = round(drop_rate · |pool|)** units without replacement with probability ∝ `softmax(β v)`
(`torch.multinomial`). Dose is fixed at every β, no cap, and the Bernoulli count variance is gone.

### Defect 3 — the pool included the dead (`a38eadd`)

Also Pavel's: sample only from the live units, `{i : v_i ≥ active_rel · q95(v)}`, the paper's own
scale-free criterion. Wasted drops become 0 **by construction** rather than by tuning β up, which
leaves β one job: how sharply to prefer the busiest of the living. `drop_rate` now multiplies the
live pool rather than N — `drop_rate * N` is unsafe here, since a 1000-unit net has ~260 live units
late in training and 0.25·N would ablate most of the working network every iteration.
`active_rel: 0.05` added to all four trainer configs.

Production path, 100 draws × 4 trained nets, `drop_rate = 0.05`:

| sampler | dropped | on live | wasted | hits in the true busiest 50 |
|---|---|---|---|---|
| SHIPPED `v_h`, β=1, pool=all | 50.0 | 24.5 | 50.9% | 3.6 |
| rate score, β=1, pool=all | 50.0 | 29.8 | 40.3% | 8.7 |
| rate score, β=1, pool=live | 18.0 | 18.0 | 0% | 5.5 |
| rate score, β=2, pool=live | 18.0 | 18.0 | 0% | 8.5 |
| rate score, β=4, pool=live | 18.0 | 18.0 | 0% | **12.7** |

Two smaller fixes rode along: `eta` defaults to 0.5 rather than 0.0 (at 0 the EMA would freeze at
its 1e-6 initialisation and participation dropout would silently degrade to *uniform* while the
config still read `participation`), and an unrecognised `sampling_method` now raises with the
offending string instead of falling through to an `UnboundLocalError`.
`tests/test_dropout_samples_on_rate.py` pins all of it, five checks.

### When the sampler is blind, and what that bounds

The sampler's own EMA has never been written to disk, so an agent instrumented a real `train_step`
for 1,000 iterations and logged it. Waste is already **36 ± 8%** by iteration 100 with selectivity
1.23 ± 0.15 — early on the sampler actively *preferred* silent units. Among the units still alive it
ranks them correctly early (ρ = 0.63 ± 0.03) and that decays to **−0.04 ± 0.05** by 150k: late in
training the shipped ordering carried no information about which living unit was busiest.

The caveat cuts the other way and belongs in any write-up: the die-off is heavily front-loaded.
**41% of the entire 0→150k loss happens in the first 100 iterations** (943 → 705 active in the
dropout arm), and 86% of the no-dropout arm's 0→40k decline is done by iteration 5,000. No sampler
improvement can touch that first chunk, whatever β is.

### Stage 1, submitted to Della

36 jobs, `NBitFlipFlop_std_dropfix`, N = 1000, k = 3, pen = none, do = dead, 40,000 iterations:
`drop_rate ∈ {0.05, 0.10, 0.175, 0.25}` × `β ∈ {1, 2, 4}` × 3 seeds. Della because Spock is
saturated (the DMTS N=4000 jobs and ~50 others are queued there) and Della's queue is empty.
The tree is `~/trainRNNbrain_dropfix`, populated by rsync since the push to GitHub was blocked, so
the launcher reads `CODE_VERSION.txt` (`a38eadd`) for provenance and aborts without it.

Pre-registered, written before the jobs existed. Read-out at 40,000, active units under the
scale-free criterion. References already on disk at the same read-out, and still valid — the only
training-path commits between them and now are the `scored_` refactor (identical for a time-index
mask, which this task uses) and the OOM chunking (exact in float32):
no dropout 387.0 ± 37.3 (n = 4); dropout dead, rate 0.05, β 1 → 487.0 ± 28.0 (n = 7).
**BAR: an arm wins only if its mean clears 571 = 487.0 + 3 × 28.0** — the drop-rate ladder's bar,
reused unchanged so the two are directly comparable. Cost guard: an arm clearing the bar whose clean
loss at 40k exceeds the no-dropout reference by >10% is reported as a trade, not a win. If no arm
clears it, the conclusion is that dropout's ceiling is structural and survives the correction of all
three defects — a stronger negative than the paper currently has, and that is what will be written.

**STAGE 2, already decided: the same grid at N = 500 and N = 2000** (72 further jobs), to test
whether any benefit scales with network size. Run in full only if stage 1 produces an arm worth
scaling; if nothing clears the bar, stage 2 becomes a single confirmatory arm at each size.

### Two manuscript corrections this forces, not yet applied

1. The quoted Spearman **ρ = 0.24 is one network**, not four — `fig_paper_F2.py:300` indexes `[0]`,
   and the folders sort by r² so net 0 is the worst one with the lowest ρ. Per-net: 0.237, 0.392,
   0.496, 0.313 → **ρ = 0.36 ± 0.10 (n = 4)**. Appears in `sections/03_dropout.tex:51`,
   `main.tex:140`, `docs/measured_facts.md:195`. The qualitative claim is unchanged.
2. "Sharper targeting is worse than useless" (`sections/03_dropout.tex:74,79`, `main.tex:155`,
   `docs/measured_facts.md:246`) rests on the void β = 4 arm and must be withdrawn.

