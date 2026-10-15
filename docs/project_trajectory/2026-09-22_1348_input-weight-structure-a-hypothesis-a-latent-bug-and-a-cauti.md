## 2026-09-22 13:48 — input-weight structure: a hypothesis, a latent bug, and a caution

### The hypothesis (Pavel)

Input drive in the unpenalised network is concentrated on a few units, and that concentration is
what leaves the rest silent; capping |W_inp| should force the drive to spread and recruit units.

Measured on trained flip-flop networks, N = 1000, no dropout, 4 seeds per arm:

| arm | input row-norm CV | active units | corr(row norm, log participation) |
|---|---|---|---|
| `none` | 2.28 | 262.5 | **+0.65** |
| `rws` | 2.54 | 221.0 | +0.51 |
| `frm` | 1.08 | 965.8 | +0.07 |
| `frm+rws` | **0.74** | 1000.0 | **−0.61** |

Input drive equalises in lockstep with the active count, and the coupling between drive and activity
**inverts**: unpenalised, a unit's input weights essentially decide whether it fires; under
`frm+rws` they do not, and if anything the units with least drive are busiest. At the weight level
the unpenalised network puts its drive into ~6% of entries — 99th percentile |W_inp| = 8.57 against
a median of 0.002, a 4000× ratio.

**This is correlational and that is the point of the experiment.** `frm` both equalises input
weights AND recruits units; the direction is unknown. Recruiting a unit may FORCE its input weights
up — a unit that fires needs drive — in which case equalisation is a consequence and capping does
nothing.

### Running: `FFinpcap` (Spock 6321178), 9 jobs

Post-step clamp on |W_inp| at 0.1 / 0.5 / 2.0, × 3 seeds, N = 1000, k = 3, pen = none, 150k.
Implemented as `model.inp_weight_cap` (null = off), mirroring the existing bias clamp.

**Not** the `inp_weights_magnitude` penalty, though it exists and is wired in as `lambda_iwm`: its
γ = 5 hinge is unusable here. At init every |W_inp| ≈ 0.03 sits under any sensible cap so the term
is exactly 0 with no gradient; on a trained network the largest weight is 41× a cap of 0.36 and the
term reaches ~1e8. Any λ is inert at one end or explosive at the other, and this project already
carries `spike_factor`/`restore_after` because `frm` misbehaved far more mildly.

Read-out is CO-PRIMARY: active units **and** clean r². A cap that recruits units by destroying the
task is not a result — an arm supports the hypothesis only if it clears 305 active units
(262.5 + 3 SD) and stays within 10% of the reference loss. Row-norm CV reported too: **if CV falls
but the count does not, the causal direction is settled the other way**, which is equally
informative and should be written up as such.

### Latent bug found and fixed (`d70b015`)

`inp_weights_magnitude_penalty` read `N, U = states.size(0), states.size(1)`, so `U` was the TRIAL
LENGTH and the cap came out as `cap100 * (T/N) * log1p(N)/log1p(T)`. Two errors: the cap depended on
how long a trial is, which is a property of the task and has no business in a weight scale (the same
1000-unit network got 0.182 on a T=300 task and 0.278 on a T=500 one), and the log ratio was
inverted relative to `out_weights_magnitude_penalty`, so the N-scaling ran too steeply — 4× the
units gave a 3.3× smaller cap where the intended form gives 1.2×. `U` was meant to be `self.UpV`,
the hard constant 100 its siblings use. **No result is affected**: `lambda_iwm` is 0 everywhere, so
the penalty has never been applied to a run on disk. `tests/test_weight_cap_scaling.py` pins both
properties. Caught by Pavel reading the formula.

### How input-weight quantiles scale with N (CDDM, 4 sizes, 3 seeds)

Figure: `img/internal_figures/fig_winp_quantiles.pdf`.

|  | q50 | q75 | q90 | q95 | q99 | max |
|---|---|---|---|---|---|---|
| `none` N=500 | 0.024 | 0.086 | 0.734 | 0.850 | 0.925 | 1.95 |
| `none` N=5000 | 0.009 | 0.014 | 0.024 | 0.130 | 0.353 | 1.07 |
| `frm+rws` N=500 | 0.053 | 0.138 | 0.576 | 0.745 | 0.927 | 2.41 |
| `frm+rws` N=5000 | 0.036 | 0.064 | 0.114 | 0.235 | 0.382 | 1.95 |

Fitted exponents for `frm+rws`, q ~ A·N^b: q50 −0.175, q75 −0.316, q90 −0.709, q95 −0.506,
q99 −0.394, max −0.102. For reference 1/√N is −0.500.

**The penalties' log form is far too flat to serve as a cap law.** `cap100·log1p(100)/log1p(N)`
falls 1.4× from N=500 to 5000 where the measured q90 falls 5×. A free fit gives
**cap(N) = 41.7·N^(−0.709)**, i.e. 0.31 at N = 1000. Either fit the power law directly or use
1/√N, which sits inside the spread of per-quantile exponents and has the merit of matching the
initialisation scale rather than being an arbitrary fit to four points. The clamp as submitted is a
plain absolute number, which is fine at a single N but **not portable to a size sweep**.

### ⚠️ The caution, which may matter more than the fit

**`frm+rws` may be the wrong target for a cap.** Its input weights are not capped: its max is
1.9–2.4, essentially unchanged from `none`. What differs is the BULK — it has far fewer near-zero
weights, its median holding at 0.036 at N=5000 where `none` decays to 0.009. So `frm+rws` **raises
the floor rather than lowering the ceiling**, and a cap can only do the latter.

If the mechanism is "give every unit enough drive" rather than "stop a few units hogging it", then
the intervention that reproduces `frm+rws` is a MINIMUM on the input row norm, or `input_row_norm`
(which already exists and equalises rows at initialisation), not a cap. The three running caps
straddle the measured q90 of 0.29 at N = 1000 so they test the question either way, but **"does
capping recruit units" and "does capping reproduce frm+rws" are now separate questions**, and the
quantiles suggest the second may be no even if the first is yes.

---

