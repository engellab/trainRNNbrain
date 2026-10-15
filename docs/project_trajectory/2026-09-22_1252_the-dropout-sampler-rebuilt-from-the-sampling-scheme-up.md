## 2026-09-22 12:52 — the dropout sampler rebuilt from the sampling scheme up

A day spent almost entirely on one question Pavel asked about Figure 2b, which turned out to reach
all the way down to how units are drawn.

### What was wrong, in the order it was found

1. **The score was blind to firing** (`f151570`, yesterday). Fixed by scoring the rate.
2. **β was a scale artefact** (`4f7d73f`). `v` grows tenfold over training, so `softmax(β·v)` meant
   different things at different times — enrichment of the busiest unit was 1.5× at iteration 100
   and 113× on a trained network *at the same β*. Fixed by having β act on normalised **rank**,
   which makes enrichment ≈ β, identical early and late and independent of pool size.
3. **`dead` is unstable under accurate targeting** (`ae52c58`). The task loss is scored on the
   ablated network, so a unit dropped on a fraction p of iterations is regularised on only (1−p);
   as p → 1 it grows unopposed. Diverged 3/3 at ρ=0.25, β=4 (p = 0.77) within 900 iterations.
4. **`rescale` was a silent no-op** (`8b5029b`). `get_dropout_mask` handed survivors a factor > 1
   and both consumers threw it away — `Wout_c * (dropout_mask > 0)` on the mute path and a
   `dm.max() <= 1` guard on the dead path. Nine GPU-hours ran as a duplicate of the unrescaled arm
   before it was caught. Testing the sampler in isolation could not catch it; the test now follows
   the mask into the forward pass.
5. **A uniform rescaling factor cannot be right under targeted dropout** — Pavel's objection. A
   single M/(M−k) leaves a unit dropped 10% of the time contributing 1.50 and one dropped 80% of
   the time contributing 0.33, when both should be 1.00.

### The design that came out of it (`dae5269`)

The fix for (5) needs each unit's marginal drop probability πᵢ, and under exactly-k
without-replacement sampling that is intractable: it is **not** k·wᵢ (the with-replacement answer,
which returns 1.27 for the heaviest unit in a 5-unit example), and the exact value sums over every
size-k subset containing the unit — ~10⁹⁶ terms at our pool sizes. Monte Carlo would have meant a
cached table per live-pool size, and M takes ~525 distinct values over a run.

Pavel's question — "can we compute π in closed form, perhaps not using rank?" — has the right
answer: **change the sampling so π is something we set.** Draw each unit independently with
probability pᵢ and πᵢ *is* pᵢ. The cost is that the count fluctuates (100 ± 7 at ρ=0.25, M=400)
rather than being exactly k, which is acceptable.

pᵢ comes from **water-filling**: pᵢ = min(p_max, c·wᵢ) with c solved so Σpᵢ = ρM. Clipping without
redistributing is what made a nominal 50 drops become 7.9 at β=4. Pavel's iterative
rescale-and-clip converges in ~10 rounds and agrees with a bisection on c to 4e-10; it is what is
implemented, being simpler and needing no bracket for c. `p_max` doubles as the ceiling on the
rescaling factor (0.9 → 10).

**Two masks, not one.** `DropoutMask(outgoing, silence)`. `outgoing` carries the rescaling and
multiplies what the unit SENDS — its rate into W_rec for `dead`, its read-out column for both
kinds. `silence` is binary and gates the unit's OWN drive and noise, which only `dead` uses:
rescaling that would feed a *surviving* unit drive it never receives in the full network. The old
single mask was disambiguated at each site by sniffing dtype and max value, which is defect (4).

Verified **per unit**: E[outgoing] = 1 for every live unit, residual falling as 1/√draws
(0.058 → 0.054 → 0.029 at 4k/16k/64k against a predicted 2 SE of 0.095 → 0.047 → 0.024) and
correlating +0.65 with p. Aggregate checks pass under the *wrong* scheme too — that is how the
uniform factor survived scrutiny.

### What the superseded sweep established before its data was deleted

Recorded in `measured_facts.md` §2.6. The headline **contradicts the withdrawn drop-rate ladder**:
drop rate is monotone and strong once the drops are aimed (+35 → +99 → +175 active units over a
316.3 ± 33.3 control at ρ = 0.05 / 0.10 / 0.175), where the ladder found it irrelevant across an 8×
range while drawing from an ordering carrying no information about firing (ρ = −0.04 among live
units). And the ρ = 0.25 loss penalty is a **train/test mismatch, not a task cost**: those nets
score 0.0706 on the pass they trained on — the best in the sweep, against a 0.117 control — and
0.2193 on the full network, a ratio of 3.11× that falls to 0.96× at ρ = 0.05.

Also measured and **not** an artefact: shrinking W_rec by 25% *raises* the active count in every arm
(+94 control, +41 at ρ=0.25), so the unit gain is not drive inflation. Weaker recurrence means less
winner-take-all competition — the paper's own mechanism, seen from the other side.

### Running now

- **Della, 72 jobs** (`14274294` mute, `14274295` dead): N = 1000 flip-flop, ρ ∈ {0.05, 0.10,
  0.175, 0.25} × β ∈ {1, 2, 4} × 3 seeds × 2 kinds, 40k iterations, new sampler, rescaling on,
  folder `NBitFlipFlop_std_bernoulli`. Tests Pavel's hypothesis that `dead` revives units more
  fairly because it pressures recurrent rather than read-out redundancy. Pre-registered: supported
  if `dead` > `mute` at a majority of the 12 cells. Note it was already refuted-ish under the old
  sampler (pooled 373.0 ± 33.9 vs 357.4 ± 23.1, p = 0.27) but that predates the rescaling.
- **Spock, 48 jobs**: DMTSv2 and DMTS36v2, below.

Three stale job sets were cancelled today, the last of them (`14271770`, 9 jobs) after it had
already recreated cells in a folder cleaned an hour earlier. It survived three rounds of checking
because I kept reading job *counts* rather than what each array was running. The `SWEEP=` folder
override added afterwards makes a stale sweep visible by name.

### ⚠️ DMTS v2 looks much harder than v1, and this needs attention

First results from the redesigned 2-stimulus task, at 16τ, N = 500, `pen=none`, 3 seeds:
**0.4516, 0.4519, 0.9991**. One seed solves it; two fail. `rws` fails 3/3 (0.4514, 0.4520, 0.4525).
At 36τ, `none` gives 0.4717 ± 0.00004 and `rws` 0.4714 ± 0.0002 — six independent runs landing
within 0.0006 of each other, which is not "training is variable", it is every seed finding the same
degenerate solution (output zero through the pre-decision epoch, which is most of the scored
timesteps, then guess).

The old 4-stimulus task was solved reliably. The difference is the class balance: at 25/75 a network
scored 75% by always answering non-match, and the r² ≈ 0.999 values that made DMTS look "solved"
may have been partly that. On the balanced task the bit genuinely has to be held.

**This threatens Figure 1c.** The scaling panel counts active units in networks that solved the
task; if 2 of 3 seeds at N = 500 never learn it, the count is being measured on failures. Decide
before the panel is rebuilt: raise the seed count and report only solvers, revert to an easier
delay, or drop DMTS from the scaling figure.

