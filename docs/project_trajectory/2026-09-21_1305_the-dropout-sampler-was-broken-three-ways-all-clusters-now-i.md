## 2026-09-21 13:05 — the dropout sampler was broken three ways; all clusters now idle

Continues the 12:01 entry. Pavel's question about Figure 2b ("does the sampler really work in h
space?") turned into four defects, three fixed, and a decision to stop and re-plan.

### Defect 4, found by calibrating instead of submitting: β was a scale artefact (`4f7d73f`)

The single-cell calibration before the 36-job grid failed loudly, which is the whole reason to run
one. ρ = 0.05, β = 4 finished 600 iterations with a healthy-looking training loss (0.207, falling)
and **r² = −78,257** on the full network. Weights were fine (W_rec |max| 0.27); the ACTIVITY had
exploded, participation 0.085 → 579 by iteration 300. Controls on disk sit at 0.8–1.3 over the same
window, so this had no precedent.

Cause: `v` is a firing-rate statistic with no fixed scale and it grows more than tenfold over
training, so `softmax(β·v)` means different things at different times. Enrichment of the busiest
live unit over uniform:

| β | raw v @ iter 100 | raw v, trained | rank-normalised |
|---|---|---|---|
| 1 | 1.5× | **112.6×** | 1.6× |
| 4 | 4.5× | **375.0×** | 4.1× |

375× on a 375-unit pool = drawn every iteration. Fixed by having β multiply the unit's **rank**
within the pool: enrichment ≈ β, identical early and late, bounded by construction.

**z-scoring was considered and rejected on measurement**, not taste: it removes a uniform rescaling
but not a heavy tail (one outlier at 400× gives z = 19.3, so exp(4z) = 4e33 → same collapse), and it
*amplifies* concentration on a healthy population (18×/93×/257× at β = 1/2/4 against raw's
2.1×/4.0×/13×).

### Defect 5, which rank scoring bounds but does not remove: `dead` is intrinsically unstable

With `dead` the unit leaves the recurrent rhs and the task loss is scored on the dropout pass, so a
unit drawn on a fraction p of iterations is regularised on only (1−p) of them. As p → 1 it grows
unopposed. Under rank scoring p(busiest) ≈ β·ρ, so the grid's own corner is dangerous, and a
pre-registered calibration at ρ = 0.25, β = 4 (threshold from n = 18 healthy controls) **failed
3/3**, blowing up inside 900 iterations.

**Sharpening the targeting IS raising p.** You cannot have accurate `dead` targeting and a loss that
sees the targeted units. This is a property of the method, not a tuning problem, and deserves a
sentence in the paper.

A wording correction that matters: the "healthy train loss" reported alongside those failures was
the *dropout-pass* loss, which is structurally blind to this failure. The full-network clean loss of
the exploded run went **0.49 → 20,590 → 53,277** over iterations 100/200/300. The manuscript is
sound here — Methods already states that every performance comparison uses the dropout-OFF clean
loss — but the claim should never have been phrased as "no cost".

`mute` is immune by construction (`RNN_torch.rhs` masks nothing unless kind == "dead"), and costs
nothing: on data already on disk, **mute 501.9 ± 43.5 vs dead 487.0 ± 26.0 at 40k, and
373.1 ± 18.4 vs 362.1 ± 8.4 at 150k** (n = 7 each, Welch p = 0.22). The launcher now defaults to
`mute`. The mute stability calibration reached iteration ~560 clean (r² 0.39–0.41 where `dead` was
at −24,146 by 360) but was cancelled before completing, so it is **unconfirmed to 5,000**.

### Also fixed

Bias is now **fixed at 0** by default (`ae52c58`): `rnn_relu_standard.yaml` had a non-degenerate
`bias_range`, making it a trainable Parameter. It was never doing anything — fixing it is worth
+5.0 units, and a permanently sub-threshold unit has `dL/dbias = 0` exactly (measured), the same
absorbing state that traps its weights. **Every run up to `4f7d73f` used a trainable bias, so
bias-free arms cannot be pooled with the existing corpus** — matched controls are required.

Launcher provenance now reads git (the clusters sync by push/fetch, not rsync) and an unclean
checkout is FATAL rather than a warning.

### Everything cancelled; both queues idle

- Spock `FFdropN` (6 running + 28 pending) and `FFdrate` (12 pending): both ran the blind sampler at
  commit `6e43d53`, so they measured the superseded method. `FFdrate` also duplicated a ladder Della
  had already completed.
- Spock `DMTSpen` N = 4000 (3 pending): cancelled pending a task-encoding decision, below.
- Della `FFdropfix` mute calibration: cancelled at Pavel's request, ~5 min short.

### Open decision: input encodings

Pavel noticed DMTS spends 4 one-hot channels on 4 stimuli because inputs used to be non-negative,
and `io_nonnegativity` is now false. Analysis:

- **CDDM reduces losslessly, 6 → 3.** `motion_r + motion_l = 1` and `motion_r − motion_l = coh`, so
  each antagonistic pair is a redundant 2-channel encoding of one signed scalar; the 2-way context
  cue is 1 bit. No geometry changes.
- **DMTS does not.** Four equidistant stimuli need 3 dimensions (tetrahedron); one-hot in 4-D has all
  pairs √2 apart, a 2-channel square gives √2 adjacent and 2 opposite. Re-encoding *changes the
  task*, so all four sizes must be re-run together.
- **Cost decides it:** DMTS is 45 trained networks, CDDM is **805** — including the N = 100–10,000
  series Figure 1c rests on. Consistency demands both or neither.
- Recommended neither: no claim in this paper depends on input dimensionality, and the encodings are
  a defensible opponent-coding construction. The more substantive DMTS wart is the **25% / 75% class
  imbalance** (4×4 pairs, 4 matches) — a network answering "non-match" always scores 75%. That is
  fixable inside the current encoding.

