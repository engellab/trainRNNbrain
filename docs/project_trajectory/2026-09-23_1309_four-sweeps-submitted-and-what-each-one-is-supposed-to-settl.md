## 2026-09-23, 13:09 — Four sweeps submitted, and what each one is supposed to settle

All on Della, 31 jobs. Spock is effectively unusable: fairshare factor **0.0098** on a 0–1 scale,
because recent usage is 5.1% of the whole cluster against a 0.33% entitlement, roughly 15× over.
Fairshare carries weight 100,000 against 1,000 for age, so these jobs score 977 where an idle user
scores ~99,000. The decay half-life is 20 days with no reset, so this does not clear overnight.
The DMTS ladder sat there 20 hours without one task starting and was moved.

### 1. DMTS delay ladder — `14318554`, 9 jobs, 8–10 h each

8, 10 and 12-step delays at N=1000, no penalties, 150k iterations, 3 seeds.

**Settles:** the shortest delay the unpenalised network solves reliably, which is the delay the
Figure 1c scaling series has to use. 16 steps is a 0/3 rung (r² 0.452, 0.452, 0.450) and 36 steps
defeats every arm including `frm`.

**Read-out, fixed before submission:** the SOLVE RATE per delay, not a mean r². Averaging a bimodal
0.45/0.99 split describes no network that exists. Active-unit counts reported separately for
solvers and failures, never pooled — at N=500 the solver had 170 active units against 123 and 113
for the failures, a 45% gap that makes any pooled count a selection artifact. Decision rule: take
the shortest delay with 3/3; if no rung reaches 3/3 the answer is more seeds, not a shorter delay.

### 2. Dropout size sweep — `14312455`, 12 jobs

`mute` and `dead` at drop rate 0.20, β=4, N ∈ {500, 2000}, 3-bit flip-flop, 40k iterations.

**Settles:** whether dropout recruits a FRACTION of the network or a FIXED COUNT. At N=1000 `dead`
at β=4 reached 963.3 ± 3.2 of 1000 units at a 25% rate and 809.7 ± 10.2 at 17.5%, against a
no-dropout control of 316.3 ± 33.3. Those sit close enough to the ceiling that one size cannot
separate the two: "96% of units" predicts ~1920 active at N=2000, "about 960 units" predicts ~960.
The scaling figure cannot use the dropout result until this is decided.

Rate 0.20 is the midpoint of `dead`'s steep region and is a NEW rate, so there is no N=1000 cell
for it; the anchors are the 17.5% and 25% cells on disk.

**Controls come from sweep 3**, same task, size, iteration count, config and commit.

### 3. Flip-flop revival — `14310257`, 18 jobs (gamma = 0)

Control, duplication and redistribution at N ∈ {500, 1000, 2000}.

**Settles:** whether duplication's equilibrium is a property of the method or of one network. On
CDDM it held ~70% of units (707 of 1000) as an equilibrium, not a ceiling — flat from iteration
11,000 while replacements continued at 450–550 per 1,000 iterations.

**Already landed, and one result is bad:**

| cell | active units | % of N | clean loss | note |
|---|---|---|---|---|
| N=1000 none | 274.7 ± 24.0 | 27.5% | 0.0775 | |
| N=500 none | 211.7 ± 14.2 | 42.3% | 0.1818 | |
| N=500 prune_copy | 285.0 ± 193.9 | 57.0% | **273,489** | **BROKEN** |

⚠️ **Duplication destabilises at N=500 on all three seeds** — 29–44% of gradient updates discarded,
27,299 replacements over 500 units (54.6 each, the treadmill signature), and a standard deviation
almost as large as the mean. By the pre-registered stability rule (>5% of updates discarded = broken
regardless of unit count) this is not a result. On CDDM at N=1000 the same mechanism ran with zero
skipped updates. Size and task differ at once between those two runs, which the N=1000 and N=2000
flip-flop cells separate.

⚠️ **The control comparability check is marginal.** The N=1000 `none` cell gives 274.7 ± 24.0 against
the dropout sweep's 316.3 ± 33.3 — Welch t = 1.76, p ≈ 0.16, so not significantly different, but
13% lower and just outside one sd. The dropout size numbers should be read against these
same-commit controls rather than against 316.3, and the 13% gap between two supposedly identical
controls is itself worth explaining.

⚠️ **The silent fraction is not constant across size even with no intervention:** 42.3% of units
active at N=500 against 27.5% at N=1000. Every claim about what an intervention recruits has to be
read against that drift.

### 4. Cubic saturation — `14319650`, 6 jobs (gamma = 0.1)

Control and duplication at N=500, 3-bit flip-flop, own `ff_revive_g01` folder because gamma changes
the dynamics for the control too.

**Settles:** whether bounding the dynamics fixes the N=500 duplication blow-up. `-gamma*x^3` bounds
x while leaving ReLU's exact-zero floor and its frozen-gradient behaviour intact, which are the two
properties the duplication argument rests on.

**Why not a bounded activation.** `rnn_sigmoid_shifted_standard` would also stop the blow-up and
would dissolve the phenomenon with it: its own config header states that no unit can be exactly zero
there and every unit has gradient everywhere, so a quiet unit is revivable by the optimiser alone
and duplication answers a question that no longer exists. Its 0.10 activity floor also breaks the
`0.05 * q95` criterion, so no sigmoid count could sit beside 316.3, 410.3 or 963.3.

**Read sweep 3's N=1000 and N=2000 cells BEFORE this one.** Size does not remove a failure mode, it
changes how often it fires, and three seeds cannot tell "rarer" from "gone". If duplication turns
out stable at N=1000 on flip-flop, then N=500 is the odd case and gamma is treating a symptom whose
cause is still unidentified.

### Not submitted, and why

**Synaptic scaling.** Still broken. At eta=0.5 it discarded 399 of its first 400 updates; at
eta=0.05 it discarded 499 of 1000 and ended at r² = −2.5e19. Two fixes have now failed the same
way — forcing the geometric mean of the scale factors to 1, and reducing eta — and **both passed
tests that had no training in the loop**. The next step has to be a diagnostic on a live run
(spectral radius and gradient norm logged at every scaling event during actual training), not a
third repair validated the way the first two were.

**`FFinpcap`** — 9 jobs, still pending on Spock since 2026-09-22, testing an intervention that did
nothing on CDDM (420.7 active units against a control of 410.3). Worth cancelling rather than
porting; it consumes queue position and future fairshare for a measured null.

---

