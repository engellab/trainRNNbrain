## 2026-09-22 13:12 — what is running, why, and what would count as an answer

A snapshot, because four experiments are live across two clusters and the reasons are no longer
obvious from the job names.

### The finding that reorganised the day

The redesigned 2-stimulus DMTS is far harder than the 4-stimulus version it replaced, and the
difference is the class balance we fixed. At 25/75 a network scored 75% by always answering
non-match, so the old r² ≈ 0.999 "solved" values were partly that artefact. On the balanced task
the bit genuinely has to be held — and at 16τ, N = 500, three seeds per arm:

| arm | r² per seed | solved |
|---|---|---|
| `none` | 0.9991, 0.4516, 0.4519 | **1 / 3** |
| `rws` | 0.4525, 0.4520, 0.4514 | **0 / 3** |
| `frm` | 0.9923, 0.9921, 0.9901 | 3 / 3 |
| `frm+rws` | 0.9927, 0.9934, 0.9930 | 3 / 3, and at HALF the iterations |

At 36τ nothing escapes: `none` 0.4717 ± 0.00004, `rws` 0.4714 ± 0.0002 — six runs within 0.0006 of
each other, every seed finding the same degenerate solution (emit zero through the pre-decision
epoch, which is most of the scored timesteps, then guess).

**This is a stronger claim than anything else in the paper.** Everywhere else `frm` recovers units
at a small r² cost; here it decides whether the task is learned at all. And `rws` ALONE is the worst
arm — worse than no penalty — while `frm+rws` is the fastest. That interaction (rws helps only in
the presence of frm) shows more cleanly here than anywhere else in the project.

Two consequences:

1. **16τ is a rescue cell, not a scaling cell.** A scaling series needs the baseline to solve at
   every size; building a four-size curve where the unpenalised arm fails most seeds would fit an
   exponent to failures. The N = 2000 and N = 4000 tasks were cancelled for this reason — 780
   GPU-hours, ~3 days of Spock — and will be re-queued at whatever delay the ladder names.
2. **Solving and active-unit count are coupled**, so conditioning on "solved" is a real selection
   effect: at N = 500 the solver had 170 active units against 123 and 113 for the two failures,
   +52 or ~45%. Any count from a solver-only subset must be reported with the solve rate beside it,
   or a rising curve cannot be told apart from a rising solve rate.

### Running now

**Spock, 33 jobs.**

- `DMTSv2` (6312325), 18 left of 48 — 16τ, N = 500 (frm/both still finishing) and N = 1000, all
  four penalty arms, 3 seeds, 150k. *Expect:* the N = 500 pattern to hold at N = 1000, i.e. none
  partial, rws worst, frm and both 3/3. If `none` solves 3/3 at N = 1000, the difficulty is
  size-dependent and the whole picture changes — that would be the informative surprise.
- `DMTS36v2` (6312326), 6 left of 12 — 36τ, N = 1000, `frm` and `both` (none and rws already
  failed 3/3). Due within 1–2 h. *Expect:* if either solves, the rescue claim holds at a delay the
  baseline provably cannot learn, which is the cleanest form of the result. Read-out is
  pre-registered in the launcher: BOTH matched-compute and best-checkpoint quoted for every arm,
  because the previous generation of this result was retracted for applying one read-out to the
  treatment and another to the control.
- `DMTSdelay` (6320918), 9 — the delay ladder, 8/10/12τ at N = 1000, pen=none, 150k. *Expect:* a
  solve rate rising as the delay shortens. Decision rule fixed in advance: the scaling series takes
  the SHORTEST delay reaching 3/3; if no rung reaches 3/3 the answer is more seeds, not a shorter
  delay. Read-out is the solve RATE, never a mean r² — averaging a bimodal 0.45/0.99 split
  describes no network that exists.

**Della, 72 jobs** — `mute` (14274294) vs `dead` (14274295) under the rebuilt sampler: independent
Bernoulli draws, water-filled probabilities, per-unit 1/(1−pᵢ) rescaling, β on rank. N = 1000
flip-flop, ρ ∈ {0.05, 0.10, 0.175, 0.25} × β ∈ {1, 2, 4} × 3 seeds, 40k, folder
`NBitFlipFlop_std_bernoulli`. Control is the completed bias-free no-dropout arm, 316.3 ± 33.3, bar
416. *Expect:* Pavel's hypothesis is that `dead` revives units more fairly because it pressures
recurrent rather than read-out redundancy — supported if `dead` > `mute` at a majority of the 12
cells. It was refuted-ish under the old sampler (pooled 373.0 ± 33.9 vs 357.4 ± 23.1, p = 0.27) but
that predates the rescaling, which is what `dead` needed most. Second, independent question: whether
rescaling removes the loss penalty that made ρ = 0.25 a trade rather than a win.

### Decisions waiting on these

- **Which delay the DMTS scaling series uses**, and whether DMTS stays in Figure 1c at all or
  becomes the rescue result instead. Waiting on the ladder.
- **Whether `dead` earns a place in the paper.** Waiting on Della.
- **Figure 2 still rests on legacy data** — old blind sampler, trainable bias, 150k — while the new
  sweep is 40k and bias-free. They answer different questions, so the figure either presents both
  or needs a fresh 150k `mute` series. Decide once the Della grid lands.
- **The −71-unit bias effect** (316.3 ± 33.3 bias-free against 387.0 ± 37.3 with a trainable bias,
  p = 0.049) is unverified and contradicts the +5.0 from the archived CDDM control. If it holds it
  belongs in Figure 1d, and it makes the DMTS tonic channel a bigger confound than represented.

