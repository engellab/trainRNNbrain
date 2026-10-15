## 2026-08-27 — Per-penalty analysis: floor, W_inp diffusion, M(N,k)

New script [`flipflop_penalties.py`](../trainRNNbrain/experiments_and_analysis/flipflop_penalties.py).
Coverage: none 64 runs (k=1..8, N=500/1000/2000), rws 43 (k=1..8, N=500/1000), frm 12 (k=1..3,
N=500/1000), **both: still no data**. Budgets differ (none 400-500k, rws 150k, frm 400k), so every
trace is truncated to a common 150k before any fit or read-out.

⚠️ **The N=4000 ceiling test has NO usable data.** The "2 completed nets" reported this morning were
the 400- and 600-iteration TIMING CALIBRATION runs, which write a full trace and are
indistinguishable from an experiment unless the trace LENGTH is checked. `load()` now drops anything
shorter than 50k and counts it. Real N=4000 jobs are at 9k-17k of 100k.

### 1. Floor law — rws does NOT reduce interference

| pen | N | a (single-channel) | b (interference) |
|---|---|---|---|
| none | 500 | 0.02069 [0.02042, 0.02123] | 0.00286 [0.00262, 0.00302] |
| none | 1000 | 0.02029 | 0.00289 |
| none | 2000 | 0.02065 | 0.00276 |
| **rws** | 500 | 0.02105 [0.02061, 0.02200] | **0.00276 [0.00236, 0.00303]** |
| **rws** | 1000 | 0.02115 | **0.00277** |

**b overlaps completely with unpenalised at both sizes.** The pre-registered interference hypothesis
(docs/experiments/penalty_vs_interference.md) predicted rws would LOWER b; it does not. `a` is if
anything slightly higher. **H is not supported for rws** — consistent with the earlier reading, now
on the full k=1..8 grid at two sizes.

⚠️ frm's floor is NOT reportable: it needs ~400k to converge, so a fit truncated at 150k is fitted to
a still-descending curve — the exact defect that forced the earlier retraction. frm floors can only
be compared against `none` at 400k, which excludes rws (150k data). Range-matching and convergence
cannot both be satisfied across all three conditions with the budgets currently on disk.

### 2. W_inp diffusion (range-matched at 150k)

| pen | settled | median onset | α at the cap |
|---|---|---|---|
| none | 60/64 | 60k | 0.50 |
| **rws** | 33/43 | **53k** | **0.43** |
| **frm** | **0/12** | NEVER | **0.87** |

rws settles the input weights faster and further than no penalty at all; frm prevents settling
entirely. Unchanged from yesterday, now on 12 frm runs including N=1000.

### 3. ★ NEW RESULT — rws makes active units DEPEND on task complexity

At MATCHED DIFFUSION (fixed compute is confounded; frm cannot be read this way at all). rws lacks
N=2000, so `none` is refitted on the same two sizes as a control:

| condition | n | b (size) | c (complexity) |
|---|---|---|---|
| none, all sizes | 60 | 0.441 [0.404, 0.478] | −0.030 [−0.083, +0.019] |
| **none, N=500/1000 (control)** | 41 | **0.493 [0.428, 0.561]** | **−0.019 [−0.077, +0.037]** |
| **rws, N=500/1000** | 33 | **0.490 [0.403, 0.585]** | **+0.136 [+0.092, +0.173]** |

**Same sizes, same criterion, same truncation: b is identical (0.493 vs 0.490), c is not.** The CIs
on c are cleanly separated — unpenalised straddles zero, rws excludes it. Going k=1 → 8 recruits
8^0.136 = **+32%** more units under rws, and nothing under none.

**Reading, offered as interpretation not measurement:** unpenalised networks recruit a
task-demand-INDEPENDENT number of units — the pathology. Under rws the count DOES scale with demand.
That is rws partially repairing the pathology, and it is a performance-independent argument for the
penalty of exactly the kind the programme was looking for. ⚠️ It rests on 33 rws runs at two sizes;
it needs N=2000 rws (20 nets already on Spock, not yet analysed here) and ideally `both` before it is
load-bearing.

### 4. ⚠️ frm's exponent is a CEILING EFFECT, not a scaling law

| pen | N | mean M/N |
|---|---|---|
| none | 500 / 1000 / 2000 | 0.409 / 0.283 / 0.183 |
| rws | 500 / 1000 | 0.384 / 0.231 |
| **frm** | **500 / 1000** | **0.993 / 0.949** |

frm drives M/N to 0.95-0.99 — it abolishes silence by construction, which is what it is for. Its
fitted b ≈ 0.91 is therefore a ceiling at M = N, not a measured exponent, and must not be set beside
none/rws's b as though it measured the same thing.

### ★ THE PRE-REGISTERED INTERFERENCE TEST RESOLVES — for frm, in the OPPOSITE direction

frm now has k=1..7 at N=500 (21 nets at 400k), enough to fit the floor law where frm is CONVERGED.
This is the comparison flagged as impossible at a 150k truncation. Both conditions fitted on
[2000, 400 000]; restricted to the common k range 2..7 because `none` covers k=2..8 and frm k=1..7.

| | a (single-channel floor) | b (interference amplitude) |
|---|---|---|
| none | 0.02068 [0.02051, 0.02087] | 0.00296 [0.00288, 0.00303] |
| **frm** | 0.02050 [0.01926, 0.02289] | **0.00460 [0.00360, 0.00532]** |
| change | **−0.9%, CIs overlap** | **+55.7%, CIs separated** |

Raw floors show the gap WIDENING with k, which is what b measures:
k=2: none 0.02488 vs frm 0.02677 (+7.6%) → k=7: none 0.02852 vs frm 0.03245 (+13.8%).

**frm INCREASES cross-channel interference by ~56%, leaving single-channel accuracy untouched.**

The pre-registration (docs/experiments/penalty_vs_interference.md) fixed three outcomes: *b falls with
a unchanged* → penalty reduces interference; *a falls with b unchanged* → not an interference story;
*neither moves* → hypothesis dead. The observed pattern is the FIRST structure — the effect really is
on the interference axis and not on single-channel accuracy — but with the sign reversed. **H is not
merely unsupported; the opposite holds.**

### Controls run before believing it

1. **k-range matched.** The uncontrolled fit (none k=2..8 vs frm k=1..7) gave +57.6%; restricted to
   the common k=2..7 it gives +55.7%. Not a range artifact.
2. **Residual convergence is not k-dependent.** frm's loss still falls 1.1–5.2% over the final third
   (none: ~0%), so frm floors are biased somewhat high — but the bias does NOT trend with k
   (slope +0.04% per bit, flat). A uniform bias inflates `a`, not `b`, and `a` did not move. The
   k=7 frm cell is the most converged (−0.4%) and shows the LARGEST gap, so the widening is not
   non-convergence.

⚠️ Caveats: N=500 only; frm's floors carry a residual high bias of a few percent from incomplete
convergence, so +56% is an estimate rather than a precise figure; and `both` — the condition that
would say whether rws rescues this — still has no data.

⚠️ Note this is the OPPOSITE sign from the rws result in the same section: rws leaves b unchanged
(0.00276 vs 0.00286), frm raises it 56%. The two penalties do different things to the floor as well
as to the weight dynamics.

### rws grid completed (N=2000 synced) — the k-dependence SHRINKS but survives

rws now has the full grid: 72 nets, k=1..8 at N=500/1000/2000. Refitting M = A N^b k^c at matched
W_inp diffusion, with `none` on the same sizes:

| condition | n | b (size) | c (complexity) |
|---|---|---|---|
| none, 3 sizes | 60 | 0.441 [0.404, 0.478] | −0.030 [−0.083, +0.019] |
| **rws, 3 sizes** | 58 | **0.440 [0.392, 0.486]** | **+0.083 [+0.044, +0.120]** |
| none, 2 sizes (previous) | 41 | 0.493 | −0.019 [−0.077, +0.037] |
| rws, 2 sizes (previous) | 38 | 0.466 | +0.124 [+0.084, +0.156] |

⚠️ **Adding N=2000 nearly HALVED the effect: c = +0.136 → +0.083.** The two-size estimate overstated
it, and the k=1→8 recruitment gain drops from +32% to **+19%**. The finding survives — the CI still
excludes zero and `none` still straddles it, with b identical between conditions (0.441 vs 0.440) —
but the magnitude reported yesterday was inflated by the narrow size range. This is why the N=2000
cells were flagged as needed before the result could be load-bearing.

### Floor law, full grid

| pen | N | a | b |
|---|---|---|---|
| none | 500/1000/2000 | 0.02069 / 0.02029 / 0.02065 | 0.00286 / 0.00289 / 0.00276 |
| **rws** | 500/1000/2000 | 0.02105 / 0.02126 / 0.02138 | **0.00276 / 0.00270 / 0.00261** |
| frm | 500 | 0.02103 | **0.00440** |

rws's b is consistently a shade below none's at every size (0.00276/0.00270/0.00261 vs
0.00286/0.00289/0.00276) but every CI overlaps, so this is not a resolved reduction — at best a hint
worth revisiting when `both` lands. frm's +50% remains the only change that clears its CI.

### W_inp diffusion, full grid at the 150k truncation

| pen | settled | median onset | α at cap |
|---|---|---|---|
| none | 60/64 | 60k | 0.50 |
| rws | 58/72 | 64k | 0.47 |
| frm | 1/27 | (113k, single run) | 0.79 |

⚠️ **Correction to yesterday's "rws settles FASTER than none".** On the full grid rws's median onset
is 64k against none's 60k — slightly SLOWER, not faster. The earlier 53k came from the two-size
subset. What survives is the α at the cap (0.47 vs 0.50, rws marginally lower) and, unambiguously,
the contrast with frm, which does not settle at all (1/27, α 0.79).

---

