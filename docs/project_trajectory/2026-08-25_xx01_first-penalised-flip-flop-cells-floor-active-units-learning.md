## 2026-08-25 — First penalised flip-flop cells: floor, active units, learning dynamics across the grid

New script [`flipflop_grid.py`](../trainRNNbrain/experiments_and_analysis/flipflop_grid.py) reads both
sweeps into one (penalty, k, N) index and emits `flipflop_grid_{floor,active,dynamics}.png`.
Coverage at this reading: `none` k=1..8 (N=500), k=2..7 (N=1000, N=2000); `rws` k=1..4 (N=500),
k=1..5 (N=1000); `frm` k=1..2 (N=500) only; `both` nothing.

### Method: the two quantities need DIFFERENT range rules, and the data forced the split

Budgets are unequal — 500k (`none` N=500/1000), 400k (`none` N=2000), 150k (k=1 and the entire
penalty grid). Truncating every floor fit to the common 150k costs ≤1.9% on most cells but **4.0% on
k=5 N=2000**, where the truncated fit falls into the A/τ degeneracy and drags that cell's √k residual
to 3.46% against 0.59% for the same cell fitted over its own budget. Bounding τ to the fitted range
was tried as a fix and **rejected**: it binds on most cells and pushes `a` at N=2000 to 0.0218,
breaking the N-independence the untruncated fits show cleanly. So both fits are reported, and:

- cross-k / cross-N **within** a condition → full-budget fits (already near-matched at 500/500/400k)
- cross-**penalty** → matched 150k on both sides, the one comparison where ranges truly differ
- **active units → matched 150k always.** That count has not converged and never gets full budget.

### 1. Floor: the √k law survives at all three N, and `rws` does not move it

Full-budget fits of `floor_per_channel(k) = a + b·√k`:

| pen | N | nk | a | b | max resid |
|---|---|---|---|---|---|
| none | 500 | 8 | 0.02052 | 0.00302 | 0.60% |
| none | 1000 | 6 | 0.02041 | 0.00304 | 0.24% |
| none | 2000 | 6 | 0.02070 | 0.00290 | 0.59% |
| rws | 500 | 4 | 0.02054 | 0.00309 | 0.20% |
| rws | 1000 | 5 | 0.02077 | 0.00303 | 0.24% |

**N-independence now confirmed at three sizes**, not two: `a` spans 0.0204–0.0207 and `b` 0.0029–0.0031
across a 4× size range.

⚠️ **`rws` is the pre-registered "neither moves" outcome.** Against `none` read at matched 150k
(a = 0.02069 / 0.01999, b = 0.00286 / 0.00305), `rws` sits inside the seed scatter on **both**
parameters. rws does not reduce cross-channel interference and does not improve single-channel
accuracy. Per `penalty_vs_interference.md` this is the outcome that kills H for this arm — it is not
to be rescued with a different statistic. **frm and both still decide the actual test**, and frm has
only 4 nets so far.

⚠️ **`frm` moves the floor the WRONG way, preliminarily.** k=1: 0.02534 vs `none` 0.02340 (+8.3%);
k=2: 0.02731 vs 0.02487 (+9.8%). On CDDM at N=5000 frm *beat* unpenalised by 37%. Two cells, one of
them n=1, so this is a flag not a finding — but the direction is opposite and it needs the rest of
the frm row before anything is said about it.

### 2. Active units: sublinear, and the penalty does not change the exponent

`M ~ N^b`, scale-free criterion, all cells read at 150k:

| pen | mean b | per-k |
|---|---|---|
| none | **0.41 ± 0.05** | 0.40 / 0.47 / 0.45 / 0.42 / 0.35 / 0.34 (k=2..7) |
| rws | **0.40 ± 0.06** | 0.50 / 0.34 / 0.40 / 0.35 (k=1..4) |

⚠️ b = 0.41 here vs **0.37** logged 2026-08-24 — that was read at each cell's endpoint, this at a
matched 150k. Shallower reading gives higher b, exactly the documented monotone depth dependence. The
comparison against CDDM's 0.31–0.35 must be redone at matched depth before it means anything.

Two things worth following up:

- **`rws` alone INCREASES silencing.** At k=1 N=500 it gives M=122 vs `none`'s 165, and hard-criterion
  M collapses from 484 to 309. This does not contradict the frm-modulated-by-rws claim (that is about
  `both`, which has not landed), but rws on its own is not "purposing all units for useful compute".
- **At N=2000, M saturates in k past k=3**: 367 / 368 / 378 / 367 / 369 for k=3..7, while N=500 and
  N=1000 are still climbing. If it holds when k=8 N=2000 lands, the active-unit ceiling is set by N
  once the task is complex enough, not by task demand.

`frm` drives the silent fraction to **0.7–1.6%** — the pre-registered prerequisite that λ is not
mis-scaled on this task **passes**, so the frm floor result above will be interpretable.

### 3. Learning dynamics: A and τ are not identifiable; report t½, t₉₀, β

Several cells fit τ ≈ 100 against a fit starting at t = 2000. Below the fit window the stretched
exponential degenerates toward a power law and `A` becomes an extrapolated t=0 amplitude the data
never constrains, with A and τ trading off along that valley. **Ratios of the excess *within* the
fitted range are identified**, so the reported quantities are t½ and t₉₀ (iterations to remove half /
90% of the excess over L∞) plus β.

| pen | β | t½ | t₉₀ vs k |
|---|---|---|---|
| none | 0.3–0.5 | 5.0k → 6.6k (k=1→8) | **17.5k → 34.6k, roughly doubles** |
| rws | 0.5–0.7 | 5.2k–6.0k | 18.3k–20.6k, **flat in k** |

β well below 1 everywhere: a broad spectrum of timescales, not a single relaxation rate. Larger N
learns *faster* at fixed k. Unpenalised, added bits cost mostly in the long tail — t₉₀ doubles while
t½ moves only ~30%. **`rws` removes that k-dependence** (over k=1..4, where `none` still rises 27%)
and raises β, i.e. it narrows the timescale spectrum. That is the closest thing yet to a mechanistic
version of the "rws makes networks less heterogeneous" claim, but it rests on four k values.

---

