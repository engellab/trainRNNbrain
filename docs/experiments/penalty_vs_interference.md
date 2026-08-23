# Does the activity penalty reduce cross-channel interference? — pre-registered plan

Written 2026-08-23 16:30, **before any penalised flip-flop network has been trained**. Predictions
and thresholds fixed in advance.

---

## 1. Why this matters more than the other open items

`frm` and `rws` are currently justified on two grounds: they remove silent units (tidiness), and they
should make networks easier to reverse-engineer (identifiability, untested). Both are arguments about
the *population*, and a referee can reasonably answer "so what — the network performs the same".

This asks whether the penalty **improves the task itself** on a multi-component task. If it does, the
recommendation stops being about tidiness and becomes about performance, which is far harder to
dismiss.

There is already a hint. At N=5000 on CDDM, `frm` beats unpenalised by **37% on clean task loss**
(0.00533 vs 0.00846) and `both` by 31%, and the advantage grows monotonically with N (−15%, +3%,
+14%, +37% at N = 500/1000/2000/5000). Two readings of that: either it is a CDDM quirk, or the
penalty reduces some form of interference that gets worse as networks get larger and sparser. The
flip-flop can distinguish them, because it has an explicit interference axis — the bit count k.

---

## 2. The measurement this builds on ✅

Measured 2026-08-23 on the fresh-batch sweep (k = 2..6, N ∈ {500, 1000}, 3 seeds):

**The per-channel loss floor rises with k and is exactly N-independent.**

| k | floor, N=500 | floor, N=1000 |
|---|---|---|
| 2 | 0.02493 | 0.02493 |
| 3 | 0.02589 | 0.02588 |
| 4 | 0.02667 | 0.02669 |
| 5 | 0.02738 | 0.02735 |
| 6 | 0.02799 | — |

Four significant figures across a 2x size change. Fitting the TOTAL floor as
`T(k) = k·f1 + g·[C(k,2)]^p`:

| model | ΔAICc | max resid |
|---|---|---|
| free exponent, **p = 0.808 ± 0.010** | 0.0 | 0.15% |
| `k·a + b·k^1.5` | +0.6 | 0.18% |
| `p = 1` — pure C(k,2), all-to-all pairwise | +66.8 | 0.85% |
| no interaction term at all | +188.1 | 8.19% |

So an interaction term is **required** (dropping it costs ΔAICc +188), but it grows as ≈ k^1.6, not
k². p = 1 is 19σ away and the exponent is stable when fitted on k=2..5 (0.806) or k=3..6 (0.800).

**The N-independence is the load-bearing part.** It rules out every capacity explanation: "each bit
gets N/k units, error ~ 1/√(N/k)" predicts the floor scales as √(k/N), and it does not. At N=500 and
N=2000 the network has 4x the units for the same job and reaches the same floor. Whatever limits
per-bit accuracy is **not the number of units** — most plausibly the single shared time constant
through which every bit must be read.

⬜ k=1 is running (tasks 64-66) as an out-of-sample test: C(k,2) predicts a per-channel floor of
0.02440 there, the k^1.5 form predicts 0.02373, a gap ~7x the residual scale, and neither model has
seen the point.

---

## 3. The claim under test

**H:** activity regularisation reduces cross-channel interference, i.e. it lowers `g` (the
interference amplitude) and/or `p` (its growth with k) relative to unpenalised networks.

**The falsifier, stated first.** If the fitted `g` and `p` for `frm` and `both` overlap the
unpenalised values within their bootstrap CIs, H is **wrong**, the CDDM N=5000 advantage is not
interference reduction, and this line is dropped rather than rescued with a different statistic.

---

## 4. Design

`none` at k ∈ {2,4,6}, N=500 **already exists** in the fresh-batch sweep, so only the penalised cells
are needed:

- k ∈ {2, 4, 6} × penalty ∈ {`frm`, `both`} × 3 seeds = **18 jobs**
- N = 500 only. This is the economy the N-independence buys: the floor is a task property, so the
  cheapest size measures it as well as the most expensive.
- 150k iterations. Floors converge by ~25k (measured: loss flat from 0.0326 at 50k to 0.0320 at
  400k), so this is 6x margin. ~10 h/job at 0.24 s/iter → **~180 GPU-h**, about a day.
- ⚠️ The existing `none` cells ran 500k. Immaterial for floors given convergence by 25k, but it must
  be stated, and the `none` floor should be re-read at 150k as a check that it has not moved.
- λ_frm = 0.1, λ_rws = 0.05, as everywhere else. ⚠️ These were tuned on CDDM, whose task loss is
  ~0.006-0.009 against the flip-flop's ~0.025 — so the penalty is relatively **weaker** here. If the
  penalty fails to zero the silent fraction, that is a mis-scaled λ and not evidence about H; check
  the silent fraction first and re-tune λ before interpreting any floor result.

---

## 5. Pre-registered predictions

| quantity | prediction under H | threshold for support |
|---|---|---|
| `g` (interference amplitude) | penalised < none | non-overlapping bootstrap 95% CIs |
| `p` (growth exponent) | penalised ≤ none | directional; a drop below 0.79 would be decisive |
| `f1` (single-bit floor) | unchanged | H is about interference, not about single-channel accuracy — if `f1` moves too, the penalty is changing something more basic and the interference reading is confounded |
| silent fraction | ~0 for `frm`/`both` | prerequisite; if not, λ is mis-scaled (see above) |

The `f1` row is the important control. A penalty that lowers the whole curve uniformly is not
reducing *interference*; it is improving the task generally. Only a change concentrated in the
k-dependent term supports H.

---

## 6. Free precursors, to run first

Both use networks already on disk and need one forward pass each. They discriminate the mechanism,
which decides whether the penalty result would even be interpretable.

**(a) Effective bits per unit vs k.** For each active unit, regress its rate on the k target bits and
compute the participation ratio of the squared coefficients:

    PR_i = (Σ_j β_ij²)² / Σ_j β_ij⁴        1 = pure single-bit, m = m bits equally

⚠️ Use PR, **not** the `max|β|/‖β‖` concentration used in `flipflop_selectivity.py`. That ratio has a
lower bound of 1/√k, so it drifts upward with k for purely geometric reasons and would manufacture
the very trend being tested. PR has no such k-dependent floor.

Two mechanism families are distinguished by the result:
- **PR rises with k** → per-pair interference weakens as k grows; the network orthogonalises harder
  when it has more to keep apart.
- **PR flat** → fewer effective pairs; bits sit in near-disjoint sub-populations and only overlapping
  ones interfere.

Limitation to state: PR sees only **linear** mixing. A unit encoding XOR of two bits has β ≈ 0 on
both and scores as unselective. Interaction terms in the regression would catch it.

**(b) The pair-interference matrix.** Measure how error on channel i responds to pulse events on
channel j, giving ε²_ij directly. This tests C(k,2)'s assumptions rather than inferring them from the
shape of a 5-point curve: is interference all-to-all (assumption 1), and equal across pairs
(assumption 2)?

---

## 7. What each outcome would mean

- **H supported, `f1` unchanged** → the penalty reduces cross-task interference. This is a
  performance argument for activity regularisation on multi-component tasks and belongs in the main
  text, not the discussion.
- **H supported but `f1` also drops** → the penalty helps generally; interesting, but it is not an
  interference story and should not be told as one.
- **H falsified** → the CDDM N=5000 advantage is something else. Drop this line; do not go looking
  for a statistic that rescues it.

⚠️ **Scope warning.** This is a new axis, and the floor-scaling thread it grew out of has already
consumed two days without moving the paper closer to submission. It is worth running **only** if
"the penalty reduces interference" is going to be a paper claim. The free precursors in §6 cost
nothing and can be run regardless; the 18-job grid should wait on that decision.
