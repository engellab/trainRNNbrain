## 2026-08-23 16:31 — the flip-flop loss floor: interference grows as k^1.6, and is N-independent

Prompted by asking whether the total floor follows `k·f1 + C(k,2)·g`. It nearly does, and the way it
fails is the informative part.

**Fits to the TOTAL floor** (k=2..6, N ∈ {500,1000}, 3 seeds, fresh-batch sweep):

| model | ΔAICc | max resid | params |
|---|---|---|---|
| `k·f1 + g·[C(k,2)]^p`, p free | 0.0 | 0.15% | **p = 0.808 ± 0.010** |
| `k·a + b·k^1.5` | +0.6 | 0.18% | a=0.02078, b=0.00295 |
| `k·f1 + C(k,2)·g`  (p = 1) | +66.8 | 0.85% | |
| `k·f1 + (k-1)·g` | +119.8 | 1.83% | |
| `k·f1` (no interaction) | +188.1 | 8.19% | |

So an interaction term is **required** — dropping it costs ΔAICc +188 and an 8% residual — but
C(k,2) grows too fast. Its residuals are systematically U-shaped (+0.00021, −0.00001, −0.00008,
−0.00002, +0.00009 at k=2..6), the signature of over-convexity. p = 1 is **19σ** from the fit, the
bootstrap CI is [0.793, 0.823] with zero mass above 1.0, and the exponent is stable when fitted on
k=2..5 (0.806) or k=3..6 (0.800).

**What C(k,2) assumes, and which assumption breaks.** Squared error is additive in variance, so
`k·f1 + C(k,2)·ε²` asserts that (1) every pair of bits interferes, (2) each pair contributes the same
ε², (3) contributions are independent. k^1.6 means at least one is false — either fewer effective
pairs than C(k,2), or per-pair strength falling as k^-0.4.

**The N-independence is the load-bearing result, not the exponent.** The floor agrees to FOUR
significant figures between N=500 and N=1000 at every k. That rules out the whole capacity family:
"each bit gets N/k units, error ~ 1/√(N/k)" predicts a floor scaling as √(k/N), and there is no N
dependence at all. At N=500 and N=2000 the network has 4x the units for the same job and hits the
same floor — **whatever limits per-bit accuracy is not the number of units.** Most plausibly it is
the single shared time constant every bit is read out through, which is a fixed parameter rather than
a resource.

Practical form: **you cannot buy multi-task capacity by enlarging the network.** Same shape of claim
as "active units saturate", reached by a completely independent measurement.

**Consistent with the selectivity result.** C(k,2) is what all-to-all mixing predicts, but units are
near-purely single-bit (concentration 0.967 at k=3) — the network separates the bits, and a modular
solution should interfere sub-quadratically.

⬜ **k=1 submitted** (Spock `5722788`, tasks 64-66, N=500, 150k iterations) as an out-of-sample test.
The model was specified as "f1 estimated from k=1", which we did not have, so f1 had to be fitted and
both survivors then have two free parameters. At k=1 they diverge: C(k,2) predicts a per-channel
floor of 0.02440, k^1.5 predicts 0.02373, a gap ~7x the residual scale, and neither has seen the
point. k=1 was appended to the END of the launcher's KS array so every already-submitted task index
keeps its meaning.

⚠️ **Caveat on the exponent.** Five points and two parameters cannot identify a mechanism. k^1.5 and
k^1.62 fit equally well and imply different per-channel exponents (0.50 vs 0.62); over k=2..6 they
are not separable. k=7 and k=8 are running and will extend the range.

**Follow-up pre-registered**: [`experiments/penalty_vs_interference.md`](experiments/penalty_vs_interference.md)
— does activity regularisation reduce the interference term? If it lowers `g` or `p` while leaving
`f1` alone, the penalty gains a *performance* argument on multi-component tasks rather than only a
tidiness one. 18 jobs, ~180 GPU-h, and `none` already exists. Two free precursors listed there run
on networks already on disk.

---

