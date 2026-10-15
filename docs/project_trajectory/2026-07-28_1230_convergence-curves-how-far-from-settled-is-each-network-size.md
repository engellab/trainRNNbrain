## 2026-07-28 (12:30) — convergence curves: how far from settled is each network size?

Direct answer to "does the participation stop changing?", computed from the participation traces of
the standard-RNN sweeps (h equation, **no penalties**, 5 nets per size, 30000 iterations). Two
measures over a trailing **1000-iteration window** (100 snapshots at `track_every=10`), bands are
95% t-intervals across the 5 networks.
[`plot_convergence_curves.py`](../trainRNNbrain/experiments_and_analysis/plot_convergence_curves.py).

![Convergence curves](../img/internal_figures/convergence_curves.png)

| N | new silent units / 1000 iters at the end | `‖Δp‖/‖p‖` at the end | decay exponent | extrapolated iterations to reach 1% |
|---|---|---|---|---|
| 100 | **0.05 ± 0.26** | 0.045 | −0.29 | ~5,600,000 |
| 500 | **3.30 ± 3.52** | 0.054 | −0.49 | ~871,000 |
| 1000 | **9.57 ± 7.90** | 0.055 | −0.57 | ~562,000 |

### What the two panels say

**Left — the silent population.** N=100 has genuinely stopped (0.05 ± 0.26 new silent units per 1000
iterations; the interval contains zero). N=500 is marginal (3.3 ± 3.5, interval contains zero).
**N=1000 is still gaining ~10 units per 1000 iterations at the end, with the interval clear of
zero.** The curve is *decelerating* — it peaks near ~27 per 1000 iterations around iteration 11000
and falls to ~10 by 30000 — but it has not reached zero, and larger networks are further from it.
Note the early negative excursion: in the first ~1500 iterations the silent count *drops* sharply,
the tail of the global collapse-and-recovery, before the steady silencing begins.

**Right — the participation vector.** The stricter test, and it is unambiguous: **no size has
converged.** At iteration 30000 the participation vector still changes by **4.5–5.5% per 1000
iterations**, at every N. The silent *count* can look flat while units continue to rearrange, which
is exactly what N=100 is doing — 0.05 new silent units per 1000 iterations, yet a 4.5% relative
change in participation.

### The uncomfortable implication

The relative change decays as a power law in iteration (exponent −0.29 to −0.57), not exponentially.
Extrapolating to a 1% criterion gives **0.5–5.6 million iterations** — 20× to 190× the current
horizon, i.e. weeks to months of GPU time per network. **On any practical horizon these networks
never fully stop changing.**

So "train to convergence" is not an available option, and the running 100000-iteration horizon
experiment will not deliver a converged state either — it will show how much further things move
with 3.3× the training. The honest resolutions are:

1. **Define convergence by a stated threshold** (e.g. "< 1 new silent unit per 1000 iterations", or
   "‖Δp‖/‖p‖ < 2% per 1000") and report the horizon required to reach it at each N — accepting that
   the horizon itself grows with N.
2. **Report every number as "at iteration X"**, and demonstrate that the *comparison* between
   conditions is stable across X even though the absolute values are not. This is the cheaper and
   probably more honest route: `frm` gives 0% silent from early on, so the `frm`-vs-`none` contrast
   is not in question — only the precise baseline value is.
3. For the size sweep specifically, **match training by drift rather than by iteration count**, since
   `lr ∝ N^(−1/3)` means a fixed iteration count buys less progress at larger N.

The saturation question (`paper.md` §6.2) is the one genuinely at risk, because it compares
*absolute* active-unit counts across N, and the residual drift is precisely the term that grows with
N. Option 2 does not rescue it; that comparison needs option 1 or 3.

