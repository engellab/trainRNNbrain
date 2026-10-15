## 2026-08-21 15:57 — Flip-flop floor fits: the floor is NOT identified at 300k, so the k-comparison is void

> # ⛔ RETRACTED — trained with `same_batch=True`
> **Every number in this entry is void.** These flip-flop networks reused ONE batch of 256
> trials for all 300k iterations, so they measure memorisation, not the task. See the
> `same_batch=True` entry below for the full account. The data is quarantined in
> `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/` and the figures were deleted.
> Kept here only so the reasoning trail is not falsified by deletion.


New script: `experiments_and_analysis/flipflop_floor.py` → `img/internal_figures/flipflop_floor.png`.
Stretched exponential `L(t) = L_inf + A exp(-(t/tau)^beta)` fitted in log space on log-binned medians,
from iteration 2000, **identical range for every cell** (all 45 runs share the 300k budget).

The training loss is a masked MSE averaged over all axes including channels, so the fitted `L_inf` IS
the per-channel floor — no normalisation needed, and comparable across k because the target variance
is k-independent (0.727–0.738).

### Two methodological corrections were needed before the answer meant anything

**1. The nested-model test was initially wrong.** The first version compared "one floor for
everything" against "one floor per SEED", which conflates the question asked (does the floor depend
on k?) with pure seed scatter — and seed scatter is large here (one k=3, N=1000 cell spans 0.0025 to
0.0110). Fixed by adding the missing middle model, **one floor per k shared across its seeds**, and
testing `one → per_k` for the k-question and `per_k → per_seed` for scatter separately.

**2. The pre-registered budget check fails.** Refitting on the first 150k and comparing:

| N | max shift in `L_inf` when the fit range is halved |
|---|---|
| 500 | **83%** |
| 1000 | **34%** |
| 2000 | **28%** (but only 5–8% for k=3,4,5) |

The 300k budget — chosen precisely so this test could be run — **was not enough**. Where the shift is
large the fit is tracking the last data point rather than an asymptote, which is visible directly in
panel (b): at N=500/1000 the floor-subtracted curves collapse to the clip at late times, meaning
`L(t) − L_inf ≤ 0`, i.e. the fitted floor sits at the data.

### What can and cannot be said

- ❌ **N=500 and N=1000: nothing.** The nominal verdicts (floors differ, p=0.0009 and p=8e-9) are
  **void** — they are differences between unidentified asymptotes. This is the same failure mode that
  manufactured a spurious decay exponent three times in the CDDM analysis, caught this time by the
  check built for it.
- 🟡 **N=2000, k=3/4/5: the floor is identified** (shifts 5–8%) and sits at **0.0032–0.0035**, with
  the shared-floor test unable to distinguish them (p=0.6, ΔAICc favours a single floor by 6.0) and
  no significant seed scatter (p=0.14). k=6 is borderline (18%); k=2 is unreliable (28%, and only 2
  seeds since one diverged, one of which is an outlier at 0.00125).

So the tentative reading is that **at N=2000 the network is no longer capacity-limited and all
complexities converge to a common, task-imposed per-channel floor near 0.0033** — consistent with the
prediction written into the launcher, that the dominant irreducible error is the tau=10 low-pass
failing to track instantaneous switches, which is k-independent because switches-per-channel is
k-independent (2.25). At N=500/1000 the loss is still descending at 300k and no floor exists to
compare.

### Does this damage the main result?

**No.** The silencing result does not depend on the floor. `flipflop_ksweep.py` deliberately reports
the k-ordering at the endpoint AND at three matched levels, requiring agreement — precisely so that
the conclusion survives not knowing where the floor is. It does: scale-free silencing falls
monotonically with k under every reading.

What the floor result does affect is whether a single universal L\* is *principled* across k. At
N=2000 it appears to be; at smaller N it is unestablished. The multi-level reporting already in place
is the correct fallback either way.

⬜ To settle it would need substantially longer runs (600k+) at N=500/1000. Not currently justified —
the main claim does not rest on it.

---

