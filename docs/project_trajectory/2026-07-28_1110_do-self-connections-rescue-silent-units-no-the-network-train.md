## 2026-07-28 (11:10) — do self-connections rescue silent units? No — the network trains them into self-*inhibition*

### Why it was worth asking

Allowing a unit to connect to itself is the **cheapest conceivable escape from silence**: a positive
self-weight lets a unit sustain its own activity with no help from the rest of the network. It was
also the last non-standard constraint in the architecture — the `W_rec` diagonal had been zeroed at
init *and* re-zeroed after every optimizer step — so if it mattered, a reviewer could fairly say the
whole phenomenon was an artifact of forbidding self-recurrence. `self_connections: true` has been
part of the standard architecture since 2026-07-27; this entry reports what it did.

### Test 1 — cross-sweep, confounded but bounding

`CDDM_ptrack_g0_nodale_trainablebias` (self-connections **off**) vs `CDDM_std_g0` (**on**), both
unconstrained with a trainable bias, N=1000, 5 nets/cell:

| eq / penalty | hard: off → on | Δ | scale-free: off → on | Δ |
|---|---|---|---|---|
| h / none | 41.6% → 42.4% | +0.9 | 59.6% → 58.6% | −1.1 |
| h / rws | 41.3% → 39.9% | −1.3 | 64.5% → 63.9% | −0.6 |
| s / none | 39.8% → 40.0% | +0.2 | 59.7% → 59.1% | −0.7 |
| s / rws | 22.5% → 22.5% | 0.0 | 53.1% → 53.1% | 0.0 |
| h,s / frm, both | 0% → 0% | 0.0 | 0% → 0% | 0.0 |

Every delta is within ±1.3 pp. **Caveat, recorded when these sweeps were submitted:** the two differ
in *two* ways — `self_connections` and `task_safe_gradients` — so a difference could not have been
attributed. But two simultaneous changes producing nothing means neither mattered, barring an
implausible exact cancellation.

### Test 2 — within-network, confound-free, and the more informative one

Everything below is measured *inside* networks that all had `self_connections: true`, so no
cross-sweep comparison is involved. Standard RNNs, h, N=1000, 5 nets:

| unpenalized (`none`) | |
|---|---|
| mean self-weight `W_rec[i,i]`, **silent** units | **−0.0073** |
| mean self-weight, **active** units | **−0.0601** |
| corr(self-weight, log participation) | **−0.51** |
| mean \|self-weight\| vs mean \|off-diagonal\| | **4.7×** larger |

**The correlation is negative, and strong.** Active units have *more negative* self-weights than
silent ones. The network does not use self-connections to keep units alive — it trains them into
**self-inhibition**, and the most active units are the ones inhibiting themselves hardest.

That is mechanistically sensible. A positive self-weight is a gain>1 loop on a single unit — the same
instability that produced the NaN divergences in the frozen-clamp arc (2026-07-07). Gradient descent
finds self-*inhibition* useful as a per-unit gain control and self-excitation dangerous. **The
cheapest escape from silence is one the network actively declines to take.**

The diagonal is not being ignored: self-weights are **4.7× the typical off-diagonal magnitude**, so
they are genuinely used, just not in the direction that would keep units alive. Under `frm` there are
no silent units left to compare and the correlation flips to a weak **+0.12** — once every unit is
forced active, the self-weight stops predicting participation.

### Conclusion

Self-connections join the list of interventions that do not rescue (intervention #7 in `paper.md`
§3). Two things follow. The residual worry that the zeroed diagonal was a hidden non-standard
constraint doing real work is settled — it was not, and we now know *why* rather than merely *that*.
And the "a unit could just excite itself" objection has an empirical answer: it could, it is allowed
to, and it chooses the opposite.

