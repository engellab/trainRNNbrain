## 2026-08-21 16:05 — the flip-flop k-effect is strongly depth-dependent; the shallow reading overstated it

> # ⛔ RETRACTED — trained with `same_batch=True`
> **Every number in this entry is void.** These flip-flop networks reused ONE batch of 256
> trials for all 300k iterations, so they measure memorisation, not the task. See the
> `same_batch=True` entry below for the full account. The data is quarantined in
> `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/` and the figures were deleted.
> Kept here only so the reasoning trail is not falsified by deletion.


Prompted by asking whether L*=0.010 was deep enough at N=2000. It was not.

**Deepest reachable level.** The worst N=2000 seed's smoothed endpoint loss is 0.00550, so that is
the deepest level every seed reaches. Iterations needed at N=2000:

| L* | R² | k=2 | k=3 | k=4 | k=5 | k=6 |
|---|---|---|---|---|---|---|
| 0.022 | 0.970 | 139k | 81k | 34k | 26k | 13k |
| 0.015 | 0.980 | 161k | 122k | 47k | 47k | 43k |
| 0.010 | 0.986 | 226k | 213k | 104k | 61k | 133k |
| 0.007 | 0.990 | 276k | 252k | 201k | 156k | 172k |
| 0.0055 | 0.993 | 297k | 293k | 267k | 256k | 259k |

L*=0.0055 consumes 256k–297k of the 300k budget, so it is the resolution floor of this sweep rather
than a chosen stopping point.

**The effect shrinks by 4.7x as the reading deepens.** Scale-free silent %, N=2000:

| L* | k=2 | k=3 | k=4 | k=5 | k=6 | k=2/k=6 |
|---|---|---|---|---|---|---|
| 0.022 | 67.5 | 46.6 | 21.5 | 18.2 | **8.0** | 8.4x |
| 0.010 | 61.1 | 49.9 | 24.3 | 16.6 | **22.1** | 2.8x |
| 0.0055 | 62.6 | 53.7 | 42.8 | 40.6 | **34.8** | 1.8x |

The hard criterion loses the ordering entirely at the deepest level: 16.4, 9.3, 17.6, 23.2, 18.2 —
non-monotone, no trend.

**Correction to the entry above.** The headline "8.0% scale-free silent at k=6" was a shallow-reading
artifact. At the deepest reachable level k=6 still leaves **34.8%** of units functionally silent.
What survives every depth is only the DIRECTION — scale-free silencing decreases with k — and the
k=2 end barely moves at all (67.5 → 62.6), so the whole effect is the high-k end rising toward the
low-k end as training deepens.

Mechanism is the same depth confound already documented: deeper levels require the high-k cells to
train 10–20x longer (13k → 259k iterations at k=6), and silencing grows with training depth, so the
cells that looked cleanest at a shallow level have simply not silenced yet when read there.

**What this means for the claim.** "Harder tasks do not silence" is not supportable. The supportable
claim is weaker and needs the depth caveat stated with it: at matched performance, silencing
decreases with task complexity, but the size of that decrease depends strongly on how deep the
networks are read, and even the hardest task tested leaves a third of the population functionally
silent. Script defaults now include L*=0.0055 so the deep reading is always plotted.

---

