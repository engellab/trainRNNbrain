## 2026-08-25 (night, 6) — ⚠️ CUDA OOM at N>=4000: a scaling bug in participation tracking

**The timing calibration caught this before 48 jobs burned.** Submitted at N=3000/4000/5000 to size the
wall-time request; instead it exposed a hard failure:

| N | result |
|---|---|
| 3000 | ✅ completed, **0.795 s/iter measured** |
| 4000 | ❌ `torch.OutOfMemoryError` — tried to allocate 13.73 GiB with 37.29 GiB already resident |
| 5000 | ❌ `torch.OutOfMemoryError` — tried to allocate 11.45 GiB |

Both died inside `Trainer.participation_from_states_` at
`fr.std(dim=1) + torch.quantile(fr.abs(), 0.9, dim=1)`. `torch.quantile` SORTS its input, so it needs
~2x the tensor, and the old code also materialised the full activation `fr` alongside `states`. At
N=4000, T=300, batch 1024 that is states (4.6 GB) + fr (4.6) + |fr| (4.6) + sort (~9) ≈ 23 GB of
transient on a 44 GiB card already holding the graph.

**The N=4000 array (5904416, 48 jobs) was HELD before any element started** — `scontrol hold`, no
compute lost.

### Fix

`participation_from_states_` now processes the unit axis in chunks of 512, applying the activation
per chunk so a second full-size copy is never held. Peak transient falls from ~4x the state tensor to
state + ~1.2 GB, independent of N.

**Equivalence verified, and my first claim about it was too strong.** It is NOT bitwise identical in
float64: `std` is a reduction whose summation order depends on block shape, giving last-bit
differences of 4e-16 to 3e-15. The docstring was corrected to say so. What matters is the production
dtype: in **float32** the measured difference over 6000 units of a realistic bimodal population is
**exactly 0.0**, and **zero** units change silence class under the hard (1e-6), task-calibrated
absolute (4e-2), or scale-free rule.

### Cost model refitted on the MEASURED point

The old model (anchored only up to N=2000) over-predicted by ~35%:

| N | s/iter | 100k median | 100k slow node |
|---|---|---|---|
| 3000 | 0.82 (measured 0.795) | 23h | 31h |
| 4000 | 1.27 | 35h | 48h |
| 5000 | 1.86 | 52h | **70h — now fits Spock's 96h cap** |

⚠️ **This revises an earlier statement.** N=5000 was described as "Della only (97h slow)"; on measured
timings it is 70h slow and fits Spock. The 10x size lever is therefore available at no extra
infrastructure cost, should the ceiling test want it.

### Provenance gap, recorded rather than hidden

The FFpenL array (frm/both @400k) was submitted BEFORE the Trainer fix was synced, so elements that
started earlier ran the unchunked participation and later ones will pick up the chunked version.
Since the two are numerically identical in float32 (above), this does not affect any result, but the
split is recorded and `CODE_VERSION.txt` on Spock now reads
`95320a6+trainer-chunked-participation-2026-08-25` so the two code states are distinguishable.
N ≤ 2000 never hit the OOM path (states at N=2000 is 2.5 GB, old peak ~10 GB on a 44 GiB card).

### Resolution: N=4000 verified and released; N=5000 blocked by a DIFFERENT limit

Re-calibration after the chunking fix:

| N | result |
|---|---|
| 4000 | ✅ **no OOM**, 400 iterations in 8:15 = **1.24 s/iter** (refit predicted 1.27) |
| 5000 | ❌ still OOM — but now in `RNN_torch.forward`, NOT in participation tracking |

**The two failures are different problems.** The participation OOM is fixed. N=5000's remaining
failure is the BPTT graph itself: 300 timesteps x 5000 units x batch 1024 of stored activations,
which no change to the read-out path can help. Resolving it needs a smaller batch (changes the
training regime and breaks comparability), gradient checkpointing (not implemented), or a larger
card. Note also that this run landed on a 39.52 GiB A100 rather than the 44.39 GiB device the N=4000
job got, so the exact N at which it breaks is partly node-dependent.

⚠️ **Therefore the earlier revision "N=5000 now fits Spock at 70h" is itself WITHDRAWN.** Wall time
was never the binding constraint at N=5000; GPU memory is. N=4000 is the largest size that runs with
the current training code and batch size.

**Array 5904416 (N=4000, 48 jobs) RELEASED** after verification. At the measured 1.24 s/iter, 100k
iterations is 34h median and 47h on a 1.35x slow node, against the 96h request — 2x margin.

**Queue as released:** 48 FFbigN (N=4000), 22 FFpenL (frm/both @400k, 19 running), 32 FFpen
(rws @150k + remaining frm @150k), 1 FFk.

---

