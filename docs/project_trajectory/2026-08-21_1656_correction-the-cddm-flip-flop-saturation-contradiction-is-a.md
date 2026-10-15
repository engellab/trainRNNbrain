## 2026-08-21 16:56 — CORRECTION: the CDDM/flip-flop saturation "contradiction" is a reading-depth artifact

> # ⛔ RETRACTED — trained with `same_batch=True`
> **Every number in this entry is void.** These flip-flop networks reused ONE batch of 256
> trials for all 300k iterations, so they measure memorisation, not the task. See the
> `same_batch=True` entry below for the full account. The data is quarantined in
> `data/trained_RNNs/RETRACTED_samebatch_NBitFlipFlop_ksweep/` and the figures were deleted.
> Kept here only so the reasoning trail is not falsified by deletion.


Pavel spotted that M saturates with N on CDDM (b = 0.31) but grows linearly on the flip-flop
(b ~ 1.0), and asked whether that is a contradiction. It is not, and my framing in the previous two
entries was wrong.

**b depends on how deep the reading is, relative to that task's own floor:**

| flip-flop L* | excess over floor | b (N=1000 -> 2000) |
|---|---|---|
| 0.0375 | 10.4x | 1.11 |
| 0.0220 | 5.7x | 1.04 |
| 0.0150 | 3.5x | 0.95 |
| 0.0100 | 2.0x | 0.87 |
| 0.0070 | 1.1x | 0.58 |
| **CDDM 0.023** | **0.08x** | **0.31-0.35** |

b falls monotonically toward the floor and is still falling at the deepest level 300k can reach. The
flip-flop cannot be read closer than ~1.1x above its floor; CDDM was read at 0.08x — about 14x deeper
in relative terms. Same curve, two sampling locations.

**Two claims retracted:**

- "M ~ N k^0.85" as a task law. b is not a stable property; b ~ 1.0 is the SHALLOW value. Any quoted
  exponent must carry the relative depth it was measured at.
- "The CDDM saturation is task-specific and does not hold on the flip-flop." Wrong — it is
  depth-specific. I made a cross-task claim without checking relative depth, which is the same
  confound that has already invalidated the endpoint k-ordering, the shallow k-effect size, and the
  contour map's diagonal structure. Fourth occurrence.

**Standing rule going forward:** no cross-task or cross-N comparison of any exponent without first
converting the reading level to excess-over-floor for each side. Absolute loss levels are not
comparable across tasks, and matched R^2 is not either, because the floors differ.

⚠️ This also means the flip-flop CANNOT currently test saturation. Reaching CDDM-comparable relative
depth needs the loss within ~10% of a floor that the budget check says is not even identified at
N=500/1000. That is a substantially longer-run experiment, not an analysis fix.

---

