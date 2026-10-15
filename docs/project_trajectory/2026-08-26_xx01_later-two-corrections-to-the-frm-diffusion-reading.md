## 2026-08-26 (later) — ⚠️ TWO CORRECTIONS to the frm diffusion reading

Both prompted by the challenge "if W_inp is still updated, how come W_rec is in a diffuse mode?" —
which was right to be suspicious, because the answer is that W_rec ISN'T.

### Correction 1 — "W_rec provides a usable diffusion read-out for frm (6/8)" is WITHDRAWN

Median α per 50k block tells a different story from the crossing detector:

| | 50k | 150k | 250k | 350k | 400k |
|---|---|---|---|---|---|
| frm W_rec | 0.91 | 0.77 | 0.78 | 0.73 | **0.74** |
| frm W_inp | 0.98 | 0.92 | 0.90 | 0.90 | **0.91** |
| none W_rec | 0.67 | 0.27 | 0.24 | 0.25 | 0.25 |
| none W_inp | 0.94 | 0.25 | 0.13 | 0.15 | 0.20 |

**Under frm, W_rec PLATEAUS at 0.74 — above the 0.6 threshold. It never settles.** The "6/8" came
from `diffusive_onset` firing on transient noise: α is noisy per probe, so a run plateauing at 0.74
still produces runs of 5 consecutive probes below 0.6 by chance. `persist=5` was insufficient.

**Fix:** `diffusive_onset` now additionally requires the TAIL MEDIAN of smoothed α to be below the
threshold — a run must still be settled at the end, not merely have dipped once. Self-check extended
with a synthetic plateau-plus-dip case. Effect: frm W_rec 6/8 → **2/8**; frm W_inp 0/8 (unchanged);
**the unpenalised headline data is unaffected — W_inp 65/65 at median 61k, W_rec 63/65.**

⚠️ Consequence: **the diffusion criterion is NOT available for frm on any variable.** frm must be
compared by matched LOSS, which does converge at 400k (median fall over the last third 1.0%). The
recommendation to "use W_rec for cross-condition diffusion matching" is withdrawn. This is the THIRD
time this one criterion has produced a plausible-looking number from a definition defect.

### Correction 2 — "there is no reason to expect W_inp ever settles under frm" was ASSERTED, not measured

Stated before checking the trajectory. Having now checked, the claim is SUPPORTED but was not
evidence-based when made: α(W_inp) under frm sits at 0.91, 0.90, 0.91, 0.90, 0.91 across the blocks
from 200k to 400k — **flat over 200 000 iterations**, while unpenalised it fell 0.94 → 0.20 over the
same span. The defensible statement is "shows no sign of settling across 200k iterations of plateau",
not "never".

**The obvious mechanism was tested and REFUTED.** If W_inp's directed motion were just norm inflation
(‖W_inp‖ is still creeping up under frm, 44 → 66, where unpenalised it froze at 89.1 by 150k), the
displacement would be radial and functionally meaningless. Decomposing |ΔW|/‖W‖ at lag 10 000 over the
second half of training:

| | total displacement | explained by norm change | cos between successive updates |
|---|---|---|---|
| frm | 0.0565 | 0.0024 (**4%**) | 0.455 |
| none | 0.0139 | 0.0003 (2%) | 0.706 |

Only 4% is norm change, so it is genuine reorganisation, and the displacement per lag is **4x larger**
than unpenalised. Note also that frm has the LOWER consecutive-update cosine (0.455 vs 0.706) yet the
HIGHER α — consecutive steps are noisier, but the direction persists over long lags. **No verified
mechanism for why frm sustains this; recorded as an open question rather than explained away.**

### rws ACCELERATES settling; frm PREVENTS it — so `both` is a genuine tug-of-war

All traces truncated to a common 150 000 iterations before measuring, so onsets are range-matched
(rws is 150k by construction; none and frm truncated from 400-500k). N=500, W_inp, corrected detector.

| penalty | settled | median onset | α at 150k |
|---|---|---|---|
| none | 20/24 | 65k | 0.53 |
| **rws** | 19/24 | **55k** | **0.44** |
| **frm** | 0/8 | — | **0.86** |

**rws settles the input weights FASTER than no penalty at all** (55k vs 65k) and reaches a lower α
(0.44 vs 0.53), while **frm stops them settling entirely** (plateau at 0.86-0.91). The two penalties
push the input-weight dynamics in OPPOSITE directions.

⚠️ Earlier "rws 43/43, median 54k" was computed with the pre-fix detector and pooled all sizes; the
range-matched N=500 figure is 19/24 at 55k. The conclusion (rws settles, frm does not) is unchanged
and in fact sharpened.

⚠️ The 55k vs 65k onset gap is suggestive, not tested — no CI computed. The α difference at a matched
150k (0.44 vs 0.53) is the more robust of the two comparisons.

**This makes `both` a sharp experiment rather than a routine cell.** The standing hypothesis has been
that rws positively modulates frm. The dynamics now give it a concrete, falsifiable form:

- if `both` settles → rws dominates, and it rescues the pathological non-settling frm induces
- if `both` plateaus like frm → frm dominates, and rws cannot rescue it
- if `both` settles LATER than none but does settle → genuine intermediate, a real modulation

⚠️ **No `both` nets exist yet** — 72 tasks queued, none started, sitting behind the frm backlog and
ahead of nothing. Both `both` and FFbigN are now scientifically interesting; FFbigN remains the one
answering a question the existing data cannot answer at all (the ceiling), whereas `both` answers a
question the existing data now poses very sharply.

---

