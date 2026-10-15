## 2026-08-28 — ★ `both` lands: rws does NOT rescue frm

First 9 `both` cells (k=1,2,3 × 3 seeds, N=500, 400k). The three-way prediction was fixed on
2026-08-26 before any `both` data existed:

> *both settles like none* → rws dominates and rescues frm's non-settling;
> *both plateaus like frm* → frm dominates;
> *both settles but later* → genuine intermediate, a real modulation.

### Axis 1 — W_inp settling. N=500, k=1-3 only, so all four conditions are compared on the same cells.

| pen | 50k | 100k | 150k | 200k | 300k | 400k | settled |
|---|---|---|---|---|---|---|---|
| none | 0.95 | 0.33 | 0.20 | 0.17 | 0.16 | **0.14** | 9/9 @44k |
| rws | 0.95 | 0.54 | 0.51 | — | — | — | 8/9 @50k |
| frm | 0.98 | 0.94 | 0.92 | 0.90 | 0.91 | **0.90** | **0/9** |
| **both** | 0.96 | 0.91 | 0.86 | 0.78 | 0.80 | **0.78** | **1/9 @202k** |

**The third outcome, but only weakly.** `both` plateaus at 0.78 against frm's 0.90 and none's 0.14 —
it recovers **16%** of the frm→none gap, and only 1 of 9 runs ever settles (frm 0/9, none 9/9).
rws shifts frm's dynamics measurably but does not rescue them.

### Axis 2 — silence. rws recovers NOTHING here.

| pen | mean M/N at 150k | per k (1/2/3) |
|---|---|---|
| none | 0.356 | 0.33 / 0.36 / 0.38 |
| rws | 0.291 | 0.24 / 0.30 / 0.33 |
| frm | 0.992 | 0.99 / 1.00 / 0.99 |
| **both** | **1.000** | 1.00 / 1.00 / 1.00 |

`both` has **literally zero silent units** — slightly more saturated than frm alone (recovery: −1%).
On a scale where none = 0.356 and frm = 0.992, rws moves the combination not at all.

### Reading

**On the two axes measured, frm dominates the combination and rws does not modulate it.** The
standing hypothesis — that rws positively modulates frm — is not supported for either the
input-weight dynamics or the silent fraction.

⚠️ **This does NOT test the hypothesis as originally stated.** The original framing was about
HETEROGENEITY: "frm alone simply makes neurons make large transients, while rws makes the transients
go away and purposes all units for useful compute... it makes networks less heterogeneous on top of
it." That is a claim about the DISTRIBUTION of activity across units, pre-registered in
docs/experiments/frm_rws_heterogeneity.md with metrics A1-A6 — **none of which have been run.**
Silence fraction and weight settling are not substitutes for it. The heterogeneity test is now the
outstanding item, and `both` cells finally exist to run it on.

⚠️ Also note `both`'s M/N = 1.000 exactly saturates the scale-free measure, so `b` and `c` cannot be
fitted for it at all — the same ceiling problem as frm, but complete. Any `both` comparison must use
a measure that is not already saturated.

⚠️ 9 cells, k=1-3, N=500 only. The k=4-8 and larger-N `both` cells are still running.

---

