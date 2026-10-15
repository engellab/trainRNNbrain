## 2026-09-26 10:20 — what a duplicated unit actually inherits, decomposed

Four controls now separate the things a copy carries. All at N=1000, 3-bit flip-flop, 40000
iterations, replacement rate 0.025, maturity 1000, three seeds each, and identical in every respect
except the one named. The corrected 2x2 self-weight block is in all of them.

| rule | incoming weights | outgoing column | active units | mean | r2 | recruitment above control |
|---|---|---|---|---|---|---|
| control | — | — | 254, 269, 301 | 275 | 0.945 | — |
| random | iid at 1/sqrt(N) | stale, decayed | 250, 261, 275 | 262 | 0.945 | −13 |
| zero_out | iid at 1/sqrt(N) | zeroed | 278, 278, 312 | 289 | 0.946 | +14 |
| **copy_iid** | **iid at 1/sqrt(N)** | **donor's, halved** | 383, 391, 467 | **414** | 0.921 | **+139 (34%)** |
| **permute** | **donor's multiset, scrambled places** | donor's, halved | 568, 615, 634 | **606** | 0.936 | **+331 (81%)** |
| copy, jitter 0 | donor's, exact | donor's, halved | 660, 672, 724 | 685 | 0.942 | +410 (100%) |
| copy, jitter 3.0 | donor's, 37% of signs flipped | donor's, halved | 740, 755, 760 | 752 | 0.944 | +477 (116%) |

**Three ingredients, and the smallest one is the one I predicted would be everything.** Handing a
new unit the donor's outgoing column and nothing else buys 34% of the recruitment. Adding the
donor's weight STATISTICS -- the exact multiset of magnitudes, signs and zeros, in randomised
places -- takes it to 81%. Putting those weights in their original places adds the last 19%. I had
written that the outgoing projection was "the whole story"; it is a third of it.

⚠️ THE LARGEST SINGLE CONTRIBUTOR IS THE INCOMING WEIGHT DISTRIBUTION, not the wiring and not the
outgoing column. Going from iid incoming weights to the donor's multiset in scrambled places adds
192 units, more than the outgoing column's 139 and far more than correct placement's 79. A trained
row is heavy-tailed -- median off-diagonal weight 0.0015 against a maximum of 2.2 -- while an iid
draw at 1/sqrt(N) = 0.032 gives every weight the same scale and no tail at all. What a copy mostly
passes on looks like the SHAPE of a trained row.

**Jitter does not hurt and appears to help.** Rotating the copy off its donor keeps every unit:
685 at jitter 0, 708 at 0.3, 751 at 1.0, 752 at 3.0, where jitter 3.0 flips 37% of the row's signs
and leaves a cosine of 0.32. Seed spreads are 40-80 units, so the 0-to-0.3 stretch is flat within
noise and only the 1.0 and 3.0 cells clearly sit above jitter 0. Multiplicative jitter preserves
each entry's magnitude scale, so it never destroys the heavy tail -- which is consistent with the
tail being what matters.

**The jitter-0 cell settles a claim this code made about itself.** Its comment said that without
injected noise the twins "take identical drive and identical gradients and never separate". They
separate perfectly well: 685 active units against 676 at the original copy_noise of 0.05. These
networks inject independent recurrent noise per unit (sigma_rec = 0.05), so identical twins receive
different inputs from the first step.

**Dimensionality, reported as the total the population spans** (Pavel: dimensions per unit is not
the right measure; what matters is the overall dimensionality of the trajectories):

| cell | active | total dimensions (noise-free) |
|---|---|---|
| control | 278 | 5.6 |
| random | 262 | 5.0 |
| jitter 0 | 707 | 7.9 |
| jitter 0.05 | 701 | 7.2 |
| jitter 0.3 | 734 | 9.7 |
| jitter 1.0 | 782 | 7.3 |
| jitter 3.0 | 781 | 7.8 |

Duplication raises what the population actually spans by 1.3-1.7x at no cost in r2. The win
condition I pre-registered was written in dimensions PER UNIT and no cell met it; on total
dimensions every cell clears the control comfortably. The pre-registration was answering the wrong
question, which is worth recording as a pre-registration failure rather than quietly re-scoring.

