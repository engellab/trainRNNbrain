## 2026-09-26 10:20 — other sweeps landed

**Dropout, all four sizes complete** (3-bit flip-flop, rate 0.20, beta 4, three seeds):

| N | control | mute | dead |
|---|---|---|---|
| 500 | 211 @ 0.946 | 398 @ 0.928 | 498 @ 0.779 |
| 1000 | 275 @ 0.945 | 524 @ 0.928 | 904 @ 0.828 |
| 2000 | 451 @ 0.947 | 720 @ 0.931 | 1370 @ 0.818 |
| 4000 | — | 1087 @ 0.931 | 1897 @ 0.817 |

`dead` dropout saturates at N=500 (496, 499, 499 of 500 units) and that is where its r2 cost is
worst, 17 points against 12–13 at the larger sizes.

**Duplication with the corrected construction, size series** (gamma 0.1): N=500 control 244 ->
duplication 495 at no r2 cost; N=1000 323 -> 742 costing 3.8 points; N=2000 462 -> 1190 costing 1.5.
Separately, in the rate grid at N=2000, 1003 active units at r2 0.936. N=4000 is still running after
two TIMEOUTs -- 12 h reached only 19448 of 40000 iterations, so it is resubmitted at 36 h.

**DMTS at 7 tau**, solve rate (clean r2 >= 0.9), three seeds per cell:

| N | none | rws | frm | both |
|---|---|---|---|---|
| 500 | 2/3 | 3/3 | 3/3 | 3/3 |
| 1000 | 3/3 | 3/3 | 3/3 | 3/3 |
| 2000 | 2/3 | 3/3 | 3/3 | 0/1 so far |

N=1000 solves 3/3 unpenalised, which is what the scaling series needs.

