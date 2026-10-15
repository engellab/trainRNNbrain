## 2026-09-22, 22:31 — The 16τ and 36τ DMTS runs are cancelled and deleted

Cancelled job 6312325 (`DMTSv2`, the 16-step delay) on Spock and deleted both
`DMTS_v2_pen` (17 nets, 367 MB) and `DMTS_v2_delay36` (12 nets, 415 MB). The 36-step jobs had
already finished; their data is gone too. 1 GB freed. Still running: `DMTSdelay` (6320918, the
8/10/12-step ladder at N=1000, no penalties) and `FFinpcap` (6321178).

**What the deleted runs showed, recorded here because the data no longer exists.**

16-step delay, read-out at r² ≥ 0.8:

| size | arm | solved | r² | active units |
|---|---|---|---|---|
| N=500 | frm | 3/3 | 0.992, 0.992, 0.991 | 500 of 500 |
| N=500 | both | 3/3 | 0.994, 0.994, 0.993 | 500 of 500 |
| N=500 | none | 1/3 | 0.999, 0.452, 0.452 | 146.7 ± 82.9 |
| N=500 | rws | 0/3 | 0.452, 0.452, 0.451 | 99.3 ± 9.3 |
| N=1000 | none | 0/3 | 0.452, 0.452, 0.450 | 151.7 ± 6.7 |
| N=1000 | rws | 0/2 | 0.452, 0.452 | 152.0 ± 15.6 |

The `frm` and `both` cells at N=1000 never ran; only the failing arms reported there.

36-step delay, N=1000 — **every arm failed**:

| arm | solved | r² | active units |
|---|---|---|---|
| both | 0/3 | 0.471, 0.470, 0.463 | 1000 of 1000 |
| frm | 0/3 | 0.470, 0.470, 0.469 | 971.3 ± 26.1 |
| none | 0/3 | 0.472, 0.472, 0.472 | 126.0 ± 14.8 |
| rws | 0/3 | 0.471, 0.471, 0.471 | 176.0 ± 36.8 |

**Recruiting units does not confer the capability.** `frm` and `both` have essentially the whole
network firing at the 36-step delay and score 0.47, the same as unpenalised networks with 126
active units. Whatever the rate penalty does for the 16-step delay, it does not extend the memory
horizon.

⚠️ Read those r² values as a floor, not as a measurement. Every arm and every seed lands in
0.45–0.47 with almost no spread, which is the signature of a network emitting a constant and
scoring whatever that yields — not of four arms failing in four different ways.

**Why 36τ was dropped rather than pursued.** No arm solves it, so it separates nothing. The 16-step
delay is dropped with it because its N=1000 cells were never going to finish before the decision
was needed, and the N=500 result it did produce is superseded by the delay ladder: 8, 10 and 12
steps at N=1000 with no penalties will say where the unpenalised network's memory horizon actually
breaks, which is the number the scaling figure needs. Picking the delay from a sweep is the right
order; picking it first and discovering nothing trains is what happened here.

---

