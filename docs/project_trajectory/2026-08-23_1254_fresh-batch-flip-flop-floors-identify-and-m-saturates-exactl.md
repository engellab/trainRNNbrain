## 2026-08-23 12:54 — fresh-batch flip-flop: floors identify, and M saturates exactly as on CDDM

26 nets across 9 cells (k=2..6 x N=500, k=2..5 x N=1000, 3 seeds each, 500k iterations, batch 1024,
fresh every step). First analysis of valid flip-flop data.

### 1. The floor is IDENTIFIED — the halved-range check passes at 2%

| | max |shift| in L_inf when the fit range is halved |
|---|---|
| fresh batch, N=500 | **2%** |
| fresh batch, N=1000 | **2%** |
| RETRACTED same_batch, N=500 | 83% |
| RETRACTED same_batch, N=1000 | 34% |

The thing that would not identify for two days was an artifact of memorisation. With fresh batches the
loss plateaus by ~25k and stays there, so L_inf is a real measurable quantity.

### 2. The floor is EXACTLY N-independent and significantly k-dependent

Empirical floor (mean loss over the final 2000 probes):

| k | N=500 | N=1000 | ratio |
|---|---|---|---|
| 2 | 0.02493 | 0.02493 | 1.000 |
| 3 | 0.02589 | 0.02588 | 0.999 |
| 4 | 0.02667 | 0.02669 | 1.001 |
| 5 | 0.02738 | 0.02735 | 0.999 |
| 6 | 0.02799 | (pending) | |

Agreement to FOUR significant figures across a 2x size change. The floor is a pure task property.
It rises monotonically with k, and the nested fit rejects a shared floor decisively (F(4,793)=141.6,
p~0, dAICc +444 at N=500; F(3,680)=104.6, dAICc +267 at N=1000). Increments shrink with k
(+0.00096, +0.00078, +0.00071, +0.00061), i.e. sublinear and possibly saturating.

**Consequence for the protocol:** matching on absolute loss ACROSS N is exactly licensed. Across k it
is not — per-k levels or relative-depth matching are required.

### 3. M SATURATES with N, at the same exponent as CDDM

Active units at matched iteration 490k, scale-free criterion:

| k | M(N=500) | M(N=1000) | b |
|---|---|---|---|
| 2 | 130.3 | 162.7 | 0.32 |
| 3 | 137.3 | 188.3 | 0.46 |
| 4 | 149.0 | 199.7 | 0.42 |
| 5 | 166.0 | 204.0 | 0.30 |

**mean b = 0.37**, against CDDM's 0.31 (hard) / 0.35 (scale-free).

⚠️ **This retracts my own explanation from 2026-08-21 16:56.** I attributed the CDDM/flip-flop
saturation contradiction to a READING-DEPTH artifact and wrote a standing rule about excess-over-floor
around it. The real cause was MEMORISATION: the same_batch runs had b ~ 1.0 because a network
memorising 256 frozen trials recruits units in proportion to its size. On valid data the two tasks
agree closely and there was never a contradiction to explain. The excess-over-floor rule is still
sound practice, but it was not the explanation here.

### 4. N now DOMINATES k, reversing the retracted conclusion

Scale-free silent %, matched iteration 490k:

| k | N=500 | N=1000 |
|---|---|---|
| 2 | 73.9 | 83.7 |
| 3 | 72.5 | 81.2 |
| 4 | 70.2 | 80.0 |
| 5 | 66.8 | 79.6 |
| 6 | 65.7 | (pending) |

- k effect, k=2->6 at fixed N=500: **8.2 pp**
- N effect, 500->1000 at fixed k=2: **9.8 pp** — from a mere 2x size change

Doubling N moves silencing more than tripling k. The retracted analysis had this backwards (it found
k explaining 65-86% of variance and N 1-14%); memorisation suppressed the N dependence, because a
memorising network scales its recruited population with N.

**This restores "too many units" as the dominant factor and puts CDDM and the flip-flop in
agreement.** Silencing is severe (66-84%), rises with N, and falls only modestly with task complexity.

⚠️ Caveats: two N values only, so b comes from a single ratio per k; k=6 at N=1000 and everything at
N=2000 still running. Revisit when the grid completes.

---

