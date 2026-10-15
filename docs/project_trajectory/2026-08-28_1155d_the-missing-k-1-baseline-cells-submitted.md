## 2026-08-28 — the missing k=1 baseline cells, submitted

**Gap found:** the unpenalised sweep runs `--array=1-63` with `KS=(2 3 4 5 6 7 8 1)` and
`K_IDX = TID/9`, so tasks 1-63 cover **k=2..8 only**. k=1 sits at `KS[7]`, reachable only by tasks
64-72, and `sacct` confirms only **64, 65, 66** were ever submitted — k=1 at N=500. Tasks 67-69
(k=1, N=1000) and 70-72 (k=1, N=2000) were never launched.

k=1 coverage when every queued job finishes, before this fix:

| penalty | N=500 | N=1000 | N=2000 | N=4000 |
|---|---|---|---|---|
| **none** | ✓ | **✗ never submitted** | **✗ never submitted** | ✓ (running) |
| rws | ✓ | ✓ | ✓ | ✓ (running) |
| frm | ✓ | ✓ | ✓ (running) | — |
| both | ✓ | queued | queued | — |

**Every penalty condition had full k=1 coverage; the baseline did not.** Consequences: every
penalty-vs-none contrast at k=1 was N=500-only; the floor law's intercept `a` (the single-channel
floor) was extrapolated from k≥2 at N=1000 and N=2000, when k=1 exists precisely to pin it — the
launcher header describes k=1 as the out-of-sample test that discriminated `C(k,2)` from `k^1.5`
(0.02440 vs 0.02373, a gap 7x the residual scale); and every "k=1→8" figure quoted so far rests on
one size at its lower end. The `·` at k=1 in the N=1000/2000 rows of every matrix figure was not data
in transit, it was data never requested.

**Submitted** (arrays 5924467, 5924468), matching the existing cells' budgets exactly so floor fits
stay range-matched:
- tasks 67-69: k=1, N=1000, 500k iterations, 72h request
- tasks 70-72: k=1, N=2000, 400k iterations (`ITERS_OVERRIDE`), 96h request

Decode verified by hand before submitting. ~290 GPU-h, negligible beside the ~5,400 already queued.

⚠️ These run on the CURRENT code (chunked `participation_from_states_`, `storable_` saver) while the
existing k=2..8 `none` cells ran the pre-fix version. The chunking was verified numerically identical
in float32 — zero units change silence class under any of the three rules — so the cells are
comparable, but the code split is recorded.

---

