## 2026-08-28 — ★ Criterion search: `excess` wins, and M vs PR DISSOCIATE

New: [`criterion_search.py`](../trainRNNbrain/experiments_and_analysis/criterion_search.py).
Scoring thresholds fixed BEFORE running any candidate (scratchpad/criteria_spec.md): coverage ≥ 90%
in every condition, within-cell seed CV ≤ 0.30, and for `none` the fitted c must straddle 0 (three
independent matched criteria say it should, and the k-compensation test showed fixed-compute's
c = +0.179 is entirely convergence depth).

### Candidates and outcomes — 180 runs across all four conditions

| criterion | coverage (none/rws/frm/both) | seed CV | c for `none` | verdict |
|---|---|---|---|---|
| `iter@150k` | 100/100/100/100 | 0.00 | **+0.179 \*** | FAILS deconfounding |
| `drift(W_inp)` | 100/81/**3**/**11** | 0.12 | −0.026 | FAILS coverage on frm, both |
| `drift(W_rec)` | 97/94/**9**/**0** | 0.12 | +0.001 | FAILS coverage |
| `slope<0.05` (new, floor-free) | 100/100/97/100 | 0.07–**0.51** | −0.011 | FAILS seed CV on `both` |
| `rho<0.05peak` (new, floor-free) | 100/100/80/100 | 0.10–**0.49** | −0.039 | FAILS coverage + CV |
| `loss` | 100/100/100/100 | 0.03–0.06 | +0.016 | passes, but see below |
| **`excess`** | **100/100/100/100** | **0.04–0.06** | **−0.029** | **PASSES everything** |

⚠️ `loss` passes the thresholds but is NOT equivalent: it disagrees with `excess` in SIGN on rws
(c = −0.113 vs +0.074) and its k-slope on frm is 0.91, i.e. it reads high-k cells far later. That is
the known k-unfairness of an absolute loss level — the floor rises with k, so one L\* sits at a
different depth for each k. `excess` corrects exactly that by using each cell's OWN floor.

**The two floor-free candidates were the interesting failures.** `slope` (read where the log-log
learning rate ρ = −dlogL/dlogt first falls below a threshold and stays) needs no fit, no floor and no
convergence assumption, and it deconfounds `none` correctly. It fails only on `both`, where its seed
CV reaches 0.44–0.51. Worth revisiting if more `both` cells make it stabler.

### ★ PROGRAM MINIMUM (within a penalty condition): use `excess`

Read each run at 1.10 × its OWN fitted floor, with the floor fitted over that condition's own
sufficient range. 100% coverage everywhere, seed CV 0.04–0.06, read-out time rises with k
(dlogT/dlogk = 0.17–0.47) as it must.

⚠️ Range-matching across conditions is NOT required here and was actively harmful: it is only ever a
proxy for "the floor is correctly estimated". Verify convergence per condition instead, then fit each
over its own range. Truncating everything to 150k to be "fair" is what made frm's floor invalid.

### ★ PROGRAM MAXIMUM (across penalties): `excess` + PARTICIPATION RATIO, not M

**The criterion was never the blocker for cross-penalty comparison — the OUTCOME MEASURE was.** At
any read-out, the thresholded count saturates:

| pen | M/N | PR/N |
|---|---|---|
| none (N=500) | 0.564 | 0.392 |
| rws (N=500) | 0.564 | 0.346 |
| frm (N=500) | **0.994** | 0.968 |
| both (N=500) | **1.000** | 0.925 |

M/N = 0.99–1.00 for frm and both: the measure cannot tell them apart or fit an exponent. **PR is
graded and does** (0.968 vs 0.925), so `common.participation_ratio` was added. ⚠️ PR is exactly
1/HHI — algebraically identical to the existing `hhi`, provided under both names because they read
differently. Never report both as independent evidence.

### ★ AND A NEW RESULT — M and PR dissociate for `none`

| criterion | c for M | c for PR |
|---|---|---|
| loss | +0.016 | **+0.053 \*** |
| excess | −0.029 | **+0.084 \*** |
| drift(W_inp) | −0.026 | **+0.098 \*** |
| drift(W_rec) | +0.001 | **+0.108 \*** |
| slope<0.05 | −0.011 | **+0.119 \*** |
| rho<0.05peak | −0.039 | **+0.089 \*** |

*(\* = CI excludes zero)*

**Under all six deconfounding criteria, M is k-independent and PR rises with k (c ≈ 0.05–0.12).**
This is not a contradiction of the headline; it is a refinement:

> As task complexity grows, the SAME NUMBER of units stay above threshold, but activity spreads MORE
> EVENLY among them. The network does not recruit more units — it uses the ones it has more uniformly.

The count does not track task demand; the effective dimensionality does. That reconnects directly to
the task-intrinsic-dimensionality literature, which measures PR-like quantities rather than
thresholded counts — the observable mismatch flagged when that literature was first checked.

⚠️ `both` remains unfittable for a two-variable law, but for COVERAGE reasons (k=1–3 at N=500 only),
not saturation. It needs k≥3 at ≥2 sizes.

---

