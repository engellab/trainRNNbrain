## 2026-07-27 (15:30) — large-N benchmark: does the ACTIVE unit count saturate? (benchmark submitted)

### The objection this addresses

If a trained RNN leaves half its units silent, the obvious rebuttal to any rescue method is:
**"why bother making every unit compute — just train a bigger network and prune the silent ones.
Compute is cheap, so what are you actually solving?"** Answering it requires knowing how the number
of *active* units grows with N. Two regimes, with completely different consequences:

- **Growth.** Active units keep increasing with N. Then "train big and prune" genuinely works, and
  the contribution is a cheaper route to the same thing, plus whatever population-level differences
  the rescued networks show (`paper.md` §5).
- **Saturation.** Active units plateau at a task-determined ceiling. Then **pruning cannot get you
  there at all** — no network size yields a large active population, and activity regularization is
  the only route. That would be a much stronger result.

### What the existing data says — and why it cannot settle it

Active-unit counts from the original Dale size sweep (`CDDM_4a031e`, h/none, 5 nets/cell):

| N | 100 | 500 | 1000 |
|---|---|---|---|
| silent | 0.0% | 23.7% | 47.1% |
| **active** | **100** | **382** | **529** |

Two models fit these three points about equally well and diverge wildly beyond them:

| Model | Fit | N=2000 | N=5000 | N=10000 |
|---|---|---|---|---|
| **A: power law** | `active = 3.57·N^0.723` | 873 | 1695 | **2798** |
| **B: saturating** | `active = 1011·N/(N+911)` | 695 | 855 | **926** |

At N=10000 they differ by 3×, so the experiment is decisive. **The local exponent is already
falling** — 0.833 between N=100 and 500, then 0.470 between 500 and 1000 — which leans toward B, a
ceiling near ~1000 active units for CDDM. That is precisely why the global "N^0.72" summary should
not be trusted: it averages over a slope that is visibly decaying.

### Explicit prediction (recorded before the run)

**I expect the power-law summary `active ≈ 3.6·N^0.72` to OVERSTATE growth, and the truth to lie
closer to the saturating fit — a ceiling of order 1000 active units.** The alternative hypothesis is
sustained power-law growth. Concretely, at N=10000 the two predict **2798 vs 926** active units
(72% vs 91% silent). Anything below ~1200 active at N=10000 confirms saturation; anything above
~2000 confirms growth.

If saturation holds, the paper's answer to the objection becomes sharp: *pruning cannot deliver a
large active population at any size, because the ceiling is set by the task, not the budget.*

### Why this needs careful engineering — three separate failure modes

1. **Wall time, not memory, binds.** Cost scales as N² per timestep, 300 sequential steps, 450
   conditions. From 0.2 s/iter at N=1000: N=5000 ≈ 41 h and N=10000 ≈ 165 h for 30000 iterations —
   beyond `gpu-short` (24 h) and even `gpu-long` (6 days). **Mitigation:** the silent fraction is
   decided in the first ~400–600 iterations and frozen thereafter (measured in the traces; our N=1000
   test gave 53% silent at iteration 3000 vs ~55% at 30000), so large-N cells can run ~5000
   iterations — *with an N=1000 control at both 5000 and 30000 to prove the truncation is harmless*.
2. **Parameter saving breaks before the GPU does.** At N=10000, `W_rec` is 100 M floats and
   `run_experiment` writes `LastParams` *and* `BestParams` as indented JSON — ~2 GB each, and
   `jsonify` first materializes Python lists at ~24 bytes/float (~2.4 GB RAM per file).
   **Mitigation:** save large parameter sets as `.npz` (400 MB binary, seconds) above a size
   threshold.
3. **A hidden per-iteration cost.** `run_training` does `deepcopy(RNN.get_params())` on every
   training-loss improvement — nearly every iteration early on. At N=10000 that is a 400 MB
   tensor→numpy conversion plus deep copy per iteration; invisible at N=1000, potentially dominant
   at N=10000.

### The benchmark (submitted — Della job `11677325`, commit `6a1119a`)

`trainRNNbrain/experiments_and_analysis/benchmark_large_N.py` runs real `Trainer.train_step`
iterations on the production task and model at N ∈ {1000, 2000, 5000, 10000} and reports peak
allocated/reserved GPU memory, measured s/iter extrapolated to full runs, and the size the saved
JSON parameter files would reach. Nothing is submitted at scale until these numbers exist.

Queued behind our own jobs: the two standard-RNN arrays hold all 44 slots of the `gpu-short`
per-user cap.

> **Process note (a near miss).** The benchmark commit was checked out in `~/trainRNNbrain_std` —
> the worktree that array `11672037` was still using. It happened to be harmless (0 tasks pending;
> the only diff under `trainRNNbrain/` was the new benchmark file), but it is exactly the mistake the
> one-worktree-per-sweep rule exists to prevent. Benchmarks and analysis get their own worktree.

