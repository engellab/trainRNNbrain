# agent-lowrank: can weight snapshots be stored at low rank?

Storage only. The networks stay full-rank and are trained as now. The question is whether a snapshot
of W_rec can be written to disk as a few factors and read back without changing any measurement the
paper makes on it.

## The schemes

- **init + low rank (the proposal).** W_rec(t) ≈ a(t)·W0 + U_r S_r V_rᵀ. W0 is the untrained
  recurrent matrix, regenerated from the chain's first run's seed (`gradient_participation.initial_params`).
  a(t) = ⟨W, W0⟩ / ⟨W0, W0⟩ absorbs the shrinkage weight decay applies to W0. U_r S_r V_rᵀ is the
  rank-r truncated SVD (singular value decomposition) of ΔW = W − a·W0. Stored: 2·r·N + r + 1 numbers.
- **plain SVD.** The rank-r truncated SVD of W itself. Stored: 2·r·N + r numbers.
- **float16 full matrix**, as the obvious competitor: N² numbers at half precision.

W_inp, W_out and y_init are kept exact in every scheme. They are N × (2 to 6) and N × 1, so storing
them exactly costs no more than a rank-6 factor; low-rank versions of them would save nothing. Variant
(ii) of the brief ("W_inp/W_out kept exact") is therefore the same thing as the proposal here.

## Pass bar (fixed before any result was computed)

Every comparison is a reconstruction against the original weights of the same snapshot, on the same
batch and the same noise seeds, so scorer noise cancels and only the weight error remains.

A reconstruction at rank r is **lossless for this paper** only if all of these hold:

1. **Held-out R².** |R²(reconstruction) − R²(original)| ≤ 0.002, both scored by
   `heldout_r2.score_net(folder, draws=36)` on a folder holding the reconstructed parameters and the
   run's config.
2. **Active units.** Participation p = std(r) + q0.9(|r|) on a noise-free pass of the task's training
   batch (`gradient_participation.unit_gradients`); a unit is active if p ≥ 0.05·q95(p). The active
   count is within 1% of the original's, AND at least 99% of the N units get the same active/silent
   label (at N = 1000: at most 10 units flip).
3. **Training loss.** The noisy training loss (masked MSE, the model's own sigma_rec/sigma_inp/sigma_out,
   one noise draw with a fixed seed, on the same fixed training batch) is within 1% of the original's.
4. **Gradient read-outs (claim C4).** Per-unit incoming gradient norm g_in = ‖∂L/∂W_rec[i,:]‖ ⊕
   ‖∂L/∂W_inp[i,:]‖ and outgoing g_out = ‖∂L/∂W_rec[:,j]‖ ⊕ ‖∂L/∂W_out[:,j]‖, from the same
   noise-free pass: Pearson correlation with the original's ≥ 0.99, for each of g_in and g_out.

**Smallest passing r, per task:** the smallest r in the rank grid at which every snapshot of that task
(3 runs × {at the floor, at the end}) passes all four tests, and every larger grid rank passes too.
Rank grid at N = 1000: 1, 2, 5, 10, 20, 50, 100, 200, 300, 500, 700.

**What counts as "a real win":** storage at the passing rank at least 4× smaller than the full matrix in
float32 (3.8 MB at N = 1000; the snapshot files are compressed npz, and float32 training noise barely
compresses). At N = 1000 that means r ≤ ~120. If the passing rank exceeds ~N/2 the factors cost more than
the full matrix and the scheme has failed outright.

**What would show the scheme is broken:** a passing rank above ~120 at N = 1000 for any task, or a
passing rank that grows in proportion to N (so the saving does not improve at N = 4000, where storage
matters). Before running, I cannot predict which way this goes: Adam's per-parameter normalised steps
and the recurrent noise could spread ΔW across every direction, and weight decay of 1e-6 is too weak to
shrink the bulk of it.

## Data

Claim-1 controls at N = 1000: CDDM_c1_snapshots, DMTS_c1_snapshots, NBitFlipFlop_c1_snapshots_c300000
(two runs; the third flip-flop seed stops at 200k in NBitFlipFlop_c1_snapshots_c200000 and is read
there). Each is read at its noisy-loss floor (`common.noisy_floor` → `fig_knobs_at_floor.snapshot_at`)
and at the end of training. Then CDDM_paper_grid N = 4000 control, end of training, to see how r
scales with N.

## Results (2026-10-14)

Script: `trainRNNbrain/experiments_and_analysis/lowrank_storage.py` (`run`, `chain`, `figure`). Records:
Della `/scratch/gpfs/TENGEL/pt1290/agent_lowrank/results/*.json`. Figure:
`img/internal_figures/lowrank_storage.{pdf,svg,png}`.

Data note: the third flip-flop seed (1258717025) never continued past 200k, so it is read in
NBitFlipFlop_c1_snapshots_c200000 (end = 200k). The other two end at 300k.

**The write-and-score path adds nothing.** The `exact` scheme writes the original weights through the
same folder path the reconstructions use. On all 18 snapshots every difference is exactly 0.

### The proposal fails because training erases W0

W0 is regenerated correctly: at iteration 500, a = 0.99 and corr(W, W0) = 0.98 in all three tasks. Then
the trained matrix moves away from it. a(t) falls to 0.01 to 0.16 at the floor and to 0.00 to 0.06 at the
end. At N = 2000 and N = 4000 the end values are 0.009 and 0.004. The trained W is also smaller than W0:
‖W0‖ / ‖W‖ at the end is 2.2 (CDDM), 3.3 to 4.0 (DMTS), 3.0 (CDDM N = 2000) and 4.0 (CDDM N = 4000).
Flip-flop is the exception at 0.87 to 0.89. So ΔW = W − a·W0 is W itself, to within 3% of its norm,
and "a·W0 + low rank" stores the same thing as a plain SVD. On every snapshot the two schemes pass at the
same rank or within one grid step. Init + low rank never beats plain SVD by a grid step, and it is one
step worse on one flip-flop floor (700 against 500).

Why W0 disappears is not tested here. One candidate: Adam applies the L2 weight-decay term after
normalising by the gradient's running RMS, so an entry with little task gradient is pulled toward 0 at
close to the full learning rate, far faster than wd = 1e-6 suggests. The falsifying test would be a run
with AdamW (decoupled decay), or with wd = 0, keeping a(t) near 1 by the floor. Treat the candidate as
interpretation.

### ΔW is low rank in energy, not in what the measures need

99% of ‖ΔW‖² sits in 66 to 254 directions at the end of training and in 160 to 590 at the floor
(a Gaussian 1000 × 1000 matrix needs 772). At the end of CDDM training r99 grows sublinearly with N:
about 250, 373 and 553 at N = 1000, 2000 and 4000. But the pass bar needs a relative Frobenius error of
about 2 to 6%, which is 99.6 to 99.96% of the energy, and the last fraction of a percent is spread over
hundreds of directions.

### Smallest passing rank (every snapshot of the task passes there and at every larger tested rank)

| task | N | floor (3 runs) | end (3 runs) | all 6 | bytes at "all 6" |
|---|---|---|---|---|---|
| CDDM | 1000 | 500 | 300 | 500 | 4.0 MB |
| DMTS | 1000 | 500 | 300 | 500 | 4.0 MB |
| 3-bit flip-flop | 1000 | 500 (init: 700) | 200 | 500 | 4.0 MB |
| CDDM | 2000 | not kept | 200 (1 run) | | 3.2 MB |
| CDDM | 4000 | not kept | 200 (1 run) | | 6.4 MB (float32 matrix: 64 MB) |

The full float32 matrix is 4.0 MB raw and 3.8 MB compressed at N = 1000. At N = 1000 the passing rank costs
as much as the matrix it replaces, so the scheme fails at the size where the claim-1 data live. Every
bar binds somewhere one grid step below: unit agreement and gradient correlation on the CDDM floors, the
noisy loss alone on the DMTS floors, and four of the five bars on one flip-flop floor.

At the end of CDDM training the passing rank stops growing with N (300, 200, 200 at
N = 1000, 2000, 4000), so the saving improves with N there. This rests on one network per large N, at
the end of training only. No N ≥ 2000 control kept snapshots, so the floor, which needed 1.7× the rank
at N = 1000, is untested at large N.

### Delta coding along the snapshot chain does not help

W_k ≈ Ŵ_{k−1} + rank-r SVD of (W_k − Ŵ_{k−1}), keyframe = the seed-regenerated W0, one snapshot per 1000
iterations, one run per task. At r = 100 per snapshot (the same bytes per snapshot as plain SVD at r = 100)
it passes on 1 of 6 snapshots (the flip-flop end); its Frobenius error at the CDDM end is 0.20, against
0.23 to 0.24 for plain SVD at r = 100. The steps between snapshots are themselves high-rank, and the
part each step leaves out piles up (at r = 5 the error exceeds ‖W‖).

### What does pass: fewer bits, not fewer dimensions

| format | bytes (N = 1000) | compressed | snapshots passing (of 18) |
|---|---|---|---|
| float32 | 4.0 MB | 3.7 to 3.8 MB | 18 (exact) |
| float16 | 2.0 MB | 1.8 to 1.9 MB | 18; every measure unchanged to 4 decimals |
| 12-bit, per-row absmax | 1.5 MB | ~1.1 MB (entropy bound) | 18 |
| 10-bit | 1.25 MB | ~0.9 MB | 14 (all 6 CDDM; fails 2 DMTS on the loss, 2 flip-flop on g_in) |
| 8-bit | 1.0 MB | 0.64 MB | 7 (all 6 CDDM; DMTS fails on loss, flip-flop on loss and g_in) |
| 6-bit, 4-bit | 0.75, 0.5 MB | | 2, 0 |

At N = 4000 (one CDDM network, end of training) float16, 8-bit and even 6-bit pass.

### Recommendation

Do not store snapshots as a·W0 + low rank: at N = 1000 the rank the paper's measures need costs as much
as the matrix. Store W_rec in float16 (2× smaller, passes everywhere with every measure unchanged to
4 decimals) or, for about 3.3×, 12-bit integers with one float32 scale per row plus a lossless coder. At
N = 4000 a plain rank-200 SVD (6.4 MB, 10× smaller than float32) of the end-of-training matrix would be smaller still, but that is one
network, end only; it should not be relied on for snapshots taken before the floor without testing a
kept N = 4000 snapshot series first.
