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

## Results

(filled in after the run)
