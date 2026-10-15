## 2026-10-13 — R² at the floor, and the revival-screen top-ups

**Top-ups submitted (Pavel: options A + B).** All keep a snapshot every 500 iterations (`KEEP=1`),
from `~/trainRNNbrain_c1top` (7739a60), launcher at 5afa577 (the continuation's root is now named
after its TARGET; it was always `_c200000`):
- A, `CDDM_screen3` → 75k, arms none / cap / prune (15499043-45, 9 runs, ~17 GPU-h).
- B, `NBitFlipFlop_replace_variants` → 200k, rules bias_kick / random / zero_out at rate 0.025
  (15499046-48, 9 runs, ~95 GPU-h).

Three of the six CDDM arms were NOT continued, because the pinned code no longer runs them as they
were trained (screen at 58d76f5): `scale` and `scale_ln` (synaptic scaling gained
`preserve_row_norm`, default true, absent from their configs) and `prune_copy` (the copy rule's
self-weight block was rewritten on 2026-09-24). For the three that were, every new prune key defaults
to the old behaviour (maturity 0, max_replace_frac 1, utility "participation"). The flip-flop
rules (ddd0228) differ from 7739a60 only in code paths for `copy` and `rescale`.

**R² at the floor (Pavel: report R² beside the active count, as a trade-off scatter).** No
weights exist at the crossing for the ladder runs, so held-out R² there is estimated from the same
run's end: R2_floor = 1 - (L(t*)/L_end)(1 - R2_end), L the noisy loss (running median at t*, median
of the last 10% at the end), R2_end = heldout_r2 on LastParams.

Test, threshold set before running: against held-out R² measured on weights saved at the crossing
(ablation JSONs, nine claim-1 networks), every network within 0.01. **Result: FAIL, worst 0.0115.**
Eight of nine are within 0.006 (DMTS within 0.0004; flip-flop within 0.0021). The miss is CDDM
s2670295995, and its cause is the scorer, not the estimator: on the SAME end weights heldout_r2
gives 0.847 and the ablation code 0.836 (the other two seeds agree to 0.002). One CDDM noise draw
has an SD of ~0.011, so two independent 9-draw scores can differ by 0.01. With the ablation code's
end scores the three CDDM estimates sit within 0.003 of the oracle. The estimator is used; CDDM R²
differences below ~0.01 are scoring noise and are not read as effects.

**The trade-off scatter** (`fig_knobs_at_floor.py --tradeoff` →
`img/internal_figures/fig_f2_knob_tradeoff.{pdf,svg,png}`): one point per run, held-out R² at the
floor against active units at the floor; dose ladders coloured in dose order, activations by marker
shape, the standard network ringed. Reading (standard CDDM: 418 active, R² 0.827-0.836):
- At the floor no CDDM knob adds more than ~120 units to the standard network's 418. The largest
  gains: the W_rec magnitude penalty at 0.2 (+115; R² 0.816 and 0.836), softplus (+100; R²
  0.820-0.832), and no weight decay or no recurrent noise (+70 each; R² within the ~0.01 scoring
  noise of the standard).
- The metabolic penalty costs R² and adds no units: λ = 10 reads ~400 active at R² 0.78.
- Weight decay at 1e-5 and 1e-4 loses ~50 units and ~0.01 of R².
- Sigmoid works with 235-255 units at R² 0.824-0.829.
- Input scale on the flip-flop is the one knob that recruits for free: 614-913 active at scale 20
  against ~300 for the default, R² 0.950-0.953 throughout.
- The low flip-flop point (334 active, R² 0.931) is the default-input run that crosses at 23k but
  ends at 0.73x its fitted floor; its loss ratio L(t*)/L_end = 1.44 is what drags the estimate down.
  It is not a converged read-out, but the rule admits it and it is drawn.

Panel titles of the trade-off scatter now give the range of floor crossings where each ladder was
read (10-73k), not the training budget, which read as the read-out point (Pavel). The 227
`RMtop200` remedy top-ups submitted at 14:56 by another session were approved by Pavel.

