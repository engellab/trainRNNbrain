## 2026-10-12 — Claim 1 on standard networks: the silent units do no work; the read-out moves to the noisy loss

**What was asked (Pavel).** Close claim 1's gap ("the ablation was run once, on pre-standard
networks, with no figure") on standard networks. Split the old Figure 1: F1 is the phenomenon only,
with panel d replaced by the ablation; F2 is everything that failed (the old panel d, "no knob fixes
it", moves there); F3 is a bird's-eye view of what worked. Tighten the read-out to 1.05x each
network's own floor, and leave the learning-rate confound of the size series (lr ∝ N^-1/3) in, to be
handled by reading each network at its own floor.

### Weights at the floor did not exist, so the controls were re-trained keeping them

No standard network had weights where its loss reaches its floor: runs save only LastParams, the
noisy-loss minimum and, for top-ups, the parent's end state, and CDDM crosses at ~25k. New option
`+trainer.checkpoint_keep=true` keeps every periodic checkpoint in its own `it<iteration>/` folder
(absolute iteration, so a continuation's snapshots sort after its parent's), without the optimiser
state (b524ac7, 7739a60; test 8 in tests/test_periodic_checkpoint.py). `SilentReLU_c1_snapshots_della.slurm`
re-trained the three standard control cells setting for setting (config diff: only keys added since,
all off), N = 1000, 3 seeds, a snapshot every 500 iterations: flip-flop 75k (15422706), CDDM 100k
(15422707), DMTS 5 tau 150k (15422708). Snapshots: /scratch/gpfs/TENGEL/pt1290/trainRNNbrain/data/c1_snapshots.

### The ablation

`ablate_silent_units.py`. Build the network from its config, assert it is standard, order units by
participation on a noise-free pass over a full training batch (ties at p = 0 broken by peak
pre-activation), remove a unit by zeroing its OUTGOING columns of W_rec and W_out (exactly its rate
clamped to 0 downstream, its output noise included), score held-out r2 as heldout_r2 does (unseen
inputs, training noise on, sigma_w 0, 9 draws) with the held-out batch drawn once and the noise
draws seeded identically at every point. Orders: quietest first, busiest first, 5 random. Gates in
every JSON: recomputed training-batch r2 against the folder name (±0.03; one CDDM seed fails by
0.033, the known +2.7 sd outlier), offline active count against the trace.

**End of training, every standard control on disk (36 networks):**

| cell | silent | r2 intact | all silent removed | as many at random |
|---|---|---|---|---|
| CDDM N = 500 / 1000 / 2000 / 4000 (100k) | 56 / 68 / 80 / 86% | 0.826 / 0.836 / 0.837 / 0.834 | 0.825 / 0.837 / 0.837 / 0.828 | −0.03 / −0.19 / −0.26 / −0.22 |
| DMTS 5 tau N = 500 / 1000 (150k) | 73 / 83% | 0.999 / 0.999 | 0.996 / 0.989 | −0.13 / −0.15 |
| flip-flop N = 500 / 1000 / 2000 / 4000 (75k) | 60 / 73 / 82 / 86% | 0.952 | 0.37 / 0.41 / 0.55 / 0.69 | ≈ 0.05 |

Removing every active unit takes CDDM to −0.33 and DMTS to −0.17; on the flip-flop the silent units,
freed of the active ones that hold them down, run away (r2 ≈ −10^11).

**The flip-flop exception, and what it is not** (diagnostics on one network per task, N = 1000):
not a noise artefact - the drop survives noise-free scoring (0.84 → 0.59) and clamping the removed
units to their mean rate under noise only partly restores it (0.38 → 0.66); not ReLU's scale
symmetry - the silent units' outgoing weights are SMALLER than the active ones' (median norm ratio
0.29 W_rec, 0.6-0.7 W_out) and the signal they send is ~10^-4 of an active unit's. Removing the
quietest 600 units (p < 0.001 q95) costs 0.06; the next ~110, between 0.001 and 0.05 q95, cost 0.4.
Units removable for at most 0.01 of r2: ~300-320 of 1000 on the flip-flop, ~780-820 on CDDM,
~835-840 on DMTS.

**Prune and retrain** (`retrain`; the silent units removed and held out, the rest fine-tuned 3,000
iterations as in training; same count removed at random; intact network fine-tuned as control),
end of training, N = 1000, 3 networks per task (re-run 2026-10-13 after the fix below):

| | flip-flop | CDDM | DMTS |
|---|---|---|---|
| intact, fine-tuned | 0.951-0.954, flat | 0.835, flat | 0.999, flat |
| silent removed: straight after → after fine-tuning | 0.37-0.48 → **0.94-0.95** | 0.836 → 0.835-0.837 | 0.98-0.99 → **0.998-0.999** |
| as many removed at random, after fine-tuning | 0.91-0.93 | 0.83 | **0.42** in all three (back to the constant-output solution) |

So every task does without its silent units: on the flip-flop they are wired in, but not needed.

⚠️ **A bug in the fine-tune loop, found and fixed 2026-10-13** (7608099). The first version reported
that fine-tuning the INTACT CDDM network cost it 0.04-0.07 while the pruned copy did not move. A
delegated investigation (/scratch/gpfs/TENGEL/pt1290/c1_finetune_probe, 10 GPU jobs) traced it:
`retrain_curve`'s scoring reseeded the model's generator for its fixed noise draws, and training
draws its noise from the same generator, so from iteration 251 the fine-tune saw the same 250 noise
realisations eleven times. On CDDM's fixed coherence grid every (trial, noise) sample recurred and
the intact network memorised them - training-sample r2 0.874 → 0.91-0.92 while held-out fell -
mostly by waking 130-180 silent units; the pruned copy could not. With the generator saved and
restored around scoring, every arm is flat (fresh or saved Adam, wd 0 or 1e-6, lr or lr/10;
−0.002 at the end of training). Not the fresh-Adam kick (real, but leaves no trace) and not coupled
weight decay. **The project's own warm start (`paths.init_from`, used by every top-up) is
unaffected**: 3,000 iterations from each CDDM control move held-out r2 by < 0.001 with or without
the saved Adam moments. Results from the buggy loop are kept in
/scratch/gpfs/TENGEL/pt1290/c1_retrain_bug_noise_reseed.

### The read-out moves to the noisy loss (Pavel)

Plotted (img/internal_figures/diag_loss_floors_standard.png): on the standard flip-flop the
noise-free probe loss bottoms out near 10k and climbs back 2-4x to a plateau of 0.13-0.18, while the
noisy loss - the one training minimises - is still falling at 75k. The August floor results
("floors identify to 2%") were on the trainable-bias networks, where the noise-free loss plateaued
and stayed. **Rule from now on: performance and convergence are read only on the noisy loss.**

`common.noisy_floor` / `noisy_loss_series` (945017f, 1d51ff3, 82f2df2): the per-iteration
training loss the trainer already saves, joined along the warm-start chain (nothing added to
training - a running median over 2,001 iterations is an average over that many noise draws and
batches); the floor fitted from the ESCAPE for a task that stalls (DMTS waits at the constant output
for 200 to 155,000 iterations); 1.05x crossing on the running median. Guards: fit on its bound,
never reaching the bar, and - judged by the median of the last 10% of training, so DMTS's bursts
back towards chance do not decide it (Pavel) - a loss ending more than 1.5x above its floor. A
second test (floor against the run's best long-run level) was tried and dropped: DMTS sits flat at
~7e-5 between bursts while its floor, ~1e-4, averages troughs and bursts, so it rejected 7 of 8 runs
that had plainly reached their floor (img/internal_figures/diag_dmts_noisy_floor.png).

**Where 1.05x of the noisy floor lands** (img/internal_figures/diag_noisy_loss_budget.png):
CDDM 24-38k in every run, floor 0.022 at every N - 100k is ample. DMTS within 1-15k of its escape,
floor ~1e-4 at every N; the budget is set by the escape, which comes later with size (5 tau
N = 2000: 33-42k; 7 tau N = 4000: 112-155k of 210k). **The flip-flop is not converged at 75k**: per
run, the fit reaches 1.05x its floor at 21k-368k (median 69k), only 8 of 15 by 75k, and every run's
loss is still falling. So every standard flip-flop control is being continued to 200k by warm start
from its own config (`SilentReLU_ff_topup200k_della.slurm`): the claim-1 snapshot runs keeping
snapshots (15431471), and the size series N = 500 / 1000 / 2000 / 4000 (15431472-15431475; N = 4000
~3 days).

### At the floor (the claim-1 snapshot runs, N = 1000)

| | CDDM, floor (24.9-25.9k) | CDDM, end (100k) | DMTS, floor (17-42k) | DMTS, end (150k) |
|---|---|---|---|---|
| active | 415-438 | 302-318 | 291-441 | 159-234 |
| r2 intact | 0.829-0.834 | 0.836-0.838 | 0.999 | 0.999 |
| all silent removed | 0.828-0.833 | 0.836-0.838 | 0.909-0.972 | 0.983-0.994 |
| as many at random | −0.14 to 0.05 | −0.06 to 0.10 | ≈ −0.1 | ≈ −0.1 |
| silent removed, then retrained 3,000 iterations | 0.825-0.830 (intact, fine-tuned: 0.824-0.830) | | **0.9987-0.9993** (as many at random: 0.998, 0.999, 0.418) | |

**The flip-flop at its floor** (the claim-1 runs continued to 200k; crossings at 27k, 39k and 69k):
266-354 active (191-194 at 200k); removing every silent unit takes held-out r2 from 0.94-0.95 to
0.25-0.48, as many at random to 0.01-0.09; pruning the silent units and retraining 3,000 iterations
restores 0.937-0.947 (intact, fine-tuned: 0.943-0.950), as many at random only 0.920-0.924.

CDDM: about 430 units work at the floor and the other ~570 can go for nothing; by 100k it has shed
another ~120 without improving. DMTS escaped late in these three runs (14-40k) and is read 1.6-6k
after; right after the escape its silent units still carry a little (0.03-0.09), which has almost
gone by the end.

### Figure 1, restructured (fig_paper_F1.py; fig_supp_ablation.py)

Every panel on standard networks. (a, b) example network: the standard flip-flop control at 75k
(242 of 1000 active). (c) size: CDDM and DMTS read at their noisy floor at every size (CDDM ∝
N^0.46, crossings 26-30k; DMTS ∝ N^0.52 from three sizes, crossings 14-40k); the flip-flop drawn at
the end of its budget (∝ N^0.48), open markers, because at 75k no run at N = 1000 or 4000 reaches
its floor - a series is read at its floor only when every size has a read-out. (d) the ablation,
one sub-panel per task, every task at its floor, with the prune-and-retrain result as a star.
(e) the noisy loss and the silent count over training, every seed, each run's crossing marked:
after its floor the flip-flop (200k) silences 116 more units, CDDM (200k) 152 and DMTS (150k) 272. The old knob panel is `panel_knobs`, kept for Figure 2. The coverage audit's signature now
includes trial length, bias range and remedy switches (a trainable-bias 30k sweep and 7 tau DMTS were
being reported as missing sizes). Supplementary: the end-of-training ablation at every size.

**Open.** Panel c's flip-flop series waits on the 200k continuations at N = 2000 and 4000.

