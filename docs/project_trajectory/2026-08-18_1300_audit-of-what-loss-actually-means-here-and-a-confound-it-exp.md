## 2026-08-18 13:00 — Audit of what "loss" actually means here, and a confound it exposes

Prompted by the question of whether the "lowest achievable loss" statistic was measuring anything
real. It was not, and checking the code turned up three facts plus one genuine confound.

### What the recorded loss is

1. **There is no validation set.** `Trainer.run_training` creates `val_losses`, never appends to it,
   and returns it empty; `run_experiment.py` saves that empty list. **Every loss number in this
   project is a training loss.**
2. **`same_batch=True`**: one fixed 450-condition batch is drawn once and reused for all 200k
   iterations. No batch resampling.
3. **The recorded loss comes from a NOISY forward pass** (`train_step` uses `w_noise=True`), so the
   only iteration-to-iteration variation is injected noise plus the weights moving.

**Consequence: the earlier "median of the 101 lowest losses" statistic is retracted.** Both it
(~0.0155) and the smoothed loss (~0.0221) are noisy training losses — the gap between them is the
lower tail of the noise distribution versus its mean, not noise-free versus noisy. Taking the minimum
of a noisy series is a noise lottery whose winner depends on how many draws the run had, so it is not
a property of the trained network at all.

### The well-defined replacement: noise-free loss of the final parameters

Deterministic, one number per seed (`eval_noisefree_loss.py`). Decomposed against the noisy loss in
the SAME implementation so the split is internally consistent:

| N | clean MSE | noisy MSE | noise share | n |
|---|---|---|---|---|
| 100 | 0.00851 | 0.03036 | 72.0% | 3 |
| 500 | 0.00851 | 0.03062 | 72.2% | 3 |
| 1000 | 0.00907 | 0.02952 | 69.3% | 3 |
| 2000 | 0.00991 | 0.03046 | 67.5% | 1 |

### The confound this exposes

**The noisy loss is flat across sizes (0.0295–0.0306, no trend) while the clean loss rises
monotonically, +16% from N=100 to N=2000.** Two effects cancel: bigger networks average injected
noise better across the readout (noise share 72.0% → 67.5%) but fit the deterministic task worse.

So **matching on training loss is not the same as matching on deterministic task performance.** At
equal noisy loss a larger network has systematically higher clean error. If that reflects being less
well fit, it biases the M(N) comparison in exactly the dangerous direction: a less-trained large
network is less silenced, M is inflated, and the result leans toward "no saturation" — the conclusion
we drew.

**This is not yet resolvable from existing data.** Clean loss can only be evaluated where weights
were saved, i.e. at the final parameters, so there is one point per run and no clean-loss trajectory
to match on.

**Cheap fix for future runs:** `track_participation_` already performs a noise-free forward pass every
`track_every` iterations. Computing the masked MSE from that same pass costs essentially nothing and
would give a noise-free loss trajectory, making the whole matching protocol re-runnable on the clean
quantity.

### Two further audit findings

- **The folder score is a single noise draw.** `run_experiment.py:119` calls
  `eval_step(..., noise=True)` once. Across 30 noise realisations that quantity has sd 0.0085 and
  range 0.818–0.854, so the saved r2 carries about +-0.02 of pure noise and should not be used as a
  performance measure.
- **Torch and numpy noisy evaluations disagree by ~4 sd.** One N=100 net: numpy gives
  r2 = 0.8395 +- 0.0085 over 30 draws, the torch-recorded score is 0.8737. The noise scaling formula
  is identical in both (`sqrt(2/alpha)*sigma`), so the cause is elsewhere and is not yet identified.
  The M(N) pipeline is unaffected (participation traces and loss curves are torch-native), but the
  numpy-based offline analyses — `population_distortion.py` in particular — carry an unquantified
  offset.

### Held-out validation: NOT memorisation. The floor is the task's ambiguous conditions.

No validation set exists in the pipeline, so one was built retrospectively: coherences INTERLEAVED
with the training grid (midpoints of the 15 training values), inside the trained range but never seen.
Evaluated on final parameters (`eval_heldout_loss.py`). Figure: `img/internal_figures/heldout_loss.png`.

**Raw gap looks alarming — held-out clean loss is ~85% above training at every size:**

| N | train clean | held-out clean | gap |
|---|---|---|---|
| 100 | 0.00851 | 0.01589 | +86.6% |
| 500 | 0.00851 | 0.01598 | +87.8% |
| 1000 | 0.00907 | 0.01664 | +83.6% |
| 2000 | 0.00991 | 0.01738 | +75.3% |

**But binning by |relevant coherence| shows the gap is entirely in the near-ambiguous conditions:**

| \|coh\| bin | train | held-out | gap (N=1000) |
|---|---|---|---|
| [0, 0.02) | 0.03504 | 0.08429 | **+140.5%** |
| [0.02, 0.05) | 0.01447 | 0.01602 | +10.7% |
| [0.05, 0.1) | 0.00081 | 0.00009 | **−89.4%** |
| [0.1, 0.2) | 0.00004 | 0.00003 | −8.1% |
| [0.2, 0.4) | 0.00003 | 0.00003 | +1.7% |
| [0.4, 0.8) | 0.00004 | 0.00004 | −1.2% |

At |coh| >= 0.05 the held-out error is **equal or lower** than training. There is no memorisation of
the trained solution; the network generalises perfectly wherever the task is well posed.

**Why the low bin behaves that way — the target is discontinuous at coh = 0.** `correct_choice` is
+1 for coh > 0 and −1 otherwise, so an infinitesimal change of stimulus flips the target by 2. No
network can generalise across that step. Training also contains coh = 0 exactly, whose label (−1) is
arbitrary and simply memorised (its MSE is 0.00078, the *lowest* of any bin). The held-out midpoint
0.0078 sits inside the discontinuity and is therefore maximally hard.

**Where the clean training loss actually comes from** (one N=1000 net, total 0.00947):

| \|rel coh\| | n conditions | mean MSE | share of total |
|---|---|---|---|
| [0, 0.001) | 30 | 0.00078 | 0.6% |
| [0.001, 0.02) | 60 | 0.05417 | **76.3%** |
| [0.02, 0.05) | 60 | 0.01532 | 21.6% |
| [0.05, 0.2) | 120 | 0.00044 | 1.2% |
| [0.2, 1.01) | 180 | 0.00008 | 0.3% |

**98% of the loss comes from the 27% of conditions with |coh| < 0.05.**

### What this settles

- **The training loss is not inflated by memorisation** — the apparent 86% gap is the ill-posed
  region, not overfitting. The matched-performance protocol survives.
- **It explains why the floor is size-independent**: the floor is set by conditions where the target
  is discontinuous, which is a property of the TASK and cannot be improved by adding units. This is a
  mechanism for the null result established earlier, not just a failure to reject it.
- **Caveat on interpretation**: because 98% of the loss lives in the ambiguous regime, "matched
  training loss" means "matched behaviour on near-ambiguous stimuli". Networks matched this way are
  NOT matched on overall task competence — but competence saturates early (MSE ~1e-4 at |coh| >= 0.05)
  and would be a useless discriminator late in training, so the loss is still the right matching
  variable. It just measures something narrower than it appears to.
- **Still recommended for future runs**: log the clean masked MSE from the noise-free probe that
  `track_participation_` already performs (free), plus a held-out batch on a coarser cadence.

### Size dependence of performance: deterministic effect survives, noise-tolerance claim RETRACTED

Prompted by the observation that noise-on MSE appears to fall with N. Decomposing the final-parameter
error into deterministic and noise-induced parts, then localising each to the coherence bins that
carry the loss. Figures: `error_decomposition.png`, `lowcoh_vs_N.png`.

**Noise tolerance — tested four ways, not supported.**

| test | noise-excess slope / decade | p |
|---|---|---|
| whole task, train | −0.00112 | 0.495 |
| whole task, held-out | −0.00314 | 0.084 |
| 0 < \|coh\| < 0.02 (76% of the loss) | −0.00590 | 0.313 |
| coh = 0 exactly (30 conditions) | **+0.00099** | 0.811 |

Only the held-out whole-task test approaches significance, and the effect does not localise: the two
bins that carry almost all the loss show nothing, one of them with the sign reversed. **The earlier
statement that larger networks tolerate noise better is withdrawn** — it rested on a single p = 0.084
result. Two successive hypotheses about where it lives (signal resolution at small coherence; then
stability of the memorised output at coh = 0) were both stated in advance and both falsified.

**Deterministic error — confirmed three ways, all monotone and same-signed.**

| test | clean-error slope / decade | p |
|---|---|---|
| whole task, train | +0.00073 | 0.079 |
| whole task, held-out | +0.00088 | 0.051 |
| 0 < \|coh\| < 0.02 | +0.00338 | 0.072 |

Larger networks are **worse at resolving the weakest real signal**: at |coh| = 0.015625 the clean MSE
rises 0.04903 → 0.05487 from N=100 to N=2000 (+12%), monotonically across all four sizes. No single
p clears 0.05, but three slices agreeing in direction and monotonicity is stronger than any one.

**Consequence for M\*.** The matching variable (total noisy loss) is flat across sizes *because* it is
dominated by the noise term, which has no size dependence — so it is a clean thing to match on. But
the deterministic quality underneath it is NOT equal at the match point, and larger networks are the
worse ones. If "deterministically worse" amounts to "effectively less trained", then M is inflated at
large N, which is the direction of the no-saturation conclusion. This is a caveat on that conclusion,
not a refutation, and it cannot be resolved without a clean-loss trajectory (see the monitoring
recommendation: compute masked MSE from the noise-free probe `track_participation_` already runs).

**Caveat throughout:** N=2000 is a single seed and sits at the end of the lever arm for every
regression here; its other two seeds land ~19:10 today.

---

