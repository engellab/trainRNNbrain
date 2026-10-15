## 2026-08-21 17:18 — ALL FLIP-FLOP RESULTS SO FAR ARE COMPROMISED: same_batch=True

Pavel caught this. `trainer_ptrack_plain.yaml` sets `same_batch: True`, which draws ONE batch and
reuses it for every iteration.

**On CDDM this is defensible.** `TaskCDDM.get_batch()` returns all 450 conditions — the full,
enumerable condition grid — so "the batch" IS the task and repeating it is repeating the task.

**On the flip-flop it is not.** Pulse times are drawn i.i.d. from an exponential inter-event
distribution, so a batch is 256 samples of a continuous space. The entire 45-run sweep trained
2000-unit networks on **256 frozen trials for 300,000 iterations**. That measures memorisation, not
the task.

### What this invalidates

Everything derived from the flip-flop sweep, specifically:

- the silent-fraction-vs-k result and its depth dependence
- every floor fit (`flipflop_floor.py`) — those are MEMORISATION floors, which is very likely why the
  loss reached R² = 0.993 against CDDM's ~0.85, and why the floor kept dropping and would not
  identify under the halved-range check
- `M ~ N k^0.85` and the whole reading-depth correction built on it
- the collapse test, the drift check, the contour map
- the conclusion that "silencing falls with task complexity"

The DIRECTION of the k-effect may well survive — but nothing measured on memorised data can be
quoted, and the floors certainly cannot.

The CDDM results are unaffected.

### Fixes made

1. **`configs/trainer/trainer_ptrack_freshbatch.yaml`** — `same_batch: False`, a new config rather
   than a launcher override, since this is a fixed task-appropriate choice and not a swept parameter.
2. **`TaskNBitFlipFlop` vectorised.** With a fresh batch every iteration the generator sits directly
   on the training loop. Two rewrites: the per-timestep target scan (which tested `i in list` at
   every step) became a forward-fill via a running maximum over event positions; and `get_batch` now
   draws all inter-event gaps for the whole batch in one pass instead of looping trial-by-trial.
   Per-channel event slicing uses `searchsorted` on the already-sorted row index rather than a mask
   per channel, which alone was 230M element comparisons at B=1024, k=8.
   Cost at k=8: 0.057 s per 256-trial batch -> 0.0099 s. At B=1024: 0.103 s -> 0.038 s.
   Verified equivalent to the replaced per-trial path on 4000 generated trials (0 mismatches), and
   the batch path matches it distributionally (target variance, switches/channel, duty cycle, mean
   inter-event gap all agree within sampling noise).
3. **Reproducibility bug fixed**: pulse signs were drawn from the GLOBAL numpy stream while event
   times came from `self.rng`, so seeding the task never reproduced a trial.
4. **batch_size 256 -> 1024.**

### Measured cost of the fix

Fresh batches at B=256 took the worst cell (k=8, N=2000) from 0.140 to 0.212 s/iter before the
vectorisation. B=1024 has to be re-timed on the GPU before any budget is set — the batch dimension
scales the forward and backward passes, so this is not a small correction.

---

