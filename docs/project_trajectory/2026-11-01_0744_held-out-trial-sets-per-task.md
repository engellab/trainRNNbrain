## 2026-11-01 07:44 — Held-out R²: which trials are new, per task (dead-relu-paper #71)

Work package `methods-r2` (#51, PR #64, review comment 6560107758). Option (b) of #71: held-out R²
means new draws of the training noise on each task's trial set; Methods M05 states per task whether
the trials are new.

- **3BFF: new in practice.** `heldout_r2.heldout_batch` draws 256 trials from `HELDOUT_SEED`. For each
  held-out trial I computed its probability under the task's generative process (per channel: each
  onset gap 10 + ⌊Exp(mean 75)⌋, so P(gap = 10 + k) = q^k (1 − q) with q = e^(−1/75); sign 1/2; the
  trial ends with no onset before step 300) and multiplied by the 200,000 × 1,024 trials of one
  training run. The largest single value is 0.0008 and the sum over all 256 is 0.0009: the expected
  number of held-out trials that also occur in training is below 0.001. Batch built from the saved
  config of `NBitFlipFlop_n100/EqType=h_N=100_arm=control` (the held-out batch is the same for every
  3BFF run: same task block, same seed).
- **CDDM: new.** The 392 held-out conditions use coherences at the midpoints of adjacent training
  coherences (`heldout_batch` l. 107-109); none is a training coherence, so no held-out trial was
  trained on.
- **DMTS: not new.** A trial is fixed by its sample-test pair (4) and the two onset shifts, each an
  integer in [−10, 9] (`TaskDMTS` l. 62-63): 4 × 20 × 20 = 1,600 noise-free trials. Training draws
  1,024 per iteration for 150,000 iterations, so it visits all of them; only the noise is new.

Also corrected: the docstrings of `heldout_r2.py` (module and `heldout_batch`), which said that
training drew from an unseeded generator and that DMTS trials are unseen.

Computation (run from `trainRNNbrain/experiments_and_analysis/`):

```python
import glob, numpy as np
from omegaconf import OmegaConf
import heldout_r2 as h
d = sorted(glob.glob('../../data/trained_RNNs/NBitFlipFlop_n100/EqType=h_N=100_arm=control/*/'))[0]
cfg = OmegaConf.load(sorted(glob.glob(d + '*_config.yaml'))[0])
bi = h.heldout_batch(cfg)[0].numpy()
T, q = 300, np.exp(-1 / 75)
logp = []
for b in range(bi.shape[2]):
    lp = 0.0
    for c in range(3):
        x = bi[c, :, b]
        last = 0
        for t in [t for t in range(T) if x[t] != 0 and (t == 0 or x[t - 1] == 0)]:   # onsets
            lp += (t - last - 10) * np.log(q) + np.log(1 - q) + np.log(0.5)
            last = t
        lp += max(T - last - 10, 0) * np.log(q)                                      # no onset after
    logp.append(lp)
print(200_000 * 1024 * np.exp(max(logp)), 200_000 * 1024 * np.exp(logp).sum())
# 0.00081 0.00086
```
