## 2026-08-25 (night) — Flip-flop vs CDDM hard silence: NOT a bug, NOT a contradiction

Prompted by "is it genuinely true? is it a bug, or a task-specific law?".

### The falsification test, and it killed my first hypothesis

I hypothesised that CDDM's exact zeros were an artifact of `same_batch=True` — the network trains on
450 fixed conditions and participation is measured on those same trials, so a unit can be driven
below threshold on that finite set while not being truly dead. New script
[`silence_is_taskset.py`](../trainRNNbrain/experiments_and_analysis/silence_is_taskset.py)
re-measures trained CDDM nets on HELD-OUT coherences (midpoints between trained ones, 392 unseen
conditions), thresholds fixed in advance (>20% relative drop = artifact, <5% = real).

**Result: hard-silent fraction 0.621 → 0.624, a −0.5% relative change. HYPOTHESIS FALSIFIED.**
CDDM's dead units are dead for any input in the trained range. The zeros are a real network property.

### What the difference actually is

Participation spectra are BIMODAL in both tasks, and both silence ~78–84% of units. They differ only
in WHERE the silent mode sits:

| | silent mode | active mode | exactly 0 | \|W_inp\| | \|W_inp\|/\|W_rec\| |
|---|---|---|---|---|---|
| flip-flop (21 nets) | 2.0–2.9e-4 | 0.79–1.1 | 0.2–0.6% | 97.6 | 2.25 |
| CDDM (3 nets) | exactly 0 | ~0.06 | 70–72% | 16.0 | 1.03 |

Both silent modes are 3–4 orders below their own active mode. **The 1e-6 absolute threshold simply
falls above CDDM's silent mode and below the flip-flop's.** The scale-free criterion agrees across
both tasks (0.84 vs 0.85 at N=2000) precisely because it does not depend on that accident.

**Mechanism, consistent with the task demand:** the flip-flop must let a sparse input pulse trigger a
decisive state flip, so training amplifies W_inp ~6x more than CDDM does (97.6 vs 16.0; inp/rec 2.25
vs 1.03). Those large input weights inject a small transient into every unit on every pulse, lifting
the silent mode off exactly-zero to ~2e-4. CDDM's inputs are sustained graded coherences and need no
such gain.

### Consequence

⚠️ **`p < 1e-6` is NOT a task-portable definition of silence.** It is calibrated to CDDM's dynamic
range and reports ~0% silence on the flip-flop, where the scale-free rule reports ~84%. `paper.md`
must define silence scale-free and present the absolute count as a task-specific diagnostic, not as
the primary measure. Reporting both remains mandatory — this is the sharpest instance yet of the two
disagreeing.

⚠️ Separate pre-existing BUG found: the saved `.npz` parameter files LOSE `activation_args`.
`np.savez` stores a dict argument as an array of its KEYS — the file literally contains
`array(['name','slope'])`, values discarded. Reconstructing a net from `.npz` alone fails with an
opaque indexing error inside `RNN_numpy`; `silence_is_taskset.py` restores it from the saved config.
Any dict-valued parameter has the same problem. Worth fixing in the saver.

### CDDM under the same read-out battery (`cddm_criteria.py`)

The double-check requested: does CDDM's saturating law depend on the criterion, as the flip-flop's
did not?

| criterion | mean T | b (scale-free) | b (hard) |
|---|---|---|---|
| endpoint | 200 000 | 0.53 | 0.52 |
| fixed iteration | 100 000 | 0.55 | 0.52 |
| matched loss L\* | 9 300 | 0.60 | 0.80 |
| matched excess | 11 000 | 0.59 | 0.78 |
| drift | — | **DISQUALIFIED** | **DISQUALIFIED** |

**Saturation is confirmed on CDDM under every usable criterion (b = 0.52–0.60, all << 1).** Compared
against the flip-flop's b = 0.41–0.46, both tasks are sublinear; CDDM's exponent is somewhat higher.

⚠️ **The `drift` criterion does NOT port to CDDM.** Alpha is already ~0.5 by the second probe for
every N ≥ 500, so there is no ballistic→diffusive transition inside the recorded window; the original
implementation returned iteration 1000 (loss still 0.033–0.040 against a final 0.019–0.025) and a
meaningless b = 0.94 that would have read as "no saturation". **`common.diffusive_onset` now requires
a prior DIRECTED phase and returns nan otherwise** — an onset presupposes a prior state. Flip-flop is
unaffected (62/64 runs, seed CV 0.12). This is the second time this one criterion has produced a
plausible-looking number from a definition defect; both times a seed-scatter or start-of-trace check
caught it.

⚠️ On CDDM the HARD criterion is strongly criterion-sensitive (b = 0.52 → 0.80) while the scale-free
one is not (0.53 → 0.60). The hard reads at 9–11k are taken before much hard silencing has happened.
Another reason the scale-free rule is the primary measure.

⚠️ The criteria land two orders of magnitude apart in depth on CDDM (9k vs 200k) but only one on the
flip-flop (27k vs 150k), and b falls with depth — so the flip-flop's tighter agreement across criteria
is partly because its criteria happen to land closer together, not only because the law is cleaner.

---

